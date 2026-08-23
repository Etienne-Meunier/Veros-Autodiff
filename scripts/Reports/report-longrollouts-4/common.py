"""report-longrollouts-4: how long a rollout can double_checkpoint sustain on the
full nz=64 `GlobalFlexibleMLDLearningSetup` (gsw/TEOS-10, streamfunction, real
ETOPO5 topography) with a 1-year trailing average of `mld` as the loss, and does
gradient descent actually calibrate c_k/c_eps there? Direct continuation of:
  - report-longrollouts-1's nz=64/mld_ma addendum, which found the setup's
    built-in `mld_ma` diagnostic (an ALWAYS-ON exact 720-step boxcar average,
    computed every single step as part of `after_timestep`, see
    global_4deg_mld_learning.py:497-512) already gives grad=-3.5e27 (garbage) by
    n=500. But that n=500 run had n < mld_ma_window=720 -- the buffer never
    filled, so that reading was almost certainly an artifact of an invalid,
    NaN-padded average, not evidence of the real (report-longrollouts-2-style)
    chaotic-adjoint blowup. Never actually tested with a *valid* buffer.
  - report-longrollouts-2, which found temp_ma (1yr average) does NOT rescue
    long-horizon temp gradients from blowup on global4deg -- but DID confirm
    gradients stay sane through n~2000-3000 there. Worth checking whether mld_ma
    behaves the same way (sane at moderate n, blows up later) once given a fair,
    fully-populated-buffer test.

Key extra problem specific to this setup: `GlobalFlexibleMLDLearningSetup`'s
`after_timestep` computes its `mld_ma` diagnostic INSIDE the model's own step
function, unconditionally, every step -- meaning its history buffer
(nx, ny, mld_ma_window=720) is part of `state` for the *entire* rollout,
automatically defeating report-longrollouts-2's whole point (bounding the
averaging buffer's cost to a fixed-length tail phase, independent of n). This is
also the likely dominant reason report-1's nz=64 addendum needed so much more
memory than gsw/streamfunction alone would predict.

Fix: `make_nz_setup_no_avg` below subclasses `GlobalFlexibleMLDLearningSetup` and
overrides `after_timestep` to compute only the raw `mld` diagnostic (identical
formula, still fully differentiable -- see that module's `mld_from_prho`), and
skips `update_mld_moving_average` entirely. `mld_history`/`mld_ma_index`/`mld_ma`
are then just unused, never-updated state -- nothing else in the model reads
them (mld/mld_ma are diagnostics bolted on for calibration, not physics inputs).
Our own boxcar average of raw `mld` is then applied report-2-style: a lead phase
(n - window steps, plain double_checkpoint, no buffer) + a fixed-length tail
phase (exactly `window` steps) that tracks the average -- so the averaging
buffer's cost is constant in n, same win as report-2, this time actually
achieved (report-2's design already proved correct on a toy model there; the
circular-buffer math is identical, just applied to `mld` -- 2D, xt/yt only --
instead of `temp`'s full 3D field, so if anything cheaper).

`mld` is NaN at land / degenerate (non-well-mixed) columns by construction (see
`mld_from_index`'s final `npx.where(well_defined, mld, npx.nan)`) -- our boxcar
average uses a plain mean (not nanmean, matching `update_mld_moving_average`'s
own convention), so any NaN day in the window makes that column's mld_ma NaN.
Loss therefore needs the same NaN-masked squared error report-1's addendum used
for the built-in mld_ma.
"""
from __init__ import PRP
import sys
import os

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

JAX_CACHE_DIR = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(
    os.environ.get("STORE", os.path.expanduser("~")), "jax_cache_veros"
)
os.makedirs(JAX_CACHE_DIR, exist_ok=True)
config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
config.update("jax_persistent_cache_min_compile_time_secs", 5)

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from veros import veros_routine
from veros.core.operators import update, at
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup, mld_from_prho
from tqdm import tqdm

# global4deg family's dt_tracer = 86400s = 1 day/step, so 365 steps = 1 year --
# same as report-longrollouts-2.
MLD_MA_WINDOW = 365


def make_nz_setup_no_avg(nz):
    """nz override (as report-1's make_nz_setup) + after_timestep override that
    computes raw `mld` only, skipping the built-in always-on moving-average
    update -- see module docstring."""

    class _NzSetupNoAvg(GlobalFlexibleMLDLearningSetup):
        @veros_routine
        def set_parameter(self, state):
            GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
            with state.settings.unlock():
                state.settings.nz = nz

        @veros_routine(dist_safe=False, local_variables=["zt", "prho", "maskT", "mld"])
        def after_timestep(self, state):
            vs = state.variables
            mld = mld_from_prho(vs.prho, vs.maskT, vs.zt, self.mld_reference_depth)
            vs.mld = update(vs.mld, at[2:-2, 2:-2], mld[2:-2, 2:-2])

    return _NzSetupNoAvg


def spin_up_mld(nz, warmup_steps=20):
    g4d = make_nz_setup_no_avg(nz)()
    g4d.setup()

    with g4d.state.variables.unlock():
        g4d.state.variables.c_k += 0.0
        g4d.state.variables.c_eps += 0.0

    def ps(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    step_jit = jax.jit(ps)

    state = g4d.state.copy()
    for _ in tqdm(range(warmup_steps), desc=f"spin-up (longrollouts-4, mld nz={nz}, no built-in avg)"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def make_diff_step(g4d):
    def pure_step(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    return pure_step


def set_vars(state, **values):
    n_state = state.copy()
    with n_state.variables.unlock():
        for name, value in values.items():
            setattr(n_state.variables, name, value)
    return n_state


def rollout(step_fn, state, iterations, chunk_size):
    """double_checkpoint, unchanged from report-longrollouts-1/2/3 -- lead phase
    of rollout_mld_ma."""
    n_full, remainder = divmod(iterations, chunk_size)
    step_ckpt = jax.checkpoint(step_fn)

    def block_fn(s, length):
        s, _ = jax.lax.scan(lambda c, _: (step_ckpt(c), None), s, length=length)
        return s

    if n_full:
        block_ckpt = jax.checkpoint(lambda s: block_fn(s, chunk_size))
        state, _ = jax.lax.scan(lambda c, _: (block_ckpt(c), None), state, length=n_full)
    if remainder:
        state = jax.checkpoint(lambda s: block_fn(s, remainder))(state)

    return state


def _make_avg_step(step_fn, window):
    def avg_step(carry):
        state, hist, idx = carry
        n_state = step_fn(state)
        hist = hist.at[..., idx].set(n_state.variables.mld)
        idx = (idx + 1) % window
        return (n_state, hist, idx)
    return avg_step


def rollout_mld_ma(step_fn, state, iterations, lead_chunk_size, tail_chunk_size, window=MLD_MA_WINDOW):
    """report-longrollouts-2's rollout_temp_ma, field swapped temp -> mld (2D,
    not 3D -- cheaper history buffer). See module docstring for why the setup's
    own built-in mld_ma is bypassed in favor of this."""
    lead = iterations - window
    if lead < 0:
        raise ValueError(f"iterations ({iterations}) must be >= window ({window})")
    if lead > 0:
        state = rollout(step_fn, state, lead, lead_chunk_size)

    mld_shape = state.variables.mld.shape
    hist0 = jnp.full(mld_shape + (window,), jnp.nan)
    idx0 = jnp.array(0)
    avg_step_ckpt = jax.checkpoint(_make_avg_step(step_fn, window))

    def block_fn(carry, length):
        carry, _ = jax.lax.scan(lambda c, _: (avg_step_ckpt(c), None), carry, length=length)
        return carry

    n_full, remainder = divmod(window, tail_chunk_size)
    carry = (state, hist0, idx0)
    if n_full:
        block_ckpt = jax.checkpoint(lambda c: block_fn(c, tail_chunk_size))
        carry, _ = jax.lax.scan(lambda c, _: (block_ckpt(c), None), carry, length=n_full)
    if remainder:
        carry = jax.checkpoint(lambda c: block_fn(c, remainder))(carry)

    state, hist, idx = carry
    mld_ma = jnp.mean(hist, axis=-1)
    return state, mld_ma


def plain_forward_rollout(step_fn, state, iterations):
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state


def plain_forward_rollout_mld_ma(step_fn, state, iterations, window=MLD_MA_WINDOW):
    lead = iterations - window
    if lead < 0:
        raise ValueError(f"iterations ({iterations}) must be >= window ({window})")
    if lead > 0:
        state = plain_forward_rollout(step_fn, state, lead)

    mld_shape = state.variables.mld.shape
    hist0 = jnp.full(mld_shape + (window,), jnp.nan)
    idx0 = jnp.array(0)
    avg_step = _make_avg_step(step_fn, window)
    (state, hist, idx), _ = jax.lax.scan(lambda c, _: (avg_step(c), None), (state, hist0, idx0), length=window)
    mld_ma = jnp.mean(hist, axis=-1)
    return state, mld_ma


def _masked_sq_error(field, target_field):
    """Same NaN-safe masking as report-1's addendum -- mld_ma is NaN at land and
    at any column with a degenerate mixed-layer read in the averaging window."""
    valid = ~jnp.isnan(field) & ~jnp.isnan(target_field)
    diff = jnp.where(valid, field - target_field, 0.0)
    return (diff**2).sum()


def mld_ma_agg_function(mld_ma, target_mld_ma):
    return _masked_sq_error(mld_ma, target_mld_ma)


def peak_gpu_memory_bytes():
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        stats = None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


STORE_DIR = os.path.join(os.environ.get("STORE", os.path.expanduser("~")), "report_longrollouts4_results")
os.makedirs(STORE_DIR, exist_ok=True)


def write_csv_incremental(rows, path):
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_worker(nz, n, param, test_val, lead_chunk_size, tail_chunk_size, timeout_s=1800):
    """Launch worker.py as a fresh subprocess for one config -- one-jit-per-
    process discipline, needed for peak-memory isolation during the n-scaling
    calibration. Kills the whole process group on timeout."""
    import subprocess
    import signal
    import time

    cmd = [
        sys.executable, f"{PRP}scripts/Reports/report-longrollouts-4/worker.py",
        "--nz", str(nz), "--n", str(n), "--param", param, "--test_val", str(test_val),
        "--lead_chunk_size", str(lead_chunk_size), "--tail_chunk_size", str(tail_chunk_size),
    ]
    label = f"[n={n}][{param}]"
    print(f"{label} launching...", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        dt = time.time() - t0
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, stderr = proc.communicate()
        returncode = "TIMEOUT"

    if returncode == "TIMEOUT":
        print(f"{label} TIMEOUT after {dt:.0f}s", flush=True)
        return dict(nz=nz, n=n, param=param, lead_chunk_size=lead_chunk_size, tail_chunk_size=tail_chunk_size, status="TIMEOUT",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    if returncode != 0:
        tail_out = stdout[-1500:] if stdout else ""
        tail_err = stderr[-1500:] if stderr else ""
        print(f"{label} FAILED returncode={returncode} ({dt:.0f}s)\n--stdout--\n{tail_out}\n--stderr--\n{tail_err}", flush=True)
        return dict(nz=nz, n=n, param=param, lead_chunk_size=lead_chunk_size, tail_chunk_size=tail_chunk_size, status=f"CRASHED(rc={returncode})",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    result_line = [ln for ln in stdout.splitlines() if ln.startswith("RESULT")][-1]
    r = dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))
    print(f"{label} OK ({dt:.0f}s): compile={r['compile_time_s']} run={r['run_time_s']} grad={r['grad']}", flush=True)
    return dict(nz=nz, n=n, param=param, lead_chunk_size=lead_chunk_size, tail_chunk_size=tail_chunk_size, status="OK",
                compile_time_s=eval(r["compile_time_s"]), run_time_s=eval(r["run_time_s"]),
                grad=eval(r["grad"]),
                peak_mem_bytes=None if r["peak_mem_bytes"] == "None" else eval(r["peak_mem_bytes"]),
                subprocess_s=dt)
