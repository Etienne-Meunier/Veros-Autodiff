"""Shared helpers for report-longrollouts-1: dloss/d{c_k,c_eps} for very long
rollouts (n=500..3000) on GPU, reporting wall time and peak GPU memory alongside the
gradient.

History (see scripts/debugging_rollouts/common.py for the full investigation this
report is downstream of): the original plan was nz=15/30/60 on the ETOPO5-based
GlobalFlexibleMLDLearningSetup with a scan-of-checkpointed-chunks rollout. Both parts
of that plan hit real, unrelated problems:
  - nz=15 on that setup gives grad=nan (ETOPO5 topography interpolated onto only 15
    levels is degenerate somewhere -- confirmed on CPU and GPU, root cause isolated,
    not fixed). nz=60 (native) works fine but is out of scope here -- see below.
  - The scan-of-checkpointed-chunks rollout (nested lax.scan, checkpoint the inner
    scan) SIGKILL'd at *compile time* (host RAM, not GPU memory) even at n=10,
    chunk_size=4 -- a real, apparently version-sensitive JAX/XLA fragility in
    composing checkpoint with nested scan under grad (see debugging_rollouts/
    common.py's module docstring and the GitHub issues cited there).

This report drops the nz sweep entirely and uses report-1's GlobalFourDegreeSetup
(global4deg: native nz=15, assets.json bathymetry, nonlin2 EOS, no streamfunction) --
the setup debugging_rollouts validated everything against. Rollout uses
rollout_double_checkpoint (below): per-step jax.checkpoint inside the inner scan,
PLUS an outer jax.checkpoint wrapping that whole inner scan, itself scanned at the
outer level. Empirically the only structure tried that (a) doesn't blow up at compile
time as chunk_size grows (flat-ish through chunk_size=16, vs scan_scan/scan_unrolled
dying by chunk_size=4) and (b) gives grad matching a plain/unchunked reference to
rel_err <1e-6 at every chunk_size tested. NOT yet confirmed to bound peak memory at
n=500..3000 specifically (only checked at n=20) -- this report's own sweep is the
first real test of that at scale.

Loss is plain squared temp error, not mld/mld_ma -- report-mld-2 found mld_ma blows
up at full-grid n=200 (grad=1.25e21 vs FD~-2e6), unresolved; temp loss on the same
config came back clean. Sidesteps that open bug entirely.
"""
from __init__ import PRP
import sys
import os

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

# Persistent XLA compilation cache -- see scripts/debugging_rollouts/common.py's copy
# of this block for the full rationale.
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
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup
from tqdm import tqdm


def spin_up(warmup_steps=20):
    g4d = GlobalFourDegreeSetup()
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
    for _ in tqdm(range(warmup_steps), desc="spin-up (longrollouts-1, global4deg)"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def make_nz_setup(nz):
    """Flip settings.nz on GlobalFlexibleMLDLearningSetup (ETOPO5, gsw+streamfunction
    -- the "full" setup, heavier than global4deg) -- same "call parent set_parameter,
    then flip a setting" pattern used throughout this repo (e.g.
    scripts/debugging_rollouts/common.py's own make_nz_setup, report-mld-2/common.py's
    GswOnlyFullGridSetup). nz=15 on this setup gives grad=nan (isolated to the ETOPO5
    topography interpolated onto only 15 levels being degenerate -- see this module's
    docstring); nz=60 (native) is known clean. nz=64 is untested -- close enough to
    nz=60 that the same degeneracy is not expected, but this is the first real check."""

    class _NzSetup(GlobalFlexibleMLDLearningSetup):
        @veros_routine
        def set_parameter(self, state):
            GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
            with state.settings.unlock():
                state.settings.nz = nz

    return _NzSetup


def spin_up_mld(nz, warmup_steps=20):
    """Spin-up for the full MLD setup (gsw+streamfunction, ETOPO5, mld/mld_ma
    diagnostics) at the given nz -- heavier per-step graph than global4deg, expect
    worse compile times."""
    g4d = make_nz_setup(nz)()
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
    for _ in tqdm(range(warmup_steps), desc=f"spin-up (longrollouts-1, mld nz={nz})"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def _masked_sq_error(field, target_field):
    """Same NaN-safe masking as report-mld-2/common.py -- mld_ma is NaN at land and
    during the moving-average buffer's warm-up."""
    valid = ~jnp.isnan(field) & ~jnp.isnan(target_field)
    diff = jnp.where(valid, field - target_field, 0.0)
    return (diff**2).sum()


def mld_ma_agg_function(state, target_state):
    """The ACTUAL target loss from the original report-mld-2 plan -- known broken at
    full-grid n=200 (grad=1.25e21 vs FD~-2e6, unresolved). Used here deliberately
    (not the temp-loss proxy the rest of this report uses) to test whether the
    double_checkpoint method can even compute *something* at very long n on the real
    loss -- correctness of the result is explicitly out of scope, see report."""
    return _masked_sq_error(state.variables.mld_ma, target_state.variables.mld_ma)


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


def temp_agg_function(state, target_state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()


def plain_forward_rollout(step_fn, state, iterations):
    """Forward-only, no checkpoint needed (no backward pass) -- used to generate
    target_state cheaply, so timing measures only the grad-path's cost."""
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state


def rollout(step_fn, state, iterations, chunk_size):
    """double_checkpoint: per-step checkpoint(step) inside an inner scan of length
    chunk_size, PLUS an outer checkpoint wrapping that whole inner scan, itself
    scanned at the outer level (length n_full = iterations // chunk_size). See this
    module's docstring / scripts/debugging_rollouts/common.py's
    rollout_double_checkpoint for the full rationale and validation history."""
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


def peak_gpu_memory_bytes():
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        stats = None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


# Output directory -- $STORE (falls back to $HOME), NOT the repo checkout. A stray
# `g5k sync code` while a run was live once deleted its log file out from under it
# (unison reconciles away anything that doesn't exist locally).
STORE_DIR = os.path.join(os.environ.get("STORE", os.path.expanduser("~")), "report_longrollouts1_results")
os.makedirs(STORE_DIR, exist_ok=True)


def write_csv_incremental(rows, path):
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_worker(n, param, test_val, chunk_size, timeout_s=1200):
    """Launch worker.py as a fresh subprocess for one (n, param) config --
    one-jit-per-process discipline, also needed for peak-memory isolation. Kills the
    whole process group on timeout."""
    import subprocess
    import signal
    import time

    cmd = [
        sys.executable, f"{PRP}scripts/Reports/report-longrollouts-1/worker.py",
        "--n", str(n), "--param", param, "--test_val", str(test_val), "--chunk_size", str(chunk_size),
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
        return dict(n=n, param=param, chunk_size=chunk_size, status="TIMEOUT",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    if returncode != 0:
        tail_out = stdout[-1500:] if stdout else ""
        tail_err = stderr[-1500:] if stderr else ""
        print(f"{label} FAILED returncode={returncode} ({dt:.0f}s)\n--stdout--\n{tail_out}\n--stderr--\n{tail_err}", flush=True)
        return dict(n=n, param=param, chunk_size=chunk_size, status=f"CRASHED(rc={returncode})",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    result_line = [ln for ln in stdout.splitlines() if ln.startswith("RESULT")][-1]
    r = dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))
    print(f"{label} OK ({dt:.0f}s): compile={r['compile_time_s']} run={r['run_time_s']} grad={r['grad']}", flush=True)
    return dict(n=n, param=param, chunk_size=chunk_size, status="OK",
                compile_time_s=eval(r["compile_time_s"]), run_time_s=eval(r["run_time_s"]),
                grad=eval(r["grad"]),
                peak_mem_bytes=None if r["peak_mem_bytes"] == "None" else eval(r["peak_mem_bytes"]),
                subprocess_s=dt)
