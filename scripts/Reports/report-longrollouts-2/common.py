"""report-longrollouts-2: does averaging temp over a trailing 1-year (365-step,
dt_tracer=86400s=1day for global4deg) window stabilize dloss/d{c_k,c_eps} at the
long horizons where report-longrollouts-1 found raw (single-final-snapshot) temp
gradients become numerically meaningless (grad=-4.5e30 at n=5000, 1.4e114 at
n=10000, both garbage, not overflow)? Same exact-boxcar-average technique as
setups/global_4deg/global_4deg_mld[_learning].py's update_mld_moving_average,
applied to the full 3D temp field instead of the 2D mld field, and to report-1's
GlobalFourDegreeSetup (global4deg -- lighter, already validated with
double_checkpoint) rather than the heavier GlobalFlexibleMLDLearningSetup.

Rollout is split into two phases to keep the averaging's extra memory cost
INDEPENDENT of the overall rollout length n -- carrying the window-length history
buffer through the *entire* n-step rollout would multiply scan's own O(n_full)
carry-history memory requirement (see report-longrollouts-1/common.py's docstring)
by the buffer size, i.e. make the original OOM problem worse, not better:
  - lead phase (n - window steps): no averaging tracked, byte-for-byte
    report-longrollouts-1's rollout() -- same memory scaling already
    characterized there (chunk_size=8 up to ~n=1500, 32 up to ~5000, 64 for
    n=10000).
  - tail phase (exactly `window` steps, fixed regardless of n): tracks an exact
    circular-buffer boxcar average of temp, itself double_checkpoint'd
    (its own lead/tail-independent chunk_size, `tail_chunk_size`). Because this
    phase's outer-scan length is fixed at window/tail_chunk_size regardless of n,
    its extra memory cost is CONSTANT in n -- verified against a numpy/FD
    reference on a toy scalar model before writing this (see
    /private/tmp/.../scratchpad/toy_temp_ma.py from that session; all 4
    (n, window, chunk) combos tried matched to <1e-9 value / <1e-5 grad rel err).

Loss compares the tail-averaged temp field to the same average computed on a
plain (uncheckpointed) forward-only reference trajectory -- mirrors report-1's
temp_agg_function exactly, just on temp_ma instead of a single final temp
snapshot.
"""
from __init__ import PRP
import sys
import os

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

# Persistent XLA compilation cache -- see debugging_rollouts/common.py for rationale.
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
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup
from tqdm import tqdm

# global4deg's dt_tracer = 86400s = 1 day/step (see global_4deg_learning.py), so
# 365 steps = 1 year.
TEMP_MA_WINDOW = 365


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
    for _ in tqdm(range(warmup_steps), desc="spin-up (longrollouts-2, global4deg)"):
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
    """double_checkpoint, unchanged from report-longrollouts-1 -- used here as the
    lead phase of rollout_temp_ma."""
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
        hist = hist.at[..., idx].set(n_state.variables.temp)
        idx = (idx + 1) % window
        return (n_state, hist, idx)
    return avg_step


def rollout_temp_ma(step_fn, state, iterations, lead_chunk_size, tail_chunk_size, window=TEMP_MA_WINDOW):
    """Differentiable path: lead phase (plain double_checkpoint, no averaging) +
    tail phase (double_checkpoint over the carry (state, hist, idx), fixed length
    `window`). See module docstring for why the split keeps averaging's memory
    cost constant in n. NaN if iterations < window (buffer never fills) --
    matches mld_ma's own "plain mean, not nanmean" convention; not expected to be
    hit in this report's sweep (n values are all >= window)."""
    lead = iterations - window
    if lead < 0:
        raise ValueError(f"iterations ({iterations}) must be >= window ({window})")
    if lead > 0:
        state = rollout(step_fn, state, lead, lead_chunk_size)

    temp_shape = state.variables.temp.shape
    hist0 = jnp.full(temp_shape + (window,), jnp.nan)
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
    temp_ma = jnp.mean(hist, axis=-1)
    return state, temp_ma


def plain_forward_rollout(step_fn, state, iterations):
    """Forward-only, no checkpoint needed -- used to generate the target
    trajectory cheaply."""
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state


def plain_forward_rollout_temp_ma(step_fn, state, iterations, window=TEMP_MA_WINDOW):
    """Forward-only version of rollout_temp_ma -- no checkpoint needed (no
    backward pass), used to generate the target temp_ma."""
    lead = iterations - window
    if lead < 0:
        raise ValueError(f"iterations ({iterations}) must be >= window ({window})")
    if lead > 0:
        state = plain_forward_rollout(step_fn, state, lead)

    temp_shape = state.variables.temp.shape
    hist0 = jnp.full(temp_shape + (window,), jnp.nan)
    idx0 = jnp.array(0)
    avg_step = _make_avg_step(step_fn, window)
    (state, hist, idx), _ = jax.lax.scan(lambda c, _: (avg_step(c), None), (state, hist0, idx0), length=window)
    temp_ma = jnp.mean(hist, axis=-1)
    return state, temp_ma


def temp_ma_agg_function(temp_ma, target_temp_ma):
    return ((temp_ma - target_temp_ma) ** 2).sum()


def peak_gpu_memory_bytes():
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        stats = None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


# Output directory -- $STORE (falls back to $HOME), NOT the repo checkout (see
# report-longrollouts-1/common.py for the "g5k sync deleted a live log" incident
# this avoids).
STORE_DIR = os.path.join(os.environ.get("STORE", os.path.expanduser("~")), "report_longrollouts2_results")
os.makedirs(STORE_DIR, exist_ok=True)


def write_csv_incremental(rows, path):
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_worker(n, param, test_val, lead_chunk_size, tail_chunk_size, timeout_s=1800):
    """Launch worker.py as a fresh subprocess for one (n, param) config --
    one-jit-per-process discipline, also needed for peak-memory isolation. Kills
    the whole process group on timeout."""
    import subprocess
    import signal
    import time

    cmd = [
        sys.executable, f"{PRP}scripts/Reports/report-longrollouts-2/worker.py",
        "--n", str(n), "--param", param, "--test_val", str(test_val),
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
        return dict(n=n, param=param, lead_chunk_size=lead_chunk_size, tail_chunk_size=tail_chunk_size, status="TIMEOUT",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    if returncode != 0:
        tail_out = stdout[-1500:] if stdout else ""
        tail_err = stderr[-1500:] if stderr else ""
        print(f"{label} FAILED returncode={returncode} ({dt:.0f}s)\n--stdout--\n{tail_out}\n--stderr--\n{tail_err}", flush=True)
        return dict(n=n, param=param, lead_chunk_size=lead_chunk_size, tail_chunk_size=tail_chunk_size, status=f"CRASHED(rc={returncode})",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    result_line = [ln for ln in stdout.splitlines() if ln.startswith("RESULT")][-1]
    r = dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))
    print(f"{label} OK ({dt:.0f}s): compile={r['compile_time_s']} run={r['run_time_s']} grad={r['grad']}", flush=True)
    return dict(n=n, param=param, lead_chunk_size=lead_chunk_size, tail_chunk_size=tail_chunk_size, status="OK",
                compile_time_s=eval(r["compile_time_s"]), run_time_s=eval(r["run_time_s"]),
                grad=eval(r["grad"]),
                peak_mem_bytes=None if r["peak_mem_bytes"] == "None" else eval(r["peak_mem_bytes"]),
                subprocess_s=dt)
