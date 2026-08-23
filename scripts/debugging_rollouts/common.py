"""Shared helpers for scripts/debugging_rollouts -- root-causing the compile-time host
RAM OOM (SIGKILL/exit 137) hit by report-longrollouts-1's rollout() at even the
smallest config tried (n=10, chunk_size=4, nz=15, on a Tesla P100-16GB, see
Results/Report/report-longrollouts-1.md once written). nvidia-smi showed the GPU
untouched (0/16384MiB) the whole time -- this is XLA blowing up *compile-time* host
memory while building the backward graph, not a runtime activation-memory problem.

Suspected cause: grad() of a checkpointed jax.lax.scan nested inside another
jax.lax.scan (scan-of-checkpointed-scanned-chunks). Nesting two control-flow
primitives' own VJP rules under jax.checkpoint's custom VJP, around a per-step graph
that's already hundreds of ops (one full Veros timestep: TKE, EKE, isoneutral
diffusion, streamfunction pressure solve with island BCs), may be what makes XLA's
HLO explode before it ever reaches the GPU.

STRATEGIES below cover the 2x2 space of {outer: scan vs Python-unrolled} x
{inner (per-chunk): scan vs Python-unrolled}, plus a "plain" unchunked/uncheckpointed
baseline. sweep.py measures compile time and run time (split via
jax.jit(...).lower(...).compile() vs the actual call, see worker.py) for each cell,
at a short and a long n, to find which structure is both compile-tractable and
memory-bounded -- before report-longrollouts-1 trusts any of them at n=500..3000.

Setup: nz=15 on GlobalFlexibleMLDLearningSetup (see
report-longrollouts-1/common.py's module docstring for why this grid/setup, and why
temp loss not mld/mld_ma). Duplicated here rather than imported cross-directory --
report dirs in this repo don't share code across each other's `common.py` (hyphenated
dir names aren't importable as packages anyway); same convention as
report-mld-2/report-mld-1 each carrying their own near-identical helpers.
"""
from __init__ import PRP
import sys
import os

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

# Persistent XLA compilation cache -- compiled executables survive across process
# runs (each sweep config is its own fresh subprocess, one-jit-per-process
# discipline, so without this every config recompiles from scratch even on a rerun).
# Keyed by the HLO program's hash, so this only saves time on an EXACT repeat of a
# (setup, strategy, n, chunk_size) config, not across different ones -- but repeats
# happen constantly here (crashed/timed-out configs rerun after a code fix, sweep
# rerun after extending N_VALUES, etc). $STORE (falls back to $HOME) rather than
# somewhere under the repo checkout: this directory must NOT be inside the
# unison-synced Veros-Autodiff tree, or `g5k sync code` will try to reconcile/delete
# cache files it doesn't recognize (it already did this once to stray log files).
JAX_CACHE_DIR = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(
    os.environ.get("STORE", os.path.expanduser("~")), "jax_cache_veros"
)
os.makedirs(JAX_CACHE_DIR, exist_ok=True)
config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
config.update("jax_persistent_cache_min_compile_time_secs", 5)

import jax

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from veros import veros_routine
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup
from tqdm import tqdm


def make_nz_setup(nz):
    class _NzSetup(GlobalFlexibleMLDLearningSetup):
        @veros_routine
        def set_parameter(self, state):
            GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
            with state.settings.unlock():
                state.settings.nz = nz

    return _NzSetup


def _spin_up_generic(g4d, warmup_steps, desc):
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
    for _ in tqdm(range(warmup_steps), desc=desc):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def spin_up(nz, warmup_steps=20):
    """nz-flipped GlobalFlexibleMLDLearningSetup (ETOPO5 topography interpolated onto
    nz levels, gsw eq_of_state + streamfunction on). This is the config that gave
    grad=nan at nz=15 -- see spin_up_global4deg below for the isolation test."""
    return _spin_up_generic(make_nz_setup(nz)(), warmup_steps, f"spin-up (debugging_rollouts, flexible nz={nz})")


def spin_up_global4deg(warmup_steps=20):
    """GlobalFourDegreeSetup (global_4deg_learning.py) -- native nz=15,
    assets.json-based bathymetry (not ETOPO5-interpolated), nonlin2 EOS,
    enable_streamfunction=False (see global_4deg_mld_learning_mini.py's docstring for
    this setup's defaults). This is report-1's setup, which never produced NaN
    gradients -- run the identical plain/n=20/c_k grad check here to isolate whether
    NaN at nz=15 comes from the ETOPO5-onto-15-levels topography interaction / the
    gsw+streamfunction physics, or is something more generic that report-1 just never
    happened to trigger."""
    return _spin_up_generic(GlobalFourDegreeSetup(), warmup_steps, "spin-up (debugging_rollouts, global4deg)")


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
    target_state cheaply for every strategy below, so the sweep measures only the
    grad-path's compile/run cost, not target-state generation."""
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state


def _run_chunk_scan(step_fn, state, length):
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=length)
    return state


def _run_chunk_unrolled(step_fn, state, length):
    for _ in range(length):
        state = step_fn(state)
    return state


def rollout_plain(step_fn, state, iterations, chunk_size=None):
    """No checkpoint, no chunking -- plain jax.lax.scan. Reference baseline: this is
    what OOM'd at *runtime* (n=400) in report-mld-2, but is the cheapest thing to
    compile. chunk_size ignored, kept for a uniform call signature across strategies."""
    return plain_forward_rollout(step_fn, state, iterations)


def rollout_split_transpose(step_fn, state, iterations, chunk_size=None):
    """Single unchunked lax.scan, no manual chunk/checkpoint nesting at all --
    suggested externally in response to the scan_scan/scan_unrolled compile-time
    blowup (see module docstring). Three specific knobs instead of manual chunking:
      - per-step jax.checkpoint with policy=nothing_saveable: maximal recompute, save
        literally nothing extra (the "extreme checkpointing" this repo's rollout()
        chunking was trying to approximate manually via chunk boundaries).
      - prevent_cse=False: checkpoint defaults to blocking XLA's common-subexpression
        elimination between the original and recomputed forward passes (so CSE can't
        silently defeat the memory saving by merging them back together) -- that
        CSE-blocking barrier may itself be adding compile complexity; disabling it
        trades a (to be verified) memory-saving guarantee for simpler HLO.
      - scan's _split_transpose=True: restructures how lax.scan's own reverse-mode
        transpose is built. Per-step checkpoint alone doesn't fix scan's O(n)
        carry-history requirement (see rollout_scan_scan's docstring) -- this is a
        scan-internal option that might, since chunking from outside scan clearly
        can't (that's what blew up compile time here). chunk_size ignored, kept for a
        uniform call signature across strategies.
    """
    cp_step = jax.checkpoint(step_fn, policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)
    state, _ = jax.lax.scan(lambda c, _: (cp_step(c), None), state, None, length=iterations, _split_transpose=True)
    return state


def rollout_scan_scan(step_fn, state, iterations, chunk_size):
    """CURRENT report-longrollouts-1 design: scan(checkpoint(scan(step))). The one
    that SIGKILL'd at compile time (n=10, chunk_size=4, nz=15) -- included here as the
    known-bad reference point, not because it's expected to pass."""
    n_full, remainder = divmod(iterations, chunk_size)

    if n_full:
        checkpointed_chunk = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, chunk_size))
        state, _ = jax.lax.scan(lambda c, _: (checkpointed_chunk(c), None), state, length=n_full)
    if remainder:
        state = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, remainder))(state)

    return state


def rollout_double_checkpoint(step_fn, state, iterations, chunk_size):
    """Double-checkpoint pattern, externally suggested: per-step checkpoint(step)
    inside the inner scan, PLUS an outer checkpoint(block_fn) wrapping that whole
    inner scan, itself scanned at the outer level --

        step_ckpt = checkpoint(step)
        block_fn(s) = scan(step_ckpt, s, length=chunk_size)
        block_ckpt = checkpoint(block_fn)
        rollout = scan(block_ckpt, state, length=n_full)

    Not redundant with rollout_split_transpose's per-step-only checkpoint: the outer
    checkpoint controls whether the block's internal *carry chain* (chunk_size steps'
    worth) is exposed to the OUTER scan's own backward bookkeeping at all, which is
    exactly the O(n) growth split_transpose couldn't avoid (see that function's
    docstring -- confirmed empirically: memory grew ~linearly n=20->100, crashed
    predictably around n=400 on the 16GB GPU). Differs from rollout_scan_scan (which
    also wraps checkpoint around an inner scan) in one way: scan_scan's inner scan
    runs the RAW, uncheckpointed step_fn -- this one's inner scan runs the
    already-checkpointed step_ckpt. Hypothesis: exposing the outer checkpoint's
    compile to a scan of opaque (already-checkpointed) steps instead of a scan of raw
    physics may be what actually made scan_scan blow up at compile time -- untested
    until now."""
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


def rollout_scan_unrolled(step_fn, state, iterations, chunk_size):
    """Proposed fix: scan(checkpoint(python_loop(step))) -- only ONE lax.scan total
    (the outer one over chunks), inner chunk body is a short Python-unrolled loop
    instead of a second scan. Keeps the outer compile flat in n_full (the property
    that makes scan attractive for n=500..3000) while removing the nested-scan-under-
    grad structure suspected of the blowup."""
    n_full, remainder = divmod(iterations, chunk_size)

    if n_full:
        checkpointed_chunk = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, chunk_size))
        state, _ = jax.lax.scan(lambda c, _: (checkpointed_chunk(c), None), state, length=n_full)
    if remainder:
        state = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, remainder))(state)

    return state


def rollout_unrolled_scan(step_fn, state, iterations, chunk_size):
    """python_loop(checkpoint(scan(step))) -- outer chunk-of-chunks loop is Python-
    unrolled (n_full separate traced calls, compile time expected to scale with
    n_full), inner chunk body stays a lax.scan. Isolates whether the inner scan alone
    (without an outer scan wrapping it under grad) is fine."""
    n_full, remainder = divmod(iterations, chunk_size)

    checkpointed_chunk = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, chunk_size)) if n_full else None
    for _ in range(n_full):
        state = checkpointed_chunk(state)
    if remainder:
        state = jax.checkpoint(lambda s: _run_chunk_scan(step_fn, s, remainder))(state)

    return state


def rollout_unrolled_unrolled(step_fn, state, iterations, chunk_size):
    """python_loop(checkpoint(python_loop(step))) -- no lax.scan anywhere. Compile
    time expected to scale with n_full (no flat-compile benefit at all), but if this
    is the only one that doesn't blow up host RAM, that tells us the problem is
    lax.scan's own VJP under nesting, not checkpoint or chunking per se."""
    n_full, remainder = divmod(iterations, chunk_size)

    checkpointed_chunk = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, chunk_size)) if n_full else None
    for _ in range(n_full):
        state = checkpointed_chunk(state)
    if remainder:
        state = jax.checkpoint(lambda s: _run_chunk_unrolled(step_fn, s, remainder))(state)

    return state


STRATEGIES = {
    "plain": rollout_plain,
    "split_transpose": rollout_split_transpose,
    "scan_scan": rollout_scan_scan,
    "double_checkpoint": rollout_double_checkpoint,
    "scan_unrolled": rollout_scan_unrolled,
    "unrolled_scan": rollout_unrolled_scan,
    "unrolled_unrolled": rollout_unrolled_unrolled,
}


def peak_gpu_memory_bytes():
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        stats = None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


# Output directory for driver scripts (phase1_correctness.py, phase2_scaling.py) --
# $STORE (falls back to $HOME), NOT under the repo checkout. A stray `g5k sync code`
# while a run was live once deleted its log file out from under it (unison reconciles
# away anything that doesn't exist locally) and nearly lost an in-progress sweep's
# results. Same rationale as JAX_CACHE_DIR above.
STORE_DIR = os.path.join(os.environ.get("STORE", os.path.expanduser("~")), "debugging_rollouts_results")
os.makedirs(STORE_DIR, exist_ok=True)


def write_csv_incremental(rows, path):
    """Rewrite the whole CSV after every completed config -- with the handful of rows
    these driver scripts produce this is cheap, and it means a kill/crash mid-run
    loses nothing (previously the CSV was only written once at the very end, so
    killing the sweep lost every already-completed result)."""
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_worker(strategy, n, chunk_size, setup="global4deg", timeout_s=600):
    """Launch worker.py as a fresh subprocess for one (strategy, n, chunk_size)
    config -- one-jit-per-process discipline, also needed for memory isolation (see
    worker.py/peak_gpu_memory_bytes docstrings). Kills the whole process group on
    timeout so a stalled/blowing-up compile can't hold up the rest of a driver
    script's loop. Returns a dict with status ("OK"/"TIMEOUT"/"CRASHED(rc=...)") and,
    on OK, compile_time_s/run_time_s/grad/peak_mem_bytes parsed from worker.py's
    RESULT line."""
    import subprocess
    import signal
    import time

    cmd = [
        sys.executable, f"{PRP}scripts/debugging_rollouts/worker.py",
        "--strategy", strategy, "--setup", setup, "--n", str(n), "--chunk_size", str(chunk_size),
    ]
    label = f"[{strategy}][n={n}][chunk={chunk_size}]"
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
        return dict(strategy=strategy, n=n, chunk_size=chunk_size, status="TIMEOUT",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    if returncode != 0:
        tail_out = stdout[-1500:] if stdout else ""
        tail_err = stderr[-1500:] if stderr else ""
        print(f"{label} FAILED returncode={returncode} ({dt:.0f}s)\n--stdout--\n{tail_out}\n--stderr--\n{tail_err}", flush=True)
        return dict(strategy=strategy, n=n, chunk_size=chunk_size, status=f"CRASHED(rc={returncode})",
                    compile_time_s=None, run_time_s=None, grad=None, peak_mem_bytes=None, subprocess_s=dt)

    result_line = [ln for ln in stdout.splitlines() if ln.startswith("RESULT")][-1]
    r = dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))
    print(f"{label} OK ({dt:.0f}s): compile={r['compile_time_s']} run={r['run_time_s']} grad={r['grad']}", flush=True)
    return dict(strategy=strategy, n=n, chunk_size=chunk_size, status="OK",
                compile_time_s=eval(r["compile_time_s"]), run_time_s=eval(r["run_time_s"]),
                grad=eval(r["grad"]),
                peak_mem_bytes=None if r["peak_mem_bytes"] == "None" else eval(r["peak_mem_bytes"]),
                subprocess_s=dt)
