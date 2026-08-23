"""report-longrollouts-3: actual gradient-descent parameter tuning of c_k/c_eps
against a temp-squared-error loss at n=2000 on global4deg -- the last horizon
report-longrollouts-1 confirmed gives sane, smoothly-growing gradients
(grad=-385.2 for c_k at n=2000, chunk_size=32; n=3000 was already the first sign
of anomaly there). Proof of concept: when gradients ARE trustworthy, do they
actually drive an optimizer to the right answer? Same double_checkpoint rollout,
same setup, self-contained copy of report-longrollouts-1/common.py's relevant
pieces (repo convention: each report is standalone).
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

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup
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
    for _ in tqdm(range(warmup_steps), desc="spin-up (longrollouts-3, global4deg)"):
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


def temp_agg_function(state, target_state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()


def plain_forward_rollout(step_fn, state, iterations):
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state


def rollout(step_fn, state, iterations, chunk_size):
    """double_checkpoint -- unchanged from report-longrollouts-1."""
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


STORE_DIR = os.path.join(os.environ.get("STORE", os.path.expanduser("~")), "report_longrollouts3_results")
os.makedirs(STORE_DIR, exist_ok=True)


def write_csv_incremental(rows, path):
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
