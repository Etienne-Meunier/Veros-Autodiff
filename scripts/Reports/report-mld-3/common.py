"""Shared helpers for report-mld-3: gradient-accuracy-vs-rollout-length probe for
direct `mld` loss (not mld_ma) on the real full grid (GlobalFlexibleMLDLearningSetup,
nz=60, ETOPO5, gsw/TEOS-10, streamfunction -- the exact config report-mld-mini-2's
n=5 (c_k, c_eps) recovery used). Precaution requested ahead of extending that
recovery scenario to longer rollouts (n=20/75/250, mirroring report-2's temp-loss
sweep): report-mld-2 phase2 only checked direct-mld gradient accuracy on the cheap
mini grid (nz=15) up to n=900, not on this heavier full grid, and mld_ma (a related
but distinct diagnostic) is known to blow up much earlier than temp on this same full
grid (report-longrollouts-4). This checks whether raw `mld`'s gradient is trustworthy
at the n values the recovery sweep would actually use, before spending compute on the
full GD runs.

The `mld` probe (run_probe.py) found the gradient breaks by n=20 (rel_err vs finite
difference already O(1), unphysical by n=75) -- far earlier than temp on `global4deg`
(sane to n~2000-3000) or even the mini-grid (nz=15) `mld` check (1.7% error at n=900).
run_probe_temp.py runs the identical probe with `temp_agg_function` instead of
`mld_agg_function` on this exact full config, to separate "is it the real
config/streamfunction/gsw/topo that's fragile" from "is it the `mld` diagnostic's own
formula (division near weak stratification, see report-longrollouts-4's note [4])."
"""
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup
from tqdm import tqdm

TRUE_PARAMS = dict(c_k=0.1, c_eps=0.7)


def spin_up_full_grid(warmup_steps=20):
    """Full grid (nz=60, ETOPO5, gsw+streamfunction) -- same setup as
    report-mld-mini-2's recovery scenario, unmodified (no window override, no flag
    flips)."""
    g4d = GlobalFlexibleMLDLearningSetup()
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
    for _ in tqdm(range(warmup_steps), desc="spin-up (mld-3, full grid)"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def make_diff_step(g4d):
    def pure_step(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    # No jax.checkpoint here: applied per-chunk in rollout() instead (see
    # report-mld-2/common.py's rollout() docstring for why per-step checkpointing
    # alone doesn't bound memory over a long scan).
    return pure_step


def set_vars(state, **values):
    n_state = state.copy()
    with n_state.variables.unlock():
        for name, value in values.items():
            setattr(n_state.variables, name, value)
    return n_state


# Verified safe on this exact full grid (nz=60) at n=6/12/16 in report-mld-2 phase1;
# reused as-is here since it bounds memory independent of total rollout length n.
DEFAULT_CHECKPOINT_CHUNK_SIZE = 4


def rollout(step_fn, state, iterations, chunk_size=DEFAULT_CHECKPOINT_CHUNK_SIZE):
    """Scan-of-checkpointed-chunks -- see report-mld-2/common.py's rollout() for the
    full OOM-avoidance rationale (a plain jax.lax.scan stores the full state carry at
    every iteration; chunked checkpointing only keeps chunk-boundary carries live)."""
    n_full, remainder = divmod(iterations, chunk_size)

    def run_chunk(state, length):
        state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=length)
        return state

    if n_full:
        checkpointed_chunk = jax.checkpoint(lambda s: run_chunk(s, chunk_size))
        state, _ = jax.lax.scan(lambda c, _: (checkpointed_chunk(c), None), state, length=n_full)
    if remainder:
        state = jax.checkpoint(lambda s: run_chunk(s, remainder))(state)

    return state


def _masked_sq_error(field, target_field):
    valid = ~jnp.isnan(field) & ~jnp.isnan(target_field)
    diff = jnp.where(valid, field - target_field, 0.0)
    return (diff**2).sum()


def mld_agg_function(state, target_state):
    """Squared error on the instantaneous mld, masked to cells valid in both states
    -- same NaN-safe pattern as report-mld-2/common.py."""
    return _masked_sq_error(state.variables.mld, target_state.variables.mld)


def temp_agg_function(state, target_state):
    """Squared error on full-field temp, unmasked -- same form as report-1/report-2's
    temp loss (temp is zero outside maskT, not NaN, so no masking needed)."""
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()
