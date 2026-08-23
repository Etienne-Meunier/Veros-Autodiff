"""Shared helpers for report-mld-2 (2-phase gradient validation ahead of the MLD_MA
parameter-fitting report): does gsw+streamfunction (both now on in
global_4deg_mld_learning.py, see that module's docstring) survive differentiation
through (1) the mld_ma moving-average mechanism itself and (2) a long rollout.

Phase 1 (mld_ma-correctness): full setup (GlobalFlexibleMLDLearningSetup, nz=60,
ETOPO5, gsw+streamfunction), `mld_ma_window` shrunk via subclass so the exact-average
buffer fills after a handful of steps instead of 720 -- isolates "does mld_ma
differentiate correctly" from rollout length.

Phase 2 (long-horizon): mini setup (GlobalFourDegreeMLDMiniSetup, nz=15, cheap grid),
gsw+streamfunction flipped on via subclass (that setup still defaults to nonlin2 +
enable_streamfunction=False -- see setups/global_4deg/global_4deg_mld_learning_mini.py),
direct `mld` loss (not mld_ma) -- isolates "does the gradient survive many chaotic
timesteps" from the MA mechanism, which phase 1 already covers separately.
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
from veros import veros_routine
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup
from setups.global_4deg.global_4deg_mld_learning_mini import GlobalFourDegreeMLDMiniSetup
from tqdm import tqdm

TRUE_PARAMS = dict(c_k=0.1, c_eps=0.7)


def _make_small_window_setup(window):
    class _SmallWindowSetup(GlobalFlexibleMLDLearningSetup):
        mld_ma_window = window

    return _SmallWindowSetup


class GswOnlyFullGridSetup(GlobalFlexibleMLDLearningSetup):
    """Full grid (nz=60, ETOPO5) with gsw (eq_of_state_type=5) but streamfunction
    forced back off -- isolates whether gsw alone (without streamfunction) reproduces
    the n=200 gradient blow-up found with both on (see diag_n200_temp_loss.py)."""

    @veros_routine
    def set_parameter(self, state):
        GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.enable_streamfunction = False


class StreamOnlyFullGridSetup(GlobalFlexibleMLDLearningSetup):
    """Full grid (nz=60, ETOPO5) with streamfunction on but eq_of_state_type forced
    back to nonlin2 (3) -- isolates whether streamfunction alone (without gsw)
    reproduces the blow-up."""

    @veros_routine
    def set_parameter(self, state):
        GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.eq_of_state_type = 3


def spin_up_full_grid(setup_cls, warmup_steps=20, desc="spin-up (mld-2, full grid)"):
    """Generic full-grid spin-up for any GlobalFlexibleMLDLearningSetup variant
    (plain, small-window, gsw-only, streamfunction-only)."""
    g4d = setup_cls()
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


def spin_up_phase1(window, warmup_steps=20):
    """Full grid (nz=60, gsw+streamfunction), mld_ma_window shrunk to `window`."""
    setup_cls = _make_small_window_setup(window)
    g4d = setup_cls()
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
    for _ in tqdm(range(warmup_steps), desc=f"spin-up (mld-2 phase1, window={window})"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


class _MiniStreamSetup(GlobalFourDegreeMLDMiniSetup):
    # Flips gsw+streamfunction on top of the mini setup's default (nonlin2,
    # enable_streamfunction=False) -- same "call parent body, then flip the flag"
    # pattern as scripts/debugging_stream/01_forward_sanity_check.py's ACCStreamSetup,
    # needed because these are set as plain assignments inside set_parameter's body,
    # not overridable class attributes.
    @veros_routine
    def set_parameter(self, state):
        GlobalFourDegreeMLDMiniSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.enable_streamfunction = True
            state.settings.eq_of_state_type = 5


def spin_up_phase2(warmup_steps=20):
    """Mini grid (nz=15), gsw+streamfunction flipped on via _MiniStreamSetup."""
    g4d = _MiniStreamSetup()
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
    for _ in tqdm(range(warmup_steps), desc="spin-up (mld-2 phase2, mini+stream+gsw)"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def make_diff_step(g4d):
    def pure_step(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    # No jax.checkpoint here: it's applied per-chunk in rollout() instead (see that
    # function's docstring for why per-step checkpointing doesn't bound memory).
    return pure_step


def set_vars(state, **values):
    n_state = state.copy()
    with n_state.variables.unlock():
        for name, value in values.items():
            setattr(n_state.variables, name, value)
    return n_state


# Steps per checkpointed chunk. This bounds the backward pass's peak memory --
# roughly chunk_size * (one step's activation memory), a property of the *grid*
# (nz=60 full grid vs. nz=15 mini grid) and the hardware's memory budget, not of
# `iterations`. Verified safe on the full grid at n=6/12/16 (16GB P100); tune up for
# a cheaper grid or more memory, down if you still OOM.
DEFAULT_CHECKPOINT_CHUNK_SIZE = 4


def rollout(step_fn, state, iterations, chunk_size=DEFAULT_CHECKPOINT_CHUNK_SIZE):
    """Scan-of-checkpointed-chunks: reverse-mode through a plain jax.lax.scan of
    length n stores the full state carry at every one of the n iterations (a
    per-step jax.checkpoint only trims each step's *internal* temporaries, not that
    O(n) carry storage -- this is what OOM'd at n=400 on a 16GB P100, see
    Results/Report/report-mld-2.md). Splitting into ceil(n/chunk_size) chunks and
    checkpointing each chunk means only the chunk-boundary carries stay live for the
    backward pass.

    Always chunked, no small-n skip: per-step memory cost depends on the grid, which
    iteration count alone can't tell you -- n=12 unchunked OOM'd on the full grid
    despite n=100 unchunked being fine on the mini grid. Pass an explicit chunk_size
    tuned to your grid/hardware rather than relying on the default, which is just a
    verified-safe starting point for the full grid.
    """
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
    """Squared error on the instantaneous mld, masked to cells valid in both states.
    Same NaN-safe pattern as scripts/Reports/report-mld-1/common.py."""
    return _masked_sq_error(state.variables.mld, target_state.variables.mld)


def mld_ma_agg_function(state, target_state):
    """Squared error on mld_ma instead -- same masking pattern (mld_ma is NaN at land
    the same way mld is, see setups/global_4deg/global_4deg_mld_learning.py)."""
    return _masked_sq_error(state.variables.mld_ma, target_state.variables.mld_ma)
