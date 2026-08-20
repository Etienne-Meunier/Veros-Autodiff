"""Shared helpers for the mld-mini-2 gradient-check / tuning scripts (60-level variant).

Same structure/shape as scripts/Reports/report-mld-mini-1/common.py -- only the setup
class differs here: GlobalFlexibleMLDLearningSetup (nz=60, ETOPO5 topography, see
setups/global_4deg/global_4deg_mld_learning.py) instead of GlobalFourDegreeMLDMiniSetup
(nz=15). Same mld diagnostic underneath (get_index_mld/mld_from_index), same
mld_agg_function.
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


def spin_up_global4deg_mld_mini2(warmup_steps=200):
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
    for _ in tqdm(range(warmup_steps), desc="spin-up (mld-mini-2)"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def make_diff_step(g4d):
    def pure_step(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    # No inner jax.jit here: rollout() traces this once via jax.lax.scan and the
    # caller jits the whole loss/grad around it -- see
    # scripts/gradient_routines/README.md (scan_checkpoint: flat compile time vs
    # rollout length, unlike an unrolled loop).
    return jax.checkpoint(pure_step)


def set_vars(state, **values):
    n_state = state.copy()
    with n_state.variables.unlock():
        for name, value in values.items():
            setattr(n_state.variables, name, value)
    return n_state


def rollout(step_fn, state, iterations):
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state


def mld_agg_function(state, target_state):
    """Squared MLD error, masked to cells where MLD is well-defined in *both* states
    (mld_from_index returns NaN at land/degenerate columns -- see
    setups/global_4deg/global_4deg_mld_learning.py). NaN-safe: the "false" branch of
    each where is a literal 0.0 constant, not a function of mld -- no gradient-NaN risk.
    """
    mld = state.variables.mld
    target_mld = target_state.variables.mld
    valid = ~jnp.isnan(mld) & ~jnp.isnan(target_mld)
    diff = jnp.where(valid, mld - target_mld, 0.0)
    return (diff**2).sum()
