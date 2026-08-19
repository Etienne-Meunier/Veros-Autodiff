"""Shared helpers for the Results/Report figure-generating scripts (global_4deg only)."""
from __init__ import PRP
import sys
sys.path.append(PRP + 'veros/')

from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros (must run before veros.core imports)
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup
from tqdm import tqdm

TRUE_PARAMS = dict(K_gm_0=1000.0, r_bot=1e-5, c_k=0.1, c_eps=0.7)


def spin_up_global4deg(warmup_steps=200):
    g4d = GlobalFourDegreeSetup()
    g4d.setup()
    with g4d.state.settings.unlock():
        g4d.state.settings.enable_eke = False
        # global_4deg_learning.py leaves bottom friction off by default (r_bot inactive,
        # gradient identically 0) -- enabled here so r_bot is a meaningful report parameter.
        g4d.state.settings.enable_bottom_friction = True

    with g4d.state.variables.unlock():
        g4d.state.variables.r_bot += TRUE_PARAMS["r_bot"]
        g4d.state.variables.K_gm_0 += TRUE_PARAMS["K_gm_0"]
        g4d.state.variables.c_k += 0.0
        g4d.state.variables.c_eps += 0.0

    def ps(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    step_jit = jax.jit(ps)

    state = g4d.state.copy()
    for _ in tqdm(range(warmup_steps), desc="spin-up"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def make_diff_step(g4d):
    def pure_step(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state
    return jax.checkpoint(jax.jit(pure_step))


def set_vars(state, **values):
    n_state = state.copy()
    with n_state.variables.unlock():
        for name, value in values.items():
            setattr(n_state.variables, name, value)
    return n_state


def rollout(step_fn, state, iterations):
    for _ in range(iterations):
        state = step_fn(state)
    return state
