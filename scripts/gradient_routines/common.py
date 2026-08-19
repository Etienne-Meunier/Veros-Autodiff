"""Shared setup for the gradient-routine benchmarks: fast ACC spin-up + step/var helpers."""
from __init__ import PRP
import sys
sys.path.append(PRP + 'veros/')

from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros (must run before veros.core imports)
from setups.acc.acc_learning import ACCSetup
from tqdm import tqdm

TRUE_R_BOT = 1e-5


def spin_up_acc(warmup_steps=200):
    """ACC is the small (30x42x15) toy setup -- fast enough to roll out several hundred
    steps in a benchmark, unlike global_4deg."""
    acc = ACCSetup()
    acc.setup()
    with acc.state.settings.unlock():
        acc.state.settings.enable_eke = False

    with acc.state.variables.unlock():
        acc.state.variables.r_bot += TRUE_R_BOT

    def ps(state):
        n_state = state.copy()
        acc.step(n_state)
        return n_state

    step_jit = jax.jit(ps)

    state = acc.state.copy()
    for _ in tqdm(range(warmup_steps), desc="spin-up"):
        state = step_jit(state)
    acc.state = state

    return acc


def pure_step(acc):
    """acc.step mutates state in place -- wrap it into a pure state -> state function."""
    def ps(state):
        n_state = state.copy()
        acc.step(n_state)
        return n_state
    return ps


def set_var(state, var_name, value):
    n_state = state.copy()
    with n_state.variables.unlock():
        setattr(n_state.variables, var_name, value)
    return n_state


def agg_sum_sq(state, var_name='temp'):
    return (getattr(state.variables, var_name) ** 2).sum()
