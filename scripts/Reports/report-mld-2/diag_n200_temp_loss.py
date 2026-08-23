# %%
# Diagnostic (not part of the formal report): full-grid n=200 mld_ma loss gave
# autodiff grad=1.25e21 vs a reliable finite-difference plateau of ~-2e6 (see
# diag_n200_fullgrid_eps_sweep.py) -- a real autodiff bug, not an FD resolution issue.
# This swaps the loss to plain squared temp error (same pattern as
# scripts/Reports/report-1/section3b_ck_ceps_scenario.py's agg_function), which never
# touches get_index_mld/mld_from_index/mld_ma at all -- same setup (full grid nz=60,
# gsw+streamfunction), same n=200, same param. If THIS is clean, the bug is in the MLD
# kernel (most likely mld_from_index's `denom = prho_above - prho_below`, only guarded
# at exact degeneracy -- a near-zero-but-nonzero denom at some real ETOPO5 column would
# blow up 1/denom in the backward pass without ever showing as NaN forward). If this is
# ALSO broken, the bug is upstream in gsw+streamfunction's core differentiability, not
# the MLD formula.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import spin_up_phase1, make_diff_step, set_vars, rollout

WINDOW = 12  # irrelevant here -- temp loss never touches mld/mld_ma
N = 200
PARAM, TEST_VAL = "c_k", 0.08

g4d, step_jit = spin_up_phase1(WINDOW, warmup_steps=20)
target_state = rollout(step_jit, g4d.state, N)


def temp_agg_function(state, target_state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()


def loss(v):
    n_state = set_vars(g4d.state, **{PARAM: v})
    n_state = rollout(make_diff_step(g4d), n_state, N)
    return temp_agg_function(n_state, target_state)


loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.value_and_grad(loss))

loss_val, grad = grad_jit(jnp.array(TEST_VAL))
print(f"n={N}  temp-loss={float(loss_val):.6e}  autodiff_grad={float(grad):.6e}", flush=True)

for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    num_grad = (loss_jit(jnp.array(TEST_VAL) + eps) - loss_jit(jnp.array(TEST_VAL) - eps)) / (2 * eps)
    rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
    print(f"eps={eps:.0e}  numerical={float(num_grad):.6e}  rel_err={rel_err:.4e}", flush=True)
