# %%
# Diagnostic (not part of the formal report): full-grid (nz=60, gsw+streamfunction),
# window=12, n=200 gave autodiff grad=1.25e21 vs finite-difference num_grad=-1.82e6
# (eps=1e-4) -- 15 orders of magnitude apart, opposite sign. Same triage as
# diag_wraparound_eps_sweep.py: sweep eps to tell apart "FD isn't resolving a real,
# razor-thin near-singular local derivative" (rel_err should shrink as eps shrinks,
# converging toward the autodiff value) from "autodiff itself is wrong" (FD stays flat
# near its own value regardless of eps, never approaches the autodiff number).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import spin_up_phase1, make_diff_step, set_vars, rollout, mld_ma_agg_function

WINDOW = 12
N = 200
PARAM, TEST_VAL = "c_k", 0.08

g4d, step_jit = spin_up_phase1(WINDOW, warmup_steps=20)
target_state = rollout(step_jit, g4d.state, N)


def loss(v):
    n_state = set_vars(g4d.state, **{PARAM: v})
    n_state = rollout(make_diff_step(g4d), n_state, N)
    return mld_ma_agg_function(n_state, target_state)


loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.value_and_grad(loss))

loss_val, grad = grad_jit(jnp.array(TEST_VAL))
print(f"n={N}  loss={float(loss_val):.6e}  autodiff_grad={float(grad):.6e}", flush=True)

for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
    num_grad = (loss_jit(jnp.array(TEST_VAL) + eps) - loss_jit(jnp.array(TEST_VAL) - eps)) / (2 * eps)
    rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
    print(f"eps={eps:.0e}  numerical={float(num_grad):.6e}  rel_err={rel_err:.4e}", flush=True)
