# %%
# Diagnostic (not part of the formal report): phase1 found rel_err jumping to ~64% at
# n=16 (window=12, i.e. past the first buffer wrap-around) vs ~0.1-0.2% at n=6/n=12.
# Same triage as scripts/debugging_stream/03_n2_eps_sweep.py: sweep eps at fixed n to
# tell apart finite-difference cancellation noise (rel_err shrinks/plateaus as eps
# grows) from a real gradient bug (rel_err stays ~constant regardless of eps).
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
N = WINDOW + 4
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
print(f"n={N}  loss={float(loss_val):.6e}  autodiff_grad={float(grad):.6e}")

for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    num_grad = (loss_jit(jnp.array(TEST_VAL) + eps) - loss_jit(jnp.array(TEST_VAL) - eps)) / (2 * eps)
    rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
    print(f"eps={eps:.0e}  numerical={float(num_grad):.6e}  rel_err={rel_err:.4e}")
