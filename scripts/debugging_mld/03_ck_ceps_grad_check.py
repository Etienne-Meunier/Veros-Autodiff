# %%
# Regression check: d(mld loss)/d(c_k), d(mld loss)/d(c_eps) through the real
# differentiable step (set_forcing_kernel incl. the solar-penetration temp_source term,
# TKE/EKE physics, thermodynamics, after_timestep's mld_from_prho), autodiff vs central
# finite difference, for n=1,2,5-step rollouts. Short warmup (not the full 200-step
# spin-up used by the actual report scripts) to keep this cheap -- this script only
# needs *a* non-trivial state, not a physically realistic one.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

sys.path.append(PRP)

sys.path.append(PRP + "scripts/Reports/report-mld-1/")
from common import spin_up_global4deg_mld, make_diff_step, set_vars, rollout, mld_agg_function, TRUE_PARAMS

g4d, step_jit = spin_up_global4deg_mld(warmup_steps=5)
step_diff = make_diff_step(g4d)

PARAM_CONFIG = {
    "c_k": (0.08, 1e-4),
    "c_eps": (0.6, 1e-4),
}
n_values = [1, 2, 5]

for n in n_values:
    target_state = rollout(step_jit, g4d.state, n)

    for name, (test_val, eps) in PARAM_CONFIG.items():

        def loss(v, name=name, n=n, target_state=target_state):
            n_state = set_vars(g4d.state, **{name: v})
            n_state = rollout(step_diff, n_state, n)
            return mld_agg_function(n_state, target_state)

        loss_jit = jax.jit(loss)
        grad_jit = jax.jit(jax.value_and_grad(loss))

        loss_val, grad = grad_jit(jnp.array(test_val))
        num_grad = (loss_jit(jnp.array(test_val) + eps) - loss_jit(jnp.array(test_val) - eps)) / (2 * eps)
        rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
        print(
            f"n={n:2d}  param={name:6s}  loss={float(loss_val):.6e}  "
            f"autodiff={float(grad):.6e}  numerical={float(num_grad):.6e}  "
            f"rel_err={rel_err:.4e}  nan_in_grad={bool(jnp.isnan(grad))}"
        )

        del loss_jit, grad_jit
        jax.clear_caches()
