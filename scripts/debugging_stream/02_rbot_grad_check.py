# %%
# Grad-check enable_streamfunction=True through the real differentiable step (prepare_forcing
# + solve_stream's linear solve + barotropic_velocity_update, INCLUDING the island-correction
# branch -- 01_forward_sanity_check.py found nisle=2 on ACC, so this actually exercises
# line_integrals / the line_psin solve, not just the plain interior Poisson solve).
# autodiff vs central finite difference, d(loss)/d(r_bot), for n=1,2,5-step rollouts.
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
from setups.acc.acc_learning import ACCSetup


class ACCStreamSetup(ACCSetup):
    @veros_routine
    def set_parameter(self, state):
        ACCSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.enable_streamfunction = True


def pure_step(acc):
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


def rollout(step_fn, state, n):
    for _ in range(n):
        state = step_fn(state)
    return state


def agg_sum_sq(state, var_name="psi"):
    return (getattr(state.variables, var_name) ** 2).sum()


acc = ACCStreamSetup()
acc.setup()
print(f"nisle = {acc.state.dimensions['isle']}")

step_fn = pure_step(acc)
base_state = acc.state

PARAM_CONFIG = {
    "r_bot": (1e-5, 1e-8),
}
n_values = [1, 2, 5]

for n in n_values:
    for name, (test_val, eps) in PARAM_CONFIG.items():

        def loss(v, name=name, n=n):
            n_state = set_var(base_state, name, jnp.full_like(getattr(base_state.variables, name), v))
            n_state = rollout(step_fn, n_state, n)
            return agg_sum_sq(n_state)

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
