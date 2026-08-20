# %%
# 02_rbot_grad_check.py's n=2 case disagreed badly (rel_err~1.0) while n=5 agreed well
# (rel_err~1.5e-5). Hypothesis: at n=2 the loss (~1e15) is huge relative to grad*eps
# (~1e3-1e4), so central finite difference is cancelling two ~1e15 floats to recover a
# ~1e3 signal -- float64 noise, not a real autodiff bug. Sweep eps: if rel_err shrinks
# and plateaus as eps grows (before truncation error takes back over at large eps),
# that confirms cancellation noise. If it stays ~1 regardless of eps, that's a real bug.
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

step_fn = pure_step(acc)
base_state = acc.state

n = 2
test_val = 1e-5


def loss(v, n=n):
    n_state = set_var(base_state, "r_bot", jnp.full_like(base_state.variables.r_bot, v))
    n_state = rollout(step_fn, n_state, n)
    return agg_sum_sq(n_state)


loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.value_and_grad(loss))

loss_val, grad = grad_jit(jnp.array(test_val))
print(f"n={n}  loss={float(loss_val):.6e}  autodiff_grad={float(grad):.6e}")

for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
    num_grad = (loss_jit(jnp.array(test_val) + eps) - loss_jit(jnp.array(test_val) - eps)) / (2 * eps)
    rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
    print(f"eps={eps:.0e}  numerical={float(num_grad):.6e}  rel_err={rel_err:.4e}")
