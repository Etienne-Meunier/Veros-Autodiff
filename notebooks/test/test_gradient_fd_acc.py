# %%
# Script version of notebooks/demonstration/gradient-computation.ipynb :
# check the autodiff gradient dL/d(r_bot) on the ACC setup against a
# central finite-difference estimate.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros
from setups.acc.acc_learning import ACCSetup

from functools import partial
from tqdm import tqdm

# %%
# Spin-up
warmup_steps = 200
acc = ACCSetup()
acc.setup()
with acc.state.settings.unlock():
    acc.state.settings.enable_eke = False

with acc.state.variables.unlock():
    acc.state.variables.r_bot += 1e-5
    acc.state.variables.K_gm_0 += 1000.0

def ps(state):
    n_state = state.copy()
    acc.step(n_state)
    return n_state

step_jit = jax.jit(ps)

state = acc.state.copy()
for _ in tqdm(range(warmup_steps), desc="spin-up"):
    state = step_jit(state)

acc.state = state

# %%
# Same autodiff helper as in the notebook (rollout n_state through step_function)
class autodiff:
    def __init__(self, step_function, agg_function, var_name):
        self.agg_function = agg_function
        self.step_function = partial(autodiff.pure, step=step_function)
        self.var_name = var_name

    @staticmethod
    def pure(state, step):
        n_state = state.copy()
        step(n_state)  # step modifies n_state in place
        return n_state

    @staticmethod
    def set_var(var_name, state, var_value):
        n_state = state.copy()
        with n_state.variables.unlock():
            setattr(n_state.variables, var_name, var_value)
        return n_state

    def rollout(self, state, iterations):
        for _ in range(iterations):
            state = self.step_function(state)
        return state

    def loss(self, state, var_value, iterations):
        n_state = autodiff.set_var(self.var_name, state, var_value)
        n_state = self.rollout(n_state, iterations)
        return self.agg_function(n_state)

    def g(self, state, var_value, iterations=1):
        return jax.value_and_grad(lambda v: self.loss(state, v, iterations))(var_value)

# %%
def agg_function(state):
    return (state.variables.temp ** 2).sum()

var_name = "r_bot"
iterations = 2   # NB: with more rollout steps, central FD and autodiff start to
                 # disagree (~10% by 3 steps) -- likely the linear solver /
                 # nonlinear physics amplifying FD truncation/roundoff noise.
eps = 1e-6

diff = autodiff(acc.step, agg_function, var_name)
diff.step_function = jax.jit(diff.step_function)
diff.agg_function = jax.jit(diff.agg_function)
diff.step_function = jax.checkpoint(diff.step_function)  # remat to save memory

# %%
# Autodiff gradient
r_bot0 = acc.state.variables.r_bot
loss0, grad_ad = diff.g(acc.state, r_bot0, iterations=iterations)

# %%
# Finite-difference gradient (central difference)
loss_plus = diff.loss(acc.state, r_bot0 + eps, iterations)
loss_minus = diff.loss(acc.state, r_bot0 - eps, iterations)
grad_fd = (loss_plus - loss_minus) / (2 * eps)

# %%
rel_err = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)

print(f"variable           : {var_name}")
print(f"rollout iterations : {iterations}")
print(f"loss(r_bot0)       = {loss0}")
print(f"autodiff grad      = {grad_ad}")
print(f"finite-diff grad   = {grad_fd}  (eps={eps})")
print(f"relative error     = {rel_err}")
