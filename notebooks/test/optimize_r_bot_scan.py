# %%
# Script version of the "Parameter fitting experiment" section of
# notebooks/demonstration/gradient-computation.ipynb :
#   - fix a target state by rolling out the ACC setup from a known r_bot
#   - scan r_bot over a range, compute loss(r_bot) and d(loss)/d(r_bot)
#   - plot both curves to check the gradient matches the shape of the loss
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from datetime import datetime
from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros
from setups.acc.acc_learning import ACCSetup

import jax.numpy as jnp
import matplotlib.pyplot as plt
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

    def g(self, state, var_value, iterations=1):
        def loss_fn(v):
            n_state = autodiff.set_var(self.var_name, state, v)
            n_state = self.rollout(n_state, iterations)
            return self.agg_function(n_state)
        return jax.value_and_grad(loss_fn)(var_value)

# %%
# Target state: roll out `pred_iter` steps from the spun-up state (true r_bot)
var_name = "r_bot"
pred_iter = 5
r_bot_true = acc.state.variables.r_bot

target_rollout = autodiff(acc.step, lambda s: s, var_name)
target_rollout.step_function = jax.jit(target_rollout.step_function)
target_state = target_rollout.rollout(acc.state, pred_iter)

# %%
# Loss: distance between the rolled-out temperature and the target temperature
def agg_function(state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()

diff = autodiff(acc.step, agg_function, var_name)
diff.step_function = jax.jit(diff.step_function)
diff.agg_function = jax.jit(diff.agg_function)
diff.step_function = jax.checkpoint(diff.step_function)  # remat to save memory

# %%
# Scan r_bot over a range around the true value and record loss + gradient
params = jnp.linspace(-0.0, 1.5e-5, 100)

losses, grads = [], []
for pr in tqdm(params, desc="scanning r_bot"):
    loss, grad = diff.g(acc.state, pr, iterations=pred_iter)
    losses.append(loss)
    grads.append(grad)

losses = jnp.array(losses)
grads = jnp.array(grads)

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f"{var_name} parameter scan ({pred_iter}-step rollout)")

axs[0].set_title("Loss")
axs[0].plot(params, losses)
axs[0].axvline(r_bot_true, color="r", linestyle="dashed", label="true r_bot")
axs[0].set_xlabel(var_name)
axs[0].set_ylabel("loss")
axs[0].legend()

axs[1].set_title("Gradient")
axs[1].plot(params, grads)
axs[1].axvline(r_bot_true, color="r", linestyle="dashed", label="true r_bot")
axs[1].axhline(0.0, color="k", linewidth=0.8)
axs[1].set_xlabel(var_name)
axs[1].set_ylabel("d(loss)/d(r_bot)")
axs[1].legend()

fig.tight_layout()

out_path = f'{PRP}notebooks/figures/r_bot_scan_{datetime.now().strftime("%d%m%y")}.png'
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
