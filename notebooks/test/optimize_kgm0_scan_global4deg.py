# %%
# Script version of the "Parameter fitting experiment" section of
# notebooks/demonstration/gradient-computation-global4deg.ipynb, for K_gm_0 :
#   - fix a target state by rolling out the global_4deg setup from a known K_gm_0
#   - scan K_gm_0 over a range, compute loss(K_gm_0) and d(loss)/d(K_gm_0)
#   - plot both curves to check the gradient matches the shape of the loss
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from datetime import datetime
from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup

import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

# %%
# Spin-up
warmup_steps = 200
g4d = GlobalFourDegreeSetup()
g4d.setup()
with g4d.state.settings.unlock():
    g4d.state.settings.enable_eke = False

with g4d.state.variables.unlock():
    g4d.state.variables.r_bot += 1e-5
    g4d.state.variables.K_gm_0 += 1000.0

def ps(state):
    n_state = state.copy()
    g4d.step(n_state)
    return n_state

step_jit = jax.jit(ps)

state = g4d.state.copy()
for _ in tqdm(range(warmup_steps), desc="spin-up"):
    state = step_jit(state)

g4d.state = state

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
# Target state: roll out `pred_iter` steps from the spun-up state (true K_gm_0)
var_name = "K_gm_0"
pred_iter = 5
var_true = g4d.state.variables.K_gm_0

target_rollout = autodiff(g4d.step, lambda s: s, var_name)
target_rollout.step_function = jax.jit(target_rollout.step_function)
target_state = target_rollout.rollout(g4d.state, pred_iter)

# %%
# Loss: distance between the rolled-out temperature and the target temperature
def agg_function(state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()

diff = autodiff(g4d.step, agg_function, var_name)
diff.step_function = jax.jit(diff.step_function)
diff.agg_function = jax.jit(diff.agg_function)
diff.step_function = jax.checkpoint(diff.step_function)  # remat to save memory

# %%
# Scan K_gm_0 over the same bounds as in the notebook (800 to 1200, 10 points)
params = jnp.linspace(800, 1200, 10)

losses, grads = [], []
for pr in tqdm(params, desc="scanning K_gm_0"):
    loss, grad = diff.g(g4d.state, pr, iterations=pred_iter)
    losses.append(loss)
    grads.append(grad)

losses = jnp.array(losses)
grads = jnp.array(grads)

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f"{var_name} parameter scan ({pred_iter}-step rollout, global_4deg)")

axs[0].set_title("Loss")
axs[0].plot(params, losses)
axs[0].axvline(var_true, color="r", linestyle="dashed", label=f"true {var_name}")
axs[0].set_xlabel(var_name)
axs[0].set_ylabel("loss")
axs[0].legend()

axs[1].set_title("Gradient")
axs[1].plot(params, grads)
axs[1].axvline(var_true, color="r", linestyle="dashed", label=f"true {var_name}")
axs[1].axhline(0.0, color="k", linewidth=0.8)
axs[1].set_xlabel(var_name)
axs[1].set_ylabel(f"d(loss)/d({var_name})")
axs[1].legend()

fig.tight_layout()

out_path = f'{PRP}notebooks/figures/kgm0_scan_global4deg_{datetime.now().strftime("%d%m%y")}.png'
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
