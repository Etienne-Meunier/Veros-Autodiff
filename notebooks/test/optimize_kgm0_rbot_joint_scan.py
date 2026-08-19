# %%
# Joint tuning of K_gm_0 and r_bot on the ACC setup :
#   - fix a target state by rolling out ACC from known (K_gm_0, r_bot)
#   - scan a 2D grid of (K_gm_0, r_bot) to build a loss colormap
#   - run plain gradient descent from a random start near the optimum,
#     minimizing distance to the target state
#   - plot the loss colormap with the GD trajectory overlaid and a star
#     at the true (K_gm_0, r_bot)
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from datetime import datetime
from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros
from setups.acc.acc_learning import ACCSetup

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
# Non-jitted step + a checkpointed/jitted variant for use under grad (remat to save memory)
def pure_step(state):
    n_state = state.copy()
    acc.step(n_state)
    return n_state

step_diff = jax.checkpoint(jax.jit(pure_step))

def set_vars(state, K_gm_0, r_bot):
    n_state = state.copy()
    with n_state.variables.unlock():
        n_state.variables.K_gm_0 = K_gm_0
        n_state.variables.r_bot = r_bot
    return n_state

def rollout(step_fn, state, iterations):
    for _ in range(iterations):
        state = step_fn(state)
    return state

# %%
# Target state: roll out `pred_iter` steps from the spun-up state (true K_gm_0, r_bot)
pred_iter = 5
K_gm_0_true = float(acc.state.variables.K_gm_0)
r_bot_true = float(acc.state.variables.r_bot)

target_state = rollout(step_jit, acc.state, pred_iter)

# %%
# Loss: distance between the rolled-out temperature and the target temperature
def agg_function(state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()

def loss_fn(params, step_fn, state):
    K_gm_0, r_bot = params
    n_state = set_vars(state, K_gm_0, r_bot)
    n_state = rollout(step_fn, n_state, pred_iter)
    return agg_function(n_state)

loss_grid_fn = jax.jit(lambda params: loss_fn(params, step_jit, acc.state))
loss_grad_fn = jax.jit(jax.value_and_grad(lambda params: loss_fn(params, step_diff, acc.state)))

# %%
# 2D scan: same bounds as the individual scans (K_gm_0 in 800-1200, r_bot in 0-1.5e-5)
n_grid = 20
K_gm_0_grid = jnp.linspace(800.0, 1200.0, n_grid)
r_bot_grid = jnp.linspace(0.0, 1.5e-5, n_grid)

losses_grid = np.zeros((n_grid, n_grid))
for i, kg in enumerate(tqdm(K_gm_0_grid, desc="scanning K_gm_0 x r_bot")):
    for j, rb in enumerate(r_bot_grid):
        losses_grid[i, j] = loss_grid_fn(jnp.array([kg, rb]))

# %%
# Gradient descent from a random start not too far from the optimum
rng = np.random.default_rng(0)
K_gm_0_start = K_gm_0_true + rng.uniform(-150.0, 150.0)
r_bot_start = r_bot_true + rng.uniform(-0.4e-5, 0.4e-5)

# Optimize in normalized units u = theta / scale (both O(1) at the start), so a single
# learning rate + a single clip threshold apply evenly to K_gm_0 (O(1e3)) and r_bot (O(1e-5))
scale = jnp.array([1000.0, 1e-5])
u = jnp.array([K_gm_0_start, r_bot_start]) / scale
lr = 0.15
max_grad = 2.0  # clip normalized gradient before stepping (physics blows up if r_bot goes negative)
n_steps = 150

trajectory = [u * scale]
for _ in tqdm(range(n_steps), desc="gradient descent"):
    loss, grad = loss_grad_fn(u * scale)
    grad_u = jnp.clip(grad * scale, -max_grad, max_grad)  # chain rule: d(loss)/du = d(loss)/dtheta * scale
    u = u - lr * grad_u
    u = u.at[1].set(jnp.clip(u[1], 1e-7 / scale[1], None))  # r_bot must stay positive
    trajectory.append(u * scale)
    tqdm.write(f"  loss={float(loss):.4e}  K_gm_0={float(u[0] * scale[0]):.2f}  r_bot={float(u[1] * scale[1]):.3e}")

theta = u * scale

trajectory = jnp.stack(trajectory)
print(f"start  (K_gm_0, r_bot) = ({K_gm_0_start:.2f}, {r_bot_start:.3e})")
print(f"final  (K_gm_0, r_bot) = ({float(theta[0]):.2f}, {float(theta[1]):.3e})")
print(f"true   (K_gm_0, r_bot) = ({K_gm_0_true:.2f}, {r_bot_true:.3e})")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(K_gm_0_grid, r_bot_grid, losses_grid.T, levels=30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="loss")

ax.plot(trajectory[:, 0], trajectory[:, 1], "o-", color="white", markersize=3,
        linewidth=1, label="GD trajectory")
ax.plot(trajectory[0, 0], trajectory[0, 1], "o", color="orange", markersize=8, label="start")
ax.plot(trajectory[-1, 0], trajectory[-1, 1], "s", color="red", markersize=8, label="final")
ax.plot(K_gm_0_true, r_bot_true, "*", color="black", markersize=18, markeredgecolor="white",
        label="true params")

ax.set_title(f"Joint (K_gm_0, r_bot) loss landscape ({pred_iter}-step rollout)")
ax.set_xlabel("K_gm_0")
ax.set_ylabel("r_bot")
ax.legend()

fig.tight_layout()

out_path = f'{PRP}notebooks/figures/kgm0_rbot_joint_scan_{datetime.now().strftime("%d%m%y")}.png'
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
