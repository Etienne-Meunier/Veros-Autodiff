# %%
# Joint tuning of c_k and c_eps on the global_4deg setup (see
# notebooks/test/optimize_ck_ceps_joint_scan.py for the ACC version) :
#   - c_k / c_eps were TKE-closure settings (fixed constants); promoted to variables the
#     same way K_gm_0 / r_bot were, so they can be differentiated w.r.t. (see
#     veros/veros/variables.py, veros/veros/settings.py, veros/veros/core/tke.py)
#   - fix a target state by rolling out global_4deg from known (c_k, c_eps)
#   - scan a 2D grid of (c_k, c_eps) to build a loss colormap
#   - run plain gradient descent from a random start near the optimum,
#     minimizing distance to the target state
#   - plot the loss colormap with the GD trajectory overlaid and a star
#     at the true (c_k, c_eps)
#
# c_eps's gradient here originally disagreed with finite differences in SIGN at several
# rollout lengths (root cause: the same safe_sqrt clip as ACC's — fixed; see
# scripts/debugging_ceps/). Matches FD to ~0.2-2.4% now, same quality as c_k, under
# global_4deg's default tke_mxl_choice (2) — no override needed.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from datetime import datetime
from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
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
    g4d.state.variables.c_k += 0.0
    g4d.state.variables.c_eps += 0.0

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
# Non-jitted step + a checkpointed/jitted variant for use under grad (remat to save memory)
def pure_step(state):
    n_state = state.copy()
    g4d.step(n_state)
    return n_state

step_diff = jax.checkpoint(jax.jit(pure_step))

def set_vars(state, c_k, c_eps):
    n_state = state.copy()
    with n_state.variables.unlock():
        n_state.variables.c_k = c_k
        n_state.variables.c_eps = c_eps
    return n_state

def rollout(step_fn, state, iterations):
    for _ in range(iterations):
        state = step_fn(state)
    return state

# %%
# Target state: roll out `pred_iter` steps from the spun-up state (true c_k, c_eps)
pred_iter = 5
c_k_true = float(g4d.state.variables.c_k)
c_eps_true = float(g4d.state.variables.c_eps)

target_state = rollout(step_jit, g4d.state, pred_iter)

# %%
# Loss: distance between the rolled-out temperature and the target temperature
def agg_function(state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()

def loss_fn(params, step_fn, state):
    c_k, c_eps = params
    n_state = set_vars(state, c_k, c_eps)
    n_state = rollout(step_fn, n_state, pred_iter)
    return agg_function(n_state)

loss_grid_fn = jax.jit(lambda params: loss_fn(params, step_jit, g4d.state))
loss_grad_fn = jax.jit(jax.value_and_grad(lambda params: loss_fn(params, step_diff, g4d.state)))

# %%
# 2D scan around the true values
n_grid = 20
c_k_grid = jnp.linspace(0.02, 0.18, n_grid)
c_eps_grid = jnp.linspace(0.3, 1.1, n_grid)

losses_grid = np.zeros((n_grid, n_grid))
for i, ck in enumerate(tqdm(c_k_grid, desc="scanning c_k x c_eps")):
    for j, ce in enumerate(c_eps_grid):
        losses_grid[i, j] = loss_grid_fn(jnp.array([ck, ce]))

# %%
# Gradient descent from a random start not too far from the optimum
rng = np.random.default_rng(0)
c_k_start = c_k_true + rng.uniform(-0.03, 0.03)
c_eps_start = c_eps_true + rng.uniform(-0.15, 0.15)

# Optimize in normalized units u = theta / scale (both O(1) at the start), so a single
# learning rate + a single clip threshold apply evenly to c_k (O(1e-1)) and c_eps (O(1))
scale = jnp.array([0.1, 0.7])
u = jnp.array([c_k_start, c_eps_start]) / scale
lr = 0.15
max_grad = 2.0  # clip normalized gradient before stepping (physics blows up if c_k/c_eps go negative)
n_steps = 150

trajectory = [u * scale]
for _ in tqdm(range(n_steps), desc="gradient descent"):
    loss, grad = loss_grad_fn(u * scale)
    grad_u = jnp.clip(grad * scale, -max_grad, max_grad)  # chain rule: d(loss)/du = d(loss)/dtheta * scale
    u = u - lr * grad_u
    u = jnp.clip(u, 1e-3 / scale, None)  # c_k, c_eps must stay positive
    trajectory.append(u * scale)
    tqdm.write(f"  loss={float(loss):.4e}  c_k={float(u[0] * scale[0]):.4f}  c_eps={float(u[1] * scale[1]):.4f}")

theta = u * scale

trajectory = jnp.stack(trajectory)
print(f"start  (c_k, c_eps) = ({c_k_start:.4f}, {c_eps_start:.4f})")
print(f"final  (c_k, c_eps) = ({float(theta[0]):.4f}, {float(theta[1]):.4f})")
print(f"true   (c_k, c_eps) = ({c_k_true:.4f}, {c_eps_true:.4f})")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(c_k_grid, c_eps_grid, losses_grid.T, levels=30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="loss")

ax.plot(trajectory[:, 0], trajectory[:, 1], "o-", color="white", markersize=3,
        linewidth=1, label="GD trajectory")
ax.plot(trajectory[0, 0], trajectory[0, 1], "o", color="orange", markersize=8, label="start")
ax.plot(trajectory[-1, 0], trajectory[-1, 1], "s", color="red", markersize=8, label="final")
ax.plot(c_k_true, c_eps_true, "*", color="black", markersize=18, markeredgecolor="white",
        label="true params")

ax.set_title(f"Joint (c_k, c_eps) loss landscape ({pred_iter}-step rollout, global_4deg)")
ax.set_xlabel("c_k")
ax.set_ylabel("c_eps")
ax.legend()

fig.tight_layout()

out_path = f'{PRP}notebooks/figures/ck_ceps_joint_scan_global4deg_{datetime.now().strftime("%d%m%y")}.png'
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
