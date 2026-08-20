# %%
# Report Section 3, Scenario A : joint (K_gm_0, r_bot) tuning on global_4deg.
#   - 3 gradient-descent runs from different random starts -> 2D loss landscape + 3
#     trajectories, all converging to the true (K_gm_0, r_bot)
#   - surface temperature snapshot : target / initial guess / optimized, for run 0
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from datetime import datetime
import jax
sys.path.append(PRP)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import spin_up_global4deg, make_diff_step, set_vars, rollout, TRUE_PARAMS

g4d, step_jit = spin_up_global4deg(200)
step_diff = make_diff_step(g4d)

pred_iter = 5
K_gm_0_true = TRUE_PARAMS["K_gm_0"]
r_bot_true = TRUE_PARAMS["r_bot"]

target_state = rollout(step_jit, g4d.state, pred_iter)


def agg_function(state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()


def loss_fn(params, step_fn, state):
    K_gm_0, r_bot = params
    n_state = set_vars(state, K_gm_0=K_gm_0, r_bot=r_bot)
    n_state = rollout(step_fn, n_state, pred_iter)
    return agg_function(n_state)


loss_grid_fn = jax.jit(lambda params: loss_fn(params, step_jit, g4d.state))
loss_grad_fn = jax.jit(jax.value_and_grad(lambda params: loss_fn(params, step_diff, g4d.state)))

# %%
# 2D scan
n_grid = 20
K_gm_0_grid = jnp.linspace(800.0, 1200.0, n_grid)
r_bot_grid = jnp.linspace(0.0, 1.5e-5, n_grid)

losses_grid = np.zeros((n_grid, n_grid))
for i, kg in enumerate(tqdm(K_gm_0_grid, desc="scanning K_gm_0 x r_bot")):
    for j, rb in enumerate(r_bot_grid):
        losses_grid[i, j] = loss_grid_fn(jnp.array([kg, rb]))

# %%
# 3 gradient-descent runs from different random starts
scale = jnp.array([1000.0, 1e-5])
lr = 0.15
max_grad = 2.0
n_steps = 150

trajectories = []
finals = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    K_gm_0_start = K_gm_0_true + rng.uniform(-150.0, 150.0)
    r_bot_start = r_bot_true + rng.uniform(-0.4e-5, 0.4e-5)

    u = jnp.array([K_gm_0_start, r_bot_start]) / scale
    traj = [u * scale]
    for _ in tqdm(range(n_steps), desc=f"GD run {seed}"):
        _, grad = loss_grad_fn(u * scale)
        grad_u = jnp.clip(grad * scale, -max_grad, max_grad)
        u = u - lr * grad_u
        u = u.at[1].set(jnp.clip(u[1], 1e-7 / scale[1], None))
        traj.append(u * scale)

    trajectories.append(jnp.stack(traj))
    finals.append(u * scale)
    print(f"run {seed}: start=({K_gm_0_start:.2f}, {r_bot_start:.3e})  "
          f"final=({float(u[0]*scale[0]):.2f}, {float(u[1]*scale[1]):.3e})")

print(f"true: ({K_gm_0_true:.2f}, {r_bot_true:.3e})")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(K_gm_0_grid, r_bot_grid, losses_grid.T, levels=30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="loss")

colors = ["white", "orange", "cyan"]
for i, traj in enumerate(trajectories):
    ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i], linewidth=1.2, label=f"run {i}")
    ax.plot(traj[0, 0], traj[0, 1], "o", color=colors[i], markersize=8)
    ax.plot(traj[-1, 0], traj[-1, 1], "s", color=colors[i], markersize=8)

ax.plot(K_gm_0_true, r_bot_true, "*", color="red", markersize=20, markeredgecolor="black",
        label="true params", zorder=5)

ax.set_title(f"(K_gm_0, r_bot) loss landscape, 3 GD runs ({pred_iter}-step rollout, global_4deg)")
ax.set_xlabel("K_gm_0")
ax.set_ylabel("r_bot")
ax.legend()
fig.tight_layout()

out_path = f"{PRP}Results/Report/figures/section3a_kgm0_rbot_landscape.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")

# %%
# Temperature snapshot : target (absolute) vs (initial - target) / (optimized - target)
# bias, run 0. Surface (z=-1) differences are tiny (SST restoring forcing damps them,
# ~1e-4 deg C) -- z=13 (one level below surface) carries the largest parameter-induced
# signal (see scripts/report/ level-variability check), so used here instead.
run0_start = trajectories[0][0]
run0_final = finals[0]

initial_rollout = rollout(step_jit, set_vars(g4d.state, K_gm_0=run0_start[0], r_bot=run0_start[1]), pred_iter)
optimized_rollout = rollout(step_jit, set_vars(g4d.state, K_gm_0=run0_final[0], r_bot=run0_final[1]), pred_iter)

Z_LEVEL = 13


def field_at_level(state, z):
    tau = state.variables.tau
    # trim the 2-cell ghost/halo border on each side (cyclic-x wraparound makes xt/yt
    # non-monotonic there -- plotting them un-trimmed distorts the map, see debug check)
    temp = np.asarray(state.variables.temp[2:-2, 2:-2, z, tau])
    mask = np.asarray(state.variables.maskT[2:-2, 2:-2, z]).astype(bool)
    return np.where(mask, temp, np.nan)


xt = np.asarray(g4d.state.variables.xt)[2:-2]
yt = np.asarray(g4d.state.variables.yt)[2:-2]

target_field = field_at_level(target_state, Z_LEVEL)
bias_initial = field_at_level(initial_rollout, Z_LEVEL) - target_field
bias_optimized = field_at_level(optimized_rollout, Z_LEVEL) - target_field

bias_max = np.nanmax(np.abs(np.concatenate([bias_initial[~np.isnan(bias_initial)],
                                             bias_optimized[~np.isnan(bias_optimized)]])))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
im0 = axs[0].pcolormesh(xt, yt, target_field.T, cmap="inferno", shading="auto")
axs[0].set_title(f"target (z={Z_LEVEL})")
fig.colorbar(im0, ax=axs[0], label="temp", shrink=0.8)

im1 = axs[1].pcolormesh(xt, yt, bias_initial.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[1].set_title("initial - target")

im2 = axs[2].pcolormesh(xt, yt, bias_optimized.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[2].set_title("optimized - target")
fig.colorbar(im2, ax=[axs[1], axs[2]], label="bias", shrink=0.8)

for ax in axs:
    ax.set_xlabel("xt")
axs[0].set_ylabel("yt")
fig.suptitle(f"Temperature bias after {pred_iter} steps, z={Z_LEVEL} (K_gm_0/r_bot scenario, global_4deg)")

out_path = f"{PRP}Results/Report/figures/section3a_kgm0_rbot_temp_snapshot.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
