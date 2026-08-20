# %%
# Report Section 3b (mld variant): joint (c_k, c_eps) tuning on global_4deg_mld_learning.
#   - 3 gradient-descent runs from different random starts -> 2D loss landscape + 3
#     trajectories
#   - mld snapshot: target / initial guess / optimized, run 0
# Same structure as scripts/Reports/report-1/section3b_ck_ceps_scenario.py, loss = mld
# instead of temp (no z-level pick needed -- mld is already a 2D field).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import jax
sys.path.append(PRP)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import spin_up_global4deg_mld, make_diff_step, set_vars, rollout, mld_agg_function, TRUE_PARAMS

g4d, step_jit = spin_up_global4deg_mld(200)
step_diff = make_diff_step(g4d)

pred_iter = 5
c_k_true = TRUE_PARAMS["c_k"]
c_eps_true = TRUE_PARAMS["c_eps"]

target_state = rollout(step_jit, g4d.state, pred_iter)


def loss_fn(params, step_fn, state):
    c_k, c_eps = params
    n_state = set_vars(state, c_k=c_k, c_eps=c_eps)
    n_state = rollout(step_fn, n_state, pred_iter)
    return mld_agg_function(n_state, target_state)


loss_grid_fn = jax.jit(lambda params: loss_fn(params, step_jit, g4d.state))
loss_grad_fn = jax.jit(jax.value_and_grad(lambda params: loss_fn(params, step_diff, g4d.state)))

# %%
# 2D scan
n_grid = 20
c_k_grid = jnp.linspace(0.02, 0.18, n_grid)
c_eps_grid = jnp.linspace(0.3, 1.1, n_grid)

losses_grid = np.zeros((n_grid, n_grid))
for i, ck in enumerate(tqdm(c_k_grid, desc="scanning c_k x c_eps (mld)")):
    for j, ce in enumerate(c_eps_grid):
        losses_grid[i, j] = loss_grid_fn(jnp.array([ck, ce]))

# %%
# 3 gradient-descent runs from different random starts
scale = jnp.array([0.1, 0.7])
lr = 0.15
max_grad = 2.0
n_steps = 150

trajectories = []
finals = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    c_k_start = c_k_true + rng.uniform(-0.03, 0.03)
    c_eps_start = c_eps_true + rng.uniform(-0.15, 0.15)

    u = jnp.array([c_k_start, c_eps_start]) / scale
    traj = [u * scale]
    for _ in tqdm(range(n_steps), desc=f"GD run {seed} (mld)"):
        _, grad = loss_grad_fn(u * scale)
        grad_u = jnp.clip(grad * scale, -max_grad, max_grad)
        u = u - lr * grad_u
        u = jnp.clip(u, 1e-3 / scale, None)
        traj.append(u * scale)

    trajectories.append(jnp.stack(traj))
    finals.append(u * scale)
    print(f"run {seed}: start=({c_k_start:.4f}, {c_eps_start:.4f})  "
          f"final=({float(u[0]*scale[0]):.4f}, {float(u[1]*scale[1]):.4f})")

print(f"true: ({c_k_true:.4f}, {c_eps_true:.4f})")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(c_k_grid, c_eps_grid, losses_grid.T, levels=30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="loss")

colors = ["white", "orange", "cyan"]
for i, traj in enumerate(trajectories):
    ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i], linewidth=1.2, label=f"run {i}")
    ax.plot(traj[0, 0], traj[0, 1], "o", color=colors[i], markersize=8)
    ax.plot(traj[-1, 0], traj[-1, 1], "s", color=colors[i], markersize=8)

ax.plot(c_k_true, c_eps_true, "*", color="red", markersize=20, markeredgecolor="black",
        label="true params", zorder=5)

ax.set_title(f"(c_k, c_eps) loss landscape, 3 GD runs ({pred_iter}-step rollout, global_4deg, mld loss)")
ax.set_xlabel("c_k")
ax.set_ylabel("c_eps")
ax.legend()
fig.tight_layout()

out_path = f"{PRP}Results/Report/figures/mld-1/section3b_ck_ceps_landscape.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")

# %%
# mld snapshot: target (absolute) vs (initial - target) / (optimized - target) bias, run 0
run0_start = trajectories[0][0]
run0_final = finals[0]

initial_rollout = rollout(step_jit, set_vars(g4d.state, c_k=run0_start[0], c_eps=run0_start[1]), pred_iter)
optimized_rollout = rollout(step_jit, set_vars(g4d.state, c_k=run0_final[0], c_eps=run0_final[1]), pred_iter)


def mld_field(state):
    # trim the 2-cell ghost/halo border on each side (cyclic-x wraparound makes xt/yt
    # non-monotonic there), same as report-1/section3b's temp field trim
    return np.asarray(state.variables.mld[2:-2, 2:-2])


xt = np.asarray(g4d.state.variables.xt)[2:-2]
yt = np.asarray(g4d.state.variables.yt)[2:-2]

target_field = mld_field(target_state)
bias_initial = mld_field(initial_rollout) - target_field
bias_optimized = mld_field(optimized_rollout) - target_field

bias_max = np.nanmax(np.abs(np.concatenate([bias_initial[~np.isnan(bias_initial)],
                                             bias_optimized[~np.isnan(bias_optimized)]])))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
im0 = axs[0].pcolormesh(xt, yt, target_field.T, cmap="inferno", shading="auto")
axs[0].set_title("target mld")
fig.colorbar(im0, ax=axs[0], label="mld", shrink=0.8)

im1 = axs[1].pcolormesh(xt, yt, bias_initial.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[1].set_title("initial - target")

im2 = axs[2].pcolormesh(xt, yt, bias_optimized.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[2].set_title("optimized - target")
fig.colorbar(im2, ax=[axs[1], axs[2]], label="bias", shrink=0.8)

for ax in axs:
    ax.set_xlabel("xt")
axs[0].set_ylabel("yt")
fig.suptitle(f"MLD bias after {pred_iter} steps (c_k/c_eps scenario, global_4deg)")

out_path = f"{PRP}Results/Report/figures/mld-1/section3b_ck_ceps_mld_snapshot.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
