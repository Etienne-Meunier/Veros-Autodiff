# %%
# Report mld-mini, Scenario B : joint (c_k, c_eps) tuning on global_4deg_learning_mini,
# loss = squared MLD error instead of temp (see scripts/Reports/report-1/
# section3b_ck_ceps_scenario.py for the temp-loss version this mirrors).
#   - 3 gradient-descent runs from different random starts -> 2D loss landscape + 3
#     trajectories, converging (or not -- MLD may be a noisier/less identifiable
#     target than temp) towards the true (c_k, c_eps)
#   - MLD snapshot: target / initial guess / optimized, for run 0
#   Budget: pred_iter kept small (5) and n_steps capped at 200 -- this is meant to be a
#   fast/cheap regression-style check, not a long sweep like report-2.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

import jax
sys.path.append(PRP)

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from common import spin_up_global4deg_mld_mini, make_diff_step, set_vars, rollout, mld_agg_function, TRUE_PARAMS

g4d, step_jit = spin_up_global4deg_mld_mini(200)
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
# 2D scan (kept modest -- 15x15, not report-1's 20x20 -- see budget note above)
n_grid = 15
c_k_grid = jnp.linspace(0.02, 0.18, n_grid)
c_eps_grid = jnp.linspace(0.3, 1.1, n_grid)

losses_grid = np.zeros((n_grid, n_grid))
for i, ck in enumerate(tqdm(c_k_grid, desc="scanning c_k x c_eps (mld)")):
    for j, ce in enumerate(c_eps_grid):
        losses_grid[i, j] = loss_grid_fn(jnp.array([ck, ce]))

# %%
# 3 gradient-descent runs from different random starts. Uses optax.adam, not report-1's
# fixed-step clipped SGD: MLD-loss gradients here are ~1e4-1e5 even at pred_iter=5 (much
# larger than report-1's temp-loss scale), which saturates a fixed clip every step and
# lands in an exact period-2 limit cycle (verified: params ping-ponged between two points
# every step, never converging) -- the same failure mode report-2's docstring documents
# for temp loss at longer rollouts. adam adapts its step size per-parameter from each
# parameter's own gradient-magnitude history, so it isn't sensitive to this raw-gradient
# scale the way a fixed clipped step is.
adam_lr = 0.01
n_steps = 200  # budget cap

trajectories = []
finals = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    c_k_start = c_k_true + rng.uniform(-0.03, 0.03)
    c_eps_start = c_eps_true + rng.uniform(-0.15, 0.15)

    params = jnp.array([c_k_start, c_eps_start])
    optimizer = optax.adam(adam_lr)
    opt_state = optimizer.init(params)
    traj = [params]
    for _ in tqdm(range(n_steps), desc=f"GD run {seed} (mld)"):
        _, grad = loss_grad_fn(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-3, None)
        traj.append(params)

    trajectories.append(jnp.stack(traj))
    finals.append(params)
    print(f"run {seed}: start=({c_k_start:.4f}, {c_eps_start:.4f})  "
          f"final=({float(params[0]):.4f}, {float(params[1]):.4f})")

print(f"true: ({c_k_true:.4f}, {c_eps_true:.4f})")

# %%
fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(c_k_grid, c_eps_grid, losses_grid.T, levels=30, cmap="viridis")
fig.colorbar(cf, ax=ax, label="loss (MLD)")

colors = ["white", "orange", "cyan"]
for i, traj in enumerate(trajectories):
    ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i], linewidth=1.2, label=f"run {i}")
    ax.plot(traj[0, 0], traj[0, 1], "o", color=colors[i], markersize=8)
    ax.plot(traj[-1, 0], traj[-1, 1], "s", color=colors[i], markersize=8)

ax.plot(c_k_true, c_eps_true, "*", color="red", markersize=20, markeredgecolor="black",
        label="true params", zorder=5)

ax.set_title(f"(c_k, c_eps) MLD-loss landscape, 3 GD runs ({pred_iter}-step rollout, global_4deg_mld_mini)")
ax.set_xlabel("c_k")
ax.set_ylabel("c_eps")
ax.legend()
fig.tight_layout()

out_path = f"{PRP}Results/Report/figures/report-mld-mini-1/section3b_ck_ceps_mld_landscape.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")

# %%
# MLD snapshot: target (absolute) vs (initial - target) / (optimized - target) bias, run 0
run0_start = trajectories[0][0]
run0_final = finals[0]

initial_rollout = rollout(step_jit, set_vars(g4d.state, c_k=run0_start[0], c_eps=run0_start[1]), pred_iter)
optimized_rollout = rollout(step_jit, set_vars(g4d.state, c_k=run0_final[0], c_eps=run0_final[1]), pred_iter)


def mld_field(state):
    # trim the 2-cell ghost/halo border on each side (cyclic-x wraparound makes xt/yt
    # non-monotonic there -- plotting them un-trimmed distorts the map, same as report-1)
    return np.asarray(state.variables.mld[2:-2, 2:-2])  # already NaN at land/degenerate columns


xt = np.asarray(g4d.state.variables.xt)[2:-2]
yt = np.asarray(g4d.state.variables.yt)[2:-2]

target_field = mld_field(target_state)
bias_initial = mld_field(initial_rollout) - target_field
bias_optimized = mld_field(optimized_rollout) - target_field

finite_bias = np.concatenate([
    bias_initial[~np.isnan(bias_initial)],
    bias_optimized[~np.isnan(bias_optimized)],
])
bias_max = np.nanmax(np.abs(finite_bias)) if finite_bias.size else 1.0

fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
im0 = axs[0].pcolormesh(xt, yt, target_field.T, cmap="viridis", shading="auto")
axs[0].set_title("target mld")
fig.colorbar(im0, ax=axs[0], label="mld (m)", shrink=0.8)

im1 = axs[1].pcolormesh(xt, yt, bias_initial.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[1].set_title("initial - target")

im2 = axs[2].pcolormesh(xt, yt, bias_optimized.T, cmap="RdBu_r", vmin=-bias_max, vmax=bias_max, shading="auto")
axs[2].set_title("optimized - target")
fig.colorbar(im2, ax=[axs[1], axs[2]], label="bias (m)", shrink=0.8)

for ax in axs:
    ax.set_xlabel("xt")
axs[0].set_ylabel("yt")
fig.suptitle(f"MLD bias after {pred_iter} steps (c_k/c_eps scenario, global_4deg_mld_mini)")

out_path = f"{PRP}Results/Report/figures/report-mld-mini-1/section3b_ck_ceps_mld_snapshot.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
