# %%
# Report 2 : extends report-1 Section 3B, (c_k, c_eps) joint recovery.
#   Same setup (global_4deg, same warmup, same true params, same 2D grid, same GD
#   hyperparameters) as report-1/section3b_ck_ceps_scenario.py -- the only change is
#   the target rollout length, swept over [5 .. 1000] (5 lengths), to test how
#   parameter recoverability degrades/improves with longer rollouts.
#   For each rollout length: ONE GD run (not 3) -> loss landscape + trajectory, plus
#   the target/initial/optimized temperature-bias snapshot.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

import jax
sys.path.append(PRP)

import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from common import spin_up_global4deg, make_diff_step, set_vars, rollout, TRUE_PARAMS

g4d, step_jit = spin_up_global4deg(200)
step_diff = make_diff_step(g4d)

c_k_true = TRUE_PARAMS["c_k"]
c_eps_true = TRUE_PARAMS["c_eps"]

# rollout lengths to sweep: 5 -> 1000, 5 values, log-spaced
rollout_lengths = [5, 20, 75, 250, 1000]

# same 2D scan grid as report-1/section3b
n_grid = 20
c_k_grid = jnp.linspace(0.02, 0.18, n_grid)
c_eps_grid = jnp.linspace(0.3, 1.1, n_grid)

# report-1's fixed-step-size clipped SGD (lr=0.15, max_grad=2.0, in a param/scale
# reparametrized space) was tuned for the n=5 gradient scale. Raw d(loss)/d(c_k,c_eps)
# grows ~35x from n=5 to n=75, so that clip saturates every step once n gtrsim 20 and
# the fixed step it produces (~0.21 in c_eps, ~26% of the whole c_eps search range)
# overshoots the minimum every time -- lands in an exact period-2 limit cycle, back on
# the start after 150 (even) steps. optax.adam adapts its step size per-parameter from
# each parameter's own gradient-magnitude history, so it needs neither the max_grad
# clip nor the c_k/c_eps reparametrization trick (`scale` in the old version) -- one
# hyperparameter set (adam_lr) works directly on raw (c_k, c_eps) across the whole
# rollout-length sweep.
adam_lr = 0.01
n_steps = 150
seed = 0
optimizer = optax.adam(adam_lr)

Z_LEVEL = 13

out_dir = f"{PRP}Results/Report/figures/report-2"
os.makedirs(out_dir, exist_ok=True)


def field_at_level(state, z):
    tau = state.variables.tau
    # trim the 2-cell ghost/halo border on each side (cyclic-x wraparound makes xt/yt
    # non-monotonic there), same as report-1/section3b
    temp = np.asarray(state.variables.temp[2:-2, 2:-2, z, tau])
    mask = np.asarray(state.variables.maskT[2:-2, 2:-2, z]).astype(bool)
    return np.where(mask, temp, np.nan)


xt = np.asarray(g4d.state.variables.xt)[2:-2]
yt = np.asarray(g4d.state.variables.yt)[2:-2]

# %%
# Sweep over rollout lengths
for pred_iter in rollout_lengths:
    print(f"\n=== rollout length = {pred_iter} ===")

    target_state = rollout(step_jit, g4d.state, pred_iter)

    def agg_function(state, target_state=target_state):
        return ((state.variables.temp - target_state.variables.temp) ** 2).sum()

    def loss_fn(params, step_fn, state, target_state=target_state, pred_iter=pred_iter):
        c_k, c_eps = params
        n_state = set_vars(state, c_k=c_k, c_eps=c_eps)
        n_state = rollout(step_fn, n_state, pred_iter)
        return agg_function(n_state, target_state)

    loss_grid_fn = jax.jit(lambda params: loss_fn(params, step_jit, g4d.state))
    loss_grad_fn = jax.jit(jax.value_and_grad(lambda params: loss_fn(params, step_diff, g4d.state)))

    # 2D scan
    losses_grid = np.zeros((n_grid, n_grid))
    for i, ck in enumerate(tqdm(c_k_grid, desc=f"[n={pred_iter}] scanning c_k x c_eps")):
        for j, ce in enumerate(c_eps_grid):
            losses_grid[i, j] = loss_grid_fn(jnp.array([ck, ce]))

    # single GD run
    rng = np.random.default_rng(seed)
    c_k_start = c_k_true + rng.uniform(-0.03, 0.03)
    c_eps_start = c_eps_true + rng.uniform(-0.15, 0.15)

    params = jnp.array([c_k_start, c_eps_start])
    opt_state = optimizer.init(params)
    traj = [params]
    for _ in tqdm(range(n_steps), desc=f"[n={pred_iter}] GD run"):
        _, grad = loss_grad_fn(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, 1e-3, None)
        traj.append(params)

    traj = jnp.stack(traj)
    final = params
    print(f"[n={pred_iter}] start=({c_k_start:.4f}, {c_eps_start:.4f})  "
          f"final=({float(final[0]):.4f}, {float(final[1]):.4f})  "
          f"true=({c_k_true:.4f}, {c_eps_true:.4f})")

    # --- loss landscape + trajectory ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(c_k_grid, c_eps_grid, losses_grid.T, levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="loss")

    ax.plot(traj[:, 0], traj[:, 1], "-", color="white", linewidth=1.2, label="GD run")
    ax.plot(traj[0, 0], traj[0, 1], "o", color="white", markersize=8)
    ax.plot(traj[-1, 0], traj[-1, 1], "s", color="white", markersize=8)

    ax.plot(c_k_true, c_eps_true, "*", color="red", markersize=20, markeredgecolor="black",
            label="true params", zorder=5)

    ax.set_title(f"(c_k, c_eps) loss landscape, 1 GD run ({pred_iter}-step rollout, global_4deg)")
    ax.set_xlabel("c_k")
    ax.set_ylabel("c_eps")
    ax.legend()
    fig.tight_layout()

    out_path = f"{out_dir}/section3b_ck_ceps_landscape_n{pred_iter:04d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")

    # --- temperature snapshot: target / initial-target / optimized-target ---
    initial_rollout = rollout(step_jit, set_vars(g4d.state, c_k=traj[0, 0], c_eps=traj[0, 1]), pred_iter)
    optimized_rollout = rollout(step_jit, set_vars(g4d.state, c_k=final[0], c_eps=final[1]), pred_iter)

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
    fig.suptitle(f"Temperature bias after {pred_iter} steps, z={Z_LEVEL} (c_k/c_eps scenario, global_4deg)")

    out_path = f"{out_dir}/section3b_ck_ceps_temp_snapshot_n{pred_iter:04d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
