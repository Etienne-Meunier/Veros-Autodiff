# %%
# report-longrollouts-4 phase 2: Adam GD joint tuning of c_k/c_eps against the
# 1yr mld_ma loss, nz=64 (built-in mld_ma averaging disabled, see common.py), at
# whatever n calibrate_n.py found feasible. Mirrors report-longrollouts-3's
# run_gd.py exactly (single process, compile once, loop the compiled
# value_and_grad) -- just swapped setup/loss. --n / --lead_chunk_size are
# required args (calibrate_n.py's result, not hardcoded here).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.append(PRP)

from common import spin_up_mld, make_diff_step, set_vars, mld_ma_agg_function, plain_forward_rollout_mld_ma, rollout_mld_ma, peak_gpu_memory_bytes, STORE_DIR, write_csv_incremental, MLD_MA_WINDOW

parser = argparse.ArgumentParser()
parser.add_argument("--nz", type=int, default=64)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--lead_chunk_size", type=int, required=True)
parser.add_argument("--tail_chunk_size", type=int, default=32)
parser.add_argument("--window", type=int, default=MLD_MA_WINDOW)
parser.add_argument("--num_steps", type=int, default=40)
parser.add_argument("--adam_lr", type=float, default=0.002)
args = parser.parse_args()

TRUE_C_K = 0.1
TRUE_C_EPS = 0.7
INIT_C_K = 0.08
INIT_C_EPS = 0.6
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
PARAM_MIN, PARAM_MAX = 0.01, 1.0

csv_path = f"{STORE_DIR}/gd_history.csv"
fig_path = f"{STORE_DIR}/gd_trajectory.png"

g4d, step_jit = spin_up_mld(args.nz, warmup_steps=20)
_, target_mld_ma = plain_forward_rollout_mld_ma(step_jit, g4d.state, args.n, args.window)


def loss_fn(c_k, c_eps):
    n_state = set_vars(g4d.state, c_k=c_k, c_eps=c_eps)
    n_state, mld_ma = rollout_mld_ma(make_diff_step(g4d), n_state, args.n, args.lead_chunk_size, args.tail_chunk_size, args.window)
    return mld_ma_agg_function(mld_ma, target_mld_ma)


value_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1))

c_k = jnp.array(INIT_C_K)
c_eps = jnp.array(INIT_C_EPS)

print(f"Compiling (nz={args.nz}, n={args.n}, lead_chunk_size={args.lead_chunk_size}, tail_chunk_size={args.tail_chunk_size})...", flush=True)
t0 = time.time()
compiled = jax.jit(value_and_grad).lower(c_k, c_eps).compile()
compile_time_s = time.time() - t0
print(f"Compiled in {compile_time_s:.1f}s", flush=True)

m_k = m_eps = v_k = v_eps = 0.0
rows = []

for step in range(args.num_steps):
    t0 = time.time()
    loss_val, (g_k, g_eps) = compiled(c_k, c_eps)
    jax.block_until_ready(loss_val)
    dt = time.time() - t0

    loss_val = float(loss_val)
    g_k = float(g_k)
    g_eps = float(g_eps)

    t = step + 1
    m_k = ADAM_B1 * m_k + (1 - ADAM_B1) * g_k
    v_k = ADAM_B2 * v_k + (1 - ADAM_B2) * g_k ** 2
    m_k_hat = m_k / (1 - ADAM_B1 ** t)
    v_k_hat = v_k / (1 - ADAM_B2 ** t)
    c_k_val = float(c_k) - args.adam_lr * m_k_hat / (v_k_hat ** 0.5 + ADAM_EPS)
    c_k_val = min(max(c_k_val, PARAM_MIN), PARAM_MAX)

    m_eps = ADAM_B1 * m_eps + (1 - ADAM_B1) * g_eps
    v_eps = ADAM_B2 * v_eps + (1 - ADAM_B2) * g_eps ** 2
    m_eps_hat = m_eps / (1 - ADAM_B1 ** t)
    v_eps_hat = v_eps / (1 - ADAM_B2 ** t)
    c_eps_val = float(c_eps) - args.adam_lr * m_eps_hat / (v_eps_hat ** 0.5 + ADAM_EPS)
    c_eps_val = min(max(c_eps_val, PARAM_MIN), PARAM_MAX)

    row = dict(step=step, loss=loss_val, c_k=float(c_k), c_eps=float(c_eps),
               grad_c_k=g_k, grad_c_eps=g_eps, step_time_s=dt)
    rows.append(row)
    write_csv_incremental(rows, csv_path)
    print(f"[step {step}] loss={loss_val:.6e} c_k={float(c_k):.6f} c_eps={float(c_eps):.6f} "
          f"grad_c_k={g_k:.4e} grad_c_eps={g_eps:.4e} ({dt:.1f}s)", flush=True)

    c_k = jnp.array(c_k_val)
    c_eps = jnp.array(c_eps_val)

peak_mem = peak_gpu_memory_bytes()
print(f"DONE final c_k={float(c_k):.6f} final c_eps={float(c_eps):.6f} peak_mem_bytes={peak_mem}", flush=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
steps = [r["step"] for r in rows]
axs[0].plot(steps, [r["loss"] for r in rows], "o-")
axs[0].set_yscale("log")
axs[0].set_title("loss vs GD step")

axs[1].plot(steps, [r["c_k"] for r in rows], "o-", label="c_k")
axs[1].axhline(TRUE_C_K, color="k", ls="--", label="true c_k")
axs[1].set_title("c_k vs GD step")
axs[1].legend()

axs[2].plot(steps, [r["c_eps"] for r in rows], "o-", label="c_eps")
axs[2].axhline(TRUE_C_EPS, color="k", ls="--", label="true c_eps")
axs[2].set_title("c_eps vs GD step")
axs[2].legend()

for ax in axs:
    ax.set_xlabel("GD step")
    ax.grid(True, alpha=0.3)

fig.suptitle(f"report-longrollouts-4: Adam GD tuning of c_k/c_eps, mld_ma (1yr), nz={args.nz}, n={args.n}, GPU")
fig.tight_layout()
fig.savefig(fig_path, dpi=150)

print(f"Saved {csv_path}")
print(f"Saved {fig_path}")
