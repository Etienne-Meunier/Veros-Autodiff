# %%
# Gradient-descent (Adam) joint tuning of c_k, c_eps against temp-squared-error
# at n=2000, chunk_size=32 (validated in report-longrollouts-1: 4.82GB, well
# under the 16GB wall). Target = trajectory from veros' native defaults
# (c_k=0.1, c_eps=0.7, see veros/variables.py). Init = (0.08, 0.6) -- the same
# test values used as arbitrary probe points throughout report-longrollouts-1/2.
#
# Single process, compile once (jax.value_and_grad, argnums=(0,1)), then loop the
# compiled function -- unlike the sweep scripts' one-jit-per-process pattern
# (that was for compile-time/memory isolation across *different* configs; here
# we WANT to reuse one compiled graph across many GD steps).
#
# Manual Adam (no optax dependency): per-parameter adaptive step size handles
# c_k's and c_eps's differing gradient scales (report-1 found c_k's gradient
# consistently ~5-10x larger magnitude than c_eps's at matching n) without
# hand-tuned per-param learning rates. Params clipped to [0.01, 1.0] as a safety
# rail (physical mixing constants, must stay positive; also guards against a
# runaway step given how steep report-1 found this loss to be).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.append(PRP)

from common import spin_up, make_diff_step, set_vars, temp_agg_function, plain_forward_rollout, rollout, peak_gpu_memory_bytes, STORE_DIR, write_csv_incremental

N = 2000
CHUNK_SIZE = 32
TRUE_C_K = 0.1
TRUE_C_EPS = 0.7
INIT_C_K = 0.08
INIT_C_EPS = 0.6
NUM_STEPS = 40
ADAM_LR = 0.002
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
PARAM_MIN, PARAM_MAX = 0.01, 1.0

csv_path = f"{STORE_DIR}/gd_history.csv"
fig_path = f"{STORE_DIR}/gd_trajectory.png"

g4d, step_jit = spin_up(warmup_steps=20)
target_state = plain_forward_rollout(step_jit, g4d.state, N)  # native defaults: c_k=0.1, c_eps=0.7


def loss_fn(c_k, c_eps):
    n_state = set_vars(g4d.state, c_k=c_k, c_eps=c_eps)
    n_state = rollout(make_diff_step(g4d), n_state, N, CHUNK_SIZE)
    return temp_agg_function(n_state, target_state)


value_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1))

c_k = jnp.array(INIT_C_K)
c_eps = jnp.array(INIT_C_EPS)

print(f"Compiling (n={N}, chunk_size={CHUNK_SIZE})...", flush=True)
t0 = time.time()
compiled = jax.jit(value_and_grad).lower(c_k, c_eps).compile()
compile_time_s = time.time() - t0
print(f"Compiled in {compile_time_s:.1f}s", flush=True)

m_k = m_eps = v_k = v_eps = 0.0
rows = []

for step in range(NUM_STEPS):
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
    c_k_val = float(c_k) - ADAM_LR * m_k_hat / (v_k_hat ** 0.5 + ADAM_EPS)
    c_k_val = min(max(c_k_val, PARAM_MIN), PARAM_MAX)

    m_eps = ADAM_B1 * m_eps + (1 - ADAM_B1) * g_eps
    v_eps = ADAM_B2 * v_eps + (1 - ADAM_B2) * g_eps ** 2
    m_eps_hat = m_eps / (1 - ADAM_B1 ** t)
    v_eps_hat = v_eps / (1 - ADAM_B2 ** t)
    c_eps_val = float(c_eps) - ADAM_LR * m_eps_hat / (v_eps_hat ** 0.5 + ADAM_EPS)
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

fig.suptitle(f"report-longrollouts-3: Adam GD tuning of c_k/c_eps, n={N}, chunk_size={CHUNK_SIZE}, GPU")
fig.tight_layout()
fig.savefig(fig_path, dpi=150)

print(f"Saved {csv_path}")
print(f"Saved {fig_path}")
