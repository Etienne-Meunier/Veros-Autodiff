# %%
# Report Section 2 : wall-clock cost of forward rollout vs forward+gradient, and the
# one-time JIT compile cost of each, as a function of the number of unrolled steps
# (global_4deg). Gradient cost is representative across parameters (shown here for
# K_gm_0) — it's dominated by the rollout length and the checkpointed backward pass,
# not by which scalar parameter is being differentiated.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')
sys.path.append(PRP)

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import spin_up_global4deg, set_vars, rollout

g4d, step_jit = spin_up_global4deg(200)


def pure_step(state):
    n_state = state.copy()
    g4d.step(n_state)
    return n_state


checkpointed_step = jax.checkpoint(pure_step)
test_val = jnp.array(900.0)  # K_gm_0, off the true value (1000)
n_repeats = 5
n_values = [2, 3, 5, 7, 10, 15, 20]


def time_call(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_time = time.perf_counter() - t0

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return compile_time, min(times)


# %%
compile_fwd, run_fwd = [], []
compile_grad, run_grad = [], []

for n in tqdm(n_values, desc="timing"):
    forward_fn = jax.jit(lambda state, n=n: rollout(pure_step, state, n))
    c, r = time_call(forward_fn, g4d.state)
    compile_fwd.append(c)
    run_fwd.append(r)

    def loss(v, n=n):
        n_state = set_vars(g4d.state, K_gm_0=v)
        n_state = rollout(checkpointed_step, n_state, n)
        return (n_state.variables.temp ** 2).sum()

    grad_fn = jax.jit(jax.value_and_grad(loss))
    c, r = time_call(grad_fn, test_val)
    compile_grad.append(c)
    run_grad.append(r)

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(n_values, run_fwd, "o-", label="forward only")
axs[0].plot(n_values, run_grad, "s-", label="forward + gradient")
axs[0].set_yscale("log")
axs[0].set_xlabel("unroll steps (n)")
axs[0].set_ylabel("steady-state wall time (s)")
axs[0].set_title("Runtime (min of 5 repeats, post-compile)")
axs[0].legend()
axs[0].grid(True, which="both", alpha=0.3)

axs[1].plot(n_values, compile_fwd, "o-", label="forward only")
axs[1].plot(n_values, compile_grad, "s-", label="forward + gradient")
axs[1].set_yscale("log")
axs[1].set_xlabel("unroll steps (n)")
axs[1].set_ylabel("first-call wall time (s)")
axs[1].set_title("Compile cost (first call, includes trace + XLA compile)")
axs[1].legend()
axs[1].grid(True, which="both", alpha=0.3)

fig.suptitle("Cost vs rollout length (global_4deg)")
fig.tight_layout()

out_path = f"{PRP}Results/Report/figures/section2_cost_vs_steps.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")

for n, cf, rf, cg, rg in zip(n_values, compile_fwd, run_fwd, compile_grad, run_grad):
    print(f"n={n:3d}  compile_fwd={cf:.3f}s  run_fwd={rf:.4f}s  "
          f"compile_grad={cg:.3f}s  run_grad={rg:.4f}s")
