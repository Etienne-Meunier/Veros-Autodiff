# %%
# Report Section 1 : gradient error (autodiff vs central finite difference) as a function
# of the number of unrolled Veros steps, for K_gm_0, r_bot, c_k, c_eps, on global_4deg.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')
sys.path.append(PRP)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from common import spin_up_global4deg, make_diff_step, set_vars, rollout

g4d, step_jit = spin_up_global4deg(200)
step_diff = make_diff_step(g4d)

# (test value away from the true value, FD step size)
PARAM_CONFIG = {
    "K_gm_0": (900.0, 0.15),
    "r_bot": (0.8e-5, 1e-7),
    "c_k": (0.08, 1e-4),
    "c_eps": (0.6, 1e-4),
}
n_values = [2, 5, 10]  # kept lean: each point is its own full JIT compile (see report text);
# n=15 dropped -- its compile alone took >3min per XLA's own "very slow compile" warning

# %%
rel_err = {name: [] for name in PARAM_CONFIG}
for n in tqdm(n_values, desc="unroll steps"):
    target_state = rollout(step_jit, g4d.state, n)

    def agg(state):
        return ((state.variables.temp - target_state.variables.temp) ** 2).sum()

    for name, (test_val, eps) in PARAM_CONFIG.items():
        def loss(v, name=name):
            n_state = set_vars(g4d.state, **{name: v})
            n_state = rollout(step_diff, n_state, n)
            return agg(n_state)

        # jit the whole n-step rollout (forward+backward) into one compiled program per
        # (n, param) instead of eagerly re-tracing/dispatching n chained calls each time
        loss_jit = jax.jit(loss)
        grad_jit = jax.jit(jax.value_and_grad(loss))

        _, grad = grad_jit(jnp.array(test_val))
        num_grad = (loss_jit(jnp.array(test_val) + eps) - loss_jit(jnp.array(test_val) - eps)) / (2 * eps)
        err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
        rel_err[name].append(err)

        del loss_jit, grad_jit
        jax.clear_caches()  # each (n, param) combo is its own compile, used once -- don't accumulate

# %%
fig, ax = plt.subplots(figsize=(7, 5))
for name, errs in rel_err.items():
    ax.plot(n_values, errs, "o-", label=name)

ax.set_yscale("log")
ax.set_xlabel("unroll steps (n)")
ax.set_ylabel("relative error (autodiff vs finite difference)")
ax.set_title("Gradient accuracy vs rollout length (global_4deg)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

out_path = f"{PRP}Results/Report/figures/section1_grad_error_vs_steps.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
