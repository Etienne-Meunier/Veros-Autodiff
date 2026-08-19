# %%
# Standalone gradient sanity check for c_eps on the ACC setup, comparing autodiff
# (jax.value_and_grad) against a central finite-difference estimate, for both values of
# tke_mxl_choice :
#   - 1 : simple per-cell bound (distance to surface/bottom)
#   - 2 : bounded length scale as in mitgcm/OPA (recursive cross-z-level min/max chain)
#
# Background : c_eps only entered the model as a fixed setting until it was promoted to a
# variable (see veros/veros/variables.py, veros/veros/settings.py, veros/veros/core/tke.py)
# so it can be differentiated w.r.t. A first check under tke_mxl_choice=2 (ACC's default)
# showed the autodiff gradient disagreeing with finite differences by up to ~90% over a
# 5-step rollout, while c_k (same TKE closure, same treatment) agreed to ~1%. Switching to
# tke_mxl_choice=1 brought c_eps back in line (~5% rel error) on ACC. This script isolates
# and reproduces that comparison so it can be re-run / poked at directly.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros
from setups.acc.acc_learning import ACCSetup

import jax.numpy as jnp
from tqdm import tqdm

# %%
def build_spun_up_acc(tke_mxl_choice, warmup_steps=200):
    acc = ACCSetup()
    acc.setup()
    with acc.state.settings.unlock():
        acc.state.settings.enable_eke = False
        acc.state.settings.tke_mxl_choice = tke_mxl_choice

    # c_k needs no explicit init: its Variable has initial=0.1 (veros/veros/variables.py),
    # which state.py fills in at acc.setup() time, same mechanism as c_eps below.
    with acc.state.variables.unlock():
        acc.state.variables.r_bot += 1e-5
        acc.state.variables.K_gm_0 += 1000.0

    def ps(state):
        n_state = state.copy()
        acc.step(n_state)
        return n_state

    step_jit = jax.jit(ps)

    state = acc.state.copy()
    for _ in tqdm(range(warmup_steps), desc=f"spin-up (tke_mxl_choice={tke_mxl_choice})"):
        state = step_jit(state)
    acc.state = state

    return acc, step_jit

def set_c_eps(state, value):
    n_state = state.copy()
    with n_state.variables.unlock():
        n_state.variables.c_eps = value
    return n_state

def rollout(step_fn, state, iterations):
    for _ in range(iterations):
        state = step_fn(state)
    return state

# %%
# For a given tke_mxl_choice : build a target state, define loss(c_eps) = distance to
# target after `pred_iter` steps, and compare autodiff grad to central finite differences
# at c_eps = 0.6 (away from the true value 0.7, so the loss isn't degenerately 0 there)
def check(tke_mxl_choice, pred_iter=5, test_val=0.6):
    acc, step_jit = build_spun_up_acc(tke_mxl_choice)

    def pure_step(state):
        n_state = state.copy()
        acc.step(n_state)
        return n_state

    step_diff = jax.checkpoint(jax.jit(pure_step))

    target_state = rollout(step_jit, acc.state, pred_iter)

    def agg_function(state):
        return ((state.variables.temp - target_state.variables.temp) ** 2).sum()

    def loss_fn(value):
        n_state = set_c_eps(acc.state, value)
        n_state = rollout(step_diff, n_state, pred_iter)
        return agg_function(n_state)

    loss, grad = jax.value_and_grad(loss_fn)(jnp.array(test_val))
    print(f"\n=== tke_mxl_choice={tke_mxl_choice}  (pred_iter={pred_iter}, c_eps={test_val}) ===")
    print(f"loss={float(loss):.6e}  autodiff_grad={float(grad):.6e}")

    # eps sweep: numerical gradient should be roughly eps-independent if it's a real signal
    # (not finite-difference noise)
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        num_grad = (loss_fn(jnp.array(test_val) + eps) - loss_fn(jnp.array(test_val) - eps)) / (2 * eps)
        rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
        print(f"  eps={eps:.0e}  numerical_grad={float(num_grad):.6e}  rel_err={rel_err:.4e}")

    # rollout-length sweep: n=1 should be exactly 0 (c_eps only affects kappaH one step
    # later, via mxl/sqrttke computed from the *previous* step's tke), so start at n=2.
    # Uses a plain sum(temp^2) loss (not the target-distance loss above) since we're only
    # probing how the mismatch grows with rollout length, not fitting to a target.
    eps = 1e-4
    for n in [1, 2, 3, 5]:
        def loss_n(value, n=n):
            n_state = set_c_eps(acc.state, value)
            n_state = rollout(step_diff, n_state, n)
            return (n_state.variables.temp ** 2).sum()

        loss_n_val, grad_n = jax.value_and_grad(loss_n)(jnp.array(test_val))
        num_grad_n = (loss_n(jnp.array(test_val) + eps) - loss_n(jnp.array(test_val) - eps)) / (2 * eps)
        rel_err_n = abs(float(grad_n) - float(num_grad_n)) / (abs(float(num_grad_n)) + 1e-30)
        print(f"  n={n}  autodiff={float(grad_n):.6e}  numerical={float(num_grad_n):.6e}  rel_err={rel_err_n:.4e}")

# %%
for choice in [1, 2]:
    check(choice)
