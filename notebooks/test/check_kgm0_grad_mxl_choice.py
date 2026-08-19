# %%
# Standalone gradient sanity check for K_gm_0 on the ACC setup, comparing autodiff
# (jax.value_and_grad) against a central finite-difference estimate, for both values of
# tke_mxl_choice :
#   - 1 : simple per-cell bound (distance to surface/bottom)
#   - 2 : bounded length scale as in mitgcm/OPA (recursive cross-z-level min/max chain)
#
# Same script as check_ceps_grad_mxl_choice.py, swapped to K_gm_0. K_gm_0 doesn't touch the
# TKE mixing-length bound at all (it only sets vs.K_gm for the skew-diffusion flux, see
# veros/veros/core/eke.py), so tke_mxl_choice shouldn't matter for it — this is the control
# case for the c_eps investigation.
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

def set_K_gm_0(state, value):
    n_state = state.copy()
    with n_state.variables.unlock():
        n_state.variables.K_gm_0 = value
    return n_state

def rollout(step_fn, state, iterations):
    for _ in range(iterations):
        state = step_fn(state)
    return state

# %%
# For a given tke_mxl_choice : build a target state, define loss(K_gm_0) = distance to
# target after `pred_iter` steps, and compare autodiff grad to central finite differences
# at K_gm_0 = 900 (away from the true value 1000, so the loss isn't degenerately 0 there)
def check(tke_mxl_choice, pred_iter=5, test_val=900.0):
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
        n_state = set_K_gm_0(acc.state, value)
        n_state = rollout(step_diff, n_state, pred_iter)
        return agg_function(n_state)

    loss, grad = jax.value_and_grad(loss_fn)(jnp.array(test_val))
    print(f"\n=== tke_mxl_choice={tke_mxl_choice}  (pred_iter={pred_iter}, K_gm_0={test_val}) ===")
    print(f"loss={float(loss):.6e}  autodiff_grad={float(grad):.6e}")

    # eps sweep: numerical gradient should be roughly eps-independent if it's a real signal
    # (not finite-difference noise). c_eps's sweep used absolute eps 1e-2..1e-6, appropriate
    # for its O(1) scale; K_gm_0 is O(1e3), so those same absolute steps are 5 orders of
    # magnitude too small in *relative* terms and just measure float cancellation noise (this
    # was confirmed empirically: the first K_gm_0 run showed rel_err climbing from ~0.3% at
    # eps=1e-2 to >100% at eps=1e-6, i.e. degrading as eps shrinks — the opposite of a real
    # mismatch, where FD is stable across eps). Rescaled here to the same *relative* range
    # (~1e-5 to ~1e-1 of the test value) by multiplying the c_eps eps list by K_gm_0/c_eps
    # magnitude (900 / 0.6 = 1500) and rounding: eps in {15, 1.5, 0.15, 0.015, 0.0015}.
    for eps in [15.0, 1.5, 0.15, 0.015, 0.0015]:
        num_grad = (loss_fn(jnp.array(test_val) + eps) - loss_fn(jnp.array(test_val) - eps)) / (2 * eps)
        rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
        print(f"  eps={eps:.4g}  numerical_grad={float(num_grad):.6e}  rel_err={rel_err:.4e}")

    # rollout-length sweep. Unlike c_eps, K_gm_0 affects the skew flux within the SAME step
    # (no one-step lag), so n=1 is expected to already be nonzero here.
    # Uses a plain sum(temp^2) loss (not the target-distance loss above) since we're only
    # probing how the mismatch grows with rollout length, not fitting to a target.
    # eps=0.15 chosen from the sweep above: same relative scale as c_eps's eps=1e-4 (~1.7e-4
    # of the test value), scaled up for K_gm_0's O(1e3) magnitude.
    eps = 0.15
    for n in [1, 2, 3, 5]:
        def loss_n(value, n=n):
            n_state = set_K_gm_0(acc.state, value)
            n_state = rollout(step_diff, n_state, n)
            return (n_state.variables.temp ** 2).sum()

        loss_n_val, grad_n = jax.value_and_grad(loss_n)(jnp.array(test_val))
        num_grad_n = (loss_n(jnp.array(test_val) + eps) - loss_n(jnp.array(test_val) - eps)) / (2 * eps)
        rel_err_n = abs(float(grad_n) - float(num_grad_n)) / (abs(float(num_grad_n)) + 1e-30)
        print(f"  n={n}  autodiff={float(grad_n):.6e}  numerical={float(num_grad_n):.6e}  rel_err={rel_err_n:.4e}")

# %%
for choice in [1, 2]:
    check(choice)
