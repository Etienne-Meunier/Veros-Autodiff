# %%
# Regression check: chains integrate_tke_kernel (updates tke from c_eps) -> time-level
# rotation (same pointer swap a real step does) -> set_tke_diffusivities_kernel (rebuilds
# kappaM/kappaH/mxl from that tke). Checks autodiff vs central finite difference for each
# output of the second kernel w.r.t. c_eps. Should match closely under both tke_mxl_choice
# values.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import *  # Setup runtime settings for veros
from setups.acc.acc_learning import ACCSetup
from veros.core.tke import integrate_tke_kernel, set_tke_diffusivities_kernel

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

    return acc

def set_c_eps(state, value):
    n_state = state.copy()
    with n_state.variables.unlock():
        n_state.variables.c_eps = value
    return n_state

def rotate_time_levels(state):
    n_state = state.copy()
    with n_state.variables.unlock():
        vs = n_state.variables
        vs.taum1, vs.tau, vs.taup1 = vs.tau, vs.taup1, vs.taum1
    return n_state

# %%
# One coupling hop : integrate_tke_kernel (at c_eps) -> rotate time levels ->
# set_tke_diffusivities_kernel. Returns the second kernel's KernelOutput.
def two_step_call(value, state):
    n_state = set_c_eps(state, value)
    tke_out = integrate_tke_kernel(n_state)
    with n_state.variables.unlock():
        n_state.variables.update(tke_out)
    n_state = rotate_time_levels(n_state)
    return set_tke_diffusivities_kernel(n_state)

# %%
def check(tke_mxl_choice, test_val=0.6, eps=1e-4):
    acc = build_spun_up_acc(tke_mxl_choice)

    out_plus = two_step_call(jnp.array(test_val) + eps, acc.state)

    print(f"\n=== tke_mxl_choice={tke_mxl_choice}  (c_eps={test_val}, eps={eps:.0e}) ===")
    for field in out_plus._fields:
        def loss(value, field=field):
            out = two_step_call(value, acc.state)
            return (getattr(out, field) ** 2).sum()

        loss_val, grad = jax.value_and_grad(loss)(jnp.array(test_val))
        num_grad = (loss(jnp.array(test_val) + eps) - loss(jnp.array(test_val) - eps)) / (2 * eps)
        rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
        print(f"  field={field:16s}  loss={float(loss_val):.6e}  autodiff={float(grad):.6e}  "
              f"numerical={float(num_grad):.6e}  rel_err={rel_err:.4e}")

# %%
for choice in [1, 2]:
    check(choice)
