# %%
# Same grad-check as 02_rbot_grad_check.py / 03_n2_eps_sweep.py, but on the real global_4deg
# grid (90x40x15, real bathymetry -- setups/global_4deg/global_4deg_learning.py) instead of
# the small ACC channel (30x42x15, nisle=2). Different topology/island count -- global
# bathymetry has many more islands than ACC's 2, so this exercises the island-correction
# branch (solve_stream.py's `if state.dimensions["isle"] > 1`) much more heavily.
#
# global_4deg_learning.py hardcodes enable_streamfunction=False in set_parameter (see its
# module docstring) -- same subclass-and-flip trick as the ACC scripts. eq_of_state_type is
# left at its existing 3 (unrelated to this check, not touching it -- see debugging_mld's
# gsw.py note for why that was set that way).
#
# eps chosen from 03_n2_eps_sweep.py's lesson: loss scale here is large (sum of squares of a
# full 3D field), so eps=1e-8 (debugging_mld's convention, tuned for O(1) losses) would just
# be float64 cancellation noise -- sweep a few eps values instead of picking one blind.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import time
import jax
import jax.numpy as jnp

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from veros import veros_routine
from setups.global_4deg.global_4deg_learning import GlobalFourDegreeSetup


class GlobalFourDegreeStreamSetup(GlobalFourDegreeSetup):
    @veros_routine
    def set_parameter(self, state):
        GlobalFourDegreeSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.enable_streamfunction = True


def pure_step(setup):
    def ps(state):
        n_state = state.copy()
        setup.step(n_state)
        return n_state

    return ps


def set_var(state, var_name, value):
    n_state = state.copy()
    with n_state.variables.unlock():
        setattr(n_state.variables, var_name, value)
    return n_state


def rollout(step_fn, state, n):
    for _ in range(n):
        state = step_fn(state)
    return state


def agg_sum_sq(state, var_name="psi"):
    return (getattr(state.variables, var_name) ** 2).sum()


t0 = time.time()
g4d = GlobalFourDegreeStreamSetup()
g4d.setup()
print(f"setup done in {time.time() - t0:.1f}s")

nisle = g4d.state.dimensions["isle"]
print(f"nisle = {nisle}  (island-correction branch exercised: {nisle > 1})")

vs = g4d.state.variables
for name in ("psi", "u", "v", "dpsi"):
    arr = getattr(vs, name)
    print(f"after setup: {name}  nan={bool(jnp.any(jnp.isnan(arr)))}  finite={bool(jnp.all(jnp.isfinite(arr)))}")

print("running 3 eager steps...")
t0 = time.time()
for i in range(3):
    g4d.step(g4d.state)
    vs = g4d.state.variables
    nan_psi = bool(jnp.any(jnp.isnan(vs.psi)))
    nan_u = bool(jnp.any(jnp.isnan(vs.u)))
    nan_v = bool(jnp.any(jnp.isnan(vs.v)))
    print(f"step {i}: nan(psi)={nan_psi}  nan(u)={nan_u}  nan(v)={nan_v}  ({time.time() - t0:.1f}s elapsed)")

step_fn = pure_step(g4d)
base_state = g4d.state

# NOTE: r_bot (used in 02/03 on ACC) is a dead end here -- global_4deg_learning.py never
# sets enable_bottom_friction=True (defaults False, veros/settings.py:50), so the bottom
# friction term referencing r_bot is never wired into the graph at all: gradient is exactly
# 0 regardless of step count, that's a physics switch, not an autodiff bug. Using c_k
# instead (TKE mixing length constant) -- it's one of the params already converted from
# Setting to Variable for this fork's differentiability work (veros/variables.py:590,
# alongside r_bot, c_eps), and enable_tke=True is active here, so it should have a real
# effect on density mixing -> hydrostatic pressure -> streamfunction forcing from step 1.
test_val = 0.1
n_values = [1, 2, 5]
eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-8]

for n in n_values:

    def loss(v, n=n):
        n_state = set_var(base_state, "c_k", jnp.full_like(base_state.variables.c_k, v))
        n_state = rollout(step_fn, n_state, n)
        return agg_sum_sq(n_state)

    t0 = time.time()
    loss_jit = jax.jit(loss)
    grad_jit = jax.jit(jax.value_and_grad(loss))

    loss_val, grad = grad_jit(jnp.array(test_val))
    print(f"n={n}  loss={float(loss_val):.6e}  autodiff_grad={float(grad):.6e}  ({time.time() - t0:.1f}s)")

    for eps in eps_values:
        num_grad = (loss_jit(jnp.array(test_val) + eps) - loss_jit(jnp.array(test_val) - eps)) / (2 * eps)
        rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
        print(f"  eps={eps:.0e}  numerical={float(num_grad):.6e}  rel_err={rel_err:.4e}")

    del loss_jit, grad_jit
    jax.clear_caches()
