# %%
# Regression check for safe_sqrt's gradient clip (veros/veros/core/operators.py). Checks
# d(safe_sqrt(x))/dx, autodiff vs central finite difference, both on the bare function
# across a range of x and on the real (spun-up) ACC state's TKE field.
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

# load_runtime must run before any veros.core import: it sets the jax backend, and
# veros.runtime forbids changing it once core modules are imported (see the crash this
# script hit when safe_sqrt was imported first, under the default numpy backend).
from scripts.load_runtime import *

import jax.numpy as jnp
from veros.core.operators import safe_sqrt

CLIP_X = 1e-6 ** 2  # gradient clip is 0.5/max(1e-6, sqrt(x)), i.e. active for x < 1e-12

# %%
# Pointwise check : d(safe_sqrt(x))/dx via autodiff vs central finite difference
xs = jnp.array([1e-6, 1e-5, 1e-4, 5e-4, 9e-4, 1e-3, 2e-3, 1e-2, 1e-1, 1.0, 10.0])
eps = 1e-7

print(f"clip threshold: x = {CLIP_X:.0e}\n")
print(f"{'x':>10s}  {'autodiff':>12s}  {'numerical':>12s}  {'rel_err':>10s}  {'below clip?':>12s}")
for x in xs:
    grad = jax.grad(safe_sqrt)(x)
    num_grad = (safe_sqrt(x + eps) - safe_sqrt(x - eps)) / (2 * eps)
    rel_err = abs(float(grad) - float(num_grad)) / (abs(float(num_grad)) + 1e-30)
    print(f"{float(x):10.2e}  {float(grad):12.6e}  {float(num_grad):12.6e}  {rel_err:10.4e}  {str(x < CLIP_X):>12s}")

# %%
# On the real ACC state : what fraction of TKE cells actually fall below the clip
# threshold, where this bias would apply ?
from setups.acc.acc_learning import ACCSetup
from tqdm import tqdm

acc = ACCSetup()
acc.setup()
with acc.state.settings.unlock():
    acc.state.settings.enable_eke = False

with acc.state.variables.unlock():
    acc.state.variables.r_bot += 1e-5
    acc.state.variables.K_gm_0 += 1000.0

def ps(state):
    n_state = state.copy()
    acc.step(n_state)
    return n_state

step_jit = jax.jit(ps)
state = acc.state.copy()
for _ in tqdm(range(200), desc="spin-up"):
    state = step_jit(state)
acc.state = state

tke = acc.state.variables.tke[:, :, :, acc.state.variables.tau]
water_cells = acc.state.variables.maskW.astype(bool)
tke_water = tke[water_cells]

frac_below_clip = float((tke_water < CLIP_X).mean())
print(f"\nfraction of water-column TKE cells with tke < {CLIP_X:.0e} (clip regime): {frac_below_clip:.2%}")
print(f"tke stats over water cells: min={float(tke_water.min()):.3e}  "
      f"median={float(jnp.median(tke_water)):.3e}  max={float(tke_water.max()):.3e}")
