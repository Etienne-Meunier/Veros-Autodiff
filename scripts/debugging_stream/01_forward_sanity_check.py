# %%
# Step 0 before any grad-check: does enable_streamfunction=True even run, and does the
# island-correction branch (solve_stream.py's `if state.dimensions["isle"] > 1`) get
# exercised on the ACC toy setup? acc_learning.py hardcodes enable_streamfunction=False
# in set_parameter -- subclass it and flip the flag right after the parent's
# set_parameter body runs, before topography/streamfunction_init happen.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from veros import veros_routine
from setups.acc.acc_learning import ACCSetup
import jax.numpy as jnp


class ACCStreamSetup(ACCSetup):
    @veros_routine
    def set_parameter(self, state):
        ACCSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.enable_streamfunction = True


acc = ACCStreamSetup()
acc.setup()

nisle = acc.state.dimensions["isle"]
print(f"nisle = {nisle}  (island-correction branch exercised: {nisle > 1})")

vs = acc.state.variables
for name in ("psi", "u", "v", "dpsi"):
    arr = getattr(vs, name)
    print(f"after setup: {name}  nan={bool(jnp.any(jnp.isnan(arr)))}  finite={bool(jnp.all(jnp.isfinite(arr)))}")

print("running 3 eager steps...")
for i in range(3):
    acc.step(acc.state)
    vs = acc.state.variables
    nan_psi = bool(jnp.any(jnp.isnan(vs.psi)))
    nan_u = bool(jnp.any(jnp.isnan(vs.u)))
    nan_v = bool(jnp.any(jnp.isnan(vs.v)))
    print(f"step {i}: nan(psi)={nan_psi}  nan(u)={nan_u}  nan(v)={nan_v}")
