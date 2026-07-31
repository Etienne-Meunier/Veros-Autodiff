# %%
from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from datetime import datetime
from jax import config
config.update("jax_enable_x64", True)

import jax
sys.path.append(PRP)

from scripts.load_runtime import * #Setup parameters for veros 
from setups.acc.acc_learning import ACCSetup

import jax.numpy as jnp
from jax import vmap

from tqdm import tqdm

# %%
# Spin-up 
warmup_steps = 200
acc = ACCSetup()
acc.setup()
with acc.state.settings.unlock() :
    acc.state.settings.enable_eke = False

with acc.state.variables.unlock() :
     acc.state.variables.r_bot += 1e-5
     acc.state.variables.K_gm_0 += 1000.0

def ps(state) : 
    n_state = state.copy()
    acc.step(n_state)
    return n_state

step_jit = jax.jit(ps)

state = acc.state.copy() # Initial state

# %%
jax.lax.scan(lambda c, _ : (ps(c), _), state, length=5)

# %%


# %%



