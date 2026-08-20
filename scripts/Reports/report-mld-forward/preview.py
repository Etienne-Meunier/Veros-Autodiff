"""Quick preview of the report-mld-forward map style (plotting.py) on a short,
cheap spin-up -- NOT the full 30y run. Just to sanity-check the look before
committing to the long run + gif.
"""
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")
sys.path.append(PRP)

from jax import config

config.update("jax_enable_x64", True)

import os
import time
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from setups.global_4deg.global_4deg_mld import GlobalFlexibleResolutionSetup
from plotting import plot_mld_map

FIG_DIR = f"{PRP}Results/Report/figures/report-mld-forward"
os.makedirs(FIG_DIR, exist_ok=True)

N_PREVIEW_STEPS = 90

g4d = GlobalFlexibleResolutionSetup()
g4d.setup()


def pure_step(state):
    n_state = state.copy()
    g4d.step(n_state)
    return n_state


step_jit = jax.jit(pure_step)

state = g4d.state.copy()
t0 = time.time()
for _ in range(N_PREVIEW_STEPS):
    state = step_jit(state)
state = jax.block_until_ready(state)
print(f"{N_PREVIEW_STEPS} steps took {time.time() - t0:.1f}s")

vs = state.variables
xt = vs.xt[2:-2]
yt = vs.yt[2:-2]
mld = vs.mld[2:-2, 2:-2]

fig, ax = plt.subplots(figsize=(8, 3.2))
plot_mld_map(ax, xt, yt, mld, label=f"MLD -- day {N_PREVIEW_STEPS}")
fig.tight_layout()
out_path = f"{FIG_DIR}/preview_style.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
