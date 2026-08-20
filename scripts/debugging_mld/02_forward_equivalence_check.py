# %%
# Forward-value equivalence check: the new NaN-gradient-safe mld_from_prho() (inf
# sentinels + max/min/_safe_mean_where) vs a literal transcription of the original
# formula (NaN sentinels + nanmax/nanmin/nanmean), both using the maskT-based land
# mask (isolates just the "safe rewrite" numerical difference from the
# isnan-vs-maskT difference, already validated separately). Should match exactly on
# every well-defined cell, and agree on which cells are NaN (undefined).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403
from veros.core.operators import numpy as npx
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup, mld_from_prho


def mld_reference_nan_style(prho, maskT, zt, ridx):
    prho_reference = prho[:, :, ridx] + 0.03
    prho_below_reference = prho[:, :, : ridx + 1]
    zt_below_reference = zt[: ridx + 1]
    valid_below_reference = maskT[:, :, : ridx + 1].astype(bool)
    zt_mask = npx.where(valid_below_reference, zt_below_reference[npx.newaxis, npx.newaxis, :], npx.nan)
    drho = prho_below_reference - prho_reference[:, :, npx.newaxis]
    depth_below_mld = npx.nanmax(npx.where(drho > 0, zt_mask, npx.nan), axis=-1)
    above_criterion = npx.logical_and(drho < 0, zt_mask > depth_below_mld[:, :, npx.newaxis])
    depth_above_mld = npx.nanmin(npx.where(above_criterion, zt_mask, npx.nan), axis=-1)
    prho_above_mld = npx.nanmean(
        npx.where(zt_mask == depth_above_mld[:, :, npx.newaxis], prho_below_reference, npx.nan), axis=-1
    )
    prho_below_mld = npx.nanmean(
        npx.where(zt_mask == depth_below_mld[:, :, npx.newaxis], prho_below_reference, npx.nan), axis=-1
    )
    mld = (prho_reference - prho_below_mld) / (prho_above_mld - prho_below_mld) * (
        depth_above_mld - depth_below_mld
    ) + depth_below_mld
    return mld


g4d = GlobalFlexibleMLDLearningSetup()
g4d.setup()
g4d.step(g4d.state)
g4d.step(g4d.state)

prho = g4d.state.variables.prho
maskT = g4d.state.variables.maskT
zt = g4d.state.variables.zt
reference_depth = g4d.mld_reference_depth
ridx = int(npx.max(npx.where(zt < reference_depth)[0]))  # concrete index, for mld_reference_nan_style's own slicing

mld_new = mld_from_prho(prho, maskT, zt, reference_depth)
mld_ref = mld_reference_nan_style(prho, maskT, zt, ridx)

both_nan = jnp.isnan(mld_new) & jnp.isnan(mld_ref)
diff = jnp.where(both_nan, 0.0, mld_new - mld_ref)
print("max abs diff (excl both-nan cells):", float(jnp.nanmax(jnp.abs(diff))))
print("nan pattern matches exactly:", bool(jnp.all(jnp.isnan(mld_new) == jnp.isnan(mld_ref))))
print("any non-nan mismatch beyond both-nan:", bool(jnp.any(jnp.isnan(diff) & ~both_nan)))
