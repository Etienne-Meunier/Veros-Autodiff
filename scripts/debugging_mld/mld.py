import numpy as np

import jax
from jax import config

config.update("jax_enable_x64", True)  # match numpy float64 precision for the finite-diff check below

import jax.numpy as npx


def mld_from_state(prho, zt, reference_depth=-10., offset_rho_ref=0.03): # MLD from state (original)
    reference_index = npx.max(npx.where(zt < reference_depth)[0]) # Extract index of the first layer below ref depth
    prho_reference = prho[:,:,reference_index] + offset_rho_ref # (2D) : density of ref layer

    prho_below_reference = prho[:,:,:reference_index+1] # Density of layers below reference one
    zt_below_reference = zt[:reference_index+1]  # Depth of layers below reference one
    zt_mask = npx.where(npx.isnan(prho_below_reference), npx.nan, zt_below_reference[npx.newaxis,npx.newaxis,:]) # Mask with land

    drho = prho_below_reference - prho_reference[:,:,npx.newaxis]

    depth_below_mld = npx.nanmax(npx.where(drho > 0, zt_mask, npx.nan), axis=-1)
    above_criterion = npx.logical_and(drho < 0, zt_mask > depth_below_mld[:,:,npx.newaxis])
    depth_above_mld = npx.nanmin(npx.where(above_criterion, zt_mask, npx.nan), axis=-1)
    prho_above_mld = npx.nanmean(npx.where(zt_mask == depth_above_mld[:,:,npx.newaxis], prho_below_reference, npx.nan), axis=-1)
    prho_below_mld = npx.nanmean(npx.where(zt_mask == depth_below_mld[:,:,npx.newaxis], prho_below_reference, npx.nan), axis=-1)

    mld = (prho_reference - prho_below_mld) / (prho_above_mld - prho_below_mld) * (depth_above_mld - depth_below_mld) + depth_below_mld
    return mld


def get_index_mld(prho, maskT, zt, ridx, reference_offset=0.03):
    """Level indices for MLD, discrete selection only -- no gradient path to prho.

    i_below/i_above come out of argmax/argmin over zt (a constant) gated by boolean
    masks built from comparisons on prho -- comparisons carry no gradient, so these
    outputs are naturally zero-gradient w.r.t. prho, same as any argmax-style index.
    well_defined marks columns with no valid below/above level (land, or too shallow
    below reference_depth); mld_from_index masks those out at the very end.
    """
    prho_reference = prho[:, :, ridx] + reference_offset
    prho_below_reference = prho[:, :, : ridx + 1]
    valid = maskT[:, :, : ridx + 1].astype(bool)
    zt_below_reference = zt[: ridx + 1]

    drho = prho_below_reference - prho_reference[:, :, npx.newaxis]

    below_mask = valid & (drho > 0)
    has_below = npx.any(below_mask, axis=-1)
    i_below = npx.argmax(npx.where(below_mask, zt_below_reference, -npx.inf), axis=-1)
    depth_below = zt_below_reference[i_below]

    above_mask = valid & (drho < 0) & (zt_below_reference > depth_below[:, :, npx.newaxis])
    has_above = npx.any(above_mask, axis=-1)
    i_above = npx.argmin(npx.where(above_mask, zt_below_reference, npx.inf), axis=-1)

    well_defined = has_below & has_above
    return i_below, i_above, well_defined


def mld_from_index(prho, zt, ridx, i_below, i_above, well_defined, reference_offset=0.03):
    """MLD from precomputed level indices -- plain gather + arithmetic, fully differentiable."""
    prho_reference = prho[:, :, ridx] + reference_offset
    prho_below = npx.take_along_axis(prho, i_below[:, :, npx.newaxis], axis=-1)[:, :, 0]
    prho_above = npx.take_along_axis(prho, i_above[:, :, npx.newaxis], axis=-1)[:, :, 0]
    depth_below = zt[i_below]
    depth_above = zt[i_above]

    # denom is 0/0 at degenerate (not well_defined) columns -- guard with a safe
    # placeholder before dividing, mask the *result* afterwards. Same necessity as
    # the original: division by (prho_above - prho_below) is inherent to the formula,
    # not something the index/value split removes.
    denom = npx.where(well_defined, prho_above - prho_below, 1.0)
    mld = (prho_reference - prho_below) / denom * (depth_above - depth_below) + depth_below
    return npx.where(well_defined, mld, npx.nan)


def mld_from_prho(prho, maskT, zt, ridx, reference_offset=0.03):
    i_below, i_above, well_defined = get_index_mld(prho, maskT, zt, ridx, reference_offset)
    return mld_from_index(prho, zt, ridx, i_below, i_above, well_defined, reference_offset)


if __name__ == "__main__":
    rng = np.random.default_rng(0)  # plain numpy: jax arrays are immutable, need in-place NaN masking below

    nx, ny, nz = 6, 5, 12
    reference_depth = -10.
    offset = 0.03

    zt = np.sort(rng.uniform(-50., -1., size=nz))  # increasing depth, matches `zt < reference_depth` convention above

    land = rng.random((nx, ny)) < 0.2  # whole-column land, like a real maskT
    increments = rng.uniform(0.01, 0.5, size=(nx, ny, nz))
    prho = np.cumsum(increments[..., ::-1], axis=-1)[..., ::-1]  # denser at depth (small index), monotonic so a crossing exists
    prho[land] = np.nan
    maskT = ~np.isnan(prho)

    ridx = int(npx.max(npx.where(zt < reference_depth)[0]))

    mld_a = mld_from_state(prho, zt, reference_depth=reference_depth, offset_rho_ref=offset)
    mld_b = mld_from_prho(prho, maskT, zt, ridx, reference_offset=offset)

    nan_a, nan_b = npx.isnan(mld_a), npx.isnan(mld_b)
    print("NaN masks match:", bool(npx.array_equal(nan_a, nan_b)))

    both_valid = ~nan_a & ~nan_b
    diff = npx.abs(mld_a[both_valid] - mld_b[both_valid])
    print("max abs diff (valid cells):", float(diff.max()) if diff.size else float("nan"))
    print("n valid cells compared:", int(both_valid.sum()), "/", nx * ny)

    # --- gradient check: finite diff through the old (non-differentiable-by-design)
    # function vs jax.grad through the new (index/value-split) one -- same directional-
    # derivative pattern as scripts/debugging_mld/01_isolate_mld_formula.py.
    prho_j = npx.array(prho)
    zt_j = npx.array(zt)
    maskT_j = npx.array(maskT)

    def loss_new(p):
        mld = mld_from_prho(p, maskT_j, zt_j, ridx, reference_offset=offset)
        valid = ~npx.isnan(mld)
        return npx.where(valid, mld, 0.0).sum() ** 2

    def loss_old(p):
        mld = mld_from_state(p, zt_j, reference_depth=reference_depth, offset_rho_ref=offset)
        valid = ~npx.isnan(mld)
        return npx.where(valid, mld, 0.0).sum() ** 2

    grad_new = jax.grad(loss_new)(prho_j)
    print("nan in grad (new):", bool(npx.any(npx.isnan(grad_new))))

    direction = rng.standard_normal(prho.shape)
    direction[land] = 0.0  # land carries no signal on either side; keep it out of the probe direction
    direction = npx.array(direction / np.linalg.norm(direction))

    print("\ndirectional derivative: jax.grad(new) . direction  vs  finite-diff(old)")
    for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
        l_plus = loss_old(prho_j + eps * direction)
        l_minus = loss_old(prho_j - eps * direction)
        num_dir_deriv = (l_plus - l_minus) / (2 * eps)
        auto_dir_deriv = npx.sum(grad_new * direction)
        rel_err = abs(float(auto_dir_deriv) - float(num_dir_deriv)) / (abs(float(num_dir_deriv)) + 1e-30)
        print(
            f"eps={eps:.0e}  autodiff(new)={float(auto_dir_deriv):.6e}  "
            f"finite_diff(old)={float(num_dir_deriv):.6e}  rel_err={rel_err:.4e}"
        )
