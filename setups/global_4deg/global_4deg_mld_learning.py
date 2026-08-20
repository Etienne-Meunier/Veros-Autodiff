#!/usr/bin/env python

"""
Differentiable variant of `global_4deg_mld.py` (itself a `veros copy-setup
global_flexible` dump with a custom `mld` mixed-layer-depth diagnostic bolted on),
adapted the same way `global_4deg_learning.py` was adapted from the built-in
`global_4deg` setup (see that module's original docstring):

 - `enable_streamfunction = True`, `eq_of_state_type = 5` (gsw/TEOS-10) : both re-enabled
   now that their differentiability blockers are fixed. `gsw.py`'s every `sqrt` call
   (the 7 `sqrt(sa)` sites plus 3 more discriminant-style sqrts inside `gsw_dyn_enthalpy`/
   `gsw_dHdT`/`gsw_dHdS`) now goes through `safe_sqrt` (veros submodule commit
   `a69b32c`, "Differentiability for teos 5") instead of plain `sqrt` -- forward-neutral
   (`safe_sqrt`'s primal is unmodified `jnp.sqrt`, only the JVP is floored), fixes the
   NaN-gradient blowup `sqrt`'s derivative has at 0 (masked/land cells have salt == 0).
   The barotropic streamfunction solve path (`solve_stream.py`, island line-integral
   solves) was never patched with `lax.cond`, but grad-checked clean as-is: see
   `scripts/debugging_stream/` (01: confirms the island-correction branch is exercised
   on the ACC toy setup, nisle=2; 02: autodiff vs central-finite-difference agrees to
   rel_err~1.5e-5 at n=5 steps through the full path incl. islands; 03: explains 02's
   n=2 disagreement as float64 cancellation noise, not a real bug). Not yet re-run on
   this setup's grid/topology specifically -- only verified on the small ACC setup.
 - `set_diagnostics` -> `diagnostics.clear()` : diagnostics do file I/O and build
   `xarray.Dataset` objects from Python-side state, which doesn't survive `jit`
   tracing. The gradient/report scripts never read `state.diagnostics` anyway.
 - `_read_setup_args`/`setup_args.txt` file-based param injection dropped : params are
   set at runtime via `set_vars` (see `scripts/Reports/report-1/common.py`), not read
   from a file. `c_k`/`c_eps`/`alpha_tke`/`kappaM_min` set to veros defaults here.
 - `after_timestep` (the MLD kernel) ported from `global_4deg_mld.py`, split into
   `get_index_mld()` (discrete level selection, no gradient path to prho) +
   `mld_from_index()` (plain gather + arithmetic, fully differentiable) below, with
   three targeted, forward-preserving fixes over the original: `reference_index`
   recomputed from `zt` fresh every call (the original also did this, via
   `npx.where(...)[0]` on a traced array every step -- but combined it with a
   dynamic-length `[:ridx+1]` slice, which breaks under `jax.jit`/`lax.scan`
   (dynamic shape); here it only ever masks/gathers over the full, statically-shaped
   `zt`/`prho` axis, so recomputing it every call from the live `zt` -- instead of
   caching a Python int once -- stays jit-safe), a `maskT`-based land mask (the
   original used `isnan(prho)`, which only reliably fires under GSW's now-fixed
   sqrt-gradient bug (see above); at land, `prho` is a finite value driven by
   `salt == 0`, not NaN, under gsw or nonlin2 alike, so `isnan` silently stops
   excluding land -- `maskT` is correct regardless of `eq_of_state_type`), and NaN-gradient
   safety (the original's NaN-sentinel + `nanmax`/`nanmin`/`nanmean` pattern produces
   NaN *gradients* -- not NaN values -- at land/degenerate columns that then
   contaminate the gradient at every other column once collected into one array/loss;
   see `scripts/debugging_mld/01_isolate_mld_formula.py` and
   `scripts/debugging_mld/mld.py`, which cross-checks this split's forward values and
   gradient against the original `nanmax`/`nanmin`/`nanmean` formula).

Everything else (grid, real ETOPO5 topography + interpolation, forcing incl. the
solar-penetration `temp_source` term) is unchanged from `global_4deg_mld.py`.
"""

import os
import h5netcdf
import scipy.ndimage

from veros import veros_routine, veros_kernel, KernelOutput, VerosSetup, runtime_settings as rs, runtime_state as rst
from veros.variables import Variable, allocate
from veros.core.utilities import enforce_boundaries
from veros.core.operators import numpy as npx, update, at
import veros.tools
import veros.time

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets("global_flexible", os.path.join(BASE_PATH, "assets_mld.json"))


def get_index_mld(prho, maskT, zt, reference_depth, reference_offset=0.03):
    """Level indices for MLD, discrete selection only -- no gradient path to prho.

    ridx (the reference level) is recomputed from `zt` on every call, not cached --
    zt is the actual grid in scope (never passed in as a stale cached int), so this
    stays correct even in a setup where zt changes at runtime. To stay jit/lax.scan
    -safe with a `zt` that may be a trace value there, this never slices to a
    dynamic-length prefix (`[:ridx+1]`, the bug the original global_4deg_mld.py had --
    breaks under jit, dynamic shape); it only ever masks/gathers over the full,
    statically-shaped (nz,) `zt`/`prho` axis, which works whether `ridx` is a concrete
    int or a jax trace value.

    i_below/i_above come out of argmax/argmin over zt (a constant) gated by boolean
    masks built from comparisons on prho -- comparisons carry no gradient, so these
    outputs are naturally zero-gradient w.r.t. prho, same as any argmax-style index.
    well_defined marks columns with no valid below/above level (land, or too shallow
    below reference_depth); mld_from_index masks those out at the very end. Land/invalid
    cells excluded via `maskT` (authoritative regardless of eq_of_state_type) instead
    of `isnan(prho)` -- under nonlin2 (see module docstring), land cells are exactly
    `0.0`, not NaN, so `isnan` would silently stop excluding land.
    """
    level = npx.arange(zt.shape[-1])
    ridx = npx.max(npx.where(zt < reference_depth, level, -1))

    prho_reference = prho[:, :, ridx] + reference_offset
    valid = maskT.astype(bool) & (level <= ridx)

    drho = prho - prho_reference[:, :, npx.newaxis]

    below_mask = valid & (drho > 0)
    has_below = npx.any(below_mask, axis=-1)
    i_below = npx.argmax(npx.where(below_mask, zt, -npx.inf), axis=-1)
    depth_below = zt[i_below]

    above_mask = valid & (drho < 0) & (zt > depth_below[:, :, npx.newaxis])
    has_above = npx.any(above_mask, axis=-1)
    i_above = npx.argmin(npx.where(above_mask, zt, npx.inf), axis=-1)

    well_defined = has_below & has_above
    return ridx, i_below, i_above, well_defined


def mld_from_index(prho, zt, ridx, i_below, i_above, well_defined, reference_offset=0.03):
    """MLD from precomputed level indices -- plain gather + arithmetic, fully differentiable.

    Same physics/formula as global_4deg_mld.py's original after_timestep (density
    crosses `prho[ridx] + reference_offset` between two levels, MLD = linear
    interpolation of that crossing depth). No NaN/sentinel values ever reach a
    differentiated primitive: i_below/i_above/well_defined are plain int/bool arrays
    (see get_index_mld), gathered via take_along_axis, and the one remaining division
    is guarded by `well_defined` before dividing, with the NaN masked in at the very
    end on a value with no further differentiated ops downstream -- see
    scripts/debugging_mld/01_isolate_mld_formula.py for why that ordering matters.
    """
    prho_reference = prho[:, :, ridx] + reference_offset
    prho_below = npx.take_along_axis(prho, i_below[:, :, npx.newaxis], axis=-1)[:, :, 0]
    prho_above = npx.take_along_axis(prho, i_above[:, :, npx.newaxis], axis=-1)[:, :, 0]
    depth_below = zt[i_below]
    depth_above = zt[i_above]

    # denom is 0/0 at degenerate (not well_defined) columns -- guard with a safe
    # placeholder before dividing, mask the *result* afterwards. Inherent to the
    # formula (division by prho_above - prho_below), not removable by this split.
    denom = npx.where(well_defined, prho_above - prho_below, 1.0)
    mld = (prho_reference - prho_below) / denom * (depth_above - depth_below) + depth_below
    return npx.where(well_defined, mld, npx.nan)


def mld_from_prho(prho, maskT, zt, reference_depth, reference_offset=0.03):
    ridx, i_below, i_above, well_defined = get_index_mld(prho, maskT, zt, reference_depth, reference_offset)
    return mld_from_index(prho, zt, ridx, i_below, i_above, well_defined, reference_offset)


def update_mld_moving_average(mld, mld_history, write_idx, window):
    """Exact boxcar moving average of mld over the last `window` after_timestep calls.

    O(window) storage (mld_history, shape (nx, ny, window)) is unavoidable for an
    *exact* windowed mean -- the value falling out of the window has to be known to
    update a running sum, which means keeping it around. This keeps that storage to
    the minimum: mld_history holds exactly the last `window` mld snapshots (NaN-filled
    before the buffer first wraps), `write_idx` (a single scalar, wraps mod window) is
    the only other state carried -- no separate unbounded step counter. Plain mean
    (not nanmean) over the window: any NaN day (land, or a degenerate/undefined MLD
    column, see mld_from_prho) propagates to the average -- a cell is only ever
    averaged from fully well-defined days, and stays NaN during warm-up until the
    buffer has wrapped at least once.

    Writes via dynamic-index scatter (`.at[:, :, write_idx].set(...)`) and reads via
    mean over the fixed-size window axis -- no variable-length slicing, so this is
    jax.jit/lax.scan-safe the same way get_index_mld/mld_from_index are above.
    """
    mld_history = update(mld_history, at[:, :, write_idx], mld)
    mld_ma = npx.mean(mld_history, axis=-1)
    next_write_idx = (write_idx + 1) % window
    return mld_history, next_write_idx, mld_ma


class GlobalFlexibleMLDLearningSetup(VerosSetup):
    """
    Global model with flexible resolution + mixed-layer-depth diagnostic --
    differentiable / learning variant (see module docstring).
    """

    # global settings
    min_depth = 4.0
    max_depth = 5400.0
    equatorial_grid_spacing_factor = 1.0
    polar_grid_spacing_factor = None

    # depth (m, negative down) above which the MLD reference density is taken;
    # matches global_4deg_mld.py's after_timestep. Recomputed from zt on every
    # after_timestep call (see get_index_mld) -- not cached.
    mld_reference_depth = -10.0

    # window (in after_timestep calls) for the exact MLD_MA moving average -- see
    # update_mld_moving_average.
    mld_ma_window = 720

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings

        settings.identifier = "global_4deg_mld_learning"
        settings.description = "Global model with flexible resolution, mld diagnostic -- differentiable/learning variant"

        settings.nx = 90
        settings.ny = 40
        settings.nz = 60
        settings.dt_mom = 1800.0
        settings.dt_tracer = 86400.0
        settings.runlen = 0.0

        settings.x_origin = 88.0
        settings.y_origin = -76.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        # friction
        settings.enable_hor_friction = True
        settings.A_h = (4 * settings.degtom) ** 3 * 2e-11
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1
        settings.enable_tempsalt_sources = True
        settings.enable_implicit_vert_friction = True

        # differentiable-veros adjustments (see module docstring)
        settings.eq_of_state_type = 5
        settings.enable_streamfunction = True

        # isoneutral
        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 1000.0
        settings.iso_dslope = 4.0 / 1000.0
        settings.iso_slopec = 1.0 / 1000.0
        settings.enable_skew_diffusion = True

        # tke
        settings.enable_tke = True
        # c_k, c_eps are state Variables now (not settings, see veros/variables.py) --
        # left at their built-in defaults (0.1, 0.7) here, same as global_4deg_learning.py.
        # Set at runtime via set_vars (see scripts/Reports/report-1/common.py).
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        # eke
        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        # idemix
        settings.enable_idemix = False
        settings.enable_eke_diss_surfbot = True
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = True
        settings.enable_idemix_hor_diffusion = True

        # custom variables
        state.dimensions["nmonths"] = 12
        state.dimensions["mld_ma_window"] = self.mld_ma_window
        state.var_meta.update(
            t_star=Variable("t_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            s_star=Variable("s_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qnec=Variable("qnec", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qnet=Variable("qnet", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qsol=Variable("qsol", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            divpen_shortwave=Variable("divpen_shortwave", ("zt",), "", "", time_dependent=False),
            taux=Variable("taux", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            tauy=Variable("tauy", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            mld=Variable("mld", ("xt", "yt"), "m", "", time_dependent=True),
            mld_history=Variable(
                "mld_history",
                ("xt", "yt", "mld_ma_window"),
                "m",
                "circular buffer of the last mld_ma_window mld snapshots (see update_mld_moving_average)",
                time_dependent=True,
                initial=npx.nan,
            ),
            mld_ma_index=Variable(
                "mld_ma_index",
                None,
                "",
                "next write position in mld_history's circular buffer",
                dtype="int32",
                initial=0,
                time_dependent=True,
            ),
            mld_ma=Variable(
                "mld_ma", ("xt", "yt"), "m", "exact moving average of mld over the last mld_ma_window steps", time_dependent=True
            ),
        )

    def _get_data(self, var, idx=None):
        if idx is None:
            idx = Ellipsis
        else:
            idx = idx[::-1]

        kwargs = {}
        if rst.proc_num > 1:
            kwargs.update(
                driver="mpio",
                comm=rs.mpi_comm,
            )

        with h5netcdf.File(DATA_FILES["forcing"], "r", **kwargs) as forcing_file:
            var_obj = forcing_file.variables[var]
            return npx.array(var_obj[idx]).T

    @veros_routine(dist_safe=False, local_variables=["dxt", "dyt", "dzt"])
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        if settings.ny % 2:
            raise ValueError("ny has to be an even number of grid cells")

        vs.dxt = update(vs.dxt, at[...], 360.0 / settings.nx)

        if self.equatorial_grid_spacing_factor is not None:
            eq_spacing = self.equatorial_grid_spacing_factor * 160.0 / settings.ny
        else:
            eq_spacing = None

        if self.polar_grid_spacing_factor is not None:
            polar_spacing = self.polar_grid_spacing_factor * 160.0 / settings.ny
        else:
            polar_spacing = None

        vs.dyt = update(
            vs.dyt,
            at[2:-2],
            veros.tools.get_vinokur_grid_steps(
                settings.ny, 160.0, eq_spacing, upper_stepsize=polar_spacing, two_sided_grid=True
            ),
        )
        vs.dzt = veros.tools.get_vinokur_grid_steps(settings.nz, self.max_depth, self.min_depth, refine_towards="lower")

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt[npx.newaxis, :] / 180.0 * settings.pi)
        )

    def _shift_longitude_array(self, vs, lon, arr):
        wrap_i = npx.where((lon[:-1] < vs.xt.min()) & (lon[1:] >= vs.xt.min()))[0][0]
        new_lon = npx.concatenate((lon[wrap_i:-1], lon[:wrap_i] + 360.0))
        new_arr = npx.concatenate((arr[wrap_i:-1, ...], arr[:wrap_i, ...]))
        return new_lon, new_arr

    @veros_routine(dist_safe=False, local_variables=["kbot", "xt", "yt", "zt"])
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings

        with h5netcdf.File(DATA_FILES["topography"], "r") as topography_file:
            topo_x, topo_y, topo_z = (npx.array(topography_file.variables[k], dtype="float").T for k in ("x", "y", "z"))

        topo_z = npx.minimum(topo_z, 0.0)

        # smooth topography to match grid resolution
        gaussian_sigma = (0.5 * len(topo_x) / settings.nx, 0.5 * len(topo_y) / settings.ny)
        topo_z_smoothed = scipy.ndimage.gaussian_filter(topo_z, sigma=gaussian_sigma)
        topo_z_smoothed = npx.where(topo_z >= -1, 0, topo_z_smoothed)

        topo_x_shifted, topo_z_shifted = self._shift_longitude_array(vs, topo_x, topo_z_smoothed)
        coords = (vs.xt[2:-2], vs.yt[2:-2])
        z_interp = allocate(state.dimensions, ("xt", "yt"), local=False)
        z_interp = update(
            z_interp,
            at[2:-2, 2:-2],
            veros.tools.interpolate((topo_x_shifted, topo_y), topo_z_shifted, coords, kind="nearest", fill=False),
        )

        depth_levels = 1 + npx.argmin(npx.abs(z_interp[:, :, npx.newaxis] - vs.zt[npx.newaxis, npx.newaxis, :]), axis=2)
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], npx.where(z_interp < 0.0, depth_levels, 0)[2:-2, 2:-2])
        vs.kbot = npx.where(vs.kbot < settings.nz, vs.kbot, 0)
        vs.kbot = enforce_boundaries(vs.kbot, settings.enable_cyclic_x, local=True)

        # remove marginal seas
        # (dilate to close 1-cell passages, fill holes, undo dilation)
        marginal = scipy.ndimage.binary_erosion(
            scipy.ndimage.binary_fill_holes(scipy.ndimage.binary_dilation(vs.kbot == 0))
        )

        vs.kbot = npx.where(marginal, 0, vs.kbot)

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)
        xt_forc, yt_forc, zt_forc = (self._get_data(k) for k in ("xt", "yt", "zt"))
        zt_forc = zt_forc[::-1]

        # coordinates must be monotonous for this to work
        assert npx.diff(xt_forc).all() > 0
        assert npx.diff(yt_forc).all() > 0

        # determine slice to read from forcing file
        data_subset = (
            slice(
                max(0, int(npx.argmax(xt_forc >= vs.xt.min())) - 1),
                len(xt_forc) - max(0, int(npx.argmax(xt_forc[::-1] <= vs.xt.max())) - 1),
            ),
            slice(
                max(0, int(npx.argmax(yt_forc >= vs.yt.min())) - 1),
                len(yt_forc) - max(0, int(npx.argmax(yt_forc[::-1] <= vs.yt.max())) - 1),
            ),
            Ellipsis,
        )

        xt_forc = xt_forc[data_subset[0]]
        yt_forc = yt_forc[data_subset[1]]

        # initial conditions
        temp_raw = self._get_data("temperature", idx=data_subset)[..., ::-1]
        temp_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), temp_raw, t_grid)
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, :], (temp_data * vs.maskT[2:-2, 2:-2, :])[..., npx.newaxis])

        salt_raw = self._get_data("salinity", idx=data_subset)[..., ::-1]
        salt_data = veros.tools.interpolate((xt_forc, yt_forc, zt_forc), salt_raw, t_grid)
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, :], (salt_data * vs.maskT[2:-2, 2:-2, :])[..., npx.newaxis])

        # wind stress on MIT grid
        time_grid = (vs.xt[2:-2], vs.yt[2:-2], npx.arange(12))
        taux_raw = self._get_data("tau_x", idx=data_subset)
        taux_data = veros.tools.interpolate((xt_forc, yt_forc, npx.arange(12)), taux_raw, time_grid)
        vs.taux = update(vs.taux, at[2:-2, 2:-2, :], taux_data)

        tauy_raw = self._get_data("tau_y", idx=data_subset)
        tauy_data = veros.tools.interpolate((xt_forc, yt_forc, npx.arange(12)), tauy_raw, time_grid)
        vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], tauy_data)

        vs.taux = enforce_boundaries(vs.taux, settings.enable_cyclic_x)
        vs.tauy = enforce_boundaries(vs.tauy, settings.enable_cyclic_x)

        # Qnet and dQ/dT and Qsol
        qnet_raw = self._get_data("q_net", idx=data_subset)
        qnet_data = veros.tools.interpolate((xt_forc, yt_forc, npx.arange(12)), qnet_raw, time_grid)
        vs.qnet = update(vs.qnet, at[2:-2, 2:-2, :], -qnet_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        qnec_raw = self._get_data("dqdt", idx=data_subset)
        qnec_data = veros.tools.interpolate((xt_forc, yt_forc, npx.arange(12)), qnec_raw, time_grid)
        vs.qnec = update(vs.qnec, at[2:-2, 2:-2, :], qnec_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        qsol_raw = self._get_data("swf", idx=data_subset)
        qsol_data = veros.tools.interpolate((xt_forc, yt_forc, npx.arange(12)), qsol_raw, time_grid)
        vs.qsol = update(vs.qsol, at[2:-2, 2:-2, :], -qsol_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        # SST and SSS
        sst_raw = self._get_data("sst", idx=data_subset)
        sst_data = veros.tools.interpolate((xt_forc, yt_forc, npx.arange(12)), sst_raw, time_grid)
        vs.t_star = update(vs.t_star, at[2:-2, 2:-2, :], sst_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        sss_raw = self._get_data("sss", idx=data_subset)
        sss_data = veros.tools.interpolate((xt_forc, yt_forc, npx.arange(12)), sss_raw, time_grid)
        vs.s_star = update(vs.s_star, at[2:-2, 2:-2, :], sss_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        if settings.enable_idemix:
            tidal_energy_raw = self._get_data("tidal_energy", idx=data_subset)
            tidal_energy_data = veros.tools.interpolate((xt_forc, yt_forc), tidal_energy_raw, t_grid[:-1])
            mask_x, mask_y = (i + 2 for i in npx.indices((vs.nx, vs.ny)))
            mask_z = npx.maximum(0, vs.kbot[2:-2, 2:-2] - 1)
            tidal_energy_data[:, :] *= vs.maskW[mask_x, mask_y, mask_z] / vs.rho_0
            vs.forc_iw_bottom[2:-2, 2:-2] = tidal_energy_data

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = vs.zw / efold1_shortwave
        swarg2 = vs.zw / efold2_shortwave
        pen = rpart_shortwave * npx.exp(swarg1) + (1.0 - rpart_shortwave) * npx.exp(swarg2)
        pen = update(pen, at[-1], 0.0)
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[1:], (pen[1:] - pen[:-1]) / vs.dzt[1:])
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[0], pen[0] / vs.dzt[0])

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.update(set_forcing_kernel(state))

    @veros_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics
        diagnostics.clear()

    @veros_routine(
        dist_safe=False,
        local_variables=["zt", "prho", "maskT", "mld", "mld_history", "mld_ma_index", "mld_ma"],
    )
    def after_timestep(self, state):
        vs = state.variables

        mld = mld_from_prho(vs.prho, vs.maskT, vs.zt, self.mld_reference_depth)
        vs.mld = update(vs.mld, at[2:-2, 2:-2], mld[2:-2, 2:-2])

        mld_history, mld_ma_index, mld_ma = update_mld_moving_average(
            vs.mld, vs.mld_history, vs.mld_ma_index, self.mld_ma_window
        )
        vs.mld_history = mld_history
        vs.mld_ma_index = mld_ma_index
        vs.mld_ma = update(vs.mld_ma, at[2:-2, 2:-2], mld_ma[2:-2, 2:-2])


@veros_kernel
def set_forcing_kernel(state):
    vs = state.variables
    settings = state.settings

    t_rest = 30.0 * 86400.0
    cp_0 = 3991.86795711963  # J/kg /K

    year_in_seconds = veros.time.convert_time(1.0, "years", "seconds")
    (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(vs.time, year_in_seconds, year_in_seconds / 12.0, 12)

    # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
    vs.surface_taux = update(vs.surface_taux, at[:-1, :], f1 * vs.taux[1:, :, n1] + f2 * vs.taux[1:, :, n2])
    vs.surface_tauy = update(vs.surface_tauy, at[:, :-1], f1 * vs.tauy[:, 1:, n1] + f2 * vs.tauy[:, 1:, n2])

    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
            )
            ** (3.0 / 2.0),
        )

    # W/m^2 K kg/J m^3/kg = K m/s
    t_star_cur = f1 * vs.t_star[..., n1] + f2 * vs.t_star[..., n2]
    qqnec = f1 * vs.qnec[..., n1] + f2 * vs.qnec[..., n2]
    qqnet = f1 * vs.qnet[..., n1] + f2 * vs.qnet[..., n2]
    vs.forc_temp_surface = (
        (qqnet + qqnec * (t_star_cur - vs.temp[..., -1, vs.tau])) * vs.maskT[..., -1] / cp_0 / settings.rho_0
    )
    s_star_cur = f1 * vs.s_star[..., n1] + f2 * vs.s_star[..., n2]
    vs.forc_salt_surface = 1.0 / t_rest * (s_star_cur - vs.salt[..., -1, vs.tau]) * vs.maskT[..., -1] * vs.dzt[-1]

    # apply simple ice mask
    mask1 = vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] > -1.8
    mask2 = vs.forc_temp_surface > 0
    ice = npx.logical_or(mask1, mask2)
    vs.forc_temp_surface *= ice
    vs.forc_salt_surface *= ice

    # solar radiation
    if settings.enable_tempsalt_sources:
        vs.temp_source = (
            (f1 * vs.qsol[..., n1, None] + f2 * vs.qsol[..., n2, None])
            * vs.divpen_shortwave[None, None, :]
            * ice[..., None]
            * vs.maskT[..., :]
            / cp_0
            / settings.rho_0
        )

    return KernelOutput(
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy,
        temp_source=vs.temp_source,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
    )
