import sys
sys.path.append('../../')

from veros.routines import veros_routine
from veros.core import momentum, thermodynamics, advection, eke, tke, utilities, isoneutral, numerics
from veros import diagnostics, restart
from jax.experimental import checkify
import jax

@veros_routine
def set_forcing(state):
    vs = state.variables
    vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

@veros_routine
def my_step(state) :
    vs = state.variables
    settings = state.settings
    #print('begin : ', vs.tke.sum())


    restart.write_restart(state)

    #set_forcing(state)


    #eke.set_eke_diffusivities(state) # (routine) Nsqr -> K_m, K_iso (L_rossby ...)

    #tke.set_tke_diffusivities(state) # (routine) Nsqr, tke -> KappaM, KappaH, Rinumber, K_diss_v


    my_momentum(state)


    #thermodynamics.thermodynamics(state)

    # if settings.enable_eke or settings.enable_tke or settings.enable_idemix:
    #     print('calculate velocity')
    #     advection.calculate_velocity_on_wgrid(state) # (routine)

    #eke.integrate_eke(state) # (routine)

    #jax.debug.print('before_integrate : {tke}', tke=vs.tke.sum())
    #if state.settings.enable_tke:
    #    jax.debug.print('Integrate tke')
    #    tke.integrate_tke(state) # (routine)

    #jax.debug.print('after_integrate : : {tke}', tke=vs.tke.sum())
    #vs.u = utilities.enforce_boundaries(vs.u, settings.enable_cyclic_x)
    #vs.v = utilities.enforce_boundaries(vs.v, settings.enable_cyclic_x)

    #if settings.enable_tke:
    #    vs.tke = utilities.enforce_boundaries(vs.tke, settings.enable_cyclic_x)
    #vs.eke = utilities.enforce_boundaries(vs.eke, settings.enable_cyclic_x)


    #momentum.vertical_velocity(state) # (routine)


    vs.itt = vs.itt + 1
    vs.time = vs.time + settings.dt_tracer

    #isoneutral.isoneutral_diag_streamfunction(state)
    #diagnostics.diagnose(state)
    #diagnostics.output(state)

    print('I')
    #checkify.check(numerics.sanity_check(state), f"solution diverged at iteration {vs.itt}")

    vs.taum1, vs.tau, vs.taup1 = vs.tau, vs.taup1, vs.taum1
    #print('end : ', vs.tke.sum())



from veros.core import advection, diffusion, isoneutral, density, utilities
from veros.core.operators import update, update_add, at
from veros.core.thermodynamics import diag_P_diss_v,advect_temp_salt_enthalpy,vertmix_tempsalt,calc_eq_of_state,surf_densityf


@veros_routine
def my_thermodynamics(state):
    """
    integrate temperature and salinity and diagnose sources of dynamic enthalpy
    """
    """
    Advection tendencies for temperature, salinity and dynamic enthalpy
    """
    print('T7')
    vs = state.variables
    settings = state.settings

    vs.update(advect_temp_salt_enthalpy(state))
    """
    horizontal diffusion
    """
    with state.timers["isoneutral"]:
        if settings.enable_hor_diffusion:
            vs.update(diffusion.tempsalt_diffusion(state))

        if settings.enable_biharmonic_mixing:
            vs.update(diffusion.tempsalt_biharmonic(state))

        """
        sources like restoring zones, etc
        """
        if settings.enable_tempsalt_sources:
            vs.update(diffusion.tempsalt_sources(state))

        """
        isopycnal diffusion
        """
        if settings.enable_neutral_diffusion:
            vs.P_diss_iso = update(vs.P_diss_iso, at[...], 0.0)
            vs.dtemp_iso = update(vs.dtemp_iso, at[...], 0.0)
            vs.dsalt_iso = update(vs.dsalt_iso, at[...], 0.0)

            vs.update(isoneutral.isoneutral_diffusion_pre(state))
            #vs.update(isoneutral.isoneutral_diffusion(state, tr=vs.temp, istemp=True))
            #vs.update(isoneutral.isoneutral_diffusion(state, tr=vs.salt, istemp=False))

    #         if settings.enable_skew_diffusion:
    #             vs.P_diss_skew = update(vs.P_diss_skew, at[...], 0.0)
    #             vs.update(isoneutral.isoneutral_skew_diffusion(state, tr=vs.temp, istemp=True))
    #             vs.update(isoneutral.isoneutral_skew_diffusion(state, tr=vs.salt, istemp=False))

    with state.timers["vmix"]:
        vs.update(vertmix_tempsalt(state))

    with state.timers["eq_of_state"]:
        vs.update(calc_eq_of_state(state, vs.taup1))

    """
    surface density flux
    """
    vs.update(surf_densityf(state))

    with state.timers["vmix"]:
        vs.update(diag_P_diss_v(state))

from veros.core.momentum import tend_coriolisf, tend_tauxyf, momentum_advection
from veros.core import friction, external

@veros_routine
def my_momentum(state):
    """
    solve for momentum for taup1
    """
    print('M6')
    vs = state.variables

    """
    time tendency due to Coriolis force
    """
    #vs.update(tend_coriolisf(state))

    # """
    # wind stress forcing
    # """
    # vs.update(tend_tauxyf(state))

    """
    advection
    """
    #vs.update(momentum_advection(state))

    with state.timers["friction"]:
        my_friction(state)

    """
    external mode
    """
    with state.timers["pressure"]:
        if state.settings.enable_streamfunction:
            external.solve_streamfunction(state)
        else:
            external.solve_pressure(state)


from veros.core.operators import numpy as npx

from veros import veros_routine, veros_kernel, KernelOutput
from veros.variables import allocate
from veros.core import numerics, utilities, isoneutral
from veros.core.operators import update, update_add, at

@veros_kernel
def my_linear_bottom_friction(state):
    """
    linear bottom friction
    dissipation is calculated and added to K_diss_bot
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_bottom_friction_var:
        """
        with spatially varying coefficient
        """
        k = npx.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
        mask = npx.arange(settings.nz) == k[:, :, npx.newaxis]
        vs.du_mix = update_add(
            vs.du_mix,
            at[1:-2, 2:-2],
            -(vs.maskU[1:-2, 2:-2] * vs.r_bot_var_u[1:-2, 2:-2, npx.newaxis]) * vs.u[1:-2, 2:-2, :, vs.tau] * mask,
        )
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(
                diss,
                at[1:-2, 2:-2],
                vs.maskU[1:-2, 2:-2]
                * vs.r_bot_var_u[1:-2, 2:-2, npx.newaxis]
                * vs.u[1:-2, 2:-2, :, vs.tau] ** 2
                * mask,
            )
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_u(state, diss))

        k = npx.maximum(vs.kbot[2:-2, 2:-1], vs.kbot[2:-2, 1:-2]) - 1
        mask = npx.arange(settings.nz) == k[:, :, npx.newaxis]
        vs.dv_mix = update_add(
            vs.dv_mix,
            at[2:-2, 1:-2],
            -(vs.maskV[2:-2, 1:-2] * vs.r_bot_var_v[2:-2, 1:-2, npx.newaxis]) * vs.v[2:-2, 1:-2, :, vs.tau] * mask,
        )
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(
                diss,
                at[2:-2, 1:-2],
                vs.maskV[2:-2, 1:-2]
                * vs.r_bot_var_v[2:-2, 1:-2, npx.newaxis]
                * vs.v[2:-2, 1:-2, :, vs.tau] ** 2
                * mask,
            )
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_v(state, diss))
    else:
        """
        with constant coefficient
        """
        k = npx.maximum(vs.kbot[1:-2, 2:-2], vs.kbot[2:-1, 2:-2]) - 1
        mask = npx.arange(settings.nz) == k[:, :, npx.newaxis]

        vs.du_mix = update_add(
            vs.du_mix, at[1:-2, 2:-2], -1 * vs.maskU[1:-2, 2:-2] * vs.r_bot * vs.u[1:-2, 2:-2, :, vs.tau] * mask
        )
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(
                diss, at[1:-2, 2:-2], vs.maskU[1:-2, 2:-2] * vs.r_bot * vs.u[1:-2, 2:-2, :, vs.tau] ** 2 * mask
            )
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_u(state, diss))

        k = npx.maximum(vs.kbot[2:-2, 2:-1], vs.kbot[2:-2, 1:-2]) - 1
        mask = npx.arange(settings.nz) == k[:, :, npx.newaxis]

        vs.dv_mix = update_add(
            vs.dv_mix, at[2:-2, 1:-2], -1 * vs.maskV[2:-2, 1:-2] * vs.r_bot * vs.v[2:-2, 1:-2, :, vs.tau] * mask
        )
        if settings.enable_conserve_energy:
            diss = allocate(state.dimensions, ("xt", "yu", "zt"))
            diss = update(
                diss, at[2:-2, 1:-2], vs.maskV[2:-2, 1:-2] * vs.r_bot * vs.v[2:-2, 1:-2, :, vs.tau] ** 2 * mask
            )
            vs.K_diss_bot = update_add(vs.K_diss_bot, at[...], numerics.calc_diss_v(state, diss))

    return KernelOutput(du_mix=vs.du_mix, dv_mix=vs.dv_mix, K_diss_bot=vs.K_diss_bot)



from veros.core.friction import *
@veros_routine
def my_friction(state):
    vs = state.variables
    settings = state.settings

    print('F1')
    # """
    # vertical friction
    # """
    # vs.K_diss_v = update(vs.K_diss_v, at[...], 0.0)

    # if settings.enable_implicit_vert_friction:
    #     vs.update(implicit_vert_friction(state))

    # if settings.enable_explicit_vert_friction:
    #     vs.update(explicit_vert_friction(state))

    # """
    # TEM formalism for eddy-driven velocity
    # """
    # if settings.enable_TEM_friction:
    #     vs.update(isoneutral.isoneutral_friction(state))

    # """
    # horizontal friction
    # """
    # if settings.enable_hor_friction:
    #     vs.update(harmonic_friction(state))

    # if settings.enable_biharmonic_friction:
    #     vs.update(biharmonic_friction(state))

    # """
    # Rayleigh and bottom friction
    # """
    # vs.K_diss_bot = update(vs.K_diss_bot, at[...], 0.0)

    # if settings.enable_ray_friction:
    #     vs.update(rayleigh_friction(state))

    if settings.enable_bottom_friction:
        vs.update(linear_bottom_friction(state))

    # if settings.enable_quadratic_bottom_friction:
    #     vs.update(quadratic_bottom_friction(state))

    # """
    # add user defined forcing
    # """
    # if settings.enable_momentum_sources:
    #     vs.update(momentum_sources(state))
