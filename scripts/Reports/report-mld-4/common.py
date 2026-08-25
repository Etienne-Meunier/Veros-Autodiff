"""Shared helpers for report-mld-4: ablation of the real config
(GlobalFlexibleMLDLearningSetup: nz=60, gsw/TEOS-10, streamfunction, real ETOPO5
topography) to find which ingredient makes the gradient break by n=20
(report-mld-3's finding, confirmed for both mld and temp loss). See PLAN.md for the
full ablation matrix and rationale.

Self-contained (no cross-import from report-mld-2/common.py, despite two classes
below being logically duplicates of ones already there) -- every report directory in
this repo is self-contained via its own `from __init__ import PRP`, and cross-report
imports would fight that pattern for no real benefit here (the classes are a few
lines each).
"""
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from veros import veros_routine
from veros.core.operators import update, at
import veros.tools
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup
from setups.acc.acc_learning import ACCSetup
from tqdm import tqdm

TRUE_PARAMS = dict(c_k=0.1, c_eps=0.7)


class GswOnlyFullGridSetup(GlobalFlexibleMLDLearningSetup):
    """Full grid (nz=60, ETOPO5) with gsw (eq_of_state_type=5) but streamfunction
    forced back off. Duplicate of report-mld-2/common.py's class of the same name."""

    @veros_routine
    def set_parameter(self, state):
        GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.enable_streamfunction = False


class StreamOnlyFullGridSetup(GlobalFlexibleMLDLearningSetup):
    """Full grid (nz=60, ETOPO5) with streamfunction on but eq_of_state_type forced
    back to nonlin2 (3). Duplicate of report-mld-2/common.py's class of the same
    name."""

    @veros_routine
    def set_parameter(self, state):
        GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.eq_of_state_type = 3


class NeitherFullGridSetup(GlobalFlexibleMLDLearningSetup):
    """Full grid (nz=60, ETOPO5) with both gsw and streamfunction forced off
    (nonlin2 + normal pressure solve) -- the fourth cell of the 2x2 gsw x
    streamfunction grid, completing GswOnlyFullGridSetup/StreamOnlyFullGridSetup."""

    @veros_routine
    def set_parameter(self, state):
        GlobalFlexibleMLDLearningSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.enable_streamfunction = False
            state.settings.eq_of_state_type = 3


class ACCFullSetup(ACCSetup):
    """Idealized ACC channel (setups/acc/acc_learning.py) bumped to nz=60,
    gsw/TEOS-10, streamfunction on -- no real bathymetry/forcing at all, unlike the
    other four rows. See PLAN.md's "Idealized-topography class definition" for the
    reasoning. ACCSetup's own set_grid hardcodes a 15-level dzt, so it's fully
    overridden here (not chained) using the same nz-parameterized dzt pattern
    global_4deg_mld_learning.py's set_grid uses.

    ACCSetup's default iso_slopec=0.01 (tuned for its original 15-level grid) fails
    veros/core/isoneutral/isoneutral.py's check_isoneutral_slope_crit once nz is
    bumped to 60 with this borrowed vertical-grid generator (delta_iso1 drops below
    iso_slopec -- RuntimeError at setup(), not a gradient issue). Rather than pick an
    arbitrary looser threshold, reuse the real config's own isoneutral tuning
    (iso_slopec=0.001, iso_dslope=0.004, K_iso_steep=1000.0 --
    global_4deg_mld_learning.py's values) since that's the same config this whole
    ablation is comparing against."""

    max_depth = 5400.0
    min_depth = 4.0

    @veros_routine
    def set_parameter(self, state):
        ACCSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.nz = 60
            state.settings.eq_of_state_type = 5
            state.settings.enable_streamfunction = True
            state.settings.K_iso_steep = 1000.0
            state.settings.iso_dslope = 4.0 / 1000.0
            state.settings.iso_slopec = 1.0 / 1000.0

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings
        vs.dxt = update(vs.dxt, at[...], 2.0)
        vs.dyt = update(vs.dyt, at[...], 2.0)
        vs.dzt = veros.tools.get_vinokur_grid_steps(
            settings.nz, self.max_depth, self.min_depth, refine_towards="lower"
        )


SETUPS = {
    "gsw_only": GswOnlyFullGridSetup,
    "stream_only": StreamOnlyFullGridSetup,
    "neither": NeitherFullGridSetup,
    "acc_full": ACCFullSetup,
}


def spin_up(setup_cls, warmup_steps=20, desc=None):
    g4d = setup_cls()
    g4d.setup()

    with g4d.state.variables.unlock():
        g4d.state.variables.c_k += 0.0
        g4d.state.variables.c_eps += 0.0

    def ps(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    step_jit = jax.jit(ps)

    state = g4d.state.copy()
    for _ in tqdm(range(warmup_steps), desc=desc or f"spin-up ({setup_cls.__name__})"):
        state = step_jit(state)
    g4d.state = state

    return g4d, step_jit


def make_diff_step(g4d):
    def pure_step(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    # No jax.checkpoint here: applied per-chunk in rollout() instead -- see
    # report-mld-2/common.py's rollout() docstring for the full OOM-avoidance
    # rationale. n=20 here is small enough it wouldn't strictly need chunking, but
    # kept for consistency with every other report's rollout().
    return pure_step


def set_vars(state, **values):
    n_state = state.copy()
    with n_state.variables.unlock():
        for name, value in values.items():
            setattr(n_state.variables, name, value)
    return n_state


DEFAULT_CHECKPOINT_CHUNK_SIZE = 4


def rollout(step_fn, state, iterations, chunk_size=DEFAULT_CHECKPOINT_CHUNK_SIZE):
    """double_checkpoint (ported from report-longrollouts-1/common.py's validated
    rollout()): per-step jax.checkpoint(step) inside the inner scan, PLUS an outer
    jax.checkpoint wrapping that whole inner scan, itself scanned at the outer level.

    The single-level version tried first here (checkpoint only around each whole
    chunk, no per-step checkpoint) OOM'd at n=250 in a non-monotonic, unpredictable
    way as chunk_size was tuned (chunk_size=2: 26.6GB: chunk_size=4: ~19GB (closest to
    fitting); chunk_size=25: 49.9GB -- worse in both directions from 4, ruling out a
    simple "just pick a better chunk_size" fix). report-longrollouts-1 established
    that the extra per-step checkpoint is what actually bounds memory reliably on
    this class of long/heavy rollout; this repo's other single-level-checkpoint
    rollout() (report-mld-2, report-mld-3, and this file's earlier version) were
    only ever validated at short n (<=100) where the difference doesn't show up.
    Checkpointing recomputes exactly regardless of structure, so this doesn't change
    the gradient value -- cached n=5/20/75 results (computed with the single-level
    version) stay valid."""
    n_full, remainder = divmod(iterations, chunk_size)
    step_ckpt = jax.checkpoint(step_fn)

    def block_fn(s, length):
        s, _ = jax.lax.scan(lambda c, _: (step_ckpt(c), None), s, length=length)
        return s

    if n_full:
        block_ckpt = jax.checkpoint(lambda s: block_fn(s, chunk_size))
        state, _ = jax.lax.scan(lambda c, _: (block_ckpt(c), None), state, length=n_full)
    if remainder:
        state = jax.checkpoint(lambda s: block_fn(s, remainder))(state)

    return state


def temp_agg_function(state, target_state):
    """Squared error on full-field temp, unmasked -- same form as report-1/report-2's
    temp loss, and report-mld-3's temp_agg_function."""
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()
