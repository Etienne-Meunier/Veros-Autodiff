"""30-year forward integration of setups/global_4deg/global_4deg_mld.py -- real
streamfunction solver, TEOS-10, real ETOPO5 topography/forcing. Plain forward
diagnostic run (no gradients, no jax.lax.scan needed: a python loop over the single
jitted step has flat dispatch cost since the same compiled program is reused every
call, see the timing note in Results/Report/report-mld-forward.md).

Tracks, once per simulated day (dt_tracer = 86400s):
  - area-weighted global-mean mld and mld_ma (720-day exact moving average, see
    setups/global_4deg/global_4deg_mld.py's update_mld_moving_average)
  - a subsampled full-field mld snapshot every GIF_FRAME_EVERY days, for the gif

Only saves raw data here -- no plotting. render.py reads these back and does all the
figure/gif rendering, so colorbar/style tweaks don't need a re-run of this (~15 min)
simulation.

Outputs (under $STORE/VerosAd -- raw data, not git-tracked, syncs via g5k_launcher's
`g5k model`; figures/report stay under git-tracked Results/Report, see render.py):
  report_mld_forward_timeseries.csv   -- day, mld_mean, mld_ma_mean
  report_mld_forward_snapshots.npz    -- frame_days, frames, xt, yt,
                                          final_mld, final_mld_ma
"""
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")
sys.path.append(PRP)

from jax import config

config.update("jax_enable_x64", True)

import os
import csv
import time

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from setups.global_4deg.global_4deg_mld import GlobalFlexibleResolutionSetup

YEARS = 30
DAYS_PER_YEAR = 360  # veros.time.YEAR_LENGTH -- this repo's convention throughout
N_STEPS = YEARS * DAYS_PER_YEAR  # 10800 daily steps (dt_tracer = 86400s)
GIF_FRAME_EVERY = 15  # days -- ~720 frames across 30y

RESULTS_DIR = os.path.join(os.environ["STORE"], "VerosAd")
os.makedirs(RESULTS_DIR, exist_ok=True)


def area_weighted_mean(field, weight):
    valid = ~jnp.isnan(field)
    w = jnp.where(valid, weight, 0.0)
    return jnp.sum(jnp.where(valid, field, 0.0) * w) / jnp.sum(w)


def main():
    g4d = GlobalFlexibleResolutionSetup()
    g4d.setup()

    def pure_step(state):
        n_state = state.copy()
        g4d.step(n_state)
        return n_state

    step_jit = jax.jit(pure_step)

    state = g4d.state.copy()
    interior = (slice(2, -2), slice(2, -2))
    area = state.variables.area_t[interior]
    xt = np.array(state.variables.xt[2:-2])
    yt = np.array(state.variables.yt[2:-2])

    mld_series = []
    mld_ma_series = []
    frames = []
    frame_days = []

    t0 = time.time()
    for day in tqdm(range(1, N_STEPS + 1), desc="30y forward (mld)"):
        state = step_jit(state)
        vs = state.variables
        mld_series.append(area_weighted_mean(vs.mld[interior], area))
        mld_ma_series.append(area_weighted_mean(vs.mld_ma[interior], area))
        if day % GIF_FRAME_EVERY == 0:
            frames.append(vs.mld[interior])
            frame_days.append(day)

    state = jax.block_until_ready(state)
    mld_series = np.array(jax.block_until_ready(mld_series))
    mld_ma_series = np.array(jax.block_until_ready(mld_ma_series))
    frames = np.array(jax.block_until_ready(frames))
    frame_days = np.array(frame_days)
    elapsed = time.time() - t0
    print(f"30y forward run took {elapsed / 60:.1f} min ({elapsed / N_STEPS * 1000:.1f} ms/step)")

    # --- time series CSV ---
    csv_path = f"{RESULTS_DIR}/report_mld_forward_timeseries.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "mld_mean_m", "mld_ma_mean_m"])
        for day, m, ma in zip(range(1, N_STEPS + 1), mld_series, mld_ma_series):
            w.writerow([day, float(m), float(ma)])
    print(f"saved {csv_path}")

    # --- raw spatial snapshots (gif frames + final state) ---
    vs = state.variables
    final_mld = np.array(vs.mld[interior])
    final_mld_ma = np.array(vs.mld_ma[interior])

    npz_path = f"{RESULTS_DIR}/report_mld_forward_snapshots.npz"
    np.savez_compressed(
        npz_path,
        xt=xt,
        yt=yt,
        frame_days=frame_days,
        frames=frames,
        final_day=N_STEPS,
        final_mld=final_mld,
        final_mld_ma=final_mld_ma,
        days_per_year=DAYS_PER_YEAR,
    )
    print(f"saved {npz_path}")


if __name__ == "__main__":
    main()
