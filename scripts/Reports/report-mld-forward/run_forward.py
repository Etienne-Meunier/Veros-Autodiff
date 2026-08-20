"""30-year forward integration of setups/global_4deg/global_4deg_mld.py -- real
streamfunction solver, TEOS-10, real ETOPO5 topography/forcing. Plain forward
diagnostic run (no gradients, no jax.lax.scan needed: a python loop over the single
jitted step has flat dispatch cost since the same compiled program is reused every
call, see the timing note in Results/Report/report-mld-forward.md).

Tracks, once per simulated day (dt_tracer = 86400s):
  - area-weighted global-mean mld and mld_ma (720-day exact moving average, see
    setups/global_4deg/global_4deg_mld.py's update_mld_moving_average)
  - a subsampled full-field mld snapshot every GIF_FRAME_EVERY days, for the gif

Outputs:
  Results/report_mld_forward_timeseries.csv        -- day, mld_mean, mld_ma_mean
  Results/Report/figures/report-mld-forward/*.png   -- final-state maps
  Results/Report/figures/report-mld-forward/*.gif   -- mld evolution gif
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from setups.global_4deg.global_4deg_mld import GlobalFlexibleResolutionSetup
from plotting import plot_mld_map

YEARS = 30
DAYS_PER_YEAR = 360  # veros.time.YEAR_LENGTH -- this repo's convention throughout
N_STEPS = YEARS * DAYS_PER_YEAR  # 10800 daily steps (dt_tracer = 86400s)
GIF_FRAME_EVERY = 15  # days -- ~720 frames across 30y

RESULTS_DIR = f"{PRP}Results"
FIG_DIR = f"{PRP}Results/Report/figures/report-mld-forward"
os.makedirs(FIG_DIR, exist_ok=True)


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

    # --- time series figure ---
    years = np.arange(1, N_STEPS + 1) / DAYS_PER_YEAR
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(years, -mld_series, lw=0.6, alpha=0.6, color="#4292c6", label="mld (daily)")
    ax.plot(years, -mld_ma_series, lw=1.8, color="#08306b", label="mld_ma (720-day moving average)")
    ax.set_xlabel("year")
    ax.set_ylabel("area-weighted mean MLD (m)")
    ax.invert_yaxis()
    ax.set_title("Global-mean MLD vs MLD_MA over 30y forward integration")
    ax.legend()
    fig.tight_layout()
    ts_path = f"{FIG_DIR}/timeseries.png"
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    print(f"saved {ts_path}")

    # --- final-state map figure: mld vs mld_ma side by side ---
    vs = state.variables
    final_mld = np.array(vs.mld[interior])
    final_mld_ma = np.array(vs.mld_ma[interior])
    vmax = np.nanpercentile(-np.stack([final_mld, final_mld_ma]), 98)

    final_year = N_STEPS / DAYS_PER_YEAR
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4))
    plot_mld_map(axes[0], xt, yt, final_mld, label=f"MLD -- year {final_year:.1f}", vmax=vmax)
    plot_mld_map(axes[1], xt, yt, final_mld_ma, label=f"MLD_MA (720d) -- year {final_year:.1f}", vmax=vmax)
    fig.tight_layout()
    map_path = f"{FIG_DIR}/final_state_maps.png"
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    print(f"saved {map_path}")

    # --- gif ---
    vmax_gif = np.nanpercentile(-frames, 98)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    gif_frames = []
    for day, frame in zip(frame_days, frames):
        ax.clear()
        plot_mld_map(ax, xt, yt, frame, label=f"MLD -- year {day / DAYS_PER_YEAR:.1f}", vmax=vmax_gif, add_colorbar=(day == frame_days[0]))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        gif_frames.append(buf[..., :3].copy())
    plt.close(fig)

    gif_path = f"{FIG_DIR}/mld_evolution.gif"
    imageio.mimsave(gif_path, gif_frames, duration=0.06, loop=0)  # seconds/frame -- ~43s total for 720 frames
    print(f"saved {gif_path} ({len(gif_frames)} frames)")


if __name__ == "__main__":
    main()
