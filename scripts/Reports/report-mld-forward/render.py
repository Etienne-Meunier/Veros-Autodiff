"""Render report-mld-forward's figures/gif from run_forward.py's saved raw data --
no model re-run needed. Edit plotting.py (colormap, vmax percentile, ...) or this
file's own choices and just re-run this script.

Reads:
  $STORE/VerosAd/report_mld_forward_timeseries.csv
  $STORE/VerosAd/report_mld_forward_snapshots.npz

Writes (git-tracked, unlike the raw data above):
  Results/Report/figures/report-mld-forward/timeseries.png
  Results/Report/figures/report-mld-forward/final_state_maps.png
  Results/Report/figures/report-mld-forward/mld_evolution.gif
"""
from __init__ import PRP
import sys

sys.path.append(PRP)

import os
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from plotting import plot_mld_map

RESULTS_DIR = os.path.join(os.environ["STORE"], "VerosAd")
FIG_DIR = f"{PRP}Results/Report/figures/report-mld-forward"
os.makedirs(FIG_DIR, exist_ok=True)

VMAX = 80.0  # m, fixed depth scale (0-80m) instead of a data-derived percentile

CENTER_LON = 180.0  # grid's x_origin=88E puts the cyclic seam through Eurasia (see
# global_4deg_mld.py); reorder columns post-hoc (data is periodic in x, so this is a
# pure relabeling, no re-run needed) so the panel runs CENTER_LON-180 .. CENTER_LON+180
# and the seam falls at 0/360E (Atlantic/Africa) instead, with 180E in the middle.


def center_on_longitude(xt, center=CENTER_LON):
    """Reindex a cyclic-in-x grid so `center` sits in the middle of the array.

    Returns (new_xt, order); apply `order` to any (..., nx, ...) array with
    `np.take(arr, order, axis=<x axis>)` to match.
    """
    lon = ((xt - (center - 180.0)) % 360.0) + (center - 180.0)
    order = np.argsort(lon)
    return lon[order], order


def load_timeseries():
    days, mld, mld_ma = [], [], []
    with open(f"{RESULTS_DIR}/report_mld_forward_timeseries.csv") as f:
        for row in csv.DictReader(f):
            days.append(int(row["day"]))
            mld.append(float(row["mld_mean_m"]))
            mld_ma.append(float(row["mld_ma_mean_m"]))
    return np.array(days), np.array(mld), np.array(mld_ma)


def render_timeseries(days_per_year):
    days, mld_series, mld_ma_series = load_timeseries()
    years = days / days_per_year

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


def render_final_state_maps(npz, days_per_year):
    xt, yt = npz["xt"], npz["yt"]
    final_mld, final_mld_ma = npz["final_mld"], npz["final_mld_ma"]
    final_year = int(npz["final_day"]) / days_per_year

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4))
    plot_mld_map(axes[0], xt, yt, final_mld, label=f"MLD -- year {final_year:.1f}", vmax=VMAX)
    plot_mld_map(axes[1], xt, yt, final_mld_ma, label=f"MLD_MA (720d) -- year {final_year:.1f}", vmax=VMAX)
    fig.tight_layout()
    map_path = f"{FIG_DIR}/final_state_maps.png"
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    print(f"saved {map_path}")


def render_gif(npz, days_per_year):
    xt, yt = npz["xt"], npz["yt"]
    frame_days, frames = npz["frame_days"], npz["frames"]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    gif_frames = []
    for i, (day, frame) in enumerate(zip(frame_days, frames)):
        ax.clear()
        plot_mld_map(
            ax, xt, yt, frame,
            label=f"MLD -- year {day / days_per_year:.1f}",
            vmax=VMAX, add_colorbar=(i == 0),
        )
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        gif_frames.append(buf[..., :3].copy())
    plt.close(fig)

    gif_path = f"{FIG_DIR}/mld_evolution.gif"
    imageio.mimsave(gif_path, gif_frames, duration=0.06, loop=0)  # seconds/frame
    print(f"saved {gif_path} ({len(gif_frames)} frames)")


def main():
    npz = np.load(f"{RESULTS_DIR}/report_mld_forward_snapshots.npz")
    days_per_year = int(npz["days_per_year"])

    data = dict(npz)
    data["xt"], order = center_on_longitude(data["xt"])
    data["frames"] = data["frames"][:, order, :]
    data["final_mld"] = data["final_mld"][order, :]
    data["final_mld_ma"] = data["final_mld_ma"][order, :]

    render_timeseries(days_per_year)
    render_final_state_maps(data, days_per_year)
    render_gif(data, days_per_year)


if __name__ == "__main__":
    main()
