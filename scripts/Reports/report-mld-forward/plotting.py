"""Shared MLD map styling for the report-mld-forward report: filled-contour map,
land in white, wide (not equal-aspect) panel, boxed corner label -- matching the
reference figure style the user gave (paper-style "CONT4 MLD" panel).
"""
import numpy as np
import matplotlib.pyplot as plt

CMAP = "YlGnBu"  # pale yellow-green (shallow) -> dark navy (deep), matches reference
N_LEVELS = 14
VMAX_PERCENTILE = 98  # robust against transient spin-up/deep-convection spikes swamping the colorbar


def plot_mld_map(ax, xt, yt, mld, label, vmax=None, add_colorbar=True):
    """mld: 2D array (nx, ny), negative-down (m). Plotted as positive depth so shallow
    is pale/light and deep is dark, matching the reference figure's convention.
    """
    depth = -mld.T  # (ny, nx), positive down
    if vmax is None:
        vmax = np.nanpercentile(depth, VMAX_PERCENTILE)
    levels = np.linspace(0, vmax, N_LEVELS)

    land = np.isnan(depth)
    ax.set_facecolor("white")
    cf = ax.contourf(
        xt, yt, np.where(land, np.nan, depth),
        levels=levels, cmap=CMAP, extend="max",
    )

    ax.set_xlim(xt.min(), xt.max())
    ax.set_ylim(yt.min(), yt.max())
    ax.set_aspect("auto")  # wide panel, not equal-aspect -- matches reference

    ax.set_xticks([60, 180, 300])
    ax.set_xticklabels(["60°E", "180°E", "300°E"])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N"])

    ax.text(
        0.03, 0.94, label, transform=ax.transAxes,
        ha="left", va="top", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", linewidth=0.8),
    )

    if add_colorbar:
        cbar = plt.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label("MLD (m)")

    return cf
