# %%
# report-longrollouts-1: dloss/d{c_k,c_eps} for n=[500,1000,2000,3000] on global4deg
# (see common.py's module docstring for why this setup / why double_checkpoint),
# chunk_size=8 fixed (not calibrated separately at this scale -- validated for
# correctness and compile-tractability up to chunk_size=16 at n=20 in
# scripts/debugging_rollouts/phase_double_checkpoint_threshold.py, but this sweep is
# the first real test of whether it also bounds peak memory at n=500..3000). Reports
# compile_time_s, run_time_s, grad, peak_mem_bytes per (n, param).
#
# Failure policy: if a config TIMEOUTs/CRASHes at some n for a given param, skip
# LARGER n for that param (assumed to only get worse) but still run the other param's
# full range independently -- not an all-or-nothing stop.
#
# Logs to $STORE (common.py's STORE_DIR), not the repo. Writes CSV + plot after every
# config, so a kill/crash/laptop-goes-to-sleep-and-this-keeps-running-on-the-remote-
# session mid-sweep still leaves usable partial results.
from __init__ import PRP
import sys

sys.path.append(PRP)

import matplotlib.pyplot as plt

from common import run_worker, write_csv_incremental, STORE_DIR

CHUNK_SIZE = 8
N_VALUES = [500, 1000, 2000, 3000]
PARAMS = {"c_k": 0.08, "c_eps": 0.6}
TIMEOUT_S = 1200  # 20min per config -- generous, this runs unattended

csv_path = f"{STORE_DIR}/scaling_raw_results.csv"
fig_path = f"{STORE_DIR}/scaling.png"

rows = []


def update_plot():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for param in PARAMS:
        pts = [r for r in rows if r["param"] == param and r["status"] == "OK"]
        pts.sort(key=lambda r: r["n"])
        if pts:
            ns = [r["n"] for r in pts]
            axs[0].plot(ns, [r["compile_time_s"] for r in pts], "o-", label=param)
            axs[1].plot(ns, [r["run_time_s"] for r in pts], "o-", label=param)
            axs[2].plot(ns, [abs(r["grad"]) for r in pts], "o-", label=param)

    axs[0].set_title("compile_time_s vs n")
    axs[1].set_title("run_time_s vs n")
    axs[2].set_title("|grad| vs n")
    for ax in axs:
        ax.set_xlabel("rollout length n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axs[2].set_yscale("log")

    fig.suptitle(f"report-longrollouts-1: global4deg, double_checkpoint chunk_size={CHUNK_SIZE}, GPU\ngaps = crash/timeout, see CSV status column")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


for param, test_val in PARAMS.items():
    for n in N_VALUES:
        r = run_worker(n, param, test_val, CHUNK_SIZE, timeout_s=TIMEOUT_S)
        rows.append(r)
        write_csv_incremental(rows, csv_path)
        update_plot()

        if r["status"] != "OK":
            print(f"\nSTOPPING param={param} at n={n} (status={r['status']}) -- skipping larger n for this param.")
            break

print(f"\nSaved {csv_path}")
print(f"Saved {fig_path}")
print("\n--- status summary ---")
for r in rows:
    mem_gb = None if r["peak_mem_bytes"] is None else r["peak_mem_bytes"] / 1e9
    print(f"[param={r['param']}][n={r['n']}] status={r['status']}  compile_time_s={r['compile_time_s']}  "
          f"run_time_s={r['run_time_s']}  grad={r['grad']}  peak_mem_GB={mem_gb}")
