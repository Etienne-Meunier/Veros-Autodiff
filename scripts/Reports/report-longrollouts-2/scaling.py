# %%
# report-longrollouts-2: dloss/d{c_k,c_eps} for a trailing 365-step (1yr) boxcar
# average of temp, n=[500,1000,2000,3000,5000,10000] on global4deg. Question:
# does averaging stabilize the gradient at n=5000/10000, where
# report-longrollouts-1's raw (single-snapshot) temp loss went to grad=-4.5e30 /
# 1.4e114 (numerically meaningless)?
#
# lead_chunk_size reuses report-1's per-n calibration (8 up to n=1000, 32 for
# 2000-5000, 64 for n=10000) as a starting point -- not re-derived here, since
# the lead phase is byte-for-byte report-1's rollout(). tail_chunk_size=32 fixed
# (window=365 is constant regardless of n, so one calibration should generalize;
# first config run is itself the check).
#
# Failure policy / logging: same as report-1's scaling.py -- stop on first
# non-OK status per param (skip larger n for that param only), log to $STORE,
# write CSV+plot after every config.
from __init__ import PRP
import sys

sys.path.append(PRP)

import matplotlib.pyplot as plt

from common import run_worker, write_csv_incremental, STORE_DIR

LEAD_CHUNK_BY_N = {500: 8, 1000: 8, 2000: 32, 3000: 32, 5000: 32, 10000: 64}
TAIL_CHUNK_SIZE = 32
N_VALUES = [500, 1000, 2000, 3000, 5000, 10000]
PARAMS = {"c_k": 0.08, "c_eps": 0.6}
TIMEOUT_S = 1800

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
    axs[2].set_title("|grad| vs n (temp_ma loss)")
    for ax in axs:
        ax.set_xlabel("rollout length n")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axs[2].set_yscale("log")

    fig.suptitle(f"report-longrollouts-2: global4deg, temp_ma (window=365) double_checkpoint, GPU\ngaps = crash/timeout, see CSV status column")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


for param, test_val in PARAMS.items():
    for n in N_VALUES:
        lead_chunk_size = LEAD_CHUNK_BY_N[n]
        r = run_worker(n, param, test_val, lead_chunk_size, TAIL_CHUNK_SIZE, timeout_s=TIMEOUT_S)
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
