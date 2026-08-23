# %%
# Phase 2 (debugging_rollouts): is scan_unrolled feasible as n grows toward the real
# target (500..3000)? Only run after phase1_correctness.py PASSes -- this trusts
# scan_unrolled's gradient without re-checking it here.
#
# scan_unrolled only (the one strategy phase1 validated), chunk_size=10 fixed, sweeping
# n=[100,200,400,700,1000]. Measures compile_time_s, run_time_s, grad, peak_mem_bytes
# per n. Setup: global4deg (see phase1_correctness.py's docstring).
#
# Logs to $STORE (common.py's STORE_DIR), not the repo -- see phase1_correctness.py's
# docstring for why. CSV and plot are rewritten after every n, not just at the end, so
# stopping early (or a crash) still leaves usable partial results.
from __init__ import PRP
import sys

sys.path.append(PRP)

import matplotlib.pyplot as plt

from common import run_worker, write_csv_incremental, STORE_DIR

CHUNK_SIZE = 10
N_VALUES = [100, 200, 400, 700, 1000]

csv_path = f"{STORE_DIR}/phase2_scaling.csv"
fig_path = f"{STORE_DIR}/phase2_scaling.png"

rows = []


def update_plot():
    ok_rows = [r for r in rows if r["status"] == "OK"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    if ok_rows:
        ns = [r["n"] for r in ok_rows]
        axs[0].plot(ns, [r["compile_time_s"] for r in ok_rows], "o-", color="tab:blue")
        axs[1].plot(ns, [r["run_time_s"] for r in ok_rows], "o-", color="tab:green")
        axs[2].plot(ns, [abs(r["grad"]) for r in ok_rows], "o-", color="tab:purple")

    axs[0].set_title("compile_time_s vs n")
    axs[1].set_title("run_time_s vs n")
    axs[2].set_title("|grad| vs n")
    for ax in axs:
        ax.set_xlabel("rollout length n")
        ax.grid(True, alpha=0.3)
    axs[2].set_yscale("log")

    fig.suptitle(f"scan_unrolled feasibility (global4deg, chunk_size={CHUNK_SIZE}, GPU)\ngaps = crash/timeout, see CSV status column")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


for n in N_VALUES:
    rows.append(run_worker("scan_unrolled", n, CHUNK_SIZE))
    write_csv_incremental(rows, csv_path)
    update_plot()

print(f"\nSaved {csv_path}")
print(f"Saved {fig_path}")
print("\n--- status summary ---")
for r in rows:
    print(f"[n={r['n']}] status={r['status']}  compile_time_s={r['compile_time_s']}  run_time_s={r['run_time_s']}  grad={r['grad']}")
