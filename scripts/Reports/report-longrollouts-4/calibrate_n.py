# %%
# report-longrollouts-4 phase 1: find the longest feasible rollout for the
# nz=64/mld_ma (1yr window, built-in avg disabled -- see common.py) setup.
# c_k only (test_val=0.08) -- this is a calibration pass, not the final result;
# once a good n is found, run_gd.py does the real GD tuning there (both params).
#
# lead_chunk_size grows with n (same idea as report-1/2's calibration, values
# picked conservatively given this setup's much heavier per-step footprint --
# report-1's addendum found chunk=32 needed 10.96GB at just n=500 with the
# built-in (always-on) mld_ma buffer; that buffer is now gone, so this may go
# further at the same chunk_size, but starting cautious). tail_chunk_size=32
# fixed (window=365, independent of n, as in report-longrollouts-2).
#
# Failure policy / logging: same as report-1/2's scaling.py -- stop on first
# non-OK status. Logs to $STORE, writes CSV after every config.
from __init__ import PRP
import sys

sys.path.append(PRP)

from common import run_worker, write_csv_incremental, STORE_DIR

NZ = 64
LEAD_CHUNK_BY_N = {500: 32, 1000: 32, 2000: 64, 3000: 64, 5000: 64, 10000: 128}
TAIL_CHUNK_SIZE = 32
N_VALUES = [500, 1000, 2000, 3000, 5000, 10000]
PARAM, TEST_VAL = "c_k", 0.08
TIMEOUT_S = 1800

csv_path = f"{STORE_DIR}/calibrate_n_raw_results.csv"

rows = []

for n in N_VALUES:
    lead_chunk_size = LEAD_CHUNK_BY_N[n]
    r = run_worker(NZ, n, PARAM, TEST_VAL, lead_chunk_size, TAIL_CHUNK_SIZE, timeout_s=TIMEOUT_S)
    rows.append(r)
    write_csv_incremental(rows, csv_path)

    if r["status"] != "OK":
        print(f"\nSTOPPING at n={n} (status={r['status']}) -- this is the feasibility boundary.")
        break

print(f"\nSaved {csv_path}")
print("\n--- status summary ---")
for r in rows:
    mem_gb = None if r["peak_mem_bytes"] is None else r["peak_mem_bytes"] / 1e9
    print(f"[n={r['n']}] status={r['status']}  lead_chunk={r['lead_chunk_size']}  compile_time_s={r['compile_time_s']}  "
          f"run_time_s={r['run_time_s']}  grad={r['grad']}  peak_mem_GB={mem_gb}")
