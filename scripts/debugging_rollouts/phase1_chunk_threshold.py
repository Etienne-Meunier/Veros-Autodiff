# %%
# Phase 1 (debugging_rollouts): where does scan_unrolled's compile actually break as
# chunk_size grows? Motivated by scan_unrolled TIMING OUT (>600s) at n=20/chunk=10,
# despite checkpointing a SMALLER unit than plain's full 20-step scan (which compiled
# in 259s) -- so it's not step count driving the blowup. Working hypothesis instead:
# report-mld-2's old design -- jax.checkpoint wrapping a SINGLE step, fed into one
# plain lax.scan -- compiled fine throughout (only failed at *runtime*, O(n) carry
# OOM at n=400). What's different about scan_scan/scan_unrolled is checkpointing a
# MULTI-step chunk and making THAT the scan body -- differentiating a custom-VJP'd
# (checkpoint) function that itself contains several steps, as the direct traced body
# of another primitive (scan) that also needs its own reverse-mode rule, may hit a
# much more expensive XLA path than either component alone.
#
# n=20 fixed, chunk_size sweeps [1, 2, 4, 8] -- starts at chunk_size=1 (the known-
# working single-step-checkpoint-per-scan-iteration pattern) DELIBERATELY, so that if
# even that fails to compile, we find out immediately rather than after burning time
# assuming bigger chunks were the problem. Stops at the first TIMEOUT/CRASH (larger
# chunk_size assumed to only get worse). Each successful chunk_size's grad is compared
# against the previous successful one as a free correctness check along the way.
#
# Logs to $STORE (common.py's STORE_DIR), not the repo -- see common.py's STORE_DIR
# docstring. Writes the CSV after every config, not just at the end.
from __init__ import PRP
import sys

sys.path.append(PRP)

from common import run_worker, write_csv_incremental, STORE_DIR

N = 20
CHUNK_SIZES = [1, 2, 4, 8]
REL_ERR_TOL = 1e-8

csv_path = f"{STORE_DIR}/phase1_chunk_threshold.csv"

rows = []
prev_ok_grad = None
prev_ok_chunk = None

for chunk_size in CHUNK_SIZES:
    r = run_worker("scan_unrolled", N, chunk_size)
    rows.append(r)
    write_csv_incremental(rows, csv_path)

    if r["status"] != "OK":
        print(f"\nSTOPPING: chunk_size={chunk_size} status={r['status']} -- not trying larger chunk sizes.")
        break

    if prev_ok_grad is not None:
        rel_err = abs(r["grad"] - prev_ok_grad) / (abs(prev_ok_grad) + 1e-30)
        cmp_status = "PASS" if rel_err < REL_ERR_TOL else "FAIL"
        print(f"chunk_size={chunk_size} vs chunk_size={prev_ok_chunk}: "
              f"grad={r['grad']:.6e} vs {prev_ok_grad:.6e}  rel_err={rel_err:.3e}  {cmp_status}")
        if cmp_status == "FAIL":
            print("WARNING: gradients disagree between chunk sizes -- correctness issue, not just compile time.")

    prev_ok_grad = r["grad"]
    prev_ok_chunk = chunk_size

print(f"\nSaved {csv_path}")
print("\n--- status summary ---")
for r in rows:
    print(f"[chunk_size={r['chunk_size']}] status={r['status']}  "
          f"compile_time_s={r['compile_time_s']}  run_time_s={r['run_time_s']}  grad={r['grad']}")
