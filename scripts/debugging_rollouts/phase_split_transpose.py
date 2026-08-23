# %%
# Phase (debugging_rollouts): does split_transpose (see common.py's
# rollout_split_transpose docstring -- single unchunked scan, per-step
# checkpoint(policy=nothing_saveable, prevent_cse=False), scan's
# _split_transpose=True) compile fast AND stay memory-bounded as n grows, without any
# manual chunking? Suggested externally after sharing the scan_scan/scan_unrolled
# compile-time blowup (see common.py's module docstring) -- simpler than anything
# tried so far, targets the actual mechanism (scan's own O(n) transpose structure)
# instead of working around it with manual chunk boundaries.
#
# n=[20, 100, 400, 1000, 3000] -- n=20 first doubles as a correctness check against
# plain's already-known reference grad (-24.656791872202238, from
# phase1_chunk_threshold.py). Tracks peak_mem_bytes explicitly: if this approach
# really avoids scan's O(n) carry-storage problem, peak memory should stay roughly
# flat across n instead of growing -- that's the real test, not just "does it
# compile". Stops at the first TIMEOUT/CRASH (larger n assumed to only get worse).
#
# Logs to $STORE (common.py's STORE_DIR), not the repo. Writes the CSV after every n.
from __init__ import PRP
import sys

sys.path.append(PRP)

from common import run_worker, write_csv_incremental, STORE_DIR

N_VALUES = [20, 100, 400, 1000, 3000]
PLAIN_REFERENCE_GRAD = -24.656791872202238  # plain, n=20, from phase1_chunk_threshold.py
REL_ERR_TOL = 1e-6  # looser than the chunk-vs-chunk 1e-8 -- different checkpoint policy/backend path

csv_path = f"{STORE_DIR}/phase_split_transpose.csv"

rows = []

for n in N_VALUES:
    r = run_worker("split_transpose", n, 0)
    rows.append(r)
    write_csv_incremental(rows, csv_path)

    if r["status"] != "OK":
        print(f"\nSTOPPING: n={n} status={r['status']} -- not trying larger n.")
        break

    if n == 20:
        rel_err = abs(r["grad"] - PLAIN_REFERENCE_GRAD) / (abs(PLAIN_REFERENCE_GRAD) + 1e-30)
        cmp_status = "PASS" if rel_err < REL_ERR_TOL else "FAIL"
        print(f"n=20 vs plain reference: grad={r['grad']:.6e} vs {PLAIN_REFERENCE_GRAD:.6e}  "
              f"rel_err={rel_err:.3e}  {cmp_status}")
        if cmp_status == "FAIL":
            print("WARNING: grad does not match plain's reference -- correctness issue, stopping.")
            break

print(f"\nSaved {csv_path}")
print("\n--- status summary ---")
for r in rows:
    mem_gb = None if r["peak_mem_bytes"] is None else r["peak_mem_bytes"] / 1e9
    print(f"[n={r['n']}] status={r['status']}  compile_time_s={r['compile_time_s']}  "
          f"run_time_s={r['run_time_s']}  grad={r['grad']}  peak_mem_GB={mem_gb}")
