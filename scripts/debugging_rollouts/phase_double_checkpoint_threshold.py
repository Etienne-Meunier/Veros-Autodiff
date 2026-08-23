# %%
# Phase (debugging_rollouts): where does double_checkpoint's compile actually break
# as chunk_size grows -- same ladder as phase1_chunk_threshold.py did for
# scan_unrolled (chunk_size=1 OK 177s, chunk_size=2 OK 462s, chunk_size=4 TIMEOUT),
# but for double_checkpoint (see common.py's rollout_double_checkpoint docstring):
# per-step checkpoint(step) inside the inner scan, PLUS an outer checkpoint(block_fn)
# around that whole inner scan. Externally suggested as not redundant with
# split_transpose's per-step-only checkpoint (which compiled flat-time but grew
# memory ~linearly with n, crashing predictably around n=400/16GB -- see
# phase_split_transpose.py's results) -- the outer checkpoint here is meant to
# actually bound that carry-chain growth the way chunking was always intended to,
# while the inner per-step checkpoint (proven compile-fast via split_transpose) may
# be what keeps the outer level's compile tractable where scan_scan's raw-inner-scan
# design wasn't.
#
# n=20 fixed (matches phase1_chunk_threshold.py exactly, for direct comparison),
# chunk_size sweeps [1, 2, 4, 8, 16] -- goes one step further than scan_unrolled's
# ladder since the hypothesis here is specifically that this design pushes the
# compile-time wall higher. Starts at chunk_size=1 deliberately (known-working
# degenerate case, should match the plain/split_transpose reference grad exactly).
# Stops at the first TIMEOUT/CRASH.
#
# Logs to $STORE (common.py's STORE_DIR), not the repo. Writes the CSV after every
# config.
from __init__ import PRP
import sys

sys.path.append(PRP)

from common import run_worker, write_csv_incremental, STORE_DIR

N = 20
CHUNK_SIZES = [1, 2, 4, 8, 16]
PLAIN_REFERENCE_GRAD = -24.656791872202238  # plain, n=20, from phase1_chunk_threshold.py
REL_ERR_TOL = 1e-6

csv_path = f"{STORE_DIR}/phase_double_checkpoint_threshold.csv"

rows = []
prev_ok_grad = None
prev_ok_chunk = None

for chunk_size in CHUNK_SIZES:
    r = run_worker("double_checkpoint", N, chunk_size)
    rows.append(r)
    write_csv_incremental(rows, csv_path)

    if r["status"] != "OK":
        print(f"\nSTOPPING: chunk_size={chunk_size} status={r['status']} -- not trying larger chunk sizes.")
        break

    ref_rel_err = abs(r["grad"] - PLAIN_REFERENCE_GRAD) / (abs(PLAIN_REFERENCE_GRAD) + 1e-30)
    ref_status = "PASS" if ref_rel_err < REL_ERR_TOL else "FAIL"
    print(f"chunk_size={chunk_size} vs plain reference: grad={r['grad']:.6e} vs {PLAIN_REFERENCE_GRAD:.6e}  "
          f"rel_err={ref_rel_err:.3e}  {ref_status}")

    if prev_ok_grad is not None:
        rel_err = abs(r["grad"] - prev_ok_grad) / (abs(prev_ok_grad) + 1e-30)
        cmp_status = "PASS" if rel_err < REL_ERR_TOL else "FAIL"
        print(f"chunk_size={chunk_size} vs chunk_size={prev_ok_chunk}: "
              f"grad={r['grad']:.6e} vs {prev_ok_grad:.6e}  rel_err={rel_err:.3e}  {cmp_status}")

    prev_ok_grad = r["grad"]
    prev_ok_chunk = chunk_size

print(f"\nSaved {csv_path}")
print("\n--- status summary ---")
for r in rows:
    print(f"[chunk_size={r['chunk_size']}] status={r['status']}  "
          f"compile_time_s={r['compile_time_s']}  run_time_s={r['run_time_s']}  grad={r['grad']}")
