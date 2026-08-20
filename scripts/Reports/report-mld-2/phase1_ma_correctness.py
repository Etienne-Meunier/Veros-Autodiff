# %%
# report-mld-2, Phase 1: does mld_ma (exact 720-step moving average, see
# update_mld_moving_average in setups/global_4deg/global_4deg_mld_learning.py)
# differentiate correctly, on the real full setup (nz=60, ETOPO5, gsw+streamfunction)?
# mld_ma_window shrunk to `window` so the buffer fills after a handful of steps
# instead of 720 -- isolates the MA mechanism from rollout length (that's Phase 2).
# 3 checkpoints: n < window (buffer still NaN-padded, warm-up), n == window (buffer
# just filled), n > window (steady state, buffer rolling over).
from __init__ import PRP
import sys
import subprocess
import time

sys.path.append(PRP)

import matplotlib.pyplot as plt

WORKER = f"{PRP}scripts/Reports/report-mld-2/phase1_worker.py"

WINDOW = 12
n_values = [6, WINDOW, WINDOW + 4]  # warm-up / just-filled / rolling
param, test_val = "c_k", 0.08
# eps=1e-4 gives a false-positive rel_err~0.64 at n=window+4: large enough to flip
# which discrete level get_index_mld selects as the density-crossing point at some
# grid cells (branch selection is exact-zero-gradient by design, see that function's
# docstring -- finite difference straddling a branch boundary isn't comparable to the
# local derivative there). eps=1e-6 stays under that flip threshold -- see
# diag_wraparound_eps_sweep.py for the full sweep confirming this converges cleanly.
eps = 1e-6


def run_worker(n, mode):
    cmd = [
        sys.executable, WORKER,
        "--window", str(WINDOW), "--n", str(n), "--param", param,
        "--test_val", str(test_val), "--eps", str(eps), "--mode", mode,
    ]
    t0 = time.time()
    print(f"[n={n}][{mode}] launching worker...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"worker failed (n={n}, mode={mode}), see output above")
    result_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")][-1]
    print(f"[n={n}][{mode}] done ({dt:.1f}s): {result_line}", flush=True)
    return dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))


rows = []
for n in n_values:
    grad_out = run_worker(n, "grad")
    fd_out = run_worker(n, "fd")
    grad = eval(grad_out["grad"])
    num_grad = eval(fd_out["num_grad"])
    err = abs(grad - num_grad) / (abs(num_grad) + 1e-30)
    stage = "warm-up" if n < WINDOW else ("just-filled" if n == WINDOW else "rolling")
    rows.append(dict(n=n, stage=stage, loss=eval(grad_out["loss"]), grad=grad, num_grad=num_grad, rel_err=err))
    print(f"[n={n}] ({stage})  loss={rows[-1]['loss']:.6e}  autodiff={grad:.6e}  "
          f"numerical={num_grad:.6e}  rel_err={err:.4e}", flush=True)

# %%
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot([r["n"] for r in rows], [r["rel_err"] for r in rows], "o-", color="tab:blue")
ax.axvline(WINDOW, color="gray", linestyle="--", alpha=0.6, label=f"window={WINDOW}")
ax.set_yscale("log")
ax.set_xlabel("unroll steps (n)")
ax.set_ylabel("relative error (autodiff vs finite difference)")
ax.set_title(f"mld_ma gradient accuracy vs rollout length\n(full grid nz=60, gsw+streamfunction, window={WINDOW})")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

import os
out_dir = f"{PRP}Results/Report/figures/report-mld-2"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/phase1_ma_correctness.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
print(rows)
