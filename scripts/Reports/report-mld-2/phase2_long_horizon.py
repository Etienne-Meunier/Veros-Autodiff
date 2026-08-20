# %%
# report-mld-2, Phase 2: does the gradient survive a long rollout through gsw +
# streamfunction (both re-enabled in global_4deg_mld_learning.py, see its docstring)?
# Mini grid (nz=15, cheap) so a several-hundred-step rollout is tractable; direct `mld`
# loss (not mld_ma) -- isolates rollout-length/chaos risk from the MA mechanism
# (Phase 1 already covers that separately). Target: this trend should stay flat/small,
# not blow up, as n approaches the ~1800-step (5y daily) scale the real report needs.
from __init__ import PRP
import sys
import subprocess
import time

sys.path.append(PRP)

import matplotlib.pyplot as plt

WORKER = f"{PRP}scripts/Reports/report-mld-2/phase2_worker.py"

n_values = [100, 400, 900]
param, test_val, eps = "c_k", 0.08, 1e-4


def run_worker(n, mode):
    cmd = [
        sys.executable, WORKER,
        "--n", str(n), "--param", param,
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
    rows.append(dict(n=n, loss=eval(grad_out["loss"]), grad=grad, num_grad=num_grad, rel_err=err))
    print(f"[n={n}]  loss={rows[-1]['loss']:.6e}  autodiff={grad:.6e}  "
          f"numerical={num_grad:.6e}  rel_err={err:.4e}", flush=True)

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].plot([r["n"] for r in rows], [r["rel_err"] for r in rows], "o-", color="tab:red")
axs[0].set_yscale("log")
axs[0].set_xlabel("unroll steps (n)")
axs[0].set_ylabel("relative error (autodiff vs finite difference)")
axs[0].set_title("gradient accuracy vs rollout length")
axs[0].grid(True, which="both", alpha=0.3)

axs[1].plot([r["n"] for r in rows], [abs(r["grad"]) for r in rows], "o-", color="tab:purple")
axs[1].set_yscale("log")
axs[1].set_xlabel("unroll steps (n)")
axs[1].set_ylabel("|dloss/dc_k|")
axs[1].set_title("gradient magnitude vs rollout length")
axs[1].grid(True, which="both", alpha=0.3)

fig.suptitle("Phase 2: long-horizon gradient sanity (mini grid nz=15, gsw+streamfunction on)")
fig.tight_layout()

import os
out_dir = f"{PRP}Results/Report/figures/report-mld-2"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/phase2_long_horizon.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
print(rows)
