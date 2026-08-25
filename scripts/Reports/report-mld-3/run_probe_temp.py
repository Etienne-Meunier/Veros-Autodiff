# %%
# report-mld-3, temp variant: same probe as run_probe.py (gradient accuracy vs
# rollout length n=5/20/75/250) on the identical real full config
# (GlobalFlexibleMLDLearningSetup: nz=60, ETOPO5, gsw/TEOS-10, streamfunction), but
# with temp_agg_function instead of mld_agg_function. run_probe.py found the mld-loss
# gradient breaks by n=20 on this config -- far earlier than temp on plain global4deg
# (sane to n~2000-3000, report-longrollouts-1). If temp also breaks early here, the
# real config itself (streamfunction/gsw/real topography) is the fragile ingredient;
# if temp stays sane like it does on global4deg, the mld diagnostic's own formula
# (division near weak stratification) is the more likely culprit.
from __init__ import PRP
import sys
import subprocess
import time

sys.path.append(PRP)

import matplotlib.pyplot as plt

WORKER = f"{PRP}scripts/Reports/report-mld-3/probe_worker_temp.py"

n_values = [5, 20, 75, 250]
param, test_val, eps = "c_k", 0.1082, 1e-6


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

fig.suptitle("report-mld-3 (temp variant): temp-loss gradient sanity vs n (full grid, gsw+streamfunction)")
fig.tight_layout()

import os
out_dir = f"{PRP}Results/Report/figures/report-mld-3"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/probe_grad_vs_n_temp.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
print(rows)
