# %%
# report-mld-4: ablates the real config (GlobalFlexibleMLDLearningSetup: nz=60,
# gsw/TEOS-10, streamfunction, real ETOPO5) to find which ingredient makes the
# gradient break by n=20 (report-mld-3's finding, confirmed for both mld and temp
# loss). Temp loss, param c_k, test_val=0.1082, n=20, eps=1e-6 -- same probe protocol
# as report-mld-3, applied to 4 new configs. See PLAN.md for the full rationale.
#
# Resumable by design (this is meant to run under oarsub's besteffort queue, which
# can pre-empt/kill the job at any time): each config's result is written to
# results/{config}.json the moment it's computed, and this script skips anything
# already on disk on every (re)run. A killed-and-resubmitted job picks up exactly
# where it left off instead of redoing finished configs.
from __init__ import PRP
import sys
import subprocess
import time
import json
import os

sys.path.append(PRP)

import matplotlib.pyplot as plt

WORKER = f"{PRP}scripts/Reports/report-mld-4/ablation_worker.py"
RESULTS_DIR = f"{PRP}scripts/Reports/report-mld-4/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = ["gsw_only", "stream_only", "neither", "acc_full"]
N, PARAM, TEST_VAL, EPS = 20, "c_k", 0.1082, 1e-6

# Baseline (full config): already computed in report-mld-3's temp-loss probe at
# n=20, same param/test_val/eps. Reused here, not recomputed.
BASELINE = dict(
    config="full",
    loss=12.010238263056102,
    grad=-1575.0766890999078,
    num_grad=2471.9064653000446,
)


def load(config):
    path = f"{RESULTS_DIR}/{config}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(config, data):
    path = f"{RESULTS_DIR}/{config}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def run_worker(config, mode):
    cmd = [
        sys.executable, WORKER,
        "--config", config, "--n", str(N), "--param", PARAM,
        "--test_val", str(TEST_VAL), "--eps", str(EPS), "--mode", mode,
    ]
    t0 = time.time()
    print(f"[{config}][{mode}] launching worker...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"worker failed (config={config}, mode={mode}), see output above")
    result_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")][-1]
    print(f"[{config}][{mode}] done ({dt:.1f}s): {result_line}", flush=True)
    return dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))


rows = [dict(BASELINE)]
for config in CONFIGS:
    data = load(config)

    if "grad" not in data:
        out = run_worker(config, "grad")
        data["loss"] = out["loss"]
        data["grad"] = out["grad"]
        save(config, data)
    else:
        print(f"[{config}][grad] cached, skipping", flush=True)

    if "fd" not in data:
        out = run_worker(config, "fd")
        data["num_grad"] = out["num_grad"]
        data["fd"] = "done"  # sentinel so the cache check above is unambiguous
        save(config, data)
    else:
        print(f"[{config}][fd] cached, skipping", flush=True)

    rows.append(dict(
        config=config,
        loss=eval(data["loss"]) if isinstance(data["loss"], str) else data["loss"],
        grad=eval(data["grad"]) if isinstance(data["grad"], str) else data["grad"],
        num_grad=eval(data["num_grad"]) if isinstance(data["num_grad"], str) else data["num_grad"],
    ))

for r in rows:
    r["rel_err"] = abs(r["grad"] - r["num_grad"]) / (abs(r["num_grad"]) + 1e-30)
    print(f"[{r['config']}]  loss={r['loss']:.6e}  autodiff={r['grad']:.6e}  "
          f"numerical={r['num_grad']:.6e}  rel_err={r['rel_err']:.4e}", flush=True)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
labels = [r["config"] for r in rows]
errs = [r["rel_err"] for r in rows]
colors = ["tab:gray"] + ["tab:red"] * len(CONFIGS)
ax.bar(labels, errs, color=colors)
ax.set_yscale("log")
ax.set_ylabel("relative error (autodiff vs finite difference)")
ax.set_title(f"report-mld-4: gradient-accuracy ablation at n={N} (temp loss, c_k)")
ax.grid(True, which="both", axis="y", alpha=0.3)
fig.tight_layout()

out_dir = f"{PRP}Results/Report/figures/report-mld-4"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/ablation_rel_err.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
print(rows)
