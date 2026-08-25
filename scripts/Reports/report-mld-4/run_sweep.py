# %%
# report-mld-4, rollout-length sweep: report-mld-4's n=20 ablation found gsw_only
# degraded (rel_err 0.43), stream_only/neither sane (~0.01), acc_full clean (8e-7),
# vs. full's broken 1.64 -- all at a single n=20 snapshot. This sweeps n=5/20/75/250
# (report-mld-3's own grid, for direct comparability with full's existing sweep) for
# the 4 non-full configs, to find each one's actual breakdown point rather than just
# its n=20 snapshot. Same temp-loss probe protocol throughout (param c_k,
# test_val=0.1082, eps=1e-6).
#
# Resumable by design (besteffort queue): each (config, n) pair's result is written
# to results_sweep/{config}_n{n}.json the moment it's computed, and this script skips
# anything already on disk on every (re)run.
from __init__ import PRP
import sys
import subprocess
import time
import json
import os

sys.path.append(PRP)

import matplotlib.pyplot as plt

WORKER = f"{PRP}scripts/Reports/report-mld-4/ablation_worker.py"
RESULTS_DIR = f"{PRP}scripts/Reports/report-mld-4/results_sweep"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = ["gsw_only", "stream_only", "neither", "acc_full"]
N_VALUES = [5, 20, 75, 250]
PARAM, TEST_VAL, EPS = "c_k", 0.1082, 1e-6

# full config's own sweep (report-mld-3, temp variant + n=20 ablation baseline) --
# reused as a reference line, not recomputed. n=75/250 for "full" were never run
# (report-mld-3's temp variant was stopped after n=20 confirmed the break) -- only
# n=5/20 are real; report-mld-4's mld-loss n=75/250 numbers are NOT substitutable
# (different loss), so those two points are left out of the reference line.
FULL_REFERENCE = {5: 2.4219e-07, 20: 1.6372}


def load(config, n):
    path = f"{RESULTS_DIR}/{config}_n{n}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(config, n, data):
    path = f"{RESULTS_DIR}/{config}_n{n}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def run_worker(config, n, mode):
    cmd = [
        sys.executable, WORKER,
        "--config", config, "--n", str(n), "--param", PARAM,
        "--test_val", str(TEST_VAL), "--eps", str(EPS), "--mode", mode,
    ]
    t0 = time.time()
    print(f"[{config}][n={n}][{mode}] launching worker...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"worker failed (config={config}, n={n}, mode={mode}), see output above")
    result_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")][-1]
    print(f"[{config}][n={n}][{mode}] done ({dt:.1f}s): {result_line}", flush=True)
    return dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))


rows = []
for config in CONFIGS:
    for n in N_VALUES:
        data = load(config, n)

        if "grad" not in data:
            out = run_worker(config, n, "grad")
            data["loss"] = out["loss"]
            data["grad"] = out["grad"]
            save(config, n, data)
        else:
            print(f"[{config}][n={n}][grad] cached, skipping", flush=True)

        if "fd" not in data:
            out = run_worker(config, n, "fd")
            data["num_grad"] = out["num_grad"]
            data["fd"] = "done"
            save(config, n, data)
        else:
            print(f"[{config}][n={n}][fd] cached, skipping", flush=True)

        rows.append(dict(
            config=config, n=n,
            loss=float(data["loss"]),
            grad=float(data["grad"]),
            num_grad=float(data["num_grad"]),
        ))

for r in rows:
    r["rel_err"] = abs(r["grad"] - r["num_grad"]) / (abs(r["num_grad"]) + 1e-30)
    print(f"[{r['config']}][n={r['n']}]  loss={r['loss']:.6e}  autodiff={r['grad']:.6e}  "
          f"numerical={r['num_grad']:.6e}  rel_err={r['rel_err']:.4e}", flush=True)

# %%
fig, ax = plt.subplots(figsize=(8, 5.5))
for config in CONFIGS:
    xs = [r["n"] for r in rows if r["config"] == config]
    ys = [r["rel_err"] for r in rows if r["config"] == config]
    ax.plot(xs, ys, "o-", label=config)

ax.plot(list(FULL_REFERENCE), list(FULL_REFERENCE.values()), "k--", marker="s", label="full (reference, n<=20 only)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("unroll steps (n)")
ax.set_ylabel("relative error (autodiff vs finite difference)")
ax.set_title("report-mld-4 sweep: gradient accuracy vs n, per ablation config (temp loss, c_k)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

out_dir = f"{PRP}Results/Report/figures/report-mld-4"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/sweep_rel_err_vs_n.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
print(rows)
