# %%
# Report Section 1 (mld variant): gradient error (autodiff vs central finite
# difference) as a function of the number of unrolled Veros steps, for c_k, c_eps,
# loss = mld (mixed-layer depth) instead of temp. Same structure/scope as
# scripts/Reports/report-1/section1_grad_error_vs_steps.py, restricted to c_k/c_eps
# (the two params this report tunes -- see Results/Report/report_mld.md).
#
# Each (n, param, grad-or-fd) case runs as its own subprocess via section1_worker.py
# -- see that file's docstring for why: a single long-lived process doing several
# jax.jit compiles back-to-back was observed to stall indefinitely after the first.
from __init__ import PRP
import sys
import subprocess
import time

sys.path.append(PRP)

import matplotlib.pyplot as plt

WORKER = f"{PRP}scripts/Reports/report-mld-1/section1_worker.py"

PARAM_CONFIG = {
    "c_k": (0.08, 1e-4),
    "c_eps": (0.6, 1e-4),
}
n_values = [2, 5, 10]


def run_worker(n, param, test_val, eps, mode):
    cmd = [
        sys.executable, WORKER,
        "--n", str(n), "--param", param,
        "--test_val", str(test_val), "--eps", str(eps),
        "--mode", mode,
    ]
    t0 = time.time()
    print(f"[n={n}][{param}][{mode}] launching worker...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"worker failed (n={n}, param={param}, mode={mode}), see output above")
    result_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")][-1]
    print(f"[n={n}][{param}][{mode}] done ({dt:.1f}s): {result_line}", flush=True)
    return dict(kv.split("=", 1) for kv in result_line.removeprefix("RESULT ").split(" "))


rel_err = {name: [] for name in PARAM_CONFIG}
for n in n_values:
    for name, (test_val, eps) in PARAM_CONFIG.items():
        grad_out = run_worker(n, name, test_val, eps, "grad")
        fd_out = run_worker(n, name, test_val, eps, "fd")

        grad = eval(grad_out["grad"])
        num_grad = eval(fd_out["num_grad"])
        err = abs(grad - num_grad) / (abs(num_grad) + 1e-30)
        rel_err[name].append(err)
        print(f"[n={n}][{name}] autodiff={grad:.6e}  numerical={num_grad:.6e}  rel_err={err:.4e}", flush=True)

# %%
fig, ax = plt.subplots(figsize=(7, 5))
for name, errs in rel_err.items():
    ax.plot(n_values, errs, "o-", label=name)

ax.set_yscale("log")
ax.set_xlabel("unroll steps (n)")
ax.set_ylabel("relative error (autodiff vs finite difference)")
ax.set_title("Gradient accuracy vs rollout length (global_4deg, mld loss)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

out_path = f"{PRP}Results/Report/figures/mld-1/section1_grad_error_vs_steps.png"
fig.savefig(out_path, dpi=150)
print(f"Saved figure to {out_path}")
print(rel_err)
