# %%
# Investigate why n=5's grad_jit compile stalled for 2+ hours in
# report-mld-1/section1_grad_error_vs_steps.py while n=2 took a normal ~2:40.
# Checks: (1) does the n=5 forward rollout itself contain NaN/Inf/denormal-range
# values (which can tank floating-point throughput independent of any bug), before
# (2) timing the grad_jit compile alone, printed with flush so this script's own
# stdout is useful even if it has to be killed under a bash-level timeout.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import time

import jax
import jax.numpy as jnp

sys.path.append(PRP)
sys.path.append(PRP + "scripts/Reports/report-mld-1/")
from common import spin_up_global4deg_mld, make_diff_step, set_vars, rollout, mld_agg_function


def log(msg):
    print(msg, flush=True)


g4d, step_jit = spin_up_global4deg_mld(200)
step_diff = make_diff_step(g4d)

n = 5
log(f"[n={n}] rolling out target_state (eager/jit step_jit, not the checkpoint+scan path)...")
t0 = time.time()
target_state = rollout(step_jit, g4d.state, n)
log(f"[n={n}] target_state done ({time.time() - t0:.1f}s)")

# Check every field in the state for NaN/Inf/denormal-range values -- a legitimate
# (not "stuck") explanation for a slow compile/run would be that some field's values
# have decayed into the subnormal float range by step 5, which can be 10-100x slower
# to compute on than normal floats, independent of any correctness bug.
DENORMAL_THRESHOLD = 1e-300  # true denormal range for float64
SUSPICIOUSLY_SMALL = 1e-30  # not denormal yet, but headed there / already tiny-scale

variables = target_state.variables
suspicious = []
for name in dir(variables):
    if name.startswith("_"):
        continue
    try:
        val = getattr(variables, name)
    except Exception:
        continue
    if not hasattr(val, "shape") or not hasattr(val, "dtype"):
        continue
    if val.size == 0 or not jnp.issubdtype(val.dtype, jnp.floating):
        continue
    try:
        has_nan = bool(jnp.any(jnp.isnan(val)))
        has_inf = bool(jnp.any(jnp.isinf(val)))
        abs_val = jnp.abs(val)
        nonzero = abs_val[abs_val > 0]
        min_nonzero = float(jnp.min(nonzero)) if nonzero.size > 0 else float("nan")
        n_denormal = int(jnp.sum((abs_val > 0) & (abs_val < DENORMAL_THRESHOLD)))
        n_tiny = int(jnp.sum((abs_val > 0) & (abs_val < SUSPICIOUSLY_SMALL)))
    except Exception as e:
        continue
    if has_nan or has_inf or n_denormal > 0 or n_tiny > 0:
        suspicious.append((name, has_nan, has_inf, min_nonzero, n_tiny, n_denormal))

log(f"[n={n}] fields with nan/inf/tiny/denormal values:")
for row in suspicious:
    log(f"  {row}")
if not suspicious:
    log(f"[n={n}] no nan/inf/denormal/tiny values found in any state field")

# %%
log(f"[n={n}] compiling+running grad_jit for c_k (this is the step that stalled)...")


def loss(v, target_state=target_state):
    n_state = set_vars(g4d.state, c_k=v)
    n_state = rollout(step_diff, n_state, n)
    return mld_agg_function(n_state, target_state)


t1 = time.time()
grad_jit = jax.jit(jax.value_and_grad(loss))
loss_val, grad = grad_jit(jnp.array(0.08))
log(f"[n={n}][c_k] grad_jit done ({time.time() - t1:.1f}s)  loss={float(loss_val):.6e}  grad={float(grad):.6e}")
