# %%
# Worker for the nz=64 + mld_ma feasibility test (addendum to report-longrollouts-1,
# see report's "nz=64 / mld_ma feasibility" section). Same measurement machinery as
# worker.py (compile/run time split via .lower().compile(), peak GPU memory), but:
#   - full GlobalFlexibleMLDLearningSetup at the given --nz (not global4deg)
#   - mld_ma_agg_function loss (the real target quantity, not the temp-loss proxy)
# Feasibility-only: can double_checkpoint compute *something* at this scale on the
# real setup/loss? Correctness is explicitly out of scope -- mld_ma is already known
# broken at full-grid n=200 (see report-mld-2), so a garbage gradient here is
# expected, not a bug in this script.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse
import time

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import spin_up_mld, make_diff_step, set_vars, mld_ma_agg_function, plain_forward_rollout, rollout, peak_gpu_memory_bytes

parser = argparse.ArgumentParser()
parser.add_argument("--nz", type=int, required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
parser.add_argument("--chunk_size", type=int, required=True)
parser.add_argument("--warmup_steps", type=int, default=20)
args = parser.parse_args()

g4d, step_jit = spin_up_mld(args.nz, warmup_steps=args.warmup_steps)

target_state = plain_forward_rollout(step_jit, g4d.state, args.n)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state = rollout(make_diff_step(g4d), n_state, args.n, args.chunk_size)
    return mld_ma_agg_function(n_state, target_state)


grad_fn = jax.grad(loss)
test_val = jnp.array(args.test_val)

t0 = time.time()
compiled = jax.jit(grad_fn).lower(test_val).compile()
t1 = time.time()
grad = compiled(test_val)
jax.block_until_ready(grad)
t2 = time.time()

compile_time_s = t1 - t0
run_time_s = t2 - t1
peak_mem = peak_gpu_memory_bytes()
peak_mem_str = "None" if peak_mem is None else str(peak_mem)

print(
    f"RESULT nz={args.nz} n={args.n} param={args.param} chunk_size={args.chunk_size} "
    f"compile_time_s={compile_time_s!r} run_time_s={run_time_s!r} grad={float(grad)!r} "
    f"peak_mem_bytes={peak_mem_str} status=OK",
    flush=True,
)
