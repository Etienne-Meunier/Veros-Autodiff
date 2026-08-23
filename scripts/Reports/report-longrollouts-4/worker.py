# %%
# Worker for calibrate_n.py -- one jax.jit compile per process. One (n, param)
# config per call: spin up nz=64 GlobalFlexibleMLDLearningSetup (built-in mld_ma
# tracking disabled -- see common.py), roll out a forward-only reference
# trajectory's 1yr trailing mld_ma, time the double_checkpoint grad compile
# separately from its execution, read peak GPU memory.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse
import time

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import (
    spin_up_mld, make_diff_step, set_vars, mld_ma_agg_function,
    plain_forward_rollout_mld_ma, rollout_mld_ma, peak_gpu_memory_bytes, MLD_MA_WINDOW,
)

parser = argparse.ArgumentParser()
parser.add_argument("--nz", type=int, required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
parser.add_argument("--lead_chunk_size", type=int, required=True)
parser.add_argument("--tail_chunk_size", type=int, required=True)
parser.add_argument("--window", type=int, default=MLD_MA_WINDOW)
parser.add_argument("--warmup_steps", type=int, default=20)
args = parser.parse_args()

g4d, step_jit = spin_up_mld(args.nz, warmup_steps=args.warmup_steps)

_, target_mld_ma = plain_forward_rollout_mld_ma(step_jit, g4d.state, args.n, args.window)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state, mld_ma = rollout_mld_ma(make_diff_step(g4d), n_state, args.n, args.lead_chunk_size, args.tail_chunk_size, args.window)
    return mld_ma_agg_function(mld_ma, target_mld_ma)


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
    f"RESULT nz={args.nz} n={args.n} param={args.param} lead_chunk_size={args.lead_chunk_size} "
    f"tail_chunk_size={args.tail_chunk_size} window={args.window} "
    f"compile_time_s={compile_time_s!r} run_time_s={run_time_s!r} grad={float(grad)!r} "
    f"peak_mem_bytes={peak_mem_str} status=OK",
    flush=True,
)
