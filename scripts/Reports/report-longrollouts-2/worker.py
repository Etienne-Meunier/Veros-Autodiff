# %%
# Worker for scaling.py -- one jax.jit compile per process. One (n, param) config
# per call: spin up global4deg, roll out a forward-only reference trajectory
# (state + trailing 365-step temp_ma), time the double_checkpoint grad compile
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
    spin_up, make_diff_step, set_vars, temp_ma_agg_function,
    plain_forward_rollout_temp_ma, rollout_temp_ma, peak_gpu_memory_bytes, TEMP_MA_WINDOW,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
parser.add_argument("--lead_chunk_size", type=int, required=True)
parser.add_argument("--tail_chunk_size", type=int, required=True)
parser.add_argument("--window", type=int, default=TEMP_MA_WINDOW)
args = parser.parse_args()

g4d, step_jit = spin_up(warmup_steps=20)

_, target_temp_ma = plain_forward_rollout_temp_ma(step_jit, g4d.state, args.n, args.window)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state, temp_ma = rollout_temp_ma(make_diff_step(g4d), n_state, args.n, args.lead_chunk_size, args.tail_chunk_size, args.window)
    return temp_ma_agg_function(temp_ma, target_temp_ma)


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
    f"RESULT n={args.n} param={args.param} lead_chunk_size={args.lead_chunk_size} "
    f"tail_chunk_size={args.tail_chunk_size} window={args.window} "
    f"compile_time_s={compile_time_s!r} run_time_s={run_time_s!r} grad={float(grad)!r} "
    f"peak_mem_bytes={peak_mem_str} status=OK",
    flush=True,
)
