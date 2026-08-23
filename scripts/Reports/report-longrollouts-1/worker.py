# %%
# Worker for scaling.py -- one jax.jit compile per process (one-jit-per-process
# discipline, also needed for peak-memory isolation). One (n, param) config per call:
# spin up global4deg, roll out a forward-only reference trajectory, time the
# double_checkpoint grad compile separately from its execution (jax.jit(...).lower(
# ...).compile() vs the actual call), read peak GPU memory.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse
import time

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import spin_up, make_diff_step, set_vars, temp_agg_function, plain_forward_rollout, rollout, peak_gpu_memory_bytes

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
parser.add_argument("--chunk_size", type=int, required=True)
args = parser.parse_args()

g4d, step_jit = spin_up(warmup_steps=20)

target_state = plain_forward_rollout(step_jit, g4d.state, args.n)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state = rollout(make_diff_step(g4d), n_state, args.n, args.chunk_size)
    return temp_agg_function(n_state, target_state)


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
    f"RESULT n={args.n} param={args.param} chunk_size={args.chunk_size} "
    f"compile_time_s={compile_time_s!r} run_time_s={run_time_s!r} grad={float(grad)!r} "
    f"peak_mem_bytes={peak_mem_str} status=OK",
    flush=True,
)
