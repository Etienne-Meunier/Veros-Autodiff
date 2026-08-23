# %%
# Worker for sweep.py -- one jax.jit compile per process (established discipline in
# this repo, see report-mld-1/section1_worker.py's docstring), also needed here so a
# host-RAM-OOM SIGKILL in one config can't take down the whole sweep. One
# (strategy, nz, n, chunk_size) config per call. Compile time and run time are split
# via jax.jit(...).lower(...).compile() (AOT) instead of timing the first call --
# that's the only way to separate "XLA is still building the graph" from "the GPU is
# running it", which is exactly the distinction this whole investigation hinges on
# (see common.py's module docstring: the failure so far has been at compile time,
# GPU memory untouched).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse
import time

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import spin_up, spin_up_global4deg, make_diff_step, set_vars, temp_agg_function, plain_forward_rollout, peak_gpu_memory_bytes, STRATEGIES

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", choices=list(STRATEGIES), required=True)
parser.add_argument("--setup", choices=["flexible_nz", "global4deg"], default="flexible_nz",
                     help="flexible_nz: GlobalFlexibleMLDLearningSetup with --nz override (ETOPO5, gsw+streamfunction). "
                          "global4deg: report-1's GlobalFourDegreeSetup, native nz=15, no gsw/streamfunction -- isolation baseline.")
parser.add_argument("--nz", type=int, default=15, help="ignored for --setup global4deg (fixed at nz=15)")
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--chunk_size", type=int, required=True)
parser.add_argument("--param", type=str, default="c_k")
parser.add_argument("--test_val", type=float, default=0.08)
args = parser.parse_args()

rollout_fn = STRATEGIES[args.strategy]

if args.setup == "global4deg":
    g4d, step_jit = spin_up_global4deg(warmup_steps=20)
else:
    g4d, step_jit = spin_up(args.nz, warmup_steps=20)

# Forward-only target state -- no checkpoint needed (see common.py), same for every
# strategy so the sweep only measures the grad-path's compile/run cost.
target_state = plain_forward_rollout(step_jit, g4d.state, args.n)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state = rollout_fn(make_diff_step(g4d), n_state, args.n, args.chunk_size)
    return temp_agg_function(n_state, target_state)


grad_fn = jax.grad(loss)
test_val = jnp.array(args.test_val)

t0 = time.time()
lowered = jax.jit(grad_fn).lower(test_val)
compiled = lowered.compile()
t1 = time.time()
grad = compiled(test_val)
jax.block_until_ready(grad)
t2 = time.time()

compile_time_s = t1 - t0
run_time_s = t2 - t1
peak_mem = peak_gpu_memory_bytes()
peak_mem_str = "None" if peak_mem is None else str(peak_mem)

print(
    f"RESULT strategy={args.strategy} setup={args.setup} nz={args.nz} n={args.n} chunk_size={args.chunk_size} "
    f"compile_time_s={compile_time_s!r} run_time_s={run_time_s!r} grad={float(grad)!r} "
    f"peak_mem_bytes={peak_mem_str} status=OK",
    flush=True,
)
