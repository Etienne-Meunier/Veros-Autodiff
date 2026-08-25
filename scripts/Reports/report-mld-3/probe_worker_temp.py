# %%
# Worker for run_probe_temp.py -- same as probe_worker.py but with temp_agg_function
# instead of mld_agg_function, to isolate whether run_probe.py's early gradient
# breakdown (n=20) comes from the mld diagnostic's own formula or from the real
# config (nz=60, gsw/TEOS-10, streamfunction, real ETOPO5) itself. One jax.jit
# compile per process, same reasoning as probe_worker.py.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import spin_up_full_grid, make_diff_step, set_vars, rollout, temp_agg_function

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
parser.add_argument("--eps", type=float, required=True)
parser.add_argument("--mode", type=str, choices=["grad", "fd"], required=True)
args = parser.parse_args()

g4d, step_jit = spin_up_full_grid(warmup_steps=20)
target_state = rollout(step_jit, g4d.state, args.n)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state = rollout(make_diff_step(g4d), n_state, args.n)
    return temp_agg_function(n_state, target_state)


if args.mode == "grad":
    grad_jit = jax.jit(jax.value_and_grad(loss))
    loss_val, grad = grad_jit(jnp.array(args.test_val))
    print(f"RESULT loss={float(loss_val)!r} grad={float(grad)!r}")
else:
    loss_jit = jax.jit(loss)
    l_plus = loss_jit(jnp.array(args.test_val) + args.eps)
    l_minus = loss_jit(jnp.array(args.test_val) - args.eps)
    num_grad = (l_plus - l_minus) / (2 * args.eps)
    print(f"RESULT num_grad={float(num_grad)!r}")
