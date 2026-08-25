# %%
# Worker for run_ablation.py -- one jax.jit compile per process (same discipline as
# every other worker in this repo, see report-mld-1/section1_worker.py's docstring).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import SETUPS, spin_up, make_diff_step, set_vars, rollout, temp_agg_function

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=list(SETUPS), required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
parser.add_argument("--eps", type=float, required=True)
parser.add_argument("--mode", type=str, choices=["grad", "fd"], required=True)
args = parser.parse_args()

g4d, step_jit = spin_up(SETUPS[args.config], warmup_steps=20, desc=f"spin-up ({args.config})")
target_state = rollout(step_jit, g4d.state, args.n)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state = rollout(make_diff_step(g4d), n_state, args.n)
    return temp_agg_function(n_state, target_state)


if args.mode == "grad":
    grad_jit = jax.jit(jax.value_and_grad(loss))
    loss_val, grad = grad_jit(jnp.array(args.test_val))
    print(f"RESULT config={args.config} loss={float(loss_val)!r} grad={float(grad)!r}")
else:
    loss_jit = jax.jit(loss)
    l_plus = loss_jit(jnp.array(args.test_val) + args.eps)
    l_minus = loss_jit(jnp.array(args.test_val) - args.eps)
    num_grad = (l_plus - l_minus) / (2 * args.eps)
    print(f"RESULT config={args.config} num_grad={float(num_grad)!r}")
