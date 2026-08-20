# %%
# Worker for section1_grad_error_vs_steps.py: does spin-up + rollout + exactly ONE
# jax.jit compile, then prints its result and exits.
#
# Why: a long-lived process doing several jax.jit compiles back-to-back (grad_jit for
# c_k, then loss_jit for c_k, then grad_jit for c_eps, ...) was observed to reliably
# take ~73s for the FIRST compile and then hang indefinitely (not crash, not error --
# genuinely stalled, confirmed via ps: minutes of wall clock with almost no CPU time
# accumulated) on the SECOND compile, non-deterministically across otherwise-identical
# runs. Every compile run in total isolation (its own fresh process) instead
# consistently took ~73s. Root cause not pinned down (XLA CPU-backend compilation
# cache/thread-pool state carried across compiles in one process, on this machine) --
# the practical fix is one compile per process, called out via subprocess by the
# driver script.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import spin_up_global4deg_mld, make_diff_step, set_vars, rollout, mld_agg_function

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
parser.add_argument("--eps", type=float, required=True)
parser.add_argument("--mode", type=str, choices=["grad", "fd"], required=True)
args = parser.parse_args()

g4d, step_jit = spin_up_global4deg_mld(200)
target_state = rollout(step_jit, g4d.state, args.n)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state = rollout(make_diff_step(g4d), n_state, args.n)
    return mld_agg_function(n_state, target_state)


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
