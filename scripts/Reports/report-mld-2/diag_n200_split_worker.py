# %%
# Worker for diag_n200_split_gsw_stream.py -- one config (gsw-only or
# streamfunction-only) per process. Same "one compile-pair per subprocess" discipline
# as phase1_worker.py/phase2_worker.py (report-mld-1/section1_worker.py's docstring:
# several back-to-back jax.jit compiles in one long-lived process reliably stalls).
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

import argparse

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

sys.path.append(PRP)

from common import GswOnlyFullGridSetup, StreamOnlyFullGridSetup, spin_up_full_grid, make_diff_step, set_vars, rollout

SETUPS = {"gsw_only": GswOnlyFullGridSetup, "stream_only": StreamOnlyFullGridSetup}

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=list(SETUPS), required=True)
parser.add_argument("--n", type=int, required=True)
parser.add_argument("--param", type=str, required=True)
parser.add_argument("--test_val", type=float, required=True)
args = parser.parse_args()


def temp_agg_function(state, target_state):
    return ((state.variables.temp - target_state.variables.temp) ** 2).sum()


g4d, step_jit = spin_up_full_grid(SETUPS[args.config], warmup_steps=20, desc=f"spin-up ({args.config})")
target_state = rollout(step_jit, g4d.state, args.n)


def loss(v):
    n_state = set_vars(g4d.state, **{args.param: v})
    n_state = rollout(make_diff_step(g4d), n_state, args.n)
    return temp_agg_function(n_state, target_state)


loss_jit = jax.jit(loss)
grad_jit = jax.jit(jax.value_and_grad(loss))

loss_val, grad = grad_jit(jnp.array(args.test_val))
print(f"RESULT config={args.config} loss={float(loss_val)!r} grad={float(grad)!r}", flush=True)

for eps in [1e-4, 1e-6]:
    num_grad = (loss_jit(jnp.array(args.test_val) + eps) - loss_jit(jnp.array(args.test_val) - eps)) / (2 * eps)
    print(f"RESULT config={args.config} eps={eps!r} num_grad={float(num_grad)!r}", flush=True)
