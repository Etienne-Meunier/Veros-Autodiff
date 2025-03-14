import sys
sys.path.append('../veros/')
sys.path.append('../setups/')
from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')

from utils import warmup_acc, autodiff

from jax import grad, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import abc

from acc.acc import ACCSetup

key = random.key(0)
key, subkey = random.split(key)

import jax, time

from jax import jacfwd, jacrev, value_and_grad
from functools import partial


def vjp_grad(state, function, var = jnp.array(1e-5, dtype=jnp.float64), iteration_grad = 10):
    next_state = state.copy()
    with next_state.variables.unlock():
        setattr(next_state.variables, 'r_bot', var)

    # Forward pass through all iterations
    current_state = next_state
    funs = []
    for i in range(iteration_grad):
        current_state, vjp_fun  = jax.vjp(function, current_state)
        funs.append(vjp_fun)

    # Compute final output
    l, vjp_agg = jax.vjp(agg_sum, current_state)

    # Backward pass using VJP
    cotangent = jnp.ones_like(l)

    # Backpropagate through agg_sum
    ds, = vjp_agg(cotangent)

    # Backpropagate through all steps
    for vjp_fun in reversed(funs):
        ds, = vjp_fun(ds)




acc = warmup_acc(20)
acc.state._diagnostics = {}

# Params auto-diff
def agg_sum(state, key_sum = 'u', cv = slice(-5,-1,1)) :
    return (getattr(state.variables, key_sum)[:,:,:].mean() - 0)**2

step_function = acc.step
agg_function =agg_sum
var_name = 'r_bot'
var = jnp.array(1e-5, dtype=jnp.float64)
iteration_grad = 5

from functools import partial

pure_step = partial(autodiff, step=acc.step)
next_state = pure_step(acc.state)

s, vjp_fun  = jax.vjp(pure_step, acc.state)
vjp_fun(acc.state)
