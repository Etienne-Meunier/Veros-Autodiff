import sys
from __init__ import PRP
sys.path.append(f'{PRP}/veros/')
sys.path.append(f'{PRP}/setups/')
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

def pure(state, step) :
    """
        Convert the state function into a "pure step" copying the input state
    """
    n_state = state.copy()
    step(n_state)  # This is a function that modifies state object inplace
    return n_state


def vjp_grad(state, function, iteration_grad = 10):
    # Forward pass through all iterations
    current_state = state
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

    return l, ds


acc = warmup_acc(20, override_settings={'enable_streamfunction' : False})

acc.state._diagnostics = {}

# Params auto-diff
def agg_sum(state, key_sum = 'u', cv = slice(-5,-1,1)) :
    return ((getattr(state.variables, key_sum)[:,:,:] - 0)**2).mean()

step_function = partial(pure, step=acc.step)
agg_function =agg_sum
var_name = 'r_bot'
var = jnp.array(1e-5, dtype=jnp.float64)
iteration_grad = 5

pred, grads = vjp_grad(acc.state, step_function, iteration_grad=10)
pred

grads.variables.u
import einops


import treescope
treescope.register_autovisualize_magic()
t = treescope.render_array(grads.variables.u)
t

with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
  contents = treescope.render_to_html(grads.variables.u[..., 0, 0])

with open("/tmp/treescope_output.html", "w") as f:
  f.write(contents)
grads.variables.u.shape

plt.imshow(grads.variables.u[..., 0, 0] == np.nan)
grads.variables.u[..., 0, 0].min()
grads.variables.u[..., 0, 0] == np.nan
t =  grads.variables.u
v = grads.variables


from jax.tree import flatten

dd = flatten(grads.variables)[0]

dd[0]
%store  dd
grads.variables.u
