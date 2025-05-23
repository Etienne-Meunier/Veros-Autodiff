import sys

sys.path.append('../../veros/')
sys.path.append('../../setups/')
sys.path.append('../../scripts/')

from veros import runtime_settings

setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')

from utils import warmup_acc
from functools import partial
from jax import random
import jax
import jax.numpy as jnp


def pure(state, step) :
    """
        Convert the state function into a "pure step" copying the input state
    """
    n_state = state.copy()
    step(n_state)  # This is a function that modifies state object inplace
    return n_state

def simple(state) :
    with state.variables.unlock() :
        state.variables.u +=  1e5 * state.variables.r_bot**2




%load_ext autoreload
%autoreload 2
from my_step import my_step

from my_step import my_linear_bottom_friction
from veros.core.external import solve_streamfunction

def lbf(state) :
    with state.variables.unlock() :
        state.variables.update(my_linear_bottom_friction(state))
        solve_streamfunction(state)


acc = warmup_acc(20, override_settings={'enable_streamfunction' : False})
acc.state._diagnostics = {}


state = acc.state
step = partial(pure, step=acc.step)#acc.step) # step = acc.step


def agg_sum(state, key_sum = 'u', cv = slice(-5,-1,1)) :
    return (getattr(state.variables, key_sum)[:,:,:].mean() - 0)**2

def agg_kernel(ret, key_sum = 'K_diss_bot') :
    return ((getattr(ret, key_sum)[:,:,:] - 0)**2).mean()

# Finite diff
s_l = state.copy()
ss = 1e-5
with s_l.variables.unlock() :
    s_l.variables.r_bot += ss

s_r =  state.copy()
with s_r.variables.unlock() :
    s_r.variables.r_bot -= ss
print('Finite diff :', (agg_sum(step(s_l)) - agg_sum(step(s_r))) / ( 2*ss ))


# JVP
s_0 = state.copy()
tangent_state = s_0.get_tangeant('r_bot')
next_state, tangent_state = jax.jvp(step, (s_0,), (tangent_state,))
center, grad_jvp = jax.jvp(agg_sum, (next_state,), (tangent_state, ))
print('JVP :', grad_jvp)

# VJP
s_0 = state.copy()
new_state, grad_fun_a = jax.vjp(step, s_0)
l, grad_fun_g = jax.vjp(agg_sum, new_state)
grad_g,  = grad_fun_g(1.0)
grad_a, = grad_fun_a(grad_g)
print('VJP :', grad_a.variables.r_bot)
