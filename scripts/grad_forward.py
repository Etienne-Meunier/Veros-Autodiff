from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')



import sys, copy
from jax import grad, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
from acc.acc import ACCSetup
import einops
from tqdm import tqdm

key = random.key(0)
key, subkey = random.split(key)
from veros.core import friction

import jax, time
from jax.tree_util import tree_flatten


from jax import jit, jacfwd, value_and_grad
from functools import partial


from mystep import my_step

def warmup_acc(steps_bf_check = 300) :
    acc = ACCSetup()
    acc.setup()
    with acc.state.variables.unlock()  :
        acc.state.variables.r_bot += 1e-5
    with acc.state.settings.unlock() :
        acc.state.settings.enable_tke = True
        acc.state.settings.enable_neutral_diffusion = True
    for step in tqdm(range(steps_bf_check)) :
        acc.step(acc.state)
    return acc


def pure_step(state, step) :
    n_state = state.copy()
    step(n_state)  # This is a function that modifies state object inplace
    return n_state

def agg_sum(state, key_sum = 'u', cv = slice(-5,-1,1)) :
    return (getattr(state.variables, key_sum)[:,:,:].mean() - 0)**2

def wrapper_var(myvar, state, step_fun, var_name, agg_func, iter):
    n_state = state.copy()
    vs = n_state.variables
    with n_state.variables.unlock():
        setattr(vs, var_name, myvar)

    for i in range(iter) :
        n_state = step_fun(n_state)

    return agg_func(n_state)


def numerical_diff(state, function, var = jnp.array(1e-5, dtype=jnp.float64), iteration_grad = 10) :
    my_wrap = partial(wrapper_var, step_fun=function, var_name='r_bot', agg_func=agg_sum, iter=iteration_grad)
    center = my_wrap(var, state)

    rgt = my_wrap(var - 1e-9, state)
    lft = my_wrap(var + 1e-9, state)
    numerical_grad =  (lft - rgt)/(2*1e-9)

    center, numerical_grad
    return center, numerical_grad

def forward_grad(state, function, var = jnp.array(1e-5, dtype=jnp.float64), iteration_grad = 10) :
    grad_wrap_forward = jacfwd(wrapper_var, argnums=0)
    grad_rbot_var_forward = grad_wrap_forward(var, state, step_fun=function, var_name='r_bot', agg_func=agg_sum, iter=iteration_grad)
    return grad_rbot_var_forward


def jvp_grad(state, function, var = jnp.array(1e-5, dtype=jnp.float64), iteration_grad = 10) :

    next_state = state.copy()
    with next_state.variables.unlock():
        setattr(next_state.variables, 'r_bot', var)

    tangent_state = next_state.get_tangeant('r_bot')

    for i in range(iteration_grad) :
        next_state, tangent_state = jax.jvp(function, (next_state,), (tangent_state,))
        center, grad = jax.jvp(agg_sum, (next_state,), (tangent_state, ))
        print('Jvp grad :', center, grad)
    center, grad = jax.jvp(agg_sum, (next_state,), (tangent_state, ))
    return center, grad

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


    return l, ds

if __name__ =='__main__' :
    var = jnp.array(1e-5, dtype=jnp.float64)
    iteration_grad = 10
    acc = warmup_acc(300)
    acc.state._diagnostics = {}
    function = partial(pure_step, step=acc.step)
    n_diff = numerical_diff(acc.state, function, var=var, iteration_grad=iteration_grad)

    start_time = time.time()
    n_diff = numerical_diff(acc.state, function, var=var, iteration_grad=iteration_grad)
    print('Numerical diff : ', n_diff)
    n_diff_time = time.time() - start_time
    print(f'Time taken for numerical_diff: {n_diff_time:.4f} seconds')

    # Timing forward_grad
    start_time = time.time()
    frwd = forward_grad(acc.state, function, var=var, iteration_grad=iteration_grad)
    print('\nForward grad : ', frwd)
    frwd_time = time.time() - start_time
    print(f'Time taken for forward_grad: {frwd_time:.4f} seconds')


    # Timing jvp_grad
    start_time = time.time()
    jvpg = jvp_grad(acc.state, function,  var=var, iteration_grad=iteration_grad)
    print('\nJVP : ' , jvpg)
    jvp_time = time.time() - start_time
    print(f'Time taken for jvp_grad: {jvp_time:.4f} seconds')

    """
    # Timing vjp_grad
    start_time = time.time()
    vjpg = vjp_grad(acc.state, function,  var=var, iteration_grad=iteration_grad)
    print('VJP : ' , vjpg[0], vjpg[1].variables.r_bot)
    vjp_time = time.time() - start_time
    print(f'Time taken for vjp_grad: {vjp_time:.4f} seconds')
    # Not working, no idea why
    """
