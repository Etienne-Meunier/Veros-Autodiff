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
from functools import partial


key = random.key(0)
key, subkey = random.split(key)

import jax, time

from jax import jacfwd, jacrev, value_and_grad

class numerical_diff(autodiff) :

    def g(self, state, var_value, iterations=1, eps=1e-9, **kwargs) :

        wrap = partial(autodiff.wrapper,
                        step_fun=self.step_function,
                        var_name=self.var_name,
                        agg_func=self.agg_function,
                        iter=iterations)
        center = wrap(var_value, state)

        rgt = wrap(var_value - eps, state)
        numerical_grad =  (center - rgt)/(eps)

        return center, numerical_grad

    def __str__(self) :
        return "numerical_diff"

class forward_diff(autodiff) :

    def g(self, state, var_value, iterations=1, eps=1e-9, **kwargs) :
        wrap = partial(autodiff.wrapper,
                        step_fun=self.step_function,
                        var_name=self.var_name,
                        agg_func=self.agg_function,
                        iter=iterations)

        grad_forward = jacfwd(wrap, argnums=0)(var_value, state)
        return grad_forward#value, grad_forward

    def __str__(self) :
        return "forward_diff"


class backward_diff(autodiff) :

    def g(self, state, var_value, iterations=1, eps=1e-9, **kwargs) :
        wrap = partial(autodiff.wrapper,
                        step_fun=self.step_function,
                        var_name=self.var_name,
                        agg_func=self.agg_function,
                        iter=iterations)

        grad_forward = jacrev(wrap, argnums=0)(var_value, state)
        return grad_forward#value, grad_forward

    def __str__(self) :
        return "backward_diff"

class jvp_grad(autodiff) :

    def g(self, state, var_value, iterations=1, **kwargs) :
        n_state = autodiff.set_var(self.var_name, state, var_value)
        tangent_state = n_state.get_tangeant(self.var_name)

        for i in range(iterations) :
            next_state, tangent_state = jax.jvp(self.step_function, (n_state,), (tangent_state,))
        center, grad_jvp = jax.jvp(self.agg_function, (n_state,), (tangent_state, ))
        return center, grad_jvp

    def __str__(self) :
        return "jvp_diff"


def vjp_grad(state, function, agg_sum, var = jnp.array(1e-5, dtype=jnp.float64), iteration_grad = 10):
    next_state = state.copy()
    with next_state.variables.unlock():
        setattr(next_state.variables, 'r_bot', var)

    # Forward pass through all iterations
    current_state = next_state
    funs = []
    for i in range(iteration_grad):
        print(f'a {i}')
        current_state, vjp_fun  = jax.vjp(function, current_state)
        funs.append(vjp_fun)

    print('back')
    # Compute final output
    l, vjp_agg = jax.vjp(agg_sum, current_state)

    # Backward pass using VJP
    cotangent = jnp.ones_like(l)

    # Backpropagate through agg_sum
    ds, = vjp_agg(cotangent)

    # Backpropagate through all steps
    for vjp_fun in reversed(funs):
        ds, = vjp_fun(ds)
    return ds

if __name__ =='__main__' :
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

    methods = [vjp_grad]#[numerical_diff, backward_diff, forward_diff, jvp_grad, vjp_grad]
    for ad_method in methods :
        ad = ad_method(step_function, agg_function,  var_name)

        start_time = time.time()
        n_diff = ad.g(acc.state, var_value = var, iterations=1)
        print(f'{ad} : ', n_diff)
        n_diff_time = time.time() - start_time
        print(f'Time taken for {ad}: {n_diff_time:.4f} seconds - its={iteration_grad} \n')

# problem with step not being pure
vjp_grad(acc.state, step_function, agg_function, var = var, iteration_grad = iteration_grad)
