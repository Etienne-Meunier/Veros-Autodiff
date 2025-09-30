import sys
sys.path.append('../veros/')
sys.path.append('../setups/')
from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')

from .utils import warmup_acc, autodiff

from jax import grad, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from operator import attrgetter

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

        grad = jacfwd(wrap, argnums=0)(var_value, state)
        return grad#value, grad_forward

    def __str__(self) :
        return "forward_diff"


class backward_diff(autodiff) :

    def g(self, state, var_value, iterations=1, eps=1e-9, **kwargs) :
        wrap = partial(autodiff.wrapper,
                        step_fun=self.step_function,
                        var_name=self.var_name,
                        agg_func=self.agg_function,
                        iter=iterations)

        grad = jacrev(wrap, argnums=0)(var_value, state)
        return grad#value, grad_forward

    def __str__(self) :
        return "backward_diff"

class jvp_grad(autodiff) :

    def g(self, state, var_value, iterations=1, **kwargs) :
        n_state = autodiff.set_var(self.var_name, state, var_value)
        tangent_state = n_state.get_tangeant(self.var_name)

        for i in range(iterations) :
            #print('j', i)
            n_state, tangent_state = jax.jvp(self.step_function, (n_state,), (tangent_state,))
        center, grad_jvp = jax.jvp(self.agg_function, (n_state,), (tangent_state, ))
        return center, grad_jvp

    def __str__(self) :
        return "jvp_diff"



class vjp_grad(autodiff) :

    def g(self, state, var_value, iterations=1, **kwargs) :

        n_state = autodiff.set_var(self.var_name, state, var_value)

        # Forward pass through all iterations
        current_state = n_state
        funs = []
        for i in range(iterations):
            #print(f'a {i}')
            current_state, vjp_fun  = jax.vjp(self.step_function, current_state)
            funs.append(vjp_fun)

        # Compute final output
        l, vjp_agg = jax.vjp(self.agg_function, current_state)

        # Backward pass using VJP
        cotangent = jnp.ones_like(l)

        # Backpropagate through self.agg_function
        ds, = vjp_agg(cotangent)

        # Backpropagate through all steps
        for vjp_fun in reversed(funs):
            ds, = vjp_fun(ds)
        return l, attrgetter(f'variables.{self.var_name}')(ds)

    def __str__(self) :
        return "vjp_diff"


if __name__ =='__main__' :
    acc = warmup_acc(20, override_settings={'enable_streamfunction' : False})
    acc.state._diagnostics = {}

    # Params auto-diff
    def agg_sum(state, key_sum = 'u', cv = slice(-5,-1,1)) :
        return ((getattr(state.variables, key_sum)[:,:,:] - 0)**2).mean()

    step_function = acc.step
    agg_function =agg_sum
    var_name = 'r_bot'
    var = jnp.array(1e-5, dtype=jnp.float64)
    iteration_grad = 5

    methods = [numerical_diff, jvp_grad, vjp_grad]#[numerical_diff, backward_diff, forward_diff, jvp_grad, vjp_grad]

    for ad_method in methods :
        ad = ad_method(step_function, agg_function,  var_name)

        start_time = time.time()
        n_diff = ad.g(acc.state, var_value = var, iterations=iteration_grad)
        print(f'{ad} : ', n_diff)
        n_diff_time = time.time() - start_time
        print(f'Time taken for {ad}: {n_diff_time:.4f} seconds - its={iteration_grad} \n')
