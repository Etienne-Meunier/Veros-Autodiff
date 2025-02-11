import sys
sys.path.append('../veros/')
sys.path.append('../setups/')
from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')

from jax import grad, random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import abc


from acc.acc import ACCSetup
from tqdm import tqdm

key = random.key(0)
key, subkey = random.split(key)

import jax, time

from jax import jacfwd, value_and_grad
from functools import partial


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

class autodiff() :
    def __init__(self, step_function, agg_function,  var_name) :
        """
            Computes derivative dL/dvar with L in R and var in R
            step_function is the function done n iterations
            agg_function is computed at the end to go from R^space -> R
            var_name : name of the variable in the state to differentiatiat w.r.t
        """
        self.agg_function = agg_function
        self.step_function = partial(autodiff.pure, step=step_function)
        self.var_name = var_name

    @staticmethod
    def pure(state, step) :
        """
            Convert the state function into a "pure step" copying the input state
        """
        n_state = state.copy()
        step(n_state)  # This is a function that modifies state object inplace
        return n_state

    @staticmethod
    def set_var(var_name, state, var_value):
        n_state = state.copy()
        vs = n_state.variables
        with n_state.variables.unlock():
            setattr(vs, var_name, var_value)
        return n_state


    @staticmethod
    def wrapper(var_value, state, step_fun, var_name, agg_func, iter):
        n_state = numerical_diff.set_var(var_name, state, var_value)

        for i in range(iter) :
            n_state = step_fun(n_state)

        return agg_func(n_state)


    @abc.abstractmethod
    def g(self, state, var_value, iterations=1, **kwargs) :
        """
            var_value : evaluation value for variable
            iterations : number of time to execute step_function
        Returns :
            output : agg_function([step_function(state)]*it) in R
            grad : (d ouput / d var_name |_var_value) in R
        """
        pass


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

class jvp_diff(autodiff) :

    def g(self, state, var_value, iterations=1, **kwargs) :
        n_state = numerical_diff.set_var(self.var_name, state, var_value)
        tangent_state = n_state.get_tangeant(self.var_name)

        for i in range(iterations) :
            next_state, tangent_state = jax.jvp(self.step_function, (n_state,), (tangent_state,))
        center, grad_jvp = jax.jvp(self.agg_function, (n_state,), (tangent_state, ))
        return center, grad_jvp

    def __str__(self) :
        return "jvp_diff"



if __name__ =='__main__' :
    acc = warmup_acc(300)
    acc.state._diagnostics = {}

    # Params auto-diff
    def agg_sum(state, key_sum = 'u', cv = slice(-5,-1,1)) :
        return (getattr(state.variables, key_sum)[:,:,:].mean() - 0)**2

    step_function = acc.step
    agg_function =agg_sum
    var_name = 'r_bot'
    var = jnp.array(1e-5, dtype=jnp.float64)
    iteration_grad = 100

    methods = [numerical_diff, forward_diff, jvp_diff]
    for ad_method in methods :
        ad = ad_method(step_function, agg_function,  var_name)

        start_time = time.time()
        n_diff = ad.g(acc.state, var_value = var, iterations=1)
        print(f'{ad} : ', n_diff)
        n_diff_time = time.time() - start_time
        print(f'Time taken for {ad}: {n_diff_time:.4f} seconds - its={iteration_grad}')
