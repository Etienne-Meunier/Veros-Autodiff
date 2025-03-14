import jax.numpy as jnp
import jax
from jax import random

from multiprocessing.dummy import Array
def f(x) :
    return (x**2)
key = random.key(0)
k1,k2,k3 = random.split(key, 3)
x_0 =  random.normal(k1, (10))
v = random.normal(k2, (10))
vd = random.normal(k3, (10))

res, gf = jax.jvp(f, (x_0,), (vd,))
print(v@gf)

res_2, grad_fun = jax.vjp(f, x_0)
gb, = grad_fun(v)
print(gb@vd)


import equinox as eqx

class State(eqx.Module):
    a : jax.Array
    b : jax.Array

    def __init__(self, a, b) :
        self.a = a
        self.b = b

    def __matmul__(self, other):
        products = jax.tree_util.tree_map(lambda x, y: jnp.sum(x * y), self, other)
        return jax.tree_util.tree_reduce(lambda x, y : x+y, products)

def f(state) :
    """
    s (n) float : state
    alpah float : param
    return new_s (n) : new state
    """
    new_s = jnp.log(state.a * state.b)
    return eqx.tree_at(lambda s : s.a, state, new_s)

key = random.key(0)
k1,k2,k3 = random.split(key, 3)
random.normal(k1, (2))
s_0 = State(*random.normal(k1, (2)))
s_v = State(*random.normal(k2, (2)))
s_vd = State(*random.normal(k3, (2)))
res, gf = jax.jvp(f, (s_0,), (s_vd,))

print(s_v@gf)
res, gf = jax.jvp(f, (s_0,), (s_vd,))
print(s_v@gf)
