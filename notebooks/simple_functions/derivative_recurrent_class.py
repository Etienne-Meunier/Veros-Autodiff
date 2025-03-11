import jax
import jax.numpy as jnp
import equinox as eqx


class State(eqx.Module) :
    s : jax.Array
    alpha : jax.Array

    def __init__(self, s_0, alpha) :
        self.s = s_0
        self.alpha = alpha

def f(state) :
    """
    s (n) float : state
    alpah float : param
    return new_s (n) : new state
    """
    new_s = jnp.log(state.s * state.alpha)
    return eqx.tree_at(lambda s : s.s, state, new_s)

def g(state) :
    """
    s (n) : state
    return : reduce of state
    """
    return (state.s**2).mean()

### Forward step

""""
         a            a           a
         v            v           v
s_0----->F---->s_1--->F--->s_2 -> F ->s_3 -> G -> l
"""


s_0 = jnp.array([4.0, 7.0, 8.0])
alpha = jnp.array(3.0)

state = State(s_0=s_0, alpha=alpha)

# Brute force way to do the derivative  : we build a loop and differentiate through it
def loop(s) :
    for i in range(5) :
        s = f(s)
    return g(s)

state_grad = jax.jacrev(loop)(state)
state_grad.alpha

# Using forward jvp
s = State(s_0=s_0, alpha=alpha)
ds = State(s_0=jnp.zeros_like(s.s), alpha=jnp.ones_like(s.alpha))
for i in range(5) :
    s, ds =  jax.jvp(f, (s,), (ds,))
s, dl_da = jax.jvp(g, (s,), (ds,))
print('Forward JVP', dl_da)
# Using backward vjp
funs = []
s = State(s_0=s_0, alpha=alpha)
for i in range(5) :
    s, vjp_fun  = jax.vjp(f, s)
    funs.append(vjp_fun)
l, vjp_agg = jax.vjp(g, s)

ds, = vjp_agg(jnp.ones_like(l)) # initial derivative (dg/ds)
for i in range(1, 6) :
    ds, = funs[-i](ds)
print(f'Backward JVP {ds.alpha=}  {ds.s=}')
