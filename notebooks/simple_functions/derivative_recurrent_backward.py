import jax
import jax.numpy as jnp

def f(s, alpha) :
    """
    s (n) float : state
    alpah float : param
    return new_s (n) : new state
    """
    return jnp.log(s*alpha)

def g(s) :
    """
    s (n) : state
    return : reduce of state
    """
    return (s**2).mean()


### Forward step

""""
         a            a           a
         v            v           v
s_0----->F---->s_1--->F--->s_2 -> F ->s_3 -> G -> l
"""

s_0 = jnp.array([4.0, 7.0, 8.0])
alpha = jnp.array(3.0)

# Brute force way to do the derivative  : we build a loop and differentiate through it
def loop(s, a) :
    for i in range(5) :
        s = f(s, a)
    return g(s)


jax.jacrev(loop, argnums=1)(s_0, alpha)

# Using forward jvp
s = s_0.copy()
ds = jnp.zeros_like(s)
for i in range(5) :
    s, ds =  jax.jvp(f, (s, alpha), (ds, jnp.array(1.0)))
s, dl_da = jax.jvp(g, (s,), (ds,))
print('Forward JVP', dl_da)

# Using backward vjp
funs = []
s = s_0.copy()
for i in range(5) :
    s, vjp_fun  = jax.vjp(f, s, alpha)
    funs.append(vjp_fun)
l, vjp_agg = jax.vjp(g, s)


ds, = vjp_agg(1.0) # initial derivative (dg/ds)
dalphas = 0 # Gradient accumulation over alpha
for i in range(1, 6) :
    ds, dalpha = funs[-i](ds)
    dalphas += dalpha
    print(ds, dalpha)
print('Backward JVP', dalphas)
