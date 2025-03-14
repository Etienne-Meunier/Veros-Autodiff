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
acc = warmup_acc(20)
acc.state._diagnostics = {}


def pure(state, step) :
    """
        Convert the state function into a "pure step" copying the input state
    """
    n_state = state.copy()
    step(n_state)  # This is a function that modifies state object inplace
    return n_state
state = acc.state
step = partial(pure, step=acc.step)
# Call function
new_state = step(state)

# JVP
s_0 = state
s_v = step(s_0)
s_vd = step(s_v)
res, gf = jax.jvp(step, (s_0,), (s_0,))

s_0
