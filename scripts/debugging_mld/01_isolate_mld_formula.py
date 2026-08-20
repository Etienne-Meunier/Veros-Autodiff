# %%
# Regression check: gradient of the MLD formula (global_4deg_mld_learning.py's
# after_timestep, ported out as a bare function) w.r.t. its own input, prho, in total
# isolation from the rest of the model. Directional-derivative check (autodiff vs
# central finite difference along a random direction) since prho is a full 3D field --
# same idea as scripts/debugging_ceps/03_test_safe_sqrt_clip.py's pointwise check, just
# batched via a random direction instead of looping every element.
from __init__ import PRP
import sys

sys.path.append(PRP + "veros/")

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(PRP)

from scripts.load_runtime import *  # noqa: F401,F403 -- sets jax backend before veros.core imports
from setups.global_4deg.global_4deg_mld_learning import GlobalFlexibleMLDLearningSetup, mld_from_prho


def build_setup():
    g4d = GlobalFlexibleMLDLearningSetup()
    g4d.setup()
    g4d.step(g4d.state)  # one eager step so prho is populated (see setup() smoke test)
    return g4d


def loss_fn(prho, maskT, zt, reference_depth):
    mld = mld_from_prho(prho, maskT, zt, reference_depth)
    valid = ~jnp.isnan(mld)
    return jnp.where(valid, mld, 0.0).sum() ** 2  # scalar, only well-defined (non-NaN) cells contribute


g4d = build_setup()
prho0 = g4d.state.variables.prho
maskT = g4d.state.variables.maskT
zt = g4d.state.variables.zt
reference_depth = g4d.mld_reference_depth

grad_fn = jax.jit(jax.grad(lambda p: loss_fn(p, maskT, zt, reference_depth)))
loss_jit = jax.jit(lambda p: loss_fn(p, maskT, zt, reference_depth))

grad0 = grad_fn(prho0)
print("nan in grad:", bool(jnp.any(jnp.isnan(grad0))), " nan in prho0:", bool(jnp.any(jnp.isnan(prho0))))

rng = np.random.default_rng(0)
direction = jnp.array(rng.standard_normal(prho0.shape))
direction = direction / jnp.linalg.norm(direction)

for eps in [1e-3, 1e-4, 1e-5]:
    l_plus = loss_jit(prho0 + eps * direction)
    l_minus = loss_jit(prho0 - eps * direction)
    num_dir_deriv = (l_plus - l_minus) / (2 * eps)
    auto_dir_deriv = jnp.sum(grad0 * direction)
    rel_err = abs(float(auto_dir_deriv) - float(num_dir_deriv)) / (abs(float(num_dir_deriv)) + 1e-30)
    print(
        f"eps={eps:.0e}  autodiff_dir_deriv={float(auto_dir_deriv):.6e}  "
        f"numerical={float(num_dir_deriv):.6e}  rel_err={rel_err:.4e}"
    )
