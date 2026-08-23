from __init__ import PRP; import sys, os
sys.path.append(PRP + 'veros/')

from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')
# VEROS_DEVICE lets a GPU run (e.g. on g5k, see README's server micro-guide) override
# the cpu default without editing this file -- a remote-only edit gets clobbered by
# the next `g5k sync code` since local is the sync's source of truth.
setattr(runtime_settings, 'device', os.environ.get('VEROS_DEVICE', 'cpu'))