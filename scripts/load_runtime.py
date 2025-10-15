from __init__ import PRP; import sys
sys.path.append(PRP + 'veros/')

from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'force_overwrite', True)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')
setattr(runtime_settings, 'device', 'gpu')