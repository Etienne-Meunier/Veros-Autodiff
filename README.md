Exemple of auto-diff leveraging veros modified code for differentiability. 

Examples are in `scripts`

## Veros fork changes (`veros/` submodule)

The `veros` submodule is a fork of the original ocean model (diverges from upstream at commit `58ae1d9`). The changes below are what make it possible to differentiate through the model with JAX.

### 1. `VerosState` made into a proper JAX pytree (`veros/state.py`)
- `VerosState.copy()` now does `jax.tree_util.tree_map(lambda x: x.copy(), self)` instead of manually deep-copying settings/variables/timers field by field. The old version wasn't safe to call while tracing under `jit`/`grad`.
- New `get_tangeant(var_key)`: builds a zero-filled tangent pytree (using `jax.dtypes.float0` for int/bool leaves, per JAX convention for non-differentiable leaves) with a `+1.0` seed at one variable. This is the seed vector `jax.jvp` needs to do forward-mode differentiation w.r.t. a single state field (used by `jvp_grad` in `scripts/grad_compare.py`).
- `VerosVariables.__setattr__`: removed the forced `asarray(val, dtype=...)` cast whenever `val` wasn't already a backend array — a blind cast there would silently detach a JAX tracer from the active trace during `grad`.

### 2. Two `Settings` promoted to `Variables` — the core enabler
- `K_gm_0` (fixed GM/isopycnal diffusivity) and `r_bot` (bottom friction coefficient) moved out of `settings.py` (left there as comments) into `variables.py`.
- Why: `Settings` are baked into the jaxpr as Python constants at trace time (not differentiable). `Variables` are array leaves of the state pytree, so JAX can trace and differentiate through them. `eke.py` and `friction.py` were updated to read `vs.K_gm_0` / `vs.r_bot` instead of `settings.K_gm_0` / `settings.r_bot`.
- This is what lets the demo notebooks compute `d(loss)/d(K_gm_0)` directly.

### 3. Python branch replaced with `lax.cond` (`veros/core/external/solve_pressure.py`)
- The `if vs.itt == 0: ... else: ...` branch selecting how to write `psi` was replaced by `update_psi()` using `jax.lax.cond`. `vs.itt` is a traced value under `jit`, so a Python `if` on it either errors or silently freezes one branch forever.

### 4. Custom safe gradient for `sqrt` (`veros/core/operators.py`, used in `eke.py`, `tke.py`)
Plain sqrt has an unbounded derivative at 0:
```
d/dx sqrt(x) = 1 / (2*sqrt(x))     ->  infinite as x -> 0
```
Added `s_sqrt` via `jax.custom_jvp` with a clipped derivative:
```
d/dx s_sqrt(x) = 0.5 / max(sqrt(0.001), sqrt(x))
```
Exposed as `safe_sqrt` and substituted for `npx.sqrt` everywhere a possibly-zero non-negative field is square-rooted: `C_rossby`, `L_rossby`, `sqrteke`, `sqrttke`. Land/masked cells and cold-start zeros made plain `sqrt` produce NaN gradients — this was the main source of exploding/NaN backprop through the EKE/TKE closures.

### 5. Hard divergence check disabled (`veros/veros.py`)
- `numerics.sanity_check(state)` (raises `RuntimeError` on divergence) is commented out in the main step. A Python `raise` on a traced boolean can't be jitted, and during training a transiently-bad state should still produce a gradient rather than crash the run.

### 6. Diagnostics reset-to-array fixes (`veros/diagnostics/{averages,energy,overturning}.py`)
- Resets like `x = 0` / `x = 0.0` replaced with `rst.backend_module.asarray(0.0)`, keeping the reset value a JAX array instead of a bare Python float, avoiding dtype/pytree-structure mismatches across steps under `jit`.

### 7. Cosmetic / scaffolding (no functional effect)
- `veros/__init__.py` docstring renamed to "AutodiffVeros"; `veros/core/__init__.py` prints a "Differentiable Veros Experimental version" banner on import, to confirm this fork (not upstream) is loaded.
- Leftover debug prints and `# (routine)`/`# (kernel)` annotations in `veros.py`'s step function, plus a stray `ipdb` import in `state.py`.

### 8. `timers`/`profile_timers` instrumentation removed entirely
- `VerosState.timers`/`.profile_timers` (`veros/state.py`), every `with state.timers[...]:` block (`veros/veros.py`, `veros/core/thermodynamics.py`, `veros/core/momentum.py`), and `profile_timers` lookups in the routine dispatcher (`veros/routines.py`) are gone. `enter_routine()` no longer takes a `timer` argument.
- Why: `VerosState`'s pytree registration (`veros_state_pytree_flatten`) put these `defaultdict`s into the pytree's static `aux_data` rather than as leaves. Since the dicts gain new keys the first time each named block actually runs, the state's pytree treedef changed step to step — harmless for the ordinary Python loop, but fatal for any driver that runs Veros inside `jax.lax.scan` (e.g. vercor's differentiable scanned runtime), which requires an identical carry structure every iteration.
- **Caveat**: `veros.py`'s `step()` used to end with a `logger.debug(" Time step took {:.2f}s", state.timers["main"].last_time)` line marked `# NOTE: benchmarks parse this, do not change / remove`. That line, and the whole end-of-run timing/profile summary (`Veros._timing_summary()` / `print_profile_summary()`), are now gone. If any external benchmark tooling parses that log format, it will break — this was a deliberate tradeoff, not an oversight, but flagging it here since nothing else in the code documents that dependency.
- If you need step timings back, use an external profiler instead of re-adding this instrumentation.

## Gradient computation method (`scripts/grad_compare.py`)

Several ways of differentiating through the timestepper were tried, converging on the simplest:
- `numerical_diff` — finite differences
- `forward_diff` / `backward_diff` — `jacfwd`/`jacrev` on the whole rollout
- `jvp_grad` — manually chaining `jax.jvp` through each step (forward-mode)
- `vjp_grad` — manually chaining `jax.vjp` through each step, storing per-step vjp closures and replaying them in reverse
- `vjp_grad_scan` — same as `vjp_grad` but with `lax.scan` instead of a Python loop
- **`vjp_grad_new` — the one kept.** Just write a plain Python rollout and let `jax.value_and_grad` differentiate through it directly:
```
def loss_fn(v):
    n_state = set_var(var_name, state, v)
    for _ in range(iterations):
        n_state = step_function(n_state)     # step_function is jit + checkpoint wrapped
    return agg_function(n_state)

loss, grad = jax.value_and_grad(loss_fn)(var_value)
```
`step_function` is wrapped in `jax.jit` then `jax.checkpoint` (remat), which recomputes forward activations during the backward pass instead of storing all of them — this is what makes backprop through a multi-step rollout tractable in memory.

See `notebooks/demonstration/gradient-computation.ipynb` (scalar parameter, e.g. `K_gm_0`), `notebooks/learning/learn_field.ipynb` (gradient descent on a full field), and `notebooks/veros_functions/clean-veros-correction.ipynb` (online-trained Flax MLP correcting `temp`/`salt` to compensate for a disabled TKE scheme) for worked examples.

## Current limitations

Structural things that had to be disabled/constrained to get differentiability working, rather than patched:

- **Backend and linear solver are fixed** (`scripts/load_runtime.py`): `runtime_settings.backend = 'jax'` (the numpy backend isn't differentiable) and `runtime_settings.linear_solver = 'scipy_jax'` (the only pressure solver that's JAX-traceable — other solvers aren't).
- **Single device only**: `runtime_settings.device = 'cpu'`, no MPI / multi-GPU distributed execution.
- **`enable_streamfunction = False`**: the streamfunction barotropic-mode solve path uses different linear algebra than `solve_pressure.py` (the one patched with `lax.cond`) and was never made differentiable.
- **`K_gm_0` can no longer be set via `settings.K_gm_0`**: since it was moved from a `Setting` to a `Variable`, `settings.K_gm_0 = ...` in `set_parameter` is a no-op — it must be set directly on `state.variables.K_gm_0` after setup instead.
- **`enable_idemix = False`**: the IDEMIX internal-wave module never got the `safe_sqrt`/`lax.cond` treatment, so it's left off in every learning run.
- **All diagnostics disabled** (`set_diagnostics` reduced to `diagnostics.clear()` in `acc_learning.py`): diagnostics do file I/O and build `xarray.Dataset` objects from Python-side state, which doesn't survive `jit` tracing — cut entirely rather than patched.
- **Hard divergence check removed** (`numerics.sanity_check` in `veros.py`): a Python `raise` on a traced boolean can't be jitted, so this safety check had to go rather than be adapted.
- **Must run in float64**: `jax.config.update("jax_enable_x64", True)` is required in every notebook — JAX's default float32 isn't numerically stable enough for reverse-mode gradients through a multi-step rollout.
