# Gradient routines

Problem: `scripts/report/common.py` (and most of `notebooks/test/`) rolls out the
state with a plain python `for` loop and jits the whole thing:

```python
def rollout(step_fn, state, iterations):
    for _ in range(iterations):
        state = step_fn(state)
    return state
```

JAX traces this loop by unrolling it `iterations` times into one XLA program. Compile
time (and the size of the program XLA has to optimize/schedule) grows with rollout
length, so this gets very slow -- sometimes minutes -- once rollouts reach the hundreds
of steps needed for real gradient-descent training.

## Fix: `jax.lax.scan`

Swap the python loop for `jax.lax.scan`:

```python
def rollout(step_fn, state, iterations):
    state, _ = jax.lax.scan(lambda c, _: (step_fn(c), None), state, length=iterations)
    return state
```

`scan` traces `step_fn` **once** and reuses it `iterations` times inside XLA --
`iterations` only sets the (static) scan length. Compile time stays ~flat as rollout
length grows. This was already prototyped in
`notebooks/demonstration/gradient-computation-scan.ipynb` (the `autodiff.rollout`
method) -- that's the "already found" result. This folder turns it into a proper,
runnable comparison.

Combine with `jax.checkpoint` (remat) on the per-step function so the backward pass
doesn't need to keep every intermediate step in memory:

```python
step_fn = jax.checkpoint(step_fn)
```

`scan` + `checkpoint` is the routine to use for long rollouts: flat compile time,
bounded backward-pass memory, and it's a two-line change from the existing loop code.

## Files

- `common.py` -- ACC spin-up (small 30x42x15 grid, fast enough for hundreds of steps)
  and step/var helpers.
- `routines.py` -- four routines, same signature, only the rollout tracing differs:
  `loop`, `loop_checkpoint` (today's approach), `scan`, `scan_checkpoint`
  (recommended).
- `benchmark.py` -- times compile + steady-state run for each routine across rollout
  lengths (short for the `loop*` routines, up to 400 for the `scan*` routines), checks
  all four agree on loss/grad at matching `n`, saves `Results/gradient_routines_benchmark.csv`.

## Usage

```bash
python scripts/gradient_routines/benchmark.py
```

To use `scan_checkpoint` in an actual training loop:

```python
from scripts.gradient_routines.routines import make_scan_checkpoint_grad

grad_fn = make_scan_checkpoint_grad(step_fn, agg_fn, "c_k", iterations=300)
loss, grad = grad_fn(var_value, state)
```

## Note for very long rollouts (1000+ steps)

`scan_checkpoint` remats every single step, so the backward pass is O(iterations) in
recomputation cost. If that becomes the bottleneck, the standard next step is
checkpointing every k steps (nested scan: outer scan over chunks, each chunk
checkpointed as a whole) instead of every step -- not implemented here since
`scan_checkpoint` alone should comfortably cover "several hundred" steps.
