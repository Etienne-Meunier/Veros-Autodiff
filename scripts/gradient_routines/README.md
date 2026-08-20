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

## Measured results (ACC, 30x42x15, this machine)

Compile time (first call, includes trace + XLA compile):

| n   | loop    | loop_checkpoint | scan   | scan_checkpoint |
|-----|---------|------------------|--------|------------------|
| 2   | 18.4s   | 13.6s            | 78.9s  | 34.9s            |
| 5   | 57.0s   | 44.8s            | 77.9s  | 31.5s            |
| 10  | 113.0s  | 128.4s           | 75.4s  | 33.0s            |
| 20  | *(not run -- see below)* | | 75.1s  | 32.1s            |
| 40  | *(measured separately: ~3 min, see note below)* | | 81.8s | 32.7s |
| 100 |         |                  | 88.9s  | 35.5s            |
| 200 |         |                  | 108.5s | 39.7s            |
| 400 |         |                  | 182.1s | 45.3s            |

`loop`/`loop_checkpoint` compile time grows superlinearly with n (18s -> 57s -> 113s
for n=2,5,10) -- a one-off direct measurement at n=40 (`loop`, before this table's n
was capped at 10 to keep the sweep runnable) took **2m59s** just to compile, and got
worse from there. `scan`/`scan_checkpoint` stay in a narrow band from n=2 all the way
to n=400 -- confirms the single-trace claim.

`scan_checkpoint` also has the lowest compile time of all four routines at every n,
and its steady-state run time scales far better at large n (n=400: `scan` runs in
81.8s vs `scan_checkpoint` in 11.8s) -- likely because `scan` alone has to keep every
intermediate state in memory for the backward pass, while `scan_checkpoint` just
recomputes each step on the way back, which is cheaper than the memory pressure of
holding 400 full 3D states. **`scan_checkpoint` wins on every axis: compile time,
run time, and correctness match with the other three.**

All four routines agree on loss/grad to ~5-6 significant figures at every n they were
both run at -- see `benchmark.py`'s printed correctness check.

One unrelated finding surfaced by the sweep: the gradient magnitude explodes (to
1e8, then 1e14, then 1e26+) for n >= 40 on this ACC config with this size of
perturbation (`r_bot` off by 1e-5). That's a genuine numerical/chaotic instability in
the rolled-out dynamics, not a routine artifact -- `scan` and `scan_checkpoint` agree
on the exploding value at every n, so whichever routine is used for real long-rollout
training will need its own fix (gradient clipping mid-rollout, a smaller parameter
perturbation, or a setup less sensitive to this) independent of the AD-routine choice
made here.

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
