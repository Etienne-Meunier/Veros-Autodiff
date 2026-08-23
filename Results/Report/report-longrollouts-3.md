# Gradient Descent Parameter Tuning at n=2000

Report-longrollouts-1 found `dloss/d{c_k,c_eps}` (raw temp-squared-error,
single-final-snapshot loss, `global4deg`) sane and smoothly growing through
n=2000 (`grad=-385.2` for `c_k` there), with the first sign of trouble only at
n=3000. Question: in that regime, do the gradients actually work as an
optimization signal -- does Adam gradient descent recover the true parameters
from a perturbed guess?

## Method

Same setup and `double_checkpoint` rollout as report-1 (`GlobalFourDegreeSetup`,
n=2000, `chunk_size=32`). Target trajectory generated from veros' own native
defaults (`c_k=0.1`, `c_eps=0.7`, see `veros/variables.py`); optimization starts
from `(c_k, c_eps) = (0.08, 0.6)` -- the same probe values used as arbitrary
test points throughout report-1/2.

Single process: `jax.value_and_grad(loss_fn, argnums=(0,1))` compiled once
(unlike the sweep scripts' one-jit-per-process pattern, which exists for
compile-time/memory *isolation* across different configs -- here we want to
reuse one compiled graph across many GD steps instead). Manual Adam
(`lr=0.002`, standard `beta1=0.9`/`beta2=0.999`) -- chosen over hand-tuned
per-parameter plain GD because report-1 found `c_k`'s gradient consistently
~5-10x larger magnitude than `c_eps`'s at matching n; Adam's per-parameter
adaptive step handles that without manual tuning. Parameters clipped to
`[0.01, 1.0]` as a safety rail (physical mixing constants, must stay positive;
also guards against a runaway step given how steep report-1 found this loss to
be).

## Results

Compiled in 317.9s. 40 steps, ~83s each (~55min total run), peak memory
4.81GB -- matches report-1's own n=2000/chunk_size=32 reading (4.82GB) almost
exactly, as expected (identical rollout structure).

| step | loss | c_k | c_eps | grad_c_k | grad_c_eps |
|---|---|---|---|---|---|
| 0 | 4.083 | 0.0800 | 0.6000 | -507.0 | 51.1 |
| 3 | 0.670 | 0.0857 | 0.5943 | -208.0 | 15.4 |
| 5 | 1.027 | 0.0887 | 0.5920 | **5.62e+06** | **-5.37e+05** |
| 7 | 0.629 | 0.0867 | 0.5940 | -157.6 | -1.2 |
| 21 | 3.405 | 0.0809 | 0.5999 | -5880 | 383 |
| 22 | 3.586 | 0.0807 | 0.6001 | **-1.31e+07** | **9.53e+05** |
| 31 | **0.612 (best of run)** | 0.0864 | 0.5948 | -0.35 | -26.8 |
| 39 (final) | 1.009 | 0.0887 | 0.5927 | -719.8 | 53.2 |

Full trajectory: `figures/report-longrollouts-3/gd_history.csv`.

![loss / c_k / c_eps vs GD step](figures/report-longrollouts-3/gd_trajectory.png)

## Bottom line

**It did not converge cleanly.** Loss dropped fast and looked promising early
(4.08 -> 0.63 by step 7), but the run never settled: loss climbed back to 3.59
by step 22, partially recovered, and ended at 1.01 -- worse than the best point
actually visited (0.61 at step 31, `c_k=0.0864`, `c_eps=0.5948`). `c_k` moved
about 45% of the way from init (0.08) toward true (0.1) by the end (0.0887);
`c_eps` moved slightly in the *wrong* direction overall, from 0.600 (init) to
0.593 (final) -- away from true 0.7, not toward it.

**The proximate cause is visible directly in the gradient log: sporadic,
enormous spikes.** Most steps show `grad_c_k` in the same few-hundred range
report-1 characterized at this n (-100 to -700), but steps 5, 21, 22, 29 and 32
show gradients 4-5 orders of magnitude larger (5.6e6, -5.9e3->-1.3e7, -1.8e4,
1.2e4) -- each one throws the Adam trajectory off course, and it takes several
steps to recover before the next spike hits.

**Important nuance this run surfaces: report-1's "n=2000 is sane" finding was
only validated along single-parameter axes, with the other parameter pinned at
its true value** (report-1's sweep varied `c_k` with `c_eps` fixed at 0.7, and
vice versa -- never both perturbed simultaneously). This GD run walks through
the doubly-off-nominal region of parameter space where both `c_k` and `c_eps`
differ from truth at once, and the gradient there is evidently not uniformly
well-behaved even at a horizon (n=2000) previously confirmed sane along the two
tested axes. Gradient sanity at a fixed n is not a single yes/no property of
the rollout length -- it also depends on where in parameter space you evaluate
it, and joint optimization necessarily visits points single-axis probing never
does.

One more candidate explanation, not yet distinguished from the above: the loss
did fall substantially (4x) even while individual parameters ended up far from
their true values, which is also consistent with `c_k`/`c_eps` being partially
degenerate for this loss/setup (multiple parameter combinations giving similar
temp trajectories) rather than -- or in addition to -- gradient noise. Not
resolved here; would need e.g. a loss-landscape scan or an ensemble of GD runs
from different inits to tell the two apart.

**Compare to reports 1/2**: this is the first time in the series a "sane at
this n" gradient has actually been driven through many sequential update steps
rather than evaluated once. The result complicates the picture from report-1 --
a single clean gradient evaluation at a given n does not guarantee that
gradient descent using that gradient, repeated over many steps and moving
through parameter space, will behave well. The long-horizon breakdown in
reports 1/2 and the mid-optimization spikes here look like the same underlying
phenomenon (chaotic trajectory sensitivity) showing up along a different axis
(parameter space instead of rollout length).
