# Does Temporal Averaging Stabilize Long-Horizon Gradients?

Follow-up to `report-longrollouts-1.md`, which found `dloss/d{c_k,c_eps}` for a
plain (single-final-snapshot) temp-squared-error loss becomes numerically
meaningless well before any memory/compile wall is hit: sane through n=1000,
garbage by n=5000 (`grad=-4.5e30`) and n=10000 (`grad=1.4e114`).

Question: does averaging the loss's target field over a trailing 1-year window --
the same technique `global_4deg_mld[_learning].py` uses for `mld_ma` (an exact
boxcar moving average, motivated there by smoothing MLD as a calibration target)
-- applied here to temp instead, stabilize the gradient at those long horizons?
Same `GlobalFourDegreeSetup` ("global4deg") as report-1, same `double_checkpoint`
rollout family, test values `c_k=0.08`, `c_eps=0.6`.

## Method

**Window**: global4deg's `dt_tracer = 86400.0` (1 day/step), so a 1-year trailing
average is a 365-step boxcar window (`TEMP_MA_WINDOW = 365`).

**Two-phase rollout**, to keep averaging's extra memory cost independent of the
overall rollout length n. Carrying a 365-step history buffer of the *full* 3D
temp field through the entire n-step `double_checkpoint` rollout would multiply
scan's own O(n_full) carry-history memory requirement (the root cause of
report-1's n-dependent memory growth) by that buffer's size -- i.e. make the
original problem worse, not better. Instead:

- **Lead phase** (`n - 365` steps): no averaging tracked, byte-for-byte
  report-1's `rollout()` -- identical memory scaling to what's already
  characterized there.
- **Tail phase** (exactly 365 steps, fixed regardless of n): its own
  `double_checkpoint` rollout over a carry `(state, temp_history, write_idx)`,
  circular-buffer boxcar average exactly like `update_mld_moving_average`,
  generalized from mld's 2D field to temp's 3D field. Because this phase's outer
  scan length is fixed at `365 / tail_chunk_size` regardless of n, its
  contribution to memory is constant in n -- only the (already-known) lead phase
  grows with n.

Loss is the squared error between this trailing temp_ma and the same quantity
computed on a plain (uncheckpointed) forward-only reference trajectory --
mirrors report-1's `temp_agg_function` exactly, just on the averaged field.

**Correctness check before touching the GPU**: the two-phase design (lead +
fixed-length averaging tail, both `double_checkpoint`) was verified on a toy
scalar model (`dyn(x,alpha)=sin(x)*alpha`) against a plain numpy/finite-difference
reference at 4 `(n, window, lead_chunk, tail_chunk)` combinations, including the
`lead=0` edge case (`n == window`). All 4 matched the reference to value error
<1e-9 and gradient relative error <1e-5 before any real Veros code was written.

`lead_chunk_size` per n reuses report-1's own calibration (8 for n=500/1000, 32
for n=2000/3000/5000, 64 for n=10000); `tail_chunk_size=32` fixed for all n (the
365-step tail is the same length regardless of overall n, so one calibration
value was expected to generalize -- the n=500 run was itself that check).

## Results

Tesla P100-16GB, `global4deg`, `window=365`.

| n | param | compile (s) | run (s) | grad | peak mem (GB) |
|---|---|---|---|---|---|
| 500 | c_k | 86.7 (first run: 563.5, XLA cache hit after) | 21.2 | -116.00 | 3.73 |
| 1000 | c_k | 561.8 | 41.8 | -215.98 | 6.74 |
| 2000 | c_k | 557.4 | 83.4 | -385.22 | 5.40 |
| 3000 | c_k | 556.1 | 124.0 | -2531.40 | 7.37 |
| 5000 | c_k | 561.2 | 243.0 | **4.00e+29** | 9.84 |
| 10000 | c_k | 562.7 | 656.7 | **9.01e+112** | 10.45 |
| 500 | c_eps | 568.0 | 21.3 | -12.01 | 3.74 |
| 1000 | c_eps | 565.7 | 42.0 | -26.79 | 6.74 |
| 2000 | c_eps | 555.6 | 82.6 | -45.31 | 5.41 |
| 3000 | c_eps | 559.9 | 123.6 | -75.42 | 7.38 |
| 5000 | c_eps | 563.4 | 220.2 | **5.23e+10** | 9.84 |
| 10000 | c_eps | 571.2 | 577.1 | **7.67e+69** | 10.44 |

All 12 configs ran to completion (`status=OK` for every one -- no crash, no
timeout; the large values above are finite-but-garbage floats, not allocation
failures).

![compile time / run time / |grad| vs n](figures/report-longrollouts-2/scaling.png)

## Bottom line

**Averaging did not stabilize the long-horizon gradient.** Both parameters show
the same qualitative failure report-1 found on the raw loss: sane, smoothly
growing gradients through n=2000 (`c_k`: -116 -> -216 -> -385; `c_eps`: -12.0 ->
-26.8 -> -45.3), an already-anomalous jump at n=3000 (`c_k` jumps ~6.5x to
-2531.4 where the trend up to n=2000 would predict roughly -550; `c_eps` grows a
more modest ~1.7x to -75.4, consistent with its trend, so the onset isn't
perfectly synchronized between the two params), then clearly unphysical values by
n=5000 (`c_k`: 4.0e29, `c_eps`: 5.2e10) and n=10000 (`c_k`: 9.0e112, `c_eps`:
7.7e69). The n=5000/n=10000 magnitudes are comparable in scale and character to
report-1's raw-loss blowup at the same n (`c_k`: -4.5e30 at n=5000, 1.4e114 at
n=10000) -- same failure mode, not a meaningfully different one. If anything the
onset moved slightly later for `c_k` (garbage confirmed by n=3000-5000 here vs.
report-1's uncharacterized 1000-5000 boundary), but this is not the order-of-
magnitude stabilization the averaging idea was hoping for.

**The two-phase design's own goal -- keep averaging's memory cost roughly
constant relative to report-1's plain rollout -- worked.** Peak memory here
tracks report-1's numbers at matching `lead_chunk_size` closely: n=500 (lead=8)
3.73GB here vs. 3.74GB in report-1; n=1000 (lead=8) 6.74GB vs. 6.75GB; n=5000
(lead=32) 9.84GB vs. 9.30GB (+0.54GB); n=10000 (lead=64) 10.45GB vs. 10.73GB
(actually slightly lower). The fixed 365-step tail phase adds at most roughly
half a GB on top of the lead phase's already-characterized cost, not a new
n-dependent term -- structurally the design does what it was built to do, it
just doesn't touch the actual problem.

**Compile time is consistently higher than report-1's raw loss**: ~556-571s here
across every n and both params (vs. report-1's 204-458s), a roughly flat extra
cost from the additional fixed-length tail-phase graph, paid once regardless of
n -- consistent with the whole point of using `lax.scan` at the outer level for
both phases.

**Why this was expected in hindsight**: averaging only changes what the forward
pass *reads out* at the end -- backprop still has to differentiate through the
*entire* unaveraged n-step chaotic trajectory to produce that average's
gradient (the tail phase's own scan is still a full `double_checkpoint`
differentiable rollout, and the lead phase feeding into it is untouched). The
butterfly-effect sensitivity blowup lives in that backward pass, not in how the
loss happens to summarize the forward state. Smoothing the *output* doesn't
remove chaotic sensitivity from the *path* the gradient has to be computed
along. A method that actually addresses this would need to change what's being
differentiated through (e.g. some form of shadowing/least-squares sensitivity
method for chaotic systems), not just what the loss reads at the end -- out of
scope here.
