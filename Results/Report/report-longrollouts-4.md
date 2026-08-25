# mld_ma on nz=64: Does It Ever Give a Usable Gradient?

Question: does `mld_ma` (tested fairly this time — buffer always fully valid, `n >= window`) behave like temp_ma — sane at moderate n — or is it broken from the start on this heavier setup?

## Method

Setup: `GlobalFlexibleMLDLearningSetup`, nz=64, gsw/TEOS-10, streamfunction, real ETOPO5 topography — much heavier than `global4deg`. The setup's built-in `mld_ma` tracking had to be disabled and replaced with report-2's memory-bounded two-phase design [1]. Test value `c_k=0.08` only; two window sizes tried, 365 days (1yr) and 10 days [2].

## Results

Tesla P100-16GB, nz=64, `test_val(c_k)=0.08`.

| n | window | status | compile (s) | run (s) | grad | peak mem (GB) |
|---|---|---|---|---|---|---|
| 365 | 365 | OK | 550.7 | 59.8 | 1.51e+30 | 10.16 |
| 500 | 365 | OK | 834.9 | 81.9 | -2.59e+37 | 11.18 |
| 10 | 10 | OK | 398.2 | 1.8 | -71,439 | 3.98 |
| 50 | 10 | OK | 531.5 | 8.3 | -1.93e+07 | 9.45 |
| 100 | 10 | **CRASHED** | -- | -- | -- | -- |

![grad magnitude vs n, both window sizes](figures/report-longrollouts-4/grad_magnitude.png)

## Bottom line

**mld_ma is broken almost immediately, regardless of window size** [3]. This is much earlier and more severe than temp_ma's breakdown on `global4deg` [4]. Combined with report-3, this closes out the current line of investigation: the `double_checkpoint` rollout itself works (flat compile, tunable memory) — the unsolved problem across this whole series is gradient *correctness* under long/chaotic backpropagation, and it's worse for `mld_ma` than for temp by a wide margin [5].

See `report-longrollouts-4-appendix.md` for the full detail behind each note.
