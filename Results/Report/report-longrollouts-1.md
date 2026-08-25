# Long-Horizon Gradient Scaling on GPU

Motivated by OOM crashes seen backpropagating through long rollouts on GPU (see
README's server micro-guide). Goal: get `dloss/d{c_k,c_eps}` for rollouts of
n=500-3000 steps, on GPU, with the wall time and peak GPU memory each config costs.

This report therefore uses `GlobalFourDegreeSetup` (nz=15, nonlin2 EOS, no streamfunction.

## Method

| structure | result |
|---|---|
| `scan(checkpoint(scan(step)))` (nested scan) | Compile bug |
| `scan(checkpoint(python_loop(step)))` | Compile bug |
| `scan(checkpoint(step, policy=nothing_saveable, prevent_cse=False), _split_transpose=True)`, no chunking | Compile good but OOM at runtime |
| **`scan(checkpoint(scan(checkpoint(step))))`** -- per-step checkpoint *and* an outer checkpoint around the chunk | Good compile + no OOM |

## Results

Tesla P100-16GB, `global4deg`, test values `c_k=0.08`, `c_eps=0.6`.

### c_k / c_eps at chunk_size=8

| n | param | status | compile (s) | run (s) | grad | peak mem (GB) |
|---|---|---|---|---|---|---|
| 500 | c_k | OK | 321.1 | 21.4 | -160.39 | 3.74 |
| 1000 | c_k | OK | 204.4 | 41.8 | -667.52 | 6.75 |
| 2000 | c_k | **OOM** | -- | -- | -- | -- |
| 500 | c_eps | OK | 317.6 | 21.2 | -14.35 | 3.74 |
| 1000 | c_eps | OK | 205.7 | 41.5 | -70.12 | 6.74 |
| 2000 | c_eps | **OOM** | -- | -- | -- | -- |

### Pushing chunk_size to reach n=10000 (c_k only, feasibility-only)

Raising `chunk_size` (fewer, bigger blocks -> fewer outer-scan iterations `n_full`) 

| n | chunk_size | status | compile (s) | run (s) | grad | peak mem (GB) |
|---|---|---|---|---|---|---|
| 2000 | 32 | OK | 456.7 | 85.3 | -1242.37 | 4.82 |
| 5000 | 32 | OK | 457.9 | 247.8 | -4.51e+30 | 9.30 |
| 10000 | 64 | OK | 458.5 | 670.0 | 1.40e+114 | 10.73 |

**These three runs are feasibility-only, c_k, no c_eps, no finite-difference check.**
The gradient values at n=5000 and n=10000 are obviously unphysical
(`-4.5e30`, `1.4e114`).

![compile time / run time / peak memory vs n](figures/report-longrollouts-1/scaling.png)

## Conclusion

1. With the new `double_checkpoint` approach compile time is constant in n

2. We can use `chunk_size` to control the memory w.r.t n -> we can compute very long gradients 

3. `double_checkpoint`strategy also works on `mld_ma` + `GlobalFlexibleMLDLearningSetup` but we need to take a much larger `chunk_size` as the state is larger 

4. Gradients are unrealistic as soon as we increase the n too musch (around 1000) -> that's the main problem now




## Modifications : 

- In the above graph peak_memory vs n it would be good to have a few more points for each method so that we can check the tendency match well
- Are the gradients depending on `chunk_size` ? Like gradients for n=100 the same for `chunk_size = 5` and `chunk_size=25` ? -> It would be good to have a figure with that 



## Open questions 

1. What are the reason why the gradient would explode like that ? How do deal with this ? 

