# Plan: report-mld-4 — ablating the real config to find what breaks the gradient by n=20

## Goal

`report-mld-3` found the real config (`GlobalFlexibleMLDLearningSetup`: nz=60, gsw/TEOS-10, streamfunction, real ETOPO5 topography) gives an unusable gradient (autodiff vs. finite-difference sign-flipped, rel_err 1.64) already by n=20, for both mld loss and temp loss — while plain `global4deg` temp loss stays sane through n~2000-3000. This plan isolates which ingredient (gsw, streamfunction, real topography) is responsible, or whether it takes more than one combined.

nz is not part of this ablation — every run so far (baseline, mld probe, temp probe) already used the config's native nz=60; nz=64 only ever existed as an unrelated synthetic override in `report-longrollouts-4`. Not varied here.

## Ablation matrix

All rows: nz=60, temp loss (`((state.variables.temp - target_state.variables.temp) ** 2).sum()`, unmasked — same as `report-1`/`diag_n200_temp_loss.py`), param `c_k`, `test_val=0.1082` (matches `report-mld-3`'s convention), n=20 only (the confirmed first breakpoint — no need to re-sweep other n here), `eps=1e-6` for finite difference (`1e-4` is known from `report-mld-2` phase1 to false-alarm via `get_index_mld` branch flips on this grid — not relevant to temp loss directly, but kept for consistency), `warmup_steps=20`.

| # | config | gsw (`eq_of_state_type`) | streamfunction | topography | class | status |
|---|---|---|---|---|---|---|
| 0 | full (baseline) | 5 (gsw) | on | real ETOPO5 | `GlobalFlexibleMLDLearningSetup` | **already have** — n=20 rel_err=1.64, grad=-1575.1, num_grad=2471.9 (from `report-mld-3`, reuse, do not rerun) |
| 1 | gsw-only | 5 (gsw) | off | real ETOPO5 | `GswOnlyFullGridSetup` | exists in `report-mld-2/common.py`, never run |
| 2 | stream-only | 3 (nonlin2) | on | real ETOPO5 | `StreamOnlyFullGridSetup` | exists in `report-mld-2/common.py`, never run |
| 3 | neither | 3 (nonlin2) | off | real ETOPO5 | `NeitherFullGridSetup` (new) | write new class: same pattern as the other two, force both flags off |
| 4 | idealized ACC | 5 (gsw) | on | idealized channel (`ACCSetup`, no real data) | `ACCFullSetup` (new) | write new class — see below |

### Idealized-topography class definition: `ACCFullSetup`

Rather than hacking real ETOPO5 into a flat shape, use the existing idealized ACC channel setup (`setups/acc/acc_learning.py`'s `ACCSetup`) — a purely analytic geometry (`vs.kbot = (x > 1.0) OR (y < -20)`, cast to int; `veros/core/numerics.py:203`'s `maskT` rule makes `kbot=1` full depth, `kbot=0` land), no bathymetry data, no interpolation, no relief at all. This rules out the realistic-topography angle more thoroughly than a flat-bottom variant of the real config would: it removes both bathymetric relief *and* the real coastline/dataset complexity in one step. (Forcing also differs from the real config's — ACC uses idealized wind/buoyancy forcing, not the interpolated real forcing fields — so a clean result here says "the real config's combination of gsw+streamfunction+real topography+real forcing isn't required," not "topography specifically is/isn't the cause" in perfect isolation. Worth keeping in mind when interpreting.)

`ACCSetup`'s own `set_grid` hardcodes a 15-level `dzt` array, so bumping `nz` to 60 needs a `set_grid` override too — reuse the real config's own nz-parameterized pattern (`global_4deg_mld_learning.py`'s `set_grid`: `vs.dzt = veros.tools.get_vinokur_grid_steps(settings.nz, self.max_depth, self.min_depth, refine_towards="lower")`, `max_depth=5400.0`, `min_depth=4.0`). Topography itself needs no change — `kbot`'s binary rule doesn't depend on `nz`.

```python
from setups.acc.acc_learning import ACCSetup
import veros.tools

class ACCFullSetup(ACCSetup):
    max_depth = 5400.0
    min_depth = 4.0

    @veros_routine
    def set_parameter(self, state):
        ACCSetup.__dict__["set_parameter"].function(self, state)
        with state.settings.unlock():
            state.settings.nz = 60
            state.settings.eq_of_state_type = 5
            state.settings.enable_streamfunction = True

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings
        vs.dxt = update(vs.dxt, at[...], 2.0)
        vs.dyt = update(vs.dyt, at[...], 2.0)
        vs.dzt = veros.tools.get_vinokur_grid_steps(
            settings.nz, self.max_depth, self.min_depth, refine_towards="lower"
        )
```

(`set_grid` fully overridden rather than calling the parent's, since the parent's `dzt` line is a fixed 15-length array — `dxt`/`dyt` lines copied verbatim from `ACCSetup.set_grid`.)

## Script layout (`scripts/Reports/report-mld-4/`)

- `__init__.py` — same `PRP` pattern as every other report dir.
- `common.py` — imports `GswOnlyFullGridSetup`, `StreamOnlyFullGridSetup`, `spin_up_full_grid`, `make_diff_step`, `set_vars`, `rollout` from `report-mld-2/common.py` (add `report-mld-2` to `sys.path` the same way scripts already cross-reference `PRP`); defines `NeitherFullGridSetup`, `ACCFullSetup`, and `temp_agg_function` locally. `spin_up_full_grid` takes the setup class as an argument already (see `report-mld-2/common.py`'s signature), so it works unchanged for `ACCFullSetup` too — only its warmup-progress `desc` string needs the config name passed through, already supported.
- `ablation_worker.py` — one-jit-per-subprocess worker (same discipline as every other worker in this repo — several JIT compiles in one long-lived process reliably stalls, per `report-mld-1/section1_worker.py`'s docstring). Args: `--config {full,gsw_only,stream_only,neither,acc_full}`, `--n`, `--param`, `--test_val`, `--eps`, `--mode {grad,fd}`. Looks up the right setup class from a `SETUPS` dict, runs `spin_up_full_grid(cls)`, computes either the autodiff grad or the finite-difference estimate, prints a `RESULT ...` line (same format as every other worker in this project).
- `run_ablation.py` — driver, loops over the 4 configs that need running (skips `full`, using the cached `report-mld-3` numbers directly in the table/plot). For each config, for each mode, **before** launching the subprocess, check whether `results/{config}.json` already has that mode's value; if yes, reuse it; if no, launch the worker and write the result to `results/{config}.json` immediately on success (not batched at the end). This is the resumability mechanism — see besteffort section below. After all configs are done (from cache or freshly run), print the summary table, compute rel_err, and save the comparison figure.

## Execution: Grid5000, besteffort queue

Per the README's GPU workflow, but on the `besteffort` job type instead of a normal reservation — besteffort jobs get scheduled opportunistically (no queueing wait) but can be pre-empted and killed at any time by a higher-priority job. Design above already assumes this: every config's result is persisted to disk the moment it finishes, so a kill mid-run only costs the in-flight config, not the whole matrix.

1. **Sync code** — but first, if a previous attempt left anything on the remote not yet pulled back (results/log/figure), fetch those *before* running `g5k sync code`: it treats local as source of truth and deletes remote-only files (this bit us once already on `report-mld-3`'s first probe — lost the figure, had to regenerate from captured log output).
   ```
   g5k sync code
   ```
2. **Submit as besteffort**:
   ```
   ssh arennes 'oarsub -t besteffort -l host=1/gpu=1,walltime=3:00:00 "cd ~/code/Veros-Autodiff/scripts/Reports/report-mld-4 && VEROS_DEVICE=gpu /home/emeunier/data/conda/envs/veros/bin/python run_ablation.py > run_ablation.log 2>&1"'
   ```
   (Use the veros env's python by full path, not bare `python` — `oarsub`'s non-interactive shell doesn't source `~/.bash_profile`/`~/.bashrc`, confirmed the hard way on `report-mld-3`'s first submission.)
3. **Poll** (`oarstat -j <id> -s`) until the state leaves `Waiting`/`Launching`/`Running`.
   - If it ends `Terminated` and `results/` has all 5 entries (4 fresh + the reused baseline folded in by the driver): done, go to step 4.
   - If it ends `Error` (pre-empted) or `Terminated` early with `results/` incomplete: **resubmit the identical command from step 2**. The driver resumes from whatever `results/*.json` already exist and only computes the missing configs. Repeat until complete.
4. **Fetch back before any further `g5k sync code`**: `run_ablation.log`, `results/*.json`, and the comparison figure, via `scp` from the remote code dir.

## Output

- Summary table: config × (autodiff grad, finite-diff grad, rel_err) at n=20, 5 rows (baseline + 4 ablations).
- One comparison figure (rel_err per config, log scale, baseline included) saved to `Results/Report/figures/report-mld-4/ablation_rel_err.png`.
- Write-up: either a new `Results/Report/report-mld-4.md` (own numbered-footnote report, same style as the rest) or folded into `report-summary-1.md` as a further appendix note extending `[6]` — decide once results are in, based on how much explanation the finding needs.

## Interpreting results (for when they come back)

- If only one ablation (gsw-only, stream-only, or idealized ACC) reproduces the n=20 break while `neither` stays sane: that ingredient alone is sufficient — clean answer.
- If `neither` (nonlin2, no streamfunction, real topography) *also* breaks: real ETOPO5 topography/nz=60 resolution alone is enough, independent of gsw/streamfunction — and if idealized ACC (nz=60, gsw+streamfunction, no real topography) stays sane, that would specifically implicate the real topography/forcing rather than nz=60 in general.
- If `neither` stays sane and *both* gsw-only and stream-only break: either ingredient alone is sufficient, worth knowing whether they compound (not separately tested here — would need a follow-up "both, real topography" cell, which is just the existing baseline).
- If idealized ACC (nz=60, gsw+streamfunction, idealized geometry+forcing) *also* breaks by n=20: the real ETOPO5 topography and real forcing aren't required either — points at gsw+streamfunction (or their combination) as sufficient on its own, independent of any topography/forcing realism.
- If no single ablation reproduces it but the full config does: it's a combined-interaction effect, not attributable to one ingredient — would need pairwise combinations as a further follow-up.
