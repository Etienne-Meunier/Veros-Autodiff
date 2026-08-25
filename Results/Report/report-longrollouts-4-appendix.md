# Appendix: report-longrollouts-4

Full detail behind each numbered note in `report-longrollouts-4.md`.

**[1] Why the built-in `mld_ma` tracking had to be disabled**

The setup's `after_timestep` computes `mld_ma` unconditionally every step via an always-on 720-step circular buffer baked into `state` for the *whole* rollout — directly defeating report-2's point of bounding the averaging buffer's cost to a fixed tail phase. `common.py`'s `make_nz_setup_no_avg` overrides `after_timestep` to compute only the raw `mld` value (same formula, still differentiable), skipping the built-in moving-average update; this report's own two-phase rollout then tracks the average of raw `mld` only during a fixed-length tail phase.

**[2] Why window=10 was also tried**

Window=10 was tried specifically to separate "window-length/chaotic-accumulation effect" from "something more fundamental" as the cause of the blowup — if a much shorter window stayed sane, that would point to accumulation over the averaging window itself as the culprit; if it broke too, that would point elsewhere.

**[3] The blowup, in detail**

With window=365, the shortest possible valid test (n=365, no lead phase) already gives `grad=1.5e30`. Shrinking the window to 10 doesn't help either: n=10 gives `-71,439` (large but not absurd), but n=50 (still window=10) jumps to `-1.93e7` — a ~270x increase for only 5x more n, exponential-looking growth rather than the smooth ~4x-per-2x-n scaling report-1 found for temp loss. n=100/window=10 crashed (OOM, 17.8GB requested) but used `lead_chunk_size=90` for a 90-step lead — effectively one giant unchunked block, a calibration mistake rather than evidence of a real n=100 memory ceiling; not re-run since the trend already answered the question this ladder was chasing.

**[4] Why mld_ma breaks so much earlier than temp_ma**

temp_ma stayed sane through n~2000-3000 on `global4deg`. Two candidate explanations for mld_ma's much earlier failure, not distinguished by this report:

- This setup's dynamics (gsw/TEOS-10 + streamfunction + real ETOPO5 topography) are intrinsically more chaotic/sensitive under autodiff than `global4deg`'s.
- The `mld` diagnostic's own formula is inherently steeper under autodiff — it's built from a division (`(prho_reference - prho_below) / (prho_above - prho_below)`, see `mld_from_index`), and division derivatives blow up as the denominator shrinks near degenerate/weakly-stratified columns, independent of what the rollout itself is doing.

Both are consistent with `mld_ma`'s history: report-mld-2 already found it broken at full-grid n=200 with *no averaging at all* (`grad=1.25e21`), long before this investigation began — the diagnostic has never produced a trustworthy gradient at any tested scale, on any setup, averaged or not.

**[5] How this connects to report-3**

report-3 found even temp's gradients (the one loss in this series that stayed sane over a wide range) don't reliably drive stable gradient descent once both parameters move jointly. Fixing the underlying correctness problem would need a fundamentally different differentiation strategy (e.g. shadowing or least-squares shadowing methods built for chaotic systems), not a different loss, window, or chunk size — out of scope here.
