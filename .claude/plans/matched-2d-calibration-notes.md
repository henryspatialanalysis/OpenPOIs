# Running notes: matched-segment 2-D calibration modes

Execution log for `.claude/plans/matched-2d-calibration-surface.md`. Every decision or
assumption the plan left open is recorded here with its reason. Newest entries go at the
bottom of each section.

## Setup

- **Branch.** `feature/matched-index-modes`, cut from `feature/ghost-closure-evidence`
  (8e8be6b), not from `main`. That branch is 6 commits ahead of `main` (manual
  overrides, same-entity ghosts, docs). `confidence-calibration.md` was edited in one of
  those commits, so cutting from `main` would set up a doc conflict at merge time. The
  new branch carries those commits; merge order is Nat's call.
- **Cron heartbeat** set at Nat's request: every 30 min at :07 / :37. It is
  session-only and deleted when the work is done.
- **Populations** were streamed with the repo's own `population_by_segment` (non-shadow,
  column-scoped) and cached in the session scratchpad:
  - 20260730 `conflated_cd`: matched 1,787,072. This equals the handoff's
    `frame_populations.matched`, so the July file has no shadow rows to drop.
  - 20260902 `conflated_cd`: matched 1,817,729.

## Decisions

1. **Which production population, where.**
   - The cross-fit (model comparison) places lookup bins and surface cells on the
     **20260730** population, the conflation version the validation round was drawn from
     (`metadata.json: conflation_version`). That keeps the evaluation internally
     consistent with the round.
   - **Deployed impact** uses the **20260902** population and the deployed curves in
     `conflation/20260902/calibration/` (July curves, copied). Each mode is fit in full
     with 20260902 bin placement, as a refit for the next release would be, and is then
     compared POI by POI with the deployed map on the same 20260902 rows.
2. **Surface cells (6×4), realized on 20260730 production, rounded to 6 dp.**
   - OSM edges: 0, 0.7168, 0.8244, 0.8842, 0.9300, 0.9661, 1.
   - Overture edges: 0.013909, 0.919912, 0.958695, 0.990219, 1.
   - With `searchsorted(side = "right")` each atom *starts* its bin: 0.919912 opens
     [0.9199, 0.9587) and 0.990219 opens [0.9902, 1]. The plan's "each spike becomes its
     own bin" is true in mass terms (the atom dominates its bin) but not literally.
   - Gold per cell: min 6, max 47. All 24 cells have ≥ 6 gold, so the sanity anchor holds.
   - 8×8 requested realizes 8×6. One cell has 0 gold and 3 phase-1 rows. The difference
     estimator is still defined there (prediction term only), so the plan's "refuse on
     zero phase-1 rows" rule does not trigger. The HT reference surface is undefined for
     that cell and is reported as missing.
3. **Constrained pool (D1).** L-BFGS-B on the normalized-weight log loss with an
   analytic gradient. It reproduces July to 4e-5 (intercept 0.20417 vs 0.20422; slopes
   0.21009 / 0.56938), inside the plan's 1e-4 anchor. The residual is sklearn's own
   solver tolerance in July, not the bound (neither slope is near it).
4. **Additive (D2) — as specified, plus two guards the plan left implicit.**
   - Step-halving on the outer IRLS step, blending old and new *components*: a
     convex blend of nondecreasing functions stays nondecreasing, so the guard never
     breaks monotonicity.
   - Knots with zero working weight are interpolated from their neighbours.
   - Tested against an exact convex solver (L-BFGS-B on nonnegative increments):
     local scoring reaches the same deviance.
   - Known property, not a bug: the isotonic MLE **spikes at the end knots**. A lone
     y = 0 at the lowest Overture score sends that block to the −9.21 clip (16 gold
     rows sit on the clip on round 20260730). The 1-D recalibration absorbs it, but it
     inflates bootstrap variability and gives the index production atoms. As a result
     only 12–25 of the 40 requested equal-mass bins survive `np.unique`.
5. **Interaction (D3).** `trust-constr` as specified, 40 ms per fit. Started from the
   constrained pool mapped to `a3 = 0` (always feasible). Interior-point results can
   sit ~1e-9 outside a bound, so the slopes are nudged back onto the feasible side
   after the solve; the dense-grid monotonicity test checks it.
6. **Surface projection (D4.3).** Dykstra–Robertson with increments, as specified.
   Each row/column PAV uses the vectorized min-max formula
   (x_i = max_{j≤i} min_{k≥i} avg), so one call projects a whole batch of matrices.
   Converged matrices drop out of the batch (2000 draws: 0.9 s). The oracle test uses
   the exact NNLS dual of the projection, not SLSQP: SLSQP itself only reaches ~1e-7,
   below the plan's 1e-8 requirement. Plain alternation (no increments) misses by 0.26,
   so the regression guard has teeth.
7. **Wald / mixture band (D5).** Implemented per Xu, Meyer & Opsomer 2021 eq. 2.6 and
   Liao, Meyer & Xu 2024 §3, both read in full-text beforehand.
   - `I − P_J` is built as W-weighted averaging within the connected components of
     the binding adjacent pairs. This equals the paper's
     `I − W⁻¹A_J'(A_J W⁻¹ A_J')⁻¹A_J` and needs no linear-independence bookkeeping
     for dependent 2-D constraint rows.
   - Binding tolerance is 1e-8.
   - Gold-ESS bound weights are floored at 0.5 (not 1): a positive floor is needed
     for uniqueness, and 1 would overweight cells with ESS ≈ 3.
8. **Published surface band.** `FitConfig.surface_band_method = "percentile"` by
   default (parallel with the shipped 1-D bands). Both bands are always computed and
   reported; the coverage simulation informs which to ship.
9. **Surface bootstrap.** A replicate that empties a cell of phase-1 rows is dropped.
   On 20260730, 0 of 500 were dropped.
10. **Cross-fit scoring through the published lookup (plan step 8).** Per fold, the
    40-bin lookup (or the cells) is built on the 20260730 production population and
    held-out gold is scored through it. The old grid-interpolated score is kept as
    `predicted_grid` and reproduces the July table exactly (pool 0.07759 / 0.27441,
    average 0.07980 / 0.29348).
    - Finding: publishing through 40 equal-mass bins costs the pool DSC
      0.00741 → 0.00588 and Brier 0.07759 → 0.07870. The primary table is therefore
      lower-scoring than July's numbers for every index mode. That is the cost of the
      deployed map, not a regression.
11. **Ladder bootstrap seeds.** `average − pool` keeps July's seed (`rng_seed + 3100`)
    so its interval reproduces the published one: Brier +0.00214 [+0.00020, +0.00406]
    against published [+0.00020, +0.00405]. Every other pair gets `rng_seed + 3200 +
    offset`.
12. **`--matched-index-mode` in the compare script** is realized as `--modes`
    (comma list), because the compare script evaluates many modes by design.
    `fit_calibration.py` got `--matched-index-mode`, `--out-dir` and
    `--allow-deployed` as specified.
13. **Deploy-side rounding is conditional.** `calibrate_frame` rounds a segment's
    scores only when its curve metadata carries `score_decimals`. The deployed July
    curves have unrounded edges, e.g. `0.9199122190475464`. Rounding a score to
    `0.919912` would drop it *below* that edge and silently change today's published
    values.
14. **Deploy support was implemented now, not in Phase 5.** `apply_surface`, an
    `index_score` dispatch in `curve_index`, and `score_decimals` plumbing are all in
    place. The deployed-impact numbers go through the real `calibrate_frame`, so they
    needed it. No config was changed: production stays `pool`.
15. **Independence surface.** The plan says to build it "from the two 1-D segment
    curves". Taken literally (osm-only and overture-only curves), it sits 0.2–0.47
    *below* every fitted matched map. POIs found by one source alone are far less
    likely to exist than a matched POI with the same score, so those curves are the
    wrong marginals and the gap says nothing about dependence. The figure instead uses
    the matched segment's **own** one-score marginal curves (the 1-D composite
    estimator on matched rows by `osm_score`, then by `overture_score`). The literal
    version is kept in `shape_diagnostics.json` as `independence_segment_curves`.
16. **Coverage simulation (D6) choices.**
    - Phase-2 gold is drawn at the round's realized per-verdict inclusion *rate*
      times the simulated class size, not at a fixed count, because simulated class
      sizes vary. Unverifiable is censused.
    - Populations are subsampled to 1M POIs per segment (fixed seed) for bin
      placement and targets, which keeps workers light. At 40 bins that is 25k POIs
      per bin.
    - Truths are each fitted point-estimate curve evaluated at its index:
      `pool`, `interaction` (a3 < 0 on this round, so no hand-set truth was needed),
      `overture_only` (the matched rows' 1-D curve in `overture_score`), and the osm /
      overture segment curves.
    - Timing: one simulation (17 fits × 200 bootstrap reps) takes 113 s serial and
      ~550 s under 11 parallel workers (hyperthreads plus memory bandwidth), about
      2.2× net throughput. 200 sims ≈ 2.8 h, well past the plan's 2 h mark. Per Nat's
      standing instruction not to ask, it was launched anyway and documented here.
17. **v4 errata vs. this doc.** `confidence-calibration.md`'s "loses about 43% of
    the discrimination" is correct as written (the average has 43% less DSC). The
    erratum is only in the v4 writeup ("~43% more"). Corrected in the doc: the
    unbiasedness wording, and the pool's "coherent / dependence discount" framing.
    Also corrected: `class_working_models`' claim that a no-gold class leaves the
    estimator unbiased. It does not; that class's rows keep an uncorrected fallback
    prediction.
18. **Housekeeping.**
    - I ran one `git stash` / `stash pop` to confirm that three
      `test_osm_history_pbf` failures pre-date this work. That was a write command
      against the read-only rule; the tree came back unchanged.
    - Three other suite failures (`test_osm_snapshot` download mocks,
      `test_constant_lambda_simulation`) sit outside calibration code.

## Late decisions (after the first results)

19. **Extra 20-seed sweep (not in the plan).** The four pre-registered runs left
    interaction − pool ambiguous: CI excluding zero in 3/4 on log score but not in the
    primary run. Seeds 101–120 (5-fold, 400 reps; pool / additive / interaction /
    surface) were added to measure sign consistency. They were cheap (~20 s each).
20. **Simulation diagnosis.** The index-mode bands under-cover (pool 0.76). I checked
    whether that is variance or bias by comparing, per bin across simulations, the
    empirical SD of the estimate with the band-implied SE: ratio 1.40 pool,
    1.31 osm, 1.18 overture, 0.96 surface-Wald.
    - A real-data recomputation without the POI-anchored second kernel pass widened
      the bands 1.15× (osm) and 1.32× (matched). That pins the main cause.
    - Nothing in the shipped band code was changed. The writeup recommends the fix.
21. **HT agreement** is evaluated at the 2,500 phase-1 rows for index modes (not grid
    points) and at the 24 cells for the surface. It is in `ht_agreement.json`.
22. **Monotonicity check limitation.** A pair involving a bin with < 5 gold gets no z.
    On the overture segment this hides the v4 §4.6 reversal behind a one-row bin
    (0.91967–0.919912). This is documented in the writeup §4.7 as a needed refinement:
    merge sub-floor bins into a neighbour instead of skipping them.
23. **Decision memo** at `~/data/library/writeups/2026-09-25-openpois-matched-segment-modes.md`
    (superseded by the 2026-09-26 IMRAD paper, item 29).
    It was drafted under the academic-humanizer rules (no em-dashes; claims tied to
    numbers). No `.docx` render has been made yet; that was left as an offer.
24. **Recommendation given (not a decision):**
    - Keep the constrained `pool` for October, with `interaction` as the
      pre-registered challenger for the next round.
    - Fix the band method, which is the higher-value change.
    - For 3+ D, extend the pool.

## Code-review follow-ups (2026-09-26)

A `pr-code-reviewer` pass found no leakage or misalignment in the new code. It raised
four points, handled as follows.
25. **POI-anchored bootstrap off-centring (older code).** This confirms and extends
    item 20.
    - Added `FitConfig.band_aggregation = "bin"`: each replicate's map is averaged
      over the production POIs in each published bin (`bin_band_from_maps`, up to
      200k POIs subsampled).
    - Left the default as `anchored_kernel`, so shipped behaviour is unchanged until
      Nat decides.
    - Tested in a targeted 200-sim re-run: coverage 0.77 → 0.89 (pool), 0.88 → 0.91
      (osm), 0.48 → 0.66 (overture).
26. **`build_lookup` edge convention (older code): FIXED.** This is a plain
    fit/deploy parity bug. Bins are now assigned with deploy's `searchsorted(side =
    "right")` (`lookup_bins`), and a parity test was added.
    - All comparison runs, the seed sweep, the fits and the figures were re-run after
      the fix. The July anchors still reproduce.
    - The first full coverage study (`coverage/`) predates the fix. For the index
      modes it is essentially unaffected (0.1% of matched rows sit on an edge). Its
      overture-1-D target had a small built-in atom bias, now gone in the targeted
      re-run (0.460 → 0.475).
27. **Simulation seed now covers the synthetic data draw.** Each run needs its own
    `--out-dir`, as the script comment says.
28. **Paired-bootstrap strata.** The compare script's fold-only strata were kept for
    July continuity, and the plan accepted them. A fold × refined-class sensitivity was
    run on every out-of-fold file (`ladder_fold_class_strata.csv`): interaction − pool
    excludes zero in 12/24 (Brier) and 10/24 (log) runs, against about 6–8/24 with
    fold-only strata. No verdict changed elsewhere.

## Decision and rollout (2026-09-26)

29. **Nat's decisions:** adopt `interaction` ("more conceptually defensible"), turn on
    the band fix, write the results up as an IMRAD paper with a sensitivity analysis
    and a 3+ score section, render to Word, wire it in for October, run a consistency
    pass, then commit and push.
    - Paper: `~/data/library/writeups/2026-09-26-openpois-matched-segment-modes.md`
      and `.docx`. That is a new date-stamped file per the versioning rule; the
      09-25 memo was kept. The Word file was rendered with pandoc
      (`--shift-heading-level-by=-1`) plus `~/repos/marketing/templates/apply_house_style.py`.
    - References list only fields present in the library's `citation.bib` records.
      Bordley (1982) is not in the library and is cited through Genest & Zidek.
30. **Wiring.**
    - `config.yaml`: `matched_index_mode: interaction`, `band_aggregation: bin`,
      `pool_min_coef: 0.001`. `fit_calibration.py` and `compare_matched_index.py`
      read all three.
    - `FitConfig` defaults were changed to the production values, so direct callers
      match config.
    - The deploy side needed no change: `curve_index` dispatches on the metadata's
      `index.form`, and the deploy round-trip test covers `interaction`.
    - `plot_calibration.curve_index_for_rows` now reads `index_mode` and `index`
      from the metadata (plan item 10).
31. **October refit is mandatory once.** The monthly drift gate would otherwise reuse
    the July pool curves. The "method-change override" was added to
    `confidence-calibration.md`, the conflate-snapshots skill and TODO.md. A dry run
    with the production config (`fit_production_config/`, bins on 20260902) gave
    a = (−11.50, 11.11, 17.42, −11.10), `a1+a3` active, and cross-fit Brier 0.0785.
32. **Consistency fixes.**
    - `fit_report.md`'s "median band width" now reports the *published* band
      (`band_width_grid_median` keeps the old grid figure).
    - Sphinx `docs/api.rst`, the CHANGELOG (Unreleased) and the project CLAUDE.md
      are updated.
