# Matched-segment calibration: comparing monotone maps f(s_OSM, s_Overture) → P(exists)

**Status 2026-09-26: EXECUTED and DECIDED.** Phases 0–4 were implemented and run on
2026-09-25/26. Nat chose the `interaction` mode and the bin-level band for the October
2026 run; Phase 5 (deploy wiring, config, docs) is done. Results:
`~/data/library/writeups/2026-09-26-openpois-matched-segment-modes.md` (+ `.docx`);
execution log and every open decision: `matched-2d-calibration-notes.md`. The text
below is the plan as approved, kept for the record.

The earlier draft proposed one candidate, a kernel-smoothed doubly-monotone surface.
The audit (`~/data/library/_staging/audit_2026-09-25_matched-2d-surface/AUDIT.md`,
evidence files `A_…`–`H_…` beside it) found that a surface-vs-pool comparison cannot tell
a nonlinear per-source shape from a genuine interaction. It also found several of the
draft's technical claims wrong: zero projection weights, the stopping rule,
monotonicity after cell averaging, and the 2-D scaling-binning claim. And it found that
the design-based literature for step 4 is already in the library. This revision:
- compares five modes;
- estimates the free surface on cells rather than kernel nodes;
- adds a Wald/mixture-covariance band and a coverage simulation;
- ends in a writeup that Nat decides from.

## Context

Nat's request: for POIs with **both** OSM and Overture sourcing (the `matched`
detection segment), replace the reduction of the two source scores to a single index
with a two-dimensional map `f(Score_OSM, Score_Overture) → Score_combined`, which must be:
1. Monotonically increasing in each input score.
2. Easy to expand to 3+ dimensions.
3. Of the same character as the current calibration: the output is a calibrated
   **P(exists and is open)** for any score combination.

Test on the **existing** round-20260730 validation data (no new validation). If the
results are promising, roll out in the next monthly OpenPOIs update.

### What exists today (read these before coding)

Design source: `~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md`
(the "v4 writeup"). Operational doc: `.claude/docs/confidence-calibration.md`. Code:

| Path | Role |
|---|---|
| `src/openpois/conflation/calibration_fit.py` | v4 estimator: refined classes, inclusion weights, working models, difference estimator (`composite_curve`), fitted log-odds pool (`fit_pool`/`pool_score`), two-phase bootstrap, equal-mass lookup (`build_lookup`), cross-fit scoring |
| `src/openpois/conflation/calibration.py` | deploy: `curve_index`, `apply_curve`, edge-rule flags, streamed rewrite of `conflated_cd.parquet` → `conflated.parquet` |
| `scripts/conflation/fit_calibration.py` | fit driver (`make fit_calibration`) → curves + `fit_report.md` |
| `scripts/conflation/apply_calibration.py` | apply driver (`make apply_calibration`) |
| `scripts/conflation/compare_matched_index.py` | pool-vs-average decision harness (cross-fit + paired bootstrap + `--with-deployed-impact`) |
| `scripts/conflation/plot_calibration.py` | diagnostic figures |
| `tests/test_calibration.py` | estimator identities + edge rules |
| `data/calibration/20260730/` | validation handoff (gitignored): `validation_rows.parquet`, `metadata.json`, `reference_curves/` |

Current matched-segment pipeline (v4 §4.5): the two scores are pooled into one index
`z = b0 + b_osm·logit(s_osm) + b_ov·logit(s_ov)`. The coefficients come from
design-weighted logistic regression on matched gold; fitted 20260730 values are 0.204 /
0.210 / 0.569. The 1-D curve is then fit against `expit(z)`:
1. Nadaraya–Watson difference estimator on a 200-point grid, in **raw index space**
   (Silverman bandwidth 0.0175).
2. Weighted PAV, weighted by a histogram of the **phase-1 validation rows'** index
   scores (not production mass; `calibration_fit.py:375`).
3. A 40-bin equal-mass lookup placed on the production population.

The pool beat the average in the 20260730 cross-fit: Brier +0.0021, log score +0.0186,
both CIs excluding zero. The average has ~43% *less* DSC than the pool (0.00426 vs
0.00741). This reproduces exactly today.

The pool already meets requirements 1–3 *parametrically*, subject to one caveat:
`fit_pool` has no positivity guard. What it cannot represent is any per-source
nonlinearity in logit, or any interaction.

### Facts about the data that shape the design (verified 2026-09-25, `H_code_data.md`)

- Matched segment: **2,500 phase-1 rows, 444 gold**. The scores are uncorrelated
  (Pearson 0.013, Spearman 0.007), with medians 0.887 OSM and 0.958 Overture.
- **The Overture axis is two spikes, not a continuum.** 31% of matched rows sit at
  0.919912 and 22% at 0.990219 (production: 30.4% / 21.9%). The validation file splits
  each spike into three float values. **Round `overture_score` to 6 dp before any
  binning.** Equal-mass edges collapse: 8×8 → 8×6 cells (17 of 48 have < 5 gold);
  6×6 → 6×4 cells (all 24 have ≥ 6 gold).
- **Both axes are monotone within matched.** The largest reversal across bins is
  z = 1.45, so v4 §4.6's non-monotone Overture finding does not carry over to this
  segment on this round.
- **The 4×4 table hints at a substitutive interaction** ("either source suffices"). At
  the lowest OSM quartile, Overture moves the rate from 0.69 to 0.99; at the top quartile
  it is flat. The additive pool cannot represent this. The driving cells have only
  18–32 gold each, so it is a hint, not a finding.
- **The unverifiable-class working model does not affect the full-data fit.** All 154
  matched unverifiable rows are gold (π = 1), so their contribution is exactly `y`. It
  matters only inside cross-fit folds, where held-out gold is demoted.
- No matched validation row has Overture = 0.5; about 25 production rows do.

### Decisions (Nat, 2026-09-25)

1. Compare **five modes**: `pool` (baseline), `average`, `additive`, `interaction`,
   `surface`.
2. Estimate the surface on **cells** (a per-cell difference estimator), not a kernel grid.
   Use atom-aware **6×4 as primary**, with **8×6 as a sensitivity** row.
3. Bands: add a **coverage simulation** under the real two-phase design, **and** a
   **Wald / mixture-covariance** band for the surface to set against the percentile
   bootstrap.
4. The pool's source coefficients are **constrained positive in the fit itself**, so a
   coefficient ≤ 0 is impossible. This applies to every mode that has slope
   coefficients, and replaces the draft's equal-weight fallback.
5. There is **no automatic adoption rule**. Phase 4 produces a full writeup of
   trade-offs, validation metrics and conceptual justification per mode, then pauses for
   Nat's decision. Parsimony-first is the likely rule.
6. Fix the inherited overstatements in `confidence-calibration.md` and in the docstrings.
   Record v4 errata in the writeup; the v4 writeup itself is a dated deliverable and is
   not edited in place.

## The five modes

Every mode yields a map that is monotone in both scores, and every mode's output is a
calibrated P(exists ∧ open). The four **index modes** share one structure: the two
scores are reduced to a monotone scalar index, and the existing 1-D machinery (kernel
difference estimator → PAV → 40-bin lookup → bootstrap with POI-anchored
re-aggregation) runs on that index unchanged. Monotonicity follows because a monotone
curve applied to an index that is monotone in each input is monotone in each input. The
**surface** skips the index and estimates the 2-D table directly.

| mode | form | params | tests | conceptual grounding |
|---|---|---|---|---|
| `average` | mean(s_osm, s_ov) | 0 | — | parameter-free baseline (July comparator) |
| `pool` | a + b₁·logit s_osm + b₂·logit s_ov, b ≥ ε | 3 | — | weighted log-odds pool (Bordley 1982; Genest & Zidek 1986 §5); a monotone single-index model |
| `additive` | a + h_osm(s_osm) + h_ov(s_ov), h nondecreasing | ~#PAV blocks | per-source shape nonlinear in logit? | additive isotonic GLM (Bacchetti 1989; Mammen & Yu 2007; Chen & Samworth 2016); Bordley's non-interaction form (Genest & Zidek p.10, eq. 5.5) |
| `interaction` | a0 + a1x + a2y + a3xy on rescaled logits, Gupta constraints | 4 | interaction present? sign of a3 | monotone bilinear (Gupta et al. 2016 p.6); cheapest direct interaction test |
| `surface` | per-cell difference estimator on a 6×4 grid, projected onto the doubly-monotone cone | 24 cells | anything additive + interaction miss | design-based order-constrained domain means (Wu, Meyer & Opsomer 2016; Oliva-Aviles et al. 2020; csurvey); 2-D isotonic regression (Dykstra & Robertson 1982; Chatterjee et al. 2018) |

Read the comparisons as a ladder:
- **additive vs pool**: are the per-source curves wrong-shaped in logit?
- **interaction vs pool, and surface vs additive**: is there an interaction?
- **surface vs interaction**: is the interaction more than bilinear?

A surface-vs-pool win alone cannot distinguish these explanations.

What theory predicts at n = 444: the 2-D isotonic LSE's worst-case risk is n^-1/2
against n^-2/3 in 1-D, and it does not adapt to an additive-in-logit truth
(Chatterjee et al. 2018, Remark 2.4). A 3-parameter pool came within 0.0011 Brier of the
ideal combination at n = 500 (Ranjan & Gneiting 2010). Expect the free surface's best
case to be roughly a tie; the additive and interaction modes have the better chance of
earning their parameters.

## Proposed design

### D0. Shared data prep

- Round `overture_score` and `osm_score` to 6 dp on load, in both the validation rows
  and the production population, inside a helper used by every mode. Document why:
  float splits of the Overture spikes misplace ~320 rows at bin edges.
- Every index mode refits its index **inside every bootstrap replicate and every
  cross-fit fold**, exactly as the pool does now.

### D1. `pool`, constrained (replaces the unguarded `fit_pool`)

Minimize the design-weighted log loss directly with
`scipy.optimize.minimize(method = "L-BFGS-B")`. The intercept is free; both slopes are
bounded below by `FitConfig.pool_min_coef` (default `1e-3`), so a coefficient ≤ 0 cannot
be fit. Record `bound_active: [...]` and `method = "constrained_log_odds_pool_v2"` in
the metadata.

The thin/one-sided-gold fallback to the equal-weight pool stays: that fallback is about
identification, not sign. On 20260730 the unconstrained optimum is interior (0.210 /
0.569), so the constrained fit must reproduce the July coefficients and the July
comparison table to within 1e-4. That is a sanity anchor.

Scipy replaces the sklearn `LogisticRegression` call. sklearn cannot bound
coefficients, and the draft's alternative, falling back to equal weights, gives half the
weight to a score the data say adds nothing.

### D2. `additive` index

Model: `logit P(Y = 1) = a + h_osm(s_osm) + h_ov(s_ov)`, each h nondecreasing and
centred. Fit on gold with design weights by **local scoring**:
- Outer loop: IRLS, with working weights = design weight × p(1 − p).
- Inner loop: backfitting, where each component update is a weighted PAV of the partial
  working residuals on that axis. This is Bacchetti's (1989) CPAV. Mammen & Yu (2007
  p.15) show backfitting onto monotone components is the dual of Dykstra's algorithm, so
  it converges.
- Stop when the relative deviance change is < 1e-8 or after 100 outer iterations. Clip
  the linear predictor to ±logit(1 − LOGIT_EPS) so all-positive top-right blocks cannot
  diverge, and record the clip count.
- Store each h as knots (unique gold score values) and PAV block values. Evaluate by
  linear interpolation between knots with constant extension, which preserves
  monotonicity.

Index = `expit(z)` → the existing 1-D pipeline. Report the realized number of PAV blocks
per component as the mode's effective parameter count. Extends to D axes by adding
components.

### D3. `interaction` index

Rescale each logit score to [0, 1] over the **fixed clip range**
[logit(LOGIT_EPS), logit(1 − LOGIT_EPS)], not the observed range, so the monotonicity
guarantee covers every deployable score. The model is
`z = a0 + a1·x + a2·y + a3·x·y`. Minimize the design-weighted log loss subject to the
linear constraints `a1 ≥ ε, a2 ≥ ε, a1 + a3 ≥ ε, a2 + a3 ≥ ε`. These are necessary and
sufficient for monotonicity on the unit square (Gupta et al. 2016 p.6). Use
`scipy.optimize.minimize(method = "trust-constr")` with a `LinearConstraint`.

a3 < 0 is a substitutive interaction ("either source suffices") and a3 > 0 a
complementary one. Report a3 with its bootstrap interval. Index = `expit(z)` → the
existing 1-D pipeline.

### D4. `surface` on cells

1. **Cells.** Per-axis equal-mass edges over the **production** matched population
   (rounded; `np.unique`-collapsed, so each Overture spike becomes its own bin). Knobs:
   `surface_osm_bins = 6`, `surface_ov_bins = 6`; 6 requested Overture bins realize 4.
   Record the realized edges. Sensitivity run: 8 × 8 requested, realizing 8 × 6. If any
   cell has zero phase-1 rows, refuse and tell the user to coarsen; do not fill.
2. **Per-cell difference estimator** (the 2-D form of v4 §4.1, with no kernel and no
   bandwidth):
   `p̂(B) = (1/N_B) [Σ_{i∈B} ŷ_i + Σ_{i∈B∩G} (y_i − ŷ_i)/π_c(i)]`, with N_B = phase-1
   rows in B.
   - The working model ŷ is the **existing** `class_working_models`, evaluated on the
     constrained pool index and refit per fold/replicate. It is genuinely unchanged from
     production, and its choice matters only in cross-fit.
   - Clip p̂ to [0, 1] before projecting; the projection does not restore the range
     (Dykstra 1983 p.5).
3. **Projection onto the doubly-monotone cone.**
   - Weighted least squares with weights = production population count per cell. This
     is the Wu/Meyer/Opsomer choice, N_d. Floor the weights at 1; zero weights make the
     solution non-unique (Dykstra & Robertson 1982 p.3).
   - Algorithm: Dykstra & Robertson (1982) row/column PAV **with Dykstra's correction
     increments**. Plain alternation settles off the projection (0.04 in a synthetic
     test).
   - Stop when max |row-pass − column-pass| < 1e-10 and the max monotonicity violation
     is < 1e-10; cap at 10,000 sweeps and raise if unconverged. The draft's "max
     increment < 1e-6" never fires, because the increments converge to nonzero values.
   - At 24–48 cells this is instantaneous.
4. **HT reference surface.** Gold-only Hájek per cell, projected the same way. It is the
   robustness overlay, as `ht_reference_curve` is in 1-D.
5. **Emission.** The projected cell matrix **is** the lookup: estimation grid =
   published grid. No averaging step, so the draft's step-5 monotonicity failure cannot
   occur, and cross-fit scores exactly the deployed map.
   - Schema: `segment, osm_lo, osm_hi, ov_lo, ov_hi, conf_mean, conf_lower, conf_upper`.
   - This yields 24 published values, versus 40 bins for the index modes.

### D5. Bands

- **Index modes**: the existing `two_phase_bootstrap`, unchanged (POI-anchored
  re-aggregation exists precisely because the index is refit per replicate).
- **Surface, percentile band**: resample within verdict class exactly as
  `two_phase_bootstrap` does (fixed phase-1 class counts, gold resampled to fixed
  per-class counts, pool refit for the working model). Then:
  - Recompute the unconstrained cell estimates and project them, per replicate.
  - Take the pointwise 2.5/97.5% quantiles per cell.
  - Project each bound onto the cone, using the same weights.
- **Surface, Wald / mixture-covariance band** (Xu, Meyer & Opsomer 2021, IBIK7RG9;
  Liao, Meyer & Xu 2024, T8RTB3XY). The executing agent must read IBIK7RG9 §2–3 and
  T8RTB3XY §2–3 before coding this and follow them, not memory.
  1. Σ̂ = bootstrap covariance of the **unconstrained** cell estimates.
  2. Mixture covariance of the constrained estimator by simulation: draw from
     N(θ̂, Σ̂), project, and mix face-wise covariances.
  3. Wald bounds θ̃ ± z·√diag.
  4. Project the upper and lower bounds separately onto the cone, weighting by gold
     effective sample size per cell.
  5. The bottom cell of the product order (low OSM × low Overture) cannot borrow
     strength downward (T8RTB3XY p.5); flag it in the report.
- Why both: the percentile bootstrap of an order-constrained estimator is inconsistent
  where constraints bind (Andrews 2000; Sen, Banerjee & Woodroofe 2010, coverage
  0.72–0.83 at nominal 0.95). Here that means the top-right plateau and the floor, which
  is where most of the mass sits. This weakness is **inherited** from the shipped 1-D
  bands. It is not new.

### D6. Coverage simulation (new script `scripts/conflation/simulate_band_coverage.py`)

This audits every mode's band **and** the shipped 1-D `osm` / `overture` bands.

- **Truths T**:
  - (a) the fitted constrained-pool surface (additive in logit);
  - (b) a substitutive-interaction surface: the fitted `interaction` model if a3 < 0,
    else a hand-set one matching the 4×4 table;
  - (c) an Overture-only truth, which is plateau-heavy;
  - for the 1-D segments, each fitted curve.
- **Generative model**, holding round 20260730's structure fixed:
  - Keep each phase-1 row's scores.
  - Draw `Y ~ Bernoulli(T(s))`.
  - Draw the refined LLM class from `P(class | Y)`, estimated design-weighted on the
    segment's gold.
  - Draw phase-2 gold within verdict class at the round's realized per-class counts,
    with unverifiable censused.
- **Target**: the published quantity, i.e. the mean of T over production scores in
  each published bin or cell.
- **Report**: pointwise coverage and median width per mode × band method, split by
  region. The regions are sloped interior; plateau (T > 0.95 with small gradient);
  floor; and edge / corner cells. Also report the share of simulations with
  simultaneous coverage.
- **Budget**: time one full fit with 200 bootstrap reps first and extrapolate before
  launching. Defaults `--sims 200 --reps 200`. Append per-simulation results to a
  parquet so the run resumes. Run backgrounded with a log per the repo's long-run rules.

## Execution plan

Work on a fresh branch off `main` (e.g. `feature/matched-index-modes`). Git is
**read-only** unless Nat says otherwise; he will drive the commit session. Environment:
`conda activate openpois`; python is `/home/nathenry/miniforge3/envs/openpois/bin/python`.
Style: Black, 90-char lines, spaces around `=` **including keyword arguments** (see the
existing calibration code). Long runs: `python -u ... > ~/data/openpois/logs/<step>.log
2>&1`, backgrounded, watched with a Monitor that has both success and failure branches.

### Phase 0 — safety, prep, docs

1. **Never write into the deployed calibration dir.**
   - `versions.conflation = "20260902"`, and `make fit_calibration` writes to
     `conflation/20260902/calibration/`, which holds the July curves copied for release.
   - Add `--out-dir` and `--matched-index-mode` CLI overrides to `fit_calibration.py`
     and `compare_matched_index.py`.
   - All evaluation output goes to
     `~/data/openpois/conflation/20260902/calibration_eval_<date>/`.
   - Assert in both scripts that `--out-dir` is not the deployed dir unless
     `--allow-deployed` is passed.
2. Rounding helper (D0), used everywhere scores are read.
3. **Doc corrections** (Nat decision 6):
   - **`calibration_fit.py:26` and `confidence-calibration.md:94`** ("design-unbiased
     *whatever* the working model does"): exact only for a working model that does not
     depend on the sample. With a model fit on the same gold it is asymptotically
     unbiased (Breidt & Opsomer 2017 p.3, p.6).
   - **`calibration_fit.py:46, 174, 178`** (pool "coherent" / "damped for dependence"):
     the fitted free-coefficient pool is Bordley/French-form. The externally Bayesian
     logarithmic pool has weights summing to 1 and cannot exceed the larger input; the
     pool that can is "not externally Bayesian" (Genest & Zidek 1986 p.6, p.9).
     Coefficients below 1 mix dependence damping with each raw score's own
     miscalibration.
   - Check `calibration_fit.py:303` for the same unbiasedness wording.
4. `matched-confidence-surface.md` is already marked superseded (done 2026-09-25).

### Phase 1 — estimators (`src/openpois/conflation/calibration_fit.py`)

5. `FitConfig`:
   - add `pool_min_coef: float = 1e-3`, `surface_osm_bins: int = 6`,
     `surface_ov_bins: int = 6`, `surface_max_sweeps: int = 10_000`,
     `surface_tol: float = 1e-10`, `wald_mixture_draws: int = 2000`;
   - extend `matched_index_mode` to accept `"additive" | "interaction" | "surface"`;
   - update the docstring and the `segment_scores` `ValueError` path.
6. Generalize "pool params" to **index params** carrying a `form` key (`pool`,
   `additive`, `interaction`). Then:
   - `fit_index(rows, weights, mode, fit_config)` dispatches to `fit_pool`
     (constrained, D1), `fit_additive_index` (D2) or `fit_interaction_index` (D3);
   - `index_score(osm, overture, params)` evaluates any form;
   - `needs_pool` becomes `needs_index`;
   - `segment_scores`, `fit_segment`, `two_phase_bootstrap` and `cross_fit_predictions`
     call these, with no mode-specific branches beyond the dispatch.
7. Surface functions: `surface_edges`, `cell_difference_estimator`,
   `project_monotone_2d` (Dykstra–Robertson with increments, D4.3),
   `surface_bootstrap`, `wald_mixture_band`, `ht_reference_surface`.
   - `fit_segment` branches to them when `index_mode == "surface"`.
   - The returned dict carries the 2-D `lookup`, the axis `edges`, `index = None`, and
     `index_mode = "surface"`.
   - `constancy_check` and the Kish ESS stay as they are.
8. `cross_fit_predictions`: every mode refits everything estimated from gold per fold,
   and **predicts held-out gold through the mode's published lookup**: the 40-bin table
   for index modes, the cells for the surface. The lookup is built per fold on the
   production population. Scoring the deployed map, rather than the grid interpolation
   used today, is what makes 40-bin and 24-cell modes comparable. Keep the current
   grid-interpolated score as a secondary column for continuity with July.
9. `curve_metadata`: record the index params or the surface edges, the knobs, the
   `score_definition` (e.g. `"surface_cells(osm_conf_mean, overture_confidence)"`), the
   bound-active flags, and the realized effective parameter count.
10. `plot_calibration.curve_index_for_rows` currently assumes pool mode
    (`plot_calibration.py:97-101`); route it through `index_score`.

### Phase 2 — bands and coverage

11. Implement D5 (surface percentile + Wald-mixture bands).
12. Implement D6 (`simulate_band_coverage.py`). Time a single fit, report the projected
    runtime to Nat if > 2 h, then launch backgrounded.

### Phase 3 — evaluation on the 20260730 scoreset (no new validation)

13. Generalize `compare_matched_index.py` to N modes. Its two-way logic is hard-coded
    today (`_verdict`, `paired_bootstrap(pool, average)`, `deployed_impact`, headings).
    - `MODES = ("pool", "average", "additive", "interaction", "surface")`, plus
      `surface_8x6` as a sensitivity mode.
    - Report every mode's Brier, log score, MCB, DSC and UNC.
    - Paired differences for the ladder pairs: additive−pool, interaction−pool,
      surface−additive, surface−interaction, interaction−additive, surface−pool, and
      pool−average for continuity.
    - Add a `--seed` flag. Run 5-fold at three seeds plus a 10-fold run.
    - Fix the stale docstring on the paired-bootstrap strata: the code strata by fold
      only.
14. **Deployed impact against the deployed curves.** Read the 20260730/20260902
    deployed matched curve and metadata (not a refit). Exclude shadow-matched rows,
    matching `population_by_segment`. Per mode, report:
    - the share of matched POIs moving by more than 0.05;
    - the count crossing a published band edge;
    - the mean absolute shift, split by the 4×4 region.

    Stream the columns only; never load the conflated parquet whole (24 GB WSL cap).
15. Shape diagnostics (`plot_calibration.py` or a small new script):
    - per-mode heatmaps on common 6×4 cells;
    - difference maps against the pool;
    - 1-D slices at each Overture bin with bands;
    - the **independence surface** `logit p̂_osm + logit p̂_ov − logit(base rate)` from
      the two 1-D segment curves (Genest & Schervish, in Genest & Zidek p.9), whose
      departures show conditional dependence directly;
    - the fitted `additive` components h_osm and h_ov;
    - the `interaction` model's a3 with its interval.
16. **Monotonicity diagnostic per axis**, as a standing per-round check: the atom-aware
    bin table with DE, HT, SE and reversal z-scores from `H_code_data.md` §10. Add it to
    `fit_report.md`. Oliva-Aviles et al. (2019)'s CIC is the formal version; note it and
    its low power with many small cells.

### Phase 4 — writeup, then STOP for Nat's decision

17. Write a date-stamped methods writeup:
    `~/data/library/writeups/<YYYY-MM-DD>-openpois-matched-segment-modes.md`. `ls` the
    directory first and never overwrite. Offer a house-style `.docx` render. Contents:
    - **Per mode**:
      - conceptual justification with library citations (table above, expanded);
      - assumptions and what each mode can and cannot represent;
      - effective parameter count;
      - behaviour in 3+ dimensions;
      - implementation and maintenance cost.
    - **Validation metrics**:
      - the cross-fit table with paired CIs along the ladder;
      - seed and fold stability;
      - band widths under both band methods;
      - coverage-simulation results by region, **including the shipped 1-D bands**;
      - HT-overlay agreement;
      - deployed impact.
    - **Shape reading**: is there an interaction, of what sign, and where?
    - **Trade-offs and a parsimony-first recommendation**: the simplest mode that beats
      the pool on one proper score with a CI excluding zero, is not worse on the other,
      and has acceptable bands and coverage. Stated as a recommendation, not a decision.
    - **v4 errata**: the §4.2 unbiasedness wording; the §4.5 "externally Bayesian" and
      "dependence discount" claims; "~43% more discrimination" (it is 43% less for the
      average, 74% more for the pool). If the simulation shows the shipped 1-D bands
      under-cover, say so plainly.
    - **Reproducibility**: commands, seeds, git SHA with dirty flag, output paths.

**Stop and hand the writeup to Nat.** If no mode earns its parameters, keep production on
`pool`, now constrained. Document the negative result in `confidence-calibration.md`: a
clean null is a real answer.

### Phase 5 — deploy path (only after Nat decides)

18. `calibration.py`:
    - `curve_index` evaluates any index form from the metadata params through the shared
      `index_score`, so the deploy side cannot drift from the fit.
    - For `surface`: `apply_surface(osm, overture, lookup)` does two `searchsorted`
      calls with `side = "right"`, clamps to the edge cells, and returns a NaN triple on
      a NaN in either score, mirroring `apply_curve`.
    - `index_modes_from_metadata` already defaults old curves to `"pool"`.
    - Edge rules (`shadow_cd`, `missing_conf`, `unnamed_extrapolated`) are untouched.
      The ~25 imputed-0.5 Overture rows ride the chosen map with no gold at 0.5; the
      surface places them in the lowest Overture cell, flagged as today.
19. Config: set `conflation.calibration.matched_index_mode` to the chosen mode, and
    confirm how it flows into `FitConfig` in `fit_calibration.py`.
20. Docs:
    - `.claude/docs/confidence-calibration.md`: segment table, the "why the matched
      index is…" section, and gotchas (the rounding helper, the 2-D lookup schema if
      applicable, the standing monotonicity check, the per-release refit rule unchanged);
    - module docstrings.
21. Rollout follows `skills/conflate-snapshots`. No new validation round is required by
    this change alone. Standing rules still apply: a turnover-model refit forces a new
    round, and curves do not transport across Overture releases.

### Phase 6 — tests (extend `tests/test_calibration.py`)

- **Constrained pool**:
  - reproduces the 20260730 coefficients (0.210 / 0.569) within 1e-4;
  - on synthetic data with a negative true slope, the fitted slope sits at
    `pool_min_coef` and `bound_active` records it;
  - the thin-gold fallback is unchanged.
- **`fit_additive_index`**:
  - components are nondecreasing;
  - recovers a known additive truth on synthetic data;
  - reduces to one flat component when an axis is dead;
  - survives an all-positive block without diverging (the clip is exercised).
- **`fit_interaction_index`**:
  - the constraints hold at the solution;
  - a monotone synthetic truth with a3 < 0 is recovered;
  - the fitted function is monotone on a dense grid over the whole unit square.
- **`project_monotone_2d`**:
  - output is doubly monotone;
  - idempotent on a monotone matrix;
  - equals 1-D weighted PAV when one axis has a single bin;
  - a heavy cell wins a violation against a light one;
  - **matches a generic QP oracle** (`scipy.optimize.minimize`, SLSQP, with explicit
    pairwise constraints) to 1e-8 on random matrices;
  - **omitting the Dykstra increments fails that oracle test** (regression guard).
- **Cell difference estimator**: with a saturated working model it equals the per-cell
  HT (Hájek) estimate; on synthetic data whose truth depends on one score, the dead
  axis's cells are flat within noise.
- **Surface edges**: the Overture spikes become their own bins after rounding; a cell
  with zero phase-1 rows raises.
- **`apply_surface`**: clamping on each axis; NaN in either score gives a NaN triple;
  exact cell choice at edges matching `apply_curve`'s side convention.
- **Wald-mixture band**: bounds are doubly monotone and bracket the point estimate.
- **Deploy roundtrip**: `calibrate_frame` with each mode's curve and metadata is
  monotone in each input on a small frame, with edge-rule flags unchanged.
- **CORP sign pinning**: keep the existing regression test intact, and add a pinned
  sign test for the N-way `_verdict`.

### Verification (end-to-end, for the executing agent)

```bash
conda activate openpois
pytest tests/test_calibration.py -q                  # all green, incl. new tests
OUT=~/data/openpois/conflation/20260902/calibration_eval_$(date +%Y%m%d)
python -u scripts/conflation/compare_matched_index.py --folds 5 --reps 400 --seed 1 \
    --out-dir "$OUT" --with-deployed-impact > ~/data/openpois/logs/compare_modes.log 2>&1
python -u scripts/conflation/simulate_band_coverage.py --sims 200 --reps 200 \
    --out-dir "$OUT" > ~/data/openpois/logs/band_coverage.log 2>&1
# Inspect: matched_index_comparison.md, coverage tables, heatmaps/slices, fit_report.md
```

Sanity anchors:
- matched n_rows = 2,500 and n_gold = 444;
- the constrained pool reproduces the July coefficients and the July table (pool Brier
  0.07759, average 0.07980, paired CIs as published);
- 6×4 surface cells all have ≥ 6 gold;
- the deployed dir's mtime is unchanged after every run.

If a sanity anchor fails, the harness changed: stop and diagnose before trusting any new
number.

## Literature grounding (all in the library; keys under `~/data/library/papers/`)

- **Algorithms**:
  - Dykstra & Robertson 1982 (ATZCTFHH): the row/column algorithm, d ≥ 2;
  - Dykstra 1983 (Y3GCGY9Q): convergence with correction increments;
  - Spouge, Wan & Wilbur 2003 (YB7IZFJR): exact 2-D, O(n²) on a grid;
  - Stout 2015 (SNII29IQ): d ≥ 2 complexity;
  - Brunk 1955 (3385U6I6): LS = binomial MLE only with count weights.
- **Rates**:
  - Chatterjee, Guntuboyina & Sen 2018 (LU9E6STB): d = 2, n^-1/2 worst case; adapts to
    rectangular level sets and to one-variable truths;
  - Han, Wang, Chatterjee & Samworth 2019 (UR2D6USP): d ≥ 3 loses adaptivity;
  - Guntuboyina & Sen 2018 (F4UXMRXK), review.
- **Design-based order constraints**:
  - Wu, Meyer & Opsomer 2016 (5TWGWCKS);
  - Oliva-Aviles, Meyer & Opsomer 2019 (NQGMZD7J, CIC check) and 2020 (4KERXHPD,
    two-way orderings);
  - Xu, Meyer & Opsomer 2021 (IBIK7RG9, mixture covariance);
  - Liao, Meyer & Xu 2024 (T8RTB3XY, projected bounds, small and empty domains);
  - Liao & Meyer 2026 (BUD7HSX7, csurvey `incr(x1)*incr(x2)`, logit family);
  - Breidt & Opsomer 2017 (NN7VFFNZ).
- **Additive / index**:
  - Bacchetti 1989 (KTDADWH5);
  - Mammen & Yu 2007 (4PFRGLEJ);
  - Chen & Samworth 2016 (YG86BUXR, sign-constrained index loadings);
  - Pya & Wood 2015 (BQ9DMLKA, doubly monotone tensor smooths; point-estimate
    benchmark only);
  - Meyer 2018 (QW4BNVM9).
- **Calibration**:
  - Gupta et al. 2016 (LU9U52MZ, monotone bilinear constraints; Lemma 1);
  - Kumar, Liang & Ma 2019 (C6722LDV: scaling-binning bins *output* values, so it
    applies to the 1-D index modes, not to 2-D cells);
  - Dimitriadis, Gneiting & Jordan 2021 (Z8WUAVS4);
  - Henzi, Ziegel & Gneiting 2021 (WTQPWIIY, IDR on partial orders; the componentwise
    order weakens with d and with uncorrelated covariates);
  - Wang & Liu 2020 (GW9BQKLG, doubly monotone bivariate score calibration at
    n ≈ 400–500; the closest prior art).
- **Pooling**:
  - Genest & Zidek 1986 (W7HWAIXY);
  - Ranjan & Gneiting 2010 (TV4P46NN, beta-transformed linear pool: single-index, so it
    cannot test interaction; not added as a mode);
  - Gneiting & Ranjan 2013 (2SIHL26B).
- **Bootstrap**:
  - Andrews 2000 (PZH68UD3);
  - Kosorok 2008 (E878P93C);
  - Sen, Banerjee & Woodroofe 2010 (4EZDV8X5);
  - Deng, Han & Zhang 2021 (R7C6CPYG: pivotal CIs, which need iid homoscedastic
    noise; a half-width sanity check only).

Considered and set aside:
- **Monotone GBM / mBART / deep lattice networks**: heavier machinery, not
  design-weighted, and no gain at n = 444.
- **Kernel-grid surface**: superseded by cells. It would need weight floors, atom
  handling and a separate deploy representation.
- **SCAM**: no design-based inference, and its default binomial intervals under-cover
  (Meyer 2018 p.13).
- **Copula pooling**: overfitting risk at this n (Ranjan & Gneiting 2010 p.17).

## Open questions for Nat

1. **Published granularity if `surface` wins**: 24 values (6×4) versus 40 bins today. Is
   that acceptable on the site and in the PMTiles, or would a win need 8×6 to be
   deployable?
2. **3+ dimensions**: the writeup will recommend additive components for new continuous
   axes (δ channel) and a coarse categorical axis where needed. Should a category axis
   be scoped now, as a follow-on plan, or wait for a round with more gold?
