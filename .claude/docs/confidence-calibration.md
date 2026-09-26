# Existence-confidence calibration

How the published `conf_mean` becomes a calibrated P(exists and is open), and what
the pipeline stage that does it assumes. Read this before touching
`src/openpois/conflation/calibration*.py`, the `calibrate` Makefile target, or the
`versions.calibration` pin.

**Design source:** `~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md`
(v4; supersedes §9 of the 2026-07-24 v3 review). The verification process that produces
the labels lives in the private `openpois-validator` repo; its
`.claude/docs/divergence-from-v3-writeup.md` records why the v3 design changed.

## What the stage does

`conflated_cd.parquet` (post change detection) → `conflated.parquet` (canonical,
calibrated). Per POI, the raw source score(s) are mapped through the POI's detection
segment's fitted curve:

| segment (`source`) | curve index |
|---|---|
| `matched` | **interaction index**: a monotone bilinear function of the two scores' rescaled logits (parameters under `index` in the curve metadata) |
| `osm` | `osm_conf_mean` (OSM turnover posterior mean) |
| `overture` | `overture_confidence` (post-imputation) |

Columns written: `conf_mean` / `conf_lower` / `conf_upper` are **overwritten** with the
calibrated triple (so the PMTiles allowlist, the site, and the published schema need no
changes); `conf_mean_uncalibrated` archives the incoming post-CD value;
`calibration_flag` records the edge rules. `original_conf_mean` (pre-CD, written by
change detection) is untouched.

## No fixed constants survive

The pre-v4 pipeline shipped three engineering defaults, all now estimated:

- `0.588·OSM + 0.412·Overture` (the matched blend, derived from
  `overture_confidence_weight = 0.7`) → replaced by a **fitted** index of both scores.
  The July 2026 fit used the log-odds pool; since the October 2026 release it is the
  monotone bilinear interaction index (next section). Either can place a
  doubly-confirmed POI above either source's own score, which the linear blend cannot.
- The flat `×0.7` on Overture-only confidence → replaced by the overture segment curve.
- The OSM-only passthrough → replaced by the osm segment curve.

`conflation.overture_confidence_weight` still drives `merge.py`'s *attribute* blending
and the archived `conf_mean_uncalibrated`, so it is not dead — but it no longer
influences the published probability.

## The matched segment: the interaction index

**Production since the October 2026 run** (`conflation.calibration.matched_index_mode:
interaction`). Design and evidence:
`~/data/library/writeups/2026-09-26-openpois-matched-segment-modes.md`; outputs in
`conflation/20260902/calibration_eval_20260925/`.

With `x`, `y` the logits of `osm_conf_mean` and `overture_confidence` rescaled to
[0, 1] over the fixed clip range [logit 1e-4, logit(1 − 1e-4)]:

    η = a0 + a1·x + a2·y + a3·x·y,   a1, a2, a1 + a3, a2 + a3 ≥ pool_min_coef (1e-3)

- **Fit.** Design-weighted Bernoulli pseudo-likelihood on matched gold
  (`fit_interaction_index`, `trust-constr`). The four linear constraints are necessary
  and sufficient for η to be nondecreasing in both scores over the whole score square
  (Gupta et al. 2016). Rescaling over the clip range, not the observed range, means
  the guarantee covers every score deploy can see.
- **Calibration.** `expit(η)` is the index that the ordinary 1-D pipeline calibrates:
  difference estimator → PAV → 40-bin lookup. The index is refit inside every
  bootstrap replicate and cross-fit fold. Only η's level sets reach the published map,
  so the model has 2 effective shape parameters (the pool has 1).
- **Reading a3.** a3 < 0 is a substitutive interaction ("either source suffices").
  Round 20260730 fit a3 = −11.1 (95% [−20.7, +0.3], 96% of bootstrap replicates
  negative) with a1 + a3 on its bound. OSM's log-odds slope falls from 0.27 at
  s_Ov = 0.75 to 0.15 at the 0.990219 atom; the pool holds it at 0.21 everywhere.

**Why it was chosen over the alternatives** (5 modes, 24 cross-fit runs, scored through
each mode's published lookup):
- **vs the constrained pool (production to 2026-09):** it nests the pool (a3 = 0).
  It was better in sign on log score in 20 of 20 extra seeds, and the interval excluded
  zero in 6 of 20 (10–12 of 24 runs with fold × class bootstrap strata). It was never
  measurably worse. The gain is small (≈ −0.0003 Brier, −0.001 log score) and sits in
  the low-Overture column (s_Ov < 0.92, 11% of matched POIs).
- **Deployed impact.** Against the July pool curve it moves 0.4% of matched POIs by
  more than 0.05.
- **Not adopted:**
  - `average` is worse on both scores in every run: Brier +0.003, log +0.021, with
    ~43% less discrimination (the July decision, reconfirmed).
  - The additive isotonic index is unstable at 444 gold (end-knot spiking, collapsed
    bins) and worse on average.
  - The doubly-monotone cell `surface` is worse than `interaction` in all 20 seeds and
    moves 10% of POIs.

All five modes remain implemented (`matched_index_mode` accepts `pool | average |
additive | interaction | surface`). Re-run the comparison when a new validation round
lands:

    python scripts/conflation/compare_matched_index.py --with-deployed-impact \
        --out-dir ~/data/openpois/conflation/<version>/calibration_eval_<date>

It refuses to write into the deployed `calibration/` directory. Read `DSC` as better
*higher* and `MCB`/Brier/log score as better *lower*; a regression test pins the signs.

**Extending to 3+ scores** (not built). The multilinear generalization of η has one
coefficient per subset of axes: 2^D terms, with D·2^(D−1) edge-derivative constraints
that again guarantee monotonicity. That is 8 coefficients and 12 constraints at
D = 3. A cheaper route adds a new axis as a main effect only, with pairwise
interactions where one is suspected. See the writeup's "Extending to three or more
scores" section.

## The estimator, in one paragraph

The validation is a two-phase sample: phase 1 is an LLM verdict on every sampled POI
(cheap, noisy), phase 2 is a human gold subsample drawn at known but very unequal rates
*within LLM-verdict class* (20260730: 11.6% of LLM-exists, 36.9% of LLM-gone, 100% of
LLM-unverifiable — a census). The curve is a **model-assisted difference estimator**:
a low-dimensional working model for P(exists | class, score) predicts every phase-1
row, and the design-weighted residuals of the gold rows correct it. That is exactly
design-unbiased for a working model that does not depend on the sample, and
asymptotically unbiased for ours, which is fit on the same gold (Breidt & Opsomer 2017
pp. 3, 6) — a poor working model costs variance, not consistency. It is where
the LLM archive earns its keep — the score *shape* comes from all 7,504 phase-1 rows,
while gold only pins the class-conditional levels. Rogan–Gladen is **not** applied: the
LLM is a stratifier, not an outcome, so there is no measurement-error model to invert.
Se/Sp are reported as diagnostics only.

## Gotchas

- **Calibration runs after change detection, never before.** CD multiplies `conf_mean`
  by a per-label δ (≈0.14). Calibrating first would leave a calibrated probability
  scaled by δ, which is not a probability of anything. The curves are fit on the
  post-CD frame.
- **Shadow-matched rows are deliberately left uncalibrated** (`calibration_flag =
  'shadow_cd'`, interval NaN). The overture curve is indexed on `overture_confidence`,
  which CD never touches, so running the curve on them would silently discard the
  demotion. They also do not influence the lookup's bin edges.
- **`overture_confidence == 0.5` is ambiguous** in the published data: `merge.py`
  imputes 0.5 for missing Overture confidence, so a stored 0.5 could be either. The
  stratum's own constant was withheld this round (3 gold labels, floor 30), so those
  rows ride the overture curve at 0.5 with `calibration_flag = 'missing_conf'`. Only
  ~1,048 Overture-only and 25 matched rows are affected. Fixing this upstream wants an
  `overture_confidence_imputed` boolean out of `merge.py`.
- **Unnamed POIs are an extrapolation.** They are excluded from the validation frame
  (the verifier needs a name to search on), and are calibrated through the osm curve
  with `calibration_flag = 'unnamed_extrapolated'`.
- **The band is computed on the published bins (`band_aggregation: bin`, since the
  October 2026 run).** The index is refit inside each bootstrap replicate, which moves
  the index scale. Comparing replicates at a fixed index value would therefore report
  reparameterization as uncertainty. Instead, each replicate's own map is applied to
  the production POIs, and the band is the percentile of each replicate's mean over
  the POIs in each published bin. The pre-2026-09 method (`anchored_kernel`)
  kernel-smoothed each replicate a second time onto the grid, which shrank and
  off-centred the band. Simulated coverage of a nominal 95% band was 0.76 (matched),
  0.87 (osm) and 0.46 (overture), against 0.89 / 0.91 / 0.66 with `bin`, at about 10%
  more width. The remaining gap is kernel smoothing bias where the curve bends, worst
  at the Overture atoms. The published bands are therefore still somewhat narrower
  than a true 95%.
- **Scores are rounded to 6 dp before indexing or binning** (`calibration_fit.round_scores`,
  both at fit and in `apply_surface`/`calibrate_frame`). The validation file splits each
  Overture atom (0.919912, 0.990219) into three float representations that production
  does not have. Curves record `score_decimals: 6`, and deploy rounds **only** when
  that key is present: curves fit before rounding have unrounded edges, and rounding
  a score to 0.919912 would drop it below an edge at 0.9199122190…
- **Fit bins edge values exactly as deploy serves them.** `build_lookup` assigns rows to
  bins with deploy's `searchsorted(side = "right")` (`lookup_bins`). Before 2026-09 it
  averaged `lo ≤ s ≤ hi`, which counted a score sitting on an interior edge in the
  lower bin while deploy served it from the upper one. 37% of overture-segment rows sit
  on an edge. The overture curve shifted at its first refit after the fix.
- **Standing per-axis monotonicity check.** `fit_report.md` has a "Monotonicity by axis"
  table per segment: atom-aware bins, difference-estimator and HT rates, bootstrap SEs
  and reversal z-scores. Bins with fewer than 5 gold rows report no z, so a one-row bin
  can hide a real reversal behind it. The overture segment's 0.85–0.92 → 0.9199-atom
  drop (v4 §4.6) is masked this way on round 20260730. Read the DE column, not only
  the z.
- **Calibration error is a binned gap, not a Brier score.** The debiased estimator
  subtracts each bin's own sampling variance (Kumar, Liang & Ma 2019). Subtracting
  per-observation Bernoulli variance instead drives it to exactly zero — a bug that
  shipped in the first 20260730 fit and is now covered by a regression test.
- **Overture's raw confidence is non-monotone in truth** (measured 20260730): the
  design-weighted existence rate is 0.68 below 0.25, dips to 0.50 at 0.50–0.70, and
  only reaches 0.94 above 0.98. The shipped curve is monotone anyway — a published
  score whose ordering inverts the input would break threshold filtering — so
  everything below ~0.70 flattens to a floor near 0.54. Expect the Overture curve to
  look like a floor plus a top-decile rise, and re-check the non-monotonicity each
  release rather than assuming it is stable.
- **The curves condition on score alone, not category** — and that is the biggest
  known weakness. The OSM curve tops out near 0.87 because ~22% of even the
  highest-scoring OSM records are LLM-unverifiable and only ~65% of those are real. The
  ceiling applies to every category equally, so stable institutional labels get pulled
  *down*: `Place of Worship` 0.92 → 0.85, with `School`, `Post Office` and
  `Public Safety` similar (see `calibration/shift_by_label.csv`). Conditioning the
  class-mix term on a coarse category grouping is the obvious next-round improvement.
- **Curves do not transport across releases.** Overture's confidence methodology drifts
  and the OSM turnover model refits monthly. `versions.calibration` pins the validation
  round. Whether an Overture release bump forces a **new validation round** is decided
  by the monthly drift check, `scripts/overture/compare_confidence.py` (matched-GERS-id
  prior-vs-current comparison, in-schema POIs only). **Decision rule (adopted
  2026-09-02):**
  - **Pass** — on matched ids, overall RMSE ≤ 0.10 **and** |mean bias| ≤ 0.03, **and**
    at most 10% of POIs move by |Δ| > 0.1: do **not** re-run `fit_calibration`. Reuse
    the most recent fitted curves verbatim via
    `apply_calibration.py --curves-dir <prior conflation>/calibration` (copy the curve
    parquets + metadata into the new version's `calibration/` dir with a provenance
    note so the version stays self-contained).
  - **Breach** — any criterion fails: the labels' `overture_score` x-axis can no longer
    be trusted. Re-export a new round from openpois-validator, bump
    `versions.calibration`, and refit before publishing.
  (Reference point: the 2026-08-19.0 release scored RMSE 0.036, bias +0.005,
  share|Δ|>0.1 = 2.0% — a comfortable pass. A turnover-model refit still forces a new
  round regardless, since it moves `osm_conf_mean`.)
  - **Method-change override.** When the calibration *method* changes
    (`matched_index_mode`, `band_aggregation`, or estimator code that changes the
    curves), reuse is off even on a pass. Refit with `make fit_calibration` against the
    current `versions.calibration` round. **The October 2026 run is such a release:**
    the prior curves are pool-mode with the old band. Refit once. Later passes reuse
    the October curves as usual.

## Files

| Path | Role |
|---|---|
| [src/openpois/conflation/calibration_fit.py](../../src/openpois/conflation/calibration_fit.py) | the estimator: classes, inclusion, working models, difference estimator, matched indices (pool / additive / interaction) and cell surface, bootstrap, cross-fit |
| [src/openpois/conflation/calibration.py](../../src/openpois/conflation/calibration.py) | deploy: curve index (any index form via `index_score`), `apply_curve` / `apply_surface`, edge rules, streamed rewrite |
| [scripts/conflation/fit_calibration.py](../../scripts/conflation/fit_calibration.py) | fit driver → curves + `fit_report.md` |
| [scripts/conflation/apply_calibration.py](../../scripts/conflation/apply_calibration.py) | apply driver |
| [scripts/conflation/plot_calibration.py](../../scripts/conflation/plot_calibration.py) | diagnostic figures |
| [scripts/conflation/compare_matched_index.py](../../scripts/conflation/compare_matched_index.py) | matched-mode comparison: cross-fit through the published lookup, paired ladder, deployed impact |
| [scripts/conflation/simulate_band_coverage.py](../../scripts/conflation/simulate_band_coverage.py) | band coverage simulation under the two-phase design (`--band-aggregation`) |
| [scripts/conflation/plot_matched_modes.py](../../scripts/conflation/plot_matched_modes.py) | matched-mode shape diagnostics (heatmaps, slices, independence surface, a3) |
| [tests/test_calibration.py](../../tests/test_calibration.py) | estimator identities + edge rules |
| `data/calibration/<round>/` | the validation handoff (**gitignored** — the labels are the moat) |
| `~/data/openpois/conflation/<version>/calibration/` | fitted curves, metadata, fit report |

## Running it

```bash
make calibrate            # fit_calibration + apply_calibration + plots
make fit_calibration      # curves only (safe to iterate)
make apply_calibration    # deploy only, needs curves
```

Refresh the handoff first when the validation round changes:

```bash
cd ~/repos/openpois-validator && python scripts/08_export_handoff.py
```

Then bump `versions.calibration` in `config.yaml` to the new round.

## Reading the fit report

`~/data/openpois/conflation/<version>/calibration/fit_report.md`. What to check:

1. **Kish ESS per segment** — precision follows the design, not the row count. A class
   audited at 1% carries a ~98× weight on few rows.
2. **Composite vs Horvitz-Thompson reference** — the HT curve is the validator's
   as-built gold-only estimator and the saturated special case of the composite. It
   should sit inside the composite band over most of the grid; a systematic gap means
   the working model is wrong.
3. **Constancy check** — the definitive classes' rates are modeled flat in score. A
   large low-vs-high gap in a definitive class argues for the isotonic treatment.
4. **Refined-class table** — confirms phase-2 inclusion is uniform within class and
   shows which (verdict × LLM-confidence) cells survived the `min_cell_gold` floor.
5. **Fitted interaction index** — a0–a3 and which constraints are active. An active
   `a1+a3` means OSM's slope is held at zero at the top of the Overture scale.
6. **Monotonicity by axis** — see the gotcha above. It is expected on the overture
   segment, and should be investigated on matched or osm.
7. **Cross-fit calibration error** — `cross-fit Brier` scores held-out gold through
   the published 40-bin lookup, built per fold. That is higher than the pre-2026-09
   curve-interpolated figure (pool 0.0787 against 0.0776 on 20260730), because
   publication in 40 bins costs some discrimination.
