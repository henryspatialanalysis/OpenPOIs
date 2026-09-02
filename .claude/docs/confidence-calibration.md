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
| `matched` | **fitted log-odds pool** of `osm_conf_mean` and `overture_confidence` |
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
  `overture_confidence_weight = 0.7`) → replaced by the **fitted** log-odds pool
  `z = b0 + b_osm·logit(s_osm) + b_ov·logit(s_ov)`, coefficients estimated by
  design-weighted logistic regression on matched gold. Unlike the linear blend, the
  pool can place a doubly-confirmed POI above either source's own score.
- The flat `×0.7` on Overture-only confidence → replaced by the overture segment curve.
- The OSM-only passthrough → replaced by the osm segment curve.

`conflation.overture_confidence_weight` still drives `merge.py`'s *attribute* blending
and the archived `conf_mean_uncalibrated`, so it is not dead — but it no longer
influences the published probability.

## Why the matched index is the fitted pool and not the simple average

`conflation.calibration.matched_index_mode` chooses between them, and the choice
is measured rather than assumed — run
`python scripts/conflation/compare_matched_index.py --with-deployed-impact`,
which writes `calibration/matched_index_comparison.md`.

The comparison is a 5-fold cross-fit with **the pool refit inside each training
fold**, so a three-parameter index gets no in-sample advantage over a
parameter-free one. Round 20260730 (444 matched gold rows):

| index | Brier | log score | MCB | DSC | UNC |
|---|---|---|---|---|---|
| `pool` | 0.07759 | 0.27441 | 0.00242 | **0.00741** | 0.08258 |
| `average` | 0.07980 | 0.29348 | 0.00148 | 0.00426 | 0.08258 |

The average is worse on both proper scoring rules with bootstrap intervals
excluding zero (Brier +0.0021 [+0.0002, +0.0041]; log score +0.0186 [+0.0092,
+0.0283]) and loses about 43% of the **discrimination** — which is the component
an index is responsible for, since miscalibration is what the isotonic step
removes anyway. Stable across six fold/seed settings (pool DSC 0.0067–0.0079 vs
average 0.0041–0.0043).

Read `DSC` as better *higher* and `MCB`/Brier/log score as better *lower*. A
comparison that treats all four as lower-is-better inverts the discrimination
verdict; there is a regression test pinning the decomposition's signs.

Two honest counterweights, in case the trade is revisited:

- The average yields a **much tighter band** (median 0.109 vs 0.180), because it
  has no coefficients to refit per bootstrap replicate. Some of that is genuine
  parsimony; some is simply having less discrimination to be uncertain about.
- Both indices discriminate weakly in absolute terms — matched POIs are
  overwhelmingly real, so `DSC` is small next to `UNC` either way. The pool
  explains ~9% of outcome variance against the average's ~5%.

The choice is not cosmetic: switching moves 28.5% of matched POIs by more than
0.05 and sends 569,129 across a published band edge.

## The estimator, in one paragraph

The validation is a two-phase sample: phase 1 is an LLM verdict on every sampled POI
(cheap, noisy), phase 2 is a human gold subsample drawn at known but very unequal rates
*within LLM-verdict class* (20260730: 11.6% of LLM-exists, 36.9% of LLM-gone, 100% of
LLM-unverifiable — a census). The curve is a **model-assisted difference estimator**:
a low-dimensional working model for P(exists | class, score) predicts every phase-1
row, and the design-weighted residuals of the gold rows correct it. That is
design-unbiased whatever the working model does (Breidt & Opsomer 2017), and it is where
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
- **The band is POI-anchored, not grid-anchored.** For the matched segment the pool is
  refit inside each bootstrap replicate, which moves the index scale; comparing
  replicates at a fixed index value would report reparameterization as uncertainty. The
  bootstrap therefore scores the *original* rows under each replicate's map and
  aggregates back onto the point estimate's grid.
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

## Files

| Path | Role |
|---|---|
| [src/openpois/conflation/calibration_fit.py](../../src/openpois/conflation/calibration_fit.py) | the estimator: classes, inclusion, working models, difference estimator, pool, bootstrap, cross-fit |
| [src/openpois/conflation/calibration.py](../../src/openpois/conflation/calibration.py) | deploy: curve index, `apply_curve`, edge rules, streamed rewrite |
| [scripts/conflation/fit_calibration.py](../../scripts/conflation/fit_calibration.py) | fit driver → curves + `fit_report.md` |
| [scripts/conflation/apply_calibration.py](../../scripts/conflation/apply_calibration.py) | apply driver |
| [scripts/conflation/plot_calibration.py](../../scripts/conflation/plot_calibration.py) | diagnostic figures |
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
