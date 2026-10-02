# Existence-confidence calibration

How the published `conf_mean` becomes a calibrated P(exists and is open), and what
the pipeline stage that does it assumes. Read this before touching
`src/openpois/conflation/calibration*.py`, the `calibrate` Makefile target, or the
`versions.calibration` pin.

**Production method since the October 2026 run:** three Bayesian monotone-spline
models with the fixed-rate mixture label layer (next sections). Design source:
[.claude/plans/bayesian-monotone-calibration.md](../plans/bayesian-monotone-calibration.md)
and its execution log. **Releases 20260730 and 20260902** were calibrated with the v4
estimator (`~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md`),
retired 2026-09-30 and kept below as history. The verification process that produces
the labels lives in the private `openpois-validator` repo.

## What the stage does

`conflated_cd.parquet` (post change detection) → `conflated.parquet` (canonical,
calibrated). Per POI, the raw source score(s) are mapped through the POI's detection
segment's fitted curve:

| segment (`source`) | curve |
|---|---|
| `matched` | a **2-D surface** over (`osm_conf_mean`, `overture_confidence`), monotone non-decreasing in both scores |
| `osm` | a 1-D curve over `osm_conf_mean` (OSM turnover posterior mean) |
| `overture` | a 1-D curve over `overture_confidence` (the provider score; never missing) |

Columns written: `conf_mean` / `conf_lower` / `conf_upper` are **overwritten** with the
calibrated triple (so the PMTiles allowlist, the site, and the published schema need no
changes); `conf_mean_uncalibrated` archives the incoming post-CD value;
`calibration_flag` records the edge rules. `original_conf_mean` (pre-CD, written by
change detection) is untouched.

## The production model: Bayesian fixed-rate mixture (from October 2026)

Decided by Nat on 2026-09-30, first deployed in the October 2026 run.

- **Three separate models**, one per segment: 1-D Overture-only, 1-D OSM-only, 2-D
  matched. Each is a monotone quadratic spline (2-D and doubly monotone for matched)
  on the logit scale, fit by NUTS in JAX / BlackJAX
  (`src/openpois/conflation/calibration_bayes.py`, `ModelSpec(arm = "C",
  label_noise = "fixed_mixture", segments = (...))`).
- **Fixed-rate mixture label layer.** Gold rows enter as Bernoulli on their truth. A
  silver (LLM-only) row enters as log[p·Se + (1−p)(1−Sp)] for an "exists" verdict and
  log[p(1−Se) + (1−p)·Sp] for "gone", with Se and Sp fixed per segment from the gold:
  design-weighted (1/π over the production refined classes, per round), pooled over
  `versions.calibration` and `conflation.calibration.pooled_rounds`, Jeffreys-smoothed
  on the Kish ESS.
- **Why three fits.** With fixed rates no parameter is shared across segments (each has
  its own α, μ, τ and spline field, with independent priors), so the joint posterior
  factorizes and separate fits give the same answer. They run in parallel, and each
  gets its own NUTS step size and mass matrix, which in a joint fit the matched surface
  set for all three. `tests/test_calibration_bayes.py` pins the factorization.
- **Acceptance gate.** Each fit must pass the §5.1 rule (`calibration_bayes.
  passes_acceptance`: R̂ ≤ 1.01, bulk and tail ESS ≥ 400, no divergences, E-BFMI ≥ 0.3,
  no tree-depth saturation, and the same R̂ and ESS on the curve values). If any segment
  fails, `export_bayes_curves.py` writes nothing deployable and the run stops for Nat.
- **Sampler settings** (`conflation.calibration.bayes`): 1,000 warmup and 1,000 samples
  × 4 chains, with a per-segment NUTS `target_accept`. Fit alone, each segment adapts a
  larger step size than the matched surface imposed in the joint fit. On the July
  round at the default 0.80, OSM had 19 divergences and matched 12 (plus R̂ 1.014 and
  tail ESS 311), failing the rule; Overture passed. At 0.99 OSM (4 min) and matched
  (29 min, sharing the CPU) pass with no divergences; 2,000 + 2,000 draws at 0.95
  still left one matched divergence (2026-10-01 tuning on openpois-01).
- **Published values.** `conf_mean` is the posterior mean of P(exists) at the POI's
  score(s); `conf_lower` / `conf_upper` are the pointwise 2.5% and 97.5% posterior
  quantiles. **The matched band under-covers** (about 0.75 of a nominal 95% band in the
  in-family simulation; smoothing bias where the surface climbs to its ceiling), a
  known limitation to fix before November; the 1-D bands cover about 0.95 (Overture)
  and 0.91 (OSM).
- **Artifacts.** The posterior is evaluated on grids (1-D: 2,001 points; matched:
  201 × 201) and written as `calibration/<segment>_curve.parquet` with `lookup: grid`
  metadata. Deploy interpolates: linear in 1-D, bilinear on the surface. Every draw is
  monotone, so the pointwise mean and quantiles are, and interpolation keeps them so.
- **Not used in a validation month before the round.** There is no provisional
  calibration: `make conflate_to_cd` stops after change detection, the validator draws
  from `conflated_cd.parquet`, and `make calibrate` runs once the handoff exists.

## No fixed constants survive

The pre-v4 pipeline shipped three engineering defaults, all now estimated:

- `0.588·OSM + 0.412·Overture` (the matched blend, derived from
  `overture_confidence_weight = 0.7`) → replaced by a **fitted** function of both
  scores: the v4 log-odds pool (July) and interaction index (September tests), and from
  October 2026 the Bayesian 2-D surface. Each can place a doubly-confirmed POI above
  either source's own score, which the linear blend cannot.
- The flat `×0.7` on Overture-only confidence → replaced by the overture segment curve.
- The OSM-only passthrough → replaced by the osm segment curve.

`conflation.overture_confidence_weight` still drives `merge.py`'s *attribute* blending
and the archived `conf_mean_uncalibrated`, so it is not dead — but it no longer
influences the published probability.

## History: the v4 matched interaction index (retired 2026-09-30)

Planned for the October 2026 run and superseded before it by the Bayesian surface; kept
as the record of the v4 matched method (`conflation.calibration.matched_index_mode`).
Design and evidence:
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

## History: the v4 estimator (releases 20260730 and 20260902)

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
- **An Overture score of exactly 0.5 is a real score** (retired `missing_conf`,
  2026-09-30). Releases before October 2026 flagged these rows `missing_conf`, on the
  belief that `merge.py` had imputed 0.5 for a missing confidence. No Overture snapshot
  from 2026-06 to 2026-08 ever had a missing value (0 nulls; 0, 1,178 and 2,767 rows at
  exactly 0.5 in the June, July and August releases), so every 0.5 was Overture's own.
  The ingest now fails on a missing or out-of-range confidence
  (`openpois.io.overture.check_overture_confidence`, run after the download, at
  conflation load and in `merge.py`), and 0.5 rows ride the overture curve unflagged.
  Round 20260730's 4 `overture_missing_conf` handoff rows stay out of the fit, since
  they were drawn under their own design.
- **Unnamed POIs are an extrapolation.** They are excluded from the validation frame
  (the verifier needs a name to search on), and are calibrated through the osm curve
  with `calibration_flag = 'unnamed_extrapolated'`.
- **(v4) The band is computed on the published bins (`band_aggregation: bin`).** The index is refit inside each bootstrap replicate, which moves
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
  both at fit and in `apply_surface`/`calibrate_frame`; the Bayesian grid metadata
  carries `score_decimals: 6`). The validation file splits each
  Overture atom (0.919912, 0.990219) into three float representations that production
  does not have. Curves record `score_decimals: 6`, and deploy rounds **only** when
  that key is present: curves fit before rounding have unrounded edges, and rounding
  a score to 0.919912 would drop it below an edge at 0.9199122190…
- **(v4) Fit bins edge values exactly as deploy serves them.** `build_lookup` assigns rows to
  bins with deploy's `searchsorted(side = "right")` (`lookup_bins`). Before 2026-09 it
  averaged `lo ≤ s ≤ hi`, which counted a score sitting on an interior edge in the
  lower bin while deploy served it from the upper one. 37% of overture-segment rows sit
  on an edge. The overture curve shifted at its first refit after the fix.
- **Standing per-axis monotonicity check.** `fit_report.md` has a "Monotonicity by axis"
  table per segment: atom-aware bins, difference-estimator and HT rates, bootstrap SEs
  and reversal z-scores. Since 2026-09-30 a bin with fewer than 5 gold rows is merged
  into a neighbour first (`calibration_fit.merge_thin_bins`; an atom keeps its own bin
  unless the thin bin has no other neighbour), so every adjacent pair gets a z. Before
  that, thin bins were skipped, and a one-row bin at 0.91967–0.919912 hid the overture
  segment's 0.85–0.92 → 0.9199-atom drop (v4 §4.6) on round 20260730.
- **Calibration error is a binned gap, not a Brier score.** The debiased estimator
  subtracts each bin's own sampling variance (Kumar, Liang & Ma 2019). Subtracting
  per-observation Bernoulli variance instead drives it to exactly zero — a bug that
  shipped in the first 20260730 fit and is now covered by a regression test.
- **Overture's raw confidence is non-monotone in truth** (measured 20260730): the
  design-weighted existence rate is 0.68 below 0.25, dips to 0.50 at 0.50–0.70, and
  only reaches 0.94 above 0.98. The shipped curve is monotone anyway — a published
  score whose ordering inverts the input would break threshold filtering — so
  everything below ~0.70 flattens to a floor (near 0.54 under v4). Expect the Overture curve to
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
    at most 10% of POIs move by |Δ| > 0.1: do **not** refit (`make fit_calibration`).
    Reuse the most recent curves verbatim via
    `apply_calibration.py --curves-dir <prior conflation>/calibration` (copy the curve
    parquets + metadata into the new version's `calibration/` dir with a provenance
    note so the version stays self-contained; grid curves copy the same way), then run
    `ht_review.py` and `make apply_manual_overrides`.
  - **Breach** — any criterion fails: the labels' `overture_score` x-axis can no longer
    be trusted. Re-export a new round from openpois-validator, bump
    `versions.calibration`, and refit before publishing.
  (Reference point: the 2026-08-19.0 release scored RMSE 0.036, bias +0.005,
  share|Δ|>0.1 = 2.0% — a comfortable pass. A turnover-model refit still forces a new
  round regardless, since it moves `osm_conf_mean`.)
  - **Method-change override.** When the calibration *method* changes (the model,
    its label layer, or code that changes the curves), reuse is off even on a pass.
    **The October 2026 run is such a release** (v4 → Bayesian mixture). Later passes
    reuse the October curves as usual.

## Files

| Path | Role |
|---|---|
| [src/openpois/conflation/calibration_bayes.py](../../src/openpois/conflation/calibration_bayes.py) | the production model: monotone splines, fixed-rate mixture layer, segment subsets, NUTS fit, acceptance rule |
| [scripts/conflation/fit_bayes_calibration.py](../../scripts/conflation/fit_bayes_calibration.py) | one fit (`--segments`); draws, diagnostics, curves, deployed-impact preview |
| [scripts/conflation/run_bayes_phase1.sh](../../scripts/conflation/run_bayes_phase1.sh) | `MODE=mixture`: the three production fits in parallel, then the report (`make fit_calibration`) |
| [scripts/conflation/export_bayes_curves.py](../../scripts/conflation/export_bayes_curves.py) | acceptance gate + grid curves, metadata and `fit_report.md` (`make export_calibration`) |
| [src/openpois/conflation/calibration_fit.py](../../src/openpois/conflation/calibration_fit.py) | the v4 estimator library; still used for design weights (`inclusion_by_class`, refined classes), bins and the HT check |
| [src/openpois/conflation/calibration.py](../../src/openpois/conflation/calibration.py) | deploy: grid, surface and step lookups, edge rules, streamed rewrite |
| [scripts/conflation/fit_calibration.py](../../scripts/conflation/fit_calibration.py) | v4 fit driver, **retired** 2026-09-30 |
| [src/openpois/conflation/calibration_ht.py](../../src/openpois/conflation/calibration_ht.py) | design-weighted (HT) check of a deployed map: Hájek rates, bins, flags |
| [scripts/conflation/ht_review.py](../../scripts/conflation/ht_review.py) | HT review PDF; standalone CLI for reused curves |
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
make calibrate            # fit + export + apply + plots + HT review
make fit_calibration      # the three Bayesian fits (MODE=mixture run_bayes_phase1.sh)
make export_calibration   # acceptance gate + grid curves into conflation/<v>/calibration/
make apply_calibration    # deploy only, needs curves
make conflate_to_cd       # conflation through change detection, no calibration
```

The three fits run in parallel (1,000 warmup and 1,000 samples × 4 chains each). The
joint fit took about 50 minutes on the laptop; the separate fits are expected to be
faster, and the first full-length run on openpois-01 times them. Run them there.

Refresh the handoff first when the validation round changes:

```bash
cd ~/repos/openpois-validator && python scripts/08_export_handoff.py
```

Then bump `versions.calibration` in `config.yaml` to the new round.

## Reading the fit report

`~/data/openpois/conflation/<version>/calibration/fit_report.md`, written by
`export_bayes_curves.py`. What to check:

1. **Acceptance per segment**: every item of the §5.1 rule, and the fit time.
2. **Forward rates (Se, Sp)** per segment, with their gold n and ESS, and the rounds
   pooled.
3. **Deployed impact** against the previous release: per-segment mean, mean absolute
   change and share moving by more than 0.05 and 0.10.
4. **The HT review** (`ht_review_<round>.pdf`, next section but one): the model-free
   check of the deployed map.

The fuller diagnostics (PPCs, rate by knot, key parameters, figures) are in the fit
directory, `conflation/<version>/calibration_bayes/` (`fit_report.md` there).

### v4 fit report (history)

What the v4 report showed, for reading the 20260730 and 20260902 releases:

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

## Design-weighted (Horvitz–Thompson) check: a standard output from October 2026

Decided 2026-09-30 (full design in TODO.md); implemented 2026-09-30 in
`src/openpois/conflation/calibration_ht.py` (numbers) and
`scripts/conflation/ht_review.py` (PDF and CLI). Revised the same day to use every
phase-1 row, silver included, with the misclassification correction applied first
(Nat). It is model-free, so it guards any deployed map against bias. The Bayesian
prototype below showed both failure directions it catches: arm A −0.035 and arm C +0.01
on matched.
- **Corrected labels.** Each usable phase-1 row gets ỹ: its gold truth if it is gold,
  else q = P(exists | segment, LLM verdict). For exists and gone, q is
  `calibration_bayes.silver_label_rates` (arm C's q, so the check and the prototype use
  the same correction); unverifiable gets the same quantity from the same helpers. Each
  is the design-weighted gold share (w = 1/π over the production refined classes, per
  round), Jeffreys-smoothed on its Kish ESS. The rates used are printed in the report.
- **Per bin:** r = mean ỹ over the bin's phase-1 rows, SD = √(r(1−r)/n) with n the row
  count (Jeffreys-smoothed r for the SD only when r is 0 or 1), and z = (model − r) /
  SD, where model is the deployed map's mean over the same rows, computed through
  `calibration.calibrate_frame` with the curve metadata. |z| > 1 is flagged and
  |z| > 2 flagged more strongly. The gold-only Hájek rate is kept as a reference column
  and a faint marker; it is not flagged on.
- **Calibration in the large** per segment: the same comparison over all its rows.

Bins: atom-aware bins on the native score for osm and overture (10 base bins); for
matched, a 2-D grid of Overture columns that isolate each atom crossed with OSM
quartiles, plus deciles of the handoff's `raw_score` (the 0.588/0.412 blend, which does
not depend on the deployed index). Every bin is merged until it holds ≥ 20 phase-1 rows
(`merge_thin_bins`; on the matched grid the OSM bins merge within each Overture column).
An atom keeps its own bin unless the stretch beside it is too thin.

Reading the shares: an exact map crosses 1 SD in about 32% of bins and 2 SD in about
5%. The SD ignores the uncertainty in q, so it is somewhat optimistic. In simulation of
the two-phase design with an exact map (accurate definitive verdicts, unverifiables
censused), 34.5% and 4.9% of bins were flagged. The correction assumes q is flat in
score within segment × verdict. That holds closely for the definitive verdicts (the fit
report's constancy check), and unverifiables are censused, so none of them is silver.
If a later round samples unverifiables instead of censusing them, their flat q would
bias the bins, since their true rate rises with score (in simulation, 50% and 14% of
bins flagged under an exact map). The curves were also fit on this gold.

The check never fails a run (Nat, 2026-10-01). `make calibrate` runs `ht_review.py` on
the deployed curves after `apply_calibration`; it writes the report section as
`ht_review_<round>.md` beside the curves, with per-view flag counts, calibration in the
large and the correction rates. (The retired v4 `fit_calibration.py` ran it in-process.) The PDF lands beside the
curves at `conflation/<version>/calibration/ht_review_<round>.pdf`, with the bin table
(row and gold counts, corrected and gold-only rates) as `ht_review_<round>_bins.csv`.
Its pages: a summary with the correction rates and the flagged bins; one reliability
page per 1-D view (corrected rate with ±1 and ±2 SD bars, the gold-only rate as a faint
marker, the deployed curve (grid line, or a step lookup for v4 curves) or, for matched
deciles, the deployed value per row,
and a gold-count strip); the matched heatmap of z with OSM slices at the two Overture
atoms; the bin table. On a reuse month, when the curves are copied rather than fit, run
it on its own:

```bash
python scripts/conflation/ht_review.py [--curves-dir DIR] [--out-dir DIR]
```

which also writes the report section as `ht_review_<round>.md`. First result
(2026-09-30, the reused September curves on round 20260730): osm 0 of 10 bins beyond
1 SD; matched 7 of 20 cells (1 beyond 2 SD) and 4 of 10 deciles (none); overture 10 of
15 beyond 1 SD and 5 beyond 2 SD. The overture misses are the known shape problem: the
monotone floor sits under the < 0.30 bin and above the 0.45–0.85 dip, and the old curve
undershoots the 0.990219 atom (rate 0.936, model 0.827, z −6.3). Calibration in the
large is within 0.003 on every segment.

## Bayesian calibration: how it got here

The Bayesian model was built and validated as a prototype on round 20260730 (Phase 1,
2026-09-27 to 2026-09-30), then made production on 2026-09-30 in its fixed-rate mixture
form (the section above). The Phase 1 record:
- **Model.** Monotone quadratic-spline curves per segment (2-D and doubly monotone for
  matched), fit in JAX / BlackJAX.
- **Data layer.** The preferred "arm C" treats gold rows as labelled and non-gold rows
  as fractional labels. The label is q = P(exists | segment, LLM verdict), the
  design-weighted gold concordance rate, passed in as data.
- **Result.** In 10-fold CV it ties the October production map: pooled relative Brier
  0.996 [0.990, 1.002].
- **Open issue.** Its matched-surface bands under-cover (0.75).


| What | Where |
|---|---|
| Design, equations, literature, results | [.claude/plans/bayesian-monotone-calibration.md](../plans/bayesian-monotone-calibration.md) |
| Execution log, every decision and benchmark | [.claude/plans/bayesian-monotone-calibration-notes.md](../plans/bayesian-monotone-calibration-notes.md) |
| Model code | `src/openpois/conflation/calibration_bayes.py` |
| Scripts | `scripts/conflation/{fit,cv}_bayes_calibration.py`, `simulate_bayes_recovery.py`, `report_bayes_calibration.py`, `run_bayes_phase1.sh`, `bayes_calibration_common.py` |
| Outputs | `~/data/openpois/conflation/20260730/calibration_eval_bayes_20260927/` (`fit_report.md`) |
| Config | `conflation.calibration.bayes` (segments, chains, grids) and `conflation.calibration.pooled_rounds` (earlier validation rounds pooled into the fit and its label rates, each under its own design) |
