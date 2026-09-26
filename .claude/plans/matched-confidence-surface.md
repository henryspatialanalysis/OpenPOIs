# Matched-segment confidence: multivariate CORP

> **SUPERSEDED 2026-09-25** by `matched-2d-calibration-surface.md` (revised after the
> literature audit in `~/data/library/_staging/audit_2026-09-25_matched-2d-surface/`).
> Kept for reference only; do not execute.

**Status 2026-09-25:** research-phase draft for Nat's line-by-line review. Nothing
built yet. Test data: validation round `20260730` (no new validation). Target
release: the October 2026 monthly update.

Design source: `~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md`
(§4.1–4.5), `.claude/docs/confidence-calibration.md`, and
`src/openpois/conflation/calibration_fit.py`.

---

## 1. Where we are now

The request describes the current matched-segment method as averaging the two
source scores. It isn't, quite, and the difference shapes this plan:

- **Production is `matched_index_mode: pool`**, not `average`. The two scores go
  into a fitted log-odds pool,
  `z = 0.204 + 0.210·logit(s_osm) + 0.569·logit(s_ov)`, estimated by
  design-weighted logistic regression on the 444 matched gold rows. The result is a
  single 1-D index, and the ordinary 1-D pipeline then calibrates it:
  kernel difference estimator → PAV → 40 equal-mass bins.
- The average survives only as a comparator. `compare_matched_index.py` showed the
  pool beats it on both proper scores (Brier +0.0021, log +0.0186; both CIs exclude
  zero).
- The deployed map is therefore already a monotone function `f(s_osm, s_ov)`. It is
  restricted to one shape, though: **additive in log-odds**, so its level sets are
  parallel lines in logit space. OSM's effect on the log-odds is the same whatever
  Overture says. The v4 writeup names this as the open limitation (§4.5: "a future
  round with more matched gold could test against an interaction term").

So the real question is whether to replace a *parametric* 2-D monotone map with a
*nonparametric* one. That also sets the bar the proposal has to clear: it must beat
the pool, not the average.

### Preliminary signal (in-sample, exploratory; not a result)

Terciles of each score. `ψ` is the per-row difference-estimator pseudo-outcome
(§2.2), so the cell mean of `ψ` is a design-unbiased estimate of the cell's
existence rate; `pool` is the deployed pool curve averaged over the same rows.

| OSM tercile | Overture tercile | n (phase 1) | ψ mean | pool | residual | naive SE |
|---|---|---|---|---|---|---|
| low  | low  | 358 | 0.780 | 0.816 | −0.036 | 0.022 |
| low  | high | 250 | 0.984 | 0.966 | +0.017 | 0.008 |
| mid  | low  | 231 | 0.823 | 0.836 | −0.013 | 0.043 |
| high | low  | 245 | 0.929 | 0.860 | **+0.069** | 0.019 |
| high | mid  | 310 | 0.893 | 0.929 | −0.036 | 0.031 |
| high | high | 278 | 0.982 | 0.978 | +0.005 | 0.007 |

The pattern: when Overture is low, the existence rate climbs steeply with OSM
(0.78 → 0.82 → 0.93); when Overture is high, it sits at ~0.98 whatever OSM says. That
is a **sub-additive, "either source vouching is enough"** shape. The pool can't
represent it, so it averages the OSM slope across Overture levels. As a result it
under-credits OSM exactly where Overture is weak and slightly over-credits it where
both are weak. A design-weighted logistic regression with a logit×logit interaction
term agrees in sign (interaction −0.108; the naive SE of 0.038 treats the HT weights
as frequency weights, so it overstates precision). §4 tests this with cross-fitting
and proper scores rather than in-sample.

The same table also shows the known Overture non-monotonicity leaking into 2-D
(high-OSM row: 0.929 at Overture-low vs 0.893 at Overture-mid, within noise).
Isotonic regression will pool those two cells.

---

## 2. Proposal: multivariate CORP

Short name: **`mcorp`** (a third value of `matched_index_mode`, beside `pool` and
`average`).

### 2.1 What CORP is, and its multivariate analogue

CORP (Dimitriadis, Gneiting & Jordan 2021) is isotonic regression of outcomes on
forecast values, solved by PAV. Its name lists its properties. It is **Consistent**.
It is **Optimally binned**: the PAV blocks are the bins, chosen by the data with no
tuning parameter. It is **Reproducible**: the result is deterministic, with no bin
count or bandwidth to choose. And it is **PAV-based**. Each block's value is the mean
outcome in that block, so every published value reads as "the existence rate among
POIs in this bin", which is the property to keep.

The multivariate extension replaces the total order of 1-D forecasts with the
**coordinatewise partial order** on score vectors: `x ≼ x'` iff `x_k ≤ x'_k` for
every source `k`. It is the weighted least-squares isotonic regression over that
partial order (Robertson, Wright & Dykstra 1988, ch. 1). What carries over from 1-D:

- **Monotone in every input by construction**, since that is the constraint set.
- **Data-chosen bins.** The solution partitions the points into level sets (blocks),
  each a convex set in the partial order, with its value equal to the block's
  weighted mean response. In 2-D the blocks are staircase-shaped regions, not
  rectangles; a rectangular lattice would force a bin shape on the data.
- **No tuning parameter**, and it is deterministic.
- **Optimal for every proper scoring rule at once.** An isotonic regression under
  *any* partial order minimizes every Bregman loss among order-preserving fits, not
  only squared error (Robertson, Wright & Dykstra 1988, Thm 1.5.1). That result is
  what makes 1-D CORP optimal for both the Brier and the log score, and it is
  order-generic. The CORP group's own multivariate version, isotonic distributional
  regression (Henzi, Ziegel & Gneiting 2021), is built on partial orders of
  covariates and reduces to exactly this estimator for a binary outcome. *(Not in the
  library; see §3.)*
- **Reduces exactly to 1-D CORP when d = 1**, which gives a clean regression test.

### 2.2 Folding in the two-phase design

The v4 estimator is a local average of a per-row pseudo-outcome:

  ψ_i = q̂_{c(i)} + 1[i ∈ G] · (y_i − q̂_{c(i)}) / π_{c(i)}

where `c(i)` is the refined LLM-verdict class, `q̂_c` its working-model rate, `G` the
gold set and `π_c` the class's phase-2 inclusion. Under the phase-2 design,
`E[ψ_i | x_i, c_i] = P(Y = 1 | x_i, c_i)` whatever `q̂` is.

Run multivariate CORP with **ψ as the response over all 2,500 phase-1 rows** (unit
weights). Each block's fitted value is then the block mean of ψ, which is *exactly
the v4 difference estimator restricted to that block*. So `mcorp` is the v4
estimator with the data choosing the bins in 2-D, instead of a kernel choosing a
neighbourhood in 1-D. The final values are clipped to [0, 1]; clipping a monotone
function keeps it monotone.

Two facts about round 20260730 keep ψ simple:

- Every score-dependent class (unverifiable) is **censused** (`π = 1`), so for those
  rows `ψ_i = y_i` exactly and the class's working model drops out. The current code
  fits an isotonic `q̂` for that class, but with `π = 1` it has no effect.
- The definitive classes (exists / gone) already use a class constant. So in d-D the
  working model is a class constant for every class, at no cost this round.

**Reference variant, `mcorp_ht`:** multivariate CORP on gold rows only, with
response `y` and HT weights `1/π`. This is the d-D version of the validator's as-built
`stratified_ht_gold_v1` (which is 1-D CORP with design weights), and it plays the
same robustness-overlay role it plays now.

**One caveat specific to ψ:** a gold row in a heavily subsampled class carries a
large residual. For example, an LLM-exists row that turns out gone has
ψ ≈ 0.987 − 0.987 × 9.6 ≈ −8.5. Least-squares isotonic regression is still
well-defined, and the block mean is still the unbiased estimator. But a single such
row can drag its block down or split blocks. The current 1-D code dilutes this with
a kernel; `mcorp` absorbs it by pooling. T2 compares `mcorp` to `mcorp_ht` so we
can see whether this makes `mcorp` less stable, and T4 checks it directly.

### 2.3 Algorithm (per pooled segment; d = 2 for matched)

1. **Tie resolution (the only discretization).** Map each score to population rank
   space, `u_k = F̂_k(s_k)`, and snap to a `G`-point rank grid per axis (`G = 100` for
   d = 2, i.e. 1% population-mass steps; `G = 40` for d = 3). Rows in the same grid
   node are tied, just as tied forecast values are one point in 1-D CORP. With 2,500
   rows this is far finer than the data can resolve. The rank transform is monotone,
   so monotone in `u` means monotone in `s`, and it absorbs Overture's heaping at
   0.9x/0.7x/0.3x.
2. **Aggregate.** For each occupied node, take the total weight and the weighted mean
   response. For `mcorp` that is ψ with unit weights; for `mcorp_ht` it is gold `y`
   with weights `1/π`.
3. **Solve the partial-order isotonic regression on the occupied nodes.** Build the
   dominance DAG among occupied nodes. Its transitive reduction is enough: sweep in
   rank order and keep only immediate successors. Then solve the weighted
   least-squares QP `min Σ w_j (f_j − r_j)²` subject to `f_j ≤ f_k` on each DAG edge.
   - **Primary solver:** OSQP. It's exact to solver tolerance, takes milliseconds at
     ≤ 2,500 variables, and is generic in d because the dominance DAG exists in any
     dimension. It needs `osqp` added to the env (question 2).
   - **No-new-dependency fallback:** Dykstra's cyclic projection on the full
     `G^d` grid. Alternate weighted 1-D PAV along each axis, with Dykstra's
     correction terms (Dykstra 1983; Bril, Dykstra, Pillers & Robertson 1984,
     algorithm AS 206, is exactly this for two variables). Empty nodes get
     weight ε, so they pass order constraints on without pulling the fit. It is
     slower and approximate as ε → 0.
   - **Test oracle:** the fallback and scipy's `minimize` on small random problems.
4. **Extend to every grid node (deploy map).** An isotonic fit is only pinned down at
   observed points. For any other node, use the **lower envelope**
   `f(x) = max{ f̂(p) : p occupied, p ≼ x }`, with nodes that dominate no occupied
   point set to the minimum fitted value. On a product grid this is a running max
   along each axis in turn (`np.maximum.accumulate`), so it is monotone by
   construction and generic in d. This matches 1-D step-function PAV:
   right-continuous, a score takes its block's value until the next block starts.
   *(Alternative: the midpoint of the lower and upper envelopes; question 4.)*
5. **Artifact.** The `G^d` grid of values, addressed by per-axis rank edges stored in
   raw score units. Deploy is a per-axis `searchsorted` followed by
   `ravel_multi_index`, which is the same kind of step lookup as `apply_curve` now.
   The number of distinct published values is the number of CORP blocks, which the
   fit report lists (block, mass, value, band).
6. **Band.** The existing two-phase bootstrap, unchanged in structure: phase 1
   resampled within verdict class, gold within class, and steps 2–4 refit per
   replicate. There is no pool to refit and the axes are fixed population
   quantities, so the band is grid-anchored without the reparameterization problem
   the pool's POI-anchored band exists to solve. Pointwise 2.5/97.5% quantiles are
   then put through the same isotonic step and bracket the point estimate, as
   `summarize_band` does. The caveat is that the naive bootstrap is not
   pointwise-consistent for isotonic estimators, and it fails wherever a constraint
   binds, i.e. inside pooled blocks (Sen, Banerjee & Woodroofe 2010; Andrews 2000).
   That caveat already applies to the 1-D curves. For parity, keep the same bootstrap
   for October. T2 also reports an m-out-of-n variant (m = n^(0.8)) as a check on
   band width. Pivotal intervals for multivariate isotonic regression (Deng, Han &
   Zhang 2021) are the longer-term fix.

### 2.4 Requirements check

| Requirement | How it is met |
|---|---|
| Monotone increasing in each score | It is the constraint set of the fit (step 3). The extension (step 4) is a running max, so it is monotone on the full grid. Verified on the deployed artifact by an adjacency test. Non-decreasing, not strictly increasing, which is the same plateau behaviour as 1-D PAV. |
| Expands to 3+ inputs | **In code, yes:** nothing in steps 1–6 is 2-D-specific; axes are a list, the dominance DAG and the running-max extension work in any d, and OSQP doesn't care about d. **Statistically, only partly** (§3): the worst-case error rate is n^(−1/2) at d = 2 but n^(−1/3) at d = 3, so a full 3-D fit needs roughly 9,000 gold labels to match what 444 buy at d = 2. A third axis therefore needs either much more gold or extra structure (additive in one axis, or a coarse axis). |
| Confidence = P(exists), histogram-binning spirit | It *is* CORP, generalized from a total order to a partial order. The bins are the PAV blocks, and each block's value is the design-unbiased existence rate among the POIs in it. |

### 2.5 Alternatives considered

*(§3 adds the citations.)*

- **Kernel surface + projection** (my first draft): a d-D Nadaraya–Watson smooth of ψ,
  projected onto the monotone cone, then binned to an equal-mass lattice. It is the
  literal d-D version of today's 1-D pipeline. But it adds a bandwidth and a lattice
  resolution, and its cells are imposed rather than data-chosen, which is the part
  of CORP worth keeping. It could come back as a T2 comparator if `mcorp` proves too
  noisy.
- **Pool + interaction term.** One extra parameter, but monotonicity is no longer
  guaranteed, because the sign of ∂z/∂s_osm then depends on s_ov. Kept only as a T3
  diagnostic.
- **Additive monotone model in log-odds** (monotone GAM / additive isotonic).
  It generalizes the pool's per-axis shapes but keeps additivity, which is the
  structure §1 argues against.
- **Monotone lattice / calibrated interpolated LUT** (TensorFlow Lattice family).
  Smooth output, but it needs an optimizer with per-input calibrators, and its
  interpolated values aren't block existence rates.
- **Monotone GBM / monotone BART.** Black-box, harder to make design-based, and
  harder to explain.

---

## 3. Literature

Scan run 2026-09-25: the local library first, then the web. I checked DOIs against
Crossref. **[LIB]** means the full text is in `~/data/library/papers/`; everything
else is web-only. The core multivariate-isotonic and survey-constrained literature
is **not in the library** (question 10).

**Multivariate CORP has a direct precedent.** Henzi, Ziegel & Gneiting (2021),
*Isotonic distributional regression*, JRSSB 83(5), 10.1111/rssb.12450, is from the
CORP group. It generalizes isotonic regression to **partial orders on multivariate
covariates**, and for a binary outcome it reduces to the partial-order isotonic
regression of §2. The closest ML analogue is Wang & Liu (2020), *Multivariate
probability calibration with isotonic Bernstein polynomials*, IJCAI,
10.24963/ijcai.2020/353. It fits coordinatewise-monotone calibration of several
classifier scores, benchmarked against multivariate isotonic regression.

**Algorithms.** Robertson, Wright & Dykstra (1988), *Order Restricted Statistical
Inference*: weighted isotonic regression on any partial order, the block/min-max
characterization, and simultaneous optimality across Bregman losses. Dykstra &
Robertson (1982), Ann. Stat., 10.1214/aos/1176345866: alternating row/column PAV
converges to the bivariate LSE on a grid (our fallback solver). Stout (2015),
Algorithmica, 10.1007/s00453-013-9814-z: scattered d-D points by reduction to a
sparse DAG, which is our primary construction. Kyng, Rao & Sachdeva (2015),
arXiv:1507.00710: near-linear isotonic regression on arbitrary DAGs. At n = 2,500,
computation is not a constraint.

**Statistical cost of dimension, which is what bounds requirement 2.**
- Han, Wang, Chatterjee & Samworth (2019), Ann. Stat., 10.1214/18-AOS1753: the
  minimax risk of the isotonic LSE is n^(−min{2/(d+2), 1/d}), i.e. n^(−2/3), n^(−1/2)
  and n^(−1/3) for d = 1, 2, 3. On 444 gold the worst-case d = 2 error is ~2.8× the
  1-D error, and d = 3 needs ~9,000 labels to match d = 2 at 444.
- Chatterjee, Guntuboyina & Sen (2018), Bernoulli, 10.3150/16-BEJ865: the 2-D LSE is
  nearly parametric (k/n) when the truth is constant on k rectangles. The §1
  pattern ("either source vouching is enough") is close to that shape, which is the
  optimistic case for d = 2.
- Deng & Zhang (2020), Ann. Stat., 10.1214/20-AOS1947: the estimator adapts to
  irrelevant inputs. If OSM turned out flat, we would pay only the 1-D rate.

These are worst-case rates on gold alone. The phase-1 rows reduce variance through
ψ, but they are not extra labels. T2 and T4 measure the real cost on our data.

**Survey designs.** The Opsomer/Meyer line gives design-based theory for
order-constrained domain means:
- Wu, Meyer & Opsomer (2016), Can. J. Stat., 10.1002/cjs.11301, for total orders;
- Oliva-Avilés, Meyer & Opsomer (2020), *Survey Methodology* 46(2), which extends it
  to **partial orders**: design-consistent and asymptotically normal when the
  constraints hold in the population;
- Liao, Meyer & Xu (2024), *Survey Methodology* 50(2), for grid orders with sparse
  and empty domains;
- the R package `csurvey` (Liao & Meyer 2025, R Journal, 10.32614/RJ-2025-032).

The caveat is that these use single-phase Hájek domain means. Using our two-phase
difference estimator (ψ) as the input is an extension we would have to argue, by
design-consistency plus asymptotic normality of the block means. It is not a
published result.

**Inference.** Sen, Banerjee & Woodroofe (2010), Ann. Stat., 10.1214/09-AOS777, and
Kosorok (2008) show the naive bootstrap is inconsistent under cube-root asymptotics.
Andrews (2000), Econometrica, 10.1111/1468-0262.00114, shows it fails at binding
constraints even in finite dimensions. Deng, Han & Zhang (2021), Ann. Stat.,
10.1214/20-AOS2025, give pivotal pointwise CIs for multivariate isotonic regression.
Xu, Meyer & Opsomer (2021), JSPI, 10.1016/j.jspi.2021.02.004, give
mixture-covariance variances for constrained survey estimators. These bear on step 6.

**Alternatives (§2.5).**
- Pya & Wood (2015), Stat. Comput., 10.1007/s11222-013-9448-7 (`scam`): a doubly
  monotone tensor-product smooth with logit link, the smooth 2-D competitor.
- Chen & Samworth (2016), JRSSB, 10.1111/rssb.12137, and Mammen & Yu (2007): additive
  monotone models with no curse of dimensionality, which makes them the literature's
  preferred route to d ≥ 3.
- Gupta et al. (2016), JMLR 17(109), arXiv:1505.06378: monotonic calibrated
  interpolated LUTs.
- Ranjan & Gneiting (2010), JRSSB, 10.1111/j.1467-9868.2009.00726.x: the
  beta-transformed linear pool.
- Kull, Silva Filho & Flach (2017), EJS, 10.1214/17-EJS1338SI: beta calibration, a
  richer monotone parametric per-source transform than logit.
- Luss, Rosset & Shahar (2012), AoAS, 10.1214/11-AOAS504: isotonic recursive
  partitioning, whose early stopping is a natural regularizer if the full LSE
  overfits.
- Chipman et al. (2022), Bayesian Anal., 10.1214/21-BA1259: monotone BART.

**Already in the library and already cited in v3/v4:** Dimitriadis, Gneiting &
Jordan (2021) [LIB]; Kumar, Liang & Ma (2019) [LIB]; Zadrozny & Elkan (2002) [LIB];
Bordley (1982) [LIB]; Clemen & Winkler (1999) [LIB]; Satopää et al. (2014) [LIB];
Breidt & Opsomer (2017) [LIB].

**What this means for the design.** The literature supports multivariate CORP at
d = 2 on our data, with Henzi et al. as the conceptual anchor and Oliva-Avilés et
al. as the design-based anchor. The pool stays as the comparator. For d = 3, the
full partial-order fit is data-hungry, and additive-monotone or coarse-axis
structure is the realistic route. That argues for keeping the **code** generic in d
while treating any real third axis as a separate decision with its own
power check.

---

## 4. Test plan on round 20260730 (no new validation)

All tests use the existing handoff (`data/calibration/20260730/`): matched segment,
2,500 phase-1 rows, 444 gold. Production population:
`conflation/20260902/conflated_cd.parquet` (1,817,729 matched rows, streamed and
column-scoped).

### T1 — Implementation correctness (unit tests, `tests/test_calibration.py`)

1. **d = 1 reduction:** `mcorp_ht` with d = 1 equals `ht_reference_curve`, i.e.
   weighted PAV on gold, on the real osm and overture segments. `mcorp` with d = 1
   equals sklearn PAV on ψ.
2. The OSQP solution equals the Dykstra fallback and scipy `minimize` on random
   2-D (30 points) and 3-D (20 points) problems, to 1e-6. It is idempotent on
   already-monotone input.
3. Block values equal the weighted mean response of the block (the property that
   makes each block a difference estimator).
4. The deployed grid is non-decreasing along every axis (adjacency check), for d = 2
   and a synthetic d = 3.
5. Universality check: on a random 2-D problem with binary y, no monotone
   perturbation of the fit lowers the log score. This is a numerical guard on the
   Thm 1.5.1 claim.
6. Deploy: `apply_mcorp` is a step lookup, NaN in either score yields NaN, and
   out-of-range scores clamp to the edge nodes.
7. `calibrate_frame` dispatches on `index_mode` and still deploys legacy 1-D pool
   curves unchanged (needed because the drift-gate policy copies prior curves).

### T2 — Cross-fit predictive comparison (the decision test)

Extend `scripts/conflation/compare_matched_index.py` from 2 modes to 4:
`pool` (reference), `average`, `mcorp`, `mcorp_ht`. Same harness as the
pool-vs-average study: 5-fold cross-fit within refined class, **whole pipeline refit
per fold** (for `mcorp`: ψ, the isotonic fit and the extension from training gold
only; held-out gold rows stay in phase 1 as non-gold), held-out gold scored with
design weights. Metrics: Brier, log score, CORP MCB/DSC/UNC, debiased calibration
error, and number of blocks. Paired bootstrap of `candidate − pool` (400 reps).
Repeat over the same six fold/seed settings used before.

Side comparison, informational only: 1-D `mcorp` (PAV on ψ) vs today's kernel
composite on the osm and overture segments. It shows whether the kernel is earning
its keep, which bears on question 6.

### T3 — Does the pool leave structure that `mcorp` removes?

Out-of-fold version of the §1 table: from the T2 cross-fit, the 3×3 (tercile)
design-weighted mean of `ψ − prediction` for `pool` and for `mcorp`, with bootstrap
SEs. Also the pool+interaction logistic coefficient with a bootstrap SE. This is the
direct test of the additivity assumption the v4 writeup flagged.

### T4 — Power and stability (simulation, small)

T2 may come back "no difference". T4 checks whether 444 gold can detect a difference
of the size §1 suggests, and how much the large-ψ rows destabilize `mcorp`. It keeps
the real design (real score pairs, classes, π, LLM-verdict structure) and simulates
`y` under two truths:

- (a) the fitted pool as truth, which measures `mcorp`'s efficiency cost when the
  pool is right;
- (b) a monotone non-additive truth fitted to the §1 pattern, which measures the
  pool's bias and whether T2 would detect it.

200 simulations each. It reports Brier regret against the truth for `pool`, `mcorp`
and `mcorp_ht`, and the spread of block counts. This decides how to read a null T2.

### T5 — Deployed impact on production

Fit `pool` and `mcorp` on full gold and apply both to the 1.82M production matched
POIs. Report:

- mean / p95 / max |Δ conf_mean|, and the share with |Δ| > 0.05;
- POIs crossing the published band edges 0.3 / 0.7 / 0.9;
- median band width;
- mean shift per label, in the style of `shift_by_label.csv`;
- heatmaps of both maps and their difference over (s_osm, s_ov), with the CORP
  block boundaries drawn on the `mcorp` map;
- a slice plot of conf vs s_osm at Overture deciles, which shows the interaction
  directly.

### T6 — Report

Everything above goes to `calibration/matched_mcorp_comparison.md` beside the
existing `matched_index_comparison.md`, plus figures from `plot_calibration.py`.

### Decision rule (pre-registered; for Nat to confirm)

Adopt `mcorp` for the October release if **all** hold (T2, over the six settings):

1. **Not measurably worse:** the upper 95% CI of Δ(mcorp − pool) is below
   **+0.001 Brier** and **+0.009 log score**. These margins are half the pool-vs-average
   effect, i.e. half of what we previously treated as a meaningful loss.
2. **At least as discriminating:** DSC(mcorp) ≥ DSC(pool) as a point estimate, and
   Δlog score < 0 as a point estimate in ≥ 5 of 6 settings.
3. **Band not much wider:** median band width ≤ 1.25 × the pool's (≤ ~0.225).
4. **Invariants pass:** T1 green; deployed grid monotone; T5 shows no block with an
   implausible value (outside the segment's gold existence-rate range); block count
   stable across folds (T4).

If (1) fails, don't adopt, and document it. If (1)–(4) pass but T2's CIs all span
zero, adopt only if T3 shows the pool's out-of-fold residual structure is real
(|residual| > 2 bootstrap SE in any cell) *and* T4(a) shows `mcorp`'s efficiency cost
is small. That is the case where it removes a documented bias at no measured cost.
Otherwise keep the pool and record the result.

Fallbacks, tried in this order and only if the primary fails on instability (block
count or band width), not on predictive loss:

1. `mcorp_ht`. It is the same CORP without the LLM archive's efficiency gain.
2. A coarse-tie-grid `mcorp` with `G = 8` per axis. This regularizes toward
   rectangle-piecewise-constant fits, the near-parametric case in Chatterjee et al.
   2018.

Each fallback is run through the same T2 harness and the same rule. The order is
fixed now to limit forking paths.

---

## 5. Implementation

### Code

- New module `src/openpois/conflation/multivariate_isotonic.py`, generic and with no
  calibration knowledge:
  - `rank_grid(population, columns, G)` → per-axis rank edges in raw units;
  - `snap(points, edges)`;
  - `dominance_edges(nodes)` → the transitive reduction of the product order;
  - `isotonic_partial_order(values, weights, edges, solver = "osqp" | "dykstra")`;
  - `lower_envelope(grid_values, occupied)`;
  - `blocks(fit)`.
- `src/openpois/conflation/calibration_fit.py`
  - `pseudo_outcomes(rows, classes, inclusion)` → ψ (§2.2).
  - `fit_segment`, `two_phase_bootstrap` and `cross_fit_predictions` branch on
    `index_mode in ("mcorp", "mcorp_ht")`; `needs_pool` stays false for them.
  - `ESTIMATOR_TAG` → `composite_model_assisted_v2_mcorp` for these curves (1-D
    curves keep v1).
- `src/openpois/conflation/calibration.py`
  - `apply_mcorp(score_matrix, grid, axes)`.
  - `curve_index` / `calibrate_frame` dispatch on the matched curve's `index_mode`.
  - `read_curves` accepts either schema.
- Curve artifact `matched_curve.parquet`, grid schema: `segment`, `cell`, `i_osm`,
  `i_overture`, `conf_mean`, `conf_lower`, `conf_upper`, `block`. Metadata adds
  `index_mode`, `axes`, per-axis edges, `G`, solver, solver residual and a block
  table. Generic in d via `i_<axis>` columns.
- `scripts/conflation/compare_matched_index.py`: four modes, plus the T3 and T5
  outputs. T4 gets its own `simulate_matched_index.py`.
- `scripts/conflation/fit_calibration.py`: fit-report section with the block table,
  solver diagnostics and `mcorp` vs `mcorp_ht` per block.
- `scripts/conflation/plot_calibration.py`: heatmap with block outlines, band-width
  heatmap, slice plot.
- `config.yaml` (`conflation.calibration`): `matched_index_mode: mcorp` (only if
  adopted), `mcorp_grid_points: 100`, `mcorp_solver: osqp`.
- `environment.yml`: `osqp`, via `make export_env`, if question 2 is a yes.

### Docs (on adoption)

- `.claude/docs/confidence-calibration.md`: index table, "why mcorp not pool", the
  new comparison table.
- `docs/api.rst`: paragraph plus the new module.
- `CHANGELOG.md`.
- The about page's band shares: `site/public/about.html` quotes per-band POI shares
  that will move.
- A `.claude/CLAUDE.md` gotcha if the reuse policy changes (below).
- Optionally, a short v4 addendum in `~/data/library/writeups/`; if wanted, it goes
  through the `academic-humanizer` skill. The addendum should also fix the stale
  sentence in v4 §8 that says the matched segment is calibrated "through the
  legacy blended score". That contradicts §4.5, and it was already out of date
  under the pool.

---

## 6. Rollout to the October monthly update

1. Build and test on a new branch from `main` (the current branch,
   `feature/ghost-closure-evidence`, is waiting on its own precision gate).
2. Run T2–T5; review `matched_mcorp_comparison.md` against the decision rule.
3. If adopted, set `matched_index_mode: mcorp`. In the October conflation run,
   `make calibrate` must **fit**, not copy. The September curves were copied from
   July under the drift-gate reuse rule, and a copied curve can't change the index
   mode. The labels are still round 20260730, so `versions.calibration` doesn't move.
4. `verify-pipeline-run` checks: matched mean conf_mean shift in line with T5;
   `calibration_flag` counts unchanged; grid monotone; band-edge crossings ≈ T5.
5. Future months: the drift-gate reuse rule applies unchanged, since copying a grid
   artifact works the same as copying a curve (T1.7).

---

## 7. Open questions for review

1. **Decision rule margins.** Are +0.001 Brier / +0.009 log (half the pool-vs-average
   effect) the right non-inferiority margins, and is the "null T2 but real T3
   residual" adoption path acceptable?
2. **Solver dependency.** Add `osqp` (conda-forge; exact, fast, generic in d), or stay
   dependency-free with the Dykstra grid solver?
3. **Response.** Is `mcorp` (ψ over all phase-1 rows) the right primary, with
   `mcorp_ht` (gold only, HT weights) as the reference and fallback? It follows v4's
   choice to let the LLM archive carry the score shape.
4. **Off-sample extension.** Lower envelope (conservative, matches 1-D step PAV) or
   the midpoint of the lower and upper envelopes?
5. **October refit scope.** Refit all three segments on the October population, or
   fit only the matched surface and copy osm/overture from July? Recommendation:
   matched only, to isolate the change.
6. **Consistency across segments.** If `mcorp` ships, matched becomes pure CORP-on-ψ
   while osm/overture stay kernel-composite. Accept that for October, and decide on
   1-D alignment from the T2 side comparison later?
7. **Scope of "3+ dimensions".** Do you have a concrete third axis in mind: a third
   source; the change-detection ghost evidence the v4 writeup calls the "natural
   next-round unification"; OSM edit recency? Category, the biggest known weakness,
   is nominal, not ordinal. It would enter as a stratifier (separate fits per group),
   not a monotone axis, and wants more gold than round 20260730 has.
8. **Branching and git.** OK to create `feature/matched-confidence-surface` from
   `main` for this work?
9. **Writeup.** Want a v4 addendum once results are in, or is the repo doc enough?
10. **Library.** The anchor papers aren't in the library: Henzi et al. 2021,
    Oliva-Avilés et al. 2020, Han et al. 2019 and Chatterjee et al. 2018. Want them
    acquired through the librarian before a writeup cites them? The plan itself
    doesn't depend on it.
