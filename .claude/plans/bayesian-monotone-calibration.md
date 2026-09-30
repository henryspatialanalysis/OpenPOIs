# Bayesian monotone-spline calibration: F1 / F2 / F3

**Status 2026-10-01: Phase 1 executed; the October changes are wired in, not run.**
- **Preferred method:** arm C (§3.5b) with fixed silver-label rates, M15c'
  (decisions 12 and 17). From October it pools validation rounds (decision 20).
- **October:** fractional labels stay the main model, with q constant in score; the
  fixed-rate mixture runs beside it as a test model (§3.5c; decisions 24, 25).
- **Production:** not wired in; Phase 2 happens only on request.
- **Results:** §11 summarises them; the full report is
  `~/data/openpois/conflation/20260730/calibration_eval_bayes_20260927/fit_report.md`.
- **Execution log:** `bayesian-monotone-calibration-notes.md` holds entries 1–25 with
  their evidence, and the questions still open.

This document and the log together are meant to be enough to reconstruct the method
and its justification in full, for example as an academic paper:
- this document: model, priors, data layer, validation design, literature and
  criticism, results;
- the log: every deviation from the plan, each benchmark, each decision and its reason.

Equations are numbered (M1, M2, …) and every prior sits in one table (§3.7). Citations
are author–year; full references are in §10. Sections 3–8 were written as the plan (with
§3.5b, §3.7 and §5 updated to what was run). §9 lists the decisions, §11 the results and
§12 the open items.

---

## 0. Summary

- **The model.** One Bayesian hierarchical model, fit in JAX (BlackJAX NUTS), with three
  monotone, range-bounded quadratic-spline curves:
  - F1, Overture-only;
  - F2, OSM-only;
  - F3, matched, 2-D. It is doubly monotone through a max recursion that imposes no sign
    on the interaction.

  A first-order random-walk smoothing prior sits on the log-slopes. There is no nugget:
  with one Bernoulli outcome per POI it is not identified (§6.1).
- **Data layer: the preferred arm C** (§3.5b). Every phase-1 row enters once: gold rows
  with the human label, and non-gold rows with the LLM's definitive verdict as a
  fractional label. The label is q = P(exists | segment, verdict), the design-weighted
  gold rate, passed in as data (M15c').
- **Arm A** is the planned joint measurement model of the LLM verdict given truth
  (M12–M16). It is **rejected**:
  - it ran biased low against the design-weighted gold rate (matched 0.871 against
    0.906), whatever its structure (four variants);
  - it failed convergence;
  - its verdict layer could not reproduce how the verdict mix changes with score (PPC
    71–81%).

  §3.5 and §3.6 are kept as the record.
- **Arm B**, a design-weighted gold-only pseudo-likelihood, is a comparator only.
- **Results** (§11), from the first-run arm C with an estimated β of about 1:
  - **10-fold CV against production as published:** pooled relative Brier 0.996 [0.990,
    1.002], a statistical tie. Arm C beats arm B (0.991 [0.986, 0.997]).
  - **Coverage of the 95% band** in simulation: Overture 0.95, OSM 0.91, matched 0.75.
    The matched shortfall is smoothing bias (§12).
  - **Sensitivity:** curves move by 0.006 or less at the validation rows under every
    structural or prior sensitivity.
- **October LLM plan** (§8): up to 1,000 new LLM-checked rows, a census of new
  unverifiables and no drift anchor. The new gold folds into the silver-label rates.

---

## 1. Why rethink, and what the current method is

Production since the October 2026 run (`.claude/docs/confidence-calibration.md`):
- **Matched segment.** The monotone bilinear interaction index
  η = a0 + a1·x + a2·y + a3·x·y is fit on the rescaled logits of both scores (Gupta et
  al., 2016). Its `expit(η)` is then recalibrated through the 1-D pipeline: kernel
  difference estimator (a model-assisted estimator on the two-phase design; Breidt &
  Opsomer, 2017) → PAV → 40-bin lookup.
- **osm and overture segments.** The same 1-D pipeline, run on the native score.
- **Bands.** A two-phase bootstrap aggregated to bins (`band_aggregation: bin`).
  Simulated coverage of a nominal 95% band is 0.89 (matched), 0.91 (osm) and 0.66
  (overture).

The weaknesses this plan targets:
1. **The matched map is two stages with one bilinear shape.** Only η's level sets reach
   the published map (two effective shape parameters). The shape is fixed by the
   bilinear form on rescaled logits, which suits the substitutive structure (a3 ≈ −11)
   but nothing beyond it.
2. **Uncertainty is bootstrap-based and under-covers**, badly for overture.
3. **The estimator is a design-based point estimator**, so it has no generative model to
   check against the data, and no natural way to add structure: category, urbanicity,
   new rounds.

What the new model keeps from v4:
- the two-phase design logic (gold is drawn within LLM-verdict class);
- the use of every phase-1 row, not only gold;
- the output definition, a calibrated P(exists ∧ open);
- monotonicity in each score.

---

## 2. Data facts that shape the design (verified 2026-09-27 on round 20260730)

Source: `data/calibration/20260730/validation_rows.parquet` (7,504 rows), with scores
rounded to 6 dp (`calibration_fit.round_scores`).

| segment | phase-1 rows | gold | gold y-rate | LLM-unverifiable (all gold) |
|---|---|---|---|---|
| overture | 2,500 (+4 `overture_missing_conf`, excluded) | 892 | 0.398 | 563 |
| osm | 2,500 | 1,026 | 0.706 | 731 |
| matched | 2,500 | 444 | 0.752 | 154 |

- **Phase 1** is stratified: segment × ten equal-mass raw-score bins (250 rows per bin;
  overture has 7 bins of 357), with urban/rural balancing (v4 §3). Conditional on the
  score(s), score-bin stratification is ignorable. For matched, `raw_score` is the
  0.588/0.412 blend, a deterministic function of both scores. **Urban/rural balancing is
  ignorable only if existence is independent of urbanicity given score.** The v4
  estimator makes the same assumption; see §6.8.
- **Phase 2 (gold)** depends on the verdict only:
  - all LLM-unverifiable rows (census);
  - a uniform ~37% of LLM-gone (the `enriched` purpose, which in the realized data
    contains only gone verdicts);
  - a uniform ~12% of LLM-exists;
  - a 75-row uniform `random_kappa` slice.

  No realized draw selected on score within class.
- **The LLM's definitive verdicts are near-gold.** Pooled over segments, exists → y = 1
  in 594/603 and gone → y = 0 in 308/311. Unverifiable is the uninformative class, and
  its gold rate differs sharply by segment (osm unverifiable:low 0.68, overture
  unverifiable:low 0.27).
- **Overture atoms.** In matched, 31% of rows sit at 0.919912 and 22% at 0.990219; in the
  overture segment, 14% and 8%. Deciles therefore coincide there.

**Knots under the requested rule** (deciles of the segment's phase-1 scores, deduplicated
at 6 dp, plus the domain ends 0 and 1, and any gap > 0.2 split into equal sub-intervals):

| curve / axis | knots (rounded) | intervals | gold per interval |
|---|---|---|---|
| F1 overture | 0, .175, .35, .55, .708, .798, .85, .9199, .9718, .9902, 1 | 10 | 32 92 93 144 105 34 126 170 42 53 |
| F2 osm | 0, .194, .388, .582, .704, .785, .840, .873, .899, .930, .955, .977, 1 | 12 | 3 24 88 102 88 103 111 125 98 112 95 77 |
| F3 x = osm | 0, .157, .314, .471, .629, .745, .810, .852, .887, .913, .941, .960, .981, 1 | 13 | 0 3 12 41 44 36 48 46 28 42 62 42 40 |
| F3 y = overture | 0, .18, .36, .54, .72, .9002, .919912, .958, .979, .9902, .9943, 1 | 11 | 1 10 21 26 27 6 199 38 20 65 31 |

Consequence: F3 has a 15 × 13 = **195-coefficient** quadratic tensor surface against 444
gold rows and 2,056 silver rows. In the matched low-Overture band (y < 0.90: 5 intervals,
10% of rows, 85 gold), the surface is carried mostly by the smoothing prior and the
monotonicity constraint. This is expected, and §5 checks it explicitly.

---

## 3. The hierarchical model, in full

### 3.1 Notation

| symbol | meaning |
|---|---|
| i = 1…n | phase-1 validation rows, n = 7,500 |
| g(i) ∈ {V, O, M} | detection segment: Overture-only, OSM-only, matched |
| $s^O_i, s^V_i \in [0, 1]$ | OSM score (`osm_score`) and Overture score (`overture_score`), rounded to 6 dp |
| $s_i$ | the score argument of the segment's curve: $s^V_i$ (V), $s^O_i$ (O), $(s^O_i, s^V_i)$ (M) |
| $r_i$ | a 1-D score covariate for the measurement layer: $s^V_i$ (V), $s^O_i$ (O), `raw_score` (M) |
| $v_i \in \mathcal{V}$ | refined LLM class: verdict × confidence, 9 levels, observed for every row |
| $G_i \in \{0, 1\}$ | gold indicator |
| $y_i \in \{0, 1\}$ | human truth (exists ∧ open), observed iff $G_i = 1$ |
| $L, U$ | range bounds: $L = \text{logit}(10^{-3}) = -6.9068$, $U = -L$ |

### 3.2 Level 1: existence

$$y_i \mid \eta_i \sim \text{Bernoulli}\big(\text{expit}(\eta_i)\big) \tag{M1}$$

$$\eta_i = F_{g(i)}(s_i) \tag{M2}$$

$$F_V(s) = f_1(s^V), \quad F_O(s) = f_2(s^O), \quad F_M(s) = f_3(s^O, s^V) \tag{M3}$$

$$p_i \equiv P(y_i = 1 \mid F) = \text{expit}\big(F_{g(i)}(s_i)\big), \qquad \log p_i = \text{log\_sigmoid}(F), \quad \log(1 - p_i) = \text{log\_sigmoid}(-F) \tag{M4}$$

There is no unit-level nugget (decision 1; §6.1 gives the reasons).
`jax.nn.log_sigmoid` keeps both log-probabilities finite and accurate across
F ∈ [L, U].

### 3.3 Level 2: the spline curves

**Basis.** For each axis a ∈ {V, O, Mx, My}:
- a clamped **quadratic** B-spline basis $\{B^a_k\}_{k=1}^{K_a}$ on the §2 knots;
  $K_a$ = intervals + 2 (12, 14, 15, 13);
- Greville abscissae $\xi^a_k = (t_{k+1} + t_{k+2})/2$ and spacings
  $h^a_k = \xi^a_k - \xi^a_{k-1}$.

The basis matrices are precomputed once in numpy
(`scipy.interpolate.BSpline.design_matrix`) because the scores are fixed data.

$$f_1(s) = \sum_{k=1}^{K_V} c^{V}_k B^V_k(s), \qquad f_2(s) = \sum_{k=1}^{K_O} c^{O}_k B^O_k(s) \tag{M5}$$

$$f_3(x, y) = \sum_{j=1}^{J}\sum_{k=1}^{K} C_{jk}\, B^{Mx}_j(x)\, B^{My}_k(y), \qquad J = 15,\ K = 13 \tag{M6}$$

**Why quadratic.** For degree ≤ 2, a B-spline is nondecreasing **iff** its coefficients
are nondecreasing:
- the derivative is a linear spline whose coefficients are the scaled coefficient
  differences;
- a linear spline is ≥ 0 iff its coefficients are.

For cubic splines, coefficient monotonicity is only sufficient. Quadratic is therefore
C¹ and carries an exact constraint (decision 3). The iff is derived here from the B-spline
derivative formula; the cited papers state only sufficiency. Brezger & Steiner (2008)
found that relaxing their cubic condition to the full class of monotone cubic splines
gave "little further predictive gain" at much higher cost, so the cubic arm (S4) is
low priority. Because B-splines are a partition of unity,
$\min_k c_k \le f(s) \le \max_k c_k$, so **bounding the coefficients bounds the curve.**

**Monotone, bounded coefficients: 1-D.** The unconstrained parameters are an anchor
$\alpha_g$ and log-slopes $\gamma_{g,k}$:

$$\tilde c_{g,1} = \alpha_g, \qquad \tilde c_{g,k} = \tilde c_{g,k-1} + h_k\, e^{\gamma_{g,k}}, \quad k = 2..K_g \tag{M7}$$

$$c_{g,k} = L + (U - L)\,\text{expit}(\tilde c_{g,k}) \tag{M8}$$

- (M8) is a fixed increasing bijection ℝ → (L, U), so $c_g$ is strictly increasing and
  inside (L, U), and so is $f_g$.
- $e^{\gamma_k}$ is the slope of $\tilde c$ per unit score, so the prior on the slope
  *level* (μ) does not depend on how unevenly the decile knots are spaced.
- **The smoothness prior (M11) does depend on spacing.** It acts per knot *step*, i.e.
  per decile of data mass, not per unit score. On the matched Overture axis the
  intervals range from 0.18 to 0.0041 (44×), so the slope may change far faster per unit
  score inside the atom cluster than on [0, 0.72].
  - This may be desirable, since it lets an atom depart from its neighbours.
  - The alternative follows the divided-difference remark in Eilers & Marx's rejoinder:
    random-walk increments with variance ∝ Greville spacing. It is sensitivity run S7.
- **Neutral shape:** constant γ makes $\tilde c$ linear in s (B-splines reproduce linear
  functions at the Greville points), so F is a logistic ramp in s between L and U.
- $\tilde c = 0$ ↔ F = 0 ↔ P = 0.5.
- **Implementation note (execution log, decision 4): the level is anchored at the
  data centre.**
  - As written, (M7) and (M9a) pin α at the first coefficient: score 0, or the
    matched corner, where there are almost no data. That makes α strongly
    correlated with the slopes.
  - The code builds $\tilde c$ from 0 and shifts it so that
    $\tilde c_{k^*} = \alpha_g$, where $k^*$ is the coefficient whose Greville
    abscissa is nearest the segment's median phase-1 score. For matched, the anchor
    cell is the pair of per-axis medians.
  - Both constructions are shift-equivariant, so the family of curves is unchanged.
    Only the meaning of α, and hence of its prior N(0, 1.5²), moves to the data
    centre.

**Monotone, bounded coefficients: 2-D (max recursion).** Axis x = OSM (index j),
y = Overture (index k):

$$\tilde C_{11} = \alpha_M \tag{M9a}$$
$$\tilde C_{j1} = \tilde C_{j-1,1} + h^{x}_j e^{\gamma_{j1}}, \qquad \tilde C_{1k} = \tilde C_{1,k-1} + h^{y}_k e^{\gamma_{1k}} \tag{M9b}$$
$$\tilde C_{jk} = \max\big(\tilde C_{j-1,k},\ \tilde C_{j,k-1}\big) + \min(h^x_j, h^y_k)\, e^{\gamma_{jk}}, \quad j, k \ge 2 \tag{M9c}$$
$$C_{jk} = L + (U - L)\,\text{expit}(\tilde C_{jk}) \tag{M10}$$

- $\tilde C$ is strictly increasing along both axes by construction.
- (M9) is a **bijection** between $(\tilde C_{11}, \text{excesses} > 0)$ and the interior
  of the doubly-monotone cone, because every doubly-increasing matrix has
  $\tilde C_{jk} - \max(\text{predecessors}) > 0$. So it imposes **no sign on the
  interaction**.
- The alternative in the shape-constrained additive model literature is the Kronecker
  cumulative-sum reparametrization β = (Σ_x ⊗ Σ_y) β̃ with β̃ > 0 (SCAM style). It forces
  every cross-difference $C_{j+1,k+1} - C_{j+1,k} - C_{j,k+1} + C_{jk}$ to be ≥ 0, that
  is, a **complementary (supermodular) interaction only**. That would rule out the
  substitutive structure the interaction model found (a3 = −11.1; 96% of bootstrap
  replicates negative), so it is rejected. §6.4 has the citation check.
- **Neutral shape:** with equal spacings, constant γ gives an additive linear
  $\tilde C_{jk} = \alpha + b h (j + k - 2)$. Unequal spacings give an approximately
  additive surface.
- **Kinks.** `max` makes the log density continuous but only piecewise-smooth. NUTS
  tolerates kinks (the turnover code already relies on this for `clip`). If divergences
  concentrate there, the fallback is a smooth max, $t\,\text{logsumexp}(a/t, b/t)$ with
  t = 0.05. It is still monotone, at a cost of ≤ t·log 2 of forced excess per cell.
- **Relation to the production model.** The interaction model is the 2 × 2 *linear*
  lattice on rescaled-logit axes (Gupta et al., 2016). F3 is a 15 × 13 quadratic lattice
  on the raw-score axes with decile knots.

### 3.4 Level 3: smoothing priors on the log-slopes

Each curve's log-slope field is a mean plus an intrinsic first-order Gaussian Markov
random field, written **non-centred** through the eigenbasis of its structure matrix:

$$\gamma_g = \mu_g \mathbf{1} + \tau_g\, V_{g,+}\,\Lambda_{g,+}^{-1/2}\, z_g, \qquad z_g \sim N(0, I) \tag{M11}$$

- **1-D (g = V, O).** The structure matrix is the path-graph Laplacian on the
  $K_g - 1$ slopes (RW1). Adjacent log-slopes then differ by ~N(0, τ_g²).
- **2-D (g = M).** It is the grid Laplacian $R_J \otimes I + I \otimes R_K$ on the
  JK − 1 active cells. $\gamma_{11}$ is unused and is dropped from the field.
- $V_{g,+}$ and $\Lambda_{g,+}$ are the eigenvectors and eigenvalues with the single zero
  eigenvalue removed. That is the sum-to-zero constraint, so μ_g carries the level
  exactly.
- Non-centring avoids the τ funnel, as the turnover model does for its random effects.

### 3.5 Level 4: the LLM measurement layer (the "silver" data): arm A, rejected

> **Arm A was the planned main model and was rejected in Phase 1** (§0, §11; execution
> log decisions 7, 11 and 13). Two changes were made in the course of Phase 1:
> - (M12) was implemented with an orthonormal sum-to-zero (Helmert) basis for the class
>   logits, not the exists:high reference class;
> - (M13) is centred: ψ_g ~ N(μ^ψ, ω_ψ²) directly.
>
> Neither change removed the bias. This section is kept as the record of the method
> compared against. The preferred data layer is arm C (§3.5b).

The verdict is modelled **given the truth** (the forward, Dawid–Skene direction), so that
F stays at the top of the model and remains monotone by construction:

$$P(v_i = v \mid y_i = y, r_i) = \theta_{g,y,v}(r_i) = \text{softmax}_v\big(\psi_{g,y,v} + \beta_{g,y,v}\,(r_i - \bar r_g)\big) \tag{M12}$$

$$\psi_{g,y,v} = \mu^{\psi}_{y,v} + \omega_{\psi}\, z^{\psi}_{g,y,v}, \qquad \beta_{g,y,v} = \mu^{\beta}_{y,v} + \omega_{\beta}\, z^{\beta}_{g,y,v} \tag{M13}$$

- **Reference class.** exists:high has ψ = β = 0 for every (g, y).
- **Parameter count.** 8 free classes × 2 truth states × 3 segments = 48 ψ and 48 β,
  partially pooled across segments through (M13). Sparse cells such as exists:low
  (≤ 2 gold per segment) borrow strength.
- **β is the differential-misclassification term.** It lets, for example, the
  unverifiable rate depend on score *given* truth. v4 found the unverifiable class's
  existence rate varies with score, and a nondifferential model would push that
  variation into F.
  - β is shrunk toward 0.
  - β ≡ 0 is a sensitivity arm (§5.5).
- **No label switching.** y is observed on 2,362 rows, so θ is identified directly from
  the gold cross-tabulation. It does not rest on the identifiability conditions for
  latent-class models without a gold standard (Hui & Walter, 1980).
- **Human gold is treated as error-free.** §6.7 covers the known "currently open"
  ceiling.

### 3.5b Arm C: the simple silver layer (the preferred model; decisions 11, 12, 17, 20)

Arm C replaces (M12)–(M15) with the simplest defensible use of the silver labels. It
began as a sensitivity arm (decision 11), became the main model after arm A failed
(decision 12), and since decision 17 takes the silver-label accuracy as **data**.

Every phase-1 row enters exactly once, with one label:
- gold rows use the human label $y_i$, treated as true;
- non-gold rows use the LLM verdict $\ell_i$. All 5,142 non-gold rows carry
  $\ell_i \in \{\text{exists}, \text{gone}\}$, because every LLM-unverifiable row was
  censused into gold, so no third label arises. This depends on the census: a future
  round without it would drop unverifiable non-gold rows on their verdict, which is
  selection on v, and the code raises.

With $p_i = \text{expit}(F_{g(i)}(s_i))$:

$$\text{gold}:\quad \log \mathcal{L}^C_i = y_i \log p_i + (1 - y_i) \log(1 - p_i) \tag{M15b}$$

$$\text{silver (preferred)}:\quad \log \mathcal{L}^C_i = q_{g(i),\ell_i} \log p_i + (1 - q_{g(i),\ell_i}) \log(1 - p_i) \tag{M15c'}$$

$$q_{g,\ell} = \frac{r_{g,\ell}\, n^{\text{eff}}_{g,\ell} + 0.5}{n^{\text{eff}}_{g,\ell} + 1}, \qquad r_{g,\ell} = \frac{\sum_{i \in \text{gold}, g, \ell} w_i\, y_i}{\sum_{i \in \text{gold}, g, \ell} w_i}, \quad w_i = 1/\pi_{c(i)} \tag{M15d'}$$

- **What q is.** $q_{g,\ell} = P(\text{exists} \mid \text{segment } g, \text{LLM verdict } \ell)$: the
  design-weighted (Hájek) share of gold rows in that segment and definitive verdict that
  truly exist. It is Jeffreys-smoothed on the Kish effective sample size
  $n^{\text{eff}}$, so a cell with no observed errors keeps a small positive error rate.
  $\pi_c$ is the realized phase-2 inclusion of the production refined class c
  (`calibration_fit.inclusion_by_class`), computed per round.
- **q is data, not a parameter.** It is fixed before sampling. It follows the "P(true
  label | data label)" direction Nat first specified, and it is exactly what gold
  measures. Phase 2 is uniform within verdict class, so the gold within a class is a
  random sample of it.
- **Values on round 20260730** (`silver_label_rates`):

  | segment | q(exists) (raw; gold n) | q(gone) (raw; gold n) |
  |---|---|---|
  | Overture | 0.973 (0.976; 181) | 0.003 (0.000; 148) |
  | OSM | 0.992 (0.994; 191) | 0.035 (0.030; 104) |
  | matched | 0.986 (0.988; 231) | 0.009 (0.000; 59) |

- **(M15c') is a fractional-label pseudo-likelihood.** Its score equation sets the
  model's p equal, locally, to the mean of the corrected labels (gold y, or q for silver
  rows).
  - Under correct q, every phase-1 row then contributes its expected true label. The
    phase-1 sample is a random sample given s, so the curve is design-consistent without
    weights.
  - Fractional labels slightly overstate the information in a silver row (a row with
    q = 0.97 counts nearly as a full label). With q within 0.035 of 0 or 1 the effect is
    negligible, but it is a known approximation.
- **How the rates are computed:** from the fit's own training gold, so in
  cross-validation held-out labels never inform them. When earlier rounds are pooled
  (next bullet), their gold enters too, each round under its own design weights. **This
  is how the October 2026 gold folds into the rates** (§8, decision 17).
- **Pooled rounds (decision 20).** Earlier validation rounds listed in
  `conflation.calibration.pooled_rounds` (or `--pooled-rounds`) join the fit as a
  whole, not only the rates: their phase-1 rows enter the curve likelihood beside the
  current round's.
  - **Design per round.** `production_classes` builds the refined classes within each
    round and prefixes them with the round id. Inclusion probabilities, arm B's weights,
    the silver-label rates and the HT reference are then all per round, as each round's
    phase-2 design requires.
  - **Why arm C may pool without weights.** Each round's phase 1 is a sample from
    that month's population. Pooled, the fit targets P(exists | s) averaged over the
    two months, which is the right target if the curve is stable across them. The
    drift check in §8 tests that before pooling.
  - **Knots** come from the deciles of the pooled phase-1 scores (the decision-5 rule,
    unchanged).
  - **CV** folds, holds out and scores the current round only. Earlier rounds stay in
    every training fit. The production comparator sees the current round only, as
    production does. So the comparison is each method as it would be run that month.
  - **Coverage study.** The simulator draws phase 2 per round, at each round's realized
    inclusion.
  - Each fit records its rounds (current first) in `summary.json`. The report and the
    simulator rebuild that exact table (`bayes_calibration_common.fit_rows`).
- **Why the rates are not estimated.** As first run, arm C estimated one symmetric
  agreement parameter, β ~ Beta(63, 1) centred on the concordance slice (63 of 64). Its
  posterior went to about 1 (0.9994), so arm C in effect treated silver labels as exact.
  - That left it about +0.01 high exactly where the LLM's exists verdicts err: Overture
    +0.013 against the design-weighted gold rate, matched +0.009, OSM 0.000. Correcting
    silver labels at the gold concordance rates implies −0.015, −0.010 and −0.001
    (log 20).
  - An asymmetric version (S2C) collapsed Sp to 0.38. Under arm C's likelihood, silver
    noise is identified only by the selection-induced gap between gold and non-gold
    rows, which it must not absorb (log 15).
  - Hence the fixed rates.
  - The estimated variants stay available as sensitivities: `label_noise =
    "symmetric" | "asymmetric" | "none"`. Under them, (M15c) replaces (M15c'):
    $\log[p\,\text{Se} + (1-p)(1-\text{Sp})]$ for "exists" and
    $\log[p(1-\text{Se}) + (1-p)\text{Sp}]$ for "gone".
- **Status of the results.** The Phase 1 results in §11 were produced with the estimated
  symmetric β (≈ 1), before decision 17. The fixed-rate model has not yet been fitted to
  round 20260730; the evidence above predicts it will sit about 0.01 lower on Overture
  and matched.
- **CV caveat.** A held-out LLM-unverifiable gold row has no silver label, so arm C drops
  it from that training fold. That is about 10% of one class per fold.
- **Parameters.** The spline fields only (225 on round 20260730).
- **Remaining assumptions:**
  - q does not vary with score within a (segment, verdict) cell. The v4 constancy check
    found definitive-class rates flat in score; §3.5c re-checks it by tercile.
  - The human gold is error-free (§6.7). Reviewers saw the LLM output.

### 3.5c Fractional labels vs a fixed-rate mixture, and score-dependent error (open)

Two ways to let a silver row inform the curve at fixed, gold-derived rates. Both give
the same answer on average over a (segment, verdict) cell; they differ in how that
average is spread over score, and in how much information a silver row carries.
**Resolved 2026-10-01:** option F stays the main model with q constant in score
(decision 25), and option M is built and run beside it in October as a test model
(decision 24; TODO.md).

**The two summaries of the gold.** For segment g and definitive verdict v:
- *backward* rates q_{g,v} = P(y = 1 | g, v), the share of gold rows with that verdict
  that truly exist (M15d'). Phase 2 samples within verdict class, so these are what gold
  measures directly.
- *forward* rates Se_g = P(v = exists | y = 1, g) and Sp_g = P(v = gone | y = 0, g),
  among definitive verdicts. They follow from the backward rates and the phase-1 verdict
  mix by Bayes' rule, both design-weighted. On round 20260730 (raw): Overture
  (1.000, 0.921), OSM (0.995, 0.965), matched (1.000, 0.861).

**Option F: fractional labels (M15c', as wired).**
$\ell_i = q\log p_i + (1-q)\log(1-p_i)$.
- The derivative in F is $q - p_i$, and the curvature is $-p_i(1-p_i)$, the same as a
  gold row's.
- Every silver row in a cell gets the same probability q of existing, whatever its
  score.
- A silver row counts as a full observation. It is a pseudo-likelihood, not a model of
  the verdict.

**Option M: fixed-rate mixture.**
$\ell_i = \log[p_i\,\text{Se} + (1-p_i)(1-\text{Sp})]$ for "exists" and
$\log[p_i(1-\text{Se}) + (1-p_i)\text{Sp}]$ for "gone". This is (M15c), the marginal
likelihood of the observed verdict with the unknown truth summed out, but with Se and
Sp fixed from the gold instead of estimated.
- The derivative in F is $r_i - p_i$, where
  $r_i = p_i\text{Se} / [p_i\text{Se} + (1-p_i)(1-\text{Sp})]$ for an "exists"
  verdict is the row's own posterior probability of existing. It depends on the score
  through $p_i$: an "exists" verdict is less likely to be right at a low score than at a
  high one.
- The curvature is $-p_i(1-p_i) + r_i(1-r_i)$. By the missing-information principle
  (Orchard & Woodbury, 1972; Louis, 1982, eqs. 3.2–3.3), a silver row's information is
  a gold row's minus the variance of its unknown label. Uncertain silver rows count for
  less, and the bands widen by exactly that amount.
- It assumes non-differential error: P(v | y) does not depend on the score given y.
  Arm A estimated such a slope; dropping it (β ≡ 0) did not change arm A's bias
  (decision 13).
- Fixed Se and Sp cannot collapse the way the estimated Sp did in S2C (0.38). Se = 1.000
  on Overture and matched would make a "gone" verdict certain, so the forward rates need
  the same Jeffreys smoothing as q.
- Cost: a fixed-rate branch in the existing asymmetric code, a test, and a refit.
- **Built 2026-10-01, not fitted:** `label_noise = "fixed_mixture"`, with the
  rates from `calibration_bayes.forward_silver_rates` (execution log, entry 26).

**Evidence on round 20260730** (does q vary with score?). Design-weighted gold rate
within (segment, verdict) by raw-score tercile, with gold n. For the exists verdicts
the table also shows the mean $r_i$ that option M implies, from the first-run arm C
curve:

| segment, verdict | q (cell) | low | mid | high | high − low z | option M: low / mid / high |
|---|---|---|---|---|---|---|
| Overture, exists | 0.973 | 0.959 (62) | 0.964 (54) | 1.000 (65) | +1.4 | 0.941 / 0.963 / 0.992 |
| OSM, exists | 0.991 | 0.980 (57) | 1.000 (71) | 1.000 (63) | +0.8 | 0.989 / 0.993 / 0.995 |
| matched, exists | 0.986 | 0.989 (78) | 0.989 (85) | 0.986 (68) | −0.1 | 0.980 / 0.990 / 0.995 |
| OSM, gone | 0.035 | 0.053 (41) | 0.000 (29) | 0.028 (34) | −0.5 | — |
| Overture, gone | 0.003 | 0.000 (49) | 0.000 (49) | 0.000 (50) | 0 | — |
| matched, gone | 0.009 | 0.000 (16) | 0.000 (21) | 0.000 (22) | 0 | — |

- No tercile differs significantly from its cell's q, and no high-vs-low gap reaches
  |z| = 2. Terciles hold 16–85 gold rows, so this test can only see gaps of about 0.04
  or more.
- The one suggestive pattern is Overture "exists": both errors among that cell's gold
  fall in the lower two terciles. Option M reproduces that direction with no extra
  parameters, whereas option F spreads the error evenly.
- **Score-dependent q directly** (q by tercile, or a logistic q(s) per cell) would cost
  one to three parameters per cell, estimated from 16–85 gold rows per tercile. The
  gone cells have no errors to fit.

### 3.6 The likelihood and the joint posterior

Row contributions, with $\ell^1_{i} = \log p_i$ and $\ell^0_{i} = \log(1 - p_i)$ from (M4):

$$\text{gold } (G_i = 1):\quad \log \mathcal{L}_i = y_i \ell^1_i + (1 - y_i)\,\ell^0_i + \log \theta_{g(i), y_i, v_i}(r_i) \tag{M14}$$

$$\text{non-gold } (G_i = 0):\quad \log \mathcal{L}_i = \text{logaddexp}\big(\ell^1_i + \log\theta_{g(i),1,v_i}(r_i),\ \ell^0_i + \log\theta_{g(i),0,v_i}(r_i)\big) \tag{M15}$$

$$p(\Theta \mid \mathcal{D}) \propto \prod_{i=1}^{n} \mathcal{L}_i \;\times\; \prod_{g \in \{V,O,M\}} p(\alpha_g)\,p(\mu_g)\,p(\tau_g)\,p(z_g) \;\times\; p(\mu^\psi)\,p(\mu^\beta)\,p(\omega_\psi)\,p(\omega_\beta)\,p(z^\psi)\,p(z^\beta) \tag{M16}$$

The parameter vector is
Θ = {α_g, μ_g, τ_g, z_g; μ^ψ, μ^β, ω_ψ, ω_β, z^ψ, z^β}, with dimension ≈ 12 + 14 +
195 + 3·3 + 16 + 16 + 2 + 96 ≈ **360**. Positive scalars (τ, ω) are sampled as logs,
with the log-Jacobian added. The turnover model instead puts priors on the log scale
directly, so the convention is stated explicitly here.

**Why there are no survey weights in (M16).** The phase-2 indicator $G_i$ depends only
on $v_i$ (and in principle could depend on $s_i$), both of which the model conditions
on. Selection is therefore missing-at-random given observed quantities, and its
mechanism shares no parameters with Θ. By the standard ignorability result, the factor
$p(G \mid v, s)$ drops out of the likelihood. This is the verification-bias setting of
Begg & Greenes (1983) that v4 §3 already invokes. So:
- the design's information about y enters through the modelled verdicts, not through
  weights;
- a phase-2 draw that selects on score within class is **still ignorable**, which v4's
  difference estimator cannot tolerate (v4 §7.1). §8 relies on this.

**Three limits on that argument** (checked against the full texts):
1. **It holds only because every phase-1 row enters through (M15).** A gold-only model
   with v as a predictor would still be biased. That is Little's truncation example in
   the discussion of Rubin (1976), and Savitsky & Toth's warning about predictors
   observed only for sampled units.
2. **Ignorability is a likelihood and Bayes result, not a frequentist one.** For
   sampling-distribution inference Rubin also requires "observed at random", which fails
   when selection depends on the observed v. Band coverage and CV behaviour of arm A are
   therefore not guaranteed by the argument and must be measured (§5.2, §5.4).
3. **"No weights" means implicit weights, not absent ones.** Little (2004) is explicit
   that a model which conditions on the design variables reproduces the design weighting.
   It is design-consistent only if it models the outcome differences across inclusion
   classes correctly: here, the form of θ(r) in (M12). Hence the misspecified-truth
   rounds in §5.2.

**Comparison arm B: gold-only, design-weighted.** This arm drops (M12)–(M15) and uses a
weighted pseudo-likelihood on gold only:

$$\log \mathcal{L}^{B} = \sum_{i: G_i = 1} \tilde w_i\,[y_i \ell^1_i + (1 - y_i)\,\ell^0_i], \qquad \tilde w_i = \frac{n^{gold}_{g}}{\sum_{j \in g,\,G_j = 1} \pi_{c(j)}^{-1}}\;\pi_{c(i)}^{-1} \tag{M17}$$

- The weights are "relative" weights: they are normalised to sum to the segment's gold
  count (Savitsky & Toth normalise to the sample size n). This puts the pseudo-posterior
  on the right gross scale, but **its spread is not calibrated to the design**.
- Under a stratified design, Williams & Savitsky (2021) found unadjusted pseudo-posterior
  intervals can be too wide as well as too narrow: their SPPS1 case over-covered, 0.99
  joint coverage against a nominal 0.90. Phase 2 here is stratified on a strongly
  predictive variable, so either direction is possible.
- π_c is the realized class inclusion (`calibration_fit.inclusion_by_class`).
- It is used as a point-prediction comparator only. Its intervals are not reported
  without the Williams & Savitsky adjustment (Algorithm 1: Hessian and replicate-score
  sandwich, then a Cholesky transform of the draws).

### 3.7 Priors (as run; the planned values, kept after the §5.2 prior-predictive check)

| parameter | prior | on scale | reasoning |
|---|---|---|---|
| $\alpha_V, \alpha_O, \alpha_M$ | N(0, 1.5²) | $\tilde c$ at the data-centre coefficient (§3.3 note) | 95% of F there within about ±6, i.e. P between 0.003 and 0.997 |
| $\mu_V, \mu_O, \mu_M$ | N(0.5, 1²) | mean log-slope of $\tilde c$ per unit score | slope e^0.5 ≈ 1.6; a curve rising from P = 0.5 to 0.98 over half the domain has log-slope ≈ 0.9 |
| $\tau_V, \tau_O, \tau_M$ | Half-N(0, 0.5²) | SD of adjacent log-slope differences | allows the slope to change ~×5 across 12 knot steps |
| $q_{g,\ell}$ (arm C, preferred) | **data, not a parameter** (M15d') | P(exists \| segment, silver verdict) | design-weighted gold rate, Jeffreys-smoothed on its Kish ESS (§3.5b) |
| $\beta_\ell$ (arm C sensitivity, `label_noise = symmetric`) | Beta(63, 1) | P(silver label = truth) | concordance slice: 63 of 64 definitive verdicts agree. The prior barely matters: Beta(9, 1) and β = 1 give the same curves (S3C) |
| Se, Sp (arm C sensitivity S2C, `asymmetric`) | Beta(0.998·64, 0.002·64), Beta(0.913·64, 0.087·64) | P(silver exists \| y = 1), P(silver gone \| y = 0) | handoff's design-weighted Se and Sp at concordance-slice concentration. Not identified: Sp collapses to 0.38 (log 15) |
| $\mu^\psi_{y,v}$ | N(0, 3²) | free sum-to-zero (Helmert) contrasts of the class logits (log 7; was log-odds vs exists:high) | class rates range from ~0.1% to ~70% |
| $\mu^\beta_{y,v}$ | N(0, 1²) | log-odds per unit score | |
| $\omega_\psi$ | Half-N(0, 1²) | between-segment SD of ψ | |
| $\omega_\beta$ | Half-N(0, 0.5²) | between-segment SD of β | |
| $z_g$ | N(0, 1) | non-centred innovations of the log-slope fields (M11) | |
| $\psi_g, \beta_g$ (arm A) | N($\mu^\psi$, $\omega_\psi^2$), N($\mu^\beta$, $\omega_\beta^2$) | **centred** segment effects (log 7; plan had non-centred $z^\psi, z^\beta$) | |

The penalised-complexity alternative for the SD parameters (τ, ω) is an exponential
prior on the SD, calibrated by a tail statement such as P(τ > 1) = 0.05 (Simpson et
al., 2017). The half-normals above are close in spirit. Either can be swapped in at
review.

### 3.8 What gets published (derived quantities)

- **The calibrated probability** for a POI in segment g with scores s is

  $$m_g(s) = \text{expit}\big(F_g(s)\big) \tag{M18}$$

  published as its posterior mean.
- **The band** is the 2.5% and 97.5% posterior quantiles of $m_g^{(d)}(s)$ over draws d.
  It is "uncertainty about the calibrated probability", which is the quantity the
  current bands claim to cover. The §5.2 coverage study measures whether it does.

---

## 4. Implementation (off the production path)

**Files (as built)**

| path | contents |
|---|---|
| `src/openpois/conflation/calibration_bayes.py` | `ModelSpec` / `PriorConfig`; knot rule; basis and Greville abscissae; RW1 / grid-GMRF eigenbases; the (M7)–(M10) transforms (data-centre anchor, smooth-max recursion); `silver_label_rates` (M15d'); `prepare_data`; log density for arms A, B and C (`label_noise`: fixed, symmetric, asymmetric, none); MAP + NUTS `fit`; arviz diagnostics and the §5.1 acceptance rule; `curve_draws`; `assign_segment_folds` (production-parity CV folds) |
| `scripts/conflation/bayes_calibration_common.py` | config and handoff loading (`load_handoff` pools the current round with `pooled_rounds`; `fit_rows` rebuilds an earlier fit's table), the output-directory guard, production-curve evaluation, design weights and Hájek bin rates, the dataviz palette |
| `scripts/conflation/fit_bayes_calibration.py` | full-data fit: prior predictive, NUTS, convergence, curves, PPC (every arm), deployed-impact preview, figures, `summary.json`; `--reuse-draws` recomputes everything but the sampling |
| `scripts/conflation/cv_bayes_calibration.py` | 10-fold CV: production comparators, per-(arm, fold) worker subprocesses, per-arm flags, design-weighted scores, paired stratified bootstrap, `cv_results.{json,md}` |
| `scripts/conflation/simulate_bayes_recovery.py` | fake-data coverage study (§5.2): in_family / realistic / category / step scenarios, coverage by segment and region |
| `scripts/conflation/report_bayes_calibration.py` | assembles `fit_report.md` from the outputs above; no fitting |
| `scripts/conflation/run_bayes_phase1.sh` | unattended, resumable pipeline: main fits → CV → coverage and sensitivity → report |
| `tests/test_calibration_bayes.py` | 17 tests; see below |
| `src/openpois/models/jax_core.py` | the one shared-code change: the `adaptation_kwargs` passthrough |
| `config.yaml` → `conflation.calibration.pooled_rounds` | earlier validation rounds pooled into the fit and the silver-label rates (empty now; October: `["20260730"]`) |
| `.claude/plans/bayesian-monotone-calibration-notes.md` | execution log: entries 1–25 and open questions |

- **Output directory:** `~/data/openpois/conflation/20260730/calibration_eval_bayes_20260927/`.
  Scripts refuse to write into any deployed `calibration/` directory (the same guard as
  `compare_matched_index.py`).
- **No Makefile, deploy or production-curve change.** The only config addition is
  `pooled_rounds`, which only the prototype reads.

**Reuse from the JAX stack** (confirmed by the deep dive of `jax_core`, `ModelFitter`
and `osm_models`):
- **Sampler.** Call `jax_core.nuts_sample_multichain(log_density, init, num_warmup,
  num_samples, num_chains = 4, key = jax.random.PRNGKey(seed), init_jitter = 0.5)`
  directly.
- **Not `ModelFitter`.** Its `predict`/`calculate_probs` are hard-wired to the Poisson
  hazard (`1 − exp(−rate)`, `data["dt"]`), and it cannot pass `init_jitter`.
- **Wider starts.** Starts are jittered wider than the default 0.05 so that R̂ can see
  multimodality.
- **Initial values:**
  - α and μ from a fit of each 1-D production curve;
  - z = 0;
  - ψ from the gold cross-tab log-odds, mapped to the sum-to-zero basis;
  - logit β_ℓ = logit(0.984) for the estimated-noise arm C variants.
  - In practice, NUTS starts from an L-BFGS-B MAP (`find_map`) with chains jittered
    N(0, 0.5).
- **One small, backward-compatible change to `jax_core`:** an optional
  `adaptation_kwargs` passthrough (`target_acceptance_rate`,
  `is_mass_matrix_diagonal`). Neither is exposed today; the defaults are 0.8 and a
  diagonal mass matrix. The existing call paths are unchanged, and a test pins that.
- **Diagnostics from arviz 1.0, not BlackJAX's.** BlackJAX's classic R̂ and non-rank ESS
  are what `diagnostics.summarize_chain_draws` reports.
  - Split-R̂ (rank-normalised), bulk and tail ESS.
  - Post-warmup divergences per chain.
  - E-BFMI from `sampler_info.energy`.
  - Tree-depth saturation (`num_integration_steps == 1023`).
- **Precision.** `enable_high_precision()` (x64) in these scripts only; at n = 7,500 it
  is cheap and removes any doubt near the ±6.9 bound.
- **Style.** New code uses `jax.Array` annotations, not the removed `jrd.KeyArray`.
  Spaces around `=` everywhere; line length 90; no Black on new files.

**Tests** (`tests/test_calibration_bayes.py`), following the repo's
simulate-and-recover and paired-implementation patterns:
1. **Monotonicity and bounds.** For 200 random parameter draws, f1/f2 on a 10,001-point
   grid and f3 on a 401 × 401 grid are nondecreasing and inside (L, U).
2. **Bijection.** A random doubly-increasing matrix recovers exactly through the inverse
   of (M9).
3. **Linear reproduction.** Constant γ with equal spacing gives an additive linear
   $\tilde C$.
4. **Boundary accuracy.** The `log_sigmoid(±F)` log-probabilities stay finite and
   accurate at F = L and F = U.
5. **Likelihood identity.** If θ is the identity (v ≡ y), (M15) reduces to the marginal
   Bernoulli. With β_ℓ = 1 and silver labels equal to y, arm C reduces to the plain
   Bernoulli. The pointwise log-likelihood sums to the total. Value and `jax.grad` match
   a numpy reference.
6. **Knot rule.** Duplicate deciles collapse; no gap exceeds 0.2; the atoms are knots.
7. **Fold parity.** The CV fold assignment reproduces `cross_fit_predictions`' `fold`
   array exactly for the same seed and folds.
8. **Recovery.** Short 2-chain fits (arms A and C) on simulated data cover the true m(s)
   at the median score.
9. **Shared code.** The `jax_core` default path equals a direct BlackJAX run with the
   same keys, and `adaptation_kwargs` reaches the adaptation.
10. **Tensor orientation.** F3 matches a direct scipy double sum on an asymmetric
    coefficient matrix.
11. **Label-noise variants.** Asymmetric with Se = Sp equals symmetric. "none" equals
    symmetric with β → 1.
12. **Fixed silver rates.** `silver_label_rates` matches a hand computation; held-out
    gold is excluded; pooled rounds add their gold. With q ∈ {0, 1}, (M15c') equals the
    Bernoulli on the label.
13. **Pooled rounds** (decision 20). A two-round table's classes are each round's own
    classes, prefixed with the round id; rates from the pooled table equal rates from
    the per-round frames; `prepare_data` uses them and keeps every pooled row.

17 test functions; all pass (2026-10-01). The repository suite has 5 pre-existing failures in
download-test mocks, unrelated (TODO.md).

---

## 5. Validation plan

### 5.1 Computation acceptance (every fit)

- 4 chains, 1,000 warmup, 1,000 draws.
- Max split-R̂ ≤ 1.01 and min bulk and tail ESS ≥ 400 over Θ, **and over m_g(s) on the
  published grid**.
- 0 post-warmup divergences, E-BFMI ≥ 0.3, and no tree-depth saturation.
- On failure, escalate in this order: target acceptance 0.95 → dense mass matrix →
  smooth-max variant of (M9c). Record each step in the notes file.
- **As run** (log 8–10):
  - Target acceptance 0.95 made things worse, and the dense mass matrix saturated the
    tree depth. The **smooth max with t = 0.05 is adopted for every arm**.
  - Main fits used 4 × (1,000 + 1,000); CV, coverage and sensitivity fits used
    4 × (600 + 400).
  - Main arm C passed every criterion except zero divergences: 21 in 4,000, from the
    steep-slope tail of exp(γ). Arm B had 150, tolerated for a comparator. Arm A failed
    outright.
  - The light fits typically reach R̂ 1.01–1.025 and ESS 200–750. That is adequate for
    posterior means and 95% bands, and they are reported as-is.

### 5.2 Before real data

- **Prior predictive.** Draw m_g(s) curves from the priors and plot them. Confirm the
  prior is weakly informative:
  - curves span most of (0, 1);
  - not concentrated on step functions;
  - no mass piled at the bounds.
- **Fake-data recovery (a coverage study of the bands).**
  - Truths: the full-data posterior-mean F1/F2/F3 and θ.
  - 20 synthetic rounds on the real design: the same scores; v drawn from θ; gold drawn
    at the realized class rates; unverifiable censused.
  - Fit each round.
  - Report the empirical coverage of the 95% m_g(s) band, pooled over population-weighted
    grid points, per segment. This is the Bayesian counterpart of the 0.89 / 0.91 / 0.66
    bootstrap result.
  - **Report coverage by region, not only pooled:**
    - interior slope;
    - flat regions (the Overture floor, the substitutive high-Overture band of F3);
    - domain edges.

    Over- and under-coverage can cancel in a pooled figure. The coverage theory
    (§6.11) covers neither flat regions nor edges.
  - **Add 10 rounds with a truth outside arm A's family**, since truths taken from arm A
    cannot reveal its own misspecification bias (Little, 2004):
    - (i) a class-specific θ(r) that is nonlinear in r (a step at the Overture atoms);
    - (ii) a θ that also depends on a simulated category variable correlated with y.

    Arm B is fit to the same rounds. Its bias there is the design-based benchmark.
  - **As run, with arm C as the fitted model** (decision 12;
    `simulate_bayes_recovery.py`). The truth curves are arm C's posterior-mean curves.
    - **in_family** (20 rounds): y ~ Bernoulli(m). The unverifiable rate u_{g,y} is the
      design-weighted rate from gold. Other verdicts come from the forward silver rates
      (symmetric β for the first-run fit; design-weighted Se_g and Sp_g for the
      fixed-rate model).
    - **realistic** (5): verdicts drawn from arm A's fitted θ (asymmetric,
      score-dependent).
    - **category** (5): a latent category shifts the existence logit by +1.0 (c − 0.3)
      and the unverifiable logit by +1.5 c.
    - **step** (5): truly-existing rows at or above the 0.919912 atom (OSM ≥ 0.9) get
      +1.2 on the unverifiable logit.
    - Arm B is fit on the 15 misspecified rounds.
    - Coverage is evaluated at 40 quantile points per 1-D segment and 150 sampled
      matched pairs. Regions: edge = outside the 5–95% score range; flat = |∂m/∂s| < 0.1
      (for matched, the smaller partial).

### 5.3 Full-data fit and checks

- **Curves.** m_V, m_O (1-D, with bands) and m_M (heatmap, plus OSM slices at the two
  Overture atoms and at y = 0.75). Each is overlaid on:
  - the production curve or lookup;
  - the HT reference points;
  - the difference-estimator bin rates from `axis_monotonicity_table`.
- **Posterior predictive checks.**
  - The verdict distribution by score decile, per segment, replicated vs observed.
  - The gold y-rate within each verdict class by score tercile.
  - The pooled y-rate per knot interval.
- **Arm C's β_ℓ.** Plot prior against posterior. Expect little movement (§3.5b).
- **Deployed-impact preview** on the 20260902 population, streamed and column-scoped
  (never a whole-file load). Report the share of POIs moving by > 0.05 against the
  production curves, by segment. The published values themselves are not changed.

### 5.4 Ten-fold cross-validation

- **Folds.**
  - Gold rows only, 10 folds per segment, assigned **within refined class** (after
    `merge_thin_cells`, `min_cell_gold` = 25).
  - The algorithm and seed are exactly those of `cross_fit_predictions`
    (`rng_seed + 700`), so the Bayesian model and the comparators score identical
    held-out rows.
  - Fold f holds out fold f of every segment at once (one joint fit per fold).
- **Holding out.** A held-out gold row stays in phase 1 with its verdict and loses y and
  gold status, as in the production cross-fit, so neither side sees held-out y.
- **What is refit.**
  - Bayesian: the whole model.
  - Comparators: `cross_fit_predictions(..., n_folds = 10)` per segment, with
    `index_mode = interaction` (matched) and native (osm and overture) and the
    20260730 population for bins. Each comparator is scored twice:
    - **as published** (40-bin lookup, `predicted`);
    - as its **curve** (`predicted_grid`).
- **Predictions.** For held-out row i, $\hat p_i = \frac{1}{D}\sum_d m^{(d)}_{g(i)}(s_i)$.
  The prediction does not condition on the row's LLM verdict: the published curve is
  P(y | s).
- **Weights.** $w_i = 1/\pi_{c(i)}$ from the training split, which is exactly the
  comparator's `weight`. For the pooled score each segment gets **equal weight**
  (decision 9): a row's weight is scaled so every segment's weights sum to the same
  total. Per-segment scores are always reported.
- **Metrics**, per segment and pooled. §5.4a gives the mbg definitions (Henry, n.d.)
  and how they are adapted. The headline comparator is the interaction model *as
  published* (decision 6).
  - Brier $B = \sum_i w_i (y_i - \hat p_i)^2 / \sum_i w_i$.
  - **Relative Brier** $B_{\text{Bayes}} / B_{\text{comparator}}$, plus the Brier skill
    score $1 - B_{\text{Bayes}}/B_{\text{comp}}$.
  - **OOS RMSE** = $\sqrt{B}$.
  - **OOS LPD** = $\sum_i w_i \log\big(\frac{1}{D}\sum_d p(y_i \mid \Theta^{(d)})\big) / \sum_i w_i$.
    For a Bernoulli outcome this equals $\sum_i w_i \log(\hat p_i^{y_i}(1 - \hat p_i)^{1 - y_i}) / \sum w$,
    i.e. minus the log score of the posterior-mean prediction. Comparators use their
    point prediction, clipped to [1e-12, 1 − 1e-12] as in `scoring_rules`.
  - CORP MCB / DSC / UNC (Dimitriadis et al., 2021) via `calibration_fit.scoring_rules`.
    Brier and log score are strictly proper (Gneiting & Raftery, 2007).
- **Uncertainty.** A paired bootstrap of the differences (2,000 reps, strata = fold ×
  refined class) for ΔBrier, the Brier ratio and ΔLPD, with 95% intervals. This is the
  robustness variant of the ladder (fold × class strata).
- **Arms under CV (as planned; see "As run" below):**
  - A (the planned main model);
  - B (weighted gold-only, §3.6);
  - C (simple silver layer, §3.5b).

  **A vs C** (paired ΔBrier and ΔLPD) is the test of whether the full measurement
  layer adds out-of-sample performance.

  That is 3 × 10 fits plus 3 full fits. At an estimated 3–8 min per fit (≤ 360
  parameters, 7,500 rows), that is roughly 2–4 h, launched in the background with
  `python -u … | tee` and one Monitor.
- **As run** (decisions 12, 14 and 15): arms C and B, 10 folds each, at 4 × (600 + 400).
  - The arm A CV was started and then deferred by Nat once the full-data evidence
    settled C over A.
  - The comparator folds match production's exactly (asserted).
  - The silver-label rates, when fixed, come from each fold's training gold only.
  - Model-comparison intervals come from the stratified bootstrap only. Posterior draws
    give the curve's uncertainty, not the evaluation sample's.

### 5.4a OOS metric definitions (mbg article)

Source: the mbg model-comparison vignette (Henry, n.d.).

**What mbg does:**
- **RMSE** $= \sqrt{\frac{1}{N}\sum_i (y_i - \hat y_i)^2}$, with $\hat y_i$ the posterior
  mean prediction. It is computed per fold, then **averaged** across folds.
- **LPD** $= \sum_i \log\big(\frac{1}{S}\sum_s p(\tilde y_i \mid \theta_s)\big)$:
  pointwise log-mean-exp over draws, **summed** over points, then **summed** across
  folds. Closer to 0 is better.
- **Selection rule:** OOS LPD is primary and OOS RMSE breaks ties.
- **Folds:** simple random 10-fold.
- **WAIC** is in-sample only.

**Adaptations here, with their reasons:**
- **Folds are stratified within refined class, not random.** Survey CV forms folds that
  mimic the sampling design: "Create SRS CV folds separately within each stratum"
  (Wieczorek et al., 2022). The LLM-verdict class *was* the phase-2 stratum.
  - Wieczorek et al. explicitly reject stratifying just to keep rare classes in every
    fold (the rationale of Kohavi, 1995), so that is not the justification here.
  - Caveat: `merge_thin_cells` pools refined cells back to the verdict. Where a pooled
    cell mixes different realized π, the folds only approximately mimic the design.
- **Every sum and mean is design-weighted.** The weights are $\tilde w_i = w_i \cdot
  n_{\text{held}} / \sum w$, normalised to sum to the held-out count, so that an mbg-style
  *sum* of LPD stays on the familiar per-row scale. This is the Hájek (ratio) form of
  their design-based test-error estimator.
- **Headline numbers are reported both ways:**
  - mbg-style: the mean of per-fold RMSE and the summed LPD;
  - pooled: √(pooled Brier) and LPD per row, which is what the relative-Brier ratio
    uses.
- **Training weights.** Wieczorek et al.'s general rule is consistency: CV training must
  fit models the way the final model will be refit. Their specific recommendation is to
  weight both training and testing "when sampling weights are informative". Their
  simulation shows that weighting only the test side can mislead in that case.
  - Arm A is unweighted in both CV and final fit, so it meets the consistency rule. It
    models the selection variable on every phase-1 row, a case the paper does not
    consider (§3.6).
  - Arm B is weighted in both, which is the version the paper would endorse. That is
    one more reason to keep it.
  - **Decision rule:** an A–B difference in held-out scores beyond bootstrap error is
    treated as a misspecification flag for arm A's measurement layer, not only as a
    ranking.

### 5.5 Sensitivity runs (full data, on arm C as run; light settings)

S2 (β ≡ 0) and S3 (3 verdict classes) tested arm A's measurement layer. After
decision 12 they were replaced by S2C and S3C, which test arm C's silver-label
assumptions. The numbering is kept.

| run | change | mean shift in P(exists) at the validation rows vs arm C (Overture / OSM / matched) |
|---|---|---|
| S2C | asymmetric label noise, estimated Se and Sp | 0.201 / 0.106 / 0.080. Not identified: Sp collapses to 0.38 (log 15) |
| S3C-a | silver labels exact (β = 1) | 0.001 / 0.001 / 0.001 |
| S3C-b | β ~ Beta(9, 1) | 0.001 / 0.001 / 0.001 |
| S4 | cubic basis (coefficient monotonicity then only sufficient) | 0.003 / 0.000 / 0.001 |
| S5 | max knot gap 0.1 | 0.002 / 0.001 / 0.004 |
| S5 | 20 equally spaced knot intervals | 0.005 / 0.002 / 0.006 |
| S6 | τ prior scales × 2 | 0.002 / 0.001 / 0.002 |
| S6 | τ prior scales × ½ | 0.002 / 0.002 / 0.004 |
| S6 | τ_M prior scale × 4 | 0.001 / 0.001 / 0.003 |
| S7 | spacing-scaled random walk | 0.003 / 0.001 / 0.004 |

The matched 95th-percentile shift is 0.015 or less for every row except S2C. Shifts
measured over the whole 81 × 81 matched grid are larger (up to 0.34), because they
include data-empty corners; see `fit_report.md` §4.

### 5.6 Deliverables at the stop point

- `fit_report.md` in the output directory, containing:
  - diagnostics;
  - the parameter table;
  - arm C's β_ℓ prior vs posterior;
  - PPC figures;
  - the CV table (relative Brier, RMSE, LPD, MCB/DSC, bootstrap intervals, per segment
    and pooled with equal segment weights), including the A-vs-C comparison;
  - the coverage study;
  - the sensitivity table;
  - the deployed-impact preview.
- A short summary in chat.
- Nothing merged into the production calibration. Commits wait for Nat's commit session.

---

## 6. Criticism from the literature, and proposed changes

*(Every citation is listed in §10 with its library ID, or marked as not yet in the
library.)*

### 6.1 Why the nugget was dropped (decided: decision 1)

The original request included an i.i.d. Gaussian unit-level nugget,
η_i = F(s_i) + ε_i with ε_i ~ N(0, σ²). It was dropped for these reasons.

- **Not identifiable from binary data.** With binary data, overdispersion cannot be
  detected (McCullagh & Nelder, 1989). Each unit's outcome is one Bernoulli draw, and
  any mixing distribution for its probability collapses to a single marginal
  probability.
  - The likelihood would depend on (F, σ) only through p_i = E_ε[expit(F + ε)].
  - For any σ, a different monotone F reproduces the same p_i, up to the range bound.
  - σ would therefore be identified only by its prior and by the bound. At σ = 2 even
    F = U gives P ≤ 0.9936; at σ = 3, P ≤ 0.975. That is an artefact of the bound, not
    evidence about unit heterogeneity.
  - A unit-level interval would be prior-determined. It would look like uncertainty
    quantification without being learned from data.
- **Marginal versus conditional.** With σ > 0, expit(F) is the ε = 0 (subject-specific)
  curve, not the population curve (Zeger et al., 1988). Their approximation is
  logit m ≈ (1 + c²σ²)^(-1/2)·F, with c = 16√3/(15π) and c² ≈ 0.346. By exact
  quadrature, at F = 2 expit(F) would overstate P by 0.01 at σ = 0.5, 0.04 at σ = 1 and
  0.11 at σ = 2.
- **Even with replication, the variance is barely identified.** In Zeger et al.'s own
  application (four binary observations per child), "there is little information
  about D": the profile likelihood for the random-intercept variance was flat.
- **The population curve needs no heterogeneity model.** Zeger et al. note that the
  population-averaged response "is directly estimable from observations without
  assumptions about the heterogeneity". That is exactly the published quantity, m_g(s).
- **Contrast with MBG.** In MBG the nugget is identified because each location carries
  a binomial count with N > 1. That replication is what is missing here.
- **The identifiable alternative, for October or later.** Grouped random effects are
  identified because many POIs share a group:
  - shared_label category group;
  - urbanicity (deferred to November, decision 7);
  - LLM-verdict confidence as a covariate.

  They also address the calibration doc's "biggest known weakness": the curves
  condition on score only, which pulls stable institutional labels down.

### 6.2 The range bound

With no nugget, the bound on F (M8, M10) is also the bound on the published
probability: m_g(s) = expit(F) ∈ [0.001, 0.999] by construction. The bound is hard,
applied through the coefficient squash (decision 2).

### 6.3 Knots at deciles when the covariate has atoms

- **Duplicate deciles.** Deciles collapse at the Overture atoms: 3 of 9 matched-Overture
  deciles fall on 0.919912 and 0.990219. This is handled by deduplication.
- **Prior-dominated regions.** The gap rule then adds knots where the data are thinnest
  (matched Overture [0, 0.90]: 5 intervals, 85 gold). Posterior shape there is mostly
  prior and monotonicity.
- **An alternative from the P-spline literature:** use many equally spaced knots and let
  the smoothing prior choose the effective complexity (Eilers & Marx, 1996; Lang &
  Brezger, 2004).
  - Eilers & Marx show that knot *number* is "largely immaterial, as long as it is large
    enough". They use only equidistant knots, and leave non-equidistant grids (which
    need divided-difference penalties) unanalysed.
  - The random-walk prior on log-slopes (M11) is *analogous* to that device, not the
    same thing: it is a nonlinear prior over 10–13 decile intervals, not a quadratic
    penalty over many equidistant knots. S5 is the real test.
  - Direct evidence under a monotone constraint: Brezger & Steiner (2008, Table 5) found
    20 quantile-placed knots did as well as or better than 20 equidistant knots.
    Monotonicity, not knot location, drove predictive accuracy. (For their
    *unconstrained* models, quantile knots were worse.)
- **Atoms as knots.** With quadratic splines and a knot *at* an atom, the atom's value
  is not a free level. If the atom behaves unlike its neighbours (v4 §4.6 found the
  overture segment's 0.85–0.92 → 0.9199 drop), the spline smooths it. A per-atom offset
  constrained to keep monotonicity is a possible extension, not proposed now.
  - Brezger & Steiner's monotone fit still captured a sales spike at a price atom
    (99 cents); monotonicity "does not preclude … steps and kinks".
  - S7 (spacing-scaled smoothness) changes how freely the curve can bend at the atom
    cluster, so it is the first check if the atoms look over-smoothed.

### 6.4 Monotone tensor splines: sufficient vs necessary conditions, and interaction sign

- **Coefficient monotonicity.** It is necessary and sufficient for monotone B-splines of
  degree ≤ 2, and only sufficient for cubic (hence quadratic, decision 3).
- **2-D conditions.** For tensor products, adjacent-coefficient monotonicity in each
  direction is sufficient. It is also necessary for the multilinear lattice (Gupta et
  al., 2016).
- **Kronecker cumulative-sum reparametrisation.** It imposes nonnegative cross
  differences (§3.3), which is why (M9) uses the max recursion.
  - **Confirmed in Pya & Wood (2015) §2.3.1.** Their double-monotone form is
    γ = (Σ₁ ⊗ Σ₂) β̃, with every element of β̃ but the first exponentiated. So
    $\gamma_{jk} = \sum_{j' \le j}\sum_{k' \le k} \tilde\beta_{j'k'}$, and every cross
    difference equals $\exp(\beta_{jk}) > 0$.
  - SCAM's own condition (adjacent coefficients nondecreasing) is only what they call
    "a sufficient condition". The reparametrisation then covers a strict subset of it.
- **Other Bayesian monotone approaches, as alternatives:**
  - Bernstein polynomials with ordered coefficients (Wang & Ghosh, 2012), whose tensor
    products extend to 2-D;
  - projection of an unconstrained Gaussian process onto the monotone cone (Lin &
    Dunson, 2014);
  - monotone P-splines with ordered coefficients under truncated priors (Brezger &
    Steiner, 2008). The sampler is Gibbs with univariate truncated normals for Gaussian
    responses, and Metropolis–Hastings with truncated-normal IWLS proposals otherwise.
    It covers 1-D only; 2-D monotone surfaces are not treated.

  - frequentist constrained regression splines with cone-projection inference for
    shape- and order-restricted GAMs (Meyer, 2018).

  All are workable. The chosen transform keeps NUTS on an unconstrained space.

### 6.5 Survey weights in Bayesian models

- **Weighted pseudo-likelihood** (arm B; Savitsky & Toth, 2016) gives consistent point
  estimates under conditions.
  - It needs a correctly specified curve model plus their design conditions A4–A6:
    bounded inverse inclusion, vanishing pairwise dependence, a constant sampling
    fraction.
  - The design side plausibly holds here: γ ≈ 8.6, and draws are uniform within class.
  - Its posterior spread is not design-calibrated without the Williams & Savitsky
    (2021) sandwich adjustment, in either direction (§3.6).
  - Savitsky & Toth motivate the method for analysts who "do not have access to the
    full design information", which is not our situation.
- **Modelling the design variable.** The model-based alternative is to condition on the
  variables the selection depends on. Gelman (2007): "models for survey responses should
  be constructed conditional on all variables that affect the probability of
  inclusion"; also Little (2004).
  - Gelman presents this as the direction the principles imply, not a solved problem.
    He calls the general reconciliation of weighting and regression open.
  - Here the selection variable (the LLM verdict) is observed on every row, so modelling
    it (arm A) makes selection ignorable for Bayesian inference. The weighting becomes
    implicit (§3.6). That is the recommended main arm.
- **The reverse factorisation.** Begg & Greenes' route is
  P(y | s) = Σ_v P(v | s) · P(y | v, s), with P(y | v, s) fit on gold and P(v | s) on
  all rows. It is the structure the production v4 estimator already uses (a class
  working model plus the phase-1 class mix). So the production comparator doubles as the
  check on arm A's forward direction; no extra arm is needed.
- **The price is model dependence.** If the measurement model (M12) is wrong, for
  instance if P(v | y) depends on something unmodelled that also drives y, arm A is
  biased where the design-based estimator is not.
  - The PPCs in §5.3 target this.
  - Arm B is kept as the model-light check.

### 6.6 Differential misclassification

- The forward (y → v) model is the Dawid & Skene (1979) and diagnostic-testing standard.
  - Their named assumptions are conditional independence of responses given the true
    class, and "no patient-by-clinician interaction".
  - Error rates are indexed by observer only, so they are implicitly common to all
    patients.
  - With one observer (the LLM) and no gold, their model would not be identified. Gold
    identifies θ here.
- (M12) relaxes the constant-error-rate assumption with the score slope β, identified
  from gold.
- **What is and is not assumed.**
  - θ_{g,y,v}(r) is by construction the *category-marginal* error rate at score r. So
    leaving category out does not break the factorisation p(y, v | s) = p(y | s) ·
    p(v | y, s). Nor does it break ignorability, since phase 2 selected on v only.
  - The assumption that carries weight is the **functional form**: logit-linear in r
    with partial pooling. This is the Little (2004) design-consistency condition (§3.6),
    which the misspecified-truth rounds in §5.2 test.
  - Category matters for the known calibration weakness (6.1c) and for transport. It is
    not a validity condition of arm A.

### 6.7 The gold standard is imperfect

- Desk-verified human labels cannot see an empty storefront (v4 §8), and κ on the random
  slice is 0.63 between LLM and human.
- Treating y as error-free makes every curve an estimate of P(human says exists).
- **The human reviewer was not blind to the LLM.** The review export
  (`openpois-validator` `audit_select.VETTING_COLUMNS`) shows the reviewer the LLM
  verdict, its confidence, the one-line reason, the research summary, the per-check
  results and `evidence_json`.
  - Begg & Greenes require verification "both independent of the test result and
    definitive", and exclude "a judgmental composite of all available information
    including the test result".
  - Anchoring would bias θ toward agreement, so the LLM would look more accurate than
    it is (κ = 0.63 is then an upper bound). Arm A would then lean too hard on the
    silver rows.
  - The v4 estimator has the same exposure through its class working models.
  - So does arm C. Its β_ℓ prior comes from the concordance slice, whose human labels
    were also made with the LLM output in view (§3.5b).
  - Measuring this would need a blinded re-review. That was considered and not funded
    (decision 8), so the limitation stands.
- A human-error layer is identifiable only with a third instrument, i.e. the phone/field
  anchor or a blinded re-review. It is noted, not proposed.

### 6.8 Phase-1 urban/rural balancing

- Phase-1 sampling balanced urban/rural within score bins. If existence depends on
  urbanicity given score, the sample's urbanicity mix at each score differs from the
  population's, and a score-only model estimates a sample-mix curve.
- The v4 estimator shares this assumption.
- The model-based fix is to add urbanicity to F's linear predictor and post-stratify to
  population shares (MRP; Gelman & Little, 1997). That needs the population's
  urbanicity by segment × score bin, which the handoff does not carry. Deferred to
  November (decision 7).
- Begg & Greenes' "referral bias" section makes the same argument for diagnostic tests:
  condition on X, then re-marginalise with the target population's distribution of X.
- Ignorability (§3.6) does not extend to phase 1 here: the balancing variable is not in
  the model (Rubin, 1976).

### 6.9 Truth is not monotone in Overture-only confidence

- v4 measured a dip: 0.68 below 0.25, 0.50 at 0.50–0.70, 0.94 above 0.98.
- Under misspecification, a monotone-constrained posterior concentrates on the
  KL-closest monotone curve, which here is a floor plus a rise, as the production PAV
  already produces.
- The PPC by knot interval will show the residual. Monotonicity is a product
  requirement, so this is accepted, not fixed.

### 6.10 Evaluation choices

- **Proper scores.** Brier and log score are strictly proper (Gneiting & Raftery, 2007).
  For Bernoulli outcomes, OOS LPD and log score coincide at the posterior-mean
  prediction (§5.4), so the two "different" metrics are one metric plus its square-loss
  cousin.
- **Survey-design CV.** Folds should respect the design's strata, and held-out scores
  should be design-weighted (Wieczorek et al., 2022). Folding within refined
  class and weighting by 1/π does both.
- **Scoring the comparator.** Scoring it through its 40-bin lookup handicaps it slightly
  (pool DSC 0.0074 → 0.0059 from binning alone), so both the published and curve
  versions are reported and the verdict states which one it rests on.

### 6.11 Coverage of Bayesian credible bands under shape constraints

- Credible intervals for monotone regression do not automatically have frequentist
  coverage.
- **Chakraborty & Ghosal (2021), 1-D projection-posterior.**
  - An unconstrained step-function posterior is PAVA-projected. Its pointwise interval at
    an interior point with f′(x₀) > 0 has asymptotic coverage free of nuisance
    parameters.
  - Coverage is *above* nominal (a 95% interval covers ≈ 96.5%; 93.2% credibility gives
    95%). The "above nominal" finding comes from Monte Carlo tables, not a proof.
  - It holds because they **undersmooth**: many bins, J ≫ n^{1/3}, so bias is negligible
    and the isotonisation does the regularising.
- **Wang & Ghosal (2023), multivariate case.**
  - They use an *immersion* posterior (block max-min), because the L2 projection has no
    tractable limit for d ≥ 2.
  - Coverage is again slightly above nominal (≈ 97.4% for a 95% interval at d = 2),
    with recalibration tables.
  - The design-density-free result needs *every* partial derivative positive at x₀. A
    locally flat direction (the substitutive high-Overture band of F3) is covered only
    in general form, without tables.
  - In their finite-sample study, unadjusted 95% intervals *under*-covered at n = 200
    for 4 of 5 test functions (91–97%). Recalibration made that worse (87–94%). The
    frequentist confidence intervals of Deng et al. (2021), "DHZ", were more accurate
    at small n.
- **Consequence for this plan.** Neither result transfers:
  - Our posterior is a direct constrained-prior posterior, with a fixed K = 12–15 below
    n^{1/3} and a smoothing prior. That is the regime where bias is comparable to
    variability and credible sets tend to **under**-cover: the "Cox phenomenon" (Cox,
    1993) that both papers contrast themselves with.
  - The outcomes are Bernoulli with a measurement layer, not homoscedastic Gaussian
    with a conjugate posterior.
  - So the band should not be presumed conservative. The fake-data study (§5.2)
    measures coverage by region on our design.
  - A PAVA-projected unconstrained fit is a possible *comparator* there, not a
    replacement: its coverage theory assumes a Gaussian working model, and the gold
    counts (444–1,026) sit in the range where Wang & Ghosal saw under-coverage.

### 6.12 Complexity vs data

- 195 surface coefficients against 444 gold is only viable because (i) the smoothing
  prior controls the effective degrees of freedom and (ii) 2,056 silver rows carry
  near-gold information through θ.
- Unconstrained by smoothness, 2-D isotonic estimation is slow: the worst-case risk of
  the bivariate isotonic least-squares estimator is of order n^{-1/2}, against n^{-2/3}
  in 1-D, and it does not adapt to an additive truth (Chatterjee et al., 2018). That is
  why F3 leans on the smoothing prior.
- Report the effective number of parameters (p_loo from PSIS-LOO; Vehtari et al.,
  2017) per curve, as a guard against a surface that is quietly the prior.

---

## 7. Budget and running order (as executed)

Numbers in parentheses in this section are execution-log entries.

1. Code, 16 tests and the `jax_core` passthrough, plus a `pr-code-reviewer` pass (eight
   findings, all adopted; decision 3).
2. Sampler benchmarks and reparameterisations: the data-centre anchor, the centred
   sum-to-zero measurement layer and the smooth max (decisions 4–9).
3. Main fits of arms A, B and C. Arm A failed; four structure variants were tested, and
   none removed its bias (decisions 11–13). Arm C became the main model (decision 12).
4. CV of arms C and B, 10 folds each; the arm A CV was deferred (decision 16).
5. The coverage study (50 fits) and sensitivity runs S2C–S7 (10 fits). They were paused
   for CPU and the machine restarted; the driver resumed them (decisions 17–18).
6. `fit_report.md` and the results summary (decision 19). Nat's answers and the switch
   to fixed silver-label rates followed (decisions 20–22). The fixed-rate model is wired
   in but **not refitted**.

Wall time was about 3 days of mostly unattended compute on 12 CPU cores. Main arm C
takes about 50 min at 4 × (1,000 + 1,000); a light arm C fit about 30–45 min under
contention.

---

## 8. Plan: ≤ 1,000 more LLM-checked observations for the October run (design only)

**Aim.** Buy the most reduction in published-band width per LLM check.

1. **No drift anchor** (decision 10). All ≤ 1,000 LLM checks go to new rows.
   - Record `template_version` on every new row.
   - **The October round pools with July** (decisions 17, 20): set
     `conflation.calibration.pooled_rounds: ["20260730"]` when `versions.calibration`
     moves to the October round. Both rounds' phase-1 rows enter the curve fit, and
     both rounds' gold enters the silver-label rates, each round under its own design
     weights (§3.5b).
   - Before pooling, run two drift checks. **LLM drift:** compare the per-round rates,
     `silver_label_rates` on each round alone; a per-(segment, verdict) difference
     beyond binomial error, or a changed template, means the rounds' rates stay
     separate. **Curve drift:** compare each round's design-weighted gold rate by score
     bin (the HT check, decision 18); a systematic gap means the score's meaning moved
     (a matcher, CD or turnover-model change), and the rounds should not be pooled.
     These free checks replace the anchor.
   - The blinded human re-review was considered and not funded (decision 8). The §6.7
     limitation stands.
2. **Targeted new phase-1 rows: up to 1,000.** Drawn from the October conflated
   population
   (non-shadow, named).
   - **Strata:** h = segment × knot interval, with the 2-D matched strata being knot-cell
     groups.
   - **Allocation:** Neyman-style on the posterior,
     $n_h \propto N_h \cdot \text{SD}_{post}\big(m_g(s)\big)_h$, where $N_h$ is the
     production count. Floor 30 per stratum in the prior-dominated regions: matched
     Overture < 0.90, OSM < 0.58, overture-segment < 0.35.
   - **Allowed because** arm C's curves are conditional on s, so score-targeted phase-1
     sampling is ignorable (as it would be for arm A, §3.6).
     - The silver-label rates are unaffected: they are rates within verdict class, and
       phase 2 stays uniform within class.
     - The inclusion probabilities are still recorded, so population-weighted scores
       stay computable.
   - **Preposterior check before committing the allocation.** For 3 candidate
     allocations × 5 synthetic draws, simulate y and verdicts on the in_family generator
     of `simulate_bayes_recovery.py`, refit arm C, and compare the expected
     population-weighted band width. That is 15 light fits, about 2–3 h.
   - Put extra weight on the matched high-Overture band, where the bands under-cover
     (§12).
3. **Gold for the new unverifiables: full census** (decision 10). About 20% of new rows
   (~200) will be LLM-unverifiable, and an unverifiable verdict carries almost no
   information. So every one is human- or desk-resolved, as in July. Arm C also
   **requires** the census: a non-gold unverifiable row has no silver label, and
   dropping it would be selection on the verdict.
4. **Covariates to collect.** shared_label group and urbanicity on every new row, so
   the §6.1 grouped effects are estimable later (urbanicity is deferred to November,
   decision 7).
5. **Protocol invariants.**
   - Blindness to scores is preserved.
   - Phase-2 draws stay uniform within class, which keeps the v4 difference estimator
     valid as a fallback.
   - The new rows form a new round directory, `data/calibration/<round>/`, with its
     metadata.

It will be implemented in `openpois-validator` and here only after the October
conflation produces the population. **Nothing in this section runs in Phase 1.**

---

## 9. Decisions (Nat, 2026-09-27 to 2026-09-30)

| # | Decision |
|---|---|
| 1 | **Nugget.** Dropped (σ ≡ 0); §6.1. The published P(exists \| s) = expit(F(s)), and its band is the posterior band of that curve. |
| 2 | **Range bound.** Hard, on F, by the coefficient squash (M8, M10). |
| 3 | **Degree.** Quadratic B-splines. |
| 4 | **Data layer** (superseded by 12 and 17). Arm A is the main model: joint, no weights, 9 refined classes pooled across segments, differential β. Arm B is the comparator. β ≡ 0 is sensitivity run S2. |
| 5 | **Knots.** Deciles of the phase-1 validation scores (§2). |
| 6 | **Headline comparator.** The interaction model as published (40-bin lookup); the curve is reported alongside. |
| 7 | **Urbanicity.** Not now; reconsider in November (§6.8). |
| 8 | **Blinded human re-review.** Skipped; the §6.7 limitation stands. |
| 9 | **Pooled CV metrics.** Equal weight per segment; per-segment metrics always reported. |
| 10 | **October plan.** Full census of new LLM-unverifiables; no drift anchor (§8). |
| 11 | **Arm C** (simple silver layer, §3.5b) runs in CV beside A and B, to test whether the full measurement layer adds out-of-sample performance. |
| 12 | **Arm C is the main model** (2026-09-28). Arm A's full-data fit failed convergence and ran biased low against the design-weighted gold rate on the matched segment (0.871 vs 0.906); arms B and C tracked it. CV: C (main), B and the best arm A variant. Sensitivity runs on arm C, with S2 and S3 replaced by S2C (asymmetric label noise) and S3C (β = 1; β ~ Beta(9, 1)). The coverage study fits arm C (execution log, decisions 11–12). |
| 13 | **Arm A comparator variant** (2026-09-28). No structural variant removed arm A's bias (3 classes, 6 merged, β ≡ 0, unpooled: matched 0.864–0.874 vs 0.906). The merged-6 variant (best convergence) was the CV comparator (execution log, decision 13). |
| 14 | **Arm A CV deferred** (Nat, 2026-09-28): the full-data evidence settles C over A. |
| 15 | **CV intervals by stratified bootstrap only** (Nat, 2026-09-28). Posterior draws carry the curve's uncertainty; the bootstrap carries the evaluation sample's. |
| 16 | **Q-a, Q-b accepted** (Nat, 2026-09-30): the data-centre anchor with α ~ N(0, 1.5²), and the centred sum-to-zero measurement layer (arm A). |
| 17 | **Arm C's silver-label accuracy is data, not a parameter** (Nat, 2026-09-30): fixed segment × verdict rates from the design-weighted gold (M15c'–d'), folding in each new round's gold (`concordance_rounds`, renamed `pooled_rounds` under decision 20). Wired in; not refitted (§3.5b). |
| 18 | **Horvitz–Thompson check becomes a standard run-cycle output** (Nat, 2026-09-30): designed in TODO.md, implemented in the October run. |
| 19 | **Matched band under-coverage (§12)**: TODO for model tweaks; skip in October, pick up before November. |
| 20 | **The curves pool validation rounds too** (Nat, 2026-10-01), not only the silver-label rates: `pooled_rounds` adds earlier rounds' phase-1 rows to the fit, each round under its own design (§3.5b). CV scores the current round only. Wired in; not run. |
| 21 | **Silver-label rate cells stay segment × verdict** (Nat, 2026-10-01), not the finer verdict × LLM-confidence classes. |
| 22 | **The HT check never fails a run** (Nat, 2026-10-01). It produces a review document with graphics (PDF) and marks bins where the deployed curve sits more than ±1 SD from the bin's design-weighted exists/checked rate (TODO.md). |
| 23 | **CHANGELOG** (Nat, 2026-10-01): first no entry while the prototype does not change published data, then reversed the same day. An "evaluated, not deployed" entry now sits under Unreleased. |
| 24 | **The fixed-rate mixture is an October test model** (Nat, 2026-10-01), beside the fractional-label arm C (§3.5c). |
| 25 | **q does not vary with score** within a segment × verdict cell (Nat, 2026-10-01). |
| 26 | **HT flag reference confirmed** (Nat, 2026-10-01): the design-weighted exists/checked ratio with SD from the Kish ESS, the raw ratio alongside, and the share flagged reported against the 32% (1 SD) and 5% (2 SD) chance rates. |

Decision numbers in this table (Nat's decisions) are not the execution-log numbers.
The log numbers every implementation choice separately. Throughout this document,
"decision N" means this table and "log N" means execution-log entry N; §7, a record of
the execution, uses log numbers only.


---

## 10. References

Format: authors, year, title, journal, volume(issue), pages, DOI, then the
`~/data/library/papers/<ID>/` library ID in brackets. Page ranges were checked against
the library full text or the publisher record.

- Begg, C. B., & Greenes, R. A. (1983). Assessment of diagnostic tests when disease
  verification is subject to selection bias. *Biometrics*, 39(1), 207–215.
  https://doi.org/10.2307/2530820 [JQ9SUSRF]
- Breidt, F. J., & Opsomer, J. D. (2017). Model-assisted survey estimation with modern
  prediction techniques. *Statistical Science*, 32(2), 190–205.
  https://doi.org/10.1214/16-STS589 [NN7VFFNZ]
- Brezger, A., & Steiner, W. J. (2008). Monotonic regression based on Bayesian
  P-splines: An application to estimating price response functions from store-level
  scanner data. *Journal of Business & Economic Statistics*, 26(1), 90–104.
  https://doi.org/10.1198/073500107000000223 [Y8VRI8HJ]
- Chakraborty, M., & Ghosal, S. (2021). Coverage of credible intervals in nonparametric
  monotone regression. *The Annals of Statistics*, 49(2), 1011–1028.
  https://doi.org/10.1214/20-AOS1989 [ZX5VTP5T]
- Chatterjee, S., Guntuboyina, A., & Sen, B. (2018). On matrix estimation under
  monotonicity constraints. *Bernoulli*, 24(2), 1072–1100.
  https://doi.org/10.3150/16-BEJ865 [LU9E6STB]
- Cox, D. D. (1993). An analysis of Bayesian inference for nonparametric regression.
  *The Annals of Statistics*, 21(2), 903–923. https://doi.org/10.1214/aos/1176349157
  [not in library]
- Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer
  error-rates using the EM algorithm. *Journal of the Royal Statistical Society,
  Series C (Applied Statistics)*, 28(1), 20–28. https://doi.org/10.2307/2346806
  [NKE2LI6T]
- Deng, H., Han, Q., & Zhang, C.-H. (2021). Confidence intervals for multiple isotonic
  regression and other monotone models. *The Annals of Statistics*, 49(4), 2021–2052.
  https://doi.org/10.1214/20-AOS2025 [not in library]
- Dimitriadis, T., Gneiting, T., & Jordan, A. I. (2021). Stable reliability diagrams
  for probabilistic classifiers. *Proceedings of the National Academy of Sciences*,
  118(8), e2016191118. https://doi.org/10.1073/pnas.2016191118 [Z8WUAVS4]
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and
  penalties. *Statistical Science*, 11(2), 89–121. https://doi.org/10.1214/ss/1038425655
  [GDGCRR5F]
- Gelman, A. (2007). Struggles with survey weighting and regression modeling.
  *Statistical Science*, 22(2), 153–164. https://doi.org/10.1214/088342306000000691
  [FFA7IDFS]
- Gelman, A., & Little, T. C. (1997). Poststratification into many categories using
  hierarchical logistic regression. *Survey Methodology*, 23(2), 127–135.
  [not in library]
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
  https://doi.org/10.1198/016214506000001437 [6TTSWEQP]
- Gupta, M., Cotter, A., Pfeifer, J., Voevodski, K., Canini, K., Mangylov, A.,
  Moczydlowski, W., & van Esbroeck, A. (2016). Monotonic calibrated interpolated
  look-up tables. *Journal of Machine Learning Research*, 17(109), 1–47.
  https://jmlr.org/papers/v17/15-243.html [LU9U52MZ]
- Henry, N. (n.d.). *Model comparison* [Vignette for the mbg R package]. Retrieved
  September 27, 2026, from
  https://henryspatialanalysis.github.io/mbg/articles/model-comparison.html
- Hui, S. L., & Walter, S. D. (1980). Estimating the error rates of diagnostic tests.
  *Biometrics*, 36(1), 167–171. https://doi.org/10.2307/2530508 [not in library]
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation
  and model selection. In *Proceedings of the 14th International Joint Conference on
  Artificial Intelligence* (Vol. 2, pp. 1137–1143). Morgan Kaufmann. [not in library]
- Lang, S., & Brezger, A. (2004). Bayesian P-splines. *Journal of Computational and
  Graphical Statistics*, 13(1), 183–212. https://doi.org/10.1198/1061860043010
  [DMJUAEAQ]
- Lin, L., & Dunson, D. B. (2014). Bayesian monotone regression using Gaussian process
  projection. *Biometrika*, 101(2), 303–317. https://doi.org/10.1093/biomet/ast063
  [not in library]
- Little, R. J. (2004). To model or not to model? Competing modes of inference for
  finite population sampling. *Journal of the American Statistical Association*,
  99(466), 546–556. https://doi.org/10.1198/016214504000000467 [EZILDGB4]
- Louis, T. A. (1982). Finding the observed information matrix when using the EM
  algorithm. *Journal of the Royal Statistical Society, Series B (Methodological)*,
  44(2), 226–233. https://doi.org/10.1111/j.2517-6161.1982.tb01203.x [U2LXHF6H]
- McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.).
  Chapman & Hall. https://doi.org/10.1007/978-1-4899-3242-6 [not in library]
- Meyer, M. C. (2018). A framework for estimation and inference in generalized additive
  models with shape and order restrictions. *Statistical Science*, 33(4), 595–614.
  https://doi.org/10.1214/18-STS671 [QW4BNVM9]
- Orchard, T., & Woodbury, M. A. (1972). A missing information principle: Theory and
  applications. In L. M. Le Cam, J. Neyman, & E. L. Scott (Eds.), *Proceedings of the
  Sixth Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1,
  pp. 697–715). University of California Press. [KDGF5K5J]
- Pya, N., & Wood, S. N. (2015). Shape constrained additive models. *Statistics and
  Computing*, 25(3), 543–559. https://doi.org/10.1007/s11222-013-9448-7 [BQ9DMLKA]
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.
  https://doi.org/10.1093/biomet/63.3.581 [PF6EIW45]
- Savitsky, T. D., & Toth, D. (2016). Bayesian estimation under informative sampling.
  *Electronic Journal of Statistics*, 10(1), 1677–1708.
  https://doi.org/10.1214/16-EJS1153 [RP9KR7VT]
- Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017). Penalising
  model component complexity: A principled, practical approach to constructing priors.
  *Statistical Science*, 32(1), 1–28. https://doi.org/10.1214/16-STS576 [RWTQDFGP]
- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using
  leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5),
  1413–1432. https://doi.org/10.1007/s11222-016-9696-4 [HTLQ5H6M]
- Wang, J., & Ghosh, S. K. (2012). Shape restricted nonparametric regression with
  Bernstein polynomials. *Computational Statistics & Data Analysis*, 56(9), 2729–2741.
  https://doi.org/10.1016/j.csda.2012.02.018 [not in library]
- Wang, K., & Ghosal, S. (2023). Coverage of credible intervals in Bayesian
  multivariate isotonic regression. *The Annals of Statistics*, 51(3), 1376–1400.
  https://doi.org/10.1214/23-AOS2298 [EGRBCSD7; the local copy is the arXiv preprint]
- Wieczorek, J., Guerin, C., & McMahon, T. (2022). K-fold cross-validation for complex
  sample surveys. *Stat*, 11(1), e454. https://doi.org/10.1002/sta4.454 [MFPRX97A]
- Williams, M. R., & Savitsky, T. D. (2021). Uncertainty estimation for pseudo-Bayesian
  inference under complex sampling. *International Statistical Review*, 89(1), 72–107.
  https://doi.org/10.1111/insr.12376 [FHKPHVLG; the local copy is the 2019 arXiv
  preprint, "Bayesian uncertainty estimation under complex sampling"]
- Zeger, S. L., Liang, K.-Y., & Albert, P. S. (1988). Models for longitudinal data: A
  generalized estimating equation approach. *Biometrics*, 44(4), 1049–1060.
  https://doi.org/10.2307/2531734 [Y8H9NZRU]

**Internal:**
- v4 writeup: `~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md`
- Matched-modes paper: `~/data/library/writeups/2026-09-26-openpois-matched-segment-modes.md`
- Prior literature evidence:
  `~/data/library/_staging/writeup_2026-09-26_matched-modes-critique/evidence/`

---

## 11. Results (Phase 1, round 20260730)

These results come from the arm C **first run** (estimated symmetric β, posterior 0.9994
[0.9979, 1.0000]). The fixed-rate model (§3.5b, M15c') is wired in but not yet fitted; the
evidence in §11.2 predicts it will sit about 0.01 lower on Overture and matched. Full
tables are in `fit_report.md`; decision numbers refer to the execution log.

### 11.1 Convergence (main fits, 4 × 1,000 + 1,000)

| arm | parameters | max R̂ | min ESS bulk / tail | divergences | verdict |
|---|---|---|---|---|---|
| C (main) | 225 | 1.006 | 1,349 / 1,053 | 21 / 4,000 | passes all but zero divergences |
| B | 224 | 1.010 | 699 / 292 | 150 / 4,000 | comparator; steep-slope tail (log 10) |
| A | 354 | 1.090 | 35 / 15 | 128 / 4,000 | fails; sparse y = 0 classes (log 11) |
| A, merged-6 variant (light) | 306 | 1.024 | 375 / 449 | 5 / 1,600 | converges, still biased |

### 11.2 Calibration against the design-weighted gold rate

These are Hájek rates over the validation rows (w = 1/π_class), set against each model's
mean posterior-mean P(exists) over the same rows:

| segment | gold rate | production (Oct config) | arm C | arm B | arm A |
|---|---|---|---|---|---|
| Overture | 0.663 | 0.663 | 0.676 | 0.658 | 0.649 |
| OSM | 0.817 | 0.817 | 0.817 | 0.812 | 0.807 |
| matched | 0.906 | 0.907 | 0.915 | 0.909 | 0.871 |

- **Arm A** is low throughout; its matched lowest raw-score quintile is 0.760 against
  0.863.
- **Arm C** sits high by exactly the amount that its silver labels are wrong. The gold
  concordance (exists verdicts wrong 5/181, 1/191 and 3/231; gone wrong 0/148, 3/104
  and 0/59) implies −0.015, −0.001 and −0.010 for Overture, OSM and matched. That is the
  evidence behind the fixed rates (log 20).

### 11.3 Ten-fold cross-validation

Design-weighted held-out scores. The pooled figures weight each segment equally. The
intervals are from the paired stratified bootstrap (fold × refined class, 2,000
replicates).

| comparison | Overture | OSM | matched | pooled |
|---|---|---|---|---|
| **C vs production as published: Brier ratio** | 0.992 [0.982, 1.003] | 1.004 [0.998, 1.010] | 0.994 [0.978, 1.009] | **0.996 [0.990, 1.002]** |
| C vs production as published: ΔLPD per row | +0.008 [−0.000, +0.016] | −0.001 [−0.004, +0.001] | +0.001 [−0.004, +0.005] | +0.003 [−0.001, +0.006] |
| C vs production curve: Brier ratio | 0.993 | 1.004 | 1.007 | 0.999 [0.993, 1.006] |
| C vs B: Brier ratio | 0.994 [0.988, 0.999] | 1.000 | 0.969 [0.952, 0.986] | 0.991 [0.986, 0.997] |

- OOS RMSE, pooled: C 0.3703; production published 0.3709; curve 0.3702; B 0.3722.
- The mbg-style summed LPD: C −1117.1; production published −1123.5; curve −1122.1;
  B −1126.7.
- **Reading.** Arm C ties production on out-of-sample accuracy (every interval
  includes 1), while offering a generative model, posterior bands and a smooth 2-D
  surface. It clearly beats the gold-only weighted arm B, most on matched.

### 11.4 Coverage of the 95% posterior band (arm C fitted; `simulate_bayes_recovery.py`)

| scenario | Overture | OSM | matched | bias (Overture / OSM / matched) |
|---|---|---|---|---|
| in_family (20 rounds) | 0.953 | 0.906 | **0.752** | −0.000 / −0.001 / −0.001 |
| realistic LLM (arm A's θ; 5) | 0.794 | 0.890 | 0.268 | +0.021 / +0.010 / +0.018 |
| category confounder (5) | 1.000 | 1.000 | 0.676 | −0.001 / −0.001 / −0.001 |
| step at the atom (5) | 0.956 | 0.805 | 0.788 | −0.006 / +0.008 / +0.002 |

- **Arm B on the misspecified rounds:** matched coverage 0.81 (realistic), 0.83
  (category) and 0.72 (step), with |bias| ≤ 0.006. The production bootstrap bands, for
  comparison, simulated at 0.89 / 0.91 / 0.66 (matched / OSM / Overture).
- **Realistic scenario.** Arm C's upward bias there comes from treating lopsided LLM
  errors as absent. The fixed rates (M15c') remove the part of it that the gold
  concordance measures. Arm A's θ is itself from a biased fit and may exaggerate the
  real errors.

### 11.5 Sensitivity

See §5.5. Every structural and prior change moves arm C's curves by 0.006 or less at
the validation rows. Estimated asymmetric noise does not (log 15).

### 11.6 Deployed-impact preview (20260902 population, unflagged rows, arm C vs the October production curves)

| segment | POIs | mean abs Δ | share abs Δ > 0.05 | share abs Δ > 0.10 |
|---|---|---|---|---|
| Overture | 11.05 M | 0.038 | 12.5% | 3.8% |
| OSM | 1.27 M | 0.010 | 0.9% | 0.1% |
| matched | 1.82 M | 0.017 | 5.5% | 0.8% |

On Overture the change is concentrated at two atoms:
- 0.919912 (39% of POIs): 0.726 → 0.679;
- 0.990219 and 1.0 (about 10%): 0.827 → 0.923. That is toward the top quintile's gold
  rate of 0.927.

The site colours confidence on a continuous gradient in 0.05 steps, so these are
one-step and two-step colour shifts (log 20).

---

## 12. Open items and limitations

1. **The matched 95% band under-covers** (0.75 in-family; 0.67 in the flat
   high-Overture region). TODO for model tweaks, before November (Nat's decision 19).
   Diagnosis (execution log, decision 20):
   - The bands have the right width for sampling noise: half-width 0.018 against
     1.96 × the estimate's SD, 0.0165.
   - Coverage fails through **pointwise smoothing bias**: corr(coverage,
     |bias| / half-width) = −0.96, and bias is 84% of the half-width in the flat region.
   - The worst points are where the true surface climbs steeply to its ceiling of about
     0.94–0.96 (bias −0.03 to −0.09). The random-walk prior rounds that off, and the
     matched segment's large effective sample makes the band narrow enough for the
     rounding to fall outside it. This is the Cox (1993) phenomenon, for a direct
     constrained-prior posterior at a fixed, modest basis size (§6.11).
   - Candidates:
     - local flexibility near the ceiling (more knots in the high-Overture band, or
       locally adaptive smoothing variances; Lang & Brezger, 2004);
     - a credible level recalibrated by simulation, using the coverage study as the
       calibration map;
     - a coarser matched grid.
   - A looser τ_M alone barely moves the surface (S6_taumatched4: 0.003).
2. **The fixed-rate arm C has not been fitted.** The October run should fit it (pooling
   the July and October rounds, decision 20) and re-run the coverage study's in_family
   and realistic scenarios. The realistic-scenario bias should fall.
3. **The fixed-rate mixture** (§3.5c) is to be built and run beside arm C in October
   as a test model (decision 24): CV, in-family coverage and the tercile check. Its
   wider bands may bear on item 1.
4. **The Horvitz–Thompson check** (Nat's decision 18) is designed in TODO.md for the
   October run. It is the standing guard against the arm C bias mode in §11.2.
5. **Limitations that stand:**
   - human gold made with the LLM output in view (§6.7);
   - phase-1 urban/rural balancing not modelled (§6.8, deferred to November);
   - curves conditioned on score only, not category (production TODO);
   - fractional labels slightly overstate the information in silver rows (§3.5b);
   - q assumed constant in score within a (segment, verdict) cell.
6. **Production (Phase 2) is not requested.** Wiring arm C into deploy would need:
   - a lookup or grid export of the posterior-mean curve and band;
   - the edge rules of `calibration.calibrate_frame` (shadow CD, missing_conf,
     unnamed);
   - a decision on whether the published band is the posterior band as-is or
     recalibrated (item 1).
