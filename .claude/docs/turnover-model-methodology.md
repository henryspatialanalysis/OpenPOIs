# Turnover model methodology

Bayesian modeling of interval-censored change from a homogeneous Poisson process, with a zero-inflated extension for instant-change events. Theoretical derivation plus a mapping to the implementation in `src/openpois/models/`.

## Context

You have intermittently-observed individuals tracked for a single binary change event. Each observation at time `t` reports `0` (unchanged) or `1` (changed). The underlying process is assumed to be a **time-invariant (homogeneous) Poisson process** with unknown rate λ, so the waiting time to first change is `Exponential(λ)`.

**What an "individual" is.** An individual is a `(POI, name-iteration)` pair. The clock `t = 0` starts either (a) when the POI is first created, or (b) at the most recent change of the tag of interest ("name"). When a name changes, that individual's history ends (with `y = 1` at the observation immediately following the change) and a new individual — same POI, new name — is spawned with its own `t = 0` anchored at the name-change event. This framing means (i) individuals are independent of each other conditional on the model parameters (a name change does not carry memory into the next iteration), and (ii) every individual has a well-defined starting time for the Poisson process.

Observation times differ across individuals, and the data contain:

- **Interval-censored events** — the first `1` arrived somewhere in the open-closed interval `(t_{k-1}, t_k]`; the exact change time is unknown.
- **Right-censored trajectories** — some individuals have only `0`s through their final observation; the change may happen later or never.

Additionally, a small fraction `δ ∈ (0, 1)` of individuals appear to have changed at (approximately) `t = 0` — i.e. there is a point mass at zero in the underlying distribution of change times. This pushes the base model from a pure `Exponential(λ)` to a **zero-inflated exponential** (§1.7). A log-odds prior `logit(δ) ~ N(-3, 1)` is specified a priori, implying a prior mean of `δ ≈ 5%`.

Goal: fit `λ` and `δ` (and eventually hierarchical extensions of `λ`) with honest uncertainty under MLE and Bayesian frameworks. This plan is a methodology writeup; it maps each statistical decision onto the openpois codebase where the pattern is already (or should be) implemented, so future code work can proceed against a clear theoretical frame.

## 1. Statistical framework for a single, constant λ

### 1.1 The individual likelihood, worked from first principles

For individual `i` observed at times `t_{i,0} < t_{i,1} < … < t_{i,n_i}`, let `T_i ~ Exponential(λ)` be the latent change time measured from `t_{i,0}`. Survival and CDF:

- `P(T_i > t) = exp(-λ·t)` (survival)
- `P(T_i ≤ t) = 1 - exp(-λ·t)` (cdf)

Two cases:

**Case A — right-censored** (no `1` observed through `t_{i,n_i}`):
```
L_i(λ) = P(T_i > t_{i,n_i} - t_{i,0}) = exp(-λ·(t_{i,n_i} - t_{i,0}))
```

**Case B — interval-censored event** (first `1` seen at `t_{i,k*}`; `0` at all prior observations):
```
L_i(λ) = P(T_i ∈ (t_{i,k*-1} - t_{i,0}, t_{i,k*} - t_{i,0}])
       = exp(-λ·(t_{i,k*-1} - t_{i,0})) · [1 - exp(-λ·(t_{i,k*} - t_{i,k*-1}))]
```

### 1.2 Memoryless decomposition into per-interval Bernoulli trials

The homogeneous Poisson process is memoryless: conditional on "no change by `t_a`", the residual waiting time from `t_a` is again `Exponential(λ)`. So for each inter-observation interval `(t_{k-1}, t_k]` with length `Δ_k = t_k - t_{k-1}` and binary outcome `y_k`,
```
P(y_k = 0 | λ, Δ_k) = exp(-λ·Δ_k)
P(y_k = 1 | λ, Δ_k) = 1 - exp(-λ·Δ_k)
```
Multiplying across intervals within an individual recovers the full likelihood in §1.1 exactly. Right censoring requires **no special indicator** — an individual with all `y=0` simply contributes `exp(-λ·Σ Δ_k) = exp(-λ·(t_{i,n_i} - t_{i,0}))`, which is the correct right-censored term.

**Consequence for the data format**: one row per `(individual, interval)` pair, carrying `(duration Δ, change_indicator y)`, is the natural representation and gives the correct likelihood for homogeneous Poisson with a *shared* λ. This is exactly what [src/openpois/osm/format_observations.py](src/openpois/osm/format_observations.py) builds and what the likelihood at [models/model_fitter.py:80-86](models/model_fitter.py#L80-L86) consumes.

### 1.3 Total log-likelihood

Stacking all rows `j = 1…N` (across individuals and their intervals):
```
log L(λ) = Σ_j [ y_j · log(1 - exp(-λ·Δ_j))  +  (1 - y_j) · (-λ·Δ_j) ]
```
Numerically: use `log(-expm1(-λΔ))` for the `y=1` term to avoid catastrophic cancellation when `λΔ` is small — this is the pattern already in the code.

### 1.4 MLE

`∂ log L / ∂λ = 0` yields
```
Σ_{j: y_j=1} [ Δ_j · exp(-λ·Δ_j) / (1 - exp(-λ·Δ_j)) ]  =  Σ_{j: y_j=0} Δ_j
```
No closed form. Solve with Newton-Raphson or Brent's method on `λ > 0`. Good starting value: the **Poisson approximation** that holds when all `λΔ_j` are small,
```
λ̂_approx = (total events) / (total exposure) = Σ y_j / Σ Δ_j
```
This is the classical exponential/Poisson MLE and is also the limit of the exact MLE as intervals shrink. For OSM-style data (years-long `Δ`s with λ on the order of 0.01–0.1/yr), the approximation is close but not identical; the exact likelihood should be used.

### 1.5 Bayesian inference

Priors — two defensible choices:

1. **Log-normal** on λ (i.e. `log λ ~ N(m, s)`). Heavier-tailed, plays well with hierarchical extensions (you can add a random-effect term directly on the log scale). This is the openpois convention (`log_lambda ~ N(0, 3)` in [ConstantModel](src/openpois/models/osm_models.py#L131-L171)). Recommended when you expect to extend to covariates / frailties.
2. **Gamma(α, β)** on λ. Conjugate in the uncensored-exponential case; not strictly conjugate under interval censoring, but a weakly-informative `Gamma(0.5, 0.5)` or `Gamma(1, small)` is common in the survival literature (Ding, Sun & Peace 2012, ch. on Bayesian methods).

Posterior: `p(λ | y, Δ) ∝ p(λ) · Π_j P(y_j | λ, Δ_j)`. No closed form under interval censoring. Inference options:

- **NUTS / HMC** — the openpois approach via BlackJAX in [models/jax_core.py](src/openpois/models/jax_core.py). Robust, self-tunes. Preferred.
- **Data augmentation Gibbs** — impute the latent change time `T_i` from its truncated-exponential conditional; given `T_i` the exponential likelihood is conjugate with `Gamma` and the full conditional for λ is closed-form. Useful for pedagogy and for very large datasets where the Gibbs step is cheap.
- **Laplace / variational** — faster, but tails of the λ posterior are under-covered with small N or rare events.

### 1.6 Poisson–exponential equivalence (useful mental model)

The exponential-survival log-likelihood equals — up to a constant — the log-likelihood of treating each interval's event count as `Poisson(λ·Δ)`. That is, the "one Bernoulli per interval" likelihood is the first-event restriction of a Poisson GLM with log-link and exposure offset `log Δ`. This equivalence is what lets many software packages (glm with `family=poisson`, offset = log-duration; Stan / rstanarm's `stan_surv(basehaz="exp")`) deliver the same estimates. Rodríguez's notes ([§7.2](https://grodri.github.io/glms/notes/c7s2)) state it directly: *"the log-likelihood for censored exponential data … coincides exactly (except for constants) with the log-likelihood that would be obtained by treating D as a Poisson random variable with mean λT."*

### 1.7 Instant-change mixture: zero-inflated exponential (the "δ" extension)

**Motivation.** Empirically, a small fraction of individuals change at (approximately) `t = 0`. A pure `Exponential(λ)` cannot put concentrated mass at zero, so a mixture is required.

**Model.** With probability `δ ∈ (0, 1)`, the individual changes instantly at `t = 0`; with probability `1 - δ`, the change time is `Exponential(λ)`. The latent change time follows a **zero-inflated exponential (ZIE)** distribution with CDF
```
F_ZIE(t) = δ · 1{t ≥ 0}  +  (1 - δ) · (1 - exp(-λ·t)),   t ≥ 0
```
so that
```
S_ZIE(t) = P(T > t) = (1 - δ) · exp(-λ·t),   t ≥ 0
```
This is the classical "atom at zero" / "inliers" / "defective" exponential studied in the reliability literature (see refs below).

**Parameterization.** Model `δ` on the log-odds scale:
```
logit(δ) = log(δ / (1 - δ)),   logit(δ) ~ N(-3, 1)
```
Prior mean `-3` implies `δ ≈ 4.7%` (prior mode), with 95% prior credible interval roughly `(0.5%, 30%)` — wide enough to let moderate data dominate without putting non-trivial mass on `δ > 0.5`. Adjust the scale if the true rate is believed to be sharper.

**Where δ enters the likelihood.** By the memoryless property of the exponential component, *only the first interval of each individual is affected*: conditional on surviving the first observation (`y_1 = 0`), the individual is known to be from the non-δ component, and subsequent intervals follow the standard exponential likelihood. Denote the first interval of individual `i` as interval `j = first(i)` with duration `Δ_{first}` and outcome `y_{first}`. Then:

| Interval type | `y=0` contribution | `y=1` contribution |
|---|---|---|
| First interval of an individual | `log(1 - δ) + (-λ·Δ)` | `log(1 - (1 - δ)·exp(-λ·Δ))` |
| Any subsequent interval | `-λ·Δ` | `log(1 - exp(-λ·Δ))` |

Derivation for the first interval uses `S_ZIE(Δ_1) = (1-δ)·exp(-λ·Δ_1)` for `y=0`, and `1 - S_ZIE(Δ_1) = 1 - (1-δ)·exp(-λ·Δ_1)` for `y=1` — i.e. the generic interval-censored form `L = S(t_L) - S(t_R)` applied to ZIE.

**Identifiability.** `δ` is identified by the **excess** first-interval change rate beyond what `exp(-λ·Δ_1)` predicts. Good identification requires (a) variability in first-interval durations `Δ_1` across individuals, and (b) a reasonable number of first-interval observations. If *every* first interval is the same length, `δ` and `λ` are weakly confounded and the posterior will rely on the prior.

**Implementation sketch (extends [model_fitter.py](src/openpois/models/model_fitter.py#L27-L91))**
- Add a column `is_first_interval` (bool) to the per-row observation DataFrame in [format_observations.py](src/openpois/osm/format_observations.py) — flagged as the earliest interval per individual (`groupby(id).transform('idxmin')` on `last_tag_timestamp`).
- Add `logit_delta` as a scalar parameter in [osm_models.py](src/openpois/models/osm_models.py) next to `log_lambda`.
- Split the log-likelihood sum into two masks (first vs. non-first) and apply the table above. The sufficient-statistics reduction still works on the non-first block; the first-interval block is `O(N_individuals)` which is tiny.
- Apply this at Level 0 **and** carry it through every higher-level model — `δ` is orthogonal to hierarchy / covariates on `λ`, so it can be bolted on to any of Levels 1-4.

**Verification for the δ parameter.** Simulate data with `λ_true = 0.05/yr` and `δ_true ∈ {0, 0.01, 0.05, 0.15}`, varying first-interval durations. Confirm posterior recovery and check that when `δ_true = 0` the posterior on `logit(δ)` concentrates toward `-∞` (or, equivalently, that the 95% posterior upper credible limit on `δ` is small).

**δ must propagate to predictions (important).** The existing `ModelFitter.predict()` at [model_fitter.py:334-375](src/openpois/models/model_fitter.py#L334-L375) computes `P(change by t) = 1 - exp(-λ·t)`. Under the ZIE model this must be replaced by `F_ZIE(t)` — but *which* form depends on what is being predicted:

| Prediction regime | Correct formula | When to use |
|---|---|---|
| "Fresh" individual at `t = 0` (no observation history) | `P(change by τ) = 1 - (1 - δ) · exp(-λ·τ)` | Population-level forecasts; predicting for a newly-created POI; any scenario where the δ-component has not yet been ruled out. |
| Individual already observed as unchanged through some duration `s ≥ first observation` | `P(change in next τ \| unchanged through s) = 1 - exp(-λ·τ)` | Rating an existing POI on the current snapshot: we have evidence the POI *wasn't* in the δ-component (it didn't change instantly), so conditional on that history the (1-δ) factor drops out. |

The second row follows from Bayes: `P(δ-component \| y_{first} = 0) = 0`, so conditional predictions recover the pure exponential form. The first row is the *unconditional* (marginal) prediction for a new individual drawn from the ZIE population.

**Concretely for openpois**: the existing predictions table (per-type `P(change in 1yr)`, `P(change in 5yr)`, etc. at [scripts/models/osm_turnover.py:164-198](scripts/models/osm_turnover.py#L164-L198)) should expose *both* formulas — a column like `p_change_{τ}_fresh` (row-1 formula) and `p_change_{τ}_conditional` (row-2 formula) — because downstream users may want either depending on context. The frontend "how likely is this POI to change" indicator likely wants the conditional form (these are existing POIs with a history of not-changing), while a system-wide forecast of how many POIs will turn over wants the fresh form.

Posterior-predictive implementation: both quantities are per-draw functions of `(λ, δ)`, so for each posterior draw `(λ^(s), δ^(s))`, compute both values and report the draws' mean / credible intervals as usual.

### 1.8 Cross-check against foundational sources

Each step of the derivation above was verified against the classical survival-analysis literature:

| Claim | Verified against |
|---|---|
| Right-censored likelihood is `S(t)` | Rodríguez §7.2 ([source](https://grodri.github.io/glms/notes/c7s2)): *"all we know … is that the lifetime exceeds `t_i`. The probability of this event is `L_i = S(t_i)`."* |
| Interval-censored likelihood is `S(t_L) - S(t_R)` | Klein & Moeschberger (2003) §4.1–4.2; Kalbfleisch & Prentice (2002) §3.1; standard across the field. |
| Exponential ↔ Poisson equivalence | Rodríguez §7.2 (direct quote above). The classic primary references are Holford (1980) "The analysis of rates and of survivorship using log-linear models" *Biometrics* 36, and Laird & Olivier (1981) "Covariance analysis of censored survival data using log-linear analysis techniques" *JASA* 76. |
| Memoryless property → per-interval Bernoulli decomposition | Follows from `P(T > t+Δ \| T > t) = exp(-λΔ)`; this is the *defining* property of the exponential. For a homogeneous Poisson process, the waiting time to the first event is exponential, so the trick is exact at Level 0. |
| Memoryless decomposition **fails** under individual frailty | Shared random effects induce intra-individual dependence even after conditioning on the global parameters — the correct handling is to sample the frailty, see §2 Level 3 and Balan & Putter (2020). |
| Zero-inflated exponential is an established construction | Dhar (2020) thesis, "Zero Inflated Exponential Distribution and its Variants"; Chaturvedi & Pati (2022) on inliers-prone distributions; Muralidharan (2000) "Lifetime model with discrete mass at zero and one." |
| Only the first interval carries the `(1-δ)` discount | Conditional on `y_{first} = 0`, posterior `P(\text{δ-component} \| y_{first}=0) = 0` because the δ-component changes at `t=0 < Δ_{first}`. So subsequent intervals are unaffected — derivation confirmed. |

Two items worth flagging that are *not* in scope for Level 0 but are good to note up front:

- **Informative censoring.** The likelihood above assumes observation times `t_{i,k}` are independent of the latent change time given covariates (non-informative censoring). If observation schedules are triggered by the event itself (e.g. edits happen because someone noticed the POI changed), the simple likelihood is biased. See *Zhang et al. (2018), "Maximum likelihood estimation for survey data with informative interval censoring"* ([AStA](https://link.springer.com/article/10.1007/s10182-018-00329-x)) for the corrected approach.
- **Left truncation.** If individuals only enter the dataset once they have been unchanged for some time (a prevalent-cohort bias), the likelihood must condition on `T > t_{entry}` — divide each contribution by `S(t_entry)`. Check this for the OSM case, since the dataset is a snapshot taken at the present moment.

## 2. Incremental extensions (in recommended order of complication)

### Level 0 — single constant λ (+ optional δ)
Status in openpois: implemented as [`ConstantModel`](src/openpois/models/osm_models.py#L131-L171). Prior: `log_lambda ~ N(0, 3)`. Inference: BlackJAX NUTS. **The core λ fit is done.**

**New work at Level 0**: add the `δ` extension from §1.7. Concretely, (a) mark the first interval per individual in [format_observations.py](src/openpois/osm/format_observations.py), (b) add `logit_delta` with prior `N(-3, 1)` to [osm_models.py](src/openpois/models/osm_models.py), and (c) split the log-likelihood in [model_fitter.py](src/openpois/models/model_fitter.py#L27-L91) into first-interval and subsequent-interval branches per the table in §1.7. `δ` should be carried through all higher levels since it is orthogonal to hierarchical structure on `λ`.

### Level 1 — group-wise random effects
```
log λ_g = log λ_0 + ε_g,   ε_g ~ N(0, σ²)
log σ   ~ N(loc, scale)    (hyperprior)
```
Partial pooling: rarely-observed groups shrink toward the pooled rate. Status in openpois: implemented as [`RandomByTypeModel`](src/openpois/models/osm_models.py#L177-L414), with both centered and non-centered reparameterizations. This is the canonical *Bayesian hierarchical* form of the model.

Extensions:
- **Multiple additive random effects + a nested interaction + geographic fixed effects — implemented (§7).** `RandomEffectsModel` generalizes `RandomByTypeModel` to a composable sum of amenity, MSA, amenity×MSA (nested, tight prior), and suburban/rural terms on `log λ`. See §7.2.
- Heavy-tailed random effects (`ε_g ~ t_ν` or Laplace) if a few groups are outliers — *not yet built*.

### Level 2 — covariates (exponential regression / Poisson GLM with offset)
```
log λ_i = β_0 + x_i^T β
```
Continuous and categorical covariates enter on the log scale. When the offset `log Δ_j` is added, this is literally a Poisson GLM and can be fit with `statsmodels.GLM(..., family=Poisson(), offset=log_dt)` for a quick MLE, or via the same JAX/NUTS pipeline for Bayesian inference. Not yet implemented.

### Level 3 — individual-level frailty
```
log λ_i = β_0 + x_i^T β + u_i,   u_i ~ N(0, τ²)
```
Captures unobserved individual heterogeneity beyond covariates. **Critical caveat**: the per-interval-row decomposition in §1.2 is only valid when rows from the same individual are conditionally independent given the shared parameters. A shared random effect `u_i` makes the rows from individual `i` correlated marginally, but **still independent conditional on `u_i`**. So MCMC that samples `u_i` alongside λ (i.e. Stan / NumPyro / BlackJAX with per-individual latent) handles this correctly — the per-interval rows just need a grouping key. The existing `RandomByTypeModel` machinery generalizes directly; you'd simply use `individual_id` as the group key if individuals have multiple intervals. Alternatively, if `u_i ~ Gamma`, the marginal over `u_i` gives a negative-binomial-like closed form (Balan & Putter 2020).

### Level 4 — non-homogeneous (time-varying) λ
```
λ(t) = exp(x^T β) · baseline(t)
```
**Piecewise-constant baseline** is the simplest non-homogeneous extension: partition time into `K` intervals, give each its own rate. The row decomposition still works, but any observation interval that *crosses a knot* must have its hazard integral split at the knot. The first such model — a single breakpoint on **tag age** — is now implemented; see §8.

## 3. Verification plan

1. **Simulation study (new, recommended at `tests/test_constant_lambda_simulation.py`)**
   - Generate `N` individuals with `T_i` drawn from ZIE(`λ_true, δ_true`); random observation schedules (e.g. Poisson-thinned observation times); mix of uncensored-event and right-censored individuals.
   - Fit `ConstantModel + δ`; confirm posterior 95% CI covers `λ_true` *and* `δ_true` in ≥ 94% of repetitions across varying `N`, `λ_true`, censoring fraction, and `δ_true ∈ {0, 0.01, 0.05, 0.15}`.
   - Sweep `Δ` magnitude: when `λ·Δ` is small, the exact MLE should approach `Σ y / Σ Δ`; when `λ·Δ` is large (heavy interval censoring), the exact MLE should diverge upward from the Poisson-approx MLE (a good diagnostic).
   - **Identifiability check**: confirm that `δ` and `λ` separate cleanly when first-interval `Δ` varies across individuals, and degrade gracefully to a weakly-identified posterior (wide on `logit(δ)`) when first intervals are all the same length.

2. **Closed-form sanity check**
   - All `Δ_j` equal: posterior under `Gamma(α, β)` prior is approximately `Gamma(α + Σ y, β + Σ Δ)` in the small-`λΔ` limit. Cross-check the NUTS posterior against this gamma.

3. **Prior sensitivity**
   - Refit under `N(0, 1)`, `N(0, 3)`, `N(0, 10)` on `log_lambda`; confirm the posterior is stable for data-rich regimes and that the prior dominates only where data is sparse.

4. **Reference-implementation cross-check**
   - Fit a subset of the data in PyMC (`pm.Potential` with the Bernoulli-on-exponential log-lik) and/or `rstanarm::stan_surv(..., basehaz="exp")`; confirm agreement within MCMC noise. PyMC is the easiest option since it lives in Python and can hit the same parquet.

## 4. Mapping theory → code

### 4.1 What every equation in §1 corresponds to in the existing code

| Theoretical concept | Equation | File | Function / lines | Status |
|---|---|---|---|---|
| Per-individual data format (one row per inter-observation interval) | `(Δ_j, y_j)` rows | [src/openpois/osm/format_observations.py](src/openpois/osm/format_observations.py) | `_advance_scan_state` (lines 46-135); `format_observations` (lines 232-241) emit one row per tag-state transition | **Done**. Builds `(id, version, obs_timestamp, last_tag_timestamp, tag_value, last_tag_value, changed, …keep_keys)` |
| Derived duration `Δ_j` | `Δ_j = t_{obs} - t_{last_tag}` | [src/openpois/models/setup.py](src/openpois/models/setup.py) | `prepare_data_for_model` (lines 8-75) adds `tag_years` column; filters `tag_years ≤ 1e-6` | **Done** |
| Log-likelihood kernel, dense per-row | `y_j·log(1-exp(-λΔ)) + (1-y_j)·(-λΔ)` | [src/openpois/models/model_fitter.py](src/openpois/models/model_fitter.py#L80-L86) | `make_log_density` body, lines 80-86. Uses `log(-expm1(-rate))` for numerical stability | **Done** |
| Log-likelihood kernel, sufficient-statistics form | `-Σ_g λ_g · Σ_{j∈g, y=0} Δ_j  +  Σ_{y=1} log(1 - exp(-λ_{g(j)}Δ_j))` | [src/openpois/models/osm_models.py](src/openpois/models/osm_models.py#L359-L372) | `RandomByTypeModel._sufficient_stats_log_lik` | **Done**. `O(K + N_changed)` instead of `O(N)` |
| Level 0 prior on `log λ` | `log λ ~ N(0, 3)` | [src/openpois/models/osm_models.py](src/openpois/models/osm_models.py#L131-L171) | `ConstantModel.param_likelihood` | **Done** |
| Level 1 hierarchical prior (non-centered) | `log λ_g = log λ_0 + exp(log_σ)·ε_raw_g;  ε_raw_g ~ N(0, 1)` | [src/openpois/models/osm_models.py](src/openpois/models/osm_models.py#L350-L390) | `RandomByTypeModel._param_likelihood_non_centered` | **Done** |
| Bayesian posterior sampling (NUTS) | `p(params \| data) ∝ p(params) · L(params; data)` | [src/openpois/models/jax_core.py](src/openpois/models/jax_core.py#L104-L283) | `nuts_sample_multichain`, BlackJAX window adaptation | **Done**. Multi-chain via `vmap` |
| Prediction `P(change by τ)` | `1 - exp(-λ·τ)` | [src/openpois/models/model_fitter.py](src/openpois/models/model_fitter.py#L334-L375) | `ModelFitter.predict` | **Done but needs δ update** — see §4.3 |
| Right-censored observation handling | `exp(-λΣΔ)` via product of per-row `y=0` terms | *implicit* | No explicit censoring flag; the row decomposition carries it for free | **Done** (implicit, by construction) |
| Interval-censored observation handling | `exp(-λ(t_L - t_0)) · [1 - exp(-λ·Δ)]` via product of earlier `y=0` rows × final `y=1` row | *implicit* | Same mechanism | **Done** (implicit, by construction) |
| Zero-inflated mixture `δ` (§1.7) | `L_first = (1-δ)^{1-y}·(1-(1-δ)e^{-λΔ})^y · e^{-λΔ(1-y)}` | — | *Not in the code yet* — see §4.2 for the implementation plan | **To build** |
| Level 2: covariates `log λ = β_0 + x^T β` | — | — | Not implemented; needs a new `CovariateModel` class in `osm_models.py` | **To build** |
| Level 3: individual frailty | — | — | Could be expressed via `RandomByTypeModel` using `id` as the group key, but only works if `id` matches the per-POI-name-iteration "individual" | **To investigate** |
| Level 4: piecewise-constant baseline λ(t) | `H = λ_1·(crossing−t1) + λ_2·(t2−crossing)`, `crossing = clip(t_B, t1, t2)` | [src/openpois/models/osm_models.py](src/openpois/models/osm_models.py) | `_breakpoint_integrated_hazard`, `ConstantBreakpointModel` — closed-form integral on the tag-age clock (no row splitting); see §8 + [docs/time-varying-models.md](time-varying-models.md) | **Done (constant); parked** — nationwide breakpoint not identified |

### 4.2 Implementation plan for the ZIE δ extension (§1.7) at Level 0

Target change: add `δ` as a model parameter so `ConstantModel` becomes `ConstantModel + ZIE`, and have the addition carry through to `RandomByTypeModel` unchanged (since δ is orthogonal to the group-level hierarchy on λ).

**Step A — Flag the first interval per individual in the observation DataFrame.**

[src/openpois/osm/format_observations.py](src/openpois/osm/format_observations.py) currently emits one row per tag-state transition. Looking at `_advance_scan_state` (lines 46-135), each row carries `last_tag_timestamp`, which marks the start of the stable-name window. An "individual" in the statistical sense is a `(POI id, name-iteration)` — so "first interval of an individual" = the first emitted row after a given `last_tag_timestamp` value. Add:

```python
# After building the observations list, before returning:
df['is_first_interval'] = (
    df.groupby(['id', 'last_tag_timestamp'])['obs_timestamp']
      .transform('idxmin') == df.index
)
```

Test: `is_first_interval.sum()` should equal the number of unique `(id, last_tag_timestamp)` pairs.

**Step B — Add `logit_delta` as a model parameter in [osm_models.py](src/openpois/models/osm_models.py).**

In `ConstantModel.build_model` (lines 131-171), extend `starting_params`:
```python
self.starting_params = {
    "log_lambda": jnp.asarray(log_lambda_init),
    "logit_delta": jnp.asarray(-3.0),  # prior mean
}
```
And in `ConstantModel.param_likelihood`, add the prior for `logit_delta`:
```python
lp += stats.norm.logpdf(params["logit_delta"], loc=-3.0, scale=1.0)
```

**Step C — Add `is_first_interval` to the JAX data dict.**

In `ModelFactory.build_model` (called via `__init__`), populate `self.data['is_first_interval']` from the corresponding DataFrame column (as float for vector ops).

**Step D — Modify the log-likelihood to apply the `(1-δ)` factor on first-interval rows.**

The cleanest path is to use the `log_likelihood_fun` hook already present at [model_fitter.py:77-78](src/openpois/models/model_fitter.py#L77-L78) rather than modifying the dense fallback. In `ConstantModel.build_model`:

```python
def log_likelihood_fun(params, data, target):
    lam = jnp.exp(params["log_lambda"])
    dt = data["dt"]
    is_first = data["is_first_interval"]
    rate = lam * dt
    log_one_minus_delta = jax.nn.log_sigmoid(-params["logit_delta"])  # log(1 - δ)

    # Standard exponential terms (for non-first rows)
    log_p_std = jnp.log(-jnp.expm1(-rate))       # log(1 - exp(-rate))
    log_1mp_std = -rate                           # log(exp(-rate))

    # ZIE first-row terms
    # y=0: log((1-δ) · exp(-rate)) = log(1-δ) - rate
    log_1mp_zie = log_one_minus_delta - rate
    # y=1: log(1 - (1-δ) · exp(-rate))
    #       stable form: log(1 - exp(log(1-δ) - rate))
    log_p_zie = jnp.log(-jnp.expm1(log_one_minus_delta - rate))

    log_p = jnp.where(is_first, log_p_zie, log_p_std)
    log_1mp = jnp.where(is_first, log_1mp_zie, log_1mp_std)
    per_obs = target * log_p + (1.0 - target) * log_1mp
    return jnp.sum(per_obs)

self.log_likelihood_fun = log_likelihood_fun
```

(The `ModelFitter` constructor already accepts `log_likelihood_fun` — see [model_fitter.py:116](src/openpois/models/model_fitter.py#L116) — and passes it into `make_log_density`.)

**Step E — Propagate δ into predictions.**

[ModelFitter.predict](src/openpois/models/model_fitter.py#L334-L375) currently returns `1 - exp(-λ·τ)`. Change the signature to accept `(..., prediction_mode: "fresh" | "conditional" = "conditional")` and emit:
- `fresh`: `1 - (1 - sigmoid(logit_delta)) · exp(-λ·τ) = 1 - sigmoid(-logit_delta)·exp(-λ·τ)`
- `conditional`: `1 - exp(-λ·τ)` (unchanged; δ is ruled out by the conditioning)

Also expose both in the exports at [scripts/models/osm_turnover.py:164-198](scripts/models/osm_turnover.py#L164-L198) — new columns `p_change_{τ}yr_fresh` and `p_change_{τ}yr_conditional` (the latter replacing the current `p_change_{τ}yr`).

**Step F — Add a starting-value diagnostic.**

The data-driven init for `logit_delta` could be done by counting first-interval changes in the first few weeks/months, but the prior (`-3` ≈ 5%) is a reasonable starting point. For `log_lambda`, the empirical rate should be computed *excluding* first-interval rows to avoid the instant-change mass contaminating the λ estimate. Minor but worth noting.

**Step G — Tests.**

Add to `tests/` (see §3 verification plan): a simulation-based test that draws from ZIE with known `(λ_true, δ_true)`, fits, and checks coverage of both parameters. Also a regression test that fitting against a dataset with δ_true ≈ 0 returns `logit(δ)` posterior pushed toward `-∞`.

### 4.3 Compatibility of δ with higher-level models

The ZIE δ extension is fully additive. It only touches the first-interval branch of the likelihood; the λ-side (including all hierarchical structure) is untouched. For `RandomByTypeModel`:

- The sufficient-statistics reduction still applies on *non-first* rows (`is_first = False`).
- First-interval rows need the full dense branch (can't be collapsed to group-level sums because the first-interval structure is per-individual, not per-group). This adds `O(N_individuals)` to the likelihood cost — negligible compared to `O(N)` total rows.
- Concretely: `_sufficient_stats_log_lik` at [osm_models.py:359-372](src/openpois/models/osm_models.py#L359-L372) should gain a `first_interval_mask` branch that computes the ZIE terms on those rows and adds them to the sufficient-statistics result from non-first rows.

### 4.4 Pipeline-level file map

| File | Role in the pipeline |
|---|---|
| [src/openpois/osm/format_observations.py](src/openpois/osm/format_observations.py) | Builds the row-per-interval observation table from OSM full-history PBFs. The shape required by §1.2. **§4.2 Step A** modifies this. |
| [src/openpois/models/setup.py](src/openpois/models/setup.py) | `prepare_data_for_model` derives `tag_years` and filters. |
| [src/openpois/models/osm_models.py](src/openpois/models/osm_models.py) | `ConstantModel` (Level 0), `RandomByTypeModel` (Level 1), model registry. **§4.2 Steps B, C, D** modify this file. |
| [src/openpois/models/model_fitter.py](src/openpois/models/model_fitter.py) | Wraps models into a BlackJAX-ready log-density and runs NUTS; exports predictions. **§4.2 Step E** modifies `predict()`. |
| [src/openpois/models/jax_core.py](src/openpois/models/jax_core.py) | BlackJAX NUTS with window adaptation; multi-chain `vmap`. No changes needed. |
| [src/openpois/models/diagnostics.py](src/openpois/models/diagnostics.py) | R-hat, bulk-ESS, InferenceData conversion. Will include `logit_delta` automatically once added to `starting_params`. |
| [scripts/models/osm_turnover.py](scripts/models/osm_turnover.py) | End-to-end wrapper (config → data → fit → exports). **§4.2 Step E** extends the predictions table. |
| [config.yaml](config.yaml) | No change needed at Level 0 + δ; `model.type: constant` continues to work. (For Level 2+ work a `covariates:` block would be added.) |
| `tests/` | **§4.2 Step G** adds simulation-based recovery tests. |

## 5. Key references (organized by topic)

### Foundations: likelihood for interval-censored exponential / Poisson data
- **Cox (1972)**, "Regression Models and Life-Tables." *JRSS-B* 34(2):187-220. In your library as `56AB5N2C` — verified against the full text. Introduces the proportional-hazards model `λ(t; z) = λ_0(t) · exp(z^T β)` and the partial likelihood. §3 briefly notes the exponential case (`λ_0(t) = const`) as "the simplest possibility" but leaves parametric development aside. **Relevant to Level 2 (covariates) and the hazard-function framing — *not* the go-to reference for interval-censored likelihood.**
- **Kalbfleisch & Prentice (2002)**, *The Statistical Analysis of Failure Time Data*, 2nd ed., Wiley. Ch. 2–3 are the canonical derivation of parametric likelihoods under exact, right-censored, left-censored, and interval-censored observations. Use this as the primary reference for the material in §1.1–1.3 of this plan.
- **Klein & Moeschberger (2003)**, *Survival Analysis: Techniques for Censored and Truncated Data*, 2nd ed., Springer. §3.5 and §4.1–4.2 give the `L = S(t_L) - S(t_R)` interval-censored form used in §1.1 Case B.
- **Rodríguez, G.**, *GLMs lecture notes*, Princeton: [Ch. 7 — Survival Models (PDF)](https://grodri.github.io/glms/notes/c7.pdf) and [§7.2 — Censoring and the Likelihood Function](https://grodri.github.io/glms/notes/c7s2). Short, rigorous, and free; see §7.2 for the exact right-censored likelihood `L_i = S(t_i)` and the Poisson-equivalence statement used in §1.6.
- **Holford (1980)**, "The analysis of rates and of survivorship using log-linear models." *Biometrics* 36(2):299-305. Original derivation of the Poisson-GLM-with-offset equivalence for piecewise-exponential survival models.
- **Laird & Olivier (1981)**, "Covariance analysis of censored survival data using log-linear analysis techniques." *JASA* 76(374):231-240. Companion to Holford; shows the Poisson-likelihood identity holds exactly (not just approximately) for right-censored exponential data.
- **Duke STA-216 Lecture Notes**, "Likelihood Function for Censored Data" ([PDF](https://www2.stat.duke.edu/courses/Fall02/sta216/lecture14.pdf)). Compact treatment of Type I, Type II, random, and interval censoring.

### Zero-inflated / instant-failure / "atom at zero" exponential (Level 0, §1.7)
- **Muralidharan (2000)**, "Analysis of lifetime model with discrete mass at zero and one." [*Journal of Statistical Theory and Practice*](https://link.springer.com/article/10.1080/15598608.2017.1303407). Most direct reference: parametric lifetime model with an explicit point mass at zero, deriving the likelihood that underpins §1.7.
- **Dhar, S. (2020)**, "Zero Inflated Exponential Distribution and Its Variants." MSc thesis, UT Rio Grande Valley. [ScholarWorks](https://scholarworks.utrgv.edu/leg_etd/280/). Full MLE and Bayesian treatment of the ZIE distribution; has the exact CDF and log-likelihood in §1.7.
- **Chaturvedi & Pati (2022)**, "Inliers prone distributions: Perspectives and future scopes." [ResearchGate](https://www.researchgate.net/publication/358721181_Inliers_prone_distributions_Perspectives_and_future_scopes). Survey of the "inliers" framework — what the reliability literature calls a mass at zero (and occasionally also at one) in a lifetime distribution.
- **Saavedra et al. (2025)**, "Bayesian Analysis of Spatial Zero-Inflated and Right-Censored Survival Data." [*J. Agric. Biol. Environ. Stat.*](https://link.springer.com/article/10.1007/s13253-025-00682-w). Recent worked example combining a ZIE with hierarchical (spatial) random effects — directly parallel to Level 1 + §1.7 in this plan.
- **Maller & Zhou (1996)**, *Survival Analysis with Long-Term Survivors*, Wiley. Cure-model treatment, the *mirror image* of ZIE (atom at `∞` instead of `0`), but the MLE / Bayesian machinery carries across.

### Interval-censored Bayesian inference
- **Ding, Sun & Peace, eds. (2012)**, *Interval-Censored Time-to-Event Data: Methods and Applications*, CRC Press. [Open chapter PDF](https://dept.stat.lsa.umich.edu/~moulib/main-intcens-21st.pdf). The most comprehensive reference; Bayesian chapters cover gamma priors, data augmentation, MCMC, and nonparametric baselines.
- **Ahmed et al. (2021)**, "Bayesian Estimations of Exponential Distribution Based on Interval-Censored Data with a Cure Fraction." [*Journal of Mathematics* 2021:9822870](https://www.hindawi.com/journals/jmath/2021/9822870/). Direct parallel to Level 0 under gamma and LINEX priors.
- **Pan & Bai (2020)**, "A Bayesian approach for analyzing partly interval-censored data under the proportional hazards model." [PMC7592883](https://pmc.ncbi.nlm.nih.gov/articles/PMC7592883/). Mixes exact, right-, and interval-censored observations in one likelihood — mirrors your data structure.
- **Laurent R. (2019)**, "Bayesian Survival Analysis: Exponential Model." [Blog post with runnable Stan code](https://jonnylaw.rocks/posts/2019-08-09-bayesian_survival/). A concrete, minimal worked example for Level 0.

### Panel-count / recurrent-event extensions (if you ever move past "first event")
- **Cook & Lawless (2007)**, *The Statistical Analysis of Recurrent Events*, Springer. Definitive reference for recurrent events, panel counts, mixed Poisson processes.
- **Wang, Tong & Sun (2024)**, "Conditional modeling of panel count data with partly interval-censored failure event." [*Biometrics* 80(1):ujae020](https://academic.oup.com/biometrics/article/80/1/ujae020/7630879). Joint modeling of panel counts and an interval-censored terminal event — close to the OSM setting.
- **Zhu, Cook & Zeng (2022)**, "Bayesian Inferences for Panel Count Data and Interval-Censored Data with Nonparametric Modeling of the Baseline Functions." [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-88658-5_14).

### Hierarchical / frailty extensions (Level 3)
- **Balan & Putter (2020)**, "A tutorial on frailty models." *Statistical Methods in Medical Research* 29(11). [Open access](https://journals.sagepub.com/doi/full/10.1177/0962280220921889). The best single starting point for individual-level random effects.
- **Hougaard (1995)**, "Frailty models for survival data." [*Lifetime Data Analysis* 1:255-273](https://link.springer.com/article/10.1007/BF00985760). Foundational paper.
- **Gutierrez (2002)**, "Parametric frailty and shared frailty survival models." [*Stata Journal* 2(1):22-44 (PDF)](https://journals.sagepub.com/doi/pdf/10.1177/1536867X0200200102). Practical, parametric-focused — aligns with your exponential-baseline setup.

### Bayesian hierarchical modeling toolkits and textbooks
- **McElreath (2020)**, *Statistical Rethinking*, 2nd ed. In your library as `SQ6QYZX8`. Ch. 11 (Poisson GLMs), Ch. 13 (multilevel models), Ch. 15 (missing data / measurement error — useful framing for censoring as missingness).
- **Gelman et al. (2013)**, *Bayesian Data Analysis*, 3rd ed. Canonical reference; see Ch. 5 (hierarchical models), Ch. 21 (survival analysis).
- **Brilleman et al. (2020)**, "Bayesian Survival Analysis Using the rstanarm R Package." [arXiv:2002.09633](https://arxiv.org/pdf/2002.09633). Exponential, Weibull, piecewise-constant baselines with interval censoring — excellent spec for what you already have and want to add.
- **INLA book**, ch. 10 — [Survival Models](https://becarioprecario.bitbucket.io/inla-gitbook/ch-survival.html). Alternative to MCMC; useful if NUTS ever becomes a bottleneck.

## 6. Confirmed framing

Confirmed in-plan:
- **Scope** — methodology writeup + mapping to the existing openpois code. The plan above reflects this: each level states the math and then points at the file(s) where it lives (or should live).
- **Event type** — terminal / first-event only. `T_i ~ Exponential(λ)` is the right primitive; recurrent-event methods are *not* required at any level.
- **What an "individual" is** — a `(POI, name-iteration)` pair. Every time the name changes, the POI spawns a new individual with its own clock anchored at the name-change event. Clock `t = 0` is either POI creation time or the most recent change of the name tag.
- **Time origin `t_{i,0}`** — falls out of the individual definition above. In the existing code, `last_tag_timestamp` at [format_observations.py](src/openpois/osm/format_observations.py) serves as the per-individual `t = 0`, which is the correct anchor.
- **The δ-component has a plausible interpretation.** In this POI-name setup, the δ-component corresponds to *names that change almost immediately after being set* — the natural candidates being vandalism reverts, typo fixes, and duplicate-edit clean-up. This is a meaningful separate data-generating process, not just a nuisance parameter, and the mixture structure of §1.7 captures it cleanly.

**No open methodology questions remain.** The next step is to implement §1.7 at Level 0 (plus propagating both predictions forms from §1.7) before moving on to the Level 2+ extensions that are not yet built.

> **Since this writeup**, the ZIE δ extension (§1.7) and a substantial Level‑1 generalization have been implemented. **§7** documents the multi-term random-effects model (`RandomEffectsModel`), the geographic indicators it consumes, the cell-based sufficient-statistics generalization, the predictive-fit / cross-validation framework (which supersedes the to-do framing in §3), and the apply-time reconstruction with its unseen-group back-off rule.

## 7. Multi-term random effects, geographic indicators, cross-validation, and apply-time reconstruction

This section documents the 2026-06 generalization of Level 1. It adds geographic
covariates to every observation, generalizes the single-factor `RandomByTypeModel`
to a composable sum of random and fixed effects (`RandomEffectsModel`),
generalizes the sufficient-statistics likelihood to arbitrary factor
combinations, adds a predictive-fit / cross-validation framework for comparing
model variants, and rates the snapshot by reconstructing each POI's curve from
the posterior (so unobserved cells are still rateable, with honest extra
uncertainty). The ZIE δ machinery of §1.7 is carried through unchanged and stays
orthogonal to everything below.

### 7.1 Geographic indicators on each observation

Two indicators are attached to every observation, **per OSM element from its
most-recent location** (one spatial join per unique element, broadcast to all of
that element's interval rows — not a per-row operation):

- **`msa_code`** — Metropolitan Statistical Area (CBSA `LSAD == "M1"`, metro
  only); points outside any MSA get the single dummy group `NO_MSA`.
- **`urban_rural`** — from the containing Census place's population and density:
  - **urban** = place population > 100k **and** density > 1000 / sq mi,
  - **suburban** = otherwise, density > 500 / sq mi,
  - **rural** = otherwise, or unincorporated (in no place).

Data sources (cartographic-boundary line, single national files): CBSA
`cb_2023_us_cbsa_500k` (filtered to metro), Place `cb_2023_us_place_500k`
(incorporated places **and** CDPs), and 2020 Decennial place population
(`P1_001N`, covers CDPs which the Population Estimates Program omits; fetched via
the Census API with a key and cached to CSV).

Code: [src/openpois/io/census_areas.py](src/openpois/io/census_areas.py)
(download/load + `classify_places`-feeding loaders),
[src/openpois/io/indicators.py](src/openpois/io/indicators.py)
(`classify_places`, `assign_indicators`, `element_coordinates`,
`element_indicator_map`). The element coordinate is the snapshot geometry's
representative point, with a fallback to the last-known node lat/lon recovered
from the `osm_changes` `lat`/`lon` pseudo-tags for deleted elements. The same
`assign_indicators` is reused at apply time on the snapshot (§7.5).

### 7.2 `RandomEffectsModel` — multiple additive random effects (generalizes Level 1)

Per-observation log-rate, with only the enabled terms present:

```
log λ_i = log_lambda_0
        + [amenity]      σ_amenity · ε_amenity_raw[a_i]        ε_raw ~ N(0,1)
        + [msa]          σ_msa     · ε_msa_raw[m_i]
        + [amenity×msa]  σ_am      · ε_am_raw[am_i]            (tight σ prior)
        + [urbanicity]   β_sub · 1{suburban_i} + β_rural · 1{rural_i}
```

- **Non-centered** everywhere (`σ_t = exp(log_sigma_t)`, `ε_raw ~ N(0,1)`), reusing
  the funnel-free parameterization that gives zero divergences (§2 Level 1).
  Each gathered term has its own `log_sigma_t ~ N(loc, scale)` hyperprior.
- **The amenity×MSA interaction is nested** and gets a **tighter** hyperprior than
  the main effects (default `log_sigma_am ~ N(-2, 0.5)` vs `N(-1, …)`), so
  interactions shrink toward zero unless strongly supported. To bound its
  dimensionality, an `(amenity, MSA)` cell only gets its own level when it has
  ≥ `interaction_min_count` **distinct POIs** (default 100); sub-threshold cells
  contribute zero interaction (fall back on the main effects).
- **Urbanicity is a fixed effect** with `urban` as the reference category: two
  coefficients `β_suburban`, `β_rural` with a `N(0, scale)` prior. (Two levels
  is effectively a fixed effect, not a partially-pooled random one.)
- **δ (zero-inflation) is independent and configurable** — grouped by amenity
  (the §1.7 behavior, default when the amenity term is on) or a single global
  scalar. δ's grouping is decoupled from the λ terms.
- **Every prior/scale is config-wired**, not hard-coded — the model is fully
  reproducible from the `osm_turnover_model.random_effects` block. Each term is
  toggled by its presence (`enabled: true`), so **any combination composes**, and
  **amenity-only reproduces `RandomByTypeModel` exactly** (asserted to machine
  precision in tests).

Code: [`RandomEffectsModel`](src/openpois/models/osm_models.py) (registry key
`random_effects`); config block `osm_turnover_model.random_effects` in
[config.yaml](config.yaml); metadata assembly + factor-aware output in
[scripts/models/osm_turnover.py](scripts/models/osm_turnover.py).

### 7.3 Cell-based sufficient statistics (generalizes §1.2 / §4.1)

The single-factor sufficient-statistics reduction (§4.1, keyed on `group`)
generalizes to multiple factors by defining a **cell** = the unique tuple of all
active factor levels (the enabled λ factors **plus** the δ-grouping factor). λ and
δ are **constant within a cell by construction**, so the ZIE Bernoulli-on-Poisson
factorization of §1.7 / §4.1 carries over verbatim with `group → cell`:

```
ll = − Σ_c λ_c · (Σ_{unchanged, first} Δ + Σ_{unchanged, non-first} Δ)_c
     + Σ_c n_first_unchanged_c · log(1 − δ_c)
     + Σ_{changed, non-first} log(1 − exp(−λ_c Δ))
     + Σ_{changed, first}     log(1 − (1−δ_c)·exp(−λ_c Δ))
```

The unchanged contribution collapses from `O(N)` to `O(C)` per-cell scalars; the
changed contribution iterates only the ~10% of rows that flipped — i.e.
`O(C + N_changed)`, the same complexity class as the single-factor case. Per-cell
`log λ_c` and `logit δ_c` are reconstructed by gathering each factor's level
contribution (the term-enable flags are resolved at JAX trace time, so the
compiled graph holds only the enabled gathers — no Python loops, no `lax.cond`).
A dense per-row path is retained for the pointwise log-likelihood (§7.4); the
two paths are asserted equal in value **and gradient** for every term combination.

### 7.4 Predictive-fit metrics and cross-validation (supersedes the §3 to-do)

[src/openpois/models/metrics.py](src/openpois/models/metrics.py) provides a
model-agnostic comparison framework built on a per-observation, per-draw ZIE
log-likelihood matrix (the model exposes `pointwise_log_likelihood`; the dense
formula is used since the sufficient-stats path can't produce pointwise values):

- **In-sample**: RMSE (`√mean((y_i − p̄_i)²)`, `p̄_i` the posterior-mean
  conditional `P(change at Δ_i)`), **LPPD** `Σ_i log mean_s p(y_i|θ_s)` (via
  `logsumexp`), and **WAIC** `= lppd − p_waic` with `p_waic = Σ_i var_s
  log p(y_i|θ_s)` (reported on the deviance scale, positive = worse).
- **Subgroup breakdowns** of all three, by amenity, MSA, urban/rural, and the
  three-way cross-section, with LPD normalized by subgroup size.
- **Out-of-sample**: a **structured 10-fold holdout** — whole POIs assigned to one
  fold (all of a POI's interval rows together, so there is no within-POI leakage)
  and **stratified** across `(MSA × amenity × urban/rural)` cells so each fold is
  balanced. Each fold refits on the other nine and scores held-out RMSE + LPD;
  the aggregate is mean RMSE and summed LPD. Methodology follows the
  [mbg model-comparison article](https://henryspatialanalysis.github.io/mbg/articles/model-comparison.html),
  with the per-POI / stratified choices specific to this dataset.

Per-observation reductions are streamed in row-blocks so the `(draws, N)` matrix
is never materialized at national scale. A small national fixture
(~10k observations, 5 MSAs × 10 amenity types, whole POIs) is built by
[scripts/exploratory/build_test_dataset.py](scripts/exploratory/build_test_dataset.py)
for fast model iteration. `scripts/models/osm_turnover.py` writes
`metrics_summary.csv` + `metrics_subgroup.csv` for any model exposing the
pointwise likelihood.

### 7.5 Apply-time prediction & the unseen-group back-off

Rating the current snapshot enriches each POI with `msa_code` / `urban_rural`
(§7.1, reusing `assign_indicators` on the snapshot geometry), assigns a
`shared_label`, and then **reconstructs that cell's posterior-predictive curve
directly from the saved draws** rather than looking it up in `predictions.csv`
(which only holds observed training cells). This makes any
`(amenity, MSA, urban/rural)` cell rateable. The conditional vs. fresh prediction
regimes of §1.7 carry over unchanged.

The novel piece is the **default case for a cell whose factor levels were not seen
at fit time** — i.e. the posterior predictive for a *new* random-effect level,
which differs by term type:

- **Nested amenity×MSA interaction** — an unobserved `(amenity, MSA)` cell
  contributes **exactly zero**; the POI falls back on the main effects. (Sparse,
  tightly-shrunk interactions should not extrapolate.)
- **All other random effects** (amenity main, MSA main, the per-amenity δ
  grouping) — a new group's effect is **drawn from the fitted random-effect
  distribution**, `ε_new ~ N(0, σ_term)` per posterior draw (using the posterior
  draws of `σ_term`). This injects both the between-group spread and the
  uncertainty in `σ`, so an unseen group gets a **wider** credible interval than an
  observed one — the statistically correct posterior predictive for a new level,
  not a flat fallback. The `N(0,1)` draw is **seeded by a stable hash of
  `(term, group-name)`**, so the same new group yields the same effect across
  batches and runs (consistent caching, reproducible output).
- **Urbanicity** is a fixed effect over known categories — there is no unseen case.

This realizes the principle that a never-before-seen MSA or amenity should be
modeled as a fresh draw from the estimated heterogeneity (with appropriately
inflated variance), while an unseen *interaction* — already weakly identified and
tightly shrunk — simply reverts to its margins.

Code: [src/openpois/models/reconstruct.py](src/openpois/models/reconstruct.py)
(posterior-predictive curves per cell, vectorized over draws, chunked over cells);
[scripts/osm_snapshot/apply_model_random_effects.py](scripts/osm_snapshot/apply_model_random_effects.py)
(streams the snapshot, enriches + labels each batch, reconstructs each unique cell
once via a cache, reads off confidence at years-since-edit). The fit step saves a
long-form `factor_lookups.csv` (incl. the interaction's `amenity`/`msa_code`
components) so the apply step can map cells back to level ids. The reconstruction
is asserted to match `ModelFitter.predict` on observed cells.

### 7.6 Data loading (implementation note)

[src/openpois/osm/format_observations.py](src/openpois/osm/format_observations.py)
gained a DuckDB **window-function** implementation (`format_observations_window`)
of the per-POI state machine (`add_to_list` as a cumulative max; `tag_value`
delete/re-add and `keep_keys` stickiness as null-preserving
`last_value(… IGNORE NULLS)` carries; the `last_*` timestamps over emitted rows
only). It produces **byte-for-byte identical** observations to the legacy row
loop (gated by a golden test over all edge-case fixtures) but runs in DuckDB's
streaming engine. Because the chained window operators exceed the memory budget
in a single national pass, it executes in `hash(id)` buckets — windows partition
by `(type, id)` and the hash depends only on `id`, so a POI's rows never split
across buckets — keeping peak RAM bounded while running ~2–3× faster than the
state machine. The observation build now also folds the §7.1 indicator enrichment
and the shared-label assignment into a single streaming pass and emits
`osm_type`, `msa_code`, and `urban_rural` columns.

### 7.7 New / changed files (this session)

| File | Role |
|---|---|
| [src/openpois/io/census_areas.py](src/openpois/io/census_areas.py) | **New.** Census CBSA / Place / population download + load (§7.1). |
| [src/openpois/io/indicators.py](src/openpois/io/indicators.py) | **New.** `classify_places`, `assign_indicators`, element coordinate + indicator map (§7.1). |
| [src/openpois/osm/format_observations.py](src/openpois/osm/format_observations.py) | DuckDB window build + `osm_type` column (§7.6). |
| [scripts/osm_data/format_tabular.py](scripts/osm_data/format_tabular.py) | Element-level enrichment + streaming label/explode folded into the build (§7.1, §7.6). |
| [src/openpois/models/osm_models.py](src/openpois/models/osm_models.py) | **`RandomEffectsModel`** + cell sufficient stats + `pointwise_log_likelihood` (§7.2, §7.3). |
| [src/openpois/models/metrics.py](src/openpois/models/metrics.py) | **New.** RMSE / LPPD / WAIC, subgroup, stratified 10-fold CV (§7.4). |
| [src/openpois/models/reconstruct.py](src/openpois/models/reconstruct.py) | **New.** Per-cell posterior-predictive reconstruction + unseen-group back-off (§7.5). |
| [scripts/models/osm_turnover.py](scripts/models/osm_turnover.py) | `random_effects` wiring, factor-aware outputs, metrics + `factor_lookups.csv` (§7.2, §7.4, §7.5). |
| [scripts/osm_snapshot/apply_model_random_effects.py](scripts/osm_snapshot/apply_model_random_effects.py) | **New.** Reconstruct-from-params snapshot rating (§7.5). |
| [scripts/exploratory/build_test_dataset.py](scripts/exploratory/build_test_dataset.py) | **New.** National test fixture (§7.4). |
| [config.yaml](config.yaml) | `download.census_areas`, `directories.census_areas`, `osm_turnover_model.random_effects`, metrics + `factor_lookups` model-output files. |
| `tests/test_osm_models.py`, `tests/test_metrics.py`, `tests/test_reconstruct.py`, `tests/test_format_observations.py` | Regression coverage for all of the above (incl. amenity-only ≡ `RandomByTypeModel`, suff-stats ≡ dense, reconstruct ≡ predict, golden window ≡ state machine). |

## 8. Time-varying λ — the breakpoint (two-rate) model (Level 4, implemented)

### 8.1 The integrated-hazard view

Every model here enters the likelihood **only** through the cumulative hazard
over each interval, `H(t1, t2) = ∫_{t1}^{t2} λ(a) da`, via `P(change) = 1 −
exp(−H)` and the ZIE δ discount on first intervals (§1.7). The fitter treats
`event_rate_fun`'s output as exactly this `H` ([model_fitter.py](src/openpois/models/model_fitter.py) `make_log_density`; [osm_models.py](src/openpois/models/osm_models.py) `_zie_pointwise_loglik`). For a **constant** λ, `H = λ·(t2 − t1) = λ·Δt`. A time-varying λ only swaps in a different closed-form `H` — nothing downstream changes. **Design rule:** only adopt λ(a) forms whose integral has a closed form (no quadrature in the inner loop).

### 8.2 Time axis = tag age

The hazard clock is the tag's **age since the current value was established**
(`last_tag_timestamp`). For each interval row, `age_start = t1 − last_tag` and
`age_end = t2 − last_tag` (in years); `age_start = 0` on the first interval, and
`age_end − age_start = tag_years` so per-row hazards telescope to each
individual's cumulative hazard. These columns are emitted by
[`prepare_data_for_model`](src/openpois/models/setup.py); constant-λ models
ignore them.

### 8.3 The two-rate breakpoint

A single breakpoint at age `t_B` with `λ_1 = exp(log_lambda_1)` before it and
`λ_2 = exp(log_lambda_2)` after. The integral over `[age_start, age_end]` is the
branchless clamp form (`_breakpoint_integrated_hazard`):

```
crossing = clip(t_B, age_start, age_end)
H        = λ_1·(crossing − age_start) + λ_2·(age_end − crossing)
```

covering all three cases (interval fully before / fully after / straddling
`t_B`) at once, and reducing to `λ·Δt` when `λ_1 = λ_2`. Cost is the same `O(N)`
dense per-row sum as `ConstantModel`. Priors: `log_lambda_1`, `log_lambda_2 ~
N(0, 3)` (the constant-λ prior); `log_t_breakpoint ~ N(0, 1)` (log-normal,
median `t_B = 1` year, config key `osm_turnover_model.t_breakpoint_prior`);
δ unchanged. `clip` is continuous and differentiable a.e. — NUTS handles the
kinks given enough straddling intervals and the log-normal `t_B` prior.

Code: `ConstantBreakpointModel` (subclass of `ConstantModel`) in
[osm_models.py](src/openpois/models/osm_models.py), registered as model type
`constant_breakpoint`; toggle with `--model-type constant_breakpoint`.

### 8.4 Rating consequence (tag age vs. last edit)

Snapshot rating ([apply_model.py](scripts/osm_snapshot/apply_model.py)) anchors
the survival clock at `last_edited` and asks `P(change within elapsed)`. That is
memoryless-correct **only for constant λ**. Under a tag-age hazard the rating
must integrate the hazard at the tag's *true age* over the `[last_edited, now]`
window:

```
Λ(a)        = λ_1·min(a, t_B) + λ_2·max(0, a − t_B)        # cumulative hazard
a_last_edit = last_edited − tag_established
a_now       = now         − tag_established
P(change since last edit) = 1 − exp( −( Λ(a_now) − Λ(a_last_edit) ) )
```

This needs a per-POI `tag_established` (the `last_tag_timestamp` that
`format_observations` already derives), joined onto the snapshot; POIs absent
from history fall back to `a_last_edit = 0`. CIs come from a modest set of
posterior draws of `(λ_1, λ_2, t_B)` pushed through `Λ`. **This rating-side work
is a scoped follow-up**; the model, training data prep, in-sample fit, and the
from-age-0 diagnostic prediction curve land first.
