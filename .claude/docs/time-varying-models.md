# Time-varying λ models (breakpoint hazard)

Status: **experimental / parked.** The constant-λ breakpoint model is implemented,
tested, and was run nationwide once (2026-06-08). The nationwide fit did not
support a meaningful breakpoint, so the planned random-effects extension was
**not** built. This note records why we built it, how it works, how to run it,
what we found, and why we stopped — so the work is reproducible and the dead end
is not re-explored from scratch.

The breakpoint code is fully **isolated and additive**: the production
`constant`, `random_by_type`, and `random_effects` models are unchanged. See also
[turnover-model-methodology.md §8](turnover-model-methodology.md) for the
statistical derivation.

## 1. Why we developed it

The turnover model assumes a **constant** per-year hazard λ over each
observation interval, so the cumulative hazard that drives the change
probability (`P(change) = 1 − exp(−rate)`) is just `λ·Δt`. We wanted to test an
"infant-mortality" hypothesis: that a POI tag churns at one rate while the tag
value is young and a different rate once it has aged past some breakpoint `t_B`
(prior centred at ~1 year). The breakpoint is the simplest time-varying λ with a
**closed-form integral**, which is the design rule for any time-varying form here
(no numerical quadrature in the NUTS inner loop).

## 2. Key idea — integrated hazard + tag-age clock

The whole likelihood consumes λ **only** through the integral over each interval,
`H(t1, t2) = ∫ λ(a) da`. Constant λ is the `λ·Δt` special case; the breakpoint
swaps in a different closed form. The time axis is the tag's **age since its
current value was established** (`last_tag_timestamp`), so per-row hazards
telescope to each POI's cumulative hazard.

Two-rate breakpoint integral (`λ_1` before age `t_B`, `λ_2` after):

```
crossing  = clip(t_B, age_start, age_end)
H(t1, t2) = λ_1·(crossing − age_start) + λ_2·(age_end − crossing)
```

This one branchless expression covers all three cases (interval fully before /
after / straddling `t_B`) and reduces to `λ·Δt` when `λ_1 == λ_2`.

**Rating consequence.** Snapshot rating anchors at `last_edited`, but a tag-age
hazard must be integrated at the POI's *true* age over the `[last_edited, now]`
window:

```
Λ(a)        = λ_1·min(a, t_B) + λ_2·max(0, a − t_B)
P(turnover) = 1 − exp( −( Λ(a_now) − Λ(a_last_edit) ) )
```

which needs a per-POI `tag_established` (joined from history); POIs absent from
history fall back to `a_last_edit = 0`.

## 3. Key files and functions

| File | What |
|---|---|
| [src/openpois/models/setup.py](src/openpois/models/setup.py) | `prepare_data_for_model` emits `age_start` / `age_end` (tag-age bounds; `age_end − age_start == tag_years`, `age_start == 0` on first intervals). Additive — constant/RE models ignore them. |
| [src/openpois/models/osm_models.py](src/openpois/models/osm_models.py) | `_breakpoint_integrated_hazard(lam1, lam2, t_b, age_start, age_end)` (the clamp formula); `ConstantBreakpointModel` (subclass of `ConstantModel`, reuses the ZIE δ machinery); registry key `constant_breakpoint`; `DEFAULT_T_BREAKPOINT_PRIOR = (0.0, 1.0)`. |
| [scripts/models/osm_turnover.py](scripts/models/osm_turnover.py) | `--model-type constant_breakpoint`; wires `osm_turnover_model.t_breakpoint_prior` into model metadata. |
| [config.yaml](config.yaml) | `osm_turnover_model.t_breakpoint_prior: [0.0, 1.0]` — (loc, scale) on log `t_B`; loc 0 → median `t_B` = 1 yr. |
| [scripts/osm_data/add_turnover_columns.py](scripts/osm_data/add_turnover_columns.py) | Final-stage augmentation: reuses an existing `osm_observations.parquet` (no history re-download), adds `age_start`/`age_end`, and emits `osm_current_tag.parquet` — one row per element with `tag_established` (latest version's `last_tag_timestamp`) + `last_seen`. |
| [scripts/osm_snapshot/apply_model_breakpoint.py](scripts/osm_snapshot/apply_model_breakpoint.py) | Rates the live snapshot via the closed-form `Λ`. `load_breakpoint_draws`, `cumulative_hazard`, `turnover_stats`. Output columns: `p_turnover_*` = **P(change)**, `conf_*` = 1 − P (P(no change)), `tag_age_years`, `matched_history`. |
| [tests/test_osm_models.py](tests/test_osm_models.py) | `test_breakpoint_integrated_hazard`, `test_constant_breakpoint_recovery`, `test_constant_breakpoint_reduces_to_constant`, `test_predictions_schema_constant_breakpoint`, `test_constant_breakpoint_requires_age_columns`, `test_prepare_data_emits_age_columns`. |

**Parameters:** `log_lambda_1`, `log_lambda_2` (each N(0, 3) on the log scale,
same prior as the constant model's λ), `log_t_breakpoint` (N(0, 1) → log-normal
`t_B`, median 1 yr), `logit_delta` (unchanged ZIE δ). Derived draws expose
`lambda_1`, `lambda_2`, `t_breakpoint`, `delta`.

## 4. How it's run (the nationwide test, end to end)

No new history download — reuse the prepared `20260521` history through the final
formatting stage:

```bash
# 1. Augment observations + emit tag_established lookup → osm_data/20260608
python scripts/osm_data/add_turnover_columns.py \
    --source-version 20260521 --target-version 20260608

# 2. Fit the constant breakpoint model (dense; ~2 h over ~9.9M rows on CPU)
python -u scripts/models/osm_turnover.py \
    --model-type constant_breakpoint \
    --observations ~/data/openpois/osm_data/20260608/osm_observations.parquet \
    --model-version 2026-06-08-breakpoint-test

# 3. Rate the live snapshot with the closed-form Λ
python -u scripts/osm_snapshot/apply_model_breakpoint.py \
    --model-version 2026-06-08-breakpoint-test \
    --current-tag-version 20260608
```

Note: the cell **sufficient-statistics** fast path cannot be used — a sampled
`t_B` enters the per-row integral through `clip`, so the likelihood is **dense**
per-row. That is why the national constant fit took ~2 h, and why a
random-effects + OOS version would cost many hours.

## 5. What we found (nationwide, 2026-06-08)

Input: `osm_data/20260608` (9,963,378 observation rows). Fit: 4 chains ×
(500 warmup + 500 draws), ~2 h 11 m wall.

**The fit did not converge — and that is the finding.**

- `max R-hat = 202.5`, `min ESS = 2` (0 divergences, accept 0.846, mean_steps 18.2).
- `λ_2` is stable across all chains: **≈ 0.054 /yr**.
- `λ_1` and `t_B` are **confounded** — per-chain `t_B` ∈ {0.0063, 0.015, 0.044,
  0.044} yr (≈ **2–16 days**), with `λ_1` swinging 1.7 → 19.4 to compensate.
  When `t_B → 0`, only the *product* `λ_1·t_B` (the brief early-time mass, which
  also overlaps δ) is identified — not `λ_1` and `t_B` separately.

So the national data wants a very short high-churn window (days) right after a
name is set, then settles to `λ_2`. There is **no meaningful breakpoint at a
≥6-month timescale**.

Apply step (sanity, despite the non-converged posterior): all **8,799,633** POIs
rated, no NaNs, `p_turnover` (= P(change)) mean 0.23 / median 0.20, range
0.003–0.68. The tag-age mechanism works as designed — an old tag edited recently
(e.g. age 10.6 yr, edited 1.2 yr ago) correctly sits in the low-`λ_2` regime
(p ≈ 0.06). History coverage on the live snapshot was **21.3%** matched to a real
`tag_established`; the other 78.7% used the last-edit fallback.

## 6. Why we are not developing it further (for now)

1. **The breakpoint collapses into δ.** The two-regime structure the data
   supports lives at a sub-month scale — i.e. "brief churn right after a name is
   set," which the **zero-inflation δ** term (instant-change mass at t = 0,
   methodology §1.7) already models. The breakpoint is largely redundant at the
   granularity the data supports.
2. **Non-identifiability.** R-hat 202 / ESS 2 is the symptom: `λ_1` and `t_B`
   trade off as `t_B → 0`. The pooled parameter summary is not trustworthy.
3. **A user-defined gate closed.** We agreed to extend to random effects **only
   if** the national constant fit showed (a) `λ_1` vs `λ_2` differing by > 10%
   **and** (b) `t_B ≥ 6 months`. Condition (a) passed (~99% apart), but
   (b) failed decisively (`t_B ≈ 0.027 yr ≈ 10 days`). Per the gate, we stopped.
4. **Cost.** Without the suff-stats fast path the model is dense; a
   random-effects breakpoint with per-amenity `t_B` plus full OOS would run many
   hours per pass.

### The random-effects breakpoint extension was scoped but NOT built

Design (for reference if revisited): a new `breakpoint_models.py` with a
`RandomEffectsBreakpointModel` adapted from `RandomEffectsModel` — the entire
log-λ predictor duplicated into independent `λ_1` / `λ_2` regimes (global
intercept + amenity/MSA/interaction random effects + urbanicity fixed effects),
a global `t_B` with an optional amenity-level random intercept, δ kept from the
best spec (Full + δ(amenity+MSA), `2026-06-06-oos-full-dmsa`), dense likelihood,
fit in-sample and OOS with the standard per-fold/aggregate/subgroup metrics.
None of this was implemented.

### If revisited, options in rough order of promise

- Drop δ and let `λ_1` absorb the early churn (does a breakpoint *replace* δ?).
- Tighten/constrain the `t_B` prior, or add an ordering / identifiability
  constraint so `λ_1`/`t_B` separate.
- Accept that a ≥6-month breakpoint is not supported nationally and keep the
  constant / random-effects λ + δ model (the production path).

## 7. Reproducibility artifacts

On disk (uncommitted; production versions untouched):
- `~/data/openpois/osm_data/20260608/` — augmented `osm_observations.parquet`
  (+ `age_start`/`age_end`) and `osm_current_tag.parquet` (3,878,564 elements).
- `~/data/openpois/osm_turnover_model/2026-06-08-breakpoint-test/` — fit
  (`fitted_params.csv`, `param_draws.parquet`, `diagnostics.csv`) +
  `osm_snapshot_turnover.parquet` (the rated snapshot).
- Logs in `~/data/openpois/logs/` (`augment_20260608`, `fit_breakpoint_20260608`,
  `apply_breakpoint_20260608`).

Production pins in [config.yaml](config.yaml) (`osm_data: 20260521`,
`model_output: 20260422_by_shared_label`) and their outputs were never modified.
