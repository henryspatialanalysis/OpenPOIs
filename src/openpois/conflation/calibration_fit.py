"""Fit per-segment existence-confidence calibration curves (v4 estimator).

Design source: the 2026-07-30 v4 writeup
(``~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md``)
(supersedes section 9 of the 2026-07-24 v3 review). Input is the condensed,
anonymized validation handoff exported by openpois-validator; output is one
monotone lookup table per detection segment, consumed by
:mod:`openpois.conflation.calibration` at deploy time.

The estimator is a **model-assisted difference estimator on a two-phase
design**. Phase 1 is the LLM verification of every sampled POI (cheap, noisy);
phase 2 is the human gold subsample, drawn at known but very unequal rates
*within LLM-verdict class* (on round 20260730: 11.6% of LLM-exists, 36.9% of
LLM-gone, 100% of LLM-unverifiable). For a score cell ``B``::

    p_hat(B) = (1 / N_B) * [ SUM_{i in B} y_pred_i
                             + SUM_{i in B and gold} (y_i - y_pred_i) / pi_c(i) ]

where ``y_pred`` is a low-dimensional working model for
``P(exists | class, score)`` fit on gold within class, and ``pi_c`` is the
realized phase-2 inclusion of the row's refined class (LLM verdict crossed with
the LLM's own verdict confidence, cells merged below a gold floor).

Two properties make this the right form (Breidt & Opsomer 2017):

- It is design-unbiased for the cell mean *whatever* the working model does,
  because the second term corrects the first in expectation. A saturated
  working model collapses it to the pure Horvitz-Thompson estimator, which is
  algebraically the validator's as-built ``stratified_ht_gold_v1`` curve --
  retained here as the robustness reference.
- With the low-dimensional working model, the large inverse-inclusion weights
  (~8.6x on LLM-exists) multiply *near-zero residuals*, because that class's
  gold existence rate is 0.985. The censused unverifiable class carries no
  phase-2 sampling variance at all. This is where the precision the gold-only
  fit gave up comes back: the score shape comes from the full phase-1 archive.

The **matched** segment carries two source scores, so it is not calibrated on
the upstream 0.588/0.412 blend. Instead the two scores are pooled on the
log-odds scale with *fitted* weights::

    z = b0 + b_osm * logit(s_osm) + b_ov * logit(s_ov)

estimated by design-weighted logistic regression on matched gold, and the
segment curve is then fit design-based against ``expit(z)``. This is the
multiplicative (log-odds) opinion pool of Bordley (1982) and Clemen & Winkler
(1999) with free rather than assumed-equal weights: coherent, monotone in each
input, and able to place a matched POI *above* either source's own confidence
when both agree, which the linear blend can never do. The fitted coefficients
replace the ``overture_confidence_weight`` constant, so no fixed downweight of
the Overture score survives anywhere in the calibrated output.

Uncertainty is a two-phase bootstrap: phase-1 rows resampled within verdict
class at fixed class counts, gold resampled within refined class at fixed
counts, refitting everything -- pool coefficients included -- per replicate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

SEGMENTS = ("matched", "osm", "overture")
# Segments whose curve is indexed on a pooled function of two source scores
# rather than on a single native score.
POOLED_SEGMENTS = ("matched",)
# Clip before the logit so scores at exactly 0 or 1 stay finite.
LOGIT_EPS = 1e-4
VERDICTS = ("exists", "gone", "unverifiable")
# Verdict classes whose gold existence rate is modeled as flat in the score.
# The definitive verdicts are near-deterministic (gold rates 0.985 / 0.010 on
# round 20260730), so a constant is both adequate and stable; the censused
# unverifiable class is genuinely score-dependent and gets an isotonic fit.
DEFINITIVE_VERDICTS = ("exists", "gone")
CURVE_COLUMNS = ("segment", "score_lo", "score_hi", "conf_mean", "conf_lower",
                 "conf_upper")
ESTIMATOR_TAG = "composite_model_assisted_v1"


@dataclass
class FitConfig:
    """Knobs for the composite fit. Defaults are the round-20260730 settings."""

    min_cell_gold: int = 25
    grid_points: int = 200
    output_bins: int = 40
    bootstrap_reps: int = 500
    band_alpha: float = 0.05
    rng_seed: int = 20260730
    # Cells below ``min_cell_gold`` merge into their parent verdict class.
    refine_by_confidence: bool = True
    # Two-source segments: "pool" fits log-odds weights on gold, "average"
    # uses the parameter-free mean of the two source scores. Compare them with
    # scripts/conflation/compare_matched_index.py before changing.
    matched_index_mode: str = "pool"
    metadata: dict = field(default_factory = dict)


def refined_class(rows: pd.DataFrame, refine: bool = True) -> pd.Series:
    """Class label per row: LLM verdict, optionally crossed with confidence.

    Refining is valid because phase-2 inclusion is uniform *within* verdict
    class, so any partition measurable at phase 1 inherits uniform inclusion
    on its cells (the realized per-cell rate is then the design rate).
    """
    verdict = rows["llm_verdict"].astype(str)
    if not refine:
        return verdict
    confidence = rows["llm_confidence"].astype(str).fillna("unknown")
    return verdict.str.cat(confidence, sep = ":")


def merge_thin_cells(classes: pd.Series, gold_mask: np.ndarray,
                     min_gold: int) -> pd.Series:
    """Collapse refined cells with too little gold back to the verdict class.

    A cell whose realized inclusion rests on a handful of labels would carry a
    noisy rate and an unstable weight; falling back to the parent class costs
    resolution and buys stability.
    """
    classes = classes.astype(str)
    gold_counts = classes[gold_mask].value_counts()
    thin = {c for c in classes.unique() if gold_counts.get(c, 0) < min_gold}
    if not thin:
        return classes
    parent = classes.str.split(":").str[0]
    return classes.where(~classes.isin(thin), parent)


def inclusion_by_class(classes: pd.Series, gold_mask: np.ndarray) -> dict:
    """Realized phase-2 inclusion rate and HT weight per class.

    ``{class: {n_pop, n_gold, inclusion, weight}}`` over the phase-1
    population. A class with no gold gets weight 0 and cannot contribute a
    correction term; its rows still contribute their working-model prediction.
    """
    pop = classes.value_counts()
    gold = classes[gold_mask].value_counts()
    out = {}
    for name, n_pop in pop.items():
        n_gold = int(gold.get(name, 0))
        inclusion = (n_gold / int(n_pop)) if n_gold else 0.0
        out[str(name)] = {
            "n_pop": int(n_pop),
            "n_gold": n_gold,
            "inclusion": float(inclusion),
            "weight": float(1.0 / inclusion) if inclusion > 0 else 0.0,
        }
    return out


def logit(values) -> np.ndarray:
    """Log-odds of scores, clipped away from the open interval's ends."""
    clipped = np.clip(np.asarray(values, dtype = float), LOGIT_EPS,
                      1.0 - LOGIT_EPS)
    return np.log(clipped / (1.0 - clipped))


def expit(values) -> np.ndarray:
    """Inverse logit."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype = float)))


def fit_pool(rows: pd.DataFrame, weights: np.ndarray) -> dict:
    """Design-weighted log-odds pool of the two source scores on gold rows.

    Returns ``{intercept, coef_osm, coef_overture, n_gold, method}``. The
    coefficients are the pool weights of Bordley (1982) freed from the
    equal-weight assumption; ``coef_* > 1`` means a source's evidence is
    sharpened, ``< 1`` that it is damped for dependence with the other source.

    Falls back to the *unfitted* equal-weight pool when gold is too thin or
    one-sided (a logistic fit on a single outcome class is not identified). The
    fallback is still a coherent pool -- it just does not learn the weights.
    """
    gold_mask = rows["gold"].to_numpy(dtype = bool)
    gold = rows[gold_mask]
    weights = np.asarray(weights, dtype = float)[gold_mask]
    y = gold["y"].to_numpy(dtype = float)
    features = np.column_stack(
        [logit(gold["osm_score"].to_numpy(dtype = float)),
         logit(gold["overture_score"].to_numpy(dtype = float))]
    )
    usable = (
        np.isfinite(features).all(axis = 1) & np.isfinite(y) & (weights > 0)
    )
    fallback = {
        "intercept": 0.0, "coef_osm": 1.0, "coef_overture": 1.0,
        "n_gold": int(usable.sum()), "method": "equal_weight_pool",
    }
    if usable.sum() < 30 or len(np.unique(y[usable])) < 2:
        return fallback
    # Effectively unpenalized: the pool has two coefficients on hundreds to
    # thousands of rows, so no shrinkage is wanted. A large finite C avoids
    # both the deprecated ``penalty = None`` and the C-is-ignored warning that
    # sklearn 1.8 raises for ``C = inf``.
    model = LogisticRegression(C = 1e10, solver = "lbfgs", max_iter = 2000)
    try:
        model.fit(features[usable], y[usable],
                  sample_weight = weights[usable])
    except (ValueError, np.linalg.LinAlgError):
        return fallback
    return {
        "intercept": float(model.intercept_[0]),
        "coef_osm": float(model.coef_[0][0]),
        "coef_overture": float(model.coef_[0][1]),
        "n_gold": int(usable.sum()),
        "method": "fitted_log_odds_pool_v1",
    }


def pool_score(osm_score, overture_score, params: dict) -> np.ndarray:
    """Pooled index in [0, 1] from the two source scores.

    Monotone increasing in each source score whenever its coefficient is
    positive, so the segment curve fit against this index stays monotone in
    both inputs.
    """
    z = (
        float(params["intercept"])
        + float(params["coef_osm"]) * logit(osm_score)
        + float(params["coef_overture"]) * logit(overture_score)
    )
    return np.clip(expit(z), 0.0, 1.0)


def average_score(osm_score, overture_score) -> np.ndarray:
    """Unweighted mean of the two source scores.

    The parameter-free alternative to :func:`pool_score` for a two-source
    segment: no coefficients to estimate, so nothing to refit per bootstrap
    replicate or per cross-fit fold. Whether it costs predictive validity is an
    empirical question -- see ``scripts/conflation/compare_matched_index.py``.
    """
    osm = np.asarray(osm_score, dtype = float)
    overture = np.asarray(overture_score, dtype = float)
    return np.clip((osm + overture) / 2.0, 0.0, 1.0)


def segment_scores(rows: pd.DataFrame, segment: str,
                   pool_params: dict = None,
                   index_mode: str = "pool") -> np.ndarray:
    """Curve-index score per row for one segment.

    For a two-source segment, ``index_mode`` selects between the fitted
    log-odds pool (``"pool"``) and the unweighted average (``"average"``).
    Single-source segments ignore it and use their native score.
    """
    if segment in POOLED_SEGMENTS:
        osm = rows["osm_score"].to_numpy(dtype = float)
        overture = rows["overture_score"].to_numpy(dtype = float)
        if index_mode == "average":
            return average_score(osm, overture)
        if index_mode != "pool":
            raise ValueError(f"Unknown index_mode {index_mode!r}")
        if pool_params is None:
            raise ValueError(f"Segment {segment} needs pool parameters")
        return pool_score(osm, overture, pool_params)
    column = "osm_score" if segment == "osm" else "overture_score"
    return rows[column].to_numpy(dtype = float)


def needs_pool(segment: str, index_mode: str) -> bool:
    """Whether this (segment, index_mode) pair estimates pool coefficients."""
    return segment in POOLED_SEGMENTS and index_mode == "pool"


def _score_definition(segment: str, index_mode: str) -> str:
    """Human-readable statement of what a segment's curve is indexed on."""
    if segment in POOLED_SEGMENTS:
        if index_mode == "average":
            return "mean(osm_conf_mean, overture_confidence)"
        return "fitted_log_odds_pool(osm_conf_mean, overture_confidence)"
    return "osm_conf_mean" if segment == "osm" else "overture_confidence"


def _isotonic_rate(scores: np.ndarray, y: np.ndarray,
                   grid: np.ndarray) -> np.ndarray:
    """Monotone P(exists | score) within one class, evaluated on ``grid``.

    Sort key ``(score asc, y desc)`` matches the CORP reference
    implementation's tie handling (Dimitriadis, Gneiting & Jordan 2021).
    """
    order = np.lexsort((-y, scores))
    model = IsotonicRegression(y_min = 0.0, y_max = 1.0, increasing = True,
                               out_of_bounds = "clip")
    model.fit(scores[order], y[order])
    return np.clip(model.predict(grid), 0.0, 1.0)


def class_working_models(rows: pd.DataFrame, classes: pd.Series,
                         grid: np.ndarray) -> dict:
    """Working model ``q_c(s)`` per class, evaluated on ``grid``.

    Definitive verdict classes (exists / gone) get a constant: their gold
    existence rate. The censused unverifiable class gets an isotonic fit in
    the score, which is what the census bought. A class with no gold falls
    back to the pooled gold rate, and its rows are then repaired only through
    other classes' corrections -- the difference estimator stays unbiased for
    the cell mean because such a class contributes no correction term either.
    """
    gold_rows = rows["gold"].to_numpy(dtype = bool)
    y_all = rows["y"].to_numpy(dtype = float)
    scores_all = rows["score"].to_numpy(dtype = float)
    pooled = float(np.nanmean(y_all[gold_rows])) if gold_rows.any() else 0.5

    models = {}
    for name in classes.unique():
        in_class = (classes == name).to_numpy()
        gold_in_class = in_class & gold_rows
        n_gold = int(gold_in_class.sum())
        verdict = str(name).split(":")[0]
        if n_gold == 0:
            models[str(name)] = np.full(len(grid), pooled)
            continue
        if verdict in DEFINITIVE_VERDICTS:
            rate = float(np.nanmean(y_all[gold_in_class]))
            models[str(name)] = np.full(len(grid), rate)
        else:
            models[str(name)] = _isotonic_rate(
                scores_all[gold_in_class], y_all[gold_in_class], grid
            )
    return models


def _kernel_matrix(scores: np.ndarray, grid: np.ndarray,
                   bandwidth: float) -> np.ndarray:
    """Gaussian weights of every row at every grid point, shape (grid, rows)."""
    return np.exp(
        -0.5 * ((scores[None, :] - grid[:, None]) / bandwidth) ** 2
    )


def composite_curve(rows: pd.DataFrame, classes: pd.Series, grid: np.ndarray,
                    inclusion: dict) -> np.ndarray:
    """Difference-estimator curve over ``grid``, then monotonized by PAV.

    At each grid point the prediction term averages every phase-1 row's class
    model *evaluated at that grid point* (which is the composite
    ``SUM_v m_v(s) q_v(s)``, since the local average over rows estimates the
    class mix ``m_v(s)``), and the correction term adds the design-weighted
    residuals of the gold rows in the same neighbourhood. Locality is a
    Nadaraya-Watson Gaussian kernel, which is what makes this a curve rather
    than a per-cell table.
    """
    scores = rows["score"].to_numpy(dtype = float)
    gold_mask = rows["gold"].to_numpy(dtype = bool)
    y = np.nan_to_num(rows["y"].to_numpy(dtype = float), nan = 0.0)
    models = class_working_models(rows, classes, grid)

    class_names = classes.astype(str).to_numpy()
    weights = np.array(
        [inclusion.get(c, {}).get("weight", 0.0) for c in class_names]
    )
    active = gold_mask & (weights > 0)

    # prediction[j, i] = q_{c(i)}(grid_j)
    prediction = np.stack([models[c] for c in class_names], axis = 1)
    contribution = prediction.copy()
    contribution[:, active] += (
        (y[active] - prediction[:, active]) * weights[active]
    )

    kernel = _kernel_matrix(scores, grid, _default_bandwidth(scores))
    total = kernel.sum(axis = 1)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        curve = (kernel * contribution).sum(axis = 1) / total
    curve[total <= 0] = np.nan

    curve = _fill_edges(curve)
    return _project_monotone(grid, np.clip(curve, 0.0, 1.0), scores)


def _default_bandwidth(scores: np.ndarray) -> float:
    """Silverman-style bandwidth, floored so sparse score regions stay smooth."""
    n = max(len(scores), 2)
    spread = float(np.std(scores))
    if spread <= 0:
        return 0.05
    return max(1.06 * spread * n ** (-1.0 / 5.0), 0.01)


def _fill_edges(curve: np.ndarray) -> np.ndarray:
    """Carry the nearest finite value into grid regions with no local support."""
    out = curve.copy()
    finite = np.isfinite(out)
    if not finite.any():
        return np.full_like(out, 0.5)
    idx = np.arange(len(out))
    return np.interp(idx, idx[finite], out[finite])


def _project_monotone(grid: np.ndarray, values: np.ndarray,
                      population_scores: np.ndarray) -> np.ndarray:
    """Isotonic projection of the fitted curve, weighted by population mass.

    Weighting by how many production POIs sit near each grid point means the
    monotonization spends its freedom where the data lives, not on empty score
    regions.
    """
    hist, _ = np.histogram(population_scores, bins = np.append(
        grid, grid[-1] + (grid[-1] - grid[-2] if len(grid) > 1 else 1e-6)
    ))
    weight = np.maximum(hist.astype(float), 1e-6)
    model = IsotonicRegression(y_min = 0.0, y_max = 1.0, increasing = True,
                               out_of_bounds = "clip")
    model.fit(grid, values, sample_weight = weight)
    return np.clip(model.predict(grid), 0.0, 1.0)


def two_phase_bootstrap(rows: pd.DataFrame, classes: pd.Series,
                        grid: np.ndarray, fit_config: FitConfig,
                        rng_offset: int = 0, segment: str = None,
                        index_mode: str = "pool") -> np.ndarray:
    """Bootstrap replicates of the composite curve, respecting both phases.

    Phase-1 rows are resampled with replacement within verdict class holding
    class counts fixed (propagating class-mix and prediction variance); within
    each resampled class the gold rows are resampled to the design's realized
    gold count (propagating correction variance). A censused class contributes
    no phase-2 variance because its gold count equals its population.

    For a pooled segment the pool coefficients are refit inside each replicate,
    so the band includes uncertainty in the pool weights themselves. That makes
    the band **POI-anchored** rather than grid-anchored: each replicate's fitted
    map is evaluated on the *original* rows (their replicate index score through
    the replicate curve), and those per-row probabilities are then aggregated
    back onto the point estimate's grid. Comparing replicates at a fixed grid
    value instead would conflate real uncertainty with the harmless
    reparameterization of the index when the pool weights move.
    """
    rng = np.random.default_rng(fit_config.rng_seed + 100 + rng_offset)
    verdicts = rows["llm_verdict"].astype(str).to_numpy()
    gold_mask = rows["gold"].to_numpy(dtype = bool)
    replicates = np.empty((fit_config.bootstrap_reps, len(grid)), dtype = float)

    groups = {v: np.flatnonzero(verdicts == v) for v in np.unique(verdicts)}
    gold_groups = {v: idx[gold_mask[idx]] for v, idx in groups.items()}
    nongold_groups = {v: idx[~gold_mask[idx]] for v, idx in groups.items()}

    # Anchor: the point estimate's own index scores, and the kernel that maps
    # per-row probabilities back onto the reporting grid.
    anchor_scores = rows["score"].to_numpy(dtype = float)
    anchor_kernel = _kernel_matrix(
        anchor_scores, grid, _default_bandwidth(anchor_scores)
    )
    anchor_total = anchor_kernel.sum(axis = 1)

    for rep in range(fit_config.bootstrap_reps):
        picks = []
        for verdict, idx in groups.items():
            n_gold = len(gold_groups[verdict])
            n_nongold = len(nongold_groups[verdict])
            if n_gold:
                picks.append(rng.choice(gold_groups[verdict], size = n_gold,
                                        replace = True))
            if n_nongold:
                picks.append(rng.choice(nongold_groups[verdict],
                                        size = n_nongold, replace = True))
        if not picks:
            replicates[rep] = np.nan
            continue
        take = np.concatenate(picks)
        boot_rows = rows.iloc[take].reset_index(drop = True)
        boot_classes = classes.iloc[take].reset_index(drop = True)
        boot_inclusion = inclusion_by_class(
            boot_classes, boot_rows["gold"].to_numpy(dtype = bool)
        )
        # Score the ORIGINAL rows under this replicate's fitted map, so the
        # band is anchored to POIs rather than to index values.
        replicate_anchor = anchor_scores
        if needs_pool(segment, index_mode):
            boot_weights = boot_classes.astype(str).map(
                lambda c: boot_inclusion.get(c, {}).get("weight", 0.0)
            ).to_numpy(dtype = float)
            boot_pool = fit_pool(boot_rows, boot_weights)
            boot_rows = boot_rows.assign(
                score = segment_scores(boot_rows, segment, boot_pool,
                                       index_mode)
            )
            replicate_anchor = segment_scores(rows, segment, boot_pool,
                                              index_mode)

        boot_scores = boot_rows["score"].to_numpy(dtype = float)
        finite = np.isfinite(boot_scores)
        if finite.sum() < 10:
            replicates[rep] = np.nan
            continue
        boot_grid = np.linspace(float(boot_scores[finite].min()),
                                float(boot_scores[finite].max()),
                                len(grid))
        boot_curve = composite_curve(boot_rows, boot_classes, boot_grid,
                                     boot_inclusion)
        per_row = np.interp(replicate_anchor, boot_grid, boot_curve,
                            left = boot_curve[0], right = boot_curve[-1])
        with np.errstate(invalid = "ignore", divide = "ignore"):
            replicates[rep] = (
                (anchor_kernel * per_row[None, :]).sum(axis = 1) / anchor_total
            )
    return replicates


def summarize_band(replicates: np.ndarray, curve: np.ndarray,
                   alpha: float = 0.05) -> dict:
    """Pointwise band around the estimate, monotonized and bracketing it."""
    lower = np.nanquantile(replicates, alpha / 2.0, axis = 0)
    upper = np.nanquantile(replicates, 1.0 - alpha / 2.0, axis = 0)
    lower = np.maximum.accumulate(np.clip(lower, 0.0, 1.0))
    upper = np.maximum.accumulate(np.clip(upper, 0.0, 1.0))
    return {
        "mean": curve,
        "lower": np.minimum(lower, curve),
        "upper": np.maximum(upper, curve),
    }


def build_lookup(segment: str, population_scores: np.ndarray,
                 grid: np.ndarray, summary: dict, n_bins: int) -> pd.DataFrame:
    """Equal-mass lookup over the population score distribution.

    Scaling-binning (Kumar, Liang & Ma 2019): fit the smooth map first, then
    average its *values* within equal-mass bins. Mirrors the validator's
    ``artifacts.build_lookup`` so both curves are directly comparable.
    """
    scores = np.asarray(population_scores, dtype = float)
    edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([scores.min(), scores.max() + 1e-9])

    def _interp(values, at):
        return np.interp(at, grid, values, left = values[0], right = values[-1])

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = scores[(scores >= lo) & (scores <= hi)]
        eval_at = in_bin if len(in_bin) else np.array([(lo + hi) / 2.0])
        rows.append(
            {
                "segment": segment,
                "score_lo": float(lo),
                "score_hi": float(hi),
                "conf_mean": float(_interp(summary["mean"], eval_at).mean()),
                "conf_lower": float(_interp(summary["lower"], eval_at).mean()),
                "conf_upper": float(_interp(summary["upper"], eval_at).mean()),
            }
        )
    lookup = pd.DataFrame(rows, columns = list(CURVE_COLUMNS))
    for column in ("conf_mean", "conf_lower", "conf_upper"):
        lookup[column] = np.maximum.accumulate(lookup[column])
    return lookup


def ht_reference_curve(rows: pd.DataFrame, classes: pd.Series,
                       grid: np.ndarray, inclusion: dict) -> np.ndarray:
    """Pure Horvitz-Thompson weighted-PAV curve on gold rows.

    The validator's as-built ``stratified_ht_gold_v1`` estimator, recomputed
    here as the robustness reference: the composite should track it within its
    band, and a systematic gap means the working model is wrong.
    """
    gold_mask = rows["gold"].to_numpy(dtype = bool)
    gold = rows[gold_mask]
    weights = classes[gold_mask].astype(str).map(
        lambda c: inclusion.get(c, {}).get("weight", 0.0)
    ).to_numpy(dtype = float)
    keep = weights > 0
    scores = gold["score"].to_numpy(dtype = float)[keep]
    y = gold["y"].to_numpy(dtype = float)[keep]
    if len(scores) < 5:
        return np.full(len(grid), np.nan)
    order = np.lexsort((-y, scores))
    model = IsotonicRegression(y_min = 0.0, y_max = 1.0, increasing = True,
                               out_of_bounds = "clip")
    model.fit(scores[order], y[order], sample_weight = weights[keep][order])
    return np.clip(model.predict(grid), 0.0, 1.0)


def cross_fit_predictions(rows: pd.DataFrame, classes: pd.Series,
                          fit_config: FitConfig, n_folds: int = 5,
                          segment: str = None,
                          index_mode: str = "pool") -> dict:
    """Out-of-fold predictions for every gold row, refitting the *whole* pipeline.

    Gold rows are folded within refined class so each training split preserves
    the design's class structure. Within a fold, everything estimated from gold
    is re-estimated on the training split alone -- **including the pool
    coefficients** for a pooled segment, which is what makes the returned
    predictions genuinely out-of-sample. Leaving the pool fit on all gold would
    quietly hand a multi-parameter index an advantage over a parameter-free one,
    which is exactly the comparison this function exists to support.

    Returns ``{predicted, actual, weight, fold}`` aligned to the gold rows.
    """
    rng = np.random.default_rng(fit_config.rng_seed + 700)
    gold_mask = rows["gold"].to_numpy(dtype = bool)
    gold_idx = np.flatnonzero(gold_mask)
    if len(gold_idx) < n_folds * 5:
        return {"n_folds": 0}

    class_names = classes.astype(str)
    fold_of = np.full(len(rows), -1)
    for name in class_names.unique():
        in_class = np.flatnonzero((class_names == name).to_numpy() & gold_mask)
        shuffled = rng.permutation(in_class)
        fold_of[shuffled] = np.arange(len(shuffled)) % n_folds

    predicted_all = np.full(len(rows), np.nan)
    weight_all = np.zeros(len(rows))
    for fold in range(n_folds):
        held = fold_of == fold
        train = rows.copy()
        # Held-out rows stay in phase 1 (they are real population members) but
        # lose their gold status, exactly as if they had not been audited.
        train.loc[held, "gold"] = False
        train.loc[held, "y"] = np.nan
        inclusion = inclusion_by_class(
            classes, train["gold"].to_numpy(dtype = bool)
        )
        # Re-derive the index from the training gold only.
        scored_rows = rows
        if needs_pool(segment, index_mode):
            train_weights = class_names.map(
                lambda c: inclusion.get(c, {}).get("weight", 0.0)
            ).to_numpy(dtype = float)
            fold_pool = fit_pool(train, train_weights)
            fold_index = segment_scores(rows, segment, fold_pool, index_mode)
            train = train.assign(score = fold_index)
            scored_rows = rows.assign(score = fold_index)

        train_scores = train["score"].to_numpy(dtype = float)
        finite = np.isfinite(train_scores)
        grid = np.linspace(float(train_scores[finite].min()),
                           float(train_scores[finite].max()),
                           fit_config.grid_points)
        curve = composite_curve(train, classes, grid, inclusion)

        held_scores = scored_rows.loc[held, "score"].to_numpy(dtype = float)
        predicted_all[held] = np.interp(held_scores, grid, curve,
                                        left = curve[0], right = curve[-1])
        w = class_names[held].map(
            lambda c: inclusion.get(c, {}).get("weight", 1.0)
        ).to_numpy(dtype = float)
        weight_all[held] = np.where(w > 0, w, 1.0)

    keep = np.isfinite(predicted_all) & (weight_all > 0)
    return {
        "n_folds": n_folds,
        "predicted": predicted_all[keep],
        "actual": rows["y"].to_numpy(dtype = float)[keep],
        "weight": weight_all[keep],
        "fold": fold_of[keep],
        "n_gold": int(keep.sum()),
    }


def scoring_rules(predicted: np.ndarray, actual: np.ndarray,
                  weight: np.ndarray) -> dict:
    """Design-weighted proper scores plus the CORP Brier decomposition.

    Brier and log score are both proper, so either ranks competing forecasts
    honestly; reporting both guards against a conclusion that rests on one
    rule's particular sensitivity. The decomposition (Dimitriadis, Gneiting &
    Jordan 2021) is the part that matters for choosing a curve *index*: ``DSC``
    is the discrimination the index supplies and ``MCB`` the miscalibration
    that recalibration removes anyway, so an index change that leaves ``DSC``
    alone has not cost predictive validity. ``UNC`` depends only on the
    outcomes and is therefore common to both candidates.
    """
    eps = 1e-12
    total = float(weight.sum())
    brier = float((weight * (actual - predicted) ** 2).sum() / total)
    clipped = np.clip(predicted, eps, 1.0 - eps)
    log_score = float(
        -(weight * (actual * np.log(clipped)
                    + (1 - actual) * np.log(1 - clipped))).sum() / total
    )
    base = float((weight * actual).sum() / total)
    uncertainty = base * (1.0 - base)

    # Recalibrate the predictions against the outcomes by PAV; the residual
    # score is the discrimination-only part, and the gap is miscalibration.
    order = np.lexsort((-actual, predicted))
    model = IsotonicRegression(y_min = 0.0, y_max = 1.0, increasing = True,
                               out_of_bounds = "clip")
    model.fit(predicted[order], actual[order], sample_weight = weight[order])
    recalibrated = np.clip(model.predict(predicted), 0.0, 1.0)
    brier_recalibrated = float(
        (weight * (actual - recalibrated) ** 2).sum() / total
    )
    return {
        "brier": brier,
        "log_score": log_score,
        "mcb": brier - brier_recalibrated,
        "dsc": uncertainty - brier_recalibrated,
        "unc": uncertainty,
        "base_rate": base,
        "n": int(len(predicted)),
    }


def cross_fit_calibration_error(rows: pd.DataFrame, classes: pd.Series,
                                grid: np.ndarray, fit_config: FitConfig,
                                n_folds: int = 5, segment: str = None,
                                index_mode: str = "pool") -> dict:
    """Design-respecting K-fold calibration error and proper scores.

    Thin wrapper over :func:`cross_fit_predictions`: the binned
    observed-minus-predicted gap on held-out gold, debiased by subtracting each
    bin's own sampling variance (Kumar, Liang & Ma 2019).
    """
    out_of_fold = cross_fit_predictions(
        rows, classes, fit_config, n_folds = n_folds, segment = segment,
        index_mode = index_mode,
    )
    if not out_of_fold.get("n_folds"):
        return {"n_folds": 0}

    predicted_scored = out_of_fold["predicted"]
    actual_scored = out_of_fold["actual"]
    w_scored = out_of_fold["weight"]
    scores = scoring_rules(predicted_scored, actual_scored, w_scored)

    # Calibration error is a BINNED gap between held-out observed rate and
    # prediction; the debiased form subtracts each bin's sampling variance of
    # its own rate estimate (Kumar, Liang & Ma 2019), not the irreducible
    # Bernoulli variance of individual outcomes.
    n_bins = min(10, max(2, int(len(predicted_scored) // 40)))
    edges = np.unique(
        np.quantile(predicted_scored, np.linspace(0.0, 1.0, n_bins + 1))
    )
    plug_in, correction, total_weight = 0.0, 0.0, float(w_scored.sum())
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (predicted_scored >= lo) & (predicted_scored <= hi)
        if in_bin.sum() < 2:
            continue
        bin_weight = float(w_scored[in_bin].sum())
        observed = float(np.average(actual_scored[in_bin],
                                    weights = w_scored[in_bin]))
        expected = float(np.average(predicted_scored[in_bin],
                                    weights = w_scored[in_bin]))
        plug_in += bin_weight * (observed - expected) ** 2
        ess = bin_weight**2 / float((w_scored[in_bin] ** 2).sum())
        if ess > 1:
            correction += bin_weight * observed * (1.0 - observed) / (ess - 1.0)

    return {
        "n_folds": out_of_fold["n_folds"],
        "n_gold": out_of_fold["n_gold"],
        "n_bins": int(len(edges) - 1),
        "index_mode": index_mode if segment in POOLED_SEGMENTS else "native",
        "brier_crossfit": scores["brier"],
        "log_score_crossfit": scores["log_score"],
        "mcb": scores["mcb"],
        "dsc": scores["dsc"],
        "unc": scores["unc"],
        "calibration_error_sq_plugin": plug_in / total_weight,
        "calibration_error_sq_debiased": max(
            (plug_in - correction) / total_weight, 0.0
        ),
    }


def constancy_check(rows: pd.DataFrame, classes: pd.Series) -> dict:
    """Gold existence rate per class in the low vs high half of the score.

    The definitive-class working models assume a flat rate in the score. This
    is the guard: a class whose two halves disagree materially wants the
    isotonic treatment instead of a constant.
    """
    gold = rows[rows["gold"].to_numpy(dtype = bool)]
    if gold.empty:
        return {}
    gold_classes = classes[rows["gold"].to_numpy(dtype = bool)].astype(str)
    median = float(gold["score"].median())
    out = {}
    for name in gold_classes.unique():
        in_class = (gold_classes == name).to_numpy()
        scores = gold["score"].to_numpy(dtype = float)[in_class]
        y = gold["y"].to_numpy(dtype = float)[in_class]
        low, high = y[scores <= median], y[scores > median]
        out[name] = {
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "rate_low": float(low.mean()) if len(low) else None,
            "rate_high": float(high.mean()) if len(high) else None,
            "gap": (
                float(abs(high.mean() - low.mean()))
                if len(low) and len(high) else None
            ),
        }
    return out


def fit_segment(rows: pd.DataFrame, segment: str, fit_config: FitConfig,
                population: pd.DataFrame = None,
                rng_offset: int = 0) -> dict:
    """Fit one segment's calibration curve and everything the report needs.

    ``population`` is the production population's raw source scores for this
    segment (columns ``osm_score`` / ``overture_score``), used only to place
    the lookup's equal-mass bins.
    """
    rows = rows.reset_index(drop = True)
    classes = merge_thin_cells(
        refined_class(rows, refine = fit_config.refine_by_confidence),
        rows["gold"].to_numpy(dtype = bool),
        fit_config.min_cell_gold,
    ).reset_index(drop = True)
    inclusion = inclusion_by_class(
        classes, rows["gold"].to_numpy(dtype = bool)
    )
    row_weights = classes.astype(str).map(
        lambda c: inclusion.get(c, {}).get("weight", 0.0)
    ).to_numpy(dtype = float)

    # Pooled segments learn their source weights from gold; single-source
    # segments index on their native score.
    index_mode = fit_config.matched_index_mode
    pool_params = (
        fit_pool(rows, row_weights) if needs_pool(segment, index_mode) else None
    )
    rows = rows.assign(
        score = segment_scores(rows, segment, pool_params, index_mode)
    )

    scores = rows["score"].to_numpy(dtype = float)
    grid = np.linspace(float(np.nanmin(scores)), float(np.nanmax(scores)),
                       fit_config.grid_points)
    curve = composite_curve(rows, classes, grid, inclusion)
    replicates = two_phase_bootstrap(rows, classes, grid, fit_config,
                                     rng_offset = rng_offset, segment = segment,
                                     index_mode = index_mode)
    summary = summarize_band(replicates, curve, alpha = fit_config.band_alpha)

    # The lookup's equal-mass bins span the PRODUCTION score distribution, so
    # for a pooled segment the population's two source columns are pushed
    # through the same fitted pool.
    if population is None:
        population_index = scores[np.isfinite(scores)]
    else:
        population_index = segment_scores(population, segment, pool_params,
                                          index_mode)
        population_index = population_index[np.isfinite(population_index)]
    lookup = build_lookup(segment, population_index, grid, summary,
                          fit_config.output_bins)

    gold_weights = classes.astype(str).map(
        lambda c: inclusion.get(c, {}).get("weight", 0.0)
    ).to_numpy(dtype = float)[rows["gold"].to_numpy(dtype = bool)]
    ess = (
        float(gold_weights.sum() ** 2 / (gold_weights**2).sum())
        if (gold_weights**2).sum() > 0 else 0.0
    )

    return {
        "segment": segment,
        "grid": grid,
        "curve": curve,
        "summary": summary,
        "lookup": lookup,
        "classes": classes,
        "inclusion": inclusion,
        "pool": pool_params,
        "scores": scores,
        "reference_curve": ht_reference_curve(rows, classes, grid, inclusion),
        "kish_ess": ess,
        "n_rows": int(len(rows)),
        "n_gold": int(rows["gold"].sum()),
        "band_width_median": float(
            np.median(summary["upper"] - summary["lower"])
        ),
        "constancy": constancy_check(rows, classes),
        "cross_fit": cross_fit_calibration_error(
            rows, classes, grid, fit_config, segment = segment,
            index_mode = index_mode,
        ),
        "index_mode": index_mode if segment in POOLED_SEGMENTS else "native",
    }


def fit_all_segments(validation_rows: pd.DataFrame, fit_config: FitConfig,
                     populations: dict = None) -> dict:
    """Fit every segment present in the handoff table.

    ``populations`` maps segment to a frame of the production population's
    ``osm_score`` / ``overture_score`` columns (bin placement only). Rows in
    the missing-confidence stratum are excluded: their Overture score is an
    upstream placeholder, and including them would put a false mass spike at
    0.5 in the curve.
    """
    results = {}
    usable = validation_rows[
        validation_rows["llm_verdict"].isin(VERDICTS)
        & validation_rows["stratum"].isin(SEGMENTS)
    ]
    for offset, segment in enumerate(SEGMENTS):
        rows = usable[usable["segment"] == segment]
        if len(rows) < 50:
            continue
        results[segment] = fit_segment(
            rows, segment, fit_config,
            population = (populations or {}).get(segment),
            rng_offset = offset,
        )
    return results


def curve_metadata(segment: str, result: dict, handoff_metadata: dict,
                   fit_config: FitConfig) -> dict:
    """Provenance + diagnostics pinned beside each fitted curve."""
    return {
        "segment": segment,
        "estimator": ESTIMATOR_TAG,
        "rogan_gladen_applied": False,
        "validation_round": handoff_metadata.get("validation_round"),
        "conflation_version": handoff_metadata.get("conflation_version"),
        "snapshot_osm": handoff_metadata.get("snapshot_osm"),
        "snapshot_overture": handoff_metadata.get("snapshot_overture"),
        "matched_collapse_method": handoff_metadata.get(
            "matched_collapse_method"
        ),
        "handoff_schema": handoff_metadata.get("export_schema"),
        "validator_git_sha": handoff_metadata.get("validator_git_sha"),
        "n_phase1_rows": result["n_rows"],
        "n_gold": result["n_gold"],
        # Pool coefficients for a two-source segment; ``null`` where the curve
        # is indexed on a single native score. The deploy step reads this.
        "pool": result["pool"],
        "index_mode": result.get("index_mode", "native"),
        "score_definition": _score_definition(
            segment, result.get("index_mode", "native")
        ),
        "effective_sample_size": result["kish_ess"],
        "band_width_median": result["band_width_median"],
        "refined_classes": result["inclusion"],
        "constancy_check": result["constancy"],
        "cross_fit": result["cross_fit"],
        "fit_config": {
            "min_cell_gold": fit_config.min_cell_gold,
            "grid_points": fit_config.grid_points,
            "output_bins": fit_config.output_bins,
            "bootstrap_reps": fit_config.bootstrap_reps,
            "band_alpha": fit_config.band_alpha,
            "rng_seed": fit_config.rng_seed,
            "refine_by_confidence": fit_config.refine_by_confidence,
            "matched_index_mode": fit_config.matched_index_mode,
        },
    }


def write_curve(out_dir, segment: str, lookup: pd.DataFrame,
                metadata: dict) -> None:
    """Write one segment's lookup parquet and metadata JSON."""
    out_dir.mkdir(parents = True, exist_ok = True)
    lookup.to_parquet(out_dir / f"{segment}_curve.parquet", index = False)
    with open(out_dir / f"{segment}_metadata.json", "w",
              encoding = "utf-8") as handle:
        json.dump(metadata, handle, indent = 2, sort_keys = True, default = str)
