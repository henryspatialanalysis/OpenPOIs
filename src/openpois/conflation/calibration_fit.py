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

- The second term corrects the first in expectation, so the estimator is
  design-unbiased for the cell mean *exactly* when the working model does not
  depend on the sample, and *asymptotically* unbiased when -- as here -- the
  working model is itself fit on the same gold (Breidt & Opsomer 2017 pp. 3, 6).
  A badly wrong working model costs variance, not consistency. A saturated
  working model collapses it to the pure Horvitz-Thompson estimator, which is
  algebraically the validator's as-built ``stratified_ht_gold_v1`` curve --
  retained here as the robustness reference.
- With the low-dimensional working model, the large inverse-inclusion weights
  (~8.6x on LLM-exists) multiply *near-zero residuals*, because that class's
  gold existence rate is 0.985. The censused unverifiable class carries no
  phase-2 sampling variance at all. This is where the precision the gold-only
  fit gave up comes back: the score shape comes from the full phase-1 archive.

The **matched** segment carries two source scores, so it is not calibrated on
the upstream 0.588/0.412 blend. Instead the two scores are combined into a
*fitted* monotone index, estimated by design-weighted logistic regression on
matched gold, and the segment curve is then fit design-based against
``expit(index)``. The production index (since the October 2026 run) is the
monotone bilinear interaction index on the scores' logits rescaled to [0, 1]::

    z = a0 + a1 * x + a2 * y + a3 * x * y,
    a1, a2, a1 + a3, a2 + a3 >= pool_min_coef

whose constraints keep it nondecreasing in both scores (Gupta et al. 2016).
It nests the log-odds pool used until then::

    z = b0 + b_osm * logit(s_osm) + b_ov * logit(s_ov)

which is the log-odds opinion pool in the free-coefficient form of Bordley
(1982) (see Genest & Zidek 1986 sec. 5). Either index can place a matched POI
*above* both sources' own confidences when they agree, which the linear blend
can never do. The pool is **not** the externally Bayesian logarithmic pool --
that one has weights summing to one and cannot exceed the larger input (Genest
& Zidek 1986 pp. 6, 9) -- and a fitted coefficient below 1 mixes damping for
dependence between the sources with each raw score's own miscalibration. The
fitted parameters replace the ``overture_confidence_weight`` constant, so no
fixed downweight of the Overture score survives anywhere in the calibrated
output.

``matched_index_mode`` selects how the two scores are combined (compared in
``scripts/conflation/compare_matched_index.py``; the production setting lives in
``config.yaml``, ``conflation.calibration.matched_index_mode``, and is
``interaction`` since the October 2026 run):

``pool``
    the constrained log-odds pool above (3 parameters; production to 2026-09)
``average``
    the unweighted mean of the two raw scores (no parameters)
``additive``
    ``a + h_osm(s_osm) + h_ov(s_ov)`` with each ``h`` nondecreasing, fit by
    local scoring with isotonic backfitting (Bacchetti 1989)
``interaction``
    a monotone bilinear model in rescaled logits (Gupta et al. 2016); the
    production index. It nests the pool (``a3 = 0``) and can represent the
    substitutive "either source suffices" structure round 20260730 showed
``surface``
    no index: a per-cell difference estimator on a coarse 2-D grid, projected
    onto the doubly-monotone cone (Dykstra & Robertson 1982)

The first four reduce the pair to a monotone scalar index and reuse the 1-D
machinery; the surface is its own lookup.

Uncertainty is a two-phase bootstrap: phase-1 rows resampled within verdict
class at fixed class counts, gold and non-gold resampled separately within
verdict class at fixed counts, refitting everything -- index parameters
included -- per replicate. With ``band_aggregation = "bin"`` (production since
October 2026) each replicate's map is averaged over the production POIs in each
published bin, so the band describes exactly the published quantity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, isotonic_regression, minimize
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

SEGMENTS = ("matched", "osm", "overture")
# Segments whose curve is indexed on a pooled function of two source scores
# rather than on a single native score.
POOLED_SEGMENTS = ("matched",)
# Clip before the logit so scores at exactly 0 or 1 stay finite.
LOGIT_EPS = 1e-4
# The largest log-odds any index can reach; the additive fit clips here so an
# all-positive block cannot diverge.
LOGIT_MAX = float(np.log((1.0 - LOGIT_EPS) / LOGIT_EPS))
# Source scores are rounded before any binning. The validation file splits each
# Overture atom (0.919912, 0.990219) into three float representations that
# production does not have; unrounded, ~320 rows land on the wrong side of an
# edge placed at the atom.
SCORE_DECIMALS = 6
SCORE_COLUMNS = ("osm_score", "overture_score")
# How a two-source segment combines its scores; see the module docstring.
INDEX_MODES = ("pool", "average", "additive", "interaction", "surface")
# The 2-D lookup schema for ``surface`` mode.
SURFACE_CURVE_COLUMNS = ("segment", "osm_lo", "osm_hi", "ov_lo", "ov_hi",
                         "conf_mean", "conf_lower", "conf_upper")
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
    # Two-source segments: one of INDEX_MODES. The defaults here match the
    # production config (conflation.calibration in config.yaml), which the
    # drivers pass explicitly. Compare modes with
    # scripts/conflation/compare_matched_index.py before changing.
    matched_index_mode: str = "interaction"
    # Lower bound on every slope of a fitted index, so no source can enter
    # with a zero or negative weight.
    pool_min_coef: float = 1e-3
    # Surface mode: requested per-axis equal-mass bins over the production
    # population. Atoms collapse duplicate edges, so fewer may be realized
    # (6 requested Overture bins realize 4 on round 20260730).
    surface_osm_bins: int = 6
    surface_ov_bins: int = 6
    surface_max_sweeps: int = 10_000
    surface_tol: float = 1e-10
    # Normal draws for the Xu-Meyer-Opsomer mixture covariance.
    wald_mixture_draws: int = 2000
    # Which surface band is published: "percentile" (bootstrap) or "wald"
    # (mixture covariance). Both are always computed and reported.
    surface_band_method: str = "percentile"
    # How the 1-D pipeline's band reaches the published bins: "bin" (each
    # replicate's map averaged over the production POIs in each published
    # bin; production since October 2026) or "anchored_kernel" (the
    # pre-2026-09 method: POI-anchored replicates re-smoothed onto the grid,
    # then bin-averaged, which shrinks and off-centres the band).
    band_aggregation: str = "bin"
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


def round_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``frame`` with its source-score columns rounded.

    Every mode reads scores through this, in the validation rows and in the
    production population alike, so a score sits in the same bin at fit time
    and at deploy time. See ``SCORE_DECIMALS`` for why.
    """
    present = [c for c in SCORE_COLUMNS if c in frame.columns]
    if not present:
        return frame
    out = frame.copy()
    for column in present:
        out[column] = pd.to_numeric(out[column], errors = "coerce").round(
            SCORE_DECIMALS
        )
    return out


def _gold_design(rows: pd.DataFrame, weights: np.ndarray):
    """Gold rows' source scores, outcomes and design weights, finite only."""
    gold_mask = rows["gold"].to_numpy(dtype = bool)
    gold = rows[gold_mask]
    w = np.asarray(weights, dtype = float)[gold_mask]
    y = gold["y"].to_numpy(dtype = float)
    osm = gold["osm_score"].to_numpy(dtype = float)
    overture = gold["overture_score"].to_numpy(dtype = float)
    usable = (
        np.isfinite(osm) & np.isfinite(overture) & np.isfinite(y) & (w > 0)
    )
    return osm[usable], overture[usable], y[usable], w[usable]


def _identified(y: np.ndarray) -> bool:
    """Whether gold is thick and two-sided enough to fit index weights."""
    return len(y) >= 30 and len(np.unique(y)) >= 2


def _equal_weight_pool(n_gold: int) -> dict:
    """The unfitted pool, used when gold cannot identify any index weights."""
    return {
        "form": "pool", "intercept": 0.0, "coef_osm": 1.0,
        "coef_overture": 1.0, "n_gold": int(n_gold),
        "method": "equal_weight_pool", "bound_active": [],
    }


def _log_loss_and_gradient(beta: np.ndarray, features: np.ndarray,
                           y: np.ndarray, w: np.ndarray):
    """Weighted Bernoulli log loss of ``features @ beta`` and its gradient."""
    z = features @ beta
    loss = float(np.sum(w * (np.logaddexp(0.0, z) - y * z)))
    residual = w * (expit(z) - y)
    return loss, features.T @ residual


def fit_pool(rows: pd.DataFrame, weights: np.ndarray,
             min_coef: float = 1e-3) -> dict:
    """Design-weighted log-odds pool of the two source scores on gold rows.

    Minimizes the design-weighted log loss of
    ``b0 + b_osm * logit(s_osm) + b_ov * logit(s_ov)`` directly
    (L-BFGS-B), with the intercept free and both slopes bounded below by
    ``min_coef``. A slope at or below zero therefore cannot be fit, which is
    what keeps the index -- and so the calibrated map -- monotone in each
    source. ``bound_active`` names any slope that ended on its bound.

    Returns ``{form, intercept, coef_osm, coef_overture, n_gold, method,
    bound_active}``. A coefficient below 1 mixes dependence damping with that
    raw score's own miscalibration; see the module docstring.

    Falls back to the *unfitted* equal-weight pool when gold is too thin or
    one-sided: a logistic fit on a single outcome class is not identified.
    That fallback is about identification, not sign.
    """
    osm, overture, y, w = _gold_design(rows, weights)
    if not _identified(y):
        return _equal_weight_pool(len(y))
    features = np.column_stack([np.ones(len(y)), logit(osm), logit(overture)])
    # Normalizing the weights leaves the optimum unchanged and keeps the loss
    # on a scale where the default tolerances mean something.
    w_norm = w / w.mean()
    base = float(np.clip(np.average(y, weights = w), 0.01, 0.99))
    start = np.array([np.log(base / (1.0 - base)), 0.5, 0.5])
    bounds = [(None, None), (min_coef, None), (min_coef, None)]
    result = minimize(
        _log_loss_and_gradient, start, args = (features, y, w_norm),
        jac = True, method = "L-BFGS-B", bounds = bounds,
        options = {"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
    )
    if not np.all(np.isfinite(result.x)):
        return _equal_weight_pool(len(y))
    beta = result.x
    names = ("coef_osm", "coef_overture")
    return {
        "form": "pool",
        "intercept": float(beta[0]),
        "coef_osm": float(beta[1]),
        "coef_overture": float(beta[2]),
        "n_gold": int(len(y)),
        "method": "constrained_log_odds_pool_v2",
        "bound_active": [
            name for name, value in zip(names, beta[1:])
            if value <= min_coef * (1.0 + 1e-6)
        ],
    }


def pool_score(osm_score, overture_score, params: dict) -> np.ndarray:
    """Pooled index in [0, 1] from the two source scores.

    Monotone increasing in each source score whenever its coefficient is
    positive, which :func:`fit_pool` guarantees for every fitted pool.
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


# --- Additive isotonic index -------------------------------------------------

def _isotonic_on_knots(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted nondecreasing fit to per-knot values (knots already sorted)."""
    return isotonic_regression(values, weights = weights,
                               increasing = True).x


def _additive_eval(knots: np.ndarray, levels: np.ndarray,
                   scores) -> np.ndarray:
    """Evaluate one monotone component at ``scores``.

    Linear interpolation between knots with constant extension beyond them:
    interpolating a nondecreasing sequence stays nondecreasing, so the
    component is monotone on the whole score axis, not only at the knots.
    """
    return np.interp(np.asarray(scores, dtype = float), knots, levels)


def fit_additive_index(rows: pd.DataFrame, weights: np.ndarray,
                       max_outer: int = 100, max_inner: int = 50,
                       tol: float = 1e-8) -> dict:
    """Additive isotonic logistic index ``a + h_osm(s_osm) + h_ov(s_ov)``.

    Each ``h`` is nondecreasing and centred. Fit on gold with design weights by
    local scoring (Hastie & Tibshirani's IRLS outer loop) around isotonic
    backfitting -- Bacchetti's (1989) cyclic PAV. The outer loop forms the
    working response ``z + (y - p) / (p (1 - p))`` with working weights
    ``w p (1 - p)``; the inner loop cycles over components, each updated by a
    weighted PAV of its partial working residual. Backfitting onto monotone
    components is the dual of Dykstra's algorithm, so the inner loop converges
    (Mammen & Yu 2007 p. 15). A step-halving guard keeps the deviance from
    rising between outer iterations; a convex blend of two nondecreasing
    components is nondecreasing, so halving never breaks monotonicity.

    The linear predictor is clipped to +/- ``LOGIT_MAX`` so an all-positive
    top-right block cannot drive a component to infinity; ``n_clipped``
    records how many gold rows sit on the clip.

    Each component is stored as its knots (unique gold score values) and
    fitted levels. ``n_blocks`` (distinct levels per component) is the mode's
    realized effective parameter count.
    """
    osm, overture, y, w = _gold_design(rows, weights)
    if not _identified(y):
        return _equal_weight_pool(len(y))
    w = w / w.mean()
    axes = []
    for values in (osm, overture):
        knots, inverse = np.unique(values, return_inverse = True)
        axes.append((knots, inverse))

    def deviance(z):
        z = np.clip(z, -LOGIT_MAX, LOGIT_MAX)
        return float(2.0 * np.sum(w * (np.logaddexp(0.0, z) - y * z)))

    base = float(np.clip(np.average(y, weights = w), 1e-3, 1.0 - 1e-3))
    intercept = float(np.log(base / (1.0 - base)))
    levels = [np.zeros(len(knots)) for knots, _ in axes]

    def linear_predictor(a, comps):
        z = a + sum(comp[inv] for comp, (_, inv) in zip(comps, axes))
        return np.clip(z, -LOGIT_MAX, LOGIT_MAX)

    z = linear_predictor(intercept, levels)
    current = deviance(z)
    n_outer = 0
    converged = False
    for n_outer in range(1, max_outer + 1):
        p = expit(z)
        var = np.maximum(p * (1.0 - p), 1e-10)
        working = z + (y - p) / var
        omega = w * var

        new_levels = [lvl.copy() for lvl in levels]
        new_intercept = intercept
        for _ in range(max_inner):
            biggest = 0.0
            for j, (knots, inverse) in enumerate(axes):
                others = sum(
                    new_levels[k][axes[k][1]] for k in range(len(axes))
                    if k != j
                )
                partial = working - new_intercept - others
                knot_w = np.bincount(inverse, weights = omega,
                                     minlength = len(knots))
                knot_r = np.bincount(inverse, weights = omega * partial,
                                     minlength = len(knots))
                positive = knot_w > 0
                fitted = np.zeros(len(knots))
                fitted[positive] = _isotonic_on_knots(
                    knot_r[positive] / knot_w[positive], knot_w[positive]
                )
                if not positive.all():
                    idx = np.flatnonzero(positive)
                    fitted = np.interp(np.arange(len(knots)), idx,
                                       fitted[positive])
                centre = float(np.sum(knot_w * fitted) / knot_w.sum())
                fitted = fitted - centre
                biggest = max(biggest,
                              float(np.max(np.abs(fitted - new_levels[j]))))
                new_levels[j] = fitted
            residual = working - sum(
                lvl[inv] for lvl, (_, inv) in zip(new_levels, axes)
            )
            new_intercept = float(np.sum(omega * residual) / omega.sum())
            if biggest < 1e-10:
                break

        # Step-halving toward the previous fit if the deviance rose.
        step = 1.0
        for _ in range(30):
            trial_levels = [
                old + step * (new - old)
                for old, new in zip(levels, new_levels)
            ]
            trial_intercept = intercept + step * (new_intercept - intercept)
            trial_z = linear_predictor(trial_intercept, trial_levels)
            trial = deviance(trial_z)
            if trial <= current * (1.0 + 1e-12):
                break
            step /= 2.0
        change = abs(current - trial) / max(abs(current), 1e-12)
        levels, intercept, z, current = (trial_levels, trial_intercept,
                                         trial_z, trial)
        if change < tol:
            converged = True
            break

    # Monotone components by construction; enforce exactly against round-off.
    levels = [np.maximum.accumulate(lvl) for lvl in levels]
    raw_z = intercept + sum(
        lvl[inv] for lvl, (_, inv) in zip(levels, axes)
    )
    return {
        "form": "additive",
        "intercept": float(intercept),
        "knots_osm": axes[0][0].tolist(),
        "levels_osm": levels[0].tolist(),
        "knots_overture": axes[1][0].tolist(),
        "levels_overture": levels[1].tolist(),
        "n_blocks_osm": int(len(np.unique(np.round(levels[0], 10)))),
        "n_blocks_overture": int(len(np.unique(np.round(levels[1], 10)))),
        "n_clipped": int(np.sum(np.abs(raw_z) >= LOGIT_MAX)),
        "n_outer_iterations": int(n_outer),
        "converged": bool(converged),
        "deviance": float(current),
        "n_gold": int(len(y)),
        "method": "additive_isotonic_local_scoring_v1",
    }


def additive_score(osm_score, overture_score, params: dict) -> np.ndarray:
    """Additive-index probability ``expit(a + h_osm + h_ov)`` in [0, 1]."""
    z = (
        float(params["intercept"])
        + _additive_eval(np.asarray(params["knots_osm"]),
                         np.asarray(params["levels_osm"]), osm_score)
        + _additive_eval(np.asarray(params["knots_overture"]),
                         np.asarray(params["levels_overture"]),
                         overture_score)
    )
    return np.clip(expit(np.clip(z, -LOGIT_MAX, LOGIT_MAX)), 0.0, 1.0)


# --- Monotone bilinear interaction index ------------------------------------

def _unit_logit(values) -> np.ndarray:
    """Logit rescaled to [0, 1] over the FIXED clip range, not the data's.

    Using the clip range means the monotonicity constraints below cover every
    score the deploy step can ever see, not just the range the gold spanned.
    """
    return (logit(values) + LOGIT_MAX) / (2.0 * LOGIT_MAX)


# Rows of C in C @ (a0, a1, a2, a3) >= eps: necessary and sufficient for
# a0 + a1 x + a2 y + a3 x y to be nondecreasing in x and y on the unit square
# (Gupta et al. 2016 p. 6).
_INTERACTION_CONSTRAINTS = np.array(
    [[0.0, 1.0, 0.0, 0.0],   # a1 >= eps        (d/dx at y = 0)
     [0.0, 0.0, 1.0, 0.0],   # a2 >= eps        (d/dy at x = 0)
     [0.0, 1.0, 0.0, 1.0],   # a1 + a3 >= eps   (d/dx at y = 1)
     [0.0, 0.0, 1.0, 1.0]]   # a2 + a3 >= eps   (d/dy at x = 1)
)
_INTERACTION_CONSTRAINT_NAMES = ("a1", "a2", "a1+a3", "a2+a3")


def fit_interaction_index(rows: pd.DataFrame, weights: np.ndarray,
                          min_coef: float = 1e-3) -> dict:
    """Monotone bilinear logistic index on rescaled logits.

    ``z = a0 + a1 x + a2 y + a3 x y`` with ``x``, ``y`` the source logits
    rescaled to [0, 1] over the fixed clip range. Minimizes the
    design-weighted log loss subject to the four linear constraints of
    ``_INTERACTION_CONSTRAINTS`` (``trust-constr``). ``a3 < 0`` is a
    substitutive interaction ("either source suffices"), ``a3 > 0`` a
    complementary one. Starts from the constrained pool, mapped onto this
    parametrization with ``a3 = 0``, which is always feasible.
    """
    osm, overture, y, w = _gold_design(rows, weights)
    if not _identified(y):
        return _equal_weight_pool(len(y))
    x_osm, x_ov = _unit_logit(osm), _unit_logit(overture)
    features = np.column_stack(
        [np.ones(len(y)), x_osm, x_ov, x_osm * x_ov]
    )
    w_norm = w / w.mean()
    pool = fit_pool(rows, weights, min_coef = min_coef)
    span = 2.0 * LOGIT_MAX
    start = np.array([
        pool["intercept"] - (pool["coef_osm"] + pool["coef_overture"])
        * LOGIT_MAX,
        max(pool["coef_osm"] * span, 2.0 * min_coef),
        max(pool["coef_overture"] * span, 2.0 * min_coef),
        0.0,
    ])

    def hessian(beta, features = features, w = w_norm):
        p = expit(features @ beta)
        return (features * (w * p * (1.0 - p))[:, None]).T @ features

    constraint = LinearConstraint(_INTERACTION_CONSTRAINTS, min_coef, np.inf)
    result = minimize(
        _log_loss_and_gradient, start, args = (features, y, w_norm),
        jac = True, hess = lambda b, *_: hessian(b), method = "trust-constr",
        constraints = [constraint],
        options = {"gtol": 1e-10, "xtol": 1e-12, "maxiter": 3000},
    )
    beta = result.x
    # trust-constr is an interior method and can sit a hair outside a bound;
    # nudge back so the monotonicity guarantee is exact.
    slack = _INTERACTION_CONSTRAINTS @ beta - min_coef
    if slack.min() < 0:
        beta = beta.copy()
        beta[1] = max(beta[1], min_coef, min_coef - beta[3])
        beta[2] = max(beta[2], min_coef, min_coef - beta[3])
        slack = _INTERACTION_CONSTRAINTS @ beta - min_coef
    return {
        "form": "interaction",
        "a0": float(beta[0]),
        "a1": float(beta[1]),
        "a2": float(beta[2]),
        "a3": float(beta[3]),
        "scale": "unit_logit_clip_range",
        "constraints_active": [
            name for name, s in zip(_INTERACTION_CONSTRAINT_NAMES, slack)
            if s <= 1e-6 * max(1.0, abs(min_coef))
        ],
        "n_gold": int(len(y)),
        "optimizer_status": int(result.status),
        "method": "monotone_bilinear_logit_v1",
    }


def interaction_score(osm_score, overture_score, params: dict) -> np.ndarray:
    """Interaction-index probability in [0, 1]."""
    x, y = _unit_logit(osm_score), _unit_logit(overture_score)
    z = (float(params["a0"]) + float(params["a1"]) * x
         + float(params["a2"]) * y + float(params["a3"]) * x * y)
    return np.clip(expit(z), 0.0, 1.0)


# --- Index dispatch ----------------------------------------------------------

def index_form(params: dict) -> str:
    """The functional form of fitted index params (old curves are pools)."""
    return (params or {}).get("form", "pool")


def index_score(osm_score, overture_score, params: dict) -> np.ndarray:
    """Evaluate any fitted index form. The deploy step calls this too."""
    form = index_form(params)
    if form == "pool":
        return pool_score(osm_score, overture_score, params)
    if form == "additive":
        return additive_score(osm_score, overture_score, params)
    if form == "interaction":
        return interaction_score(osm_score, overture_score, params)
    raise ValueError(f"Unknown index form {form!r}")


def fit_index(rows: pd.DataFrame, weights: np.ndarray, index_mode: str,
              fit_config: "FitConfig") -> dict:
    """Fit the index a mode needs, or ``None`` for the parameter-free average.

    ``surface`` has no index of its own, but its difference estimator needs a
    working-model index; it uses the constrained pool.
    """
    if index_mode == "average":
        return None
    if index_mode in ("pool", "surface"):
        return fit_pool(rows, weights, min_coef = fit_config.pool_min_coef)
    if index_mode == "additive":
        return fit_additive_index(rows, weights)
    if index_mode == "interaction":
        return fit_interaction_index(rows, weights,
                                     min_coef = fit_config.pool_min_coef)
    raise ValueError(f"Unknown index_mode {index_mode!r}")


def segment_scores(rows: pd.DataFrame, segment: str,
                   index_params: dict = None,
                   index_mode: str = "pool") -> np.ndarray:
    """Curve-index score per row for one segment.

    For a two-source segment, ``index_mode`` selects the combination (see
    ``INDEX_MODES``); every mode but ``average`` needs fitted ``index_params``
    (for ``surface`` these are the working-model pool). Single-source segments
    ignore both and use their native score.
    """
    if segment in POOLED_SEGMENTS:
        osm = rows["osm_score"].to_numpy(dtype = float)
        overture = rows["overture_score"].to_numpy(dtype = float)
        if index_mode not in INDEX_MODES:
            raise ValueError(f"Unknown index_mode {index_mode!r}")
        if index_mode == "average":
            return average_score(osm, overture)
        if index_params is None:
            raise ValueError(f"Segment {segment} needs index parameters")
        return index_score(osm, overture, index_params)
    column = "osm_score" if segment == "osm" else "overture_score"
    return rows[column].to_numpy(dtype = float)


def needs_index(segment: str, index_mode: str) -> bool:
    """Whether this (segment, index_mode) pair estimates index parameters."""
    return segment in POOLED_SEGMENTS and index_mode != "average"


def _score_definition(segment: str, index_mode: str) -> str:
    """Human-readable statement of what a segment's curve is indexed on."""
    if segment in POOLED_SEGMENTS:
        inputs = "(osm_conf_mean, overture_confidence)"
        return {
            "average": f"mean{inputs}",
            "pool": f"constrained_log_odds_pool{inputs}",
            "additive": f"additive_isotonic_logit{inputs}",
            "interaction": f"monotone_bilinear_logit{inputs}",
            "surface": f"surface_cells{inputs}",
        }.get(index_mode, f"{index_mode}{inputs}")
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
    back to the pooled gold rate. Such a class contributes no correction term,
    so its rows keep that fallback prediction uncorrected: the estimator is no
    longer unbiased for the part of the cell mean those rows carry, and the
    bias is bounded by their share of the cell.
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
    """Isotonic projection of the fitted curve, weighted by score mass.

    ``population_scores`` is whatever the caller passes; :func:`composite_curve`
    passes the **phase-1 validation rows'** index scores, not the production
    population. On round 20260730 the two are close (the matched phase-1 cells
    sit near production blend deciles), but the weighting is the sample's.
    Weighting by mass means the monotonization spends its freedom where the
    data lives, not on empty score regions.
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
                        index_mode: str = "pool",
                        return_maps: bool = False):
    """Bootstrap replicates of the composite curve, respecting both phases.

    Phase-1 rows are resampled with replacement within verdict class holding
    class counts fixed (propagating class-mix and prediction variance); within
    each resampled class the gold rows are resampled to the design's realized
    gold count (propagating correction variance). A censused class contributes
    no phase-2 variance because its gold count equals its population.

    For a pooled segment the index parameters are refit inside each replicate,
    so the band includes uncertainty in the index itself. That makes
    the band **POI-anchored** rather than grid-anchored: each replicate's fitted
    map is evaluated on the *original* rows (their replicate index score through
    the replicate curve), and those per-row probabilities are then aggregated
    back onto the point estimate's grid. Comparing replicates at a fixed grid
    value instead would conflate real uncertainty with the harmless
    reparameterization of the index when its parameters move.

    With ``return_maps`` it also returns each replicate's
    ``(index_params, grid, curve)`` so :func:`bin_band_from_maps` can build the
    band on the published bins without the second kernel pass above (which
    shrinks and off-centres the band; see the 2026-09-25 coverage study).
    """
    rng = np.random.default_rng(fit_config.rng_seed + 100 + rng_offset)
    replicates = np.empty((fit_config.bootstrap_reps, len(grid)), dtype = float)
    maps = [None] * fit_config.bootstrap_reps
    groups, gold_groups, nongold_groups = _verdict_groups(rows)

    # Anchor: the point estimate's own index scores, and the kernel that maps
    # per-row probabilities back onto the reporting grid.
    anchor_scores = rows["score"].to_numpy(dtype = float)
    anchor_kernel = _kernel_matrix(
        anchor_scores, grid, _default_bandwidth(anchor_scores)
    )
    anchor_total = anchor_kernel.sum(axis = 1)

    for rep in range(fit_config.bootstrap_reps):
        take = _resample_two_phase(rows, rng, groups, gold_groups,
                                   nongold_groups)
        if not len(take):
            replicates[rep] = np.nan
            continue
        boot_rows = rows.iloc[take].reset_index(drop = True)
        boot_classes = classes.iloc[take].reset_index(drop = True)
        boot_inclusion = inclusion_by_class(
            boot_classes, boot_rows["gold"].to_numpy(dtype = bool)
        )
        # Score the ORIGINAL rows under this replicate's fitted map, so the
        # band is anchored to POIs rather than to index values.
        replicate_anchor = anchor_scores
        boot_index = None
        if needs_index(segment, index_mode):
            boot_weights = boot_classes.astype(str).map(
                lambda c: boot_inclusion.get(c, {}).get("weight", 0.0)
            ).to_numpy(dtype = float)
            boot_index = fit_index(boot_rows, boot_weights, index_mode,
                                   fit_config)
            boot_rows = boot_rows.assign(
                score = segment_scores(boot_rows, segment, boot_index,
                                       index_mode)
            )
            replicate_anchor = segment_scores(rows, segment, boot_index,
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
        maps[rep] = (boot_index, boot_grid, boot_curve)
        per_row = np.interp(replicate_anchor, boot_grid, boot_curve,
                            left = boot_curve[0], right = boot_curve[-1])
        with np.errstate(invalid = "ignore", divide = "ignore"):
            replicates[rep] = (
                (anchor_kernel * per_row[None, :]).sum(axis = 1) / anchor_total
            )
    if return_maps:
        return replicates, maps
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


def lookup_edges(population_scores: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-mass bin edges over the population index (atoms collapse)."""
    scores = np.asarray(population_scores, dtype = float)
    edges = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([scores.min(), scores.max() + 1e-9])
    return edges


def lookup_bins(scores, edges: np.ndarray) -> np.ndarray:
    """Bin of each score under the DEPLOY convention.

    ``searchsorted(edges, s, side = "right") - 1`` clipped to the end bins:
    a score equal to an interior edge belongs to the bin that starts there,
    exactly as :func:`apply_step_lookup` serves it. (Before 2026-09 the fit
    averaged ``lo <= s <= hi``, which counted edge values in the lower bin
    while deploy served them from the upper one; 37% of overture-segment
    rows sit on an edge.)
    """
    return np.clip(np.searchsorted(edges, np.asarray(scores, dtype = float),
                                   side = "right") - 1, 0, len(edges) - 2)


def build_lookup(segment: str, population_scores: np.ndarray,
                 grid: np.ndarray, summary: dict, n_bins: int,
                 bin_band: dict = None) -> pd.DataFrame:
    """Equal-mass lookup over the population score distribution.

    Scaling-binning (Kumar, Liang & Ma 2019): fit the smooth map first, then
    average its *values* over the population rows in each equal-mass bin,
    assigned with the deploy convention (:func:`lookup_bins`).

    ``bin_band`` (``{"lower", "upper"}`` arrays, one value per bin) replaces
    the grid band's bin averages -- the ``band_aggregation = "bin"`` path,
    whose band is computed directly on each bin's published quantity.
    """
    scores = np.asarray(population_scores, dtype = float)
    edges = lookup_edges(scores, n_bins)
    bins = lookup_bins(scores, edges)
    n_out = len(edges) - 1

    def _bin_means(values):
        fitted = np.interp(scores, grid, values, left = values[0],
                           right = values[-1])
        total = np.bincount(bins, weights = fitted, minlength = n_out)
        count = np.bincount(bins, minlength = n_out)
        mids = (edges[:-1] + edges[1:]) / 2.0
        fallback = np.interp(mids, grid, values, left = values[0],
                             right = values[-1])
        with np.errstate(invalid = "ignore", divide = "ignore"):
            return np.where(count > 0, total / np.maximum(count, 1), fallback)

    mean = _bin_means(summary["mean"])
    if bin_band is None:
        lower, upper = _bin_means(summary["lower"]), _bin_means(summary["upper"])
    else:
        lower = np.asarray(bin_band["lower"], dtype = float)
        upper = np.asarray(bin_band["upper"], dtype = float)
    lookup = pd.DataFrame({
        "segment": segment,
        "score_lo": edges[:-1].astype(float),
        "score_hi": edges[1:].astype(float),
        "conf_mean": mean,
        "conf_lower": lower,
        "conf_upper": upper,
    }, columns = list(CURVE_COLUMNS))
    for column in ("conf_mean", "conf_lower", "conf_upper"):
        lookup[column] = np.maximum.accumulate(lookup[column])
    if bin_band is not None:
        lookup["conf_lower"] = np.minimum(lookup["conf_lower"],
                                          lookup["conf_mean"])
        lookup["conf_upper"] = np.maximum(lookup["conf_upper"],
                                          lookup["conf_mean"])
    return lookup


def bin_band_from_maps(maps: list, population: pd.DataFrame, segment: str,
                       index_mode: str, edges: np.ndarray, alpha: float,
                       max_rows: int = 200_000, seed: int = 0) -> dict:
    """Band on each published bin from the bootstrap replicates' own maps.

    Each replicate's fitted map (its refit index and its curve) is applied
    to the production POIs, and the replicate's value for a bin is the mean
    over the POIs the point estimate places in that bin -- the published
    quantity itself. No second kernel pass, so the replicates centre on the
    point estimate. Up to ``max_rows`` POIs (a fixed random subsample) are
    used.
    """
    frame = population
    if len(frame) > max_rows:
        rng = np.random.default_rng(seed)
        frame = frame.iloc[np.sort(rng.choice(len(frame), max_rows,
                                              replace = False))]
    frame = frame.reset_index(drop = True)
    n_out = len(edges) - 1
    means = np.full((len(maps), n_out), np.nan)
    base_bins = None
    count = np.zeros(n_out)
    for rep, entry in enumerate(maps):
        if entry is None:
            continue
        params, grid, curve, point_index = entry
        if base_bins is None:
            base_bins = lookup_bins(
                segment_scores(frame, segment, point_index, index_mode), edges
            )
            count = np.bincount(base_bins, minlength = n_out)
        index = segment_scores(frame, segment, params, index_mode)
        values = np.interp(index, grid, curve, left = curve[0],
                           right = curve[-1])
        total = np.bincount(base_bins, weights = values, minlength = n_out)
        with np.errstate(invalid = "ignore", divide = "ignore"):
            means[rep] = np.where(count > 0, total / np.maximum(count, 1),
                                  np.nan)
    lower = np.nanquantile(means, alpha / 2.0, axis = 0)
    upper = np.nanquantile(means, 1.0 - alpha / 2.0, axis = 0)
    return {"lower": np.clip(lower, 0.0, 1.0), "upper": np.clip(upper, 0.0, 1.0)}


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
                          index_mode: str = "pool",
                          population: pd.DataFrame = None,
                          surface_bins: tuple = None) -> dict:
    """Out-of-fold predictions for every gold row, refitting the *whole* pipeline.

    Gold rows are folded within refined class so each training split preserves
    the design's class structure. Within a fold, everything estimated from gold
    is re-estimated on the training split alone -- **including the index
    parameters** for a pooled segment, which is what makes the returned
    predictions genuinely out-of-sample. Leaving an index fit on all gold would
    quietly hand a multi-parameter index an advantage over a parameter-free
    one, which is exactly the comparison this function exists to support.

    Held-out gold is predicted through the mode's **published lookup**, built
    per fold on the production ``population``: the equal-mass bin table for
    index modes, the cells for ``surface``. Scoring the deployed map is what
    makes a 40-bin mode and a 24-cell mode comparable. ``predicted_grid`` keeps
    the pre-2026-09 score (the fold curve interpolated at the held-out index)
    for continuity with the July comparison. Without a ``population`` the
    validation rows place the bins.

    Returns ``{predicted, predicted_grid, actual, weight, fold}`` aligned to
    the gold rows.
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

    base = population if population is not None and len(population) else None
    if base is not None:
        base = round_scores(base)
    surface = segment in POOLED_SEGMENTS and index_mode == "surface"
    if surface:
        n_osm, n_ov = surface_bins or (fit_config.surface_osm_bins,
                                       fit_config.surface_ov_bins)
        edges = surface_edges(base if base is not None else rows, n_osm, n_ov)
        shape = (len(edges["osm"]) - 1, len(edges["overture"]) - 1)
        i_osm, i_ov = surface_cells(rows["osm_score"], rows["overture_score"],
                                    edges)
        cell_base = base if base is not None else rows
        p_osm, p_ov = surface_cells(cell_base["osm_score"],
                                    cell_base["overture_score"], edges)
        projection_weights = np.maximum(
            _cell_sums(p_osm, p_ov, np.ones(len(cell_base)), shape), 1.0
        )

    predicted_all = np.full(len(rows), np.nan)
    predicted_grid_all = np.full(len(rows), np.nan)
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

        if surface:
            estimate, _, _, _ = _surface_estimate(
                train, classes, i_osm, i_ov, shape, fit_config,
                segment = segment,
            )
            estimate = np.where(np.isfinite(estimate), estimate,
                                np.nanmean(estimate))
            theta = project_monotone_2d(
                estimate, projection_weights,
                max_sweeps = fit_config.surface_max_sweeps,
                tol = fit_config.surface_tol,
            )
            predicted_all[held] = theta[i_osm[held], i_ov[held]]
            predicted_grid_all[held] = predicted_all[held]
        else:
            # Re-derive the index from the training gold only.
            scored_rows = rows
            fold_params = None
            if needs_index(segment, index_mode):
                train_weights = class_names.map(
                    lambda c: inclusion.get(c, {}).get("weight", 0.0)
                ).to_numpy(dtype = float)
                fold_params = fit_index(train, train_weights, index_mode,
                                        fit_config)
                fold_index = segment_scores(rows, segment, fold_params,
                                            index_mode)
                train = train.assign(score = fold_index)
                scored_rows = rows.assign(score = fold_index)

            train_scores = train["score"].to_numpy(dtype = float)
            finite = np.isfinite(train_scores)
            grid = np.linspace(float(train_scores[finite].min()),
                               float(train_scores[finite].max()),
                               fit_config.grid_points)
            curve = composite_curve(train, classes, grid, inclusion)

            held_scores = scored_rows.loc[held, "score"].to_numpy(dtype = float)
            predicted_grid_all[held] = np.interp(
                held_scores, grid, curve, left = curve[0], right = curve[-1]
            )
            if base is None:
                population_index = train_scores[finite]
            else:
                population_index = segment_scores(base, segment, fold_params,
                                                  index_mode)
                population_index = population_index[
                    np.isfinite(population_index)
                ]
            summary = {"mean": curve, "lower": curve, "upper": curve}
            lookup = build_lookup(segment, population_index, grid, summary,
                                  fit_config.output_bins)
            predicted_all[held] = apply_step_lookup(
                held_scores, lookup
            )["conf_mean"].to_numpy()

        w = class_names[held].map(
            lambda c: inclusion.get(c, {}).get("weight", 1.0)
        ).to_numpy(dtype = float)
        weight_all[held] = np.where(w > 0, w, 1.0)

    keep = np.isfinite(predicted_all) & (weight_all > 0)
    return {
        "n_folds": n_folds,
        "predicted": predicted_all[keep],
        "predicted_grid": predicted_grid_all[keep],
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
                                index_mode: str = "pool",
                                population: pd.DataFrame = None) -> dict:
    """Design-respecting K-fold calibration error and proper scores.

    Thin wrapper over :func:`cross_fit_predictions`: the binned
    observed-minus-predicted gap on held-out gold, debiased by subtracting each
    bin's own sampling variance (Kumar, Liang & Ma 2019).
    """
    out_of_fold = cross_fit_predictions(
        rows, classes, fit_config, n_folds = n_folds, segment = segment,
        index_mode = index_mode, population = population,
    )
    if not out_of_fold.get("n_folds"):
        return {"n_folds": 0}
    grid_scores = scoring_rules(out_of_fold["predicted_grid"],
                                out_of_fold["actual"], out_of_fold["weight"])

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
        # The pre-2026-09 score (fold curve interpolated, not the lookup).
        "brier_crossfit_grid": grid_scores["brier"],
        "log_score_crossfit_grid": grid_scores["log_score"],
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


# --- Surface: per-cell difference estimator on a doubly-monotone grid --------

def surface_edges(population: pd.DataFrame, n_osm: int,
                  n_overture: int) -> dict:
    """Per-axis equal-mass cell edges over the production population.

    Edges are ``np.unique``-collapsed quantiles of the rounded scores, so an
    atom holding more than one quantile's mass yields a single edge and opens
    its own (mass-dominated) bin. Fewer bins than requested may be realized.
    """
    population = round_scores(population)
    edges = {}
    for axis, column, n_bins in (("osm", "osm_score", n_osm),
                                 ("overture", "overture_score", n_overture)):
        values = population[column].to_numpy(dtype = float)
        values = values[np.isfinite(values)]
        axis_edges = np.unique(
            np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        )
        if len(axis_edges) < 2:
            axis_edges = np.array([values.min(), values.max() + 1e-9])
        edges[axis] = axis_edges
    return edges


def _bin_index(scores, edges: np.ndarray) -> np.ndarray:
    """Bin per score with the deploy convention: an edge value opens its bin,
    and scores outside the edges clamp to the end bins."""
    scores = np.asarray(scores, dtype = float)
    return np.clip(np.searchsorted(edges, scores, side = "right") - 1, 0,
                   len(edges) - 2)


def surface_cells(osm_score, overture_score, edges: dict):
    """``(osm_bin, overture_bin)`` per row under ``edges``."""
    return (_bin_index(osm_score, edges["osm"]),
            _bin_index(overture_score, edges["overture"]))


def _cell_sums(i_osm: np.ndarray, i_ov: np.ndarray, values: np.ndarray,
               shape: tuple) -> np.ndarray:
    """Sum of ``values`` per cell, as a ``shape`` matrix."""
    flat = i_osm * shape[1] + i_ov
    return np.bincount(flat, weights = values,
                       minlength = shape[0] * shape[1]).reshape(shape)


def _row_predictions(rows: pd.DataFrame, classes: pd.Series) -> np.ndarray:
    """Each row's working-model prediction ``q_c(i)(score_i)``."""
    scores = rows["score"].to_numpy(dtype = float)
    models = class_working_models(rows, classes, scores)
    names = classes.astype(str).to_numpy()
    out = np.empty(len(rows))
    for name, fitted in models.items():
        in_class = names == name
        out[in_class] = fitted[in_class]
    return out


def cell_difference_estimator(rows: pd.DataFrame, classes: pd.Series,
                              inclusion: dict, i_osm: np.ndarray,
                              i_ov: np.ndarray, shape: tuple):
    """Per-cell difference estimator (v4 sec. 4.1 in 2-D, with no kernel).

    ``p_hat(B) = (1 / N_B) [SUM_{i in B} q_i + SUM_{i in B and gold}
    (y_i - q_i) / pi_c(i)]`` with ``N_B`` the phase-1 rows in cell ``B`` and
    ``q`` the unchanged class working models, evaluated on ``rows["score"]``
    (the working-model index). Clipped to [0, 1]: the projection that follows
    does not restore the range (Dykstra 1983 p. 5). Returns
    ``(estimate, n_phase1)``; a cell with no phase-1 rows is NaN.
    """
    gold = rows["gold"].to_numpy(dtype = bool)
    y = np.nan_to_num(rows["y"].to_numpy(dtype = float), nan = 0.0)
    weights = classes.astype(str).map(
        lambda c: inclusion.get(c, {}).get("weight", 0.0)
    ).to_numpy(dtype = float)
    prediction = _row_predictions(rows, classes)
    active = gold & (weights > 0)
    contribution = prediction.copy()
    contribution[active] += (y[active] - prediction[active]) * weights[active]
    totals = _cell_sums(i_osm, i_ov, contribution, shape)
    counts = _cell_sums(i_osm, i_ov, np.ones(len(rows)), shape)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        estimate = np.where(counts > 0, totals / counts, np.nan)
    return np.clip(estimate, 0.0, 1.0), counts


def _isotonic_last_axis(values: np.ndarray, weights: np.ndarray
                        ) -> np.ndarray:
    """Weighted nondecreasing projection along the last axis, vectorized.

    Uses the min-max representation of the isotonic solution,
    ``x_i = max_{j <= i} min_{k >= i} avg_w(v_j..v_k)`` (Robertson, Wright &
    Dykstra 1988 Thm 1.4.4), which is O(n^3) per row but fully vectorized over
    every leading axis -- the right trade for rows of 4-8 cells projected
    thousands of times. ``weights`` broadcasts against ``values`` and must be
    strictly positive.
    """
    weights = np.broadcast_to(weights, values.shape)
    n = values.shape[-1]
    zero = np.zeros(values.shape[:-1] + (1,))
    cum_v = np.concatenate([zero, np.cumsum(weights * values, axis = -1)],
                           axis = -1)
    cum_w = np.concatenate([zero, np.cumsum(weights, axis = -1)], axis = -1)
    # avg[..., j, k] = weighted mean of v_j..v_k (j <= k), +inf below diagonal.
    num = cum_v[..., None, 1:] - cum_v[..., :-1, None]
    den = cum_w[..., None, 1:] - cum_w[..., :-1, None]
    upper = np.triu(np.ones((n, n), dtype = bool))
    with np.errstate(invalid = "ignore", divide = "ignore"):
        avg = np.where(upper, num / den, np.inf)
    # min over k >= i: reverse cumulative minimum along k.
    tail_min = np.flip(np.minimum.accumulate(np.flip(avg, -1), axis = -1), -1)
    # max over j <= i of tail_min[..., j, i].
    tail_min = np.where(upper, tail_min, -np.inf)
    return tail_min.max(axis = -2)


def _max_violation(values: np.ndarray) -> np.ndarray:
    """Largest monotonicity violation per matrix, along either axis."""
    along_ov = np.clip(values[..., :, :-1] - values[..., :, 1:], 0.0, None)
    along_osm = np.clip(values[..., :-1, :] - values[..., 1:, :], 0.0, None)
    worst = np.zeros(values.shape[:-2])
    if along_ov.size:
        worst = np.maximum(worst, along_ov.max(axis = (-2, -1)))
    if along_osm.size:
        worst = np.maximum(worst, along_osm.max(axis = (-2, -1)))
    return worst


def project_monotone_2d(values: np.ndarray, weights: np.ndarray,
                        max_sweeps: int = 10_000, tol: float = 1e-10,
                        use_increments: bool = True) -> np.ndarray:
    """Weighted least-squares projection onto the doubly-monotone cone.

    ``values[..., i_osm, i_ov]`` is projected so it is nondecreasing along both
    of its last two axes; leading axes are a batch. The algorithm is Dykstra &
    Robertson (1982): alternate row-wise and column-wise weighted PAV, carrying
    Dykstra's correction increments ``p`` and ``q``. Plain alternation (the
    ``use_increments = False`` regression path) reaches *a* doubly-monotone
    matrix but not the projection. Stops when every matrix's row and column
    passes agree to ``tol`` and its worst violation is below ``tol``; the
    increments themselves converge to nonzero values, so they are no stopping
    signal. Raises if ``max_sweeps`` pass without convergence.

    ``weights`` must be strictly positive: zero weights make the projection
    non-unique (Dykstra & Robertson 1982 p. 3). Callers floor them.
    """
    values = np.asarray(values, dtype = float)
    shape = values.shape
    weights = np.broadcast_to(np.asarray(weights, dtype = float), shape)
    if np.any(weights <= 0):
        raise ValueError("Projection weights must be strictly positive")
    # Flatten any batch axes; each matrix stops iterating once it converges.
    x = values.reshape((-1,) + shape[-2:]).copy()
    w = weights.reshape((-1,) + shape[-2:])
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    active = np.arange(len(x))
    for _ in range(max_sweeps):
        xa, wa = x[active], w[active]
        wa_t = np.swapaxes(wa, -1, -2)
        if use_increments:
            a = _isotonic_last_axis(xa + p[active], wa)
            p[active] = xa + p[active] - a
            b = np.swapaxes(
                _isotonic_last_axis(np.swapaxes(a + q[active], -1, -2), wa_t),
                -1, -2,
            )
            q[active] = a + q[active] - b
        else:
            a = _isotonic_last_axis(xa, wa)
            b = np.swapaxes(
                _isotonic_last_axis(np.swapaxes(a, -1, -2), wa_t), -1, -2
            )
        gap = np.abs(a - b).max(axis = (-2, -1))
        x[active] = b
        done = (gap < tol) & (_max_violation(b) < tol)
        active = active[~done]
        if not len(active):
            return x.reshape(shape)
    raise RuntimeError(
        f"Doubly-monotone projection did not converge in {max_sweeps} sweeps"
    )


def binding_blocks(theta: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Level-set labels of a projected matrix: cells joined by a binding
    adjacent constraint share a label.

    The binding constraints ``A_J`` of the projection are the adjacent pairs
    it made equal. The null space of ``A_J`` is the vectors constant on the
    connected components of those pairs, so these labels define
    ``I - P_J`` as W-weighted block averaging (Xu, Meyer & Opsomer 2021 eq.
    2.1), with no rank bookkeeping for dependent constraint rows.
    """
    rows, cols = theta.shape
    parent = list(range(rows * cols))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for r in range(rows):
        for c in range(cols):
            here = r * cols + c
            if c + 1 < cols and abs(theta[r, c + 1] - theta[r, c]) <= tol:
                union(here, here + 1)
            if r + 1 < rows and abs(theta[r + 1, c] - theta[r, c]) <= tol:
                union(here, here + cols)
    return np.array([find(i) for i in range(rows * cols)])


def _block_average_matrix(labels: np.ndarray, weights: np.ndarray
                          ) -> np.ndarray:
    """``I - P_J``: replace each cell by its block's W-weighted mean."""
    same = labels[:, None] == labels[None, :]
    mat = same * weights[None, :]
    return mat / mat.sum(axis = 1, keepdims = True)


def mixture_covariance(theta: np.ndarray, sigma: np.ndarray,
                       weights: np.ndarray, draws: int,
                       rng: np.random.Generator, max_sweeps: int = 10_000,
                       tol: float = 1e-10) -> dict:
    """Xu, Meyer & Opsomer (2021) mixture covariance of the projected surface.

    Draws ``y ~ N(theta, sigma)`` (``theta`` the constrained estimate,
    ``sigma`` the covariance of the *unconstrained* cell estimates), projects
    each draw with the estimator's own weights, records its binding set ``J``,
    and averages ``(I - P_J) sigma (I - P_J)'`` over draws (their eq. 2.6).
    Unlike the covariance at the observed ``J`` alone, this does not condition
    on which constraints happened to bind in this sample.
    """
    shape = theta.shape
    flat_theta = theta.ravel()
    flat_w = np.broadcast_to(weights, shape).ravel()
    # Symmetrize and jitter so a rank-deficient bootstrap covariance still
    # yields draws.
    sigma = (sigma + sigma.T) / 2.0
    jitter = 1e-12 * max(float(np.trace(sigma)), 1e-12)
    samples = rng.multivariate_normal(
        flat_theta, sigma + jitter * np.eye(len(flat_theta)), size = draws,
        method = "eigh",
    ).reshape((draws,) + shape)
    projected = project_monotone_2d(samples, weights, max_sweeps = max_sweeps,
                                    tol = tol)
    patterns = {}
    for draw in projected:
        labels = binding_blocks(draw)
        key = labels.tobytes()
        if key in patterns:
            patterns[key][1] += 1
        else:
            patterns[key] = [labels, 1]
    covariance = np.zeros_like(sigma)
    for labels, count in patterns.values():
        m = _block_average_matrix(labels, flat_w)
        covariance += count * (m @ sigma @ m.T)
    covariance /= draws
    return {"covariance": covariance, "n_patterns": len(patterns)}


def _kish_ess_by_cell(rows: pd.DataFrame, classes: pd.Series,
                      inclusion: dict, i_osm: np.ndarray, i_ov: np.ndarray,
                      shape: tuple) -> np.ndarray:
    """Kish effective sample size of the gold design weights in each cell."""
    gold = rows["gold"].to_numpy(dtype = bool)
    weights = classes.astype(str).map(
        lambda c: inclusion.get(c, {}).get("weight", 0.0)
    ).to_numpy(dtype = float) * gold
    total = _cell_sums(i_osm, i_ov, weights, shape)
    squares = _cell_sums(i_osm, i_ov, weights**2, shape)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        return np.where(squares > 0, total**2 / squares, 0.0)


def wald_mixture_band(theta: np.ndarray, unconstrained_replicates: np.ndarray,
                      projection_weights: np.ndarray, bound_weights: np.ndarray,
                      fit_config: "FitConfig", rng: np.random.Generator
                      ) -> dict:
    """Wald band from the mixture covariance, bounds projected onto the cone.

    1. ``sigma`` = bootstrap covariance of the unconstrained cell estimates.
    2. Mixture covariance of the constrained estimator
       (:func:`mixture_covariance`).
    3. ``theta +/- z * sqrt(diag)``.
    4. Each bound projected onto the doubly-monotone cone separately, weighted
       by the cell's gold effective sample size (Liao, Meyer & Xu 2024 sec.
       3), so thin cells borrow the most from their neighbours.

    Coverage theory for this band assumes the constraints hold strictly
    (Xu, Meyer & Opsomer 2021 Thm 2); where they bind -- plateaus and floors
    -- the coverage simulation is the check.
    """
    shape = theta.shape
    finite = np.all(np.isfinite(unconstrained_replicates.reshape(
        len(unconstrained_replicates), -1)), axis = 1)
    flat = unconstrained_replicates[finite].reshape(int(finite.sum()), -1)
    sigma = np.cov(flat, rowvar = False)
    mixture = mixture_covariance(
        theta, sigma, projection_weights, fit_config.wald_mixture_draws, rng,
        max_sweeps = fit_config.surface_max_sweeps, tol = fit_config.surface_tol,
    )
    z = float(norm.ppf(1.0 - fit_config.band_alpha / 2.0))
    se = np.sqrt(np.clip(np.diag(mixture["covariance"]), 0.0, None)).reshape(
        shape
    )
    floor_weights = np.maximum(bound_weights, 0.5)
    lower = project_monotone_2d(theta - z * se, floor_weights,
                                max_sweeps = fit_config.surface_max_sweeps,
                                tol = fit_config.surface_tol)
    upper = project_monotone_2d(theta + z * se, floor_weights,
                                max_sweeps = fit_config.surface_max_sweeps,
                                tol = fit_config.surface_tol)
    return {
        "lower": np.minimum(np.clip(lower, 0.0, 1.0), theta),
        "upper": np.maximum(np.clip(upper, 0.0, 1.0), theta),
        "se": se,
        "n_patterns": mixture["n_patterns"],
        "sigma_unconstrained": sigma,
    }


def _resample_two_phase(rows: pd.DataFrame, rng: np.random.Generator,
                        groups: dict, gold_groups: dict,
                        nongold_groups: dict) -> np.ndarray:
    """Row positions for one two-phase bootstrap replicate.

    Within each verdict class, gold and non-gold rows are resampled
    separately at their realized counts: phase-1 class counts stay fixed and
    the phase-2 draw is repeated at the design's gold count.
    """
    picks = []
    for verdict in groups:
        gold_idx, nongold_idx = gold_groups[verdict], nongold_groups[verdict]
        if len(gold_idx):
            picks.append(rng.choice(gold_idx, size = len(gold_idx),
                                    replace = True))
        if len(nongold_idx):
            picks.append(rng.choice(nongold_idx, size = len(nongold_idx),
                                    replace = True))
    return np.concatenate(picks) if picks else np.array([], dtype = int)


def _verdict_groups(rows: pd.DataFrame):
    verdicts = rows["llm_verdict"].astype(str).to_numpy()
    gold_mask = rows["gold"].to_numpy(dtype = bool)
    groups = {v: np.flatnonzero(verdicts == v) for v in np.unique(verdicts)}
    gold_groups = {v: idx[gold_mask[idx]] for v, idx in groups.items()}
    nongold_groups = {v: idx[~gold_mask[idx]] for v, idx in groups.items()}
    return groups, gold_groups, nongold_groups


def _surface_estimate(rows: pd.DataFrame, classes: pd.Series,
                      i_osm: np.ndarray, i_ov: np.ndarray, shape: tuple,
                      fit_config: "FitConfig", segment: str = "matched"):
    """Refit the working index and return the unconstrained cell estimate.

    Everything estimated from gold -- inclusion weights and the pool behind
    the working model -- is fit on ``rows`` alone, so this is the unit a
    bootstrap replicate or a cross-fit fold repeats.
    """
    inclusion = inclusion_by_class(classes, rows["gold"].to_numpy(dtype = bool))
    weights = classes.astype(str).map(
        lambda c: inclusion.get(c, {}).get("weight", 0.0)
    ).to_numpy(dtype = float)
    pool = fit_pool(rows, weights, min_coef = fit_config.pool_min_coef)
    scored = rows.assign(score = segment_scores(rows, segment, pool, "pool"))
    estimate, counts = cell_difference_estimator(scored, classes, inclusion,
                                                 i_osm, i_ov, shape)
    return estimate, counts, pool, inclusion


def surface_bootstrap(rows: pd.DataFrame, classes: pd.Series,
                      i_osm: np.ndarray, i_ov: np.ndarray, shape: tuple,
                      projection_weights: np.ndarray, fit_config: "FitConfig",
                      rng_offset: int = 0):
    """Two-phase bootstrap of the surface: unconstrained and projected.

    Resamples exactly as :func:`two_phase_bootstrap` does, refits the pool
    behind the working model per replicate, recomputes the cell estimates and
    projects each replicate. Cells are fixed (their edges come from the
    production population, not from gold), so replicates are compared cell by
    cell with no re-anchoring. A replicate that leaves any cell without
    phase-1 rows is dropped (NaN).
    """
    rng = np.random.default_rng(fit_config.rng_seed + 100 + rng_offset)
    groups, gold_groups, nongold_groups = _verdict_groups(rows)
    reps = fit_config.bootstrap_reps
    unconstrained = np.full((reps,) + shape, np.nan)
    for rep in range(reps):
        take = _resample_two_phase(rows, rng, groups, gold_groups,
                                   nongold_groups)
        if not len(take):
            continue
        boot_rows = rows.iloc[take].reset_index(drop = True)
        boot_classes = classes.iloc[take].reset_index(drop = True)
        estimate, _, _, _ = _surface_estimate(
            boot_rows, boot_classes, i_osm[take], i_ov[take], shape,
            fit_config,
        )
        if np.all(np.isfinite(estimate)):
            unconstrained[rep] = estimate
    ok = np.all(np.isfinite(unconstrained.reshape(reps, -1)), axis = 1)
    projected = np.full_like(unconstrained, np.nan)
    if ok.any():
        projected[ok] = project_monotone_2d(
            unconstrained[ok], projection_weights,
            max_sweeps = fit_config.surface_max_sweeps,
            tol = fit_config.surface_tol,
        )
    return unconstrained, projected


def percentile_surface_band(projected_replicates: np.ndarray,
                            theta: np.ndarray, projection_weights: np.ndarray,
                            fit_config: "FitConfig") -> dict:
    """Pointwise percentile band of the projected replicates, each bound then
    projected onto the cone with the estimator's weights and made to bracket
    the point estimate.

    Inherits the weakness of every percentile bootstrap of an
    order-constrained estimator: inconsistent where constraints bind
    (Andrews 2000; Sen, Banerjee & Woodroofe 2010).
    """
    alpha = fit_config.band_alpha
    lower = np.nanquantile(projected_replicates, alpha / 2.0, axis = 0)
    upper = np.nanquantile(projected_replicates, 1.0 - alpha / 2.0, axis = 0)
    lower = project_monotone_2d(lower, projection_weights,
                                max_sweeps = fit_config.surface_max_sweeps,
                                tol = fit_config.surface_tol)
    upper = project_monotone_2d(upper, projection_weights,
                                max_sweeps = fit_config.surface_max_sweeps,
                                tol = fit_config.surface_tol)
    return {
        "lower": np.minimum(np.clip(lower, 0.0, 1.0), theta),
        "upper": np.maximum(np.clip(upper, 0.0, 1.0), theta),
    }


def ht_reference_surface(rows: pd.DataFrame, classes: pd.Series,
                         inclusion: dict, i_osm: np.ndarray, i_ov: np.ndarray,
                         shape: tuple, projection_weights: np.ndarray,
                         fit_config: "FitConfig") -> np.ndarray:
    """Gold-only Hajek estimate per cell, projected the same way.

    The robustness overlay, as :func:`ht_reference_curve` is in 1-D. All-NaN
    if any cell has no gold, since the projection needs every cell.
    """
    gold = rows["gold"].to_numpy(dtype = bool)
    weights = classes.astype(str).map(
        lambda c: inclusion.get(c, {}).get("weight", 0.0)
    ).to_numpy(dtype = float) * gold
    y = np.nan_to_num(rows["y"].to_numpy(dtype = float), nan = 0.0)
    total = _cell_sums(i_osm, i_ov, weights * y, shape)
    mass = _cell_sums(i_osm, i_ov, weights, shape)
    if np.any(mass <= 0):
        return np.full(shape, np.nan)
    return project_monotone_2d(total / mass, projection_weights,
                               max_sweeps = fit_config.surface_max_sweeps,
                               tol = fit_config.surface_tol)


def surface_lookup(segment: str, edges: dict, mean: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray) -> pd.DataFrame:
    """The 2-D lookup: one row per cell, row-major over (osm, overture).

    Estimation grid = published grid, so there is no averaging step that
    could break monotonicity, and cross-fit scores exactly this table.
    """
    out = []
    osm_edges, ov_edges = edges["osm"], edges["overture"]
    for i in range(len(osm_edges) - 1):
        for j in range(len(ov_edges) - 1):
            out.append({
                "segment": segment,
                "osm_lo": float(osm_edges[i]),
                "osm_hi": float(osm_edges[i + 1]),
                "ov_lo": float(ov_edges[j]),
                "ov_hi": float(ov_edges[j + 1]),
                "conf_mean": float(mean[i, j]),
                "conf_lower": float(lower[i, j]),
                "conf_upper": float(upper[i, j]),
            })
    return pd.DataFrame(out, columns = list(SURFACE_CURVE_COLUMNS))


def surface_edges_from_lookup(lookup: pd.DataFrame) -> dict:
    """Recover the axis edges from a 2-D lookup table."""
    osm = np.unique(np.concatenate([lookup["osm_lo"], lookup["osm_hi"]]))
    overture = np.unique(np.concatenate([lookup["ov_lo"], lookup["ov_hi"]]))
    return {"osm": osm, "overture": overture}


def apply_surface(osm_score, overture_score, lookup: pd.DataFrame
                  ) -> pd.DataFrame:
    """Calibrated triple per row from a 2-D lookup.

    Two ``searchsorted(side = "right")`` calls, each clamped to the edge cells,
    mirroring :func:`apply_step_lookup`. A NaN in either score yields a NaN
    triple. Scores are rounded first, as at fit time.
    """
    osm = np.round(np.asarray(osm_score, dtype = float), SCORE_DECIMALS)
    overture = np.round(np.asarray(overture_score, dtype = float),
                        SCORE_DECIMALS)
    edges = surface_edges_from_lookup(lookup)
    i_osm, i_ov = surface_cells(osm, overture, edges)
    n_ov = len(edges["overture"]) - 1
    flat = i_osm * n_ov + i_ov
    out = pd.DataFrame({
        column: lookup[column].to_numpy(dtype = float)[flat]
        for column in ("conf_mean", "conf_lower", "conf_upper")
    })
    missing = ~(np.isfinite(osm) & np.isfinite(overture))
    if missing.any():
        out.loc[missing, :] = np.nan
    return out


def apply_step_lookup(scores, lookup: pd.DataFrame) -> pd.DataFrame:
    """Step-function lookup of the calibrated triple for 1-D index scores.

    An index equal to an edge goes to the bin that starts there; scores below
    the first bin or above the last clamp to them; NaN yields NaN. The deploy
    step's ``calibration.apply_curve`` is this function.
    """
    scores = np.asarray(scores, dtype = float)
    edges = lookup["score_lo"].to_numpy()
    idx = np.clip(np.searchsorted(edges, scores, side = "right") - 1, 0,
                  len(lookup) - 1)
    out = pd.DataFrame(
        {
            "conf_mean": lookup["conf_mean"].to_numpy()[idx],
            "conf_lower": lookup["conf_lower"].to_numpy()[idx],
            "conf_upper": lookup["conf_upper"].to_numpy()[idx],
        }
    )
    missing = ~np.isfinite(scores)
    if missing.any():
        out.loc[missing, :] = np.nan
    return out


def fit_surface_segment(rows: pd.DataFrame, classes: pd.Series,
                        fit_config: "FitConfig",
                        population: pd.DataFrame = None,
                        rng_offset: int = 0, segment: str = "matched",
                        n_osm: int = None, n_overture: int = None) -> dict:
    """Fit the doubly-monotone cell surface and both of its bands."""
    n_osm = n_osm or fit_config.surface_osm_bins
    n_overture = n_overture or fit_config.surface_ov_bins
    base = population if population is not None and len(population) else rows
    base = round_scores(base)
    edges = surface_edges(base, n_osm, n_overture)
    shape = (len(edges["osm"]) - 1, len(edges["overture"]) - 1)
    i_osm, i_ov = surface_cells(rows["osm_score"], rows["overture_score"],
                                edges)
    p_osm, p_ov = surface_cells(base["osm_score"], base["overture_score"],
                                edges)
    population_counts = _cell_sums(p_osm, p_ov, np.ones(len(base)), shape)
    projection_weights = np.maximum(population_counts, 1.0)

    unconstrained, counts, pool, inclusion = _surface_estimate(
        rows, classes, i_osm, i_ov, shape, fit_config, segment = segment
    )
    if np.any(counts == 0):
        empty = [(int(a), int(b)) for a, b in np.argwhere(counts == 0)]
        raise ValueError(
            f"Surface cells {empty} have no phase-1 rows; coarsen the grid "
            f"(surface_osm_bins / surface_ov_bins)"
        )
    theta = project_monotone_2d(unconstrained, projection_weights,
                                max_sweeps = fit_config.surface_max_sweeps,
                                tol = fit_config.surface_tol)
    boot_raw, boot_projected = surface_bootstrap(
        rows, classes, i_osm, i_ov, shape, projection_weights, fit_config,
        rng_offset = rng_offset,
    )
    percentile = percentile_surface_band(boot_projected, theta,
                                         projection_weights, fit_config)
    ess = _kish_ess_by_cell(rows, classes, inclusion, i_osm, i_ov, shape)
    wald = wald_mixture_band(
        theta, boot_raw, projection_weights, ess, fit_config,
        np.random.default_rng(fit_config.rng_seed + 900 + rng_offset),
    )
    bands = {"percentile": percentile, "wald": wald}
    method = fit_config.surface_band_method
    if method not in bands:
        raise ValueError(f"Unknown surface_band_method {method!r}")
    chosen = bands[method]
    lookup = surface_lookup(segment, edges, theta, chosen["lower"],
                            chosen["upper"])
    gold = rows["gold"].to_numpy(dtype = bool)
    return {
        "edges": edges,
        "shape": shape,
        "theta": theta,
        "unconstrained": unconstrained,
        "bands": bands,
        "band_method": method,
        "lookup": lookup,
        "pool": pool,
        "inclusion": inclusion,
        "phase1_counts": counts,
        "gold_counts": _cell_sums(i_osm[gold], i_ov[gold],
                                  np.ones(int(gold.sum())), shape),
        "gold_ess": ess,
        "population_counts": population_counts,
        "reference_surface": ht_reference_surface(
            rows, classes, inclusion, i_osm, i_ov, shape, projection_weights,
            fit_config,
        ),
        "bootstrap_unconstrained": boot_raw,
        "bootstrap_projected": boot_projected,
        "n_bootstrap_dropped": int(np.sum(~np.all(np.isfinite(
            boot_raw.reshape(len(boot_raw), -1)), axis = 1))),
        # The bottom cell of the product order has nothing below it to
        # borrow from (Liao, Meyer & Xu 2024 p. 5).
        "corner_cell_flag": {"cell": [0, 0], "gold": int(
            _cell_sums(i_osm[gold], i_ov[gold], np.ones(int(gold.sum())),
                       shape)[0, 0])},
    }


def fit_segment(rows: pd.DataFrame, segment: str, fit_config: FitConfig,
                population: pd.DataFrame = None,
                rng_offset: int = 0, surface_bins: tuple = None,
                cross_fit: bool = True) -> dict:
    """Fit one segment's calibration curve and everything the report needs.

    ``population`` is the production population's raw source scores for this
    segment (columns ``osm_score`` / ``overture_score``), used only to place
    the lookup's equal-mass bins (or, in ``surface`` mode, the cell edges and
    the projection weights). Scores are rounded to ``SCORE_DECIMALS`` first.
    ``surface_bins`` overrides the requested ``(osm, overture)`` cell counts.
    ``cross_fit = False`` skips the K-fold diagnostic (simulation runs).
    """
    rows = round_scores(rows.reset_index(drop = True))
    if population is not None:
        population = round_scores(population)
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
    gold_weights = row_weights[rows["gold"].to_numpy(dtype = bool)]
    ess = (
        float(gold_weights.sum() ** 2 / (gold_weights**2).sum())
        if (gold_weights**2).sum() > 0 else 0.0
    )

    # Pooled segments learn their index from gold; single-source segments
    # index on their native score.
    index_mode = fit_config.matched_index_mode
    if segment in POOLED_SEGMENTS and index_mode not in INDEX_MODES:
        raise ValueError(f"Unknown index_mode {index_mode!r}")
    index_params = (
        fit_index(rows, row_weights, index_mode, fit_config)
        if needs_index(segment, index_mode) else None
    )
    # In surface mode this is the working-model (pool) index.
    rows = rows.assign(
        score = segment_scores(rows, segment, index_params, index_mode)
    )
    scores = rows["score"].to_numpy(dtype = float)
    common = {
        "segment": segment,
        "classes": classes,
        "inclusion": inclusion,
        "scores": scores,
        "kish_ess": ess,
        "n_rows": int(len(rows)),
        "n_gold": int(rows["gold"].sum()),
        "constancy": constancy_check(rows, classes),
        "index_mode": index_mode if segment in POOLED_SEGMENTS else "native",
    }

    if segment in POOLED_SEGMENTS and index_mode == "surface":
        surface = fit_surface_segment(
            rows, classes, fit_config, population = population,
            rng_offset = rng_offset, segment = segment,
            n_osm = (surface_bins or (None, None))[0],
            n_overture = (surface_bins or (None, None))[1],
        )
        chosen = surface["bands"][surface["band_method"]]
        return {
            **common,
            "grid": None,
            "curve": None,
            "summary": None,
            "reference_curve": None,
            "lookup": surface["lookup"],
            "pool": None,
            "index": None,
            "working_index": surface["pool"],
            "surface": surface,
            "band_width_median": float(
                np.median(chosen["upper"] - chosen["lower"])
            ),
            "cross_fit": cross_fit_calibration_error(
                rows, classes, None, fit_config, segment = segment,
                index_mode = index_mode, population = population,
            ) if cross_fit else {"n_folds": 0},
        }

    grid = np.linspace(float(np.nanmin(scores)), float(np.nanmax(scores)),
                       fit_config.grid_points)
    curve = composite_curve(rows, classes, grid, inclusion)
    if fit_config.band_aggregation not in ("anchored_kernel", "bin"):
        raise ValueError(
            f"Unknown band_aggregation {fit_config.band_aggregation!r}"
        )
    replicates, maps = two_phase_bootstrap(
        rows, classes, grid, fit_config, rng_offset = rng_offset,
        segment = segment, index_mode = index_mode, return_maps = True,
    )
    summary = summarize_band(replicates, curve, alpha = fit_config.band_alpha)

    # The lookup's equal-mass bins span the PRODUCTION score distribution, so
    # for a pooled segment the population's two source columns are pushed
    # through the same fitted index.
    if population is None:
        population_index = scores[np.isfinite(scores)]
    else:
        population_index = segment_scores(population, segment, index_params,
                                          index_mode)
        population_index = population_index[np.isfinite(population_index)]
    bin_band = None
    if fit_config.band_aggregation == "bin":
        base = population if population is not None else rows
        mode = index_mode if segment in POOLED_SEGMENTS else "pool"
        bin_band = bin_band_from_maps(
            [None if m is None else (m[0], m[1], m[2], index_params)
             for m in maps],
            base, segment, mode,
            lookup_edges(population_index, fit_config.output_bins),
            fit_config.band_alpha, seed = fit_config.rng_seed + 1500,
        )
    lookup = build_lookup(segment, population_index, grid, summary,
                          fit_config.output_bins, bin_band = bin_band)

    return {
        **common,
        "grid": grid,
        "curve": curve,
        "summary": summary,
        "lookup": lookup,
        # ``pool`` stays populated for pool mode so curves remain readable by
        # deploy code that predates ``index``.
        "pool": (index_params if index_form(index_params) == "pool"
                 and index_params is not None else None),
        "index": index_params,
        "replicates": replicates,
        "reference_curve": ht_reference_curve(rows, classes, grid, inclusion),
        # The PUBLISHED band's median width over bins, and the grid band's
        # (the pre-2026-09 figure, dominated by empty low-index grid points).
        "band_width_median": float(
            np.median(lookup["conf_upper"] - lookup["conf_lower"])
        ),
        "band_width_grid_median": float(
            np.median(summary["upper"] - summary["lower"])
        ),
        "cross_fit": cross_fit_calibration_error(
            rows, classes, grid, fit_config, segment = segment,
            index_mode = index_mode, population = population,
        ) if cross_fit else {"n_folds": 0},
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


def bootstrap_index_params(rows: pd.DataFrame, index_mode: str,
                           fit_config: FitConfig, reps: int = None,
                           rng_offset: int = 0) -> list:
    """Index parameters refit on two-phase bootstrap replicates.

    The resampling of :func:`two_phase_bootstrap`, keeping each replicate's
    fitted params instead of its curve -- for intervals on the params
    themselves (the interaction's ``a3``, the additive components).
    """
    rows = round_scores(rows.reset_index(drop = True))
    classes = merge_thin_cells(
        refined_class(rows, refine = fit_config.refine_by_confidence),
        rows["gold"].to_numpy(dtype = bool), fit_config.min_cell_gold,
    ).reset_index(drop = True)
    rng = np.random.default_rng(fit_config.rng_seed + 500 + rng_offset)
    groups, gold_groups, nongold_groups = _verdict_groups(rows)
    out = []
    for _ in range(reps or fit_config.bootstrap_reps):
        take = _resample_two_phase(rows, rng, groups, gold_groups,
                                   nongold_groups)
        boot_rows = rows.iloc[take].reset_index(drop = True)
        boot_classes = classes.iloc[take].reset_index(drop = True)
        inclusion = inclusion_by_class(
            boot_classes, boot_rows["gold"].to_numpy(dtype = bool)
        )
        weights = boot_classes.astype(str).map(
            lambda c: inclusion.get(c, {}).get("weight", 0.0)
        ).to_numpy(dtype = float)
        out.append(fit_index(boot_rows, weights, index_mode, fit_config))
    return out


def atom_aware_edges(values, n_bins: int, atom_share: float = 0.05
                     ) -> np.ndarray:
    """Bin edges that give every atom its own bin.

    An atom is a single (rounded) score holding at least ``atom_share`` of
    the mass. Each atom ``a`` gets the bin ``[a, a + 10**-SCORE_DECIMALS)``,
    which under ``searchsorted(side = "right")`` holds exactly the atom; the
    remaining mass is split at equal-mass quantiles of the non-atom values.
    """
    values = np.round(np.asarray(values, dtype = float), SCORE_DECIMALS)
    values = values[np.isfinite(values)]
    unique, counts = np.unique(values, return_counts = True)
    atoms = unique[counts >= atom_share * len(values)]
    rest = values[~np.isin(values, atoms)]
    step = 10.0 ** -SCORE_DECIMALS
    edges = [values.min(), values.max()]
    if len(rest):
        edges += list(np.quantile(rest, np.linspace(0.0, 1.0, n_bins + 1)))
    for atom in atoms:
        edges += [atom, atom + step]
    return np.unique(np.round(edges, SCORE_DECIMALS + 2))


def axis_monotonicity_table(rows: pd.DataFrame, column: str,
                            edges: np.ndarray, fit_config: FitConfig,
                            reps: int = 200,
                            segment: str = "matched",
                            min_gold: int = 5) -> pd.DataFrame:
    """Per-bin existence rate along one score axis, with reversal z-scores.

    The standing per-round monotonicity check. Per bin: the difference
    estimator ``de`` (with the working-model index -- the pool for the
    matched segment, the native score otherwise), the gold-only Hajek
    ``ht``, and two-phase bootstrap standard errors. ``drop_z`` is the fall
    from this bin to the next divided by the bootstrap SE of that
    difference; only positive drops are reversals, and pairs involving a bin
    with fewer than ``min_gold`` gold rows get no z. The formal version is
    Oliva-Aviles, Meyer & Opsomer's (2019) CIC, which has little power with
    many small cells.
    """
    rows = round_scores(rows.reset_index(drop = True))
    classes = merge_thin_cells(
        refined_class(rows, refine = fit_config.refine_by_confidence),
        rows["gold"].to_numpy(dtype = bool), fit_config.min_cell_gold,
    ).reset_index(drop = True)
    bins = _bin_index(rows[column], edges)
    shape = (len(edges) - 1, 1)
    zeros = np.zeros(len(rows), dtype = int)

    def estimate(frame, frame_classes, frame_bins):
        inclusion = inclusion_by_class(
            frame_classes, frame["gold"].to_numpy(dtype = bool)
        )
        weights = frame_classes.astype(str).map(
            lambda c: inclusion.get(c, {}).get("weight", 0.0)
        ).to_numpy(dtype = float)
        if segment in POOLED_SEGMENTS:
            params = fit_pool(frame, weights,
                              min_coef = fit_config.pool_min_coef)
            frame = frame.assign(
                score = segment_scores(frame, segment, params, "pool")
            )
        else:
            frame = frame.assign(score = segment_scores(frame, segment))
        de, counts = cell_difference_estimator(
            frame, frame_classes, inclusion, frame_bins, zeros[:len(frame)],
            shape,
        )
        gold = frame["gold"].to_numpy(dtype = bool)
        y = np.nan_to_num(frame["y"].to_numpy(dtype = float), nan = 0.0)
        w = weights * gold
        mass = _cell_sums(frame_bins, zeros[:len(frame)], w, shape)
        with np.errstate(invalid = "ignore", divide = "ignore"):
            ht = _cell_sums(frame_bins, zeros[:len(frame)], w * y, shape) / mass
        return de[:, 0], ht[:, 0], counts[:, 0]

    de, ht, counts = estimate(rows, classes, bins)
    rng = np.random.default_rng(fit_config.rng_seed + 1300)
    groups, gold_groups, nongold_groups = _verdict_groups(rows)
    boot_de = np.full((reps, len(de)), np.nan)
    boot_ht = np.full((reps, len(de)), np.nan)
    for rep in range(reps):
        take = _resample_two_phase(rows, rng, groups, gold_groups,
                                   nongold_groups)
        boot_de[rep], boot_ht[rep], _ = estimate(
            rows.iloc[take].reset_index(drop = True),
            classes.iloc[take].reset_index(drop = True), bins[take],
        )
    gold = rows["gold"].to_numpy(dtype = bool)
    n_gold = np.bincount(bins[gold], minlength = len(de))
    drops = boot_de[:, :-1] - boot_de[:, 1:]
    drop_se = np.nanstd(drops, axis = 0)
    point_drop = de[:-1] - de[1:]
    with np.errstate(invalid = "ignore", divide = "ignore"):
        drop_z = point_drop / drop_se
    # A bin with a handful of gold has a degenerate bootstrap SE (a single
    # censused row gives SE 0 and an infinite z); report no z for it.
    thin = np.minimum(n_gold[:-1], n_gold[1:]) < min_gold
    drop_z = np.append(np.where(thin, np.nan, drop_z), np.nan)
    return pd.DataFrame({
        "bin": np.arange(len(de)),
        "lo": edges[:-1],
        "hi": edges[1:],
        "n_phase1": counts.astype(int),
        "n_gold": n_gold,
        "de": de,
        "de_se": np.nanstd(boot_de, axis = 0),
        "ht": ht,
        "ht_se": np.nanstd(boot_ht, axis = 0),
        "drop_to_next": np.append(point_drop, np.nan),
        "drop_z": drop_z,
    })


def _jsonable(value):
    """numpy -> plain Python, recursively, for the metadata JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def effective_parameters(result: dict) -> dict:
    """Realized parameter count of a fitted matched-segment map."""
    mode = result.get("index_mode")
    if mode == "surface":
        return {"cells": int(np.prod(result["surface"]["shape"])),
                "distinct_levels": int(len(np.unique(
                    np.round(result["surface"]["theta"], 10))))}
    params = result.get("index") or {}
    form = index_form(params) if params else mode
    if mode == "average" or not params:
        return {"index_params": 0}
    if form == "additive":
        return {"index_params": 1 + params["n_blocks_osm"]
                + params["n_blocks_overture"] - 2,
                "n_blocks_osm": params["n_blocks_osm"],
                "n_blocks_overture": params["n_blocks_overture"]}
    if form == "interaction":
        return {"index_params": 4}
    return {"index_params": 3}


def curve_metadata(segment: str, result: dict, handoff_metadata: dict,
                   fit_config: FitConfig) -> dict:
    """Provenance + diagnostics pinned beside each fitted curve."""
    surface = result.get("surface")
    out = {
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
        # Pool coefficients for a pool-mode two-source segment; ``null``
        # elsewhere. Kept for deploy code that predates ``index``.
        "pool": result.get("pool"),
        # The fitted index of any form (``form`` key); ``null`` for average,
        # surface and single-source segments. The deploy step reads this.
        "index": result.get("index"),
        "index_mode": result.get("index_mode", "native"),
        "score_definition": _score_definition(
            segment, result.get("index_mode", "native")
        ),
        # Scores are rounded to this many decimals before indexing or binning;
        # the deploy step must do the same. Absent on pre-2026-09 curves.
        "score_decimals": SCORE_DECIMALS,
        "effective_sample_size": result["kish_ess"],
        "band_width_median": result["band_width_median"],
        "band_width_grid_median": result.get("band_width_grid_median"),
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
            "pool_min_coef": fit_config.pool_min_coef,
            "surface_osm_bins": fit_config.surface_osm_bins,
            "surface_ov_bins": fit_config.surface_ov_bins,
            "surface_max_sweeps": fit_config.surface_max_sweeps,
            "surface_tol": fit_config.surface_tol,
            "wald_mixture_draws": fit_config.wald_mixture_draws,
            "surface_band_method": fit_config.surface_band_method,
            "band_aggregation": fit_config.band_aggregation,
        },
    }
    if segment in POOLED_SEGMENTS:
        out["effective_parameters"] = effective_parameters(result)
        bound = (result.get("index") or result.get("working_index") or {})
        out["bound_active"] = (bound.get("bound_active")
                               or bound.get("constraints_active") or [])
    if surface is not None:
        out["working_index"] = result.get("working_index")
        out["surface"] = {
            "osm_edges": surface["edges"]["osm"],
            "overture_edges": surface["edges"]["overture"],
            "shape": list(surface["shape"]),
            "band_method": surface["band_method"],
            "phase1_counts": surface["phase1_counts"],
            "gold_counts": surface["gold_counts"],
            "gold_ess": surface["gold_ess"],
            "population_counts": surface["population_counts"],
            "unconstrained": surface["unconstrained"],
            "reference_surface": surface["reference_surface"],
            "wald_se": surface["bands"]["wald"]["se"],
            "wald_n_patterns": surface["bands"]["wald"]["n_patterns"],
            "n_bootstrap_dropped": surface["n_bootstrap_dropped"],
            "corner_cell_flag": surface["corner_cell_flag"],
        }
    return _jsonable(out)


def write_curve(out_dir, segment: str, lookup: pd.DataFrame,
                metadata: dict) -> None:
    """Write one segment's lookup parquet and metadata JSON."""
    out_dir.mkdir(parents = True, exist_ok = True)
    lookup.to_parquet(out_dir / f"{segment}_curve.parquet", index = False)
    with open(out_dir / f"{segment}_metadata.json", "w",
              encoding = "utf-8") as handle:
        json.dump(metadata, handle, indent = 2, sort_keys = True, default = str)
