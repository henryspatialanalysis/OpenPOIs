#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""Bayesian monotone-spline calibration of existence confidence (Phase 1).

**Not on the production path.** This is the prototype described in
``.claude/plans/bayesian-monotone-calibration.md`` (the "design doc"; equation
numbers M1-M18 below refer to it). It is validated against the production
interaction model by ``scripts/conflation/cv_bayes_calibration.py`` and is not
read by any deploy code.

The model calibrates P(exists and open) for each detection segment with a
monotone, range-bounded spline of the source score(s)::

    y_i ~ Bernoulli(expit(F_g(s_i)))                                  (M1-M4)
    F_overture = f1(s_overture), F_osm = f2(s_osm),
    F_matched  = f3(s_osm, s_overture)

- f1 and f2 are quadratic B-splines whose coefficients are nondecreasing, which
  for degree <= 2 is exactly equivalent to a nondecreasing curve. f3 is a
  tensor-product quadratic B-spline whose coefficient matrix is nondecreasing
  along both axes.
- Coefficients are squashed into (L, U) = (logit 1e-3, logit(1 - 1e-3)), so
  every curve is bounded by construction (partition of unity; M8, M10).
- Monotone coefficients are built from exponentiated log-slopes (M7). In 2-D the
  build is a "max recursion" (M9) that covers the whole doubly-monotone cone
  and so imposes no sign on the interaction. The Kronecker cumulative-sum form
  of Pya & Wood (2015) forces a supermodular (complementary) interaction.
- The log-slopes carry an intrinsic first-order GMRF smoothing prior: a path
  graph in 1-D, the four-neighbour grid in 2-D, as in the Bayesian P-splines of
  Lang & Brezger (2004). It is written non-centred through the eigenbasis of
  the structure matrix (M11).

Three data layers ("arms") are implemented. Arm C is the preferred model (design
doc decisions 12 and 17):

``C`` (preferred)
    The simple silver layer (M15b, M15c'). Every phase-1 row enters once. Gold
    rows use the human label. Non-gold rows use the LLM's definitive verdict as a
    fractional label: q log p + (1 - q) log(1 - p), where q = P(exists |
    segment, verdict) is the design-weighted gold concordance rate
    (``silver_label_rates``), passed in as data (``label_noise = "fixed"``). The
    estimated-noise variants ``symmetric`` (one agreement parameter, as first
    run), ``asymmetric`` (Se, Sp) and ``none`` (labels exact) are kept as
    sensitivities. Under arm C's likelihood, estimated noise is identified only
    through the selection-induced gap between gold and non-gold rows.
``A`` (rejected in Phase 1)
    The joint measurement model. The LLM verdict v is modelled given the truth
    (the forward direction of Dawid & Skene, 1979), with class probabilities
    that may depend on score (the differential slope, M12) and are partially
    pooled across segments (M13; centred, sum-to-zero class logits). Gold rows
    contribute p(y, v | s) (M14); non-gold rows contribute p(v | s),
    marginalising the unobserved y (M15). Because phase-2 (gold) selection
    depends only on v, which the model conditions on, the selection is
    ignorable for likelihood and Bayesian inference (Rubin, 1976; Begg &
    Greenes, 1983). It ran biased low against the design-weighted gold rate:
    its single-slope verdict layer cannot follow how the verdict mix varies
    with score, and the curve absorbs the misfit.
``B`` (comparator)
    A gold-only, design-weighted pseudo-likelihood (Savitsky & Toth, 2016),
    with relative weights normalised to the segment's gold count (M17). Used as
    a point-prediction comparator only: its intervals are not design-calibrated
    without the Williams & Savitsky (2021) adjustment.

There is no unit-level nugget. With one binary outcome per unit it is not
identified (McCullagh & Nelder, 1989; Zeger, Liang & Albert, 1988).

References
----------
Begg, C. B., & Greenes, R. A. (1983). Assessment of diagnostic tests when
disease verification is subject to selection bias. Biometrics, 39(1), 207-215.

Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer
error-rates using the EM algorithm. Applied Statistics, 28(1), 20-28.

Lang, S., & Brezger, A. (2004). Bayesian P-splines. Journal of Computational
and Graphical Statistics, 13(1), 183-212.

McCullagh, P., & Nelder, J. A. (1989). Generalized linear models (2nd ed.).
Chapman & Hall.

Pya, N., & Wood, S. N. (2015). Shape constrained additive models. Statistics
and Computing, 25(3), 543-559.

Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.

Savitsky, T. D., & Toth, D. (2016). Bayesian estimation under informative
sampling. Electronic Journal of Statistics, 10(1), 1677-1708.

Williams, M. R., & Savitsky, T. D. (2021). Uncertainty estimation for
pseudo-Bayesian inference under complex sampling. International Statistical
Review, 89(1), 72-107.

Zeger, S. L., Liang, K.-Y., & Albert, P. S. (1988). Models for longitudinal
data: A generalized estimating equation approach. Biometrics, 44(4),
1049-1060.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.flatten_util import ravel_pytree
from jax.scipy.stats import norm as jnorm
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from openpois.conflation import calibration_fit as cf

# Curve range: logit(1e-3) to logit(1 - 1e-3) (design doc, decision 2).
L_BOUND = float(np.log(1e-3 / (1.0 - 1e-3)))
U_BOUND = -L_BOUND
# Segment order used for every per-segment parameter vector.
SEGMENT_ORDER = ("overture", "osm", "matched")
ONE_D_SEGMENTS = ("overture", "osm")
# The 1-D curve's score column, and the measurement layer's score covariate r.
SCORE_COLUMN = {"overture": "overture_score", "osm": "osm_score"}
R_COLUMN = {"overture": "overture_score", "osm": "osm_score",
            "matched": "raw_score"}
VERDICTS = ("exists", "gone", "unverifiable")
CONFIDENCES = ("high", "medium", "low")
# Reference class first: exists:high has psi = beta = 0 (M12).
REFINED_CLASSES = tuple(f"{v}:{c}" for v in VERDICTS for c in CONFIDENCES)
# Thin refined cells merged (execution log, decision 11): exists:low into
# exists:medium, gone:low into gone:medium, unverifiable:high into
# unverifiable:medium. Each merged-away cell has at most ~20 rows over all three
# segments, and its y = 0 / y = 1 split is almost unidentified.
MERGED_CLASSES = ("exists:high", "exists:medium", "gone:high", "gone:medium",
                  "unverifiable:medium", "unverifiable:low")
MERGE_MAP = {"exists:low": "exists:medium", "gone:low": "gone:medium",
             "unverifiable:high": "unverifiable:medium"}
CLASS_SCHEMES = ("refined9", "merged6", "verdict3")
KNOT_PROBS = np.linspace(0.1, 0.9, 9)
ARMS = ("A", "B", "C")
# NUTS tree-depth ceiling (BlackJAX default max_num_doublings).
MAX_DOUBLINGS = 10
# Log-slopes are capped before exponentiation. e^25 is ~7e10 per unit score,
# far outside the prior (mu ~ N(0.5, 1), tau ~ Half-N(0.5)) and far past the
# point where the squash saturates, so the cap never binds in practice. It
# keeps fp32 finite: without it an overflowed step gives inf - inf = NaN once
# the level is anchored mid-curve.
GAMMA_CAP = 25.0


@dataclass(frozen = True)
class PriorConfig:
    """Priors of the design doc's table 3.7 (all provisional)."""

    alpha_sd: float = 1.5
    mu_mean: float = 0.5
    mu_sd: float = 1.0
    tau_scale: float = 0.5
    # Optional separate half-normal scale for the matched surface's tau (S6).
    tau_scale_matched: float | None = None
    mu_psi_sd: float = 3.0
    mu_beta_sd: float = 1.0
    omega_psi_scale: float = 1.0
    omega_beta_scale: float = 0.5
    # Arm C: beta_label ~ Beta(a, b), the concordance slice (63 of 64 agree).
    beta_label_a: float = 63.0
    beta_label_b: float = 1.0
    # Arm C with asymmetric noise (S2C): Se = P(silver says exists | exists),
    # Sp = P(silver says gone | gone), Beta priors centred on the handoff's
    # design-weighted Se 0.998 and Sp 0.913, each with the concordance slice's
    # concentration (64).
    se_a: float = 0.998 * 64
    se_b: float = 0.002 * 64
    sp_a: float = 0.913 * 64
    sp_b: float = 0.087 * 64


@dataclass(frozen = True)
class ModelSpec:
    """Structural choices; the defaults are the approved main model."""

    arm: str = "A"
    # Score slope in the measurement layer (M12); False is sensitivity S2.
    differential: bool = True
    # 9 refined verdict x confidence classes; False (3 verdicts) is S3.
    refined_classes: bool = True
    # "refined9", "merged6" or "verdict3"; overrides refined_classes when set.
    class_scheme: str | None = None
    # Quadratic B-splines; 3 is sensitivity S4.
    degree: int = 2
    # Decile knots with this maximum gap (S5 uses 0.1).
    max_gap: float = 0.2
    # If set, use this many equally spaced knot intervals instead (S5).
    equal_knots: int | None = None
    # Random-walk increments with variance proportional to spacing (S7).
    spacing_scaled_rw: bool = False
    # Temperature of the smooth max in (M9c); 0 is the exact max. 0.05 is the
    # default: the exact max left kinks that caused divergences (execution log,
    # decision 9); the forced excess is at most 0.05 * log 2 per cell.
    smooth_max_t: float = 0.05
    # Where alpha pins the level of c~: "center" (the coefficient nearest the
    # segment's median score) or "origin" (the first coefficient, as written in
    # M7/M9a). A pure reparametrisation of the level; "center" decorrelates
    # alpha from the slopes because it sits where the data are.
    anchor: str = "center"
    # Measurement-layer hierarchy (M13). Centred (psi, beta sampled directly
    # around mu with SD omega) mixes far better here than non-centred: each of
    # the 3 segments has ~2,500 rows, so the data pin every psi_g and the
    # non-centred z's fight mu (execution log, decision 7).
    centered_measurement: bool = True
    # Softmax identification for (M12): "sum_to_zero" maps the C - 1 free
    # coefficients through an orthonormal (Helmert) basis of the sum-to-zero
    # subspace; "reference" pins exists:high at 0. A rare reference class (e.g.
    # exists:high given y = 0) makes every logit share one weakly identified
    # offset, a ridge NUTS mixes badly on (execution log, decision 7).
    softmax_basis: str = "sum_to_zero"
    # Partial pooling of the measurement layer across segments (M13). False
    # fits each segment's psi and beta independently under N(0, mu_psi_sd^2) and
    # N(0, mu_beta_sd^2) (execution log, decision 12).
    pool_segments: bool = True
    # Arm C label noise (design doc §3.5b; execution log, decision 21):
    # "fixed" (default) passes the silver-label accuracy in as data: the
    # design-weighted gold rate of existence among LLM-exists and among LLM-gone
    # rows, per segment (``silver_label_rates``); "symmetric" (one estimated
    # beta_label, M15c as first run), "asymmetric" (estimated Se and Sp,
    # sensitivity S2C) and "none" (silver labels exact, S3C-a) are kept as
    # sensitivities. Estimated noise is not identified from arm C's likelihood
    # except through the selection-induced gold/silver gap (decision 15).
    label_noise: str = "fixed"
    priors: PriorConfig = field(default_factory = PriorConfig)

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.degree not in (1, 2, 3):
            raise ValueError("degree must be 1, 2 or 3")
        if self.label_noise not in ("fixed", "symmetric", "asymmetric", "none"):
            raise ValueError(
                "label_noise must be fixed, symmetric, asymmetric or none")
        if self.class_scheme is not None and self.class_scheme not in CLASS_SCHEMES:
            raise ValueError(f"class_scheme must be one of {CLASS_SCHEMES}")

    @property
    def scheme(self) -> str:
        if self.class_scheme is not None:
            return self.class_scheme
        return "refined9" if self.refined_classes else "verdict3"

    @property
    def class_levels(self) -> tuple:
        return {"refined9": REFINED_CLASSES, "merged6": MERGED_CLASSES,
                "verdict3": VERDICTS}[self.scheme]


# ---------------------------------------------------------------------------
# Knots and bases
# ---------------------------------------------------------------------------

def knot_breaks(scores, max_gap: float = 0.2, equal_knots: int | None = None,
                decimals: int = cf.SCORE_DECIMALS) -> np.ndarray:
    """Spline breakpoints on [0, 1] under the approved rule (decision 5).

    The breakpoints are the deciles of ``scores`` (rounded, deduplicated) plus 0
    and 1; any gap wider than ``max_gap`` is split into equal sub-intervals.
    Point masses such as the Overture atoms make several deciles coincide; they
    collapse to one breakpoint, which then sits exactly on the atom.
    ``equal_knots`` replaces the rule with that many equal intervals (S5).
    """
    if equal_knots is not None:
        return np.linspace(0.0, 1.0, int(equal_knots) + 1)
    values = np.round(np.asarray(scores, dtype = float), decimals)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite scores to place knots on")
    deciles = np.unique(np.round(np.quantile(values, KNOT_PROBS), decimals))
    deciles = deciles[(deciles > 0.0) & (deciles < 1.0)]
    anchors = np.unique(np.concatenate([[0.0], deciles, [1.0]]))
    breaks = [anchors[0]]
    for lo, hi in zip(anchors[:-1], anchors[1:]):
        pieces = int(np.ceil((hi - lo) / max_gap - 1e-9))
        breaks.extend(lo + (hi - lo) * np.arange(1, pieces + 1) / pieces)
    return np.asarray(breaks, dtype = float)


def clamped_knot_vector(breaks: np.ndarray, degree: int) -> np.ndarray:
    """Full knot vector with the end breakpoints repeated ``degree`` times."""
    breaks = np.asarray(breaks, dtype = float)
    return np.concatenate([
        np.full(degree, breaks[0]), breaks, np.full(degree, breaks[-1])
    ])


def n_basis(breaks: np.ndarray, degree: int) -> int:
    """Number of B-spline basis functions: intervals + degree."""
    return len(breaks) - 1 + degree


def basis_matrix(scores, breaks: np.ndarray, degree: int) -> np.ndarray:
    """Dense B-spline design matrix, rows = scores, columns = basis functions.

    Scores are rounded to the production precision and clipped to the knot
    span, so a score of exactly 1 evaluates at the right end. Scores must be
    finite: callers filter missing scores first (``curve_draws`` returns NaN
    for them).
    """
    x = np.round(np.asarray(scores, dtype = float), cf.SCORE_DECIMALS)
    if not np.all(np.isfinite(x)):
        raise ValueError("basis_matrix needs finite scores")
    x = np.clip(x, breaks[0], breaks[-1])
    knots = clamped_knot_vector(breaks, degree)
    return BSpline.design_matrix(x, knots, degree).toarray()


def greville(breaks: np.ndarray, degree: int) -> np.ndarray:
    """Greville abscissae: the mean of each basis function's interior knots."""
    knots = clamped_knot_vector(breaks, degree)
    count = n_basis(breaks, degree)
    if degree == 0:
        return 0.5 * (knots[:-1] + knots[1:])[:count]
    return np.array([knots[k + 1:k + degree + 1].mean() for k in range(count)])


# ---------------------------------------------------------------------------
# Smoothing-prior eigenbases (M11)
# ---------------------------------------------------------------------------

def _laplacian(n_nodes: int, edges: list, weights: np.ndarray) -> np.ndarray:
    lap = np.zeros((n_nodes, n_nodes))
    for (a, b), w in zip(edges, weights):
        lap[a, a] += w
        lap[b, b] += w
        lap[a, b] -= w
        lap[b, a] -= w
    return lap


def _null_free_basis(lap: np.ndarray) -> np.ndarray:
    """Columns V_+ Lambda_+^(-1/2) for a connected graph Laplacian.

    The single zero eigenvalue (the constant vector) is dropped: that is the
    sum-to-zero constraint, so the separate mean parameter carries the level.
    """
    values, vectors = np.linalg.eigh(lap)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    if values[0] > 1e-8 * max(1.0, values[-1]) or values[1] < 1e-10:
        raise ValueError("Smoothing graph must be connected with one null mode")
    return vectors[:, 1:] / np.sqrt(values[1:])


def path_basis(locations: np.ndarray, spacing_scaled: bool = False) -> np.ndarray:
    """RW1 basis for a 1-D log-slope field at the given locations.

    With ``spacing_scaled`` each increment's variance is proportional to the
    distance between neighbouring locations (normalised to mean 1), i.e.
    smoothness per unit score rather than per knot step (sensitivity S7).
    """
    n = len(locations)
    edges = [(k, k + 1) for k in range(n - 1)]
    weights = np.ones(n - 1)
    if spacing_scaled:
        gaps = np.diff(np.asarray(locations, dtype = float))
        gaps = np.maximum(gaps, 1e-9)
        weights = 1.0 / (gaps / gaps.mean())
    return _null_free_basis(_laplacian(n, edges, weights))


def active_cells(n_x: int, n_y: int) -> np.ndarray:
    """Row-major (j, k) indices of the 2-D log-slope field; (0, 0) is unused."""
    cells = [(j, k) for j in range(n_x) for k in range(n_y) if (j, k) != (0, 0)]
    return np.asarray(cells, dtype = int)


def grid_basis(x_locations: np.ndarray, y_locations: np.ndarray,
               spacing_scaled: bool = False) -> np.ndarray:
    """Four-neighbour grid-GMRF basis on the active cells of the 2-D field."""
    n_x, n_y = len(x_locations), len(y_locations)
    cells = active_cells(n_x, n_y)
    position = {tuple(c): i for i, c in enumerate(cells)}
    gx = np.maximum(np.diff(np.asarray(x_locations, dtype = float)), 1e-9)
    gy = np.maximum(np.diff(np.asarray(y_locations, dtype = float)), 1e-9)
    gx, gy = gx / gx.mean(), gy / gy.mean()
    edges, weights = [], []
    for (j, k), i in position.items():
        if (j + 1, k) in position:
            edges.append((i, position[(j + 1, k)]))
            weights.append(1.0 / gx[j] if spacing_scaled else 1.0)
        if (j, k + 1) in position:
            edges.append((i, position[(j, k + 1)]))
            weights.append(1.0 / gy[k] if spacing_scaled else 1.0)
    return _null_free_basis(_laplacian(len(cells), edges, np.asarray(weights)))


# ---------------------------------------------------------------------------
# Monotone, bounded coefficients (M7-M10)
# ---------------------------------------------------------------------------

def squash(c_tilde):
    """(M8)/(M10): increasing bijection from the real line onto (L, U)."""
    return L_BOUND + (U_BOUND - L_BOUND) * jax.nn.sigmoid(c_tilde)


def coefficients_1d(alpha, gamma, h, anchor: int = 0):
    """(M7)-(M8): c~_k = c~_{k-1} + h_k exp(gamma_k), squashed.

    ``gamma`` and ``h`` have length K - 1 (one log-slope per basis step).
    ``alpha`` is the value of c~ at coefficient ``anchor`` (0 reproduces M7's
    c~_1 = alpha; the default model anchors at the data centre).
    """
    steps = h * jnp.exp(jnp.minimum(gamma, GAMMA_CAP))
    c0 = jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps)])
    return squash(alpha + c0 - c0[anchor])


def _pairmax(a, b, t: float):
    if t > 0:
        return t * jnp.logaddexp(a / t, b / t)
    return jnp.maximum(a, b)


def coefficients_2d_tilde(alpha, gamma, hx, hy, smooth_max_t: float = 0.0,
                          anchor: tuple = (0, 0)):
    """(M9): the unsquashed doubly-increasing coefficient matrix C~.

    ``gamma`` is (J, K) with gamma[0, 0] unused. The first column and row are
    cumulative sums; every other cell is the larger of its two predecessors
    plus min(hx_j, hy_k) exp(gamma_jk). Every doubly-increasing matrix has a
    unique such representation, so no sign is imposed on the interaction
    (contrast the Kronecker cumulative sum of Pya & Wood, 2015).

    The recursion is shift-equivariant (max(a + c, b + c) = max(a, b) + c, and
    likewise for the smooth max), so it is run from 0 and then shifted so that
    C~[anchor] = alpha. ``anchor = (0, 0)`` is (M9a) as written.
    """
    gamma = jnp.minimum(gamma, GAMMA_CAP)
    first_col = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(hx * jnp.exp(gamma[1:, 0]))]
    )
    first_row = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(hy * jnp.exp(gamma[0, 1:]))]
    )
    excess = jnp.minimum(hx[:, None], hy[None, :]) * jnp.exp(gamma[1:, 1:])

    def row_step(previous_row, inputs):
        left0, excess_row = inputs

        def cell(left, xs):
            up, e = xs
            value = _pairmax(up, left, smooth_max_t) + e
            return value, value

        _, rest = jax.lax.scan(cell, left0, (previous_row[1:], excess_row))
        row = jnp.concatenate([left0[None], rest])
        return row, row

    _, rows = jax.lax.scan(row_step, first_row, (first_col[1:], excess))
    c0 = jnp.concatenate([first_row[None, :], rows], axis = 0)
    return alpha + c0 - c0[anchor[0], anchor[1]]


def coefficients_2d(alpha, gamma, hx, hy, smooth_max_t: float = 0.0,
                    anchor: tuple = (0, 0)):
    """(M9)-(M10): doubly-increasing coefficients squashed into (L, U)."""
    return squash(coefficients_2d_tilde(alpha, gamma, hx, hy, smooth_max_t,
                                        anchor))


def invert_coefficients_2d(c_tilde: np.ndarray, hx: np.ndarray,
                           hy: np.ndarray) -> tuple:
    """Inverse of (M9) for a strictly doubly-increasing C~ (tests only)."""
    c_tilde = np.asarray(c_tilde, dtype = float)
    n_x, n_y = c_tilde.shape
    gamma = np.zeros((n_x, n_y))
    for j in range(n_x):
        for k in range(n_y):
            if j == 0 and k == 0:
                continue
            if k == 0:
                step = (c_tilde[j, 0] - c_tilde[j - 1, 0]) / hx[j - 1]
            elif j == 0:
                step = (c_tilde[0, k] - c_tilde[0, k - 1]) / hy[k - 1]
            else:
                step = (
                    (c_tilde[j, k] - max(c_tilde[j - 1, k], c_tilde[j, k - 1]))
                    / min(hx[j - 1], hy[k - 1])
                )
            gamma[j, k] = np.log(step)
    return float(c_tilde[0, 0]), gamma


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def usable_rows(validation_rows: pd.DataFrame) -> pd.DataFrame:
    """Validation rows the calibration uses, rounded, in segment order.

    Mirrors ``calibration_fit.fit_all_segments``: the missing-confidence
    stratum and unknown verdicts are excluded.
    """
    rows = validation_rows[
        validation_rows["llm_verdict"].isin(cf.VERDICTS)
        & validation_rows["stratum"].isin(cf.SEGMENTS)
        & validation_rows["segment"].isin(SEGMENT_ORDER)
    ].copy()
    rows["_order"] = rows["segment"].map(
        {s: i for i, s in enumerate(SEGMENT_ORDER)}
    )
    rows = rows.sort_values("_order", kind = "stable").drop(columns = "_order")
    return cf.round_scores(rows.reset_index(drop = True))


def production_classes(rows: pd.DataFrame, fit_config: cf.FitConfig) -> pd.Series:
    """Production's refined classes (merged below the gold floor), per segment.

    These define CV strata and design weights, exactly as
    ``calibration_fit.fit_segment`` builds them for each segment.

    Pooled rounds (execution log, decision 23): when ``validation_round`` holds
    more than one round, the classes are built within each round and prefixed
    with the round id ("20260730|exists_high"). Every consumer groups by class,
    so inclusion probabilities, design weights and silver-label rates are then
    computed per round, as each round's phase-2 design requires. A single-round
    table gets exactly the unprefixed production classes.
    """
    rounds = (rows["validation_round"].astype(str) if "validation_round" in rows
              else pd.Series("", index = rows.index))
    pooled = rounds.nunique() > 1
    out = pd.Series(index = rows.index, dtype = object)
    for round_id in rounds.unique():
        in_round = (rounds == round_id).to_numpy()
        for segment in SEGMENT_ORDER:
            mask = in_round & (rows["segment"] == segment).to_numpy()
            if not mask.any():
                continue
            seg = rows[mask]
            classes = cf.merge_thin_cells(
                cf.refined_class(seg, refine = fit_config.refine_by_confidence),
                seg["gold"].to_numpy(dtype = bool),
                fit_config.min_cell_gold,
            ).astype(str)
            out[mask] = (f"{round_id}|" + classes).to_numpy() if pooled \
                else classes.to_numpy()
    return out.astype(str)


def segment_knots(rows: pd.DataFrame, spec: ModelSpec) -> dict:
    """Breakpoints per curve axis, from all phase-1 rows (fixed across folds)."""
    knots = {}
    for segment in ONE_D_SEGMENTS:
        seg = rows[rows["segment"] == segment]
        knots[segment] = knot_breaks(seg[SCORE_COLUMN[segment]], spec.max_gap,
                                     spec.equal_knots)
    matched = rows[rows["segment"] == "matched"]
    knots["matched_x"] = knot_breaks(matched["osm_score"], spec.max_gap,
                                     spec.equal_knots)
    knots["matched_y"] = knot_breaks(matched["overture_score"], spec.max_gap,
                                     spec.equal_knots)
    return knots


def class_codes(rows: pd.DataFrame, spec: ModelSpec) -> np.ndarray:
    """Integer class index per row in ``spec.class_levels`` (reference = 0)."""
    if spec.scheme == "verdict3":
        labels = rows["llm_verdict"].astype(str)
    else:
        labels = (rows["llm_verdict"].astype(str) + ":"
                  + rows["llm_confidence"].astype(str))
        if spec.scheme == "merged6":
            labels = labels.replace(MERGE_MAP)
    lookup = {name: i for i, name in enumerate(spec.class_levels)}
    codes = labels.map(lookup)
    if codes.isna().any():
        missing = sorted(set(labels[codes.isna()]))
        raise ValueError(f"Unknown LLM classes: {missing}")
    return codes.to_numpy(dtype = int)


@dataclass
class PreparedData:
    """Everything the log density needs, as numpy arrays per segment."""

    spec: ModelSpec
    knots: dict
    segments: dict
    geometry: dict
    # Arm C "fixed" noise: the silver-label rates used (``silver_label_rates``).
    silver_rates: dict = None

    def to_jax(self) -> dict:
        return jax.tree_util.tree_map(jnp.asarray, self.segments)


def _nearest(xi: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(xi) - value)))


def _geometry(knots: dict, spec: ModelSpec, medians: dict) -> dict:
    """Basis spacings, smoothing bases and level anchors per curve.

    ``medians`` holds each axis's median phase-1 score; with
    ``spec.anchor = "center"`` alpha pins the coefficient nearest it.
    """
    geometry = {}
    center = spec.anchor == "center"
    for segment in ONE_D_SEGMENTS:
        xi = greville(knots[segment], spec.degree)
        geometry[segment] = {
            "h": np.diff(xi),
            "field_basis": path_basis(xi[1:], spec.spacing_scaled_rw),
            "n_basis": len(xi),
            "anchor": _nearest(xi, medians[segment]) if center else 0,
        }
    xi_x = greville(knots["matched_x"], spec.degree)
    xi_y = greville(knots["matched_y"], spec.degree)
    geometry["matched"] = {
        "hx": np.diff(xi_x),
        "hy": np.diff(xi_y),
        "field_basis": grid_basis(xi_x, xi_y, spec.spacing_scaled_rw),
        "cells": active_cells(len(xi_x), len(xi_y)),
        "shape": (len(xi_x), len(xi_y)),
        "anchor": ((_nearest(xi_x, medians["matched_x"]),
                    _nearest(xi_y, medians["matched_y"])) if center else (0, 0)),
    }
    return geometry


def silver_label_rates(frames, fit_config: cf.FitConfig = None,
                       gold_masks = None) -> dict:
    """P(exists | segment, LLM verdict) for silver labels, from gold (M15c').

    For each segment and definitive verdict (exists, gone), the design-weighted
    (Hajek) share of gold rows that truly exist. The weights are 1 / pi over the
    production refined classes, computed per validation round, so the rate is
    design-consistent even though phase 2 sampled verdict classes at different
    rates. The rate is then Jeffreys-smoothed on its Kish effective sample size,
    q = (r n_eff + 0.5) / (n_eff + 1), so a class with no observed errors (e.g.
    gone verdicts wrong 0/148) still gets a small positive error rate.

    ``frames`` is one validation table or a list of them (one per round); rows
    from every round are pooled, which is how new rounds' gold (e.g. the
    October 2026 checks) fold into the rates. ``gold_masks`` optionally
    restricts each frame's gold (cross-validation passes the training gold
    only, so held-out labels never inform the rates).

    Returns ``{segment: {"exists": q_e, "gone": q_g, "n_exists": n,
    "n_gone": n, "ess_exists": .., "ess_gone": ..}}``.
    """
    fit_config = fit_config or cf.FitConfig()
    if isinstance(frames, pd.DataFrame):
        frames = [frames]
    if gold_masks is None:
        gold_masks = [None] * len(frames)
    pieces = []
    for frame, gold_mask in zip(frames, gold_masks):
        frame = frame.reset_index(drop = True)
        gold = (frame["gold"].to_numpy(dtype = bool) if gold_mask is None
                else np.asarray(gold_mask, dtype = bool))
        classes = production_classes(frame, fit_config)
        weights = np.zeros(len(frame))
        for segment in SEGMENT_ORDER:
            mask = (frame["segment"] == segment).to_numpy()
            inclusion = cf.inclusion_by_class(
                pd.Series(classes[mask]).reset_index(drop = True), gold[mask])
            weights[mask] = np.array([
                inclusion.get(c, {}).get("weight", 0.0) for c in classes[mask]
            ]) * gold[mask]
        pieces.append(pd.DataFrame({
            "segment": frame["segment"].to_numpy(),
            "verdict": frame["llm_verdict"].astype(str).to_numpy(),
            "y": np.where(gold, frame["y"].to_numpy(dtype = float), np.nan),
            "w": weights,
        }))
    pooled = pd.concat(pieces, ignore_index = True)
    out = {}
    for segment in SEGMENT_ORDER:
        entry = {}
        for verdict in ("exists", "gone"):
            sel = ((pooled["segment"] == segment) & (pooled["verdict"] == verdict)
                   & (pooled["w"] > 0)).to_numpy()
            w = pooled.loc[sel, "w"].to_numpy()
            y = pooled.loc[sel, "y"].to_numpy()
            if len(w) == 0:
                raise ValueError(f"No gold {verdict} verdicts in {segment}")
            rate = float(np.sum(w * y) / np.sum(w))
            ess = float(np.sum(w) ** 2 / np.sum(w ** 2))
            entry[verdict] = (rate * ess + 0.5) / (ess + 1.0)
            entry[f"n_{verdict}"] = int(len(w))
            entry[f"ess_{verdict}"] = ess
            entry[f"raw_{verdict}"] = rate
        out[segment] = entry
    return out


def prepare_data(rows: pd.DataFrame, spec: ModelSpec, knots: dict = None,
                 held_out: np.ndarray = None,
                 fit_config: cf.FitConfig = None,
                 silver_rates: dict = None) -> PreparedData:
    """Build per-segment arrays for one fit.

    ``held_out`` marks gold rows held out of this fit: they stay phase-1 rows
    with their LLM verdict but lose y and gold status, exactly as the
    production cross-fit demotes them. In arm B they drop out (gold only); in
    arm C a held-out row with an unverifiable verdict has no label and drops
    out (design doc §3.5b).

    ``silver_rates`` (arm C, ``label_noise = "fixed"``) overrides the silver-label
    rates; by default they come from this fit's own training gold
    (``silver_label_rates``). Pass pooled multi-round rates here to fold other
    rounds' gold in.
    """
    fit_config = fit_config or cf.FitConfig()
    rows = rows.reset_index(drop = True)
    knots = knots or segment_knots(rows, spec)
    held_out = (np.zeros(len(rows), dtype = bool) if held_out is None
                else np.asarray(held_out, dtype = bool))
    gold = rows["gold"].to_numpy(dtype = bool) & ~held_out
    y_raw = rows["y"].to_numpy(dtype = float)
    if not np.all(np.isfinite(y_raw[gold])):
        raise ValueError("Gold rows must carry a finite y")
    y = np.where(gold, y_raw, 0.0)
    codes = class_codes(rows, spec)
    verdict = rows["llm_verdict"].astype(str).to_numpy()
    prod_classes = production_classes(rows, fit_config)
    if spec.arm == "C" and spec.label_noise == "fixed" and silver_rates is None:
        silver_rates = silver_label_rates(rows, fit_config, gold_masks = [gold])

    segments = {}
    for segment in SEGMENT_ORDER:
        mask = (rows["segment"] == segment).to_numpy()
        seg = rows[mask]
        r = seg[R_COLUMN[segment]].to_numpy(dtype = float)
        own = ([SCORE_COLUMN[segment]] if segment in ONE_D_SEGMENTS
               else ["osm_score", "overture_score"])
        for column in own + [R_COLUMN[segment]]:
            if not np.all(np.isfinite(seg[column].to_numpy(dtype = float))):
                raise ValueError(f"{segment} rows need a finite {column}")
        entry = {
            "gold": gold[mask].astype(float),
            "y": y[mask],
            "cls": codes[mask],
            "r": r - r.mean(),
        }
        if segment in ONE_D_SEGMENTS:
            entry["basis"] = basis_matrix(seg[SCORE_COLUMN[segment]],
                                          knots[segment], spec.degree)
        else:
            entry["basis_x"] = basis_matrix(seg["osm_score"], knots["matched_x"],
                                            spec.degree)
            entry["basis_y"] = basis_matrix(seg["overture_score"],
                                            knots["matched_y"], spec.degree)
        # Arm B: relative design weights on training gold (M17).
        inclusion = cf.inclusion_by_class(
            pd.Series(prod_classes[mask]).reset_index(drop = True), gold[mask]
        )
        w = np.array([
            inclusion.get(c, {}).get("weight", 0.0) for c in prod_classes[mask]
        ]) * gold[mask]
        entry["weight_b"] = (w * gold[mask].sum() / w.sum()) if w.sum() > 0 else w
        # Arm C: label 1 = exists, 0 = gone, -1 = no silver label.
        silver = np.where(verdict[mask] == "exists", 1,
                          np.where(verdict[mask] == "gone", 0, -1))
        entry["silver_label"] = np.where(gold[mask], -1, silver)
        # Fixed noise: P(exists | segment, silver label) per row (0 if no label).
        if spec.arm == "C" and spec.label_noise == "fixed":
            rates = silver_rates[segment]
            entry["silver_q"] = np.where(
                entry["silver_label"] == 1, rates["exists"],
                np.where(entry["silver_label"] == 0, rates["gone"], 0.0))
        if spec.arm == "C":
            # Only held-out rows may lack a label: a non-gold unverifiable row
            # that is not held out would be dropped on its verdict, which is
            # selection on v (design doc §3.5b needs the unverifiable census).
            unlabelled = (entry["silver_label"] == -1) & ~gold[mask] & ~held_out[mask]
            if unlabelled.any():
                raise ValueError(
                    f"Arm C: {int(unlabelled.sum())} non-gold {segment} rows "
                    f"have no silver label (unverifiable, not censused)"
                )
        segments[segment] = entry
    medians = {s: float(np.median(rows.loc[rows["segment"] == s,
                                           SCORE_COLUMN[s]]))
               for s in ONE_D_SEGMENTS}
    matched = rows[rows["segment"] == "matched"]
    medians["matched_x"] = float(np.median(matched["osm_score"]))
    medians["matched_y"] = float(np.median(matched["overture_score"]))
    return PreparedData(spec = spec, knots = knots, segments = segments,
                        geometry = _geometry(knots, spec, medians),
                        silver_rates = silver_rates)


# ---------------------------------------------------------------------------
# Parameters, curves and log density
# ---------------------------------------------------------------------------

def parameter_template(prepared: PreparedData) -> dict:
    """A zero-valued parameter pytree with the model's shapes."""
    spec, geometry = prepared.spec, prepared.geometry
    n_classes = len(spec.class_levels)
    params = {
        "alpha": jnp.zeros(3),
        "mu": jnp.zeros(3),
        "log_tau": jnp.zeros(3),
        "z_overture": jnp.zeros(geometry["overture"]["field_basis"].shape[1]),
        "z_osm": jnp.zeros(geometry["osm"]["field_basis"].shape[1]),
        "z_matched": jnp.zeros(geometry["matched"]["field_basis"].shape[1]),
    }
    if spec.arm == "A" and not spec.pool_segments:
        params["psi"] = jnp.zeros((3, 2, n_classes - 1))
        if spec.differential:
            params["beta"] = jnp.zeros((3, 2, n_classes - 1))
    elif spec.arm == "A":
        leaf = "psi" if spec.centered_measurement else "z_psi"
        params.update({
            "mu_psi": jnp.zeros((2, n_classes - 1)),
            "log_omega_psi": jnp.zeros(()),
            leaf: jnp.zeros((3, 2, n_classes - 1)),
        })
        if spec.differential:
            leaf = "beta" if spec.centered_measurement else "z_beta"
            params.update({
                "mu_beta": jnp.zeros((2, n_classes - 1)),
                "log_omega_beta": jnp.zeros(()),
                leaf: jnp.zeros((3, 2, n_classes - 1)),
            })
    if spec.arm == "C" and spec.label_noise == "symmetric":
        params["logit_beta_label"] = jnp.zeros(())
    if spec.arm == "C" and spec.label_noise == "asymmetric":
        params["logit_se"] = jnp.zeros(())
        params["logit_sp"] = jnp.zeros(())
    return params


def curve_coefficients(params: dict, geometry: dict, spec: ModelSpec) -> dict:
    """Squashed spline coefficients for all three curves (M7-M11)."""
    tau = jnp.exp(params["log_tau"])
    out = {}
    for g, segment in enumerate(SEGMENT_ORDER):
        basis = jnp.asarray(geometry[segment]["field_basis"])
        field_values = params["mu"][g] + tau[g] * (basis @ params[f"z_{segment}"])
        if segment in ONE_D_SEGMENTS:
            out[segment] = coefficients_1d(params["alpha"][g], field_values,
                                           jnp.asarray(geometry[segment]["h"]),
                                           geometry[segment]["anchor"])
        else:
            n_x, n_y = geometry["matched"]["shape"]
            cells = geometry["matched"]["cells"]
            gamma = jnp.zeros((n_x, n_y)).at[cells[:, 0], cells[:, 1]].set(
                field_values
            )
            out[segment] = coefficients_2d(
                params["alpha"][g], gamma, jnp.asarray(geometry["matched"]["hx"]),
                jnp.asarray(geometry["matched"]["hy"]), spec.smooth_max_t,
                geometry["matched"]["anchor"],
            )
    return out


def segment_logits(coefficients: dict, data: dict) -> dict:
    """F_g(s_i) for every row, per segment (M3, M5, M6)."""
    out = {}
    for segment in SEGMENT_ORDER:
        seg = data[segment]
        if segment in ONE_D_SEGMENTS:
            out[segment] = seg["basis"] @ coefficients[segment]
        else:
            out[segment] = jnp.sum(
                (seg["basis_x"] @ coefficients[segment]) * seg["basis_y"], axis = 1
            )
    return out


def _half_normal_log_scale(log_value, scale):
    """log p(log x) for x ~ Half-N(0, scale^2), including the log-Jacobian."""
    value = jnp.exp(log_value)
    return jnp.log(2.0) + jnorm.logpdf(value, 0.0, scale) + log_value


def log_prior(params: dict, spec: ModelSpec) -> jnp.ndarray:
    """Sum of the priors in design doc table 3.7 (plus Jacobians)."""
    pr = spec.priors
    lp = jnp.sum(jnorm.logpdf(params["alpha"], 0.0, pr.alpha_sd))
    lp += jnp.sum(jnorm.logpdf(params["mu"], pr.mu_mean, pr.mu_sd))
    tau_scales = jnp.array([
        pr.tau_scale, pr.tau_scale,
        pr.tau_scale_matched if pr.tau_scale_matched is not None else pr.tau_scale,
    ])
    lp += jnp.sum(_half_normal_log_scale(params["log_tau"], tau_scales))
    for segment in SEGMENT_ORDER:
        lp += jnp.sum(jnorm.logpdf(params[f"z_{segment}"]))
    if spec.arm == "A" and not spec.pool_segments:
        lp += jnp.sum(jnorm.logpdf(params["psi"], 0.0, pr.mu_psi_sd))
        if spec.differential:
            lp += jnp.sum(jnorm.logpdf(params["beta"], 0.0, pr.mu_beta_sd))
    elif spec.arm == "A":
        lp += jnp.sum(jnorm.logpdf(params["mu_psi"], 0.0, pr.mu_psi_sd))
        lp += _half_normal_log_scale(params["log_omega_psi"], pr.omega_psi_scale)
        if spec.centered_measurement:
            lp += jnp.sum(jnorm.logpdf(params["psi"], params["mu_psi"][None],
                                       jnp.exp(params["log_omega_psi"])))
        else:
            lp += jnp.sum(jnorm.logpdf(params["z_psi"]))
        if spec.differential:
            lp += jnp.sum(jnorm.logpdf(params["mu_beta"], 0.0, pr.mu_beta_sd))
            lp += _half_normal_log_scale(params["log_omega_beta"],
                                         pr.omega_beta_scale)
            if spec.centered_measurement:
                lp += jnp.sum(jnorm.logpdf(params["beta"], params["mu_beta"][None],
                                           jnp.exp(params["log_omega_beta"])))
            else:
                lp += jnp.sum(jnorm.logpdf(params["z_beta"]))
    if spec.arm == "C" and spec.label_noise == "symmetric":
        # beta ~ Beta(a, b) on the logit scale: beta^a (1 - beta)^b with the
        # Jacobian beta (1 - beta) folded in.
        x = params["logit_beta_label"]
        lp += (pr.beta_label_a * jax.nn.log_sigmoid(x)
               + pr.beta_label_b * jax.nn.log_sigmoid(-x))
    if spec.arm == "C" and spec.label_noise == "asymmetric":
        for name, a, b in (("logit_se", pr.se_a, pr.se_b),
                           ("logit_sp", pr.sp_a, pr.sp_b)):
            x = params[name]
            lp += a * jax.nn.log_sigmoid(x) + b * jax.nn.log_sigmoid(-x)
    return lp


def measurement_arrays(params: dict, spec: ModelSpec) -> tuple:
    """(psi, beta) of (M13), each (3 segments, 2 truth states, C - 1).

    ``beta`` is None when the differential slope is off.
    """
    if spec.centered_measurement or not spec.pool_segments:
        psi = params["psi"]
        beta = params["beta"] if spec.differential else None
    else:
        psi = (params["mu_psi"][None]
               + jnp.exp(params["log_omega_psi"]) * params["z_psi"])
        beta = ((params["mu_beta"][None]
                 + jnp.exp(params["log_omega_beta"]) * params["z_beta"])
                if spec.differential else None)
    return psi, beta


def helmert_basis(n_classes: int) -> np.ndarray:
    """Orthonormal basis (C x (C - 1)) of the sum-to-zero subspace of R^C."""
    centering = np.eye(n_classes) - 1.0 / n_classes
    q, _ = np.linalg.qr(centering[:, :-1])
    return q


def _full_logits(free, spec: ModelSpec):
    """Map (..., C - 1) free coefficients to (..., C) class logits."""
    n_classes = free.shape[-1] + 1
    if spec.softmax_basis == "sum_to_zero":
        return free @ jnp.asarray(helmert_basis(n_classes)).T
    zeros = jnp.zeros(free.shape[:-1] + (1,))
    return jnp.concatenate([zeros, free], axis = -1)


def class_log_probs(params: dict, spec: ModelSpec, g: int, r):
    """log theta_{g,y,v}(r_i) for every class v and y in {0, 1}: (n, 2, C)."""
    psi_all, beta_all = measurement_arrays(params, spec)
    psi = _full_logits(psi_all[g], spec)                               # (2, C)
    logits = jnp.broadcast_to(psi[None], (r.shape[0],) + psi.shape)  # (n, 2, C)
    if beta_all is not None:
        beta = _full_logits(beta_all[g], spec)
        logits = logits + beta[None] * r[:, None, None]
    return jax.nn.log_softmax(logits, axis = -1)


def measurement_log_probs(params: dict, spec: ModelSpec, g: int, cls, r):
    """log theta_{g,y,v_i}(r_i) for y = 0 and 1, shape (n, 2) (M12-M13)."""
    log_theta = class_log_probs(params, spec, g, r)
    return jnp.take_along_axis(
        log_theta, jnp.broadcast_to(cls[:, None, None], (r.shape[0], 2, 1)),
        axis = 2,
    )[..., 0]


def silver_log_rates(params: dict, spec: ModelSpec) -> tuple:
    """(log Se, log(1 - Se), log Sp, log(1 - Sp)) of the arm C label noise.

    Symmetric noise uses Se = Sp = beta_label; "none" makes silver labels exact
    (log 1 = 0 and log 0 approximated by -1e3 so the logaddexp stays finite).
    """
    if spec.label_noise == "symmetric":
        x = params["logit_beta_label"]
        log_b, log_1mb = jax.nn.log_sigmoid(x), jax.nn.log_sigmoid(-x)
        return log_b, log_1mb, log_b, log_1mb
    if spec.label_noise == "asymmetric":
        se, sp = params["logit_se"], params["logit_sp"]
        return (jax.nn.log_sigmoid(se), jax.nn.log_sigmoid(-se),
                jax.nn.log_sigmoid(sp), jax.nn.log_sigmoid(-sp))
    zero, never = jnp.asarray(0.0), jnp.asarray(-1e3)
    return zero, never, zero, never


def pointwise_log_likelihood(params: dict, data: dict, geometry: dict,
                             spec: ModelSpec) -> dict:
    """Per-row log-likelihood contributions, per segment.

    Rows that do not enter the chosen arm (non-gold rows in arm B, rows with no
    label in arm C) contribute exactly zero.
    """
    coefficients = curve_coefficients(params, geometry, spec)
    logits = segment_logits(coefficients, data)
    out = {}
    for g, segment in enumerate(SEGMENT_ORDER):
        seg, f = data[segment], logits[segment]
        lp1, lp0 = jax.nn.log_sigmoid(f), jax.nn.log_sigmoid(-f)
        y, gold = seg["y"], seg["gold"]
        bernoulli = y * lp1 + (1.0 - y) * lp0
        if spec.arm == "A":
            log_theta = measurement_log_probs(params, spec, g, seg["cls"], seg["r"])
            gold_term = bernoulli + jnp.where(y > 0.5, log_theta[:, 1],
                                              log_theta[:, 0])
            silver_term = jnp.logaddexp(lp1 + log_theta[:, 1],
                                        lp0 + log_theta[:, 0])
            out[segment] = jnp.where(gold > 0.5, gold_term, silver_term)
        elif spec.arm == "B":
            out[segment] = seg["weight_b"] * bernoulli * gold
        elif spec.label_noise == "fixed":
            # (M15c'): a silver row is a fractional label q = P(exists |
            # segment, verdict) from gold: q log p + (1 - q) log(1 - p).
            q = seg["silver_q"]
            silver = jnp.where(seg["silver_label"] >= 0,
                               q * lp1 + (1.0 - q) * lp0, 0.0)
            out[segment] = jnp.where(gold > 0.5, bernoulli, silver)
        else:
            log_se, log_1mse, log_sp, log_1msp = silver_log_rates(params, spec)
            label = seg["silver_label"]
            # P(label | s) = sum_y P(y | s) P(label | y)   (M15c)
            says_exists = jnp.logaddexp(lp1 + log_se, lp0 + log_1msp)
            says_gone = jnp.logaddexp(lp1 + log_1mse, lp0 + log_sp)
            silver = jnp.where(label == 1, says_exists,
                               jnp.where(label == 0, says_gone, 0.0))
            out[segment] = jnp.where(gold > 0.5, bernoulli, silver)
    return out


def make_log_density(prepared: PreparedData):
    """Jitted ``params -> log posterior`` closed over the prepared data."""
    data = prepared.to_jax()
    geometry, spec = prepared.geometry, prepared.spec

    @jax.jit
    def log_density(params):
        pointwise = pointwise_log_likelihood(params, data, geometry, spec)
        total = sum(jnp.sum(v) for v in pointwise.values())
        return total + log_prior(params, spec)

    return log_density


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def initial_params(prepared: PreparedData) -> dict:
    """A sensible, feasible starting point before the MAP search."""
    params = parameter_template(prepared)
    center = prepared.spec.anchor == "center"
    params["alpha"] = jnp.full(3, 0.3 if center else -1.0)
    params["mu"] = jnp.full(3, 1.0)
    params["log_tau"] = jnp.full(3, np.log(0.3))
    spec = prepared.spec
    if spec.arm == "A" and spec.pool_segments:
        params["log_omega_psi"] = jnp.asarray(np.log(0.5))
        if spec.differential:
            params["log_omega_beta"] = jnp.asarray(np.log(0.3))
    if spec.arm == "A":
        # psi from pooled class log-odds against the reference class, by truth.
        counts = np.ones((2, len(spec.class_levels)))
        for seg in prepared.segments.values():
            gold = seg["gold"] > 0.5
            for yv in (0, 1):
                sel = gold & (seg["y"] == yv)
                counts[yv] += np.bincount(seg["cls"][sel],
                                          minlength = counts.shape[1])
        if spec.softmax_basis == "sum_to_zero":
            log_counts = np.log(counts)
            centered = log_counts - log_counts.mean(axis = 1, keepdims = True)
            psi = centered @ helmert_basis(counts.shape[1])
        else:
            psi = np.log(counts[:, 1:]) - np.log(counts[:, :1])
        if spec.pool_segments:
            params["mu_psi"] = jnp.asarray(psi)
        if "psi" in params:
            params["psi"] = jnp.broadcast_to(jnp.asarray(psi), params["psi"].shape)
    if spec.arm == "C" and spec.label_noise == "symmetric":
        params["logit_beta_label"] = jnp.asarray(np.log(0.984 / 0.016))
    if spec.arm == "C" and spec.label_noise == "asymmetric":
        params["logit_se"] = jnp.asarray(np.log(0.99 / 0.01))
        params["logit_sp"] = jnp.asarray(np.log(0.913 / 0.087))
    return params


def find_map(log_density, start: dict, max_iter: int = 2000) -> tuple:
    """L-BFGS-B maximum a posteriori estimate, used to start NUTS."""
    flat, unravel = ravel_pytree(start)
    value_and_grad = jax.jit(jax.value_and_grad(lambda v: -log_density(unravel(v))))

    def objective(v):
        value, grad = value_and_grad(jnp.asarray(v))
        return float(value), np.asarray(grad, dtype = float)

    result = minimize(objective, np.asarray(flat, dtype = float), jac = True,
                      method = "L-BFGS-B", options = {"maxiter": max_iter})
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        print(f"find_map: non-finite result ({result.message}); "
              f"starting NUTS from the initial values", flush = True)
        return start, result
    return unravel(jnp.asarray(result.x)), result


@dataclass
class FitResult:
    """Posterior draws and sampler diagnostics for one fit."""

    prepared: PreparedData
    chain_draws: dict
    info: object
    warmup: dict
    map_params: dict
    map_result: object
    diagnostics: dict

    @property
    def draws(self) -> dict:
        """Draws flattened over chains: leading axis = chains x draws."""
        return jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), self.chain_draws
        )


def fit(prepared: PreparedData, num_warmup: int = 1000, num_samples: int = 1000,
        num_chains: int = 4, seed: int = 0, init_jitter: float = 0.5,
        adaptation_kwargs: dict = None, map_init: bool = True) -> FitResult:
    """MAP search then multi-chain NUTS (``jax_core.nuts_sample_multichain``)."""
    from openpois.models.jax_core import nuts_sample_multichain

    if not jax.config.jax_enable_x64:
        print("calibration_bayes.fit: jax_enable_x64 is off, sampling in fp32; "
              "call jax_core.enable_high_precision() first (design doc §4)",
              flush = True)
    log_density = make_log_density(prepared)
    start = initial_params(prepared)
    map_result = None
    if map_init:
        start, map_result = find_map(log_density, start)
    chain_draws, info, warmup = nuts_sample_multichain(
        log_density = log_density,
        init_position = start,
        num_warmup = num_warmup,
        num_samples = num_samples,
        num_chains = num_chains,
        key = jax.random.PRNGKey(seed),
        init_jitter = init_jitter,
        adaptation_kwargs = adaptation_kwargs,
    )
    result = FitResult(prepared = prepared, chain_draws = chain_draws,
                       info = info, warmup = warmup, map_params = start,
                       map_result = map_result, diagnostics = {})
    result.diagnostics = sampler_diagnostics(chain_draws, info)
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _scalar_series(chain_draws: dict) -> dict:
    """{name[i]: (chains, draws)} for every scalar component."""
    out = {}
    for name, value in chain_draws.items():
        arr = np.asarray(value, dtype = float)
        chains, draws = arr.shape[:2]
        flat = arr.reshape(chains, draws, -1)
        for i in range(flat.shape[2]):
            label = name if flat.shape[2] == 1 else f"{name}[{i}]"
            out[label] = flat[:, :, i]
    return out


def convergence_table(series: dict) -> pd.DataFrame:
    """Rank-normalised split R-hat and bulk/tail ESS (arviz) per scalar."""
    import arviz as az

    rows = []
    for name, x in series.items():
        x = np.asarray(x, dtype = float)
        if np.allclose(x, x.flat[0]):
            rows.append({"parameter": name, "rhat": 1.0, "ess_bulk": np.inf,
                         "ess_tail": np.inf})
            continue
        rows.append({
            "parameter": name,
            "rhat": float(az.rhat(x, method = "rank")),
            "ess_bulk": float(az.ess(x, method = "bulk")),
            "ess_tail": float(az.ess(x, method = "tail", prob = (0.05, 0.95))),
        })
    return pd.DataFrame(rows)


def sampler_diagnostics(chain_draws: dict, info) -> dict:
    """Convergence and NUTS health summary for the acceptance rule (§5.1)."""
    table = convergence_table(_scalar_series(chain_draws))
    divergent = np.asarray(info.is_divergent)
    energy = np.asarray(info.energy, dtype = float)
    bfmi = (np.sum(np.diff(energy, axis = 1) ** 2, axis = 1)
            / np.sum((energy - energy.mean(axis = 1, keepdims = True)) ** 2,
                     axis = 1))
    # Saturation = a full-length trajectory (Stan's treedepth), not merely a
    # 10th expansion attempt, which BlackJAX counts even when it stops early.
    steps = np.asarray(info.num_integration_steps)
    return {
        "table": table,
        "max_rhat": float(table["rhat"].max()),
        "min_ess_bulk": float(table["ess_bulk"].min()),
        "min_ess_tail": float(table["ess_tail"].min()),
        "divergences_per_chain": divergent.sum(axis = 1).tolist(),
        "ebfmi_per_chain": bfmi.tolist(),
        "treedepth_saturated": int((steps >= 2 ** MAX_DOUBLINGS - 1).sum()),
        "mean_accept": float(np.mean(np.asarray(info.acceptance_rate))),
        "mean_steps": float(np.mean(np.asarray(info.num_integration_steps))),
    }


def passes_acceptance(diagnostics: dict, curve_table: pd.DataFrame = None) -> dict:
    """The §5.1 acceptance rule, item by item."""
    checks = {
        "rhat": diagnostics["max_rhat"] <= 1.01,
        "ess_bulk": diagnostics["min_ess_bulk"] >= 400,
        "ess_tail": diagnostics["min_ess_tail"] >= 400,
        "divergences": sum(diagnostics["divergences_per_chain"]) == 0,
        "ebfmi": min(diagnostics["ebfmi_per_chain"]) >= 0.3,
        "treedepth": diagnostics["treedepth_saturated"] == 0,
    }
    if curve_table is not None and len(curve_table):
        checks["curve_rhat"] = float(curve_table["rhat"].max()) <= 1.01
        checks["curve_ess"] = float(
            min(curve_table["ess_bulk"].min(), curve_table["ess_tail"].min())
        ) >= 400
    checks["all"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def curve_draws(draws: dict, prepared: PreparedData, segment: str, osm = None,
                overture = None, chunk: int = 250) -> np.ndarray:
    """Posterior draws of m_g(s) = expit(F_g(s)) at new scores (M18).

    ``draws`` has a leading draw axis. Returns an array (draws, n points);
    points with a missing score are NaN.
    """
    spec, knots, geometry = prepared.spec, prepared.knots, prepared.geometry
    if segment in ONE_D_SEGMENTS:
        scores = np.asarray(osm if segment == "osm" else overture, dtype = float)
        finite = np.isfinite(scores)
        basis = jnp.asarray(basis_matrix(scores[finite], knots[segment],
                                         spec.degree))

        def one(p):
            c = curve_coefficients(p, geometry, spec)[segment]
            return jax.nn.sigmoid(basis @ c)
    else:
        osm = np.asarray(osm, dtype = float)
        overture = np.asarray(overture, dtype = float)
        finite = np.isfinite(osm) & np.isfinite(overture)
        bx = jnp.asarray(basis_matrix(osm[finite], knots["matched_x"],
                                      spec.degree))
        by = jnp.asarray(basis_matrix(overture[finite], knots["matched_y"],
                                      spec.degree))

        def one(p):
            c = curve_coefficients(p, geometry, spec)["matched"]
            return jax.nn.sigmoid(jnp.sum((bx @ c) * by, axis = 1))

    batched = jax.jit(jax.vmap(one))
    n_draws = jax.tree_util.tree_leaves(draws)[0].shape[0]
    parts = []
    for start in range(0, n_draws, chunk):
        sub = jax.tree_util.tree_map(lambda x: x[start:start + chunk], draws)
        parts.append(np.asarray(batched(sub)))
    values = np.concatenate(parts, axis = 0)
    out = np.full((n_draws, len(finite)), np.nan, dtype = values.dtype)
    out[:, finite] = values
    return out


def summarize_draws(values: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Posterior mean and equal-tailed band per column of a (draws, n) array."""
    return pd.DataFrame({
        "mean": values.mean(axis = 0),
        "lower": np.quantile(values, alpha / 2, axis = 0),
        "upper": np.quantile(values, 1 - alpha / 2, axis = 0),
    })


def sample_prior_curve_params(prepared: PreparedData, n_draws: int,
                              rng: np.random.Generator) -> dict:
    """Draws of the curve parameters from their priors (prior predictive).

    Only the parameters that shape the curves are drawn; the measurement-layer
    leaves are zero (they do not enter m_g(s)).
    """
    pr = prepared.spec.priors
    template = parameter_template(prepared)
    out = jax.tree_util.tree_map(
        lambda x: np.zeros((n_draws,) + np.shape(x)), template
    )
    out["alpha"] = rng.normal(0.0, pr.alpha_sd, (n_draws, 3))
    out["mu"] = rng.normal(pr.mu_mean, pr.mu_sd, (n_draws, 3))
    scales = np.array([
        pr.tau_scale, pr.tau_scale,
        pr.tau_scale_matched if pr.tau_scale_matched is not None else pr.tau_scale,
    ])
    out["log_tau"] = np.log(np.abs(rng.normal(0.0, 1.0, (n_draws, 3))) * scales)
    for segment in SEGMENT_ORDER:
        key = f"z_{segment}"
        out[key] = rng.normal(0.0, 1.0, out[key].shape)
    return jax.tree_util.tree_map(jnp.asarray, out)


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------

def assign_gold_folds(classes: pd.Series, gold_mask: np.ndarray, n_folds: int,
                      seed: int) -> np.ndarray:
    """Fold per row (-1 for non-gold), identical to ``cross_fit_predictions``.

    ``calibration_fit.cross_fit_predictions`` draws a permutation of each
    class's gold rows from ``default_rng(rng_seed + 700)``, in the order of
    ``classes.unique()``. ``seed`` here is that ``rng_seed``.
    """
    rng = np.random.default_rng(seed + 700)
    class_names = pd.Series(classes).astype(str).reset_index(drop = True)
    gold_mask = np.asarray(gold_mask, dtype = bool)
    fold_of = np.full(len(class_names), -1)
    for name in class_names.unique():
        in_class = np.flatnonzero((class_names == name).to_numpy() & gold_mask)
        shuffled = rng.permutation(in_class)
        fold_of[shuffled] = np.arange(len(shuffled)) % n_folds
    return fold_of


def assign_segment_folds(rows: pd.DataFrame, classes: pd.Series, n_folds: int,
                         seed: int) -> np.ndarray:
    """Folds for the whole table, one production cross-fit call per segment.

    Production runs ``cross_fit_predictions`` separately per segment, each with
    a fresh ``default_rng(seed + 700)``. Calling ``assign_gold_folds`` once on
    the full table would share one stream and merge same-named classes across
    segments, so this loops over segments. Returns a full-length array aligned
    to ``rows`` (-1 for non-gold rows).
    """
    rows = rows.reset_index(drop = True)
    classes = pd.Series(classes).astype(str).reset_index(drop = True)
    folds = np.full(len(rows), -1)
    for segment in SEGMENT_ORDER:
        mask = (rows["segment"] == segment).to_numpy()
        folds[mask] = assign_gold_folds(
            classes[mask].reset_index(drop = True),
            rows.loc[mask, "gold"].to_numpy(dtype = bool), n_folds, seed,
        )
    return folds


def with_spec(prepared_or_spec, **changes) -> ModelSpec:
    """Convenience: a copy of a spec with fields replaced."""
    spec = getattr(prepared_or_spec, "spec", prepared_or_spec)
    return replace(spec, **changes)
