#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
JAX-based models for OSM POI turnover rate estimation.

Each model class is self-contained: it ingests a raw observations DataFrame
plus a metadata dict, prepares the JAX arrays that ``ModelFitter`` needs, and
exposes ``event_rate_fun`` and ``param_likelihood`` as bound instance methods.

The fitted rate is interpreted as a Poisson event rate per observation; the
change probability is recovered inside ``ModelFitter`` via P = 1 - exp(-rate).
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy import stats


DEFAULT_LOG_LAMBDA_PRIOR_SCALE = 3.0
DEFAULT_VAR_PRIOR = (0.0, 1.0)
# RandomEffectsModel hyperprior defaults (overridable per term via config).
# Main effects (amenity, MSA) use a moderately diffuse hyperprior on
# log_sigma; the amenity x MSA interaction uses a tighter one so deviations
# shrink toward zero unless strongly supported, mirroring the tight
# DEFAULT_LOGIT_DELTA_VAR_PRIOR below.
DEFAULT_VAR_PRIOR_MAIN = (-1.0, 1.0)
DEFAULT_VAR_PRIOR_INTERACTION = (-2.0, 0.5)
# Normal prior on the suburban / rural fixed-effect coefficients (urban = ref).
DEFAULT_URBANICITY_PRIOR = (0.0, 1.0)
# Minimum distinct POIs in an (amenity, MSA) cell to warrant its own
# interaction level; smaller cells fall back to the main effects.
DEFAULT_INTERACTION_MIN_COUNT = 100
# Zero-inflated exponential (ZIE) δ prior on the log-odds scale, per
# turnover-model-methodology.md §1.7. Prior mean −3 → δ ≈ 5 %.
DEFAULT_LOGIT_DELTA_PRIOR = (-3.0, 1.0)
# Tight hyperprior on log_tau (random-effect scale for per-group logit_delta).
# Tau median ≈ exp(-2) ≈ 0.135 on the logit scale.
DEFAULT_LOGIT_DELTA_VAR_PRIOR = (-2.0, 0.5)
# Log-normal prior on the breakpoint age t_B (years since the tag was
# established) for ConstantBreakpointModel: (loc, scale) on log t_B. loc = 0 →
# median t_B = exp(0) = 1 year.
DEFAULT_T_BREAKPOINT_PRIOR = (0.0, 1.0)
# Normal prior on log_shape for WeibullModel (p = exp(log_shape)); loc = 0 →
# median p = 1 (constant hazard), with p < 1 (decreasing) well inside 2σ.
DEFAULT_WEIBULL_LOG_SHAPE_PRIOR = (0.0, 1.0)
# Normal prior on log_theta for LomaxModel (θ = exp(log_theta), the gamma-
# frailty variance / overdispersion knob); loc = 0 → median θ = 1, θ → 0
# recovers the constant model.
DEFAULT_LOMAX_LOG_THETA_PRIOR = (0.0, 1.5)
# Floor applied when taking the log of an empirical rate. Groups with zero
# observed changes would otherwise give log(0) = -inf on init.
_EMPIRICAL_RATE_FLOOR = 1e-8
# Bounds on the empirical-Bayes log_sigma initializer, to guard against
# single-observation groups driving the starting point to extremes.
_LOG_SIGMA_INIT_BOUNDS = (-3.0, 1.0)
# Supported parameterizations for RandomByTypeModel.
_VALID_REPARAMS = ("centered", "non_centered")
_DEFAULT_REPARAM = "non_centered"


def _empirical_rate_from_nonfirst(
    raw_data: pd.DataFrame, dt_col: str,
) -> float:
    """
    Pooled change rate using non-first-interval rows only.

    Per methodology §4.2 Step F: the δ-component changes at t = 0, so including
    first-interval rows in the empirical λ init would let instant-change mass
    inflate the starting point. Falls back to the full frame if no non-first
    rows exist (e.g. tests that don't emit the flag).
    """
    if "is_first_interval" in raw_data.columns:
        mask = ~raw_data["is_first_interval"].astype(bool)
        if mask.sum() > 0:
            sub = raw_data.loc[mask]
            return float(sub["changed"].mean() / sub[dt_col].mean())
    return float(raw_data["changed"].mean() / raw_data[dt_col].mean())


def _zie_pointwise_loglik(
    rate: jnp.ndarray,
    is_first: jnp.ndarray,
    log_1md: jnp.ndarray,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-row ZIE Bernoulli-on-Poisson log-likelihood (methodology §4.2 Step D),
    without the final sum. Shared by the constant and random-effects models so
    the summed likelihood and the pointwise metric stay in lockstep.

    First-interval rows (``is_first = 1``) carry the ``log(1−δ)`` discount;
    ``log_1md`` may be a scalar (constant δ) or a per-row vector (grouped δ).
    """
    log_p_std = jnp.log(-jnp.expm1(-rate))
    log_1mp_std = -rate
    log_p_zie = jnp.log(-jnp.expm1(log_1md - rate))
    log_1mp_zie = log_1md - rate
    log_p = jnp.where(is_first > 0, log_p_zie, log_p_std)
    log_1mp = jnp.where(is_first > 0, log_1mp_zie, log_1mp_std)
    return target * log_p + (1.0 - target) * log_1mp


class ModelFactory(ABC):
    """
    Base class for OSM turnover models.

    Subclasses must implement ``build_model()`` to populate ``starting_params``,
    ``param_ids`` (and ``group_lookup`` for random-effects variants), plus any
    per-observation columns in ``self.data`` beyond ``dt``.
    """

    def __init__(self, dataset: pd.DataFrame, metadata: dict):
        """
        Args:
            dataset: Observations DataFrame (already filtered/prepared by
                ``prepare_data_for_model``).
            metadata: Model configuration. Required keys vary by subclass; all
                subclasses honor ``dt_col`` (default ``"tag_years"``).
        """
        self.raw_data = dataset
        self.metadata = metadata or {}
        self.data: dict[str, jnp.ndarray] = {}
        self.target: jnp.ndarray | None = None
        self.starting_params: dict[str, jnp.ndarray] = {}
        self.param_ids: pd.DataFrame | None = None
        self.group_lookup: pd.DataFrame | None = None
        # Optional sufficient-statistics log-likelihood override. If set by
        # ``build_model``, the fitter bypasses the per-observation dense path
        # and calls ``log_likelihood_fun(params, data, target)`` directly.
        self.log_likelihood_fun: callable | None = None
        self.validate_inputs()
        self.build_model()
        self.assign_targets()

    def validate_inputs(self):
        """Override to validate ``raw_data`` / ``metadata`` before build_model."""
        if not isinstance(self.raw_data, pd.DataFrame):
            raise ValueError("Raw data must be a pandas DataFrame")
        if 'changed' not in self.raw_data.columns:
            raise ValueError("Raw data must include a 'changed' column")

    @abstractmethod
    def build_model(self):
        """Populate ``starting_params``, ``param_ids``, and any extra data columns."""

    def assign_targets(self):
        """Create ``self.data['dt']``, ``self.target``, and ``is_first_interval``."""
        dt_col = self.metadata.get('dt_col', 'tag_years')
        if dt_col not in self.raw_data.columns:
            raise ValueError(
                f"dt_col '{dt_col}' not found in raw_data columns"
            )
        self.data['dt'] = jnp.asarray(
            self.raw_data[dt_col].to_numpy(), dtype = jnp.float32
        )
        self.target = jnp.asarray(
            self.raw_data['changed'].to_numpy(), dtype = jnp.float32
        )
        # is_first_interval flags per-individual first rows for the ZIE δ
        # extension (methodology §1.7). Default zeros for legacy frames that
        # don't carry the column — in that case δ has no effect on the
        # likelihood (no first-interval rows), so the sampler just exercises
        # the prior on logit_delta.
        if 'is_first_interval' in self.raw_data.columns:
            self.data['is_first_interval'] = jnp.asarray(
                self.raw_data['is_first_interval'].to_numpy().astype(bool),
                dtype = jnp.float32,
            )
        else:
            self.data['is_first_interval'] = jnp.zeros_like(self.target)

    @abstractmethod
    def event_rate_fun(
        self,
        params: dict[str, jnp.ndarray],
        data: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Poisson event rate per observation."""

    def param_likelihood(self, params: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Optional log-prior contribution. Default is flat (0.0)."""
        return jnp.asarray(0.0)

    def pointwise_log_likelihood(
        self,
        params: dict[str, jnp.ndarray],
        data: dict[str, jnp.ndarray],
        target: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Per-observation ZIE log-likelihood, shape ``(N,)``.

        This is the dense per-row likelihood *without* the final sum — the
        common input to the LPPD / WAIC computations in
        :mod:`openpois.models.metrics`. Always uses the dense per-row formula
        (the sufficient-statistics path cannot produce pointwise values).
        Subclasses with a ZIE likelihood must implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pointwise_log_likelihood"
        )

    def derive_draws(
        self,
        draws: dict[str, jnp.ndarray],
    ) -> dict[str, jnp.ndarray]:
        """
        Augment posterior draws with any derived/back-transformed parameters.

        Default is an identity map. Override in subclasses that sample a
        reparameterised form (e.g. non-centered ``epsilon_raw``) to expose the
        natural parameter (``epsilon``) to downstream consumers.
        """
        return draws

    @abstractmethod
    def build_predict_data(
        self,
        times: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Build the ``data`` dict passed to ``ModelFitter.predict`` for a time grid."""


# Constant rate model -------------------------------------------------------->


class ConstantModel(ModelFactory):
    """
    Constant change rate with ZIE δ mixture.

    λ = exp(log_lambda); δ = sigmoid(logit_delta). A fraction δ of individuals
    change at t = 0 (methodology §1.7); the remaining 1−δ fraction follow
    Exponential(λ). Only the first interval of each individual carries the
    (1−δ) discount — see ``log_likelihood_fun``.

    Metadata keys:
        dt_col: Column containing per-observation interval length in years
            (default ``"tag_years"``).
        log_lambda_prior_scale: Standard deviation of the N(0, scale) prior on
            ``log_lambda`` (default ``DEFAULT_LOG_LAMBDA_PRIOR_SCALE``).
        logit_delta_prior: (loc, scale) tuple for the Normal prior on
            ``logit_delta`` (default ``DEFAULT_LOGIT_DELTA_PRIOR`` = (−3, 1)).
    """

    def build_model(self):
        """Define ``log_lambda`` and ``logit_delta`` parameters + ZIE likelihood."""
        dt_col = self.metadata.get("dt_col", "tag_years")
        empirical_rate = _empirical_rate_from_nonfirst(self.raw_data, dt_col)
        log_lambda_init = float(
            np.log(max(empirical_rate, _EMPIRICAL_RATE_FLOOR))
        )
        logit_delta_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        self.starting_params = {
            "log_lambda": jnp.array(log_lambda_init),
            "logit_delta": jnp.array(logit_delta_init),
        }
        self.param_ids = pd.DataFrame({
            "parameter": ["log_lambda", "logit_delta", "delta"],
            "param_name": ["log_lambda", "logit_delta", "delta"],
            "group_id": [np.nan, np.nan, np.nan],
        })
        self.group_lookup = None
        self.log_likelihood_fun = self._zie_log_likelihood

    @staticmethod
    def _zie_log_likelihood(params, data, target):
        """
        Bernoulli-on-Poisson log-likelihood with the ZIE δ discount on the
        first interval of each individual (methodology §4.2 Step D).

        First-interval rows (``is_first = 1``):
            y=0: log(1−δ) − λ·Δ
            y=1: log(1 − (1−δ)·exp(−λ·Δ))
        Non-first rows:
            y=0: −λ·Δ
            y=1: log(1 − exp(−λ·Δ))
        """
        per_obs = ConstantModel._zie_pointwise(params, data, target)
        return jnp.sum(per_obs)

    @staticmethod
    def _zie_pointwise(params, data, target):
        """Per-row ZIE log-likelihood (no sum) — shared by the summed
        likelihood and :meth:`pointwise_log_likelihood`."""
        rate = jnp.exp(params["log_lambda"]) * data["dt"]
        # log(1−δ) = log(sigmoid(−logit_delta)) via log_sigmoid for numerical
        # stability at extreme logit_delta values.
        log_1md = jax.nn.log_sigmoid(-params["logit_delta"])
        return _zie_pointwise_loglik(
            rate, data["is_first_interval"], log_1md, target
        )

    def pointwise_log_likelihood(self, params, data, target):
        return ConstantModel._zie_pointwise(params, data, target)

    def event_rate_fun(self, params, data):
        return jnp.exp(params["log_lambda"]) * data["dt"]

    def param_likelihood(self, params):
        scale = self.metadata.get(
            "log_lambda_prior_scale", DEFAULT_LOG_LAMBDA_PRIOR_SCALE
        )
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        lp = stats.norm.logpdf(
            params["log_lambda"], loc = 0.0, scale = scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["logit_delta"], loc = delta_loc, scale = delta_scale
        ).sum()
        return lp

    def derive_draws(self, draws):
        """Expose ``delta = sigmoid(logit_delta)`` alongside raw draws."""
        return {**draws, "delta": jax.nn.sigmoid(draws["logit_delta"])}

    def build_predict_data(self, times):
        return {"dt": jnp.asarray(times, dtype = jnp.float32)}


# Constant rate with a time-varying (breakpoint) hazard ----------------------->


def _breakpoint_integrated_hazard(
    lam1: jnp.ndarray,
    lam2: jnp.ndarray,
    t_b: jnp.ndarray,
    age_start: jnp.ndarray,
    age_end: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integral of a two-rate piecewise-constant hazard over ``[age_start,
    age_end]`` (the interval's tag-age bounds).

    The hazard is λ(a) = λ_1 for a < t_b and λ_2 for a ≥ t_b. Its integral
    ``∫_{age_start}^{age_end} λ(a) da`` has the branchless closed form

        crossing = clip(t_b, age_start, age_end)
        H = λ_1·(crossing − age_start) + λ_2·(age_end − crossing)

    which handles all three cases (interval fully before / fully after /
    straddling t_b) at once, and reduces to λ·(age_end − age_start) when
    λ_1 == λ_2. ``age_start`` ≤ ``age_end`` always holds (dt > 0), so ``clip``
    is well-defined. This is the time-varying analogue of the constant model's
    ``λ·Δt`` rate; everything downstream (ZIE δ, ``P = 1 − exp(−rate)``) is
    unchanged.
    """
    crossing = jnp.clip(t_b, age_start, age_end)
    return lam1 * (crossing - age_start) + lam2 * (age_end - crossing)


class ConstantBreakpointModel(ConstantModel):
    """
    Two-rate (breakpoint) change model with the ZIE δ mixture.

    A single global hazard that switches from λ_1 = exp(log_lambda_1) to
    λ_2 = exp(log_lambda_2) at tag age t_B = exp(log_t_breakpoint) years (years
    since the tag value was established, per ``age_start`` / ``age_end``). The
    change probability over an interval is 1 − exp(−H), where H is the integral
    of the piecewise-constant hazard over the interval's tag-age bounds (see
    ``_breakpoint_integrated_hazard``). δ = sigmoid(logit_delta) carries the
    instant-change mass on first intervals exactly as ``ConstantModel``.

    Reduces to ``ConstantModel`` when λ_1 == λ_2 (the rate becomes λ·Δt for any
    t_B). Toggle via ``--model-type constant_breakpoint``.

    Metadata keys (in addition to ConstantModel's):
        t_breakpoint_prior: (loc, scale) tuple for the Normal prior on
            ``log_t_breakpoint`` (default ``DEFAULT_T_BREAKPOINT_PRIOR`` =
            (0.0, 1.0) → median t_B = 1 year).
    """

    def validate_inputs(self):
        super().validate_inputs()
        for col in ("age_start", "age_end"):
            if col not in self.raw_data.columns:
                raise ValueError(
                    f"{type(self).__name__} requires an '{col}' column; run "
                    "prepare_data_for_model (which emits age_start / age_end) "
                    "before fitting."
                )

    def build_model(self):
        """Define ``log_lambda_1/2``, ``log_t_breakpoint``, ``logit_delta``."""
        dt_col = self.metadata.get("dt_col", "tag_years")
        empirical_rate = _empirical_rate_from_nonfirst(self.raw_data, dt_col)
        log_lambda_init = float(
            np.log(max(empirical_rate, _EMPIRICAL_RATE_FLOOR))
        )
        logit_delta_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        tb_loc = float(
            self.metadata.get(
                "t_breakpoint_prior", DEFAULT_T_BREAKPOINT_PRIOR
            )[0]
        )
        self.starting_params = {
            "log_lambda_1": jnp.array(log_lambda_init),
            "log_lambda_2": jnp.array(log_lambda_init),
            "log_t_breakpoint": jnp.array(tb_loc),
            "logit_delta": jnp.array(logit_delta_init),
        }
        names = [
            "log_lambda_1", "log_lambda_2", "log_t_breakpoint", "logit_delta",
            "lambda_1", "lambda_2", "t_breakpoint", "delta",
        ]
        self.param_ids = pd.DataFrame({
            "parameter": names,
            "param_name": names,
            "group_id": [np.nan] * len(names),
        })
        self.group_lookup = None
        self.log_likelihood_fun = self._zie_log_likelihood

    def assign_targets(self):
        """Add ``age_start`` / ``age_end`` on top of dt / target / is_first."""
        super().assign_targets()
        self.data["age_start"] = jnp.asarray(
            self.raw_data["age_start"].to_numpy(), dtype = jnp.float32
        )
        self.data["age_end"] = jnp.asarray(
            self.raw_data["age_end"].to_numpy(), dtype = jnp.float32
        )

    @staticmethod
    def _zie_log_likelihood(params, data, target):
        per_obs = ConstantBreakpointModel._zie_pointwise(params, data, target)
        return jnp.sum(per_obs)

    @staticmethod
    def _zie_pointwise(params, data, target):
        """Per-row ZIE log-likelihood with the breakpoint integrated hazard."""
        rate = _breakpoint_integrated_hazard(
            jnp.exp(params["log_lambda_1"]),
            jnp.exp(params["log_lambda_2"]),
            jnp.exp(params["log_t_breakpoint"]),
            data["age_start"],
            data["age_end"],
        )
        log_1md = jax.nn.log_sigmoid(-params["logit_delta"])
        return _zie_pointwise_loglik(
            rate, data["is_first_interval"], log_1md, target
        )

    def pointwise_log_likelihood(self, params, data, target):
        return ConstantBreakpointModel._zie_pointwise(params, data, target)

    def event_rate_fun(self, params, data):
        return _breakpoint_integrated_hazard(
            jnp.exp(params["log_lambda_1"]),
            jnp.exp(params["log_lambda_2"]),
            jnp.exp(params["log_t_breakpoint"]),
            data["age_start"],
            data["age_end"],
        )

    def param_likelihood(self, params):
        scale = self.metadata.get(
            "log_lambda_prior_scale", DEFAULT_LOG_LAMBDA_PRIOR_SCALE
        )
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        tb_loc, tb_scale = self.metadata.get(
            "t_breakpoint_prior", DEFAULT_T_BREAKPOINT_PRIOR
        )
        lp = stats.norm.logpdf(
            params["log_lambda_1"], loc = 0.0, scale = scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["log_lambda_2"], loc = 0.0, scale = scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["log_t_breakpoint"], loc = tb_loc, scale = tb_scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["logit_delta"], loc = delta_loc, scale = delta_scale
        ).sum()
        return lp

    def derive_draws(self, draws):
        """Expose natural-scale λ_1, λ_2, t_B, δ alongside the raw draws."""
        return {
            **draws,
            "lambda_1": jnp.exp(draws["log_lambda_1"]),
            "lambda_2": jnp.exp(draws["log_lambda_2"]),
            "t_breakpoint": jnp.exp(draws["log_t_breakpoint"]),
            "delta": jax.nn.sigmoid(draws["logit_delta"]),
        }

    def build_predict_data(self, times):
        """From-establishment diagnostic curve: age 0 → t for each t.

        The production rating instead integrates the hazard over each POI's
        actual ``[last_edited, now]`` window at its true tag age (closed-form
        Λ); this curve is for inspection / the saved predictions.csv.
        """
        times = jnp.asarray(times, dtype = jnp.float32)
        return {
            "dt": times,
            "age_start": jnp.zeros_like(times),
            "age_end": times,
        }


# Declining-hazard tag-age baselines (Weibull / Lomax) ----------------------->


def _weibull_integrated_hazard(
    lam: jnp.ndarray,
    shape: jnp.ndarray,
    age_start: jnp.ndarray,
    age_end: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integral of a Weibull tag-age hazard over ``[age_start, age_end]``.

    The hazard is ``λ(a) = λ·p·a^{p-1}`` with cumulative ``H(a) = λ·a^p``, so the
    interval integral is ``λ·(age_end^p − age_start^p)``. ``p < 1`` gives a
    decreasing ("infant-mortality") hazard; ``p = 1`` reduces to the constant
    model's ``λ·Δt`` (since ``age_end − age_start = Δt``).

    A first interval starts at ``age_start = 0``; ``0^p = 0`` is the correct term
    but its derivative w.r.t. ``p`` is ``0·log 0`` (NaN). We use the standard
    double-``where`` guard so the masked branch evaluates a finite power and
    contributes zero gradient.
    """
    def _term(a):
        safe_a = jnp.where(a > 0, a, 1.0)
        return jnp.where(a > 0, lam * safe_a ** shape, 0.0)

    return _term(age_end) - _term(age_start)


def _lomax_integrated_hazard(
    lam: jnp.ndarray,
    theta: jnp.ndarray,
    age_start: jnp.ndarray,
    age_end: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integral of the marginal Lomax (gamma-frailty) tag-age hazard over the
    interval.

    A Gamma(mean 1, variance θ) frailty on a constant rate λ integrates to the
    Lomax / Pareto-II marginal with cumulative hazard ``H(a) = θ^{-1}·log(1 +
    θ·λ·a)`` and strictly decreasing marginal hazard ``λ/(1 + θ·λ·a)`` — the
    survival analogue of Poisson → Negative-Binomial. The interval integral is
    ``θ^{-1}·[log1p(θ·λ·age_end) − log1p(θ·λ·age_start)]``. ``θ → 0`` recovers the
    constant model (``log1p(θλa)/θ → λa``); ``log1p`` keeps it stable for small
    arguments.
    """
    return (
        jnp.log1p(theta * lam * age_end) - jnp.log1p(theta * lam * age_start)
    ) / theta


class WeibullModel(ConstantBreakpointModel):
    """
    Weibull tag-age baseline hazard with the ZIE δ mixture.

    ``λ = exp(log_lambda)``, ``p = exp(log_shape)``; the change probability over
    an interval is ``1 − exp(−H)`` with ``H = λ·(age_end^p − age_start^p)`` (see
    ``_weibull_integrated_hazard``). ``p < 1`` is a decreasing hazard — the
    single-curve analogue of an early-burst-then-decline pattern; ``p = 1``
    reduces to ``ConstantModel``. δ = sigmoid(logit_delta) carries the instant-
    change mass on first intervals exactly as ``ConstantModel``. Toggle via
    ``--model-type weibull``.

    **Status (2026-06): parked diagnostic — registered but NOT used by the
    pipeline.** Built to test whether a declining baseline hazard explains the
    turnover overdispersion. It does not: nationally ``p ≈ 0.86`` (barely
    decreasing) and School-only ``p ≈ 1.51`` (*increasing*), and neither flattens
    the dispersion. The overdispersion turned out to be an edit-triggered
    *informative-sampling* artifact, not a hazard-shape effect — see methodology
    doc §9.3–9.5. Kept for reference / re-use; do not adopt without re-reading it.

    Metadata keys (besides ConstantModel's):
        weibull_log_shape_prior: (loc, scale) Normal prior on ``log_shape``
            (default ``DEFAULT_WEIBULL_LOG_SHAPE_PRIOR`` = (0, 1) → p median 1).
    """

    def build_model(self):
        dt_col = self.metadata.get("dt_col", "tag_years")
        empirical_rate = _empirical_rate_from_nonfirst(self.raw_data, dt_col)
        log_lambda_init = float(
            np.log(max(empirical_rate, _EMPIRICAL_RATE_FLOOR))
        )
        logit_delta_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        self.starting_params = {
            "log_lambda": jnp.array(log_lambda_init),
            "log_shape": jnp.array(0.0),
            "logit_delta": jnp.array(logit_delta_init),
        }
        names = ["log_lambda", "log_shape", "logit_delta",
                 "lambda", "shape", "delta"]
        self.param_ids = pd.DataFrame({
            "parameter": names,
            "param_name": names,
            "group_id": [np.nan] * len(names),
        })
        self.group_lookup = None
        self.log_likelihood_fun = self._zie_log_likelihood

    @staticmethod
    def _zie_log_likelihood(params, data, target):
        return jnp.sum(WeibullModel._zie_pointwise(params, data, target))

    @staticmethod
    def _zie_pointwise(params, data, target):
        rate = _weibull_integrated_hazard(
            jnp.exp(params["log_lambda"]),
            jnp.exp(params["log_shape"]),
            data["age_start"],
            data["age_end"],
        )
        log_1md = jax.nn.log_sigmoid(-params["logit_delta"])
        return _zie_pointwise_loglik(
            rate, data["is_first_interval"], log_1md, target
        )

    def pointwise_log_likelihood(self, params, data, target):
        return WeibullModel._zie_pointwise(params, data, target)

    def event_rate_fun(self, params, data):
        return _weibull_integrated_hazard(
            jnp.exp(params["log_lambda"]),
            jnp.exp(params["log_shape"]),
            data["age_start"],
            data["age_end"],
        )

    def param_likelihood(self, params):
        scale = self.metadata.get(
            "log_lambda_prior_scale", DEFAULT_LOG_LAMBDA_PRIOR_SCALE
        )
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        sh_loc, sh_scale = self.metadata.get(
            "weibull_log_shape_prior", DEFAULT_WEIBULL_LOG_SHAPE_PRIOR
        )
        lp = stats.norm.logpdf(
            params["log_lambda"], loc = 0.0, scale = scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["log_shape"], loc = sh_loc, scale = sh_scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["logit_delta"], loc = delta_loc, scale = delta_scale
        ).sum()
        return lp

    def derive_draws(self, draws):
        """Expose natural-scale λ, p, δ alongside the raw draws."""
        return {
            **draws,
            "lambda": jnp.exp(draws["log_lambda"]),
            "shape": jnp.exp(draws["log_shape"]),
            "delta": jax.nn.sigmoid(draws["logit_delta"]),
        }


class LomaxModel(ConstantBreakpointModel):
    """
    Marginal Gamma-frailty (Lomax / Pareto-II) tag-age hazard with ZIE δ.

    A Gamma(mean 1, variance θ) frailty on a constant exponential rate λ
    marginalises to the Lomax hazard ``λ/(1 + θ·λ·a)`` with cumulative
    ``H(a) = θ^{-1}·log(1 + θ·λ·a)`` (see ``_lomax_integrated_hazard``) — the
    survival analogue of Poisson → Negative-Binomial. ``λ = exp(log_lambda)``;
    ``θ = exp(log_theta)`` is the frailty variance / overdispersion parameter,
    and ``θ → 0`` recovers ``ConstantModel``. δ = sigmoid(logit_delta) carries
    the instant-change mass on first intervals exactly as ``ConstantModel``.
    Toggle via ``--model-type lomax``.

    **Status (2026-06): parked diagnostic — registered but NOT used by the
    pipeline.** Built as the explicit Poisson → Negative-Binomial (frailty)
    response to the overdispersion question. ``θ`` collapses toward 0 (reducing to
    ``ConstantModel``) both nationally and within a single amenity — i.e. the data
    show no global frailty-in-tag-age once δ is present. The overdispersion turned
    out to be an edit-triggered *informative-sampling* artifact, not unobserved
    frailty — see methodology doc §9.3–9.6. Kept for reference; do not adopt
    without re-reading it.

    Metadata keys (besides ConstantModel's):
        lomax_log_theta_prior: (loc, scale) Normal prior on ``log_theta``
            (default ``DEFAULT_LOMAX_LOG_THETA_PRIOR`` = (0, 1.5) → θ median 1).
    """

    def build_model(self):
        dt_col = self.metadata.get("dt_col", "tag_years")
        empirical_rate = _empirical_rate_from_nonfirst(self.raw_data, dt_col)
        log_lambda_init = float(
            np.log(max(empirical_rate, _EMPIRICAL_RATE_FLOOR))
        )
        logit_delta_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        self.starting_params = {
            "log_lambda": jnp.array(log_lambda_init),
            "log_theta": jnp.array(0.0),
            "logit_delta": jnp.array(logit_delta_init),
        }
        names = ["log_lambda", "log_theta", "logit_delta",
                 "lambda", "theta", "delta"]
        self.param_ids = pd.DataFrame({
            "parameter": names,
            "param_name": names,
            "group_id": [np.nan] * len(names),
        })
        self.group_lookup = None
        self.log_likelihood_fun = self._zie_log_likelihood

    @staticmethod
    def _zie_log_likelihood(params, data, target):
        return jnp.sum(LomaxModel._zie_pointwise(params, data, target))

    @staticmethod
    def _zie_pointwise(params, data, target):
        rate = _lomax_integrated_hazard(
            jnp.exp(params["log_lambda"]),
            jnp.exp(params["log_theta"]),
            data["age_start"],
            data["age_end"],
        )
        log_1md = jax.nn.log_sigmoid(-params["logit_delta"])
        return _zie_pointwise_loglik(
            rate, data["is_first_interval"], log_1md, target
        )

    def pointwise_log_likelihood(self, params, data, target):
        return LomaxModel._zie_pointwise(params, data, target)

    def event_rate_fun(self, params, data):
        return _lomax_integrated_hazard(
            jnp.exp(params["log_lambda"]),
            jnp.exp(params["log_theta"]),
            data["age_start"],
            data["age_end"],
        )

    def param_likelihood(self, params):
        scale = self.metadata.get(
            "log_lambda_prior_scale", DEFAULT_LOG_LAMBDA_PRIOR_SCALE
        )
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        th_loc, th_scale = self.metadata.get(
            "lomax_log_theta_prior", DEFAULT_LOMAX_LOG_THETA_PRIOR
        )
        lp = stats.norm.logpdf(
            params["log_lambda"], loc = 0.0, scale = scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["log_theta"], loc = th_loc, scale = th_scale
        ).sum()
        lp = lp + stats.norm.logpdf(
            params["logit_delta"], loc = delta_loc, scale = delta_scale
        ).sum()
        return lp

    def derive_draws(self, draws):
        """Expose natural-scale λ, θ, δ alongside the raw draws."""
        return {
            **draws,
            "lambda": jnp.exp(draws["log_lambda"]),
            "theta": jnp.exp(draws["log_theta"]),
            "delta": jax.nn.sigmoid(draws["logit_delta"]),
        }


# Random effects by group ---------------------------------------------------->


class RandomByTypeModel(ModelFactory):
    """
    Random-effects model on both λ and δ, grouped by a shared label.

    Per-group log-rate:
        log λ_g = log_lambda_0 + ε_g,   ε_g ~ N(0, exp(log_sigma))
        log_sigma ~ N(var_prior[0], var_prior[1])

    Per-group zero-inflated mixture mass on the logit scale:
        logit δ_g = logit_delta_0 + η_g, η_g ~ N(0, exp(log_tau))
        log_tau ~ N(logit_delta_var_prior[0], logit_delta_var_prior[1])

    Two equivalent parameterisations are supported via ``metadata["reparam"]``
    and apply symmetrically to both ε and η:

    * ``"centered"`` (legacy): the sampler traces ``epsilon`` / ``eta``
      directly. Simple, but the posterior is funnel-shaped near sparse groups
      and NUTS tends to diverge there.
    * ``"non_centered"`` (**default**): the sampler traces
      ``epsilon_raw ~ N(0, 1)`` / ``eta_raw ~ N(0, 1)`` and we reconstruct
      ``epsilon = exp(log_sigma) * epsilon_raw`` and
      ``eta = exp(log_tau) * eta_raw`` post-hoc. Removes the funnel, usually
      yielding zero divergences and higher ESS on small groups with no change
      to well-identified groups.

    Under ``non_centered`` the natural-scale ``epsilon`` and ``eta`` draws are
    exposed via ``derive_draws`` along with the per-group ``logit_delta`` and
    ``delta``, so downstream consumers (``predict``, parameter tables, saved
    ``param_draws.csv``) see the same derived names either way.

    Metadata keys:
        group: Column name in raw_data holding the grouping variable. **Required.**
        dt_col: Column containing per-observation interval length (default
            ``"tag_years"``).
        var_prior: ``(loc, scale)`` tuple for the hyperprior on ``log_sigma``
            (default ``DEFAULT_VAR_PRIOR``).
        logit_delta_prior: ``(loc, scale)`` tuple for the prior on the
            intercept ``logit_delta_0`` (default ``DEFAULT_LOGIT_DELTA_PRIOR``).
        logit_delta_var_prior: ``(loc, scale)`` tuple for the tight hyperprior
            on ``log_tau`` (default ``DEFAULT_LOGIT_DELTA_VAR_PRIOR``).
        reparam: ``"centered"`` or ``"non_centered"`` (default
            ``"non_centered"``).
    """

    def validate_inputs(self):
        """Require a valid ``group`` metadata entry that is a column in raw_data."""
        super().validate_inputs()
        group_key = self.metadata.get("group") if self.metadata else None
        if group_key is None:
            raise ValueError("Key 'group' is required in metadata")
        if group_key not in self.raw_data.columns:
            raise ValueError(
                f"Group key '{group_key}' not found in raw data columns: "
                f"{', '.join(self.raw_data.columns.tolist())}"
            )
        reparam = (self.metadata or {}).get("reparam", _DEFAULT_REPARAM)
        if reparam not in _VALID_REPARAMS:
            raise ValueError(
                f"metadata['reparam']={reparam!r} not in "
                f"{_VALID_REPARAMS}"
            )
        suff = (self.metadata or {}).get("use_sufficient_stats", True)
        if not isinstance(suff, bool):
            raise ValueError(
                f"metadata['use_sufficient_stats']={suff!r} must be a bool"
            )

    def build_model(self):
        """Encode group IDs and allocate per-group epsilon + logit_delta."""
        group_key = self.metadata["group"]
        dt_col = self.metadata.get("dt_col", "tag_years")
        self.raw_data = self.raw_data.dropna(subset = [group_key]).copy()
        self.raw_data["group_id"] = (
            self.raw_data[group_key].astype("category").cat.codes
        )
        self.data["group"] = jnp.asarray(
            self.raw_data["group_id"].to_numpy(), dtype = jnp.int32
        )
        self.group_lookup = (
            self.raw_data
            .loc[:, [group_key, "group_id"]]
            .rename(columns = {group_key: "group_name"})
            .drop_duplicates()
            .sort_values("group_id", ascending = True)
            .reset_index(drop = True)
        )
        n_groups = self.group_lookup.shape[0]

        # Empirical-Bayes init so warmup doesn't burn steps escaping a bad
        # starting point. log_lambda_0: non-first-only pooled log-rate
        # (excludes δ-mass). log_sigma: log(std of per-group empirical
        # log-rates on non-first rows), bounded.
        if "is_first_interval" in self.raw_data.columns:
            init_df = self.raw_data.loc[
                ~self.raw_data["is_first_interval"].astype(bool)
            ]
            if init_df.empty:
                init_df = self.raw_data
        else:
            init_df = self.raw_data
        overall_rate = (
            init_df["changed"].mean() / init_df[dt_col].mean()
        )
        log_lambda_0_init = float(
            np.log(max(overall_rate, _EMPIRICAL_RATE_FLOOR))
        )
        per_group = (
            init_df.groupby("group_id", observed = True)
            .agg(
                changed_mean = ("changed", "mean"),
                dt_mean = (dt_col, "mean"),
            )
            .reindex(range(n_groups))
        )
        per_group["changed_mean"] = per_group["changed_mean"].fillna(
            float(init_df["changed"].mean())
        )
        per_group["dt_mean"] = per_group["dt_mean"].fillna(
            float(init_df[dt_col].mean())
        )
        per_group_rates = np.maximum(
            (per_group["changed_mean"] / per_group["dt_mean"]).to_numpy(),
            _EMPIRICAL_RATE_FLOOR,
        )
        per_group_log_rates = np.log(per_group_rates)
        if len(per_group_log_rates) > 1:
            empirical_log_sigma = float(np.log(
                max(float(np.std(per_group_log_rates)), 1e-3)
            ))
        else:
            empirical_log_sigma = 0.0
        log_sigma_init = float(np.clip(
            empirical_log_sigma,
            _LOG_SIGMA_INIT_BOUNDS[0],
            _LOG_SIGMA_INIT_BOUNDS[1],
        ))
        logit_delta_0_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        log_tau_init = float(
            self.metadata.get(
                "logit_delta_var_prior", DEFAULT_LOGIT_DELTA_VAR_PRIOR,
            )[0]
        )
        self._reparam = self.metadata.get("reparam", _DEFAULT_REPARAM)
        self._use_sufficient_stats = self.metadata.get(
            "use_sufficient_stats", True
        )
        eps_key = "epsilon" if self._reparam == "centered" else "epsilon_raw"
        eta_key = "eta" if self._reparam == "centered" else "eta_raw"
        self.starting_params = {
            "log_lambda_0": jnp.array(log_lambda_0_init),
            "log_sigma": jnp.array(log_sigma_init),
            "logit_delta_0": jnp.array(logit_delta_0_init),
            "log_tau": jnp.array(log_tau_init),
            eps_key: jnp.zeros(n_groups),
            eta_key: jnp.zeros(n_groups),
        }
        if self._use_sufficient_stats:
            self._build_sufficient_stats(
                dt_col = dt_col, n_groups = n_groups,
            )
            self.log_likelihood_fun = self._suff_stats_log_likelihood
        else:
            # Dense per-row ZIE path — needed whenever suff-stats is off so the
            # log-likelihood stays methodologically consistent with §4.2.
            self.log_likelihood_fun = self._zie_dense_log_likelihood
        # Under non-centered, expose BOTH *_raw (what NUTS sees) and the
        # back-transformed natural-scale draws (what consumers expect) in
        # param_ids. Per-group logit_delta and delta are always exposed so
        # end users can read off per-category instant-change fractions.
        param_rows = ["log_lambda_0", "log_sigma", "logit_delta_0", "log_tau"]
        param_names = list(param_rows)
        group_ids: list = [np.nan, np.nan, np.nan, np.nan]
        if self._reparam == "non_centered":
            param_rows += [f"epsilon_raw[{i}]" for i in range(n_groups)]
            param_names += ["epsilon_raw"] * n_groups
            group_ids += list(range(n_groups))
        param_rows += [f"epsilon[{i}]" for i in range(n_groups)]
        param_names += ["epsilon"] * n_groups
        group_ids += list(range(n_groups))
        if self._reparam == "non_centered":
            param_rows += [f"eta_raw[{i}]" for i in range(n_groups)]
            param_names += ["eta_raw"] * n_groups
            group_ids += list(range(n_groups))
        param_rows += [f"eta[{i}]" for i in range(n_groups)]
        param_names += ["eta"] * n_groups
        group_ids += list(range(n_groups))
        param_rows += [f"logit_delta[{i}]" for i in range(n_groups)]
        param_names += ["logit_delta"] * n_groups
        group_ids += list(range(n_groups))
        param_rows += [f"delta[{i}]" for i in range(n_groups)]
        param_names += ["delta"] * n_groups
        group_ids += list(range(n_groups))
        self.param_ids = pd.DataFrame({
            "parameter": param_rows,
            "param_name": param_names,
            "group_id": group_ids,
        })

    def _build_sufficient_stats(self, dt_col: str, n_groups: int):
        """
        Precompute first/non-first unchanged-time sums and changed-only arrays.

        The ZIE Bernoulli-on-Poisson log-likelihood factors as
            -Σ_g λ_g · (sum_dt_unchanged_nonfirst[g] + sum_dt_unchanged_first[g])
            + Σ_g n_first_unchanged_by_group[g] · log(1−δ_g)
            + Σ log(1 − exp(−λ_g·Δ))                over non-first y=1 rows
            + Σ log(1 − (1−δ_g)·exp(−λ_g·Δ))        over first y=1 rows
        (methodology §4.3). The unchanged contribution collapses from N terms
        to 3K scalars; the changed contribution runs over only the ~10 %
        of observations that flipped. Stashes six arrays into ``self.data``
        for the jitted log-likelihood to gather.
        """
        dt_np = self.raw_data[dt_col].to_numpy().astype(np.float32)
        target_np = self.raw_data["changed"].to_numpy().astype(np.int32)
        group_np = self.raw_data["group_id"].to_numpy().astype(np.int32)
        if "is_first_interval" in self.raw_data.columns:
            is_first_np = (
                self.raw_data["is_first_interval"].to_numpy().astype(bool)
            )
        else:
            is_first_np = np.zeros(len(self.raw_data), dtype = bool)

        mask_unchanged = target_np == 0
        mask_unchanged_first = mask_unchanged & is_first_np
        mask_unchanged_nonfirst = mask_unchanged & ~is_first_np
        sum_dt_first = np.zeros(n_groups, dtype = np.float32)
        sum_dt_nonfirst = np.zeros(n_groups, dtype = np.float32)
        n_first_unchanged_by_group = np.zeros(n_groups, dtype = np.float32)
        np.add.at(
            sum_dt_first,
            group_np[mask_unchanged_first],
            dt_np[mask_unchanged_first],
        )
        np.add.at(
            sum_dt_nonfirst,
            group_np[mask_unchanged_nonfirst],
            dt_np[mask_unchanged_nonfirst],
        )
        np.add.at(
            n_first_unchanged_by_group,
            group_np[mask_unchanged_first],
            1.0,
        )

        mask_changed = ~mask_unchanged
        mask_changed_first = mask_changed & is_first_np
        mask_changed_nonfirst = mask_changed & ~is_first_np

        self.data["sum_dt_unchanged_first"] = jnp.asarray(sum_dt_first)
        self.data["sum_dt_unchanged_nonfirst"] = jnp.asarray(sum_dt_nonfirst)
        self.data["n_first_unchanged_by_group"] = jnp.asarray(
            n_first_unchanged_by_group
        )
        self.data["group_changed_nonfirst"] = jnp.asarray(
            group_np[mask_changed_nonfirst], dtype = jnp.int32,
        )
        self.data["dt_changed_nonfirst"] = jnp.asarray(
            dt_np[mask_changed_nonfirst], dtype = jnp.float32,
        )
        self.data["group_changed_first"] = jnp.asarray(
            group_np[mask_changed_first], dtype = jnp.int32,
        )
        self.data["dt_changed_first"] = jnp.asarray(
            dt_np[mask_changed_first], dtype = jnp.float32,
        )

    def _log_lambda_per_group(self, params):
        """Shared helper: per-group log-rate under either parameterisation."""
        if self._reparam == "centered":
            return params["log_lambda_0"] + params["epsilon"]
        return (
            params["log_lambda_0"]
            + jnp.exp(params["log_sigma"]) * params["epsilon_raw"]
        )

    def _logit_delta_per_group(self, params):
        """Shared helper: per-group logit_delta under either parameterisation."""
        if self._reparam == "centered":
            return params["logit_delta_0"] + params["eta"]
        return (
            params["logit_delta_0"]
            + jnp.exp(params["log_tau"]) * params["eta_raw"]
        )

    def event_rate_fun(self, params, data):
        log_lambda_g = self._log_lambda_per_group(params)
        return jnp.exp(log_lambda_g[data["group"]]) * data["dt"]

    def log_1md_fun(self, params, data):
        """Per-observation log(1-δ_{g(i)}) for fresh-mode predictions."""
        logit_delta_g = self._logit_delta_per_group(params)
        return jax.nn.log_sigmoid(-logit_delta_g[data["group"]])

    def _zie_dense_log_likelihood(self, params, data, target):
        """
        Per-row ZIE Bernoulli-on-Poisson log-likelihood (methodology §4.2
        Step D). Used when sufficient stats are disabled.
        """
        return jnp.sum(self.pointwise_log_likelihood(params, data, target))

    def pointwise_log_likelihood(self, params, data, target):
        rate = self.event_rate_fun(params, data)
        logit_delta_g = self._logit_delta_per_group(params)
        log_1md_per_row = jax.nn.log_sigmoid(-logit_delta_g[data["group"]])
        return _zie_pointwise_loglik(
            rate, data["is_first_interval"], log_1md_per_row, target
        )

    def _suff_stats_log_likelihood(self, params, data, target):
        """
        ZIE Bernoulli-on-Poisson log-likelihood via K-vectors + changed folds.

        ``target`` is intentionally unused — its information is captured by
        the precomputed sufficient-stats arrays built in
        ``_build_sufficient_stats``. ``log_sigmoid(-logit_delta_g)`` gives
        ``log(1−δ_g)`` with numerical stability at extreme logits.
        """
        del target
        lam_g = jnp.exp(self._log_lambda_per_group(params))
        logit_delta_g = self._logit_delta_per_group(params)
        log_1md_g = jax.nn.log_sigmoid(-logit_delta_g)  # shape (K,)

        # y=0 contribution: exponential survival on all unchanged rows, plus
        # Σ_g n_first_unchanged[g] · log(1−δ_g) for first-interval y=0 rows.
        ll_unchanged = -jnp.sum(
            lam_g * (
                data["sum_dt_unchanged_nonfirst"]
                + data["sum_dt_unchanged_first"]
            )
        )
        ll_unchanged = ll_unchanged + jnp.sum(
            data["n_first_unchanged_by_group"] * log_1md_g
        )

        # y=1 non-first: log(1 − exp(−λ_g·Δ)).
        rate_nf = (
            lam_g[data["group_changed_nonfirst"]]
            * data["dt_changed_nonfirst"]
        )
        ll_changed_nf = jnp.sum(jnp.log(-jnp.expm1(-rate_nf)))

        # y=1 first: log(1 − (1−δ_g)·exp(−λ_g·Δ)) via log(−expm1(log(1−δ_g) − rate)).
        log_1md_f = log_1md_g[data["group_changed_first"]]
        rate_f = (
            lam_g[data["group_changed_first"]] * data["dt_changed_first"]
        )
        ll_changed_f = jnp.sum(jnp.log(-jnp.expm1(log_1md_f - rate_f)))

        return ll_unchanged + ll_changed_nf + ll_changed_f

    def param_likelihood(self, params):
        var_loc, var_scale = self.metadata.get("var_prior", DEFAULT_VAR_PRIOR)
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        tau_loc, tau_scale = self.metadata.get(
            "logit_delta_var_prior", DEFAULT_LOGIT_DELTA_VAR_PRIOR
        )
        ll = stats.norm.logpdf(
            params["log_sigma"], loc = var_loc, scale = var_scale
        ).sum()
        ll = ll + stats.norm.logpdf(
            params["logit_delta_0"], loc = delta_loc, scale = delta_scale
        ).sum()
        ll = ll + stats.norm.logpdf(
            params["log_tau"], loc = tau_loc, scale = tau_scale
        ).sum()
        if self._reparam == "centered":
            ll = ll + stats.norm.logpdf(
                params["epsilon"],
                loc = 0.0,
                scale = jnp.exp(params["log_sigma"]),
            ).sum()
            ll = ll + stats.norm.logpdf(
                params["eta"],
                loc = 0.0,
                scale = jnp.exp(params["log_tau"]),
            ).sum()
        else:
            ll = ll + stats.norm.logpdf(
                params["epsilon_raw"], loc = 0.0, scale = 1.0,
            ).sum()
            ll = ll + stats.norm.logpdf(
                params["eta_raw"], loc = 0.0, scale = 1.0,
            ).sum()
        return ll

    def derive_draws(self, draws):
        """Expose natural-scale epsilon/eta and per-group logit_delta/delta."""
        out = dict(draws)
        if self._reparam != "centered":
            # draws["log_sigma"]: (n_draws,). draws["epsilon_raw"]: (n_draws, K).
            out["epsilon"] = (
                jnp.exp(draws["log_sigma"])[:, None] * draws["epsilon_raw"]
            )
            out["eta"] = (
                jnp.exp(draws["log_tau"])[:, None] * draws["eta_raw"]
            )
        # Per-group logit_delta and delta (always K-vectors per draw).
        eta_natural = out["eta"]
        out["logit_delta"] = draws["logit_delta_0"][:, None] + eta_natural
        out["delta"] = jax.nn.sigmoid(out["logit_delta"])
        return out

    def build_predict_data(self, times):
        if self.group_lookup is None:
            raise RuntimeError(
                "group_lookup is unset; build_model must run first"
            )
        n_groups = self.group_lookup.shape[0]
        times = jnp.asarray(times, dtype = jnp.float32)
        n_periods = times.shape[0]
        return {
            "dt": jnp.tile(times, n_groups),
            "group": jnp.repeat(
                jnp.arange(n_groups, dtype = jnp.int32), n_periods
            ),
        }


# Multiple additive random effects ------------------------------------------->


# Canonical term names for RandomEffectsModel.
_AMENITY_TERM = "amenity"
_MSA_TERM = "msa"
_INTERACTION_TERM = "amenity_msa"
_URBANICITY_TERM = "urbanicity"
_MAIN_TERMS = (_AMENITY_TERM, _MSA_TERM)
# Non-centered gathered terms (each owns a log_sigma + eps_<t>_raw vector).
_GATHERED_TERMS = (_AMENITY_TERM, _MSA_TERM, _INTERACTION_TERM)
_VALID_TERMS = (_AMENITY_TERM, _MSA_TERM, _INTERACTION_TERM, _URBANICITY_TERM)

# δ (zero-inflation) random-intercept terms, each independently toggleable and
# separate from the λ terms above. ``amenity`` groups δ by shared_label,
# ``msa`` by msa_code; each owns a log_tau_<t> scale + eta_<t>_raw vector.
_DELTA_VALID_TERMS = (_AMENITY_TERM, _MSA_TERM)
_DELTA_DEFAULT_COLUMNS = {_AMENITY_TERM: "shared_label", _MSA_TERM: "msa_code"}


def _resolve_delta_terms(meta: dict) -> dict[str, dict]:
    """Resolve the enabled δ random-intercept terms from metadata.

    Preferred form is ``meta["delta_terms"]`` — a dict ``term -> {"column":...,
    "var_prior":...}`` holding only the enabled terms (mirrors ``meta["terms"]``
    for λ). For backward compatibility a legacy single ``meta["delta_group"]``
    column is translated into one δ term (``amenity`` for shared_label, ``msa``
    for msa_code, else a term named after the column).
    """
    dterms = meta.get("delta_terms")
    if dterms is not None:
        out = {}
        for name, cfg in dterms.items():
            col = cfg.get("column", _DELTA_DEFAULT_COLUMNS.get(name, name))
            out[name] = {"column": col, "var_prior": cfg.get("var_prior")}
        return out
    dg = meta.get("delta_group")
    if not dg:
        return {}
    name = next(
        (t for t, c in _DELTA_DEFAULT_COLUMNS.items() if c == dg), dg
    )
    return {name: {"column": dg, "var_prior": None}}


class RandomEffectsModel(ModelFactory):
    """
    Additive multi-term random-effects model on log λ, with ZIE δ.

    Per-observation log-rate (only the enabled terms appear)::

        log λ_i = log_lambda_0
                + amenity_active   · σ_amenity · eps_amenity_raw[a_i]
                + msa_active       · σ_msa     · eps_msa_raw[m_i]
                + am_active        · σ_am      · eps_am_raw[am_i]     (tight σ)
                + β_sub · is_sub_i + β_rural · is_rural_i             (urban = ref)

    where ``σ_t = exp(log_sigma_t)`` and ``eps_t_raw ~ N(0, 1)`` (non-centered).
    Each term is toggled by its presence in ``metadata["terms"]``; ``amenity``
    alone reproduces :class:`RandomByTypeModel`. The ``amenity_msa`` interaction
    only instantiates a level for an ``(amenity, MSA)`` cell with at least
    ``min_count`` distinct POIs; smaller cells get ``am_active = 0`` and fall
    back on the main effects.

    δ (zero-inflation) is configured independently of the λ terms via
    ``metadata["delta_terms"]`` — a composable, separately-toggleable set of
    random intercepts on ``logit δ`` (``amenity`` grouped by shared_label,
    ``msa`` by msa_code), each with its own ``log_tau_<t>`` scale and
    ``eta_<t>_raw`` vector::

        logit δ_i = logit_delta_0 + Σ_t active_{t,i}·exp(log_tau_t)·eta_t_raw[idx]

    With no δ terms enabled δ is a single global ``logit_delta_0``. Unseen δ
    levels back off to the global intercept (``active = 0``) exactly as the λ
    terms do. A legacy scalar ``metadata["delta_group"]`` is still accepted and
    maps to one δ term.

    The likelihood uses a **cell** sufficient-statistics fast path: a cell is a
    unique tuple of all active factor levels (λ factors + δ group), within which
    λ and δ are constant, so the ZIE factorisation of :class:`RandomByTypeModel`
    carries over with ``group → cell``. The per-row dense path (used by
    :meth:`pointwise_log_likelihood`) is reconstructed from the same gathers.

    Metadata keys:
        dt_col: per-observation interval column (default ``"tag_years"``).
        terms: dict mapping term name → config. ``amenity``/``msa`` take
            ``{"column": str, "var_prior": (loc, scale)}``; ``amenity_msa`` takes
            ``{"columns": [amenity_col, msa_col], "var_prior": (loc, scale),
            "min_count": int}``; ``urbanicity`` takes ``{"column": str,
            "prior": (loc, scale)}`` with values ``urban``/``suburban``/``rural``.
        delta_terms: dict mapping δ term name (``amenity``/``msa``) → config
            ``{"column": str, "var_prior": (loc, scale)}`` (only enabled terms);
            empty/absent → global δ. (Legacy: ``delta_group`` scalar column.)
        logit_delta_prior, logit_delta_var_prior, use_sufficient_stats: as in
            :class:`RandomByTypeModel`. ``logit_delta_var_prior`` is the default
            ``log_tau`` hyperprior for δ terms that don't set their own.
    """

    # -- validation --------------------------------------------------------->

    def validate_inputs(self):
        super().validate_inputs()
        meta = self.metadata or {}
        terms = meta.get("terms")
        if not terms:
            raise ValueError(
                "metadata['terms'] must be a non-empty dict for "
                "RandomEffectsModel (use ConstantModel for no random effects)"
            )
        for name, cfg in terms.items():
            if name not in _VALID_TERMS:
                raise ValueError(
                    f"Unknown term {name!r}; valid: {_VALID_TERMS}"
                )
            cols = cfg.get("columns") if name == _INTERACTION_TERM else [cfg["column"]]
            for col in cols:
                if col not in self.raw_data.columns:
                    raise ValueError(
                        f"Term {name!r} column {col!r} not in raw_data columns"
                    )
        for name, cfg in _resolve_delta_terms(meta).items():
            if name not in _DELTA_VALID_TERMS:
                raise ValueError(
                    f"Unknown delta term {name!r}; valid: {_DELTA_VALID_TERMS}"
                )
            if cfg["column"] not in self.raw_data.columns:
                raise ValueError(
                    f"delta term {name!r} column {cfg['column']!r} not in "
                    "raw_data columns"
                )

    # -- build -------------------------------------------------------------->

    def build_model(self):
        meta = self.metadata
        terms = meta["terms"]
        dt_col = meta.get("dt_col", "tag_years")
        self._terms = {k: dict(v) for k, v in terms.items()}
        self._has = {t: (t in self._terms) for t in _VALID_TERMS}
        # δ random-intercept terms (composable, separate from the λ terms).
        self._delta_terms = _resolve_delta_terms(meta)
        self._delta_terms_on = list(self._delta_terms)
        self._delta_grouped = bool(self._delta_terms_on)
        self._use_sufficient_stats = meta.get("use_sufficient_stats", True)

        df = self.raw_data
        # Drop rows missing any active factor column so codings are clean.
        active_cols = []
        for name, cfg in self._terms.items():
            active_cols += (
                cfg["columns"] if name == _INTERACTION_TERM else [cfg["column"]]
            )
        for cfg in self._delta_terms.values():
            active_cols.append(cfg["column"])
        df = df.dropna(subset = list(dict.fromkeys(active_cols))).copy()
        self.raw_data = df

        self.factor_lookups: dict[str, pd.DataFrame] = {}
        # Per-row level index + active mask for each enabled gathered term.
        self._row_idx: dict[str, np.ndarray] = {}
        self._row_active: dict[str, np.ndarray] = {}
        self._n_levels: dict[str, int] = {}

        for term in _MAIN_TERMS:
            if not self._has[term]:
                continue
            col = self._terms[term]["column"]
            codes, levels = pd.factorize(df[col].astype("string"), sort = True)
            self._row_idx[term] = codes.astype(np.int32)
            self._row_active[term] = np.ones(len(df), dtype = np.float32)
            self._n_levels[term] = len(levels)
            self.factor_lookups[term] = pd.DataFrame({
                "level_id": np.arange(len(levels), dtype = np.int64),
                "level_name": list(levels),
            })

        if self._has[_INTERACTION_TERM]:
            self._build_interaction(df)

        if self._has[_URBANICITY_TERM]:
            ucol = self._terms[_URBANICITY_TERM]["column"]
            uvals = df[ucol].astype("string").to_numpy()
            self._is_sub = (uvals == "suburban").astype(np.float32)
            self._is_rural = (uvals == "rural").astype(np.float32)

        # δ grouping coding — one random-intercept per enabled δ term.
        self._delta_row_idx: dict[str, np.ndarray] = {}
        self._delta_row_active: dict[str, np.ndarray] = {}
        self._n_delta: dict[str, int] = {}
        for term, cfg in self._delta_terms.items():
            codes, levels = pd.factorize(
                df[cfg["column"]].astype("string"), sort = True
            )
            self._delta_row_idx[term] = codes.astype(np.int32)
            self._delta_row_active[term] = np.ones(len(df), dtype = np.float32)
            self._n_delta[term] = len(levels)
            self.factor_lookups[f"delta_{term}"] = pd.DataFrame({
                "level_id": np.arange(len(levels), dtype = np.int64),
                "level_name": list(levels),
            })

        self._build_starting_params(df, dt_col)
        self._build_param_ids()
        self._build_cells(df, dt_col)
        # Back-compat: expose the amenity lookup as group_lookup when amenity is
        # the only gathered factor (lets legacy single-factor merges keep working).
        gathered_on = [t for t in _GATHERED_TERMS if self._has[t]]
        self.group_lookup = (
            self.factor_lookups[_AMENITY_TERM].rename(
                columns = {"level_id": "group_id", "level_name": "group_name"}
            )
            if gathered_on == [_AMENITY_TERM]
            else None
        )

        if self._use_sufficient_stats:
            self.log_likelihood_fun = self._suff_stats_log_likelihood
        else:
            self.log_likelihood_fun = self._dense_log_likelihood

    def _build_interaction(self, df: pd.DataFrame):
        """Code the (amenity, MSA) interaction with a distinct-POI min-count floor."""
        cfg = self._terms[_INTERACTION_TERM]
        a_col, m_col = cfg["columns"]
        min_count = int(cfg.get("min_count", DEFAULT_INTERACTION_MIN_COUNT))
        combo = (
            df[a_col].astype("string") + "\x1f" + df[m_col].astype("string")
        )
        # Distinct POIs per cell (unique element id), not row count.
        poi_counts = df.assign(_combo = combo).groupby("_combo")["id"].nunique()
        active_combos = poi_counts[poi_counts >= min_count].index
        active_sorted = sorted(active_combos)
        combo_to_idx = {c: i for i, c in enumerate(active_sorted)}
        combo_np = combo.to_numpy()
        idx = np.array(
            [combo_to_idx.get(c, -1) for c in combo_np], dtype = np.int64
        )
        active = (idx >= 0).astype(np.float32)
        self._row_idx[_INTERACTION_TERM] = np.where(idx >= 0, idx, 0).astype(np.int32)
        self._row_active[_INTERACTION_TERM] = active
        self._n_levels[_INTERACTION_TERM] = max(len(active_sorted), 1)
        names = [c.split("\x1f") for c in active_sorted]
        self.factor_lookups[_INTERACTION_TERM] = pd.DataFrame({
            "level_id": np.arange(len(active_sorted), dtype = np.int64),
            "level_name": ["|".join(p) for p in names],
            "amenity": [p[0] for p in names] if names else [],
            "msa_code": [p[1] for p in names] if names else [],
        })
        n_active = int(active.sum())
        print(
            f"  amenity_msa interaction: {len(active_sorted)} cells "
            f">= {min_count} POIs ({n_active:,}/{len(df):,} rows active)"
        )

    def _build_starting_params(self, df: pd.DataFrame, dt_col: str):
        """Empirical-Bayes starting position, mirroring RandomByTypeModel."""
        if "is_first_interval" in df.columns:
            init_df = df.loc[~df["is_first_interval"].astype(bool)]
            if init_df.empty:
                init_df = df
        else:
            init_df = df
        overall_rate = init_df["changed"].mean() / init_df[dt_col].mean()
        log_lambda_0_init = float(np.log(max(overall_rate, _EMPIRICAL_RATE_FLOOR)))

        params: dict[str, jnp.ndarray] = {
            "log_lambda_0": jnp.array(log_lambda_0_init),
        }
        for term in _GATHERED_TERMS:
            if not self._has[term]:
                continue
            params[f"log_sigma_{term}"] = jnp.array(
                self._init_log_sigma(init_df, term, dt_col)
            )
            params[f"eps_{term}_raw"] = jnp.zeros(self._n_levels[term])
        if self._has[_URBANICITY_TERM]:
            params["beta_urban"] = jnp.zeros(2)

        logit_delta_0_init = float(
            self.metadata.get("logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR)[0]
        )
        params["logit_delta_0"] = jnp.array(logit_delta_0_init)
        for term in self._delta_terms_on:
            log_tau_init = float(self._delta_var_prior(term)[0])
            params[f"log_tau_{term}"] = jnp.array(log_tau_init)
            params[f"eta_{term}_raw"] = jnp.zeros(self._n_delta[term])
        self.starting_params = params

    def _delta_var_prior(self, term: str) -> tuple:
        """(loc, scale) hyperprior on a δ term's log_tau — the term's own
        ``var_prior`` if set, else the model-level ``logit_delta_var_prior``."""
        vp = self._delta_terms[term].get("var_prior")
        if vp is None:
            vp = self.metadata.get(
                "logit_delta_var_prior", DEFAULT_LOGIT_DELTA_VAR_PRIOR
            )
        return tuple(vp)

    def _init_log_sigma(self, init_df, term, dt_col) -> float:
        """log(std of per-level empirical log-rates), bounded — EB init."""
        idx = self._row_idx[term]
        active = self._row_active[term] > 0
        n_levels = self._n_levels[term]
        # Restrict to the init (non-first) rows and active levels.
        if "is_first_interval" in self.raw_data.columns:
            nonfirst = ~self.raw_data["is_first_interval"].astype(bool).to_numpy()
        else:
            nonfirst = np.ones(len(self.raw_data), dtype = bool)
        use = active & nonfirst
        if use.sum() == 0:
            return 0.0
        changed = self.raw_data["changed"].to_numpy()[use]
        dt = self.raw_data[dt_col].to_numpy()[use]
        lev = idx[use]
        sums = np.bincount(lev, weights = changed, minlength = n_levels)
        dts = np.bincount(lev, weights = dt, minlength = n_levels)
        counts = np.bincount(lev, minlength = n_levels)
        seen = counts > 0
        rates = np.maximum(
            sums[seen] / np.maximum(dts[seen], _EMPIRICAL_RATE_FLOOR),
            _EMPIRICAL_RATE_FLOOR,
        )
        log_rates = np.log(rates)
        if len(log_rates) > 1:
            val = float(np.log(max(float(np.std(log_rates)), 1e-3)))
        else:
            val = 0.0
        return float(np.clip(val, _LOG_SIGMA_INIT_BOUNDS[0], _LOG_SIGMA_INIT_BOUNDS[1]))

    def _build_param_ids(self):
        """Long-form, self-describing parameter table."""
        rows = ["log_lambda_0", "logit_delta_0"]
        names = ["log_lambda_0", "logit_delta_0"]
        factors: list = [np.nan, np.nan]
        level_ids: list = [np.nan, np.nan]
        for term in _GATHERED_TERMS:
            if not self._has[term]:
                continue
            rows.append(f"log_sigma_{term}")
            names.append(f"log_sigma_{term}")
            factors.append(np.nan)
            level_ids.append(np.nan)
            n = self._n_levels[term]
            for i in range(n):
                rows.append(f"eps_{term}[{i}]")
                names.append(f"eps_{term}")
                factors.append(term)
                level_ids.append(i)
        if self._has[_URBANICITY_TERM]:
            for i, nm in enumerate(["beta_suburban", "beta_rural"]):
                rows.append(f"beta_urban[{i}]")
                names.append(nm)
                factors.append(_URBANICITY_TERM)
                level_ids.append(np.nan)
        for term in self._delta_terms_on:
            rows.append(f"log_tau_{term}")
            names.append(f"log_tau_{term}")
            factors.append(np.nan)
            level_ids.append(np.nan)
            for i in range(self._n_delta[term]):
                rows.append(f"eta_{term}[{i}]")
                names.append(f"eta_{term}")
                factors.append(f"delta_{term}")
                level_ids.append(i)
        self.param_ids = pd.DataFrame({
            "parameter": rows,
            "param_name": names,
            "factor": factors,
            "level_id": level_ids,
        })

    def _row_data_arrays(self, df: pd.DataFrame) -> dict:
        """Per-row index/active/urbanicity arrays for a frame using the trained
        factor codings. Unseen levels back off (active = 0)."""
        out: dict[str, np.ndarray] = {}
        n = len(df)
        for term in _MAIN_TERMS:
            if not self._has[term]:
                continue
            col = self._terms[term]["column"]
            lut = self.factor_lookups[term]
            name_to_id = dict(zip(lut["level_name"], lut["level_id"]))
            vals = df[col].astype("string").to_numpy()
            idx = np.array([name_to_id.get(v, -1) for v in vals], dtype = np.int64)
            out[f"{term}_idx"] = np.where(idx >= 0, idx, 0).astype(np.int32)
            out[f"{term}_active"] = (idx >= 0).astype(np.float32)
        if self._has[_INTERACTION_TERM]:
            a_col, m_col = self._terms[_INTERACTION_TERM]["columns"]
            lut = self.factor_lookups[_INTERACTION_TERM]
            combo_to_id = {
                f"{a}\x1f{m}": i
                for i, (a, m) in enumerate(zip(lut["amenity"], lut["msa_code"]))
            }
            combo = (
                df[a_col].astype("string") + "\x1f" + df[m_col].astype("string")
            ).to_numpy()
            idx = np.array(
                [combo_to_id.get(c, -1) for c in combo], dtype = np.int64
            )
            out["amenity_msa_idx"] = np.where(idx >= 0, idx, 0).astype(np.int32)
            out["amenity_msa_active"] = (idx >= 0).astype(np.float32)
        if self._has[_URBANICITY_TERM]:
            uvals = df[self._terms[_URBANICITY_TERM]["column"]].astype("string").to_numpy()
            out["is_sub"] = (uvals == "suburban").astype(np.float32)
            out["is_rural"] = (uvals == "rural").astype(np.float32)
        for term, cfg in self._delta_terms.items():
            lut = self.factor_lookups[f"delta_{term}"]
            name_to_id = dict(zip(lut["level_name"], lut["level_id"]))
            vals = df[cfg["column"]].astype("string").to_numpy()
            idx = np.array([name_to_id.get(v, -1) for v in vals], dtype = np.int64)
            # Unseen δ levels back off (active = 0 → global logit_delta_0), the
            # same out-of-sample fallback used for the λ random effects.
            out[f"delta_{term}_idx"] = np.where(idx >= 0, idx, 0).astype(np.int32)
            out[f"delta_{term}_active"] = (idx >= 0).astype(np.float32)
        return out

    def build_row_data(self, df: pd.DataFrame | None = None) -> dict:
        """JAX ``data`` dict at per-row granularity for the dense / pointwise
        path (used by :mod:`openpois.models.metrics`)."""
        if df is None:
            df = self.raw_data
        dt_col = self.metadata.get("dt_col", "tag_years")
        out = {
            k: jnp.asarray(v) for k, v in self._row_data_arrays(df).items()
        }
        out["dt"] = jnp.asarray(df[dt_col].to_numpy(), dtype = jnp.float32)
        if "is_first_interval" in df.columns:
            out["is_first_interval"] = jnp.asarray(
                df["is_first_interval"].to_numpy().astype(bool), dtype = jnp.float32
            )
        else:
            out["is_first_interval"] = jnp.zeros(len(df), dtype = jnp.float32)
        return out

    def _build_cells(self, df: pd.DataFrame, dt_col: str):
        """Factorise rows into cells and precompute per-cell sufficient stats."""
        # Build the per-row tuple of active factor codings; identical tuples
        # share a cell (λ and δ constant within).
        components = []
        comp_names = []
        for term in _GATHERED_TERMS:
            if self._has[term]:
                components.append(self._row_idx[term])
                comp_names.append(f"{term}_idx")
                components.append(self._row_active[term].astype(np.int32))
                comp_names.append(f"{term}_active")
        if self._has[_URBANICITY_TERM]:
            components.append(self._is_sub.astype(np.int32))
            comp_names.append("is_sub")
            components.append(self._is_rural.astype(np.int32))
            comp_names.append("is_rural")
        for term in self._delta_terms_on:
            components.append(self._delta_row_idx[term])
            comp_names.append(f"delta_{term}_idx")

        comp_df = pd.DataFrame({name: c for name, c in zip(comp_names, components)})
        cell_id, _ = pd.factorize(
            pd.MultiIndex.from_frame(comp_df), sort = False
        )
        cell_id = cell_id.astype(np.int64)
        n_cells = int(cell_id.max()) + 1 if len(cell_id) else 0
        self._n_cells = n_cells

        # Per-cell component values (first occurrence per cell).
        cell_first_row = (
            pd.Series(np.arange(len(df)))
            .groupby(cell_id)
            .first()
            .reindex(range(n_cells))
            .to_numpy()
        )
        data: dict[str, jnp.ndarray] = {}
        for term in _GATHERED_TERMS:
            if not self._has[term]:
                continue
            data[f"{term}_idx"] = jnp.asarray(
                self._row_idx[term][cell_first_row], dtype = jnp.int32
            )
            data[f"{term}_active"] = jnp.asarray(
                self._row_active[term][cell_first_row], dtype = jnp.float32
            )
        if self._has[_URBANICITY_TERM]:
            data["is_sub"] = jnp.asarray(
                self._is_sub[cell_first_row], dtype = jnp.float32
            )
            data["is_rural"] = jnp.asarray(
                self._is_rural[cell_first_row], dtype = jnp.float32
            )
        for term in self._delta_terms_on:
            data[f"delta_{term}_idx"] = jnp.asarray(
                self._delta_row_idx[term][cell_first_row], dtype = jnp.int32
            )
            data[f"delta_{term}_active"] = jnp.ones(n_cells, dtype = jnp.float32)

        # Sufficient statistics per cell (mirrors RandomByTypeModel).
        dt_np = df[dt_col].to_numpy().astype(np.float32)
        target_np = df["changed"].to_numpy().astype(np.int32)
        if "is_first_interval" in df.columns:
            is_first_np = df["is_first_interval"].to_numpy().astype(bool)
        else:
            is_first_np = np.zeros(len(df), dtype = bool)

        unchanged = target_np == 0
        u_first = unchanged & is_first_np
        u_nonfirst = unchanged & ~is_first_np
        sum_dt_first = np.zeros(n_cells, dtype = np.float32)
        sum_dt_nonfirst = np.zeros(n_cells, dtype = np.float32)
        n_first_unchanged = np.zeros(n_cells, dtype = np.float32)
        np.add.at(sum_dt_first, cell_id[u_first], dt_np[u_first])
        np.add.at(sum_dt_nonfirst, cell_id[u_nonfirst], dt_np[u_nonfirst])
        np.add.at(n_first_unchanged, cell_id[u_first], 1.0)

        changed = ~unchanged
        c_first = changed & is_first_np
        c_nonfirst = changed & ~is_first_np
        data["sum_dt_unchanged_first"] = jnp.asarray(sum_dt_first)
        data["sum_dt_unchanged_nonfirst"] = jnp.asarray(sum_dt_nonfirst)
        data["n_first_unchanged"] = jnp.asarray(n_first_unchanged)
        data["cell_changed_nonfirst"] = jnp.asarray(
            cell_id[c_nonfirst], dtype = jnp.int32
        )
        data["dt_changed_nonfirst"] = jnp.asarray(
            dt_np[c_nonfirst], dtype = jnp.float32
        )
        data["cell_changed_first"] = jnp.asarray(
            cell_id[c_first], dtype = jnp.int32
        )
        data["dt_changed_first"] = jnp.asarray(dt_np[c_first], dtype = jnp.float32)
        self.data.update(data)

        # Cell → human-readable factor names, for labeling predictions.
        # Label cells by the source column names (shared_label, msa_code,
        # urban_rural) so predictions join cleanly to observations/snapshots.
        cell_lookup = pd.DataFrame({"cell_id": np.arange(n_cells)})
        for term in _MAIN_TERMS:
            if self._has[term]:
                col = self._terms[term]["column"]
                lut = self.factor_lookups[term]
                ids = self._row_idx[term][cell_first_row]
                cell_lookup[col] = lut.set_index("level_id").loc[ids, "level_name"].to_numpy()
        if self._has[_URBANICITY_TERM]:
            ucol = self._terms[_URBANICITY_TERM]["column"]
            cell_lookup["urban_rural"] = df[ucol].astype("string").to_numpy()[cell_first_row]
        self.cell_lookup = cell_lookup

    # -- log λ / δ reconstruction (granularity-agnostic gathers) ----------->

    def _log_lambda(self, params, data):
        out = params["log_lambda_0"]
        for term in _GATHERED_TERMS:
            if not self._has[term]:
                continue
            out = out + (
                data[f"{term}_active"]
                * jnp.exp(params[f"log_sigma_{term}"])
                * params[f"eps_{term}_raw"][data[f"{term}_idx"]]
            )
        if self._has[_URBANICITY_TERM]:
            out = out + (
                params["beta_urban"][0] * data["is_sub"]
                + params["beta_urban"][1] * data["is_rural"]
            )
        return out

    def _logit_delta(self, params, data, ref):
        if not self._delta_terms_on:
            return jnp.broadcast_to(params["logit_delta_0"], jnp.shape(ref))
        out = params["logit_delta_0"]
        for term in self._delta_terms_on:
            out = out + (
                data[f"delta_{term}_active"]
                * jnp.exp(params[f"log_tau_{term}"])
                * params[f"eta_{term}_raw"][data[f"delta_{term}_idx"]]
            )
        return out

    def event_rate_fun(self, params, data):
        return jnp.exp(self._log_lambda(params, data)) * data["dt"]

    def log_1md_fun(self, params, data):
        log_lambda = self._log_lambda(params, data)
        return jax.nn.log_sigmoid(-self._logit_delta(params, data, log_lambda))

    # -- likelihoods -------------------------------------------------------->

    def _suff_stats_log_likelihood(self, params, data, target):
        del target
        log_lambda_c = self._log_lambda(params, data)
        lam_c = jnp.exp(log_lambda_c)
        log_1md_c = jax.nn.log_sigmoid(
            -self._logit_delta(params, data, log_lambda_c)
        )
        ll_unchanged = -jnp.sum(
            lam_c * (
                data["sum_dt_unchanged_nonfirst"] + data["sum_dt_unchanged_first"]
            )
        )
        ll_unchanged = ll_unchanged + jnp.sum(
            data["n_first_unchanged"] * log_1md_c
        )
        rate_nf = lam_c[data["cell_changed_nonfirst"]] * data["dt_changed_nonfirst"]
        ll_changed_nf = jnp.sum(jnp.log(-jnp.expm1(-rate_nf)))
        log_1md_f = log_1md_c[data["cell_changed_first"]]
        rate_f = lam_c[data["cell_changed_first"]] * data["dt_changed_first"]
        ll_changed_f = jnp.sum(jnp.log(-jnp.expm1(log_1md_f - rate_f)))
        return ll_unchanged + ll_changed_nf + ll_changed_f

    def _dense_log_likelihood(self, params, data, target):
        return jnp.sum(self.pointwise_log_likelihood(params, data, target))

    def pointwise_log_likelihood(self, params, data, target):
        log_lambda = self._log_lambda(params, data)
        rate = jnp.exp(log_lambda) * data["dt"]
        log_1md = jax.nn.log_sigmoid(-self._logit_delta(params, data, log_lambda))
        return _zie_pointwise_loglik(
            rate, data["is_first_interval"], log_1md, target
        )

    # -- priors / derived draws / prediction grid -------------------------->

    def param_likelihood(self, params):
        ll = jnp.asarray(0.0)
        for term in _GATHERED_TERMS:
            if not self._has[term]:
                continue
            default = (
                DEFAULT_VAR_PRIOR_INTERACTION if term == _INTERACTION_TERM
                else DEFAULT_VAR_PRIOR_MAIN
            )
            loc, scale = self._terms[term].get("var_prior", default)
            ll = ll + stats.norm.logpdf(
                params[f"log_sigma_{term}"], loc = loc, scale = scale
            ).sum()
            ll = ll + stats.norm.logpdf(
                params[f"eps_{term}_raw"], loc = 0.0, scale = 1.0
            ).sum()
        if self._has[_URBANICITY_TERM]:
            loc, scale = self._terms[_URBANICITY_TERM].get(
                "prior", DEFAULT_URBANICITY_PRIOR
            )
            ll = ll + stats.norm.logpdf(
                params["beta_urban"], loc = loc, scale = scale
            ).sum()
        delta_loc, delta_scale = self.metadata.get(
            "logit_delta_prior", DEFAULT_LOGIT_DELTA_PRIOR
        )
        ll = ll + stats.norm.logpdf(
            params["logit_delta_0"], loc = delta_loc, scale = delta_scale
        ).sum()
        for term in self._delta_terms_on:
            tau_loc, tau_scale = self._delta_var_prior(term)
            ll = ll + stats.norm.logpdf(
                params[f"log_tau_{term}"], loc = tau_loc, scale = tau_scale
            ).sum()
            ll = ll + stats.norm.logpdf(
                params[f"eta_{term}_raw"], loc = 0.0, scale = 1.0
            ).sum()
        return ll

    def derive_draws(self, draws):
        out = dict(draws)
        for term in _GATHERED_TERMS:
            if not self._has[term]:
                continue
            out[f"eps_{term}"] = (
                jnp.exp(draws[f"log_sigma_{term}"])[:, None]
                * draws[f"eps_{term}_raw"]
            )
        for term in self._delta_terms_on:
            out[f"eta_{term}"] = (
                jnp.exp(draws[f"log_tau_{term}"])[:, None]
                * draws[f"eta_{term}_raw"]
            )
        if not self._delta_terms_on:
            out["delta"] = jax.nn.sigmoid(draws["logit_delta_0"])
        return out

    def build_predict_data(self, times):
        """One row per observed cell × time, carrying that cell's factor
        gathers so :meth:`event_rate_fun` works unchanged."""
        times = jnp.asarray(times, dtype = jnp.float32)
        n_periods = times.shape[0]
        n_cells = self._n_cells
        out = {
            "dt": jnp.tile(times, n_cells),
        }
        for term in _GATHERED_TERMS:
            if not self._has[term]:
                continue
            out[f"{term}_idx"] = jnp.repeat(self.data[f"{term}_idx"], n_periods)
            out[f"{term}_active"] = jnp.repeat(
                self.data[f"{term}_active"], n_periods
            )
        if self._has[_URBANICITY_TERM]:
            out["is_sub"] = jnp.repeat(self.data["is_sub"], n_periods)
            out["is_rural"] = jnp.repeat(self.data["is_rural"], n_periods)
        for term in self._delta_terms_on:
            out[f"delta_{term}_idx"] = jnp.repeat(
                self.data[f"delta_{term}_idx"], n_periods
            )
            out[f"delta_{term}_active"] = jnp.repeat(
                self.data[f"delta_{term}_active"], n_periods
            )
        return out


# Registry ------------------------------------------------------------------->


MODEL_REGISTRY = {
    "constant": ConstantModel,
    # Parked diagnostics — registered but not used by the pipeline. The
    # breakpoint / Weibull / Lomax families were built to probe the turnover
    # overdispersion; it proved to be an edit-triggered informative-sampling
    # artifact rather than a hazard-shape / frailty effect (methodology §8-§9).
    "constant_breakpoint": ConstantBreakpointModel,
    "weibull": WeibullModel,
    "lomax": LomaxModel,
    "random_by_type": RandomByTypeModel,
    "random_effects": RandomEffectsModel,
}


def get_model_class(model_name: str) -> type[ModelFactory]:
    """
    Return a ``ModelFactory`` subclass by name from ``MODEL_REGISTRY``.

    Args:
        model_name: Registry key (``"constant"`` or ``"random_by_type"``).

    Returns:
        The corresponding ``ModelFactory`` subclass.

    Raises:
        ValueError: If ``model_name`` is not a registered model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: "
            f"{', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
