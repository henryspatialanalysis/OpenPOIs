#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for the declining-hazard tag-age turnover models (Weibull, Lomax).

Coverage:
* closed-form integrated-hazard helpers match their analytic forms,
* both reduce to the constant model in their respective limits (Weibull p=1,
  Lomax θ→0),
* the Weibull power is gradient-safe at the age=0 first-interval boundary,
* the registry exposes both, and
* a short NUTS fit recovers the generating shape / frailty parameter from data
  simulated under each model.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import (
    LomaxModel,
    WeibullModel,
    _lomax_integrated_hazard,
    _weibull_integrated_hazard,
    get_model_class,
)

A_START = jnp.asarray([0.0, 1.0, 4.0, 0.0])
A_END = jnp.asarray([1.0, 4.0, 9.0, 0.5])


def test_weibull_helper_matches_analytic():
    lam, p = 0.13, 0.4
    got = np.asarray(_weibull_integrated_hazard(lam, p, A_START, A_END))
    want = lam * (np.asarray(A_END) ** p - np.asarray(A_START) ** p)
    np.testing.assert_allclose(got, want, rtol = 1e-5)


def test_lomax_helper_matches_analytic():
    lam, theta = 0.13, 2.5
    got = np.asarray(_lomax_integrated_hazard(lam, theta, A_START, A_END))
    want = (
        np.log1p(theta * lam * np.asarray(A_END))
        - np.log1p(theta * lam * np.asarray(A_START))
    ) / theta
    np.testing.assert_allclose(got, want, rtol = 1e-5)


def test_weibull_p1_reduces_to_constant():
    """p = 1 ⇒ H = λ·(age_end − age_start) = λ·Δt (the constant model)."""
    lam = 0.2
    got = np.asarray(_weibull_integrated_hazard(lam, 1.0, A_START, A_END))
    want = lam * (np.asarray(A_END) - np.asarray(A_START))
    np.testing.assert_allclose(got, want, rtol = 1e-6)


def test_lomax_theta_to_zero_reduces_to_constant():
    """θ → 0 ⇒ H → λ·Δt."""
    lam = 0.2
    got = np.asarray(_lomax_integrated_hazard(lam, 1e-4, A_START, A_END))
    want = lam * (np.asarray(A_END) - np.asarray(A_START))
    np.testing.assert_allclose(got, want, rtol = 1e-3)


def test_weibull_gradient_safe_at_age_zero():
    """The age=0 first-interval start must not produce NaN gradients in p."""
    def loss(log_lambda, log_shape):
        rate = _weibull_integrated_hazard(
            jnp.exp(log_lambda), jnp.exp(log_shape), A_START, A_END
        )
        return jnp.sum(rate)

    g = jax.grad(loss, argnums = (0, 1))(jnp.array(-1.5), jnp.array(-0.7))
    assert all(np.isfinite(np.asarray(x)) for x in g)


def test_registry_exposes_new_models():
    assert get_model_class("weibull") is WeibullModel
    assert get_model_class("lomax") is LomaxModel


def _sim(kind, n = 4000, lam = 0.2, par = 0.5, seed = 0):
    """Rows simulated from the named hazard with a known shape/frailty param.

    ``is_first_interval`` is False throughout so δ does not enter; ``changed``
    is Bernoulli with the model's exact interval probability ``1 − exp(−H)``.
    """
    rng = np.random.default_rng(seed)
    age_start = rng.uniform(0.0, 8.0, n)
    dt = rng.uniform(0.2, 2.0, n)
    age_end = age_start + dt
    if kind == "weibull":
        h = lam * (age_end ** par - age_start ** par)
    else:
        h = (np.log1p(par * lam * age_end)
             - np.log1p(par * lam * age_start)) / par
    changed = (rng.random(n) < (1.0 - np.exp(-h))).astype(int)
    return pd.DataFrame({
        "id": rng.integers(0, n, n),
        "changed": changed,
        "tag_years": dt,
        "age_start": age_start,
        "age_end": age_end,
        "is_first_interval": np.zeros(n, dtype = bool),
    })


def _fit(model, seed = 0):
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data, target = model.target,
        num_warmup = 250, num_samples = 250, num_chains = 1,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        rng_key = jax.random.PRNGKey(seed),
    )
    fitter.fit()
    return fitter


def test_weibull_recovers_decreasing_shape():
    df = _sim("weibull", lam = 0.2, par = 0.45, seed = 1)
    model = WeibullModel(dataset = df, metadata = {"dt_col": "tag_years"})
    fitter = _fit(model, seed = 1)
    tab = fitter.get_parameter_table().set_index("parameter")
    shape_mean = float(tab.loc["shape", "mean"])
    # True p = 0.45 ⇒ clearly decreasing; allow generous slack for short NUTS.
    assert shape_mean < 0.8
    # Probabilities stay in (0, 1).
    probs = np.asarray(fitter.calculate_probs(
        {k: v[0] for k, v in fitter.param_draws.items()}, model.data
    ))
    assert probs.min() > 0.0 and probs.max() < 1.0


def test_lomax_recovers_positive_frailty():
    df = _sim("lomax", lam = 0.2, par = 2.5, seed = 2)
    model = LomaxModel(dataset = df, metadata = {"dt_col": "tag_years"})
    fitter = _fit(model, seed = 2)
    tab = fitter.get_parameter_table().set_index("parameter")
    theta_mean = float(tab.loc["theta", "mean"])
    # True θ = 2.5 ⇒ substantial frailty; the constant model would be θ = 0.
    assert theta_mean > 1.0
