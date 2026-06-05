#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for JAX-based OSM turnover models.

Uses small synthetic frames + short NUTS runs so the suite stays fast. The
recovery tolerances are loose on purpose: ``num_draws = 300`` produces noisy
estimates, but we still want to catch outright regressions (wrong math, wrong
sign, broken priors).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import pandas as pd
import pytest

from openpois.models.jax_core import jax_rng
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import (
    MODEL_REGISTRY,
    ConstantModel,
    RandomByTypeModel,
    RandomEffectsModel,
    get_model_class,
)


NUM_DRAWS = 300


def _simulate_frame(
    key,
    n: int,
    true_log_lambda_by_group: np.ndarray,
    group_names: list[str],
    is_first_interval: np.ndarray | None = None,
    true_delta_by_group: np.ndarray | None = None,
    dt_range: tuple[float, float] = (0.5, 5.0),
) -> pd.DataFrame:
    """Build a DataFrame matching the per-group (ZIE) likelihood.

    ``is_first_interval`` defaults to all-False so the emitted frame behaves
    like a pure Exponential(λ) fit under the ZIE likelihood (log(1−δ)
    contributes nothing). Pass a boolean array to mark individual rows as
    first-interval for tests that need to exercise the δ branch.

    ``true_delta_by_group`` (natural scale, length K) optionally adds an
    instant-change mass on first-interval rows: with probability δ_g the row
    is forced to y=1 (δ-component fires at t=0), otherwise the standard
    Exponential(λ_g) draw holds.
    """
    key_group, key_dt, key_y = jrd.split(key, 3)
    g = np.asarray(
        jrd.randint(key_group, (n,), 0, len(group_names))
    )
    dt = np.asarray(
        jrd.uniform(key_dt, (n,), minval = dt_range[0], maxval = dt_range[1])
    )
    lam = np.exp(np.asarray(true_log_lambda_by_group)[g])
    p = 1.0 - np.exp(-lam * dt)
    y = np.asarray(jrd.bernoulli(key_y, jnp.asarray(p))).astype(np.int32)
    if is_first_interval is None:
        is_first_interval = np.zeros(n, dtype = bool)
    if true_delta_by_group is not None:
        # Instant-change mass fires only on first-interval rows. Fold a fixed
        # tag into the simulation key so old legacy-signature callers (no
        # true_delta_by_group) stay bit-identical to pre-change behavior.
        key_delta = jrd.fold_in(key, 99)
        delta_per_row = np.asarray(true_delta_by_group)[g]
        fire = np.asarray(
            jrd.bernoulli(key_delta, jnp.asarray(delta_per_row))
        )
        y = np.where(is_first_interval & fire, 1, y).astype(np.int32)
    return pd.DataFrame({
        "tag_years": dt,
        "changed": y,
        "group_col": [group_names[i] for i in g],
        "is_first_interval": is_first_interval,
    })


def _run_fitter(model, key) -> ModelFitter:
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_draws = NUM_DRAWS,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        log_1md_fun = getattr(model, "log_1md_fun", None),
        rng_key = key,
    )
    fitter.fit()
    return fitter


def test_constant_model_recovery():
    """ConstantModel posterior should bracket the true log_lambda."""
    true_log_lambda = -1.2
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_frame(
        key_sim,
        n = 2_000,
        true_log_lambda_by_group = np.array([true_log_lambda]),
        group_names = ["only"],
    )
    model = ConstantModel(dataset = df, metadata = {"dt_col": "tag_years"})
    fitter = _run_fitter(model, key_fit)

    row = fitter.get_parameter_table().iloc[0]
    assert row["parameter"] == "log_lambda"
    assert row["lower"] <= true_log_lambda <= row["upper"], (
        f"true log_lambda {true_log_lambda:+.3f} not covered by "
        f"[{row['lower']:+.3f}, {row['upper']:+.3f}]"
    )
    assert abs(row["mean"] - true_log_lambda) < 0.3


def _build_random_by_type(df, reparam = None):
    metadata = {
        "dt_col": "tag_years",
        "group": "group_col",
        "var_prior": (0.0, 1.0),
    }
    if reparam is not None:
        metadata["reparam"] = reparam
    return RandomByTypeModel(dataset = df, metadata = metadata)


@pytest.mark.parametrize("reparam", ["non_centered", "centered"])
def test_random_by_type_recovery(reparam):
    """Recover per-group mean lambdas under both parameterisations."""
    group_names = ["aaa", "bbb", "ccc"]
    true_log_lambda_0 = -1.0
    true_epsilons = np.array([-0.5, 0.0, 0.6])
    true_log_lambda_by_group = true_log_lambda_0 + true_epsilons
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_frame(
        key_sim,
        n = 3_000,
        true_log_lambda_by_group = true_log_lambda_by_group,
        group_names = group_names,
    )
    model = _build_random_by_type(df, reparam = reparam)
    fitter = _run_fitter(model, key_fit)

    # Recovered per-group mean log-lambda. ``derive_draws`` makes ``epsilon``
    # available regardless of parameterisation.
    draws = fitter.get_parameter_draws()
    assert "epsilon" in draws
    group_log_lambdas = (
        draws["log_lambda_0"][:, None] + draws["epsilon"]
    )
    post_mean = np.asarray(jnp.mean(group_log_lambdas, axis = 0))
    for i, truth in enumerate(true_log_lambda_by_group):
        assert abs(post_mean[i] - truth) < 0.4, (
            f"group {group_names[i]}: posterior mean {post_mean[i]:+.3f} "
            f"far from truth {truth:+.3f}"
        )

    # param_ids layout: 4 scalars (log_lambda_0, log_sigma, logit_delta_0,
    # log_tau), then per-group epsilon* rows, then per-group eta* rows, then
    # per-group logit_delta + delta rows. Under non-centered the *_raw
    # variants precede their natural-scale counterparts.
    names = list(model.param_ids["param_name"])
    k = len(group_names)
    assert names[:4] == [
        "log_lambda_0", "log_sigma", "logit_delta_0", "log_tau",
    ]
    if reparam == "centered":
        assert names[4:] == (
            ["epsilon"] * k
            + ["eta"] * k
            + ["logit_delta"] * k
            + ["delta"] * k
        )
    else:
        assert names[4:] == (
            ["epsilon_raw"] * k
            + ["epsilon"] * k
            + ["eta_raw"] * k
            + ["eta"] * k
            + ["logit_delta"] * k
            + ["delta"] * k
        )
    assert sorted(model.group_lookup["group_name"]) == sorted(group_names)


@pytest.mark.parametrize("reparam", ["non_centered", "centered"])
def test_random_by_type_sufficient_stats_matches_dense(reparam):
    """Sufficient-stats and dense ZIE log-likelihoods agree to tight tolerance."""
    import jax

    group_names = ["aaa", "bbb", "ccc", "ddd"]
    key = jax_rng()
    # Mark ~30 % of rows as first-interval so both ZIE branches are exercised.
    n = 1_500
    is_first = np.asarray(
        jrd.bernoulli(jrd.fold_in(key, 42), 0.3, shape = (n,))
    )
    df = _simulate_frame(
        key,
        n = n,
        true_log_lambda_by_group = np.array([-1.2, -0.8, -0.4, -0.1]),
        group_names = group_names,
        is_first_interval = is_first,
    )
    md_dense = {
        "dt_col": "tag_years",
        "group": "group_col",
        "var_prior": (0.0, 1.0),
        "reparam": reparam,
        "use_sufficient_stats": False,
    }
    md_suff = {**md_dense, "use_sufficient_stats": True}
    model_dense = RandomByTypeModel(dataset = df, metadata = md_dense)
    model_suff = RandomByTypeModel(dataset = df, metadata = md_suff)

    # Pick a not-all-zero parameter point with a non-prior-mean logit_delta_0
    # and non-zero per-group eta so the per-group δ path is exercised in both
    # the dense and suff-stats implementations.
    log_sigma = jnp.array(-0.2)
    logit_delta_0 = jnp.array(-2.0)
    log_tau = jnp.array(-1.5)
    if reparam == "centered":
        params = {
            "log_lambda_0": jnp.array(-1.0),
            "log_sigma": log_sigma,
            "logit_delta_0": logit_delta_0,
            "log_tau": log_tau,
            "epsilon": jnp.array([0.3, -0.2, 0.1, -0.4]),
            "eta": jnp.array([0.15, -0.05, 0.20, -0.10]),
        }
    else:
        params = {
            "log_lambda_0": jnp.array(-1.0),
            "log_sigma": log_sigma,
            "logit_delta_0": logit_delta_0,
            "log_tau": log_tau,
            "epsilon_raw": jnp.array([0.3, -0.2, 0.1, -0.4]),
            "eta_raw": jnp.array([0.8, -0.3, 1.1, -0.6]),
        }

    # Dense ZIE path via the model's own dense log_likelihood_fun.
    ll_dense = float(
        model_dense.log_likelihood_fun(
            params, model_dense.data, model_dense.target,
        )
        + model_dense.param_likelihood(params)
    )

    # Suff-stats path: log_likelihood_fun(params, data, target) + prior.
    ll_suff = float(
        model_suff.log_likelihood_fun(
            params, model_suff.data, model_suff.target,
        )
        + model_suff.param_likelihood(params)
    )
    np.testing.assert_allclose(ll_suff, ll_dense, rtol = 1e-4, atol = 1e-3)

    # Gradients must agree too — this is what NUTS actually uses.
    def _wrap_dense(p):
        return (
            model_dense.log_likelihood_fun(p, model_dense.data, model_dense.target)
            + model_dense.param_likelihood(p)
        )

    def _wrap_suff(p):
        return (
            model_suff.log_likelihood_fun(p, model_suff.data, model_suff.target)
            + model_suff.param_likelihood(p)
        )

    g_dense = jax.grad(_wrap_dense)(params)
    g_suff = jax.grad(_wrap_suff)(params)
    for k in params:
        np.testing.assert_allclose(
            np.asarray(g_suff[k]), np.asarray(g_dense[k]),
            rtol = 1e-3, atol = 1e-3,
        )


def test_random_by_type_reparam_likelihoods_agree():
    """Centered and non-centered log-densities agree at matching parameter points."""
    group_names = ["aaa", "bbb"]
    key = jax_rng()
    df = _simulate_frame(
        key,
        n = 500,
        true_log_lambda_by_group = np.array([-1.2, -0.8]),
        group_names = group_names,
    )
    model_c = _build_random_by_type(df, reparam = "centered")
    model_nc = _build_random_by_type(df, reparam = "non_centered")

    # Pick an arbitrary (not-all-zero) point. epsilon = exp(log_sigma)*eps_raw
    # and eta = exp(log_tau)*eta_raw under the non-centered parameterisation.
    log_lambda_0 = jnp.array(-1.0)
    log_sigma = jnp.array(-0.3)
    epsilon_raw = jnp.array([0.5, -0.7])
    epsilon = jnp.exp(log_sigma) * epsilon_raw

    logit_delta_0 = jnp.array(-2.5)
    log_tau = jnp.array(-1.2)
    eta_raw = jnp.array([0.9, -0.4])
    eta = jnp.exp(log_tau) * eta_raw

    params_c = {
        "log_lambda_0": log_lambda_0,
        "log_sigma": log_sigma,
        "logit_delta_0": logit_delta_0,
        "log_tau": log_tau,
        "epsilon": epsilon,
        "eta": eta,
    }
    params_nc = {
        "log_lambda_0": log_lambda_0,
        "log_sigma": log_sigma,
        "logit_delta_0": logit_delta_0,
        "log_tau": log_tau,
        "epsilon_raw": epsilon_raw,
        "eta_raw": eta_raw,
    }

    # Event rates must match (deterministic transform).
    r_c = model_c.event_rate_fun(params_c, model_c.data)
    r_nc = model_nc.event_rate_fun(params_nc, model_nc.data)
    np.testing.assert_allclose(np.asarray(r_c), np.asarray(r_nc), rtol = 1e-5)

    # Full log-priors differ only by the Jacobian of the N(0, exp(·)) vs
    # N(0, 1) parameterisations on both random effects; here we compare in
    # closed form. Change-of-variables:
    #   log p_c(epsilon | log_sigma) = log p_nc(eps_raw) - K * log_sigma
    #   log p_c(eta | log_tau)       = log p_nc(eta_raw) - K * log_tau
    lp_c = float(model_c.param_likelihood(params_c))
    lp_nc = float(model_nc.param_likelihood(params_nc))
    k = len(group_names)
    expected_gap = float(k * log_sigma + k * log_tau)
    np.testing.assert_allclose(lp_c - lp_nc, -expected_gap, atol = 1e-4)


def test_predictions_schema_constant():
    """predict() output for ConstantModel has the expected columns + row count."""
    key = jax_rng()
    df = _simulate_frame(
        key,
        n = 400,
        true_log_lambda_by_group = np.array([-1.5]),
        group_names = ["only"],
    )
    model = ConstantModel(dataset = df, metadata = {"dt_col": "tag_years"})
    fitter = _run_fitter(model, jrd.fold_in(key, 1))

    times = jnp.arange(11) / 10.0
    preds = fitter.predict(data = model.build_predict_data(times))
    assert list(preds.columns) == ["p_mean", "p_lower", "p_upper"]
    assert len(preds) == 11
    # P(change) must be in [0, 1] and monotonically non-decreasing in time.
    pm = preds["p_mean"].values
    assert np.all((pm >= 0.0) & (pm <= 1.0))
    assert np.all(np.diff(pm) >= -1e-6)


def test_predictions_schema_random_by_type():
    """predict() output for RandomByTypeModel has one row per (group, time)."""
    group_names = ["aaa", "bbb"]
    key = jax_rng()
    df = _simulate_frame(
        key,
        n = 600,
        true_log_lambda_by_group = np.array([-1.2, -0.6]),
        group_names = group_names,
    )
    model = RandomByTypeModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years",
            "group": "group_col",
            "var_prior": (0.0, 1.0),
        },
    )
    fitter = _run_fitter(model, jrd.fold_in(key, 1))

    times = jnp.arange(5) / 10.0
    pred_data = model.build_predict_data(times)
    preds = fitter.predict(data = pred_data)
    assert list(preds.columns) == ["p_mean", "p_lower", "p_upper"]
    assert len(preds) == len(group_names) * len(times)
    assert int(pred_data["group"].max()) == len(group_names) - 1


def test_model_registry():
    """Registry exposes the supported models and rejects removed ones."""
    assert set(MODEL_REGISTRY) == {"constant", "random_by_type", "random_effects"}
    assert get_model_class("constant") is ConstantModel
    assert get_model_class("random_by_type") is RandomByTypeModel
    assert get_model_class("random_effects") is RandomEffectsModel
    with pytest.raises(ValueError, match = "Unknown model"):
        get_model_class("pseudo_varying")


def test_random_by_type_multichain_diagnostics():
    """Multi-chain NUTS produces per-chain draws plus R-hat / ESS diagnostics."""
    group_names = ["aaa", "bbb"]
    key = jax_rng()
    key_sim, key_first, key_fit = jrd.split(key, 3)
    # Mark ~30 % of rows as first-interval and emit a nontrivial δ component so
    # the per-group δ parameters have identifying data (otherwise eta/eta_raw
    # are driven entirely by the prior and ESS collapses).
    n = 1_500
    is_first = np.asarray(
        jrd.bernoulli(key_first, 0.3, shape = (n,))
    )
    df = _simulate_frame(
        key_sim,
        n = n,
        true_log_lambda_by_group = np.array([-1.0, -0.6]),
        group_names = group_names,
        is_first_interval = is_first,
        true_delta_by_group = np.array([0.04, 0.08]),
    )
    model = _build_random_by_type(df, reparam = "non_centered")
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_draws = NUM_DRAWS,
        num_chains = 2,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        log_1md_fun = model.log_1md_fun,
        rng_key = key_fit,
    )
    fitter.fit()

    # chain_draws keeps the (chain, draw, ...) axis; param_draws is flattened.
    log_lambda_chains = fitter.chain_draws["log_lambda_0"]
    assert log_lambda_chains.shape == (2, NUM_DRAWS)
    log_lambda_flat = fitter.param_draws["log_lambda_0"]
    assert log_lambda_flat.shape == (2 * NUM_DRAWS,)

    # Diagnostics DataFrame: R-hat and ESS finite for every sampled parameter.
    diag = fitter.diagnostics
    assert set(diag.columns) == {"parameter", "rhat", "ess_bulk"}
    assert not diag["rhat"].isna().any()
    assert (diag["rhat"] < 1.5).all(), (
        f"max rhat = {diag['rhat'].max():.3f} — suspiciously high for a "
        "well-specified model"
    )
    assert (diag["ess_bulk"] > 10.0).all()


def test_random_by_type_delta_recovery():
    """RandomByTypeModel recovers per-group δ when the truth varies by group."""
    group_names = ["aaa", "bbb", "ccc", "ddd"]
    # Keep λ small and Δ short so the ZIE instant-change mass δ stands out
    # from Poisson-driven change in the likelihood — with λ ≈ 0.13 and Δ ~ 0.3
    # the base change probability is ~4 %, so a 12 % δ dominates the first-
    # interval y=1 signal and the posterior can identify it.
    true_log_lambda_by_group = np.array([-2.0, -2.0, -2.0, -2.0])
    true_delta_by_group = np.array([0.02, 0.08, 0.16, 0.30])
    n = 5_000
    key = jax_rng()
    key_sim, key_first, key_fit = jrd.split(key, 3)
    # All rows first-interval so every row carries δ signal.
    is_first = np.ones(n, dtype = bool)
    df = _simulate_frame(
        key_sim,
        n = n,
        true_log_lambda_by_group = true_log_lambda_by_group,
        group_names = group_names,
        is_first_interval = is_first,
        true_delta_by_group = true_delta_by_group,
        dt_range = (0.1, 0.5),
    )
    # Loosen the tight default prior so the test data can pull posterior δ to
    # the truth within a short chain — otherwise heavy shrinkage fights the
    # recovery signal.
    model = RandomByTypeModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years",
            "group": "group_col",
            "var_prior": (0.0, 1.0),
            "logit_delta_var_prior": (0.0, 1.0),
            "reparam": "non_centered",
        },
    )
    fitter = _run_fitter(model, key_fit)

    draws = fitter.get_parameter_draws()
    # Map posterior delta to the same group-name order as the truth array.
    group_order = (
        model.group_lookup
        .set_index("group_name")
        .loc[group_names, "group_id"]
        .to_numpy()
    )
    post_mean_delta = np.asarray(
        jnp.mean(draws["delta"], axis = 0)
    )[group_order]
    # δ is weakly identified relative to λ when first-interval Δ is not small,
    # so the posterior is noisy. Loose coverage check + rank check keeps the
    # test meaningful as a regression sentinel.
    for i, truth in enumerate(true_delta_by_group):
        assert abs(post_mean_delta[i] - truth) < 0.10, (
            f"group {group_names[i]}: posterior δ {post_mean_delta[i]:.3f} "
            f"far from truth {truth:.3f}"
        )
    # Rank of posterior means should match rank of truth.
    assert np.all(
        np.argsort(post_mean_delta) == np.argsort(true_delta_by_group)
    ), (
        f"posterior ranking {np.argsort(post_mean_delta)} doesn't match truth "
        f"ranking {np.argsort(true_delta_by_group)}"
    )


def test_random_by_type_tight_tau_prior_shrinks():
    """Under uniform true δ, the tight log_tau prior concentrates τ near zero."""
    group_names = ["aaa", "bbb", "ccc", "ddd"]
    true_delta = 0.05
    true_delta_by_group = np.full(len(group_names), true_delta)
    n = 2_000
    key = jax_rng()
    key_sim, key_first, key_fit = jrd.split(key, 3)
    is_first = np.asarray(
        jrd.bernoulli(key_first, 0.3, shape = (n,))
    )
    df = _simulate_frame(
        key_sim,
        n = n,
        true_log_lambda_by_group = np.array([-1.0, -1.0, -1.0, -1.0]),
        group_names = group_names,
        is_first_interval = is_first,
        true_delta_by_group = true_delta_by_group,
    )
    # Use the default tight prior explicitly.
    model = RandomByTypeModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years",
            "group": "group_col",
            "var_prior": (0.0, 1.0),
            "logit_delta_var_prior": (-2.0, 0.5),
            "reparam": "non_centered",
        },
    )
    fitter = _run_fitter(model, key_fit)

    draws = fitter.get_parameter_draws()
    log_tau_upper = float(
        jnp.quantile(draws["log_tau"], 0.975, method = "linear")
    )
    # Under the tight prior (mean -2, scale 0.5), uniform truth should keep the
    # upper credible bound well below -1 (i.e. τ < ~0.37). If this fails the
    # prior is effectively not shrinking.
    assert log_tau_upper < -0.75, (
        f"log_tau 97.5% upper = {log_tau_upper:+.3f} — tight prior isn't "
        "shrinking τ as expected"
    )


def test_random_by_type_requires_group():
    """Missing 'group' metadata key is a construction-time error."""
    df = pd.DataFrame({
        "tag_years": [1.0, 2.0],
        "changed": [0, 1],
        "group_col": ["a", "b"],
    })
    with pytest.raises(ValueError, match = "group"):
        RandomByTypeModel(dataset = df, metadata = {"dt_col": "tag_years"})


# Multi-term random effects -------------------------------------------------->


def _re_frame(seed = 0, n = 4000):
    """Synthetic observations carrying the amenity / MSA / urban_rural columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": rng.integers(0, 1500, n),
        "shared_label": rng.choice([f"a{i}" for i in range(6)], n),
        "msa_code": rng.choice(["12345", "31080", "NO_MSA"], n),
        "urban_rural": rng.choice(["urban", "suburban", "rural"], n),
        "tag_years": rng.uniform(0.2, 4.0, n),
        "changed": rng.binomial(1, 0.12, n),
        "is_first_interval": rng.binomial(1, 0.5, n).astype(bool),
    })


_ALL_TERMS = {
    "amenity": {"column": "shared_label"},
    "msa": {"column": "msa_code"},
    "amenity_msa": {"columns": ["shared_label", "msa_code"], "min_count": 5},
    "urbanicity": {"column": "urban_rural"},
}


def _perturbed_params(model, seed = 1):
    key = jrd.PRNGKey(seed)
    return {
        k: v + 0.1 * jrd.normal(jrd.fold_in(key, i), v.shape)
        for i, (k, v) in enumerate(model.starting_params.items())
    }


@pytest.mark.parametrize("terms", [
    {"amenity": {"column": "shared_label"}},
    {"msa": {"column": "msa_code"}},
    {"urbanicity": {"column": "urban_rural"}},
    _ALL_TERMS,
])
def test_random_effects_suff_stats_matches_dense(terms):
    """Cell sufficient-stats log-density == dense per-row, value AND gradient,
    for every term combination."""
    df = _re_frame()
    model = RandomEffectsModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years", "terms": terms, "delta_group": "shared_label",
        },
    )
    params = _perturbed_params(model)
    row = model.build_row_data()
    ss = model._suff_stats_log_likelihood(params, model.data, model.target)
    dense = model._dense_log_likelihood(params, row, model.target)
    assert abs(float(ss) - float(dense)) < 1e-3

    g_ss = jax.grad(
        lambda p: model._suff_stats_log_likelihood(p, model.data, model.target)
    )(params)
    g_dense = jax.grad(
        lambda p: model._dense_log_likelihood(p, row, model.target)
    )(params)
    for k in params:
        assert np.allclose(
            np.asarray(g_ss[k]), np.asarray(g_dense[k]), atol = 1e-3
        ), f"gradient mismatch for {k}"


def test_random_effects_pointwise_sums_to_total():
    """pointwise_log_likelihood sums to the dense total log-likelihood."""
    df = _re_frame()
    model = RandomEffectsModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years", "terms": _ALL_TERMS, "delta_group": "shared_label",
        },
    )
    params = _perturbed_params(model)
    row = model.build_row_data()
    pw = jnp.sum(model.pointwise_log_likelihood(params, row, model.target))
    total = model._dense_log_likelihood(params, row, model.target)
    assert abs(float(pw) - float(total)) < 1e-4


def test_random_effects_amenity_only_matches_random_by_type():
    """amenity-only RandomEffectsModel reproduces RandomByTypeModel's
    log-density exactly at a shared parameter point."""
    df = _re_frame()
    re_model = RandomEffectsModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years",
            "terms": {"amenity": {"column": "shared_label"}},
            "delta_group": "shared_label",
        },
    )
    rbt = RandomByTypeModel(
        dataset = df, metadata = {"dt_col": "tag_years", "group": "shared_label"},
    )
    rng = np.random.default_rng(3)
    k = re_model._n_levels["amenity"]
    eps = jnp.asarray(rng.normal(size = k))
    eta = jnp.asarray(rng.normal(size = re_model._n_delta))
    re_p = {
        "log_lambda_0": jnp.array(-2.0), "log_sigma_amenity": jnp.array(-0.5),
        "eps_amenity_raw": eps, "logit_delta_0": jnp.array(-3.0),
        "log_tau": jnp.array(-2.0), "eta_raw": eta,
    }
    rbt_p = {
        "log_lambda_0": jnp.array(-2.0), "log_sigma": jnp.array(-0.5),
        "epsilon_raw": eps, "logit_delta_0": jnp.array(-3.0),
        "log_tau": jnp.array(-2.0), "eta_raw": eta,
    }
    ll_re = re_model._suff_stats_log_likelihood(re_p, re_model.data, re_model.target)
    ll_rbt = rbt._suff_stats_log_likelihood(rbt_p, rbt.data, rbt.target)
    assert abs(float(ll_re) - float(ll_rbt)) < 1e-4


def test_random_effects_interaction_min_count_floor():
    """(amenity, MSA) cells below the distinct-POI floor get am_active = 0."""
    df = _re_frame()
    model = RandomEffectsModel(
        dataset = df,
        metadata = {
            "dt_col": "tag_years",
            "terms": {
                "amenity_msa": {
                    "columns": ["shared_label", "msa_code"], "min_count": 10_000,
                },
            },
            "delta_group": None,
        },
    )
    # No cell can reach 10k distinct POIs in a 4k-row frame → all inactive.
    assert float(np.asarray(model.data["amenity_msa_active"]).sum()) == 0.0


def test_random_effects_requires_terms():
    """Empty/absent terms is a construction-time error."""
    df = _re_frame(n = 50)
    with pytest.raises(ValueError, match = "terms"):
        RandomEffectsModel(dataset = df, metadata = {"dt_col": "tag_years"})


def test_random_effects_in_registry():
    assert get_model_class("random_effects") is RandomEffectsModel
