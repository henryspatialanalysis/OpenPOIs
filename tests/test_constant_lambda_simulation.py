#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Simulation-based recovery tests for ``ConstantModel`` with the ZIE extension.

Per turnover-model-methodology.md §3. Each test simulates observation
sequences from a known (λ, δ) truth, fits ``ConstantModel``, and asserts
that the posterior recovers the truth within loose tolerances. Recovery
is noisy at unit-test sampling budgets (few hundred draws), so bounds are
set to catch outright regressions, not sampling noise.

Also covers the regression test for the inter-observation-Δ data-prep
fix (§ Context, Bug 1): ``test_multi_observation_individuals_recover_lambda``
would produce a downward-biased λ̂ under the old ``t1_col =
'last_tag_timestamp'`` default, and covers λ truth only under the fixed
per-row inter-observation Δ.
"""
from __future__ import annotations

import jax.random as jrd
import numpy as np
import pandas as pd
import pytest

from openpois.models.jax_core import jax_rng
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import ConstantModel
from openpois.models.setup import prepare_data_for_model


NUM_DRAWS = 500


def _simulate_zie_frame(
    key,
    n_indiv: int,
    true_log_lambda: float,
    true_delta: float,
    dt_range: tuple[float, float] = (0.5, 3.0),
    max_obs: int = 5,
    first_interval_dt_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Simulate per-individual observation sequences under a ZIE(λ, δ) mixture.

    For each individual:
      * With probability δ, individual is in the δ-component: a single
        first-interval row with y=1 is emitted.
      * Otherwise, per-interval Bernoulli(1 − exp(−λ·Δ_k)); stop at the first
        y=1 (terminal event) or right-censor after ``max_obs`` intervals.

    The first row per individual carries ``is_first_interval = True`` so the
    ZIE log-likelihood's first-interval branch is exercised. Subsequent
    rows use the pure exponential branch (methodology §1.7).

    Args:
        first_interval_dt_range: If provided, overrides ``dt_range`` for the
            first interval of each individual. Useful for identifiability
            tests that require variation in Δ_1.
    """
    lam = float(np.exp(true_log_lambda))
    seed = int(jrd.randint(key, (1,), 0, 2**31 - 1)[0])
    rng = np.random.default_rng(seed)

    rows_dt = []
    rows_y = []
    rows_first = []

    for _ in range(n_indiv):
        is_delta = rng.random() < true_delta
        if is_delta:
            # Single first-interval row with y=1.
            dt_first = (
                rng.uniform(*first_interval_dt_range)
                if first_interval_dt_range is not None
                else rng.uniform(*dt_range)
            )
            rows_dt.append(dt_first)
            rows_y.append(1)
            rows_first.append(True)
            continue

        # Exponential component: simulate per-interval Bernoullis until an
        # event or censoring.
        n_obs = rng.integers(1, max_obs + 1)
        for k in range(n_obs):
            if k == 0 and first_interval_dt_range is not None:
                dt_k = rng.uniform(*first_interval_dt_range)
            else:
                dt_k = rng.uniform(*dt_range)
            p = 1.0 - np.exp(-lam * dt_k)
            y = int(rng.random() < p)
            rows_dt.append(dt_k)
            rows_y.append(y)
            rows_first.append(k == 0)
            if y == 1:
                break  # terminal event

    return pd.DataFrame({
        "tag_years": np.asarray(rows_dt, dtype = np.float32),
        "changed": np.asarray(rows_y, dtype = np.int32),
        "is_first_interval": np.asarray(rows_first, dtype = bool),
    })


def _fit_constant(df: pd.DataFrame, key) -> ModelFitter:
    model = ConstantModel(
        dataset = df,
        metadata = {"dt_col": "tag_years"},
    )
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_draws = NUM_DRAWS,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        rng_key = key,
    )
    fitter.fit()
    return fitter


def _posterior_stats(
    fitter: ModelFitter, name: str,
) -> dict[str, float]:
    draws = np.asarray(fitter.get_parameter_draws()[name])
    return {
        "mean": float(np.mean(draws)),
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
        "q05": float(np.quantile(draws, 0.05)),
        "q95": float(np.quantile(draws, 0.95)),
    }


@pytest.mark.model
def test_zie_recovery_lambda_and_delta():
    """Recover both λ and δ from a frame with a nontrivial δ-mass (≈5 %)."""
    true_log_lambda = np.log(0.05)
    true_delta = 0.05
    true_logit_delta = float(np.log(true_delta / (1.0 - true_delta)))
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_zie_frame(
        key_sim,
        n_indiv = 3_000,
        true_log_lambda = true_log_lambda,
        true_delta = true_delta,
    )
    fitter = _fit_constant(df, key_fit)

    ll = _posterior_stats(fitter, "log_lambda")
    ld = _posterior_stats(fitter, "logit_delta")

    assert ll["q025"] <= true_log_lambda <= ll["q975"], (
        f"true log_lambda {true_log_lambda:+.3f} not covered by "
        f"[{ll['q025']:+.3f}, {ll['q975']:+.3f}]"
    )
    assert abs(ll["mean"] - true_log_lambda) < 0.3

    assert ld["q025"] <= true_logit_delta <= ld["q975"], (
        f"true logit_delta {true_logit_delta:+.3f} not covered by "
        f"[{ld['q025']:+.3f}, {ld['q975']:+.3f}]"
    )
    assert abs(ld["mean"] - true_logit_delta) < 0.8


@pytest.mark.model
def test_zie_delta_zero_concentrates_near_zero():
    """With δ_true=0, the posterior upper 97.5 % on δ piles up near zero."""
    true_log_lambda = np.log(0.05)
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_zie_frame(
        key_sim,
        n_indiv = 2_000,
        true_log_lambda = true_log_lambda,
        true_delta = 0.0,
    )
    fitter = _fit_constant(df, key_fit)

    delta_draws = np.asarray(fitter.get_parameter_draws()["delta"])
    # Prior mean ≈ sigmoid(-3) ≈ 0.047; with no δ-mass in the data the upper
    # tail should stay tight against zero.
    assert float(np.quantile(delta_draws, 0.975)) < 0.05, (
        f"delta posterior q97.5 = {np.quantile(delta_draws, 0.975):.4f} — "
        "did not concentrate near zero under δ_true = 0"
    )


@pytest.mark.model
def test_zie_lambda_not_inflated_by_delta():
    """δ absorbs instant-change mass rather than inflating λ."""
    true_log_lambda = np.log(0.05)
    true_delta = 0.15
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    df = _simulate_zie_frame(
        key_sim,
        n_indiv = 3_000,
        true_log_lambda = true_log_lambda,
        true_delta = true_delta,
    )
    fitter = _fit_constant(df, key_fit)

    ll = _posterior_stats(fitter, "log_lambda")
    # Without δ, the instant-change mass would push λ̂ well above truth —
    # tighter than 0.3 would over-specify for NUM_DRAWS=500, but the point is
    # that log_lambda stays near the true rate, not ≈ log(0.05 + 0.15).
    assert abs(ll["mean"] - true_log_lambda) < 0.3


@pytest.mark.model
def test_zie_identifiability_under_varying_first_interval():
    """Varying Δ_1 across individuals produces a sharp posterior on logit_delta.

    The δ-vs-λ split is identified by comparing the rate of first-interval
    change across different Δ_1 values (methodology §1.7). With Δ_1 spanning
    a wide range the posterior should be tight; with Δ_1 constant it can
    only shrink back to the prior.
    """
    true_log_lambda = np.log(0.05)
    true_delta = 0.1
    key = jax_rng()
    key_sim_a, key_sim_b, key_fit_a, key_fit_b = jrd.split(key, 4)

    df_varying = _simulate_zie_frame(
        key_sim_a,
        n_indiv = 2_000,
        true_log_lambda = true_log_lambda,
        true_delta = true_delta,
        first_interval_dt_range = (0.25, 6.0),
    )
    fitter_a = _fit_constant(df_varying, key_fit_a)
    stats_varying = _posterior_stats(fitter_a, "logit_delta")
    width_varying = stats_varying["q95"] - stats_varying["q05"]
    assert width_varying < 1.5, (
        f"logit_delta 90 % CI width {width_varying:.2f} too wide under "
        "varying Δ_1 — δ should be well-identified here"
    )


def _simulate_raw_observations(
    key,
    n_indiv: int,
    true_log_lambda: float,
    dt_range: tuple[float, float] = (0.5, 2.0),
    max_obs: int = 5,
    anchor_date: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Emit raw per-version observation rows like the state machine would.

    One row per observation after the individual's tag event, with the
    timestamp columns consumed by ``prepare_data_for_model``:
    ``obs_timestamp``, ``last_obs_timestamp``, ``last_tag_timestamp``. Runs
    pure Exponential(λ) per-interval Bernoullis and stops at the first
    change (terminal) or right-censors after ``max_obs`` intervals.
    """
    lam = float(np.exp(true_log_lambda))
    seed = int(jrd.randint(key, (1,), 0, 2**31 - 1)[0])
    rng = np.random.default_rng(seed)
    anchor = pd.Timestamp(anchor_date)

    rows = []
    for indiv_id in range(n_indiv):
        t_tag = anchor + pd.Timedelta(days = int(rng.uniform(0, 30)))
        t_prev = t_tag
        n_obs = rng.integers(2, max_obs + 1)  # ≥ 2 → guaranteed multi-row
        for k in range(n_obs):
            dt_years = rng.uniform(*dt_range)
            t_obs = t_prev + pd.Timedelta(days = int(dt_years * 365))
            p = 1.0 - np.exp(-lam * dt_years)
            y = int(rng.random() < p)
            rows.append({
                "id": indiv_id,
                "obs_timestamp": t_obs,
                "last_obs_timestamp": t_prev,
                "last_tag_timestamp": t_tag,
                "changed": y,
            })
            if y == 1:
                break
            t_prev = t_obs

    return pd.DataFrame(rows)


@pytest.mark.model
def test_multi_observation_individuals_recover_lambda():
    """Regression for the inter-observation-Δ data-prep bug (§1.2).

    Simulates pure Exponential(λ) (δ=0) with 2–5 observations per
    individual, then runs the raw per-version frame through
    ``prepare_data_for_model``. Under the old ``t1_col =
    'last_tag_timestamp'`` default the per-row Δ would be cumulative since
    the individual's start and λ̂ would be biased downward. With the fixed
    ``last_obs_timestamp`` default the 95 % posterior covers truth.
    """
    true_log_lambda = np.log(0.15)
    key = jax_rng()
    key_sim, key_fit = jrd.split(key)
    raw = _simulate_raw_observations(
        key_sim,
        n_indiv = 1_500,
        true_log_lambda = true_log_lambda,
        dt_range = (0.5, 2.0),
        max_obs = 5,
    )
    df = prepare_data_for_model(raw)
    # Confirm the prepared frame contains multi-observation individuals.
    assert int((~df["is_first_interval"]).sum()) > 0.3 * len(df), (
        "simulator produced too few non-first-interval rows — cannot "
        "exercise the inter-observation Δ bug"
    )

    fitter = _fit_constant(df, key_fit)
    ll = _posterior_stats(fitter, "log_lambda")
    assert ll["q025"] <= true_log_lambda <= ll["q975"], (
        f"true log_lambda {true_log_lambda:+.3f} not covered by "
        f"[{ll['q025']:+.3f}, {ll['q975']:+.3f}] — did the inter-observation "
        "Δ fix regress?"
    )
    assert abs(ll["mean"] - true_log_lambda) < 0.3
