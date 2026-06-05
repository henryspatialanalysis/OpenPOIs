#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.models.metrics (predictive-fit + cross-validation).

Small synthetic frames + short NUTS runs keep the suite fast; the assertions
target structural invariants (leak-free folds, LPPD/WAIC sanity, subgroup
reconciliation) rather than tight numeric recovery.
"""
from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest

from openpois.models import metrics
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import RandomEffectsModel


def _frame(seed = 0, n = 2500):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": rng.integers(0, 900, n),
        "shared_label": rng.choice([f"a{i}" for i in range(5)], n),
        "msa_code": rng.choice(["12345", "31080", "NO_MSA"], n),
        "urban_rural": rng.choice(["urban", "suburban", "rural"], n),
        "tag_years": rng.uniform(0.2, 4.0, n),
        "changed": rng.binomial(1, 0.15, n),
        "is_first_interval": rng.binomial(1, 0.5, n).astype(bool),
    })


_META = {
    "dt_col": "tag_years",
    "terms": {
        "amenity": {"column": "shared_label"},
        "msa": {"column": "msa_code"},
        "urbanicity": {"column": "urban_rural"},
    },
    "delta_group": "shared_label",
}


def _fit(df, seed = 0):
    model = RandomEffectsModel(dataset = df, metadata = _META)
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data, target = model.target,
        num_warmup = 120, num_samples = 120, num_chains = 1,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        log_1md_fun = model.log_1md_fun,
        rng_key = jax.random.PRNGKey(seed),
    )
    fitter.fit()
    return model, fitter


def test_assign_holdout_folds_leak_free_and_balanced():
    df = _frame().reset_index(drop = True)
    folds = metrics.assign_holdout_folds(df, n_folds = 10, seed = 1)
    framed = df.assign(_f = folds.to_numpy())
    # Every POI lands in exactly one fold (no within-POI leakage).
    assert int(framed.groupby("id")["_f"].nunique().max()) == 1
    # All folds populated; sizes within a sane band.
    counts = framed["_f"].value_counts()
    assert set(counts.index) == set(range(1, 11))
    assert counts.min() > 0


def test_in_sample_metrics_sane():
    model, fitter = _fit(_frame())
    m = metrics.in_sample_metrics(model, fitter)
    assert m["n"] == len(model.raw_data)
    assert 0.0 <= m["rmse"] <= 1.0
    assert m["lppd"] < 0.0           # log density of probabilities
    assert m["p_waic"] > 0.0         # effective-parameter penalty is positive
    assert np.isfinite(m["waic"])


def test_subgroup_lppd_reconciles_to_total():
    model, fitter = _fit(_frame())
    total = metrics.in_sample_metrics(model, fitter)["lppd"]
    sub = metrics.subgroup_metrics(model, fitter)
    # Each single-factor grouping partitions the data, so its per-level LPPD
    # must sum back to the overall LPPD.
    for grouping in ["shared_label", "msa_code", "urban_rural"]:
        part = sub[sub["grouping"] == grouping]
        assert part["n"].sum() == total_rows(model)
        assert abs(part["lppd"].sum() - total) < 1e-3


def total_rows(model):
    return len(model.raw_data)


def test_cross_validate_structure():
    df = _frame().reset_index(drop = True)
    out = metrics.cross_validate(
        df, _META, n_folds = 3,
        num_warmup = 80, num_samples = 80, num_chains = 1, seed = 2,
    )
    pf = out["per_fold"]
    assert len(pf) == 3
    assert set(pf["fold"]) == {1, 2, 3}
    # Held-out rows across folds sum to the full dataset (a partition).
    assert pf["n"].sum() == len(df)
    assert np.isfinite(out["aggregate"]["rmse_oos_mean"])
    assert out["aggregate"]["lpd_oos_sum"] < 0.0
