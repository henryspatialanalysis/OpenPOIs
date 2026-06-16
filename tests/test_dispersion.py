#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.models.dispersion (overdispersion diagnostics).

Two families of checks, both on small synthetic frames with short NUTS runs:

* **Structural** — the report has the expected shape on both model paths
  (``ConstantModel`` per-row data, ``RandomEffectsModel`` ``build_row_data``).
* **Calibration of the custom statistic** — simulate data with and without an
  injected per-POI frailty and confirm the POI-frailty diagnostic separates
  them (higher φ̂ and smaller posterior-predictive p-value under frailty). The
  assertions are comparative rather than tight-numeric, so they are robust to
  the short sampler runs.
"""
from __future__ import annotations

import jax
import numpy as np
import pandas as pd

from openpois.models import dispersion
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import ConstantModel, RandomEffectsModel


def _simulate(seed = 0, n_poi = 400, k = 4, lam = 0.25, tau = 0.0):
    """Synthetic interval data with optional per-physical-POI frailty.

    Each physical POI gets ``k`` intervals; ``changed`` is an independent
    Bernoulli with the Poisson-correct probability ``1 − exp(−λ_i·Δ)`` where
    ``λ_i = lam·exp(N(0, tau))`` is constant within a POI. ``tau = 0`` is the
    homogeneous null; ``tau > 0`` injects exactly the within-POI clustering the
    frailty statistic targets.
    """
    rng = np.random.default_rng(seed)
    labels = [f"a{i}" for i in range(5)]
    msas = ["12345", "31080", "NO_MSA"]
    urb = ["urban", "suburban", "rural"]
    ids, dts, ch, first, age = [], [], [], [], []
    lab, msa, ur = [], [], []
    for poi in range(n_poi):
        lam_i = lam * np.exp(rng.normal(0.0, tau)) if tau > 0 else lam
        poi_lab = rng.choice(labels)
        poi_msa = rng.choice(msas)
        poi_ur = rng.choice(urb)
        a = 0.0
        for j in range(k):
            dt = float(rng.uniform(0.3, 2.0))
            p = 1.0 - np.exp(-lam_i * dt)
            ids.append(poi)
            dts.append(dt)
            ch.append(int(rng.random() < p))
            first.append(j == 0)
            age.append(a)
            lab.append(poi_lab)
            msa.append(poi_msa)
            ur.append(poi_ur)
            a += dt
    return pd.DataFrame({
        "id": ids,
        "shared_label": lab,
        "msa_code": msa,
        "urban_rural": ur,
        "tag_years": dts,
        "changed": ch,
        "is_first_interval": np.asarray(first, dtype = bool),
        "age_start": age,
    })


def _fit_constant(df, seed = 0):
    model = ConstantModel(dataset = df, metadata = {"dt_col": "tag_years"})
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data, target = model.target,
        num_warmup = 150, num_samples = 150, num_chains = 1,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        rng_key = jax.random.PRNGKey(seed),
    )
    fitter.fit()
    return model, fitter


_RE_META = {
    "dt_col": "tag_years",
    "terms": {
        "amenity": {"column": "shared_label"},
        "msa": {"column": "msa_code"},
        "urbanicity": {"column": "urban_rural"},
    },
    "delta_group": "shared_label",
}


def _fit_random_effects(df, seed = 0):
    model = RandomEffectsModel(dataset = df, metadata = _RE_META)
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


def test_report_structure_constant():
    model, fitter = _fit_constant(_simulate(seed = 1))
    out = dispersion.dispersion_report(
        model, fitter, n_ppc_draws = 60, n_age_bins = 3,
    )
    summary = out["summary"]
    assert set(summary["statistic"]) == {"covariate_cell", "poi_frailty"}
    assert summary["ppp"].between(0.0, 1.0).all()
    assert (summary["phi_hat"] > 0).all()
    # Calibration table partitions all observations.
    assert out["calibration"]["n"].sum() == len(model.raw_data)
    # Subgroup breakdown covers every grouping dimension.
    assert set(out["subgroup"]["grouping"]) >= {
        "shared_label", "msa_code", "urban_rural", "age_bin"
    }
    assert out["meta"]["n_poi_multi"] > 0


def test_report_structure_random_effects():
    """The production model path goes through ``build_row_data``."""
    model, fitter = _fit_random_effects(_simulate(seed = 2))
    out = dispersion.dispersion_report(model, fitter, n_ppc_draws = 40)
    assert set(out["summary"]["statistic"]) == {"covariate_cell", "poi_frailty"}
    assert np.isfinite(out["summary"]["phi_hat"]).all()


def test_frailty_is_detected_relative_to_null():
    """Injected per-POI frailty inflates the POI φ̂ and shrinks its ppp."""
    null_model, null_fitter = _fit_constant(
        _simulate(seed = 10, tau = 0.0), seed = 10
    )
    fr_model, fr_fitter = _fit_constant(
        _simulate(seed = 10, tau = 1.0), seed = 11
    )
    null_out = dispersion.dispersion_report(
        null_model, null_fitter, n_ppc_draws = 120, seed = 0
    )
    fr_out = dispersion.dispersion_report(
        fr_model, fr_fitter, n_ppc_draws = 120, seed = 0
    )

    def _poi(out):
        row = out["summary"].set_index("statistic").loc["poi_frailty"]
        return float(row["phi_hat"]), float(row["ppp"])

    null_phi, null_ppp = _poi(null_out)
    fr_phi, fr_ppp = _poi(fr_out)

    # Frailty produces materially more within-POI clustering than the null.
    assert fr_phi > null_phi
    assert fr_phi > 1.3
    # ... and the predictive check flags it (replicates rarely as extreme).
    assert fr_ppp < 0.05
    assert fr_ppp <= null_ppp


def test_null_poi_statistic_not_flagged():
    """Under the homogeneous null the POI φ̂ sits near 1 and ppp is non-extreme."""
    model, fitter = _fit_constant(_simulate(seed = 20, tau = 0.0), seed = 20)
    out = dispersion.dispersion_report(model, fitter, n_ppc_draws = 120, seed = 1)
    row = out["summary"].set_index("statistic").loc["poi_frailty"]
    assert 0.6 < float(row["phi_hat"]) < 1.6
    assert float(row["ppp"]) > 0.05
