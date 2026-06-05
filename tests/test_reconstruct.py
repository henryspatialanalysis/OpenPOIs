#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.models.reconstruct.

The reconstruction must reproduce ``ModelFitter.predict`` exactly on observed
cells (it uses the same posterior draws and the same additive log-λ / ZIE-δ
math), and must degrade gracefully — via the model's active-mask back-off — on
cells whose factor levels were never seen.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from openpois.models import reconstruct
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import RandomEffectsModel


_META = {
    "dt_col": "tag_years",
    "terms": {
        "amenity": {"column": "shared_label"},
        "msa": {"column": "msa_code"},
        "amenity_msa": {"columns": ["shared_label", "msa_code"], "min_count": 5},
        "urbanicity": {"column": "urban_rural"},
    },
    "delta_group": "shared_label",
}


def _frame(seed = 0, n = 3000):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": rng.integers(0, 1200, n),
        "shared_label": rng.choice([f"a{i}" for i in range(6)], n),
        "msa_code": rng.choice(["12345", "31080", "NO_MSA"], n),
        "urban_rural": rng.choice(["urban", "suburban", "rural"], n),
        "tag_years": rng.uniform(0.2, 4.0, n),
        "changed": rng.binomial(1, 0.12, n),
        "is_first_interval": rng.binomial(1, 0.5, n).astype(bool),
    })


def _flatten(pytree):
    """Flatten a posterior pytree to {column: (S,)}, matching param_draws.csv."""
    cols = {}
    for name, arr in pytree.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            cols[name] = arr
        else:
            flat = arr.reshape(arr.shape[0], -1)
            for i in range(flat.shape[1]):
                cols[f"{name}[{i}]"] = flat[:, i]
    return cols


def _maps_from_model(model):
    fl = model.factor_lookups
    inter = fl["amenity_msa"]
    return {
        "amenity": dict(zip(
            fl["amenity"]["level_name"], fl["amenity"]["level_id"].astype(int)
        )),
        "msa": dict(zip(
            fl["msa"]["level_name"], fl["msa"]["level_id"].astype(int)
        )),
        "amenity_msa": {
            (r.amenity, r.msa_code): int(r.level_id) for r in inter.itertuples()
        },
        "delta": dict(zip(
            fl["delta"]["level_name"], fl["delta"]["level_id"].astype(int)
        )),
    }


def _fit(df):
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
        rng_key = jax.random.PRNGKey(0),
    )
    fitter.fit()
    return model, fitter


def test_reconstruct_matches_predict_on_observed_cells():
    model, fitter = _fit(_frame())
    draws = _flatten(fitter.get_parameter_draws())
    maps = _maps_from_model(model)

    cells = model.cell_lookup.sort_values("cell_id").reset_index(drop = True)
    n_cells = len(cells)
    times = np.arange(11) / 10.0

    # Model's own posterior-predictive curves.
    pred_data = model.build_predict_data(jnp.asarray(times))
    cond = fitter.predict(data = pred_data, mode = "conditional")
    fresh = fitter.predict(data = pred_data, mode = "fresh")
    cond_mean = cond["p_mean"].to_numpy().reshape(n_cells, len(times))
    fresh_mean = fresh["p_mean"].to_numpy().reshape(n_cells, len(times))

    out = reconstruct.reconstruct_cell_curves(draws, maps, cells, times)

    assert np.allclose(out["p_cond_mean"], cond_mean, atol = 1e-5)
    assert np.allclose(out["p_fresh_mean"], fresh_mean, atol = 1e-5)
    # Credible bounds should also line up (same draws, same linear quantile).
    cond_lo = cond["p_lower"].to_numpy().reshape(n_cells, len(times))
    assert np.allclose(out["p_cond_lower"], cond_lo, atol = 1e-4)


def test_reconstruct_unseen_main_effect_is_drawn_and_deterministic():
    """An unseen amenity/MSA draws its effect from N(0, σ): the result is finite,
    deterministic (hash-seeded), and has a WIDER credible interval than the
    intercept-only curve (extra between-group + σ variance)."""
    model, fitter = _fit(_frame())
    draws = _flatten(fitter.get_parameter_draws())
    maps = _maps_from_model(model)
    times = np.arange(11) / 10.0

    unseen = pd.DataFrame({
        "shared_label": ["UNSEEN_LABEL"],
        "msa_code": ["99999"],        # unseen MSA too
        "urban_rural": ["urban"],     # urban = reference → no urbanicity term
    })
    out1 = reconstruct.reconstruct_cell_curves(draws, maps, unseen, times)
    out2 = reconstruct.reconstruct_cell_curves(draws, maps, unseen, times)

    assert np.all(np.isfinite(out1["p_cond_mean"]))
    # Deterministic: the hash-seeded draw is identical across calls.
    assert np.array_equal(out1["p_cond_mean"], out2["p_cond_mean"])
    assert np.array_equal(out1["p_cond_lower"], out2["p_cond_lower"])

    # Wider interval than intercept-only at the far horizon.
    lam0 = np.exp(draws["log_lambda_0"])
    p0 = 1.0 - np.exp(-lam0[None, :] * times[:, None])  # (T, S)
    w0 = np.quantile(p0, 0.975, axis = 1) - np.quantile(p0, 0.025, axis = 1)
    w_unseen = out1["p_cond_upper"][0] - out1["p_cond_lower"][0]
    assert w_unseen[-1] > w0[-1]


def test_reconstruct_interaction_unseen_contributes_zero():
    """A seen amenity + seen MSA whose interaction cell wasn't kept gets no
    interaction term — λ matches the main-effects-only reconstruction."""
    model, fitter = _fit(_frame())
    draws = _flatten(fitter.get_parameter_draws())
    maps = _maps_from_model(model)
    times = np.arange(6) / 10.0

    # Use a SEEN amenity + SEEN MSA (so no unseen main-effect sampling fires),
    # but treat the interaction as unobserved by emptying the interaction map.
    seen_label = next(iter(maps["amenity"]))
    seen_msa = next(iter(maps["msa"]))
    maps_unseen_inter = {**maps, "amenity_msa": {}}
    cell = pd.DataFrame({
        "shared_label": [seen_label], "msa_code": [seen_msa],
        "urban_rural": ["urban"],
    })
    out = reconstruct.reconstruct_cell_curves(draws, maps_unseen_inter, cell, times)

    # Stripping the interaction family entirely must give the identical curve —
    # i.e. an unobserved interaction cell contributes exactly zero.
    draws_no_inter = {
        k: v for k, v in draws.items() if not k.startswith("eps_amenity_msa")
    }
    out_no_inter = reconstruct.reconstruct_cell_curves(
        draws_no_inter, maps_unseen_inter, cell, times
    )
    assert np.allclose(
        out["p_cond_mean"], out_no_inter["p_cond_mean"], atol = 1e-12
    )
