#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Cross-validation and predictive-fit metrics for the OSM turnover models.

Provides three families of diagnostics, all built on a per-observation,
per-draw log-likelihood matrix (the common input to LPPD / WAIC):

1. In-sample RMSE, log pointwise predictive density (LPPD), and WAIC
   (:func:`in_sample_metrics`).
2. The same broken down by subgroup — amenity, MSA, urban/rural, and their
   three-way cross-section — with LPD normalised by subgroup size
   (:func:`subgroup_metrics`).
3. Out-of-sample RMSE and LPD via a structured, per-POI-stratified 10-fold
   holdout (:func:`assign_holdout_folds`, :func:`cross_validate`).

Methodology follows
https://henryspatialanalysis.github.io/mbg/articles/model-comparison.html, with
two project-specific choices: folds are assigned at the **individual POI** level
(all of a POI's interval rows share a fold, avoiding leakage) and **stratified**
across (MSA × amenity × urban/rural) cells so each fold is balanced.

The pointwise log-likelihood always uses the model's dense per-row formula
(``pointwise_log_likelihood``); the sufficient-statistics fit path cannot
produce pointwise values. To stay memory-bounded at national scale the
per-observation reductions are computed in row-blocks rather than materialising
the full ``(draws, N)`` matrix.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import logsumexp

from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import get_model_class
from openpois.models.setup import prepare_data_for_model


def _row_data_and_target(model, df):
    """Per-row JAX data dict + target array for a frame (or the model's own
    rows when ``df is None``)."""
    data = model.build_row_data(df)
    if df is None:
        target = jnp.asarray(model.target, dtype = jnp.float32)
    else:
        target = jnp.asarray(df["changed"].to_numpy(), dtype = jnp.float32)
    return data, target


def per_observation_stats(
    model,
    fitter: ModelFitter,
    df: pd.DataFrame | None = None,
    chunk: int = 200_000,
) -> dict[str, np.ndarray]:
    """
    Per-observation predictive statistics, computed in row-blocks.

    Returns a dict of length-N arrays:
        ``lppd``    — log pointwise predictive density per obs,
                      ``log mean_s p(y_i | θ_s)``.
        ``var_ll``  — sample variance over draws of the pointwise log-lik
                      (the per-obs WAIC penalty term).
        ``p_mean``  — posterior-mean conditional P(change at Δ_i).
        ``target``  — observed 0/1 outcome.
    """
    data, target = _row_data_and_target(model, df)
    draws = fitter.param_draws
    n = int(target.shape[0])

    @jax.jit
    def block_stats(sub_data, sub_target):
        ll = jax.vmap(
            lambda p: model.pointwise_log_likelihood(p, sub_data, sub_target)
        )(draws)  # (S, b)
        probs = jax.vmap(
            lambda p: fitter.calculate_probs(p, sub_data, mode = "conditional")
        )(draws)  # (S, b)
        s = ll.shape[0]
        lppd = logsumexp(ll, axis = 0) - jnp.log(s)
        var_ll = jnp.var(ll, axis = 0, ddof = 1)
        return lppd, var_ll, jnp.mean(probs, axis = 0)

    lppd = np.empty(n, dtype = np.float64)
    var_ll = np.empty(n, dtype = np.float64)
    p_mean = np.empty(n, dtype = np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = {k: v[start:end] for k, v in data.items()}
        b_lppd, b_var, b_pmean = block_stats(sub, target[start:end])
        lppd[start:end] = np.asarray(b_lppd, dtype = np.float64)
        var_ll[start:end] = np.asarray(b_var, dtype = np.float64)
        p_mean[start:end] = np.asarray(b_pmean, dtype = np.float64)
    return {
        "lppd": lppd,
        "var_ll": var_ll,
        "p_mean": p_mean,
        "target": np.asarray(target, dtype = np.float64),
    }


def _summarize(lppd: np.ndarray, var_ll: np.ndarray, p_mean: np.ndarray,
               target: np.ndarray) -> dict[str, float]:
    """RMSE / LPPD / WAIC summary over a set of per-observation stats."""
    n = len(target)
    rmse = float(np.sqrt(np.mean((target - p_mean) ** 2)))
    lppd_sum = float(np.sum(lppd))
    p_waic = float(np.sum(var_ll))
    elpd_waic = lppd_sum - p_waic
    return {
        "n": n,
        "rmse": rmse,
        "lppd": lppd_sum,
        "lppd_per_obs": lppd_sum / n if n else float("nan"),
        "p_waic": p_waic,
        "elpd_waic": elpd_waic,
        # Article convention: WAIC on the deviance scale (positive; lower better).
        "waic": -2.0 * elpd_waic,
    }


def in_sample_metrics(
    model,
    fitter: ModelFitter,
    chunk: int = 200_000,
) -> dict[str, float]:
    """In-sample RMSE, LPPD, p_waic, and WAIC over all observations."""
    stats = per_observation_stats(model, fitter, df = None, chunk = chunk)
    return _summarize(
        stats["lppd"], stats["var_ll"], stats["p_mean"], stats["target"]
    )


def _aggregate_subgroups(
    frame: pd.DataFrame,
    by: tuple[str, ...],
    include_cross: bool = True,
) -> pd.DataFrame:
    """
    Aggregate a per-observation stats ``frame`` into per-subgroup RMSE / LPD /
    WAIC rows, one block per column in ``by`` plus (optionally) their
    cross-section.

    ``frame`` must carry ``lppd``, ``var_ll``, ``sq_err`` and every column in
    ``by``. Returns a long-form DataFrame with columns ``[grouping, level, n,
    rmse, lppd, lppd_per_obs, p_waic, elpd_waic, waic]``.
    """
    def _agg(group_cols: list[str], label: str) -> pd.DataFrame:
        rows = []
        for level, g in frame.groupby(group_cols, observed = True):
            n = len(g)
            lppd_sum = float(g["lppd"].sum())
            p_waic = float(g["var_ll"].sum())
            elpd = lppd_sum - p_waic
            rows.append({
                "grouping": label,
                "level": level if isinstance(level, str) else " | ".join(map(str, level)),
                "n": n,
                "rmse": float(np.sqrt(g["sq_err"].mean())),
                "lppd": lppd_sum,
                "lppd_per_obs": lppd_sum / n,
                "p_waic": p_waic,
                "elpd_waic": elpd,
                "waic": -2.0 * elpd,
            })
        return pd.DataFrame(rows)

    parts = [_agg([col], col) for col in by]
    if include_cross and len(by) > 1:
        parts.append(_agg(list(by), " x ".join(by)))
    return pd.concat(parts, ignore_index = True)


def subgroup_metrics(
    model,
    fitter: ModelFitter,
    by: tuple[str, ...] = ("shared_label", "msa_code", "urban_rural"),
    include_cross: bool = True,
    chunk: int = 200_000,
) -> pd.DataFrame:
    """
    RMSE / LPD / WAIC per subgroup, with LPD normalised by subgroup size.

    Computes the heavy per-observation stats once, then aggregates by each
    column in ``by`` plus (optionally) their cross-section. Returns a long-form
    DataFrame with columns ``[grouping, level, n, rmse, lppd, lppd_per_obs,
    p_waic, elpd_waic, waic]``.
    """
    stats = per_observation_stats(model, fitter, df = None, chunk = chunk)
    base = model.raw_data.reset_index(drop = True)
    frame = pd.DataFrame({
        "lppd": stats["lppd"],
        "var_ll": stats["var_ll"],
        "sq_err": (stats["target"] - stats["p_mean"]) ** 2,
        "target": stats["target"],
        "p_mean": stats["p_mean"],
    })
    for col in by:
        frame[col] = base[col].to_numpy()
    return _aggregate_subgroups(frame, by, include_cross = include_cross)


# Cross-validation ----------------------------------------------------------->


def assign_holdout_folds(
    df: pd.DataFrame,
    n_folds: int = 10,
    strata: tuple[str, ...] = ("msa_code", "shared_label", "urban_rural"),
    individual_col: str = "id",
    seed: int = 0,
) -> pd.Series:
    """
    Assign each row a holdout fold id in ``1..n_folds``.

    Whole POIs (``individual_col``) are kept together — all of a POI's interval
    rows land in the same fold — so the holdout has no within-POI leakage.
    Within each stratum (the unique combination of ``strata``, taken from each
    POI's first row) the POIs are shuffled and dealt round-robin across folds,
    keeping every fold balanced across the subgroup structure.

    Returns:
        Series of fold ids aligned to ``df.index``.
    """
    rng = np.random.default_rng(seed)
    first = (
        df.reset_index(drop = True)
        .groupby(individual_col, observed = True)[list(strata)]
        .first()
    )
    fold_of_id: dict = {}
    for _, stratum_ids in first.groupby(list(strata), observed = True):
        ids = stratum_ids.index.to_numpy().copy()
        rng.shuffle(ids)
        # Round-robin from a random starting fold so small strata don't all
        # pile onto fold 1.
        offset = int(rng.integers(0, n_folds))
        for i, poi_id in enumerate(ids):
            fold_of_id[poi_id] = (i + offset) % n_folds + 1
    return df[individual_col].map(fold_of_id).astype(int)


def _fit_for_fold(
    model,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    rng_key,
) -> ModelFitter:
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_warmup = num_warmup,
        num_samples = num_samples,
        num_chains = num_chains,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        log_1md_fun = getattr(model, "log_1md_fun", None),
        rng_key = rng_key,
    )
    fitter.fit()
    return fitter


def cross_validate(
    observations_df: pd.DataFrame,
    metadata: dict,
    model_name: str = "random_effects",
    n_folds: int = 10,
    num_warmup: int = 400,
    num_samples: int = 400,
    num_chains: int = 2,
    strata: tuple[str, ...] = ("msa_code", "shared_label", "urban_rural"),
    seed: int = 0,
    prepared: bool = True,
    fold_ids: np.ndarray | pd.Series | None = None,
    subgroup_by: tuple[str, ...] | None = None,
) -> dict:
    """
    Structured per-POI 10-fold cross-validation.

    For each fold, the model is refit on the other folds and evaluated on the
    held-out fold (held-out rows are mapped onto the trained factor codings;
    unseen levels back off via the model's per-term active masks). Reports
    held-out RMSE and LPD per fold plus aggregates (mean RMSE, summed LPD).

    Args:
        observations_df: Observations. If ``prepared`` is False, it is passed
            through :func:`prepare_data_for_model` first.
        metadata: Model metadata dict (terms, priors, delta_group, ...).
        model_name: Registry key (must expose ``build_row_data(df)``).
        n_folds, num_warmup, num_samples, num_chains: CV + NUTS controls.
        strata: Stratification columns for fold assignment.
        seed: RNG seed for folds + NUTS.
        prepared: Whether ``observations_df`` already has tag_years /
            is_first_interval (skip ``prepare_data_for_model``).
        fold_ids: Precomputed fold id per row (``1..n_folds``), aligned to
            ``observations_df`` row order. When given, fold assignment is taken
            from here instead of calling :func:`assign_holdout_folds` — the way
            to score several model specifications on one fixed holdout set.
        subgroup_by: When given, also pool the held-out per-observation stats
            across folds and aggregate them by these columns, returned under
            ``per_subgroup``. This is a genuine out-of-sample subgroup
            breakdown (every observation scored from a model that never saw it).

    Returns:
        dict with ``per_fold`` (DataFrame), ``aggregate`` (dict), and — when
        ``subgroup_by`` is given — ``per_subgroup`` (DataFrame).
    """
    model_cls = get_model_class(model_name)
    df = observations_df if prepared else prepare_data_for_model(observations_df)
    df = df.reset_index(drop = True)
    if fold_ids is not None:
        fold_arr = np.asarray(
            fold_ids.to_numpy() if isinstance(fold_ids, pd.Series) else fold_ids
        )
        if len(fold_arr) != len(df):
            raise ValueError(
                f"fold_ids length {len(fold_arr)} != observations {len(df)}"
            )
        df = df.assign(_fold = fold_arr.astype(int))
    else:
        folds = assign_holdout_folds(
            df, n_folds = n_folds, strata = strata, seed = seed,
        )
        df = df.assign(_fold = folds.to_numpy())

    per_fold = []
    pooled_frames: list[pd.DataFrame] = []
    for k in range(1, n_folds + 1):
        train = df[df["_fold"] != k]
        test = df[df["_fold"] == k]
        if test.empty or train.empty:
            continue
        model = model_cls(dataset = train, metadata = metadata)
        fitter = _fit_for_fold(
            model, num_warmup, num_samples, num_chains,
            jax.random.PRNGKey(seed + k),
        )
        stats = per_observation_stats(model, fitter, df = test)
        summary = _summarize(
            stats["lppd"], stats["var_ll"], stats["p_mean"], stats["target"]
        )
        summary["fold"] = k
        per_fold.append(summary)
        if subgroup_by is not None:
            pooled = pd.DataFrame({
                "lppd": stats["lppd"],
                "var_ll": stats["var_ll"],
                "sq_err": (stats["target"] - stats["p_mean"]) ** 2,
            })
            for col in subgroup_by:
                pooled[col] = test[col].to_numpy()
            pooled_frames.append(pooled)

    per_fold_df = pd.DataFrame(per_fold)
    aggregate = {
        "rmse_oos_mean": float(per_fold_df["rmse"].mean()),
        "lpd_oos_sum": float(per_fold_df["lppd"].sum()),
        "lpd_oos_per_obs": float(
            per_fold_df["lppd"].sum() / per_fold_df["n"].sum()
        ),
        "n_folds": int(len(per_fold_df)),
    }
    result = {"per_fold": per_fold_df, "aggregate": aggregate}
    if subgroup_by is not None:
        pooled_all = pd.concat(pooled_frames, ignore_index = True)
        result["per_subgroup"] = _aggregate_subgroups(
            pooled_all, subgroup_by, include_cross = False
        )
    return result
