#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Overdispersion diagnostics for the OSM turnover model.

Background — why a *custom* check is needed
-------------------------------------------
The turnover likelihood is Bernoulli-per-interval, derived from an exponential
waiting time under a homogeneous Poisson process: for an interval of length Δ,
``P(changed = 1 | λ, Δ) = 1 − exp(−λ·Δ)`` (first-interval rows carry the ZIE δ
factor; see methodology §1.2 / §1.7). The data are therefore **ungrouped
Bernoulli (n = 1) trials**, and for ungrouped binary data overdispersion is
*undefined* — the per-row deviance/Pearson statistics carry no dispersion
information (McCullagh & Nelder 1989 §4.5). Overdispersion only becomes
observable once rows are **aggregated into replicated subpopulations**.

In a survival/exponential model, "overdispersion" is unobserved heterogeneity,
i.e. *frailty*: a latent multiplier on λ. A Gamma frailty turns the exponential
into a Lomax/Pareto marginal with a decreasing apparent hazard — the survival
analog of Poisson → Negative Binomial (Rodríguez, *Unobserved Heterogeneity*).
The honest question is: **after the amenity / MSA / urbanicity random effects and
the ZIE δ, is there residual heterogeneity in λ that the homogeneous assumption
misses?**

What this module computes
-------------------------
Three complementary discrepancy statistics, each with a posterior-predictive
Bayesian p-value (Gelman et al. *BDA3* ch. 6) and a readable dispersion ratio φ̂:

* ``cov`` — **covariate-cell Pearson dispersion.** Rows are aggregated into
  ``(shared_label, msa_code, urban_rural, age_bin)`` cells. For cell ``c`` the
  observed change count ``O_c`` has, under the model, a **Poisson-binomial**
  mean ``μ_c = Σ p_i`` and variance ``V_c = Σ p_i(1−p_i)`` (this handles the
  per-row Δ heterogeneity exactly — no duration binning needed). The χ²
  discrepancy is ``X² = Σ_c (O_c − μ_c)²/V_c``; ``φ̂ = mean_c (O_c − μ_c)²/V_c``
  (≈ 1 ⇒ well specified, > 1 ⇒ residual heterogeneity in the mean structure).
* ``poi`` — **per-physical-POI frailty statistic** (the negative-binomial-
  targeted test). The model treats each ``(POI, name-iteration)`` interval as
  independent; if some physical POIs are intrinsically volatile, change events
  cluster within ``id`` beyond independence. Computed over POIs with ≥ 2
  intervals (single-interval POIs carry no frailty signal). This is the shared-
  frailty / recurrent-event signature (Balan & Putter 2020).
* ``calibration`` — observed vs expected change rate across deciles of the
  fitted probability (Hosmer–Lemeshow / DHARMa style), to separate *mean
  misspecification* from *pure dispersion*.

A tag-age-stratified φ̂ table disambiguates frailty (dispersion ~uniform across
age) from a genuinely time-varying baseline hazard (dispersion tracks age →
the breakpoint model, not NB).

Memory / compute (designed for a 16 GB host, N ≈ 10M, S ≈ 1–4k draws)
---------------------------------------------------------------------
The full ``(S, N)`` probability matrix (≈ 40 GB at S = 1000) is **never**
formed. The diagnostics stream **one posterior draw at a time**: each iteration
materialises a single ``(N,)`` probability vector (≈ 40 MB at fp32), reduces it
to per-cell / per-POI ``bincount`` sums, computes the scalar discrepancies, and
discards it. Peak working set is O(N) plus the ``(C,)`` / ``(P,)`` accumulators
— well under 1 GB. The number of replicate draws is capped by ``n_ppc_draws``
(default 200), which is ample for a Bayesian p-value. No model refitting is done
(unlike cross-validation), so the whole report is minutes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from openpois.models.model_fitter import ModelFitter

# A covariate cell needs a non-trivial expected count before its Pearson
# residual is informative; cells below this expected-change threshold are
# dropped from the χ² aggregation (standard rule of thumb for Pearson GoF).
DEFAULT_MU_MIN = 1.0
DEFAULT_N_PPC_DRAWS = 200
DEFAULT_N_AGE_BINS = 5
DEFAULT_N_CALIBRATION_BINS = 10
DEFAULT_CELL_COLS = ("shared_label", "msa_code", "urban_rural")


def _per_row_data(model) -> dict:
    """Per-row JAX ``data`` dict aligned to ``model.target``.

    ``RandomEffectsModel`` stores cell-based sufficient statistics in
    ``model.data`` and exposes the dense per-row dict via ``build_row_data``;
    ``ConstantModel`` already keeps ``model.data`` at per-row granularity.
    """
    if hasattr(model, "build_row_data"):
        return model.build_row_data(None)
    return model.data


def _make_prob_fn(model, fitter: ModelFitter, row_data: dict) -> callable:
    """Jitted ``params -> (N,)`` marginal change probability per row.

    The per-row marginal ``P(y_i = 1)`` under the fitted model is the *fresh*
    (ZIE, ``1 − (1−δ)e^{−λΔ}``) form on first-interval rows and the conditional
    (``1 − e^{−λΔ}``) form elsewhere — see methodology §1.7. ``calculate_probs``
    already encodes both regimes (and per-row δ via ``log_1md_fun``); we select
    by ``is_first_interval`` so the simulated replicates match the data-
    generating process exactly.
    """
    is_first = row_data["is_first_interval"] > 0.5

    @jax.jit
    def prob_fn(params):
        p_cond = fitter.calculate_probs(params, row_data, mode = "conditional")
        p_fresh = fitter.calculate_probs(params, row_data, mode = "fresh")
        return jnp.where(is_first, p_fresh, p_cond)

    return prob_fn


def _draw_at(param_draws: dict, s: int) -> dict:
    """Slice posterior draw ``s`` out of the stacked ``param_draws`` pytree."""
    return {k: v[s] for k, v in param_draws.items()}


def _codes(frame: pd.DataFrame, cols: tuple[str, ...]) -> tuple[np.ndarray, pd.DataFrame]:
    """Factorise the tuple of ``cols`` into integer cell codes.

    Returns ``(codes, levels)`` where ``codes`` is length-N int32 and ``levels``
    is a ``(C, len(cols))`` frame mapping each code to its attribute values.
    """
    keys = list(cols)
    mi = pd.MultiIndex.from_frame(frame[keys].astype("object").fillna("NA"))
    codes, uniques = pd.factorize(mi, sort = False)
    levels = pd.DataFrame(
        [list(u) if isinstance(u, tuple) else [u] for u in uniques],
        columns = keys,
    )
    return codes.astype(np.int32), levels


def _age_bins(age_start: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bins of tag age (years), as string labels for grouping."""
    finite = age_start[np.isfinite(age_start)]
    if finite.size == 0:
        return np.array(["NA"] * len(age_start), dtype = object)
    try:
        binned = pd.qcut(age_start, q = n_bins, duplicates = "drop")
        return binned.astype(str).to_numpy()
    except (ValueError, IndexError):
        return np.array(["NA"] * len(age_start), dtype = object)


def dispersion_report(
    model,
    fitter: ModelFitter,
    cell_cols: tuple[str, ...] = DEFAULT_CELL_COLS,
    n_ppc_draws: int = DEFAULT_N_PPC_DRAWS,
    n_age_bins: int = DEFAULT_N_AGE_BINS,
    n_calibration_bins: int = DEFAULT_N_CALIBRATION_BINS,
    mu_min: float = DEFAULT_MU_MIN,
    min_poi_intervals: int = 2,
    id_col: str = "id",
    age_col: str = "age_start",
    seed: int = 0,
) -> dict:
    """
    Posterior-predictive overdispersion diagnostics for a fitted turnover model.

    Streams one posterior draw at a time (memory-bounded; see module docstring).
    For each of up to ``n_ppc_draws`` draws it computes, from a single ``(N,)``
    probability vector, the covariate-cell and per-POI χ² discrepancies for both
    the observed data and a simulated replicate, plus running sums for the
    posterior-mean calibration table.

    Args:
        model: A fitted model factory (``ConstantModel`` / ``RandomEffectsModel``
            / ...) exposing ``raw_data``, ``target``, and either
            ``build_row_data`` or a per-row ``data`` dict.
        fitter: The ``ModelFitter`` whose ``param_draws`` hold the posterior.
        cell_cols: Columns defining a covariate subpopulation. A tag-age bin is
            appended automatically for the age-stratified breakdown.
        n_ppc_draws: Number of posterior draws used for the predictive check
            (capped at the number available). 200 is ample for a Bayesian
            p-value and keeps the run to minutes.
        n_age_bins: Quantile bins of tag age for the disambiguation table.
        n_calibration_bins: Decile-style bins for the calibration table.
        mu_min: Minimum expected change count for a covariate cell to enter the
            χ² aggregation.
        min_poi_intervals: Minimum interval rows for a physical POI to enter the
            frailty statistic (single-interval POIs carry no signal).
        id_col, age_col: Column names for the physical POI id and tag age.
        seed: RNG seed for the replicate simulation.

    Returns:
        dict with:
          ``summary`` — one-row-per-statistic DataFrame
              ``[statistic, phi_hat, ppp, n_groups, n_obs]``.
          ``subgroup`` — long-form φ̂ by ``shared_label`` / ``urban_rural`` /
              ``msa_code`` / ``age_bin``.
          ``calibration`` — per-bin observed vs expected change rate + HL stat.
          ``meta`` — run parameters (draws used, N, cell/POI counts).
    """
    if fitter.param_draws is None:
        raise ValueError("Run fitter.fit() before computing dispersion diagnostics")

    df = model.raw_data.reset_index(drop = True)
    target = np.asarray(model.target, dtype = np.float64)
    if len(df) != len(target):
        raise ValueError(
            f"raw_data rows ({len(df)}) != target length ({len(target)}); "
            "model.raw_data must be aligned to model.target"
        )

    # The per-draw probability source is the live fitter: a jitted per-row fn
    # evaluated on each selected posterior draw.
    row_data = _per_row_data(model)
    prob_fn = _make_prob_fn(model, fitter, row_data)
    n_avail = int(jax.tree_util.tree_leaves(fitter.param_draws)[0].shape[0])
    n_use = min(n_ppc_draws, n_avail)
    draw_ids = np.linspace(0, n_avail - 1, n_use).round().astype(int)

    def get_probs(i):
        return np.asarray(
            prob_fn(_draw_at(fitter.param_draws, int(draw_ids[i]))),
            dtype = np.float64,
        )

    return dispersion_report_from_probs(
        df, get_probs, n_use,
        target = target,
        cell_cols = cell_cols, n_age_bins = n_age_bins,
        n_calibration_bins = n_calibration_bins, mu_min = mu_min,
        min_poi_intervals = min_poi_intervals, id_col = id_col,
        age_col = age_col, seed = seed, n_draws_available = n_avail,
    )


def dispersion_report_from_probs(
    df: pd.DataFrame,
    get_probs: callable,
    n_use: int,
    target: np.ndarray | None = None,
    cell_cols: tuple[str, ...] = DEFAULT_CELL_COLS,
    n_age_bins: int = DEFAULT_N_AGE_BINS,
    n_calibration_bins: int = DEFAULT_N_CALIBRATION_BINS,
    mu_min: float = DEFAULT_MU_MIN,
    min_poi_intervals: int = 2,
    id_col: str = "id",
    age_col: str = "age_start",
    seed: int = 0,
    n_draws_available: int | None = None,
) -> dict:
    """
    Model-agnostic core of :func:`dispersion_report`.

    Decoupled from the JAX fitter: the per-draw marginal change probabilities
    are supplied by ``get_probs(i) -> (N,) array`` for ``i`` in ``0..n_use-1``.
    This lets the same statistics run either from a live ``ModelFitter`` or from
    posterior draws reconstructed off disk (e.g. via
    :func:`openpois.models.reconstruct.cell_log_params`), without rebuilding the
    JAX model. ``df`` supplies the grouping columns (``cell_cols`` + id + age)
    and, when ``target`` is omitted, the observed ``changed`` outcome.

    Memory: one ``(N,)`` probability vector per draw plus the ``(C,)`` / ``(P,)``
    bincount accumulators; the ``(n_use, N)`` matrix is never formed.
    """
    df = df.reset_index(drop = True)
    if target is None:
        target = df["changed"].to_numpy()
    target = np.asarray(target, dtype = np.float64)
    n = len(target)
    if len(df) != n:
        raise ValueError(f"df rows ({len(df)}) != target length ({n})")

    # --- Grouping codes (computed once) ----------------------------------->
    age_bin = _age_bins(df[age_col].to_numpy(dtype = float), n_age_bins) \
        if age_col in df.columns else np.array(["NA"] * n, dtype = object)
    cell_frame = df[list(cell_cols)].copy()
    cell_frame["age_bin"] = age_bin
    cell_code, cell_levels = _codes(cell_frame, tuple(cell_cols) + ("age_bin",))
    n_cells = len(cell_levels)

    poi_code, _ = _codes(df, (id_col,))
    n_poi = int(poi_code.max()) + 1 if n else 0
    poi_n = np.bincount(poi_code, minlength = n_poi)
    poi_multi = poi_n >= min_poi_intervals

    # Observed counts (constant across draws).
    o_obs_cell = np.bincount(cell_code, weights = target, minlength = n_cells)
    o_obs_poi = np.bincount(poi_code, weights = target, minlength = n_poi)

    # --- Stream draws ----------------------------------------------------->
    rng = np.random.default_rng(seed)

    cov_obs = np.empty(n_use)
    cov_rep = np.empty(n_use)
    cov_ngrp = np.empty(n_use)
    poi_obs = np.empty(n_use)
    poi_rep = np.empty(n_use)
    poi_ngrp = np.empty(n_use)
    # Running per-cell mean Pearson (observed) for the subgroup/age breakdown.
    cell_pear_sum = np.zeros(n_cells)
    cell_pear_cnt = np.zeros(n_cells)
    sum_p = np.zeros(n)  # for the calibration table (posterior-mean prob)

    for i in range(n_use):
        probs = np.asarray(get_probs(i), dtype = np.float64)
        sum_p += probs
        pv = probs * (1.0 - probs)
        y_rep = (rng.random(n) < probs).astype(np.float64)

        # Covariate-cell χ² discrepancy (Poisson-binomial mean/variance).
        mu_c = np.bincount(cell_code, weights = probs, minlength = n_cells)
        v_c = np.bincount(cell_code, weights = pv, minlength = n_cells)
        o_rep_c = np.bincount(cell_code, weights = y_rep, minlength = n_cells)
        keep_c = (v_c > 0) & (mu_c >= mu_min)
        safe_vc = np.where(v_c > 0, v_c, 1)
        pear_obs_c = np.where(keep_c, (o_obs_cell - mu_c) ** 2 / safe_vc, 0.0)
        pear_rep_c = np.where(keep_c, (o_rep_c - mu_c) ** 2 / safe_vc, 0.0)
        cov_obs[i] = pear_obs_c[keep_c].sum()
        cov_rep[i] = pear_rep_c[keep_c].sum()
        cov_ngrp[i] = int(keep_c.sum())
        cell_pear_sum[keep_c] += pear_obs_c[keep_c]
        cell_pear_cnt[keep_c] += 1

        # Per-POI frailty discrepancy (multi-interval POIs only).
        mu_p = np.bincount(poi_code, weights = probs, minlength = n_poi)
        v_p = np.bincount(poi_code, weights = pv, minlength = n_poi)
        o_rep_p = np.bincount(poi_code, weights = y_rep, minlength = n_poi)
        keep_p = poi_multi & (v_p > 0)
        safe_vp = np.where(v_p > 0, v_p, 1)
        pear_obs_p = np.where(keep_p, (o_obs_poi - mu_p) ** 2 / safe_vp, 0.0)
        pear_rep_p = np.where(keep_p, (o_rep_p - mu_p) ** 2 / safe_vp, 0.0)
        poi_obs[i] = pear_obs_p[keep_p].sum()
        poi_rep[i] = pear_rep_p[keep_p].sum()
        poi_ngrp[i] = int(keep_p.sum())

    # --- Reduce ----------------------------------------------------------->
    def _phi(obs, ngrp):
        ok = ngrp > 0
        return float(np.mean(obs[ok] / ngrp[ok])) if ok.any() else float("nan")

    summary = pd.DataFrame([
        {
            "statistic": "covariate_cell",
            "phi_hat": _phi(cov_obs, cov_ngrp),
            "ppp": float(np.mean(cov_rep >= cov_obs)),
            "n_groups": int(np.median(cov_ngrp)),
            "n_obs": n,
        },
        {
            "statistic": "poi_frailty",
            "phi_hat": _phi(poi_obs, poi_ngrp),
            "ppp": float(np.mean(poi_rep >= poi_obs)),
            "n_groups": int(np.median(poi_ngrp)),
            "n_obs": int(poi_multi[poi_code].sum()),
        },
    ])

    subgroup = _subgroup_phi(cell_levels, cell_pear_sum, cell_pear_cnt, cell_cols)
    calibration = _calibration_table(
        sum_p / max(n_use, 1), target, n_calibration_bins
    )

    meta = {
        "n_obs": n,
        "n_cells_total": n_cells,
        "n_poi_total": n_poi,
        "n_poi_multi": int(poi_multi.sum()),
        "n_ppc_draws": n_use,
        "n_draws_available": n_draws_available if n_draws_available else n_use,
        "mu_min": mu_min,
        "min_poi_intervals": min_poi_intervals,
    }
    return {
        "summary": summary,
        "subgroup": subgroup,
        "calibration": calibration,
        "meta": meta,
    }


def _subgroup_phi(
    cell_levels: pd.DataFrame,
    cell_pear_sum: np.ndarray,
    cell_pear_cnt: np.ndarray,
    cell_cols: tuple[str, ...],
) -> pd.DataFrame:
    """φ̂ aggregated by each cell attribute (posterior-mean per-cell Pearson)."""
    has = cell_pear_cnt > 0
    cell_phi = np.where(has, cell_pear_sum / np.where(has, cell_pear_cnt, 1), np.nan)
    levels = cell_levels.copy()
    levels["_phi"] = cell_phi
    levels["_has"] = has
    rows = []
    for col in tuple(cell_cols) + ("age_bin",):
        sub = levels[levels["_has"]]
        for level, g in sub.groupby(col, observed = True):
            rows.append({
                "grouping": col,
                "level": str(level),
                "n_cells": int(len(g)),
                "phi_hat": float(np.mean(g["_phi"])),
            })
    return pd.DataFrame(rows)


def _calibration_table(
    p_mean: np.ndarray, target: np.ndarray, n_bins: int
) -> pd.DataFrame:
    """Observed vs expected change rate across fitted-probability bins.

    Includes a Hosmer–Lemeshow-style χ² contribution per bin; their sum is the
    HL statistic (large ⇒ mean miscalibration, distinct from dispersion).
    """
    try:
        bins = pd.qcut(p_mean, q = n_bins, duplicates = "drop")
    except (ValueError, IndexError):
        bins = pd.cut(p_mean, bins = min(n_bins, 2))
    frame = pd.DataFrame({"bin": bins, "p": p_mean, "y": target})
    rows = []
    for label, g in frame.groupby("bin", observed = True):
        n_b = len(g)
        exp_rate = float(g["p"].mean())
        obs_rate = float(g["y"].mean())
        exp_count = exp_rate * n_b
        denom = exp_count * (1.0 - exp_rate)
        hl = ((g["y"].sum() - exp_count) ** 2 / denom) if denom > 0 else np.nan
        rows.append({
            "bin": str(label),
            "n": n_b,
            "expected_rate": exp_rate,
            "observed_rate": obs_rate,
            "hl_contribution": float(hl) if np.isfinite(hl) else np.nan,
        })
    table = pd.DataFrame(rows)
    return table
