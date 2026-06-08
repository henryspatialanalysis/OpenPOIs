#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Reconstruct per-cell change-probability curves from a fitted ``random_effects``
model, for the model-apply step.

A snapshot POI's cell ``(shared_label, msa_code, urban_rural)`` may not have
been among the model's training cells, so rather than look curves up from the
saved ``predictions.csv`` (observed cells only) we rebuild each cell's curve
directly from the posterior draws in ``param_draws.csv``::

    log λ = log_lambda_0
            + eps_amenity[label]            (if the label was seen)
            + eps_msa[msa]                  (if the MSA was seen)
            + eps_amenity_msa[(label, msa)] (if that interaction cell was kept)
            + beta_suburban / beta_rural    (urban = reference)
    logit δ = logit_delta[label]            (if δ is grouped and label seen)
              else logit_delta_0

Unseen factor levels are handled by their type:

* the **amenity x MSA interaction** (a nested effect) drops to zero for an
  unobserved cell — the POI falls back on the main effects;
* every **other** random effect (amenity main, MSA main, the per-amenity δ
  grouping) draws a fresh effect from the fitted random-effect distribution,
  ``ε_new ~ N(0, σ_term)`` per posterior draw (using the posterior draws of
  ``σ_term``). This injects both the between-group spread and the uncertainty
  in ``σ``, so a group unseen at fit time gets a *wider* credible interval than
  an observed one. The ``N(0, 1)`` draw is seeded by a stable hash of
  ``(term, group-name)`` so the same new group gets the same effect across
  batches and runs.

Probabilities are the posterior-predictive mean / credible interval over draws,
matching ``ModelFitter.predict`` on observed cells:

    conditional: P(change by t) = 1 − exp(−λ·t)
    fresh:       P(change by t) = 1 − (1−δ)·exp(−λ·t)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

PARAM_DRAWS_FILE = "param_draws.parquet"
# Legacy CSV name still accepted on read for fits made before the Parquet switch.
PARAM_DRAWS_FILE_LEGACY = "param_draws.csv"
FACTOR_LOOKUPS_FILE = "factor_lookups.csv"


def resolve_param_draws(model_dir: Path) -> Path:
    """Return the param-draws path in ``model_dir``, preferring Parquet and
    falling back to the legacy CSV. Raises ``FileNotFoundError`` if neither
    exists."""
    model_dir = Path(model_dir)
    parquet = model_dir / PARAM_DRAWS_FILE
    if parquet.exists():
        return parquet
    legacy = model_dir / PARAM_DRAWS_FILE_LEGACY
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Neither {PARAM_DRAWS_FILE} nor {PARAM_DRAWS_FILE_LEGACY} found in "
        f"{model_dir}; fit with save_full_model: true."
    )


def load_random_effects_draws(model_dir: Path) -> dict[str, np.ndarray]:
    """Load the param draws as a dict of per-column draw arrays (Parquet,
    or legacy CSV)."""
    path = resolve_param_draws(model_dir)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return {c: df[c].to_numpy() for c in df.columns}


def load_factor_maps(model_dir: Path) -> dict[str, dict]:
    """Build name → level-id maps from ``factor_lookups.csv``.

    Returns a dict with keys ``amenity``, ``msa`` (name → id), ``amenity_msa``
    (``(amenity, msa_code)`` → id), and ``delta`` (name → id). Missing factors
    yield empty maps.
    """
    fl = pd.read_csv(
        Path(model_dir) / FACTOR_LOOKUPS_FILE,
        dtype = {"msa_code": str, "amenity": str, "level_name": str},
    )

    def _name_map(factor: str) -> dict:
        sub = fl[fl["factor"] == factor]
        return dict(zip(sub["level_name"], sub["level_id"].astype(int)))

    inter = fl[fl["factor"] == "amenity_msa"]
    am_map = {
        (row.amenity, row.msa_code): int(row.level_id)
        for row in inter.itertuples()
    }
    return {
        "amenity": _name_map("amenity"),
        "msa": _name_map("msa"),
        "amenity_msa": am_map,
        # Legacy single-group δ map, plus the per-term δ maps (new schema).
        "delta": _name_map("delta"),
        "delta_amenity": _name_map("delta_amenity"),
        "delta_msa": _name_map("delta_msa"),
    }


def _stack(draws: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    """Stack the indexed draw columns ``name[0], name[1], ...`` into ``(K, S)``,
    or return None if the family is absent."""
    pat = re.compile(rf"^{re.escape(name)}\[(\d+)\]$")
    idx = {}
    for col in draws:
        m = pat.match(col)
        if m:
            idx[int(m.group(1))] = draws[col]
    if not idx:
        return None
    k = max(idx) + 1
    return np.stack([idx[i] for i in range(k)], axis = 0)  # (K, S)


def _gather(mat: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Per-cell contribution ``mat[ids]`` with ``ids < 0`` (unseen) → 0.

    ``mat`` is ``(K, S)``; ``ids`` is ``(n,)``; result is ``(n, S)``.
    """
    clamped = np.where(ids >= 0, ids, 0)
    vals = mat[clamped]  # (n, S)
    return np.where((ids >= 0)[:, None], vals, 0.0)


def _stable_seed(term: str, name: str) -> int:
    """Deterministic 64-bit seed for an unseen group's effect draw."""
    digest = hashlib.blake2b(
        f"{term}\x1f{name}".encode("utf-8"), digest_size = 8
    ).digest()
    return int.from_bytes(digest, "big")


def _unseen_contrib(
    names: np.ndarray,
    unseen: np.ndarray,
    sigma_draws: np.ndarray,
    term: str,
) -> np.ndarray:
    """
    Random-effect contribution for unseen groups: ``ε_new ~ N(0, σ_term)`` per
    posterior draw, shared across all cells of the same new group.

    Args:
        names: per-cell group name ``(n,)``.
        unseen: bool mask ``(n,)`` of cells whose group was not in training.
        sigma_draws: per-draw ``σ_term`` ``(S,)``.
        term: term label, mixed into the per-group seed.

    Returns:
        ``(n, S)`` contribution (0 for seen cells).
    """
    n = len(names)
    s = len(sigma_draws)
    out = np.zeros((n, s), dtype = np.float64)
    if not unseen.any():
        return out
    names = np.asarray(names, dtype = object)
    for name in pd.unique(names[unseen]):
        z = np.random.default_rng(_stable_seed(term, str(name))).standard_normal(s)
        rows = unseen & (names == name)
        out[rows] = (z * sigma_draws)[None, :]
    return out


def cell_log_params(
    draws: dict[str, np.ndarray],
    maps: dict[str, dict],
    cells: pd.DataFrame,
    delta_group_col: str = "shared_label",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct per-cell ``(log λ, logit δ)`` draws.

    Args:
        draws: Output of :func:`load_random_effects_draws`.
        maps: Output of :func:`load_factor_maps`.
        cells: DataFrame with ``shared_label``, ``msa_code``, ``urban_rural``.
        delta_group_col: Column δ is grouped by (default ``shared_label``).

    Returns:
        ``(log_lambda, logit_delta)`` arrays, each ``(n_cells, n_draws)``.
    """
    n = len(cells)
    s = len(draws["log_lambda_0"])
    log_lambda = np.tile(draws["log_lambda_0"], (n, 1)).astype(np.float64)

    label = cells["shared_label"].astype(str).to_numpy()
    msa = cells["msa_code"].astype(str).to_numpy()
    urban = cells["urban_rural"].astype(str).to_numpy()

    # Main effects: gather for seen levels; for unseen groups, draw from the
    # fitted random-effect distribution N(0, σ_term).
    eps_amenity = _stack(draws, "eps_amenity")
    if eps_amenity is not None:
        a_idx = np.array([maps["amenity"].get(v, -1) for v in label])
        log_lambda += _gather(eps_amenity, a_idx)
        if "log_sigma_amenity" in draws:
            log_lambda += _unseen_contrib(
                label, a_idx < 0, np.exp(draws["log_sigma_amenity"]), "amenity"
            )

    eps_msa = _stack(draws, "eps_msa")
    if eps_msa is not None:
        m_idx = np.array([maps["msa"].get(v, -1) for v in msa])
        log_lambda += _gather(eps_msa, m_idx)
        if "log_sigma_msa" in draws:
            log_lambda += _unseen_contrib(
                msa, m_idx < 0, np.exp(draws["log_sigma_msa"]), "msa"
            )

    # Interaction (nested): unobserved cells contribute zero (fall back to mains).
    eps_inter = _stack(draws, "eps_amenity_msa")
    if eps_inter is not None and maps["amenity_msa"]:
        am_idx = np.array([
            maps["amenity_msa"].get((a, m), -1) for a, m in zip(label, msa)
        ])
        log_lambda += _gather(eps_inter, am_idx)

    if "beta_urban[0]" in draws:
        is_sub = (urban == "suburban")[:, None]
        is_rural = (urban == "rural")[:, None]
        log_lambda += np.where(is_sub, draws["beta_urban[0]"][None, :], 0.0)
        log_lambda += np.where(is_rural, draws["beta_urban[1]"][None, :], 0.0)

    # δ: composable random intercepts on logit δ. Each enabled δ term adds
    # eta_<t>[level]; unseen levels draw a fresh η ~ N(0, exp(log_tau_<t>))
    # about the global intercept (same "from the distribution" fallback as the
    # λ main effects). Falls back to the legacy single-group schema when the
    # new per-term draws are absent.
    delta_term_col = {"amenity": "shared_label", "msa": "msa_code"}
    new_delta_terms = [
        t for t in delta_term_col if _stack(draws, f"eta_{t}") is not None
    ]
    if new_delta_terms:
        logit_delta = np.tile(draws["logit_delta_0"], (n, 1)).astype(np.float64)
        for t in new_delta_terms:
            eta_t = _stack(draws, f"eta_{t}")  # (K, S)
            vals = cells[delta_term_col[t]].astype(str).to_numpy()
            d_idx = np.array([maps.get(f"delta_{t}", {}).get(v, -1) for v in vals])
            logit_delta += _gather(eta_t, d_idx)
            if f"log_tau_{t}" in draws:
                logit_delta += _unseen_contrib(
                    vals, d_idx < 0, np.exp(draws[f"log_tau_{t}"]), f"delta_{t}"
                )
        return log_lambda, logit_delta

    # Legacy single-group δ schema (logit_delta + "delta" map) or global δ.
    logit_delta_grouped = _stack(draws, "logit_delta")
    delta_group_vals = cells[delta_group_col].astype(str).to_numpy()
    if logit_delta_grouped is not None:
        d_idx = np.array([maps["delta"].get(v, -1) for v in delta_group_vals])
        gathered = _gather(logit_delta_grouped, d_idx)
        seen = d_idx >= 0
        unseen_eta = (
            _unseen_contrib(
                delta_group_vals, ~seen, np.exp(draws["log_tau"]), "delta"
            )
            if "log_tau" in draws else np.zeros_like(gathered)
        )
        logit_delta = np.where(
            seen[:, None],
            gathered,
            draws["logit_delta_0"][None, :] + unseen_eta,
        )
    else:
        logit_delta = np.tile(draws["logit_delta_0"], (n, 1)).astype(np.float64)
    return log_lambda, logit_delta


def reconstruct_cell_curves(
    draws: dict[str, np.ndarray],
    maps: dict[str, dict],
    cells: pd.DataFrame,
    times: np.ndarray,
    delta_group_col: str = "shared_label",
    ui_width: float = 0.95,
    chunk: int = 256,
) -> dict[str, np.ndarray]:
    """
    Posterior-predictive change-probability curves for each cell.

    Args:
        draws, maps: fitted-model artifacts.
        cells: DataFrame of unique cells (``shared_label``/``msa_code``/
            ``urban_rural``).
        times: 1-D array of horizons (years).
        delta_group_col: column δ is grouped by.
        ui_width: central credible-interval width (default 0.95).
        chunk: cells processed per block (caps the ``cells × draws × times``
            temporary).

    Returns:
        dict of ``(n_cells, n_times)`` arrays: ``p_cond_mean/lower/upper`` and
        ``p_fresh_mean/lower/upper``.
    """
    times = np.asarray(times, dtype = np.float64)
    log_lambda, logit_delta = cell_log_params(
        draws, maps, cells, delta_group_col = delta_group_col
    )
    lam = np.exp(log_lambda)                       # (n, S)
    one_minus_delta = 1.0 - _sigmoid(logit_delta)  # (n, S)

    lb = (1.0 - ui_width) / 2.0
    ub = 1.0 - lb
    n = len(cells)
    out = {
        key: np.empty((n, len(times)), dtype = np.float64)
        for key in (
            "p_cond_mean", "p_cond_lower", "p_cond_upper",
            "p_fresh_mean", "p_fresh_lower", "p_fresh_upper",
        )
    }
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        # (c, S, T)
        rate = lam[start:end, :, None] * times[None, None, :]
        surv = np.exp(-rate)
        p_cond = 1.0 - surv
        p_fresh = 1.0 - one_minus_delta[start:end, :, None] * surv
        out["p_cond_mean"][start:end] = p_cond.mean(axis = 1)
        out["p_cond_lower"][start:end] = np.quantile(p_cond, lb, axis = 1)
        out["p_cond_upper"][start:end] = np.quantile(p_cond, ub, axis = 1)
        out["p_fresh_mean"][start:end] = p_fresh.mean(axis = 1)
        out["p_fresh_lower"][start:end] = np.quantile(p_fresh, lb, axis = 1)
        out["p_fresh_upper"][start:end] = np.quantile(p_fresh, ub, axis = 1)
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
