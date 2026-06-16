#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Run the overdispersion diagnostics against an already-fitted random-effects
model, reconstructing per-cell posterior draws off disk (no model refit, no
JAX). Drives :func:`openpois.models.dispersion.dispersion_report_from_probs`
from :func:`openpois.models.reconstruct.cell_log_params`, so the per-row change
probabilities use exactly the same parameterisation the apply step trusts.

Memory: one ``(N,)`` probability vector per draw plus per-cell ``(C, n_use)``
parameter arrays; the ``(n_use, N)`` matrix is never formed. Safe on a 16 GB
host at national scale (N ~ 10M). Stream stdout to a log:

    python -u scripts/exploratory/run_dispersion_on_fit.py \
        --model-dir ~/data/openpois/osm_turnover_model/2026-06-05-nationwide-full \
        --observations ~/data/openpois/osm_data/20260521/osm_observations.parquet \
        2>&1 | tee ~/data/openpois/logs/dispersion_2026-06-05-nationwide-full.log
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openpois.models import dispersion, reconstruct
from openpois.models.setup import prepare_data_for_model

CELL_COLS = ["shared_label", "msa_code", "urban_rural"]
LOAD_COLS = [
    "id", "shared_label", "msa_code", "urban_rural", "changed",
    "last_obs_timestamp", "obs_timestamp", "last_tag_timestamp",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush = True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required = True)
    ap.add_argument("--observations", required = True)
    ap.add_argument("--n-ppc-draws", type = int, default = 200)
    ap.add_argument("--seed", type = int, default = 0)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    obs_path = Path(args.observations).expanduser()

    _log(f"Loading observations: {obs_path}")
    df = pd.read_parquet(obs_path, columns = LOAD_COLS)
    _log(f"  raw rows: {len(df):,}")
    n_raw = len(df)
    df = prepare_data_for_model(df)
    n_valid_dt = len(df)
    df = df[df[CELL_COLS].notna().all(axis = 1)].reset_index(drop = True)
    _log(f"  after valid-Δ filter: {n_valid_dt:,} (dropped {n_raw - n_valid_dt:,} "
         f"near-zero/NaN intervals)")
    _log(f"  modelled rows (also non-null cell): {len(df):,}")
    # Free the timestamp columns we no longer need.
    keep = CELL_COLS + ["id", "changed", "tag_years", "is_first_interval",
                        "age_start"]
    df = df[keep].copy()

    _log(f"Loading posterior draws + factor maps: {model_dir.name}")
    draws_full = reconstruct.load_random_effects_draws(model_dir)
    maps = reconstruct.load_factor_maps(model_dir)
    n_avail = len(draws_full["log_lambda_0"])
    n_use = min(args.n_ppc_draws, n_avail)
    idx = np.linspace(0, n_avail - 1, n_use).round().astype(int)
    draws = {k: np.asarray(v)[idx] for k, v in draws_full.items()}
    _log(f"  draws available: {n_avail}; using {n_use} for the PPC")

    # Reconstruct per-cell (log λ, logit δ) for every unique covariate cell.
    row_cell, uniques = pd.factorize(
        pd.MultiIndex.from_frame(df[CELL_COLS].astype(str)), sort = False
    )
    uniq_df = pd.DataFrame(list(uniques), columns = CELL_COLS)
    _log(f"  unique covariate cells: {len(uniq_df):,}")
    log_lambda_cell, logit_delta_cell = reconstruct.cell_log_params(
        draws, maps, uniq_df, delta_group_col = "shared_label"
    )
    one_minus_delta_cell = 1.0 / (1.0 + np.exp(logit_delta_cell))  # = 1 − δ

    dt = df["tag_years"].to_numpy(dtype = np.float64)
    is_first = df["is_first_interval"].to_numpy().astype(bool)
    row_cell = np.asarray(row_cell)

    def get_probs(i: int) -> np.ndarray:
        """Per-row marginal P(changed=1) for posterior draw ``i``.

        ``1 − exp(−λΔ)`` on non-first intervals; ``1 − (1−δ)exp(−λΔ)`` on the
        first interval of each individual (ZIE, methodology §1.7).
        """
        rate = np.exp(log_lambda_cell[row_cell, i]) * dt
        e = np.exp(-rate)
        p_cond = 1.0 - e
        p_fresh = 1.0 - one_minus_delta_cell[row_cell, i] * e
        p = np.where(is_first, p_fresh, p_cond)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    _log("Running dispersion diagnostics...")
    report = dispersion.dispersion_report_from_probs(
        df, get_probs, n_use,
        cell_cols = tuple(CELL_COLS),
        seed = args.seed,
        n_draws_available = n_avail,
    )

    _log("=== META ===")
    for k, v in report["meta"].items():
        print(f"  {k}: {v}", flush = True)

    _log("=== SUMMARY (φ̂ ≈ 1 ⇒ well specified; ppp near 0/1 ⇒ misfit) ===")
    print(report["summary"].to_string(index = False), flush = True)

    _log("=== CALIBRATION (observed vs expected change rate by fitted-prob bin) ===")
    print(report["calibration"].to_string(index = False), flush = True)

    _log("=== SUBGROUP φ̂ — by tag-age bin (frailty vs time-varying hazard) ===")
    sub = report["subgroup"]
    print(sub[sub["grouping"] == "age_bin"].to_string(index = False), flush = True)

    _log("=== SUBGROUP φ̂ — urban_rural ===")
    print(sub[sub["grouping"] == "urban_rural"].to_string(index = False),
          flush = True)

    _log("=== SUBGROUP φ̂ — top 12 shared_label by φ̂ ===")
    lab = sub[sub["grouping"] == "shared_label"].sort_values(
        "phi_hat", ascending = False
    )
    print(lab.head(12).to_string(index = False), flush = True)

    out_dir = model_dir
    report["summary"].to_csv(out_dir / "dispersion_summary.csv", index = False)
    report["subgroup"].to_csv(out_dir / "dispersion_subgroup.csv", index = False)
    report["calibration"].to_csv(
        out_dir / "dispersion_calibration.csv", index = False
    )
    _log(f"Wrote dispersion_*.csv to {out_dir}")


if __name__ == "__main__":
    main()
