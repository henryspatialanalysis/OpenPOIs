#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Experiment: re-fit the two-rate `constant_breakpoint` turnover model with the
breakpoint-time prior re-centred away from zero (median t_B ~ 4 yr), then run
the overdispersion diagnostics and compare the tag-age φ̂ gradient against the
production random-effects fit (which has no age term: φ̂ ≈ 5.3 → 8.2 → 105.7).

Motivation: the earlier breakpoint fit used `log_t_breakpoint ~ N(0, 1)`
(median 1 yr, with prior mass near zero); t_B collapsed to ~10 days and became
confounded with the ZIE δ instant-change mass. Re-centring the prior on a
multi-year breakpoint, with negligible mass near zero, tests whether a properly
identified time-varying hazard absorbs the age-driven overdispersion.

The 4 global parameters are fit on a random subsample (they are population-level,
so a subsample is statistically ample and keeps the dense NUTS run fast and
memory-light); the diagnostic then runs on the full modelled population. A
constant-λ (no age term) baseline is reported on the same rows for contrast.

    python -u scripts/exploratory/refit_breakpoint_and_diagnose.py \
        --observations ~/data/openpois/osm_data/20260521/osm_observations.parquet \
        2>&1 | tee ~/data/openpois/logs/breakpoint_refit_diagnose.log
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd

from openpois.models import dispersion
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import ConstantBreakpointModel
from openpois.models.setup import prepare_data_for_model

CELL_COLS = ["shared_label", "msa_code", "urban_rural"]
LOAD_COLS = [
    "id", "shared_label", "msa_code", "urban_rural", "changed",
    "last_obs_timestamp", "obs_timestamp", "last_tag_timestamp",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush = True)


def _age_gradient(report: dict) -> pd.DataFrame:
    sub = report["subgroup"]
    return sub[sub["grouping"] == "age_bin"].reset_index(drop = True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", required = True)
    ap.add_argument("--tb-median", type = float, default = 4.0,
                    help = "Prior median for t_breakpoint (years).")
    ap.add_argument("--tb-scale", type = float, default = 0.5,
                    help = "Prior log-scale for t_breakpoint.")
    ap.add_argument("--fit-sample", type = int, default = 1_500_000)
    ap.add_argument("--n-ppc-draws", type = int, default = 200)
    ap.add_argument("--n-warmup", type = int, default = 400)
    ap.add_argument("--n-samples", type = int, default = 400)
    ap.add_argument("--n-chains", type = int, default = 2)
    ap.add_argument("--seed", type = int, default = 0)
    args = ap.parse_args()

    tb_loc = float(np.log(args.tb_median))
    lo, hi = np.exp(tb_loc - 2 * args.tb_scale), np.exp(tb_loc + 2 * args.tb_scale)
    _log(f"t_breakpoint prior: log_t_B ~ N({tb_loc:.3f}, {args.tb_scale}) "
         f"⇒ median {args.tb_median} yr, ~95% in [{lo:.2f}, {hi:.2f}] yr")
    metadata = {
        "dt_col": "tag_years",
        "t_breakpoint_prior": [tb_loc, args.tb_scale],
    }

    _log(f"Loading observations: {args.observations}")
    df = pd.read_parquet(Path(args.observations).expanduser(), columns = LOAD_COLS)
    df = prepare_data_for_model(df)
    df = df[df[CELL_COLS].notna().all(axis = 1)].reset_index(drop = True)
    keep = CELL_COLS + ["id", "changed", "tag_years", "is_first_interval",
                        "age_start", "age_end"]
    df = df[keep].copy()
    _log(f"  modelled rows: {len(df):,}; overall change rate "
         f"{df['changed'].mean():.4f}")

    # --- Fit the 4 global params on a subsample ---------------------------->
    n_fit = min(args.fit_sample, len(df))
    df_fit = df.sample(n = n_fit, random_state = args.seed).reset_index(drop = True)
    _log(f"Fitting constant_breakpoint on {n_fit:,} sampled rows "
         f"({args.n_warmup}+{args.n_samples} x {args.n_chains} chains)...")
    model_fit = ConstantBreakpointModel(dataset = df_fit, metadata = metadata)
    fitter = ModelFitter(
        event_rate_fun = model_fit.event_rate_fun,
        starting_params = model_fit.starting_params,
        data = model_fit.data, target = model_fit.target,
        num_warmup = args.n_warmup, num_samples = args.n_samples,
        num_chains = args.n_chains,
        param_likelihood = model_fit.param_likelihood,
        derive_draws = model_fit.derive_draws,
        rng_key = jax.random.PRNGKey(args.seed),
        verbose = True,
    )
    fitter.fit()
    ptab = fitter.get_parameter_table()
    _log("Fitted parameters:")
    show = ptab[ptab["parameter"].isin(
        ["lambda_1", "lambda_2", "t_breakpoint", "delta"]
    )]
    print(show.to_string(index = False), flush = True)

    # --- Diagnose on the full population ----------------------------------->
    _log("Running dispersion diagnostics on full data (breakpoint model)...")
    model_full = ConstantBreakpointModel(dataset = df, metadata = metadata)
    report_bp = dispersion.dispersion_report(
        model_full, fitter, n_ppc_draws = args.n_ppc_draws, seed = args.seed,
    )
    _log("=== BREAKPOINT — SUMMARY ===")
    print(report_bp["summary"].to_string(index = False), flush = True)
    _log("=== BREAKPOINT — φ̂ by tag-age bin ===")
    print(_age_gradient(report_bp).to_string(index = False), flush = True)

    # --- Constant-λ (no age) baseline on the same rows --------------------->
    nonfirst = ~df["is_first_interval"].to_numpy().astype(bool)
    lam_global = float(
        df["changed"].to_numpy()[nonfirst].sum()
        / df["tag_years"].to_numpy()[nonfirst].sum()
    )
    delta_hat = float(show.set_index("parameter").loc["delta", "mean"]) \
        if "delta" in set(show["parameter"]) else 0.05
    _log(f"Constant-λ baseline: λ={lam_global:.4f}/yr, δ={delta_hat:.3f}")
    dt = df["tag_years"].to_numpy(dtype = np.float64)
    is_first = df["is_first_interval"].to_numpy().astype(bool)
    one_minus_delta = 1.0 - delta_hat

    def get_probs_const(_i: int) -> np.ndarray:
        e = np.exp(-lam_global * dt)
        p = np.where(is_first, 1.0 - one_minus_delta * e, 1.0 - e)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    report_c = dispersion.dispersion_report_from_probs(
        df, get_probs_const, n_use = 1, cell_cols = tuple(CELL_COLS),
        seed = args.seed,
    )
    _log("=== CONSTANT-λ BASELINE — φ̂ by tag-age bin (same rows) ===")
    print(_age_gradient(report_c).to_string(index = False), flush = True)

    out = Path(args.observations).expanduser().parent
    report_bp["summary"].to_csv(out / "breakpoint_refit_dispersion_summary.csv",
                                index = False)
    report_bp["subgroup"].to_csv(out / "breakpoint_refit_dispersion_subgroup.csv",
                                 index = False)
    _log(f"Wrote breakpoint_refit_dispersion_*.csv to {out}")


if __name__ == "__main__":
    main()
