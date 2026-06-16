#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Fit a declining-hazard turnover model (``weibull`` / ``lomax`` /
``constant_breakpoint``) and a ``constant`` baseline on the same data, then run
the overdispersion diagnostics on the full population and compare the tag-age
φ̂ gradient + WAIC.

The point: the production random-effects fit and the two-rate breakpoint both
leave a steep φ̂-by-tag-age gradient (5.3 → 8.2 → ~106/135), because past the
breakpoint they revert to a single constant rate. A continuous declining hazard
— Weibull (duration dependence) or Lomax (gamma-frailty, the NB analogue) —
tests whether modelling the decline as a continuum flattens that gradient.

Global parameters (3 each) are fit on a random subsample (ample for population-
level params, keeps the dense NUTS fast); diagnostics + WAIC run on / sample
from the full modelled population. Memory-safe on a 16 GB host (the dispersion
loop streams one draw at a time; WAIC uses a bounded eval subset).

    python -u scripts/exploratory/fit_hazard_model_and_diagnose.py \
        --model-type lomax \
        --observations ~/data/openpois/osm_data/20260521/osm_observations.parquet \
        2>&1 | tee ~/data/openpois/logs/hazard_lomax.log
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.special import logsumexp

from openpois.models import dispersion
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import get_model_class
from openpois.models.setup import prepare_data_for_model

CELL_COLS = ["shared_label", "msa_code", "urban_rural"]
LOAD_COLS = [
    "id", "shared_label", "msa_code", "urban_rural", "changed",
    "last_obs_timestamp", "obs_timestamp", "last_tag_timestamp",
]
NATURAL = ["lambda", "shape", "theta", "t_breakpoint", "lambda_1", "lambda_2",
           "delta"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush = True)


def _metadata(model_type: str, args) -> dict:
    md = {"dt_col": "tag_years"}
    if model_type == "weibull":
        md["weibull_log_shape_prior"] = [args.shape_loc, args.shape_scale]
    elif model_type == "lomax":
        md["lomax_log_theta_prior"] = [args.theta_loc, args.theta_scale]
    elif model_type == "constant_breakpoint":
        md["t_breakpoint_prior"] = [float(np.log(args.tb_median)), args.tb_scale]
    return md


def _fit(model_type: str, df_fit: pd.DataFrame, metadata: dict, args):
    model = get_model_class(model_type)(dataset = df_fit, metadata = metadata)
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data, target = model.target,
        num_warmup = args.n_warmup, num_samples = args.n_samples,
        num_chains = args.n_chains,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        rng_key = jax.random.PRNGKey(args.seed),
        verbose = True,
    )
    fitter.fit()
    return model, fitter


def _waic(model_type, fitter, df_eval, metadata, n_use):
    """In-sample WAIC on a bounded eval subset (deviance scale, lower better)."""
    model = get_model_class(model_type)(dataset = df_eval, metadata = metadata)
    n_avail = int(jax.tree_util.tree_leaves(fitter.param_draws)[0].shape[0])
    idx = np.linspace(0, n_avail - 1, min(n_use, n_avail)).round().astype(int)
    draws = {k: v[idx] for k, v in fitter.param_draws.items()}
    target = model.target

    @jax.jit
    def ll_matrix(d):
        return jax.vmap(
            lambda p: model.pointwise_log_likelihood(p, model.data, target)
        )(d)

    ll = ll_matrix(draws)                       # (S, M)
    s = ll.shape[0]
    lppd = logsumexp(ll, axis = 0) - jnp.log(s)
    p_waic = jnp.var(ll, axis = 0, ddof = 1)
    return float(-2.0 * jnp.sum(lppd - p_waic)), int(len(target))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required = True,
                    choices = ["weibull", "lomax", "constant_breakpoint"])
    ap.add_argument("--observations", required = True)
    ap.add_argument("--fit-sample", type = int, default = 600_000)
    ap.add_argument("--eval-sample", type = int, default = 500_000)
    ap.add_argument("--n-ppc-draws", type = int, default = 150)
    ap.add_argument("--n-warmup", type = int, default = 300)
    ap.add_argument("--n-samples", type = int, default = 300)
    ap.add_argument("--n-chains", type = int, default = 2)
    ap.add_argument("--shape-loc", type = float, default = 0.0)
    ap.add_argument("--shape-scale", type = float, default = 1.0)
    ap.add_argument("--theta-loc", type = float, default = 0.0)
    ap.add_argument("--theta-scale", type = float, default = 1.5)
    ap.add_argument("--tb-median", type = float, default = 4.0)
    ap.add_argument("--tb-scale", type = float, default = 0.5)
    ap.add_argument("--amenity", default = None,
                    help = "Restrict to a single shared_label (e.g. 'School'); "
                           "fits on the full filtered set without subsampling.")
    ap.add_argument("--seed", type = int, default = 0)
    args = ap.parse_args()

    _log(f"Loading observations: {args.observations}")
    df = pd.read_parquet(Path(args.observations).expanduser(), columns = LOAD_COLS)
    df = prepare_data_for_model(df)
    df = df[df[CELL_COLS].notna().all(axis = 1)].reset_index(drop = True)
    keep = CELL_COLS + ["id", "changed", "tag_years", "is_first_interval",
                        "age_start", "age_end"]
    df = df[keep].copy()
    _log(f"  modelled rows: {len(df):,}; change rate {df['changed'].mean():.4f}")

    if args.amenity:
        df = df[df["shared_label"] == args.amenity].reset_index(drop = True)
        _log(f"  filtered to amenity '{args.amenity}': {len(df):,} rows; "
             f"change rate {df['changed'].mean():.4f}")
        # Single small amenity: fit + evaluate on the full set, no subsampling.
        df_fit = df_eval = df
    else:
        df_fit = df.sample(n = min(args.fit_sample, len(df)),
                           random_state = args.seed).reset_index(drop = True)
        df_eval = df.sample(n = min(args.eval_sample, len(df)),
                            random_state = args.seed + 1).reset_index(drop = True)

    md = _metadata(args.model_type, args)
    _log(f"Fitting '{args.model_type}' on {len(df_fit):,} rows; metadata={md}")
    model_full_meta = md
    model, fitter = _fit(args.model_type, df_fit, md, args)
    tab = fitter.get_parameter_table()
    _log(f"Fitted '{args.model_type}' parameters:")
    print(tab[tab["parameter"].isin(NATURAL)].to_string(index = False),
          flush = True)

    # Constant baseline on the same fit subsample, for WAIC reference.
    _log("Fitting 'constant' baseline on the same rows...")
    const_model, const_fitter = _fit("constant", df_fit, {"dt_col": "tag_years"},
                                     args)

    waic_m, n_eval = _waic(args.model_type, fitter, df_eval, model_full_meta,
                           args.n_ppc_draws)
    waic_c, _ = _waic("constant", const_fitter, df_eval, {"dt_col": "tag_years"},
                      args.n_ppc_draws)
    _log(f"=== WAIC on {n_eval:,} eval rows (deviance scale, lower=better) ===")
    print(f"  {args.model_type:>20}: {waic_m:,.1f}", flush = True)
    print(f"  {'constant':>20}: {waic_c:,.1f}", flush = True)
    print(f"  Δ (model − constant): {waic_m - waic_c:,.1f} "
          f"({'better' if waic_m < waic_c else 'worse'})", flush = True)

    # Dispersion diagnostics on the full population for the fitted model.
    _log(f"Running dispersion diagnostics on full data ('{args.model_type}')...")
    model_full = get_model_class(args.model_type)(dataset = df, metadata = md)
    report = dispersion.dispersion_report(
        model_full, fitter, n_ppc_draws = args.n_ppc_draws, seed = args.seed,
    )
    _log("=== SUMMARY ===")
    print(report["summary"].to_string(index = False), flush = True)
    _log("=== φ̂ by tag-age bin (target: a flatter gradient than 5.3→8.2→~106) ===")
    sub = report["subgroup"]
    print(sub[sub["grouping"] == "age_bin"].to_string(index = False), flush = True)

    # Same-handicap (no-covariate) constant baseline age gradient, so the
    # comparison isolates the hazard SHAPE rather than the missing covariates.
    _log("Running dispersion diagnostics on full data ('constant' baseline)...")
    const_full = get_model_class("constant")(
        dataset = df, metadata = {"dt_col": "tag_years"})
    report_c = dispersion.dispersion_report(
        const_full, const_fitter, n_ppc_draws = args.n_ppc_draws, seed = args.seed,
    )
    _log("=== CONSTANT baseline — φ̂ by tag-age bin (same rows) ===")
    subc = report_c["subgroup"]
    print(subc[subc["grouping"] == "age_bin"].to_string(index = False),
          flush = True)
    phi_m = report["summary"].set_index("statistic").loc[
        "covariate_cell", "phi_hat"]
    phi_c = report_c["summary"].set_index("statistic").loc[
        "covariate_cell", "phi_hat"]
    _log(f"covariate_cell φ̂ — {args.model_type}: {phi_m:.2f} vs "
         f"constant: {phi_c:.2f}")

    out = Path(args.observations).expanduser().parent
    report["summary"].to_csv(
        out / f"hazard_{args.model_type}_dispersion_summary.csv", index = False)
    report["subgroup"].to_csv(
        out / f"hazard_{args.model_type}_dispersion_subgroup.csv", index = False)
    _log(f"Wrote hazard_{args.model_type}_dispersion_*.csv to {out}")


if __name__ == "__main__":
    main()
