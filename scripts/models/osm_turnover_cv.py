"""
Out-of-sample cross-validation for one ``random_effects`` turnover spec.

Fits the requested subset of random-effects terms and scores it on the shared
10-fold holdout set (built once by ``make_holdout_folds.py``), so several
specifications can be compared head-to-head on identical folds. For each fold
the model is refit on the other nine and scored on the held-out fold; unseen
factor levels (including an unseen δ group) back off to the main / global terms
via the model's per-term active masks.

Two artefacts are produced per spec (no predictions / viz — those are only for
the in-sample production fit):
  * the model "output parameters" from a single fit on the full prepared frame,
  * detailed out-of-sample performance metrics (per fold, aggregate, and a
    pooled per-subgroup breakdown).

δ rule: δ is grouped by ``shared_label`` when the amenity term is part of the λ
structure, and a single global δ otherwise (``--delta-group`` overrides). Both
back off to the global ``logit_delta_0`` for shared_labels unseen in a fold's
training data.

Example:
    python scripts/models/osm_turnover_cv.py \
        --model-version 2026-06-04-oos-msa --terms msa

Config keys used (config.yaml):
    directories.osm_data.osm_observations / holdout_folds
    directories.model_output (+ oos_metrics_* file stubs)
    osm_turnover_model.group_key / group_values / min_value_count /
        random_effects.*

Output files (in the model_output version directory):
    fitted_params.csv           — posterior summaries from the full-data fit
    oos_metrics_per_fold.csv     — RMSE / LPPD / WAIC per held-out fold
    oos_metrics_aggregate.csv    — mean OOS RMSE, summed / per-obs OOS LPD
    oos_metrics_subgroup.csv     — OOS RMSE / LPD pooled by shared_label, MSA,
                                   urban/rural
    config.yaml                  — snapshot of the exact spec that was run
"""
import argparse

import pandas as pd
from config_versioned import Config

from openpois.models import metrics
from openpois.models.model_fitter import ModelFitter
from openpois.models.osm_models import get_model_class
from openpois.models.setup import prepare_data_for_model

from _re_metadata import build_random_effects_metadata


config = Config("~/repos/openpois/config.yaml")

VALID_TERMS = ("amenity", "msa", "amenity_msa", "urbanicity")
DELTA_VALID_TERMS = ("amenity", "msa")
SUBGROUP_BY = ("shared_label", "msa_code", "urban_rural")


def build_fitted_params(model, fitter) -> pd.DataFrame:
    """Posterior parameter table with factor level names attached — identical
    layout to ``scripts/models/osm_turnover.py``."""
    fitted_params = (
        fitter.get_parameter_table()
        .merge(model.param_ids, on = "parameter", how = "left")
    )
    factor_lookups = getattr(model, "factor_lookups", None)
    if factor_lookups:
        long_lookup = pd.concat(
            [lut.assign(factor = fac) for fac, lut in factor_lookups.items()],
            ignore_index = True,
        )
        fitted_params = fitted_params.merge(
            long_lookup.loc[:, ["factor", "level_id", "level_name"]],
            on = ["factor", "level_id"], how = "left",
        )
    elif getattr(model, "group_lookup", None) is not None:
        fitted_params = fitted_params.merge(
            model.group_lookup, on = "group_id", how = "left"
        )
    return fitted_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Cross-validate one random_effects turnover spec.",
    )
    parser.add_argument("--model-version", required = True)
    parser.add_argument(
        "--terms", required = True,
        help = (
            "Comma-separated subset of "
            f"{{{','.join(VALID_TERMS)}}} to enable on log λ."
        ),
    )
    parser.add_argument(
        "--delta-terms", default = None,
        help = (
            "Comma-separated δ random-intercept terms from {amenity,msa} "
            "(amenity→shared_label, msa→msa_code), or 'none'/'global' for a "
            "single global δ. Default: auto — δ on amenity if the amenity λ "
            "term is enabled, else global δ."
        ),
    )
    parser.add_argument("--n-folds", type = int, default = 5)
    parser.add_argument("--num-warmup", type = int, default = 400)
    parser.add_argument("--num-samples", type = int, default = 400)
    parser.add_argument("--num-chains", type = int, default = 2)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument(
        "--stats-chunk", type = int, default = 50_000,
        help = (
            "Row-block size for held-out scoring; peak memory ~ draws * chunk. "
            "Kept modest for memory-constrained hosts."
        ),
    )
    parser.add_argument("--observations", default = None)
    parser.add_argument(
        "--folds-file", default = None,
        help = "Override the configured osm_data.holdout_folds path.",
    )
    args = parser.parse_args()

    terms = [t.strip() for t in args.terms.split(",") if t.strip()]
    bad = [t for t in terms if t not in VALID_TERMS]
    if bad:
        raise SystemExit(f"Unknown term(s) {bad}; valid: {VALID_TERMS}")
    if "amenity_msa" in terms and not ("amenity" in terms and "msa" in terms):
        raise SystemExit(
            "amenity_msa requires both amenity and msa to be enabled."
        )

    # δ terms (auto unless overridden): default mirrors the prior behavior —
    # δ on amenity (shared_label) when the amenity λ term is enabled, else
    # global δ. Both back off to the global intercept for unseen levels.
    if args.delta_terms is None:
        delta_terms = ["amenity"] if "amenity" in terms else []
    elif args.delta_terms.lower() in ("none", "global", ""):
        delta_terms = []
    else:
        delta_terms = [d.strip() for d in args.delta_terms.split(",") if d.strip()]
    bad_d = [d for d in delta_terms if d not in DELTA_VALID_TERMS]
    if bad_d:
        raise SystemExit(f"Unknown delta term(s) {bad_d}; valid: {DELTA_VALID_TERMS}")
    print(
        f"Spec {args.model_version}: terms={terms}, "
        f"delta_terms={delta_terms}, folds={args.n_folds}, "
        f"NUTS={args.num_chains}x({args.num_warmup}+{args.num_samples})"
    )

    metadata = build_random_effects_metadata(
        config, enabled_terms = terms, enabled_delta_terms = delta_terms,
    )

    # Faithful config snapshot: record the exact spec under this run version.
    otm = config.config["osm_turnover_model"]
    otm["default_model_type"] = "random_effects"
    re_cfg = otm["random_effects"]
    for name in re_cfg["terms"]:
        re_cfg["terms"][name]["enabled"] = name in terms
    for name in re_cfg.get("delta_terms", {}):
        re_cfg["delta_terms"][name]["enabled"] = name in delta_terms
    model_dir = config.get_dir_path(
        "model_output", custom_version = args.model_version
    )
    model_dir.mkdir(parents = True, exist_ok = True)
    config.write_self("model_output", custom_version = args.model_version)

    # Data preparation (identical to the in-sample fit / fold builder) -------->
    group_key = config.get(
        "osm_turnover_model", "group_key", fail_if_none = False
    )
    group_values = config.get(
        "osm_turnover_model", "group_values", fail_if_none = False
    )
    min_value_count = config.get(
        "osm_turnover_model", "min_value_count", fail_if_none = False
    )
    obs_path = args.observations or config.get_file_path(
        "osm_data", "osm_observations"
    )
    print(f"Loading observations from {obs_path} ...")
    df = prepare_data_for_model(
        data = pd.read_parquet(obs_path),
        group_key = group_key,
        group_values = group_values,
        min_value_count = min_value_count,
        t2_col = "obs_timestamp",
    ).reset_index(drop = True)
    print(f"Prepared {len(df):,} rows ({df['id'].nunique():,} POIs).")

    # Shared folds → row-aligned fold ids.
    folds_file = args.folds_file or config.get_file_path(
        "osm_data", "holdout_folds"
    )
    fold_map = pd.read_parquet(folds_file)
    merged = df.merge(fold_map, on = "id", how = "left")
    if merged["fold"].isna().any():
        n_missing = int(merged["fold"].isna().sum())
        raise SystemExit(
            f"{n_missing:,} rows have no fold in {folds_file}; rebuild folds "
            "from the same observations/prepare settings."
        )
    fold_ids = merged["fold"].to_numpy()

    # Full-data fit → output parameters -------------------------------------->
    print("\nFitting on the full prepared frame (output parameters) ...")
    model = get_model_class("random_effects")(dataset = df, metadata = metadata)
    fitter = ModelFitter(
        event_rate_fun = model.event_rate_fun,
        starting_params = model.starting_params,
        data = model.data,
        target = model.target,
        num_warmup = args.num_warmup,
        num_samples = args.num_samples,
        num_chains = args.num_chains,
        param_likelihood = model.param_likelihood,
        derive_draws = model.derive_draws,
        log_likelihood_fun = model.log_likelihood_fun,
        log_1md_fun = getattr(model, "log_1md_fun", None),
        verbose = True,
    )
    fitter.fit()
    config.write(
        build_fitted_params(model, fitter), "model_output", "fitted_params",
        custom_version = args.model_version,
    )

    # Cross-validation -------------------------------------------------------->
    print("\nRunning cross-validation ...")
    cv = metrics.cross_validate(
        df, metadata,
        model_name = "random_effects",
        n_folds = args.n_folds,
        num_warmup = args.num_warmup,
        num_samples = args.num_samples,
        num_chains = args.num_chains,
        seed = args.seed,
        prepared = True,
        fold_ids = fold_ids,
        subgroup_by = SUBGROUP_BY,
        stats_chunk = args.stats_chunk,
    )

    per_fold = cv["per_fold"]
    aggregate = pd.DataFrame([{"model_version": args.model_version, **cv["aggregate"]}])
    config.write(
        per_fold, "model_output", "oos_metrics_per_fold",
        custom_version = args.model_version,
    )
    config.write(
        aggregate, "model_output", "oos_metrics_aggregate",
        custom_version = args.model_version,
    )
    config.write(
        cv["per_subgroup"], "model_output", "oos_metrics_subgroup",
        custom_version = args.model_version,
    )

    print("\nOut-of-sample aggregate:")
    for k, v in cv["aggregate"].items():
        print(f"  {k}: {v}")
