"""
Create the shared 10-fold holdout assignment for OSM turnover cross-validation.

Built once, up front, so every out-of-sample model specification
(``scripts/models/osm_turnover_cv.py``) is scored on *identical* folds — the
only way a head-to-head OOS comparison is fair. Folds are assigned at the
individual-POI level (all of a POI's interval rows share a fold, so the holdout
has no within-POI leakage) and stratified across (MSA × shared_label ×
urban/rural) cells so each fold is balanced.

The observations are prepared with the same ``group_key`` / ``min_value_count``
the model fits use, so the prepared row set — and therefore the POI ids — match
what the fits and the CV runs see.

Config keys used (config.yaml):
    directories.osm_data.osm_observations  — input observations
    directories.osm_data.holdout_folds     — output id → fold map
    osm_turnover_model.group_key / group_values / min_value_count

Output (an ``id, fold`` parquet at ``osm_data.holdout_folds``):
    one row per POI id, ``fold`` in ``1..n_folds``.
"""
import argparse

import pandas as pd
from config_versioned import Config

from openpois.models import metrics
from openpois.models.setup import prepare_data_for_model


config = Config("~/repos/openpois/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Assign shared k-fold holdouts for OSM turnover CV.",
    )
    parser.add_argument("--n-folds", type = int, default = 10)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument(
        "--observations", default = None,
        help = "Override the configured osm_data.osm_observations path.",
    )
    args = parser.parse_args()

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
    df = pd.read_parquet(obs_path)
    df = prepare_data_for_model(
        data = df,
        group_key = group_key,
        group_values = group_values,
        min_value_count = min_value_count,
        t2_col = "obs_timestamp",
    ).reset_index(drop = True)
    print(f"Prepared {len(df):,} observation rows ({df['id'].nunique():,} POIs).")

    folds = metrics.assign_holdout_folds(
        df, n_folds = args.n_folds, seed = args.seed,
    )
    df = df.assign(_fold = folds.to_numpy())

    # Whole POIs share a fold, so one (id → fold) row per POI is sufficient and
    # robust to row order when the CV runs map it back on.
    fold_map = (
        df.loc[:, ["id", "_fold"]]
        .drop_duplicates(subset = "id")
        .rename(columns = {"_fold": "fold"})
        .reset_index(drop = True)
    )
    if not fold_map["id"].is_unique:
        raise RuntimeError("A POI id was assigned to more than one fold.")

    out_path = config.get_file_path("osm_data", "holdout_folds")
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fold_map.to_parquet(out_path, index = False)
    print(f"\nWrote {len(fold_map):,} (id, fold) rows → {out_path}")

    # Balance report -------------------------------------------------------->
    print("\nPOIs per fold:")
    print(fold_map["fold"].value_counts().sort_index().to_string())
    print("\nObservation rows per fold:")
    print(df["_fold"].value_counts().sort_index().to_string())
    if "urban_rural" in df.columns:
        print("\nRow share of each urban_rural class within each fold:")
        bal = (
            df.groupby("_fold")["urban_rural"]
            .value_counts(normalize = True)
            .unstack()
            .round(3)
        )
        print(bal.to_string())
