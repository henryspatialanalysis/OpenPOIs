"""
Visualise a fitted ``random_effects`` turnover model against observed stability.

``scripts/osm_data/data_viz.py`` overlays predictions only for the
``constant`` / ``random_by_type`` families (keyed on a single ``group_name``).
A ``random_effects`` fit predicts one curve per (shared_label × MSA ×
urban/rural) cell, so this script collapses those per-cell curves to one curve
per ``shared_label`` — the observation-count-weighted average across the cells
of that label — and overlays it on the observed Kaplan-Meier-style curve.

Output PNGs (in ``<model_output>/<model-version>/viz/``):
    osm_changes_all_preds.png        — overall observed curve + model overlay
    by_type/osm_changes_<label>.png  — per-shared_label observed + model overlay

Config keys used (config.yaml):
    directories.model_output         — locate the fit's predictions.csv + viz/
    directories.osm_data.osm_observations
    osm_data.tag_key / timestamp_cols / top_n_types
    download.osm.end_date            — right-censoring date
"""
import argparse

import numpy as np
import pandas as pd
from config_versioned import Config

import matplotlib
matplotlib.use("Agg")  # noqa: E402

from openpois.osm.change_plots import change_plot_create  # noqa: E402


config = Config("~/repos/openpois/config.yaml")

CELL_COLS = ["shared_label", "msa_code", "urban_rural"]
P_COLS = ["p_fresh_mean", "p_fresh_lower", "p_fresh_upper"]
MAX_DAYS = 365 * 10


def weighted_label_curves(
    predictions: pd.DataFrame, cell_weights: pd.DataFrame
) -> pd.DataFrame:
    """Collapse per-cell predicted curves to one curve per shared_label, weighting
    each cell by its observation count. Returns columns ``shared_label, t2,
    conf_mean, conf_lower, conf_upper`` (conf = 1 − P(change), fresh regime)."""
    # CSV inference can read numeric msa codes as int; the parquet keeps them as
    # strings. Coerce both sides so the merge keys line up.
    predictions = predictions.copy()
    cell_weights = cell_weights.copy()
    for col in CELL_COLS:
        predictions[col] = predictions[col].astype(str)
        cell_weights[col] = cell_weights[col].astype(str)
    preds = predictions.merge(cell_weights, on = CELL_COLS, how = "left")
    preds["w"] = preds["w"].fillna(0.0)
    for col in P_COLS:
        preds[f"_wx_{col}"] = preds[col] * preds["w"]
    grp = preds.groupby(["shared_label", "t2"], observed = True)
    agg = grp.agg(
        wsum = ("w", "sum"),
        **{f"wx_{c}": (f"_wx_{c}", "sum") for c in P_COLS},
        **{f"mean_{c}": (c, "mean") for c in P_COLS},
    ).reset_index()
    out = pd.DataFrame({
        "shared_label": agg["shared_label"],
        "t2": agg["t2"],
    })
    for col in P_COLS:
        wavg = np.where(
            agg["wsum"] > 0, agg[f"wx_{col}"] / agg["wsum"].replace(0, np.nan),
            agg[f"mean_{col}"],
        )
        out[col] = np.where(np.isfinite(wavg), wavg, agg[f"mean_{col}"])
    # conf = 1 - P(change); the interval flips.
    out["conf_mean"] = 1.0 - out["p_fresh_mean"]
    out["conf_lower"] = 1.0 - out["p_fresh_upper"]
    out["conf_upper"] = 1.0 - out["p_fresh_lower"]
    return out.loc[:, ["shared_label", "t2", "conf_mean", "conf_lower", "conf_upper"]]


def build_observed(observations: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    """Reshape observations into the changed/unchanged frame the change plots
    consume (mirrors scripts/osm_data/data_viz.py)."""
    obs = observations.copy()
    obs["latest_version"] = (
        obs.groupby("id")["version"].transform(lambda x: x == x.max()).astype(int)
    )
    changed = (
        obs.query("changed == 1").assign(
            no_change = (obs["last_obs_timestamp"] - obs["last_tag_timestamp"]).dt.days,
            change = (obs["obs_timestamp"] - obs["last_tag_timestamp"]).dt.days,
        )
    )
    unchanged = (
        obs.query("(changed == 0) & (latest_version == 1)").assign(
            no_change = (obs["obs_timestamp"] - obs["last_tag_timestamp"]).dt.days,
            change = np.inf,
        )
    )
    to_plot = pd.concat([changed, unchanged])
    to_plot["final_obs"] = np.inf
    return to_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Overlay a random_effects fit on observed stability curves.",
    )
    parser.add_argument("--model-version", required = True)
    parser.add_argument(
        "--observations", default = None,
        help = "Override the configured osm_data.osm_observations path.",
    )
    args = parser.parse_args()

    model_base = config.get_dir_path("model_output").parent
    model_dir = model_base / args.model_version
    predictions_path = model_dir / "predictions.csv"
    if not predictions_path.exists():
        raise SystemExit(f"predictions.csv not found in {model_dir}")
    predictions = pd.read_csv(predictions_path)
    missing = [c for c in CELL_COLS + ["t2"] + P_COLS if c not in predictions.columns]
    if missing:
        raise SystemExit(
            f"predictions.csv missing {missing} — not a random_effects fit?"
        )

    tag_key = config.get("osm_data", "tag_key")
    timestamp_cols = config.get("osm_data", "timestamp_cols")
    end_date = pd.Timestamp(config.get("download", "osm", "end_date"), tz = "UTC")
    top_n = config.get("osm_data", "top_n_types")

    obs_path = args.observations or config.get_file_path(
        "osm_data", "osm_observations"
    )
    observations = (
        pd.read_parquet(obs_path)
        .dropna(subset = timestamp_cols)
    )
    for col in timestamp_cols:
        observations[col] = pd.to_datetime(observations[col])

    cell_weights = (
        observations.dropna(subset = CELL_COLS)
        .groupby(CELL_COLS, observed = True).size().rename("w").reset_index()
    )
    label_curves = weighted_label_curves(predictions, cell_weights)
    to_plot = build_observed(observations, end_date)

    viz_dir = model_dir / "viz"
    by_type_dir = viz_dir / "by_type"
    by_type_dir.mkdir(parents = True, exist_ok = True)

    def save(fig, path):
        fig.save(
            filename = path, width = 10, height = 6, units = "in", dpi = 300,
            verbose = False,
        )

    # Overall: weight all cells together into a single label="__all__".
    overall = weighted_label_curves(
        predictions.assign(shared_label = "__all__"),
        cell_weights.assign(shared_label = "__all__")
        .groupby(CELL_COLS, observed = True)["w"].sum().reset_index(),
    )
    fig = change_plot_create(
        observations = to_plot,
        predictions = overall,
        no_change_col = "no_change",
        change_col = "change",
        final_observation_col = "final_obs",
        day_range = MAX_DAYS,
        title = f"Stability of the `{tag_key}` tag: random-effects model",
        x_label = "Years since tag",
        y_label = "Proportion remaining unchanged",
    )
    save(fig, viz_dir / "osm_changes_all_preds.png")
    print(f"Wrote {viz_dir / 'osm_changes_all_preds.png'}")

    observed_labels = set(to_plot["shared_label"].dropna().unique())
    top_labels = (
        to_plot["shared_label"].value_counts().head(top_n).index.tolist()
    )
    for label in top_labels:
        if label not in observed_labels:
            continue
        preds_label = label_curves.query("shared_label == @label")
        if preds_label.empty:
            continue
        fig = change_plot_create(
            observations = to_plot.query("shared_label == @label"),
            predictions = preds_label,
            no_change_col = "no_change",
            change_col = "change",
            final_observation_col = "final_obs",
            day_range = MAX_DAYS,
            title = f"Stability of the `{tag_key}` tag: {label}",
            x_label = "Years since tag",
            y_label = "Proportion remaining unchanged",
        )
        save(fig, by_type_dir / f"osm_changes_{label}.png")
        print(f"Wrote {by_type_dir / f'osm_changes_{label}.png'}")
