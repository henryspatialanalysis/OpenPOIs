"""
Plot OSM tag stability curves from observation data.

Reads ``osm_observations.parquet`` (one row per (POI version, shared_label)) and
computes Kaplan-Meier-style survival estimates showing what fraction of
observations remain unchanged over time. Saves two types of PNG figures:

    1. Overall stability curve — all observations pooled into a single panel.
       (Note: rows are duplicated per shared_label, so POIs mapping to
       multiple taxonomy categories are over-represented here.)
    2. Per-shared-label multi-panel curves — top-N shared labels by row
       count, shown as separate facets.

Config keys used (config.yaml):
    directories.osm_data           — directory containing input parquet and viz/ output
    osm_data.tag_key               — tag key whose changes define observation
                                     events (used only in plot titles)
    osm_data.timestamp_cols        — columns to parse as timestamps (rows with
                                     nulls dropped)
    osm_data.top_n_types           — number of top shared labels in the multi-panel figure
    download.osm.end_date          — right-censoring date for still-unchanged tags
    osm_data.apply_model.model_stub — stub for loading model predictions

Prerequisites:
    Run osm_data/format_tabular.py first.

Output files (in osm_data/viz/):
    osm_changes_all.png                     — overall survival curve
    osm_changes_all_preds.png               — overall curve with constant-model
                                              prediction overlay
    osm_changes_by_shared_label.png         — per-label facet grid (top N)
    by_type/osm_changes_<label>.png         — per-label curves with
                                              shared-label model predictions,
                                              one file per shared_label with a
                                              fitted prediction
"""

import numpy as np
import pandas as pd
from config_versioned import Config

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import plotnine as gg  # noqa: E402

from openpois.osm.change_plots import (  # noqa: E402
    change_plot_create, change_multiplot_create
)

# ----------------------------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

SAVE_DIR = config.get_dir_path("osm_data")
OBSERVATIONS_PATH = config.get_file_path("osm_data", "osm_observations")
VIZ_DIR = SAVE_DIR / "viz"
TAG_KEY = config.get("osm_data", "tag_key")
END_DATE = pd.Timestamp(config.get("download", "osm", "end_date"), tz='UTC')
MODEL_BASE = config.get_dir_path("model_output").parent
MODEL_STUB = config.get("osm_data", "apply_model", "model_stub")
SHARED_LABEL_VERSION = f"{MODEL_STUB}_by_shared_label"

max_days = 365 * 10
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------------------


def fig_save(
    fig: gg.ggplot,
    stub: str,
    width: float = 10,
    height: float = 6,
    subdir: str | None = None,
    **kwargs,
) -> None:
    """
    Save a ggplot figure as a PNG file to VIZ_DIR (or a subdirectory of it).

    Args:
        fig: The ggplot figure to save.
        stub: Output filename stem (without extension).
        width: Figure width in inches.
        height: Figure height in inches.
        subdir: Optional subdirectory under VIZ_DIR. Created if it doesn't
            exist.
        **kwargs: Additional keyword arguments forwarded to fig.save().
    """
    out_dir = VIZ_DIR if subdir is None else VIZ_DIR / subdir
    out_dir.mkdir(parents = True, exist_ok = True)
    fig.save(
        filename = out_dir / f"{stub}.png",
        width = width,
        height = height,
        units = 'in',
        dpi = 300,
        verbose = False,
        **kwargs
    )


def get_preds_dict(model_stub: str | None) -> dict[str, pd.DataFrame]:
    """Load constant and shared-label model predictions."""
    if model_stub is None:
        return dict()

    def get_preds_df(version: str) -> pd.DataFrame | None:
        preds_fp = MODEL_BASE / version / "predictions.csv"
        if not preds_fp.exists():
            return None
        return pd.read_csv(preds_fp).assign(
            year = pd.col('t2'),
            conf_mean = (1.0 - pd.col('p_fresh_mean')),
            conf_lower = (1.0 - pd.col('p_fresh_upper')),
            conf_upper = (1.0 - pd.col('p_fresh_lower')),
        )

    return {
        "constant": get_preds_df(f"{model_stub}_constant"),
        "shared_label": get_preds_df(SHARED_LABEL_VERSION),
    }


# ----------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Read model predictions
    preds = get_preds_dict(MODEL_STUB)
    # Read observations
    # Drop the first observation for each POI (when the POI was first added) - the last
    #   observation timestamp will be missing for these rows
    timestamp_cols = config.get("osm_data", "timestamp_cols")
    observations_df = (
        pd.read_parquet(OBSERVATIONS_PATH)
        .dropna(subset = timestamp_cols)
    )
    for timestamp_col in timestamp_cols:
        observations_df[timestamp_col] = pd.to_datetime(observations_df[timestamp_col])
    # Add a column that is 1 for the highest value of 'version' within each 'id' grouping
    observations_df['latest_version'] = (
        observations_df.groupby('id')['version']
        .transform(lambda x: x == x.max())
        .astype(int)
    )
    # Prepare timediffs in days:
    # no_change: Time elapsed until the final confirmation of the previous tag
    # change: Time elapsed from previous tag to changed tag
    # final_obs: Time elapsed from previous tag to data download
    changed_tags = (
        observations_df
        .query('changed == 1')
        .assign(
            no_change=(
                pd.col('last_obs_timestamp') - pd.col('last_tag_timestamp')
            ).dt.days,
            change=(pd.col('obs_timestamp') - pd.col('last_tag_timestamp')).dt.days,
            final_obs=(END_DATE - pd.col('last_tag_timestamp')).dt.days
        )
    )
    unchanged_tags = (
        observations_df
        .query('(changed == 0) & (latest_version == 1)')
        .assign(
            no_change=(pd.col('obs_timestamp') - pd.col('last_tag_timestamp')).dt.days,
            change=np.inf,
            final_obs=(END_DATE - pd.col('last_tag_timestamp')).dt.days
        )
    )
    # Format changes
    to_plot_df = pd.concat([changed_tags, unchanged_tags])
    to_plot_df['final_obs'] = np.inf
    # Create a plot for all tags
    fig = change_plot_create(
        observations = to_plot_df,
        no_change_col = 'no_change',
        change_col = 'change',
        final_observation_col = 'final_obs',
        day_range = max_days,
        title = f"Stability of the `{TAG_KEY}` tag over time",
        x_label = "Years since tag",
        y_label = "Proportion remaining unchanged",
    )
    fig_save(fig, stub = "osm_changes_all")

    if preds.get('constant') is not None:
        fig = change_plot_create(
            observations = to_plot_df,
            predictions = preds['constant'],
            no_change_col = 'no_change',
            change_col = 'change',
            final_observation_col = 'final_obs',
            day_range = max_days,
            title = f"Stability of the `{TAG_KEY}` tag over time",
            x_label = "Years since tag",
            y_label = "Proportion remaining unchanged",
        )
        fig_save(fig, stub = "osm_changes_all_preds")

    # Multi-panel plot faceted by the top shared labels. Catch-all "Other ..."
    # labels are excluded so the top N reflects specific POI categories.
    TOP_N_TYPES = config.get("osm_data", "top_n_types")
    other_labels = sorted(
        lbl for lbl in to_plot_df["shared_label"].dropna().unique()
        if str(lbl).startswith("Other ")
    )
    fig = change_multiplot_create(
        observations = to_plot_df,
        col = "shared_label",
        top_n = TOP_N_TYPES,
        exclude_values = other_labels,
        no_change_col = 'no_change',
        change_col = 'change',
        final_observation_col = 'final_obs',
        color_label = "Amenity",
        title = f"Stability of the `{TAG_KEY}` tag over time by shared label",
        subtitle = (
            f"Top {TOP_N_TYPES} shared labels by number of observations"
        ),
        x_label = "Years since tag",
        y_label = "Proportion remaining unchanged",
        day_range = max_days,
    )
    fig_save(fig = fig, stub = "osm_changes_by_shared_label")

    if preds.get('shared_label') is not None:
        observed_labels = set(to_plot_df["shared_label"].dropna().unique())
        pred_groups = set(preds['shared_label']['group_name'].unique())
        for pred_label in sorted(pred_groups & observed_labels):
            print(f"Plotting shared_label = {pred_label}")
            fig = change_plot_create(
                observations = to_plot_df.query(
                    "shared_label == @pred_label"
                ),
                predictions = preds['shared_label'].query(
                    "group_name == @pred_label"
                ),
                no_change_col = 'no_change',
                change_col = 'change',
                final_observation_col = 'final_obs',
                day_range = max_days,
                title = (
                    f"Stability of the `{TAG_KEY}` tag over time: {pred_label}"
                ),
                x_label = "Years since tag",
                y_label = "Proportion remaining unchanged",
            )
            fig_save(
                fig,
                stub = f"osm_changes_{pred_label}",
                subdir = "by_type",
            )
