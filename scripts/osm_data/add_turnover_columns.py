"""
Final-stage augmentation of the OSM observations for time-varying λ models.

Reuses an already-prepared ``osm_observations.parquet`` (no history re-download
or re-format) and writes a new ``osm_data`` version that carries the metadata a
time-varying λ model needs to compute turnover probabilities:

  1. ``age_start`` / ``age_end`` — the tag's age (years since its current value
     was established, ``last_tag_timestamp``) at the start / end of each
     observation interval. These are the integration bounds for the breakpoint
     hazard; ``age_end − age_start == tag_years`` and ``age_start == 0`` on first
     intervals. When ``last_tag_timestamp`` is missing the interval start is used
     as the origin (age_start = 0). (Same definition as
     ``openpois.models.setup.prepare_data_for_model``.)

  2. ``osm_current_tag.parquet`` — one row per element ``(osm_type, id)`` giving
     ``tag_established`` (the establishment timestamp of the element's *current*
     name value) and ``last_seen`` (its latest observed timestamp). For the
     latest version of an element ``last_tag_timestamp`` is exactly when the
     current name began, so this is read straight off the max-``obs_timestamp``
     row per element. ``apply_model_breakpoint.py`` joins this onto the live
     snapshot to recover each POI's current tag age.

Usage:
    python scripts/osm_data/add_turnover_columns.py \
        --source-version 20260521 --target-version 20260608
"""

import argparse

import pandas as pd
from config_versioned import Config


config = Config("~/repos/openpois/config.yaml")

CURRENT_TAG_FILE = "osm_current_tag.parquet"


def _to_dt(series: pd.Series) -> pd.Series:
    """Parse an ISO timestamp column to tz-aware UTC datetimes."""
    return pd.to_datetime(series, utc = True, errors = "coerce")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Add age / tag-establishment metadata to osm_observations."
    )
    parser.add_argument("--source-version", required = True)
    parser.add_argument("--target-version", required = True)
    args = parser.parse_args()

    in_path = config.get_file_path(
        "osm_data", "osm_observations", custom_version = args.source_version
    )
    out_dir = config.get_dir_path(
        "osm_data", custom_version = args.target_version
    )
    out_dir.mkdir(parents = True, exist_ok = True)
    out_obs = config.get_file_path(
        "osm_data", "osm_observations", custom_version = args.target_version
    )
    out_cur = out_dir / CURRENT_TAG_FILE

    print(f"Reading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} observation rows.")

    # Tag-age integration bounds (years since the current value was established).
    ts_origin = _to_dt(df["last_tag_timestamp"])
    ts_lastobs = _to_dt(df["last_obs_timestamp"])
    ts_obs = _to_dt(df["obs_timestamp"])
    origin = ts_origin.fillna(ts_lastobs)
    df["age_start"] = (ts_lastobs - origin).dt.days / 365.0
    df["age_end"] = (ts_obs - origin).dt.days / 365.0
    print(
        "  age_start range "
        f"[{df['age_start'].min():.2f}, {df['age_start'].max():.2f}]; "
        f"age_end range [{df['age_end'].min():.2f}, {df['age_end'].max():.2f}]"
    )

    # Per-element current-state lookup: the latest version's establishment time.
    idx = (
        ts_obs.groupby([df["osm_type"], df["id"]]).idxmax()
    )
    cur = pd.DataFrame({
        "osm_type": df.loc[idx, "osm_type"].to_numpy(),
        "id": df.loc[idx, "id"].to_numpy(),
        "tag_established": origin.loc[idx].to_numpy(),
        "last_seen": ts_obs.loc[idx].to_numpy(),
    })
    print(f"  {len(cur):,} distinct elements → {out_cur.name}")

    df.to_parquet(out_obs, index = False)
    cur.to_parquet(out_cur, index = False)
    print(f"Saved augmented observations to {out_obs}")
    print(f"Saved current-tag lookup to {out_cur}")
