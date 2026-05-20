#!/usr/bin/env python
"""
Diff the baseline conflated parquet against the change-detection output
and write a manual-review CSV of the demoted POIs.

The script joins the two outputs on ``unified_id`` (stable across runs
when the row identity is preserved), filters to Overture-source rows
whose ``conf_mean`` decreased, and writes a sorted CSV with the audit
trail (ghost id, event type, event timestamp, ghost prior name) so the
user can verify each shadow match manually.

Usage:
    python scripts/conflation/diff_change_detection.py \
        --baseline-suffix=baseline --cd-suffix=cd \
        --output=~/data/openpois/logs/seattle_demoted_pois.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from config_versioned import Config


def _suffixed_path(base: Path, suffix: str | None) -> Path:
    if not suffix:
        return base
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description = (
            "Diff baseline vs change-detection conflated parquets "
            "and emit a demoted-POI review CSV."
        )
    )
    parser.add_argument(
        "--baseline-suffix", default = "baseline",
        help = "Suffix on the baseline parquet (default: baseline).",
    )
    parser.add_argument(
        "--cd-suffix", default = "cd",
        help = "Suffix on the change-detection parquet (default: cd).",
    )
    parser.add_argument(
        "--output",
        default = "~/data/openpois/logs/demoted_pois.csv",
        help = "Output CSV path (~ is expanded).",
    )
    parser.add_argument(
        "--ghosts-path",
        default = None,
        help = (
            "Optional ghosts.parquet path. If set, the ghost's "
            "prior_name and event_type are joined onto the CSV by "
            "shadow_ghost_id. Defaults to the configured ghost_osm dir."
        ),
    )
    args = parser.parse_args()

    config = Config("~/repos/openpois/config.yaml")
    base = config.get_file_path("conflation", "conflated")

    baseline_path = _suffixed_path(base, args.baseline_suffix)
    cd_path = _suffixed_path(base, args.cd_suffix)
    ghosts_path = (
        Path(args.ghosts_path).expanduser()
        if args.ghosts_path
        else config.get_file_path("ghost_osm", "ghosts")
    )
    output_path = Path(args.output).expanduser()

    print(f"Baseline:        {baseline_path}")
    print(f"Change-detect:   {cd_path}")
    print(f"Ghosts:          {ghosts_path}")
    print(f"Output CSV:      {output_path}")

    print("\nLoading parquets ...")
    base_gdf = gpd.read_parquet(
        baseline_path,
        columns = [
            "unified_id", "source", "overture_id",
            "name", "shared_label", "conf_mean", "geometry",
        ],
    )
    cd_gdf = gpd.read_parquet(
        cd_path,
        columns = [
            "unified_id", "source", "overture_id",
            "name", "shared_label", "conf_mean",
            "shadow_matched", "shadow_ghost_id",
            "shadow_event_type", "shadow_event_timestamp",
            "shadow_score", "shadow_distance_m",
            "original_conf_mean", "geometry",
        ],
    )

    print(f"  baseline rows: {len(base_gdf):,}")
    print(f"  CD rows:       {len(cd_gdf):,}")

    if len(base_gdf) != len(cd_gdf):
        print(
            "  WARNING: row counts differ — the diff still proceeds "
            "by joining on unified_id."
        )

    merged = cd_gdf.merge(
        base_gdf[["unified_id", "conf_mean"]].rename(
            columns = {"conf_mean": "conf_mean_baseline"},
        ),
        on = "unified_id",
        how = "left",
    )

    # Demoted = Overture-source rows where CD conf_mean < baseline
    demoted = merged[
        (merged["source"] == "overture")
        & merged["shadow_matched"].fillna(False)
        & (merged["conf_mean"] < merged["conf_mean_baseline"])
    ].copy()

    demoted["delta_conf"] = (
        demoted["conf_mean_baseline"] - demoted["conf_mean"]
    )
    demoted = demoted.sort_values(
        "delta_conf", ascending = False,
    ).reset_index(drop = True)

    print(f"\nDemoted Overture rows: {len(demoted):,}")
    if len(demoted) == 0:
        print("Nothing to write.")
        return

    # Join ghost prior_name (the OSM-history name at the time of the
    # removal/rename event) onto the audit row.
    ghosts = pd.read_parquet(
        ghosts_path,
        columns = [
            "ghost_id", "prior_name",
            "event_type", "event_timestamp",
        ],
    )
    demoted = demoted.merge(
        ghosts.rename(
            columns = {
                "prior_name": "ghost_prior_name",
                "event_type": "ghost_event_type",
                "event_timestamp": "ghost_event_timestamp",
            },
        ),
        left_on = "shadow_ghost_id",
        right_on = "ghost_id",
        how = "left",
    ).drop(columns = ["ghost_id"])

    # Project geometry to lon/lat for readability.
    demoted["lon"] = demoted.geometry.x
    demoted["lat"] = demoted.geometry.y

    output_cols = [
        "overture_id", "name", "shared_label",
        "lon", "lat",
        "conf_mean_baseline", "conf_mean", "delta_conf",
        "shadow_score", "shadow_distance_m",
        "shadow_event_type", "shadow_event_timestamp",
        "shadow_ghost_id", "ghost_prior_name",
        "ghost_event_type", "ghost_event_timestamp",
        "unified_id",
    ]
    output_path.parent.mkdir(parents = True, exist_ok = True)
    demoted[output_cols].to_csv(output_path, index = False)
    print(f"Wrote {len(demoted):,} rows to {output_path}")

    by_event = (
        demoted["shadow_event_type"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
    )
    print("\nDemoted-by-event breakdown:")
    for et, n in sorted(by_event.items(), key = lambda kv: -kv[1]):
        print(f"  {et}: {n}")

    print("\nTop 10 by drop:")
    for _, r in demoted.head(10).iterrows():
        print(
            f"  {r.get('name') or '(no name)'} "
            f"({r['shared_label'] or '-'}): "
            f"{r['conf_mean_baseline']:.3f} → {r['conf_mean']:.3f} "
            f"[event={r.get('ghost_event_type')}, "
            f"prior={r.get('ghost_prior_name')}]"
        )


if __name__ == "__main__":
    main()
