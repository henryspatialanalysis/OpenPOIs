#!/usr/bin/env python
"""
Summarize the conflated dataset by shared_label and source.

Reads conflated.parquet and produces a CSV with one row per shared_label
showing POI counts broken down by source (matched, osm, overture) and the
average composite match score for matched pairs.

Config keys used (config.yaml):
    conflation.conflated        — input GeoParquet path (conflated.parquet)
    conflation.summary_by_label — output CSV path
    conflation.match_status     — output CSV path (match-status view)

Prerequisites:
    Run scripts/conflation/conflate.py first.

Output files:
    summary_by_label.csv — columns: shared_label, matched, osm, overture,
        total, avg_match_score; sorted by total descending
    match_status_by_label.csv — columns: shared_label, matched, osm_only,
        overture_only, total, match_%; sorted by total descending. This is
        the table published in docs; labels with a zero in any column are
        single-source and are called out on stdout.
"""
from __future__ import annotations

import pandas as pd
from config_versioned import Config

config = Config("~/repos/openpois/config.yaml")
INPUT_PATH = config.get_file_path("conflation", "conflated")
OUTPUT_DIR = config.get_dir_path("conflation")
output_path = config.get_file_path("conflation", "summary_by_label")
match_status_path = config.get_file_path("conflation", "match_status")

if __name__ == "__main__":
    print(f"Reading {INPUT_PATH} ...")
    df = pd.read_parquet(
        INPUT_PATH,
        columns = [
            "shared_label", "source", "match_score",
        ],
    )
    print(f"  {len(df):,} rows")

    # Pivot: count by (shared_label, source)
    counts = (
        df.groupby(["shared_label", "source"])
        .size()
        .unstack(fill_value = 0)
    )
    # Reorder columns
    for col in ["matched", "osm", "overture"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["matched", "osm", "overture"]]
    counts["total"] = counts.sum(axis = 1)

    # Average match score per label (matched pairs only)
    matched = df[df["source"] == "matched"]
    avg_score = (
        matched.groupby("shared_label")["match_score"]
        .mean()
        .rename("avg_match_score")
    )
    summary = counts.join(avg_score).sort_values(
        "total", ascending = False,
    )
    summary.index.name = "shared_label"

    summary.to_csv(output_path)
    print(f"\nSaved to {output_path}")
    print(f"\n{summary.to_string()}")

    # Match-status view: same counts, named for the published
    # per-label table. A label with a zero in any column is
    # single-source and cannot match — the metric the taxonomy
    # crosswalks are tuned against.
    status = counts.rename(
        columns = {"osm": "osm_only", "overture": "overture_only"},
    ).sort_values("total", ascending = False)
    status["match_%"] = (
        100 * status["matched"] / status["total"]
    ).round(1)
    status.index.name = "shared_label"
    status.to_csv(match_status_path)
    print(f"\nSaved to {match_status_path}")

    totals = status[["matched", "osm_only", "overture_only", "total"]].sum()
    overall = 100 * totals["matched"] / max(totals["total"], 1)
    zero_cols = status[
        (status[["matched", "osm_only", "overture_only"]] == 0).any(axis = 1)
    ]
    print(
        f"\nTOTAL  matched {totals['matched']:,}  "
        f"OSM-only {totals['osm_only']:,}  "
        f"Overture-only {totals['overture_only']:,}  "
        f"total {totals['total']:,}  ({overall:.1f}% matched)"
    )
    print(
        f"Single-source labels (a zero in any column): "
        f"{len(zero_cols)}"
    )
    for lbl, row in zero_cols.iterrows():
        print(
            f"  {lbl:<22} matched {row['matched']:>9,}  "
            f"OSM {row['osm_only']:>9,}  "
            f"Overture {row['overture_only']:>9,}"
        )
