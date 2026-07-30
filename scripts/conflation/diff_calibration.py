#!/usr/bin/env python
"""
Review CSV of the POIs the confidence calibration moved most.

Companion to ``diff_change_detection.py``: reads the calibrated conflated
parquet, ranks rows by ``|conf_mean - conf_mean_uncalibrated|``, and writes a
hand-reviewable sample per detection segment together with a per-`shared_label`
summary of the mean shift.

Config keys used (config.yaml):
    conflation.conflated — the calibrated parquet
    conflation           — directory; outputs land in its calibration/ subdir

Prerequisites:
    scripts/conflation/apply_calibration.py has run.

Output files (in conflation/<version>/calibration/):
    biggest_movers.csv
    shift_by_label.csv

Usage:
    python scripts/conflation/diff_calibration.py [--per-segment 200]
"""
from __future__ import annotations

import argparse

import pandas as pd
from config_versioned import Config

REVIEW_COLUMNS = [
    "unified_id", "source", "name", "shared_label", "calibration_flag",
    "osm_conf_mean", "overture_confidence", "original_conf_mean",
    "conf_mean_uncalibrated", "conf_mean", "conf_lower", "conf_upper",
    "delta", "shadow_matched", "match_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--per-segment", type = int, default = 200,
                        help = "Rows to sample per segment (default 200).")
    args = parser.parse_args()

    config = Config("~/repos/openpois/config.yaml")
    conflated = config.get_file_path("conflation", "conflated")
    out_dir = config.get_dir_path("conflation") / "calibration"
    out_dir.mkdir(parents = True, exist_ok = True)

    columns = [c for c in REVIEW_COLUMNS if c != "delta"]
    print(f"Reading {conflated} ...")
    frame = pd.read_parquet(conflated, columns = columns)
    frame["delta"] = frame["conf_mean"] - frame["conf_mean_uncalibrated"]
    print(f"  {len(frame):,} rows")

    # Biggest movers in both directions, per segment, so a reviewer sees the
    # promotions and the demotions rather than whichever direction dominates.
    picks = []
    for segment, rows in frame.groupby("source"):
        half = max(args.per_segment // 2, 1)
        picks.append(rows.nlargest(half, "delta"))
        picks.append(rows.nsmallest(half, "delta"))
    movers = pd.concat(picks).loc[:, REVIEW_COLUMNS]
    movers = movers.sort_values(["source", "delta"], ascending = [True, False])
    movers_path = out_dir / "biggest_movers.csv"
    movers.to_csv(movers_path, index = False)
    print(f"Biggest movers -> {movers_path} ({len(movers):,} rows)")

    label_shift = (
        frame.groupby(["shared_label", "source"])
        .agg(n = ("delta", "size"),
             mean_before = ("conf_mean_uncalibrated", "mean"),
             mean_after = ("conf_mean", "mean"),
             mean_delta = ("delta", "mean"))
        .reset_index()
        .sort_values("mean_delta")
    )
    label_path = out_dir / "shift_by_label.csv"
    label_shift.to_csv(label_path, index = False)
    print(f"Shift by label -> {label_path} ({len(label_shift):,} rows)")

    print("\nLargest mean decreases by (label, segment):")
    print(label_shift.head(8).to_string(index = False))
    print("\nLargest mean increases:")
    print(label_shift.tail(8).to_string(index = False))


if __name__ == "__main__":
    main()
