#!/usr/bin/env python
"""
Apply the change-detection penalty to a baseline conflated dataset.

Reads:
  - the baseline ``conflated.parquet`` (no change detection)
  - ``ghosts.parquet`` (from ``scripts/conflation/build_ghosts.py``)
  - ``fitted_params.csv`` for the active ``model_output`` version

Writes a new conflated parquet (suffix ``_cd`` by default) whose
unmatched-Overture rows have had ``conf_mean`` re-weighted by
``δ_group`` for any spatial+name+taxonomy match against a ghost. Audit
columns are appended (``shadow_*`` + ``original_conf_mean``) so the
demoted rows can be inspected by hand.

Usage:
    python scripts/conflation/apply_change_detection.py \
        --baseline-suffix=baseline --output-suffix=cd [--test]

Both ``--baseline-suffix`` and ``--output-suffix`` are inserted into
the conflated filename before ``.parquet`` (e.g. ``conflated_cd.parquet``).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config_versioned import Config

from openpois.conflation.change_detection import apply_shadow_match


def _suffixed_path(base_path: Path, suffix: str | None) -> Path:
    """Insert ``suffix`` before the parquet extension."""
    if not suffix:
        return base_path
    return base_path.with_name(
        f"{base_path.stem}_{suffix}{base_path.suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description = (
            "Apply change-detection penalty to a baseline conflated "
            "dataset using OSM-history-derived ghost POIs."
        )
    )
    parser.add_argument(
        "--baseline-suffix",
        default = "baseline",
        help = (
            "Suffix inserted into the input parquet filename "
            "(default: 'baseline' → conflated_baseline.parquet). "
            "Pass an empty string to read conflated.parquet directly."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default = "cd",
        help = (
            "Suffix inserted into the output parquet filename "
            "(default: 'cd' → conflated_cd.parquet)."
        ),
    )
    parser.add_argument(
        "--test",
        action = "store_true",
        help = (
            "Restrict ghosts to the configured conflation.test_bbox. "
            "Use when the baseline was produced with --test."
        ),
    )
    parser.add_argument(
        "--min-prior-name-score",
        type = float,
        default = None,
        help = (
            "Override config's min_prior_name_match_score. Higher "
            "values require a stricter Overture-name vs ghost-prior-"
            "name token_set_ratio match before a penalty fires. "
            "Default config value implements decision rule A "
            "(name-match required)."
        ),
    )
    parser.add_argument(
        "--no-survivor-filter",
        action = "store_true",
        help = (
            "Disable the current-OSM-survivor post-filter for this "
            "run. Used for ablation against the vetted set."
        ),
    )
    args = parser.parse_args()

    config = Config("~/repos/openpois/config.yaml")

    conflated_base = config.get_file_path("conflation", "conflated")
    baseline_path = _suffixed_path(
        conflated_base, args.baseline_suffix,
    )
    output_path = _suffixed_path(
        conflated_base, args.output_suffix,
    )
    ghosts_path = config.get_file_path("ghost_osm", "ghosts")

    model_dir = Path(config.get_dir_path("model_output"))
    fitted_params_path = model_dir / config.get(
        "directories", "model_output", "files", "fitted_params",
    )

    cd_cfg = config.get("conflation", "change_detection")
    min_match_score = float(cd_cfg["min_shadow_match_score"])
    default_delta = float(cd_cfg["default_delta"])
    min_prior_name_match_score = float(
        cd_cfg.get("min_prior_name_match_score", 0)
    )
    if args.min_prior_name_score is not None:
        min_prior_name_match_score = float(args.min_prior_name_score)

    survivor_filter = cd_cfg.get("suppress_if_current_survivor") or {}
    if args.no_survivor_filter:
        survivor_filter = dict(survivor_filter)
        survivor_filter["enabled"] = False
        print("Current-OSM-survivor filter disabled for this run.")

    max_radius_m = float(config.get("conflation", "max_radius_m"))
    default_radius_m = float(
        config.get("conflation", "default_radius_m")
    )
    distance_weight = float(config.get("conflation", "distance_weight"))
    name_weight = float(config.get("conflation", "name_weight"))
    type_weight = float(config.get("conflation", "type_weight"))
    identifier_weight = float(
        config.get("conflation", "identifier_weight")
    )

    # R1 needs the rated snapshot; no other auxiliary inputs are
    # needed by the simplified pipeline.
    rated_snapshot_path = config.get_file_path(
        "snapshot_osm", "rated_snapshot",
    )

    test_bbox = (
        config.get("conflation", "test_bbox") if args.test else None
    )

    print(f"Baseline: {baseline_path}")
    print(f"Ghosts:   {ghosts_path}")
    print(f"Fitted params: {fitted_params_path}")
    print(f"Output:   {output_path}")
    print(f"Rated snapshot (survivor filter): {rated_snapshot_path}")
    print(
        f"min_match_score={min_match_score} "
        f"max_radius_m={max_radius_m} "
        f"default_delta={default_delta} "
        f"min_prior_name_match_score={min_prior_name_match_score}"
    )
    if args.test:
        print(f"Test bbox: {test_bbox}")

    t0 = time.time()
    summary = apply_shadow_match(
        conflated_path = baseline_path,
        ghosts_path = ghosts_path,
        fitted_params_path = fitted_params_path,
        output_path = output_path,
        min_match_score = min_match_score,
        max_radius_m = max_radius_m,
        default_radius_m = default_radius_m,
        distance_weight = distance_weight,
        name_weight = name_weight,
        type_weight = type_weight,
        identifier_weight = identifier_weight,
        default_delta = default_delta,
        test_bbox = test_bbox,
        rated_snapshot_path = rated_snapshot_path,
        survivor_filter = survivor_filter,
        min_prior_name_match_score = min_prior_name_match_score,
    )
    elapsed = time.time() - t0

    print(f"\nApplied change-detection in {elapsed:.0f}s")
    print(f"  Total conflated rows:       {summary['n_total']:,}")
    print(
        f"  Unmatched Overture rows:    "
        f"{summary['n_unmatched_overture']:,}"
    )
    print(f"  Ghosts considered:          {summary['n_ghosts']:,}")
    print(
        f"  Shadow matches (final):     "
        f"{summary['n_shadow_matches']:,}"
    )
    print(
        f"  Dropped by survivor filter: "
        f"{summary['n_survivor_dropped']}"
    )
    print(
        f"  Mean penalty factor (Δ/old): "
        f"{summary['mean_penalty_factor']:.4f}"
    )
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
