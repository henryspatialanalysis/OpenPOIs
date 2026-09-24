#!/usr/bin/env python
"""
Apply the change-detection penalty to a baseline conflated dataset.

Reads:
  - the baseline ``conflated.parquet`` (no change detection)
  - ``ghosts.parquet`` (from ``scripts/conflation/build_ghosts.py``),
    dropping ghosts older than ``max_ghost_age_years`` at load time
  - ``fitted_params.csv`` for the active ``model_output`` version
  - the full filtered ``osm_snapshot.parquet`` (survivor filter, when
    ``suppress_if_current_survivor.use_full_snapshot`` is true) or the
    rated snapshot otherwise

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
            "Override config's min_prior_name_match_score: the "
            "token_set_ratio a ghost's prior name/brand must reach "
            "against the Overture name/brand (after normalisation) "
            "before a penalty can fire. Both names are always required; "
            "0 accepts any ratio. Config default 70 (same-entity rule)."
        ),
    )
    parser.add_argument(
        "--max-ghost-age-years",
        type = float,
        default = None,
        help = (
            "Override config's max_ghost_age_years (ghosts older than "
            "this are dropped before matching; 0 keeps all)."
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

    max_ghost_age_years = cd_cfg.get("max_ghost_age_years")
    max_ghost_age_years = (
        None if max_ghost_age_years is None else float(max_ghost_age_years)
    )
    if args.max_ghost_age_years is not None:
        max_ghost_age_years = float(args.max_ghost_age_years)

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
    # Drop conflated points that never got a shared_label (uncategorized
    # POIs). Default on; the canonical conflated.parquet is label-only.
    drop_unlabeled = config.get(
        "conflation", "drop_unlabeled", fail_if_none = False
    )
    drop_unlabeled = True if drop_unlabeled is None else bool(drop_unlabeled)

    # The survivor filter (R1) reads the full filtered snapshot — nodes,
    # ways and relations by centroid — so a POI that survives in OSM as a
    # building way still suppresses the penalty. The rated snapshot is
    # the fallback when use_full_snapshot is false.
    rated_snapshot_path = config.get_file_path(
        "snapshot_osm", "rated_snapshot",
    )
    full_snapshot_path = config.get_file_path("snapshot_osm", "snapshot")
    use_full_snapshot = bool(survivor_filter.get("use_full_snapshot", False))

    test_bbox = (
        config.get("conflation", "test_bbox") if args.test else None
    )

    print(f"Baseline: {baseline_path}")
    print(f"Ghosts:   {ghosts_path}")
    print(f"Fitted params: {fitted_params_path}")
    print(f"Output:   {output_path}")
    print(
        "Survivor-filter snapshot: "
        + (
            f"{full_snapshot_path} (full)" if use_full_snapshot
            else f"{rated_snapshot_path} (rated)"
        )
        + f", radius_m={survivor_filter.get('radius_m', 150)}"
    )
    print(
        f"min_match_score={min_match_score} "
        f"max_radius_m={max_radius_m} "
        f"default_delta={default_delta} "
        f"min_prior_name_match_score={min_prior_name_match_score} "
        f"max_ghost_age_years={max_ghost_age_years}"
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
        full_snapshot_path = full_snapshot_path,
        survivor_filter = survivor_filter,
        min_prior_name_match_score = min_prior_name_match_score,
        drop_unlabeled = drop_unlabeled,
        max_ghost_age_years = max_ghost_age_years,
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
