#!/usr/bin/env python
"""
Apply fitted existence-confidence calibration curves to a conflated dataset.

Reads a change-detected conflated parquet plus the per-segment curves written
by ``fit_calibration.py``, and writes a conflated parquet whose ``conf_mean`` /
``conf_lower`` / ``conf_upper`` are calibrated probabilities that the POI
exists and is open. The pre-calibration value is archived in
``conf_mean_uncalibrated`` and each edge rule is recorded in
``calibration_flag``.

Runs AFTER change detection: the CD penalty multiplies ``conf_mean`` by a
per-label delta, and calibrating first would leave a calibrated probability
scaled by ~0.14. Shadow-matched (CD-demoted) rows keep their penalized value.

Config keys used:
  - versions.conflation, versions.calibration
  - directories.conflation.files.conflated

Prerequisites:
  - make apply_cd (or apply_change_detection.py) has produced the input
  - scripts/conflation/fit_calibration.py has written the segment curves

Output file(s):
  - the suffixed conflated parquet (default: canonical conflated.parquet)

Usage:
    python scripts/conflation/apply_calibration.py \
        --input-suffix=cd --output-suffix="" [--test]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config_versioned import Config

from openpois.conflation import calibration


def _suffixed_path(base_path: Path, suffix: str | None) -> Path:
    """Insert ``suffix`` before the parquet extension."""
    if not suffix:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--input-suffix", default = "cd",
        help = ("Suffix of the input parquet (default: 'cd' -> "
                "conflated_cd.parquet, the change-detection output)."),
    )
    parser.add_argument(
        "--output-suffix", default = "",
        help = ("Suffix of the output parquet (default: none -> the canonical "
                "conflated.parquet that downstream steps read)."),
    )
    parser.add_argument(
        "--curves-dir", default = None,
        help = ("Directory holding {segment}_curve.parquet (default: the "
                "conflation version's calibration/ subdirectory)."),
    )
    parser.add_argument(
        "--test", action = "store_true",
        help = "Read/write the *_test.parquet variants.",
    )
    args = parser.parse_args()
    started = time.time()

    config = Config("~/repos/openpois/config.yaml")
    conflated_base = config.get_file_path("conflation", "conflated")
    if args.test:
        conflated_base = conflated_base.with_name(
            f"{conflated_base.stem}_test{conflated_base.suffix}"
        )
    input_path = _suffixed_path(conflated_base, args.input_suffix)
    output_path = _suffixed_path(conflated_base, args.output_suffix)
    if not input_path.exists():
        raise SystemExit(f"Input parquet not found: {input_path}")
    if output_path == input_path:
        raise SystemExit(
            "Input and output resolve to the same path; pass distinct suffixes"
        )

    curves_dir = Path(
        args.curves_dir
        or (config.get_dir_path("conflation") / "calibration")
    )
    curves = calibration.read_curves(curves_dir)
    metadata = calibration.read_curve_metadata(curves_dir)
    pool_params = calibration.pool_params_from_metadata(metadata)
    index_modes = calibration.index_modes_from_metadata(metadata)

    print(f"Calibration curves: {curves_dir}")
    for segment, lookup in sorted(curves.items()):
        meta = metadata.get(segment, {})
        print(f"  {segment}: {len(lookup)} bins, "
              f"index = {meta.get('score_definition', 'unknown')}")
    for segment, pool in sorted((pool_params or {}).items()):
        if pool:
            print(f"    {segment} pool: intercept {pool['intercept']:.4f}, "
                  f"osm {pool['coef_osm']:.4f}, "
                  f"overture {pool['coef_overture']:.4f} "
                  f"({pool['method']})")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    calibration.apply_calibration(
        input_path, output_path, curves, pool_params = pool_params,
        index_modes = index_modes,
    )
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
