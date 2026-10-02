#!/usr/bin/env python
"""
Apply the hand-curated manual overrides to the calibrated conflated dataset.

Runs LAST in ``make conflate`` — after ``apply_calibration.py`` — so a forced
``conf_mean`` is never re-scaled by the calibration curves. Reads the
canonical ``conflated.parquet`` and rewrites it in place (atomic swap) with
``exclude`` rows forced to ``conf_mean = conf_lower = conf_upper = 0``
(``calibration_flag = 'manual_exclude'``) and ``include`` rows forced to 1
(``calibration_flag = 'manual_include'``). Idempotent.

Config keys used:
  - versions.conflation, versions.manual_overrides
  - directories.conflation.files.conflated
  - directories.manual_overrides.files.overrides   (default CSV location)
  - conflation.manual_overrides.enabled / .path    (.path overrides the
    versioned default when set)

A missing CSV is a no-op: the stage logs one line and leaves the canonical
file untouched.

Usage:
    python scripts/conflation/apply_manual_overrides.py [--test]
    python scripts/conflation/apply_manual_overrides.py \
        --input-suffix="" --output-suffix=overridden      # ablation copy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from config_versioned import Config

from openpois.conflation import manual_overrides


def _suffixed_path(base_path: Path, suffix: str | None) -> Path:
    """Insert ``suffix`` before the parquet extension."""
    if not suffix:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def resolve_overrides_path(config: Config, explicit: str | None = None) -> Path:
    """The CSV to read: CLI flag > ``conflation.manual_overrides.path`` >
    the versioned ``directories.manual_overrides`` file."""
    if explicit:
        return Path(explicit).expanduser()
    mo_cfg = config.get("conflation", "manual_overrides", fail_if_none = False)
    mo_cfg = mo_cfg or {}
    configured = mo_cfg.get("path")
    if configured:
        return Path(str(configured)).expanduser()
    return config.get_file_path("manual_overrides", "overrides")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--input-suffix", default = "",
        help = ("Suffix of the input parquet (default: none -> the canonical "
                "conflated.parquet written by apply_calibration.py)."),
    )
    parser.add_argument(
        "--output-suffix", default = "",
        help = ("Suffix of the output parquet (default: none -> rewrite the "
                "canonical file in place)."),
    )
    parser.add_argument(
        "--overrides", default = None,
        help = "Explicit CSV path (default: resolved from config.yaml).",
    )
    parser.add_argument(
        "--test", action = "store_true",
        help = "Read/write the *_test.parquet variants.",
    )
    args = parser.parse_args()
    started = time.time()

    config = Config("~/repos/openpois/config.yaml")
    mo_cfg = config.get("conflation", "manual_overrides", fail_if_none = False)
    mo_cfg = mo_cfg or {}
    if not bool(mo_cfg.get("enabled", True)):
        print("conflation.manual_overrides.enabled is false; nothing to do.")
        return

    conflated_base = config.get_file_path("conflation", "conflated")
    if args.test:
        conflated_base = conflated_base.with_name(
            f"{conflated_base.stem}_test{conflated_base.suffix}"
        )
    input_path = _suffixed_path(conflated_base, args.input_suffix)
    output_path = _suffixed_path(conflated_base, args.output_suffix)
    if not input_path.exists():
        raise SystemExit(f"Input parquet not found: {input_path}")

    overrides_path = resolve_overrides_path(config, args.overrides)
    print(f"Overrides: {overrides_path}")
    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}"
          + (" (in place)" if output_path == input_path else ""))
    if not overrides_path.exists():
        print(
            f"No manual overrides file at {overrides_path}; nothing to do "
            f"({input_path.name} left as calibrated)."
        )
        return

    overrides = manual_overrides.read_overrides(overrides_path)
    print(f"  {len(overrides):,} override row(s): "
          + ", ".join(
              f"{n} {a}" for a, n in
              overrides["action"].value_counts().sort_index().items()
          ))
    stats = manual_overrides.apply_manual_overrides(
        input_path, output_path, overrides,
    )
    print(f"Done in {time.time() - started:.1f}s "
          f"({stats['n_excluded']:,} excluded, {stats['n_included']:,} "
          f"included, {stats['n_unmatched_overrides']:,} unmatched)")


if __name__ == "__main__":
    main()
