#!/usr/bin/env python
"""
Drop unnamed private-property POIs using the landuse=residential layer.

An unnamed POI whose primary tag is in
``download.osm.residential_exclusion.scoped_tags`` and whose representative
point falls inside a ``landuse=residential`` polygon is dropped. See the
"Exclusion" section of docs/data-sources.md for the rationale, and
``openpois.osm.residential`` for the predicate.

TRANSITIONAL, in the same sense as ``apply_access_exclusion.py``: from the
2026-08 pull onward ``scripts/osm_snapshot/download.py`` applies this at
snapshot build time, so ``osm_snapshot.parquet`` arrives already filtered and
this script reports 0 dropped. It exists for the 2026-07 snapshot, which was
built before the wiring, and as the tuning tool — re-scoping the rule is a
config edit plus a re-run here, not another 11 GB Geofabrik pull.

``--target rated_snapshot`` filters the *scored* file so conflation can be
re-run without re-rating. Note that ``make rate`` regenerates the rated file
from ``osm_snapshot.parquet``, so filter the snapshot too (the default target)
or a later rating pass silently resurrects the rows.

Always pass ``--expect-kept`` when the target count is known — it is what
caught a pyarrow null-propagation bug that over-dropped 2.44M rows in the
access exclusion.

The pre-exclusion file is kept alongside the output as
``*.preresidential.parquet`` unless ``--no-archive`` is passed.

Config keys used (config.yaml):
    download.osm.residential_exclusion — landuse_values, scoped_tags
    download.osm.filter_keys           — primary-tag precedence
    snapshot_osm.residential_areas     — the polygon layer
    snapshot_osm.snapshot / rated_snapshot — input and output (rewritten)

Usage:
    python -u scripts/osm_snapshot/apply_residential_exclusion.py --report-only
    python -u scripts/osm_snapshot/apply_residential_exclusion.py \\
        --target rated_snapshot --expect-kept 4900000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from config_versioned import Config

from openpois.osm.residential import (
    filter_parquet_by_residential,
    load_residential_areas,
)

config = Config("~/repos/openpois/config.yaml")

RESIDENTIAL = config.get(
    "download", "osm", "residential_exclusion", fail_if_none = False,
) or {}
LANDUSE_VALUES = RESIDENTIAL.get("landuse_values") or []
SCOPED_TAGS = RESIDENTIAL.get("scoped_tags") or {}
FILTER_KEYS = config.get("download", "osm", "filter_keys")
AREAS_PATH = Path(config.get_file_path("snapshot_osm", "residential_areas"))


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--target", choices = ["snapshot", "rated_snapshot"],
        default = "snapshot",
        help = "which snapshot file to filter (default: snapshot)",
    )
    parser.add_argument(
        "--report-only", action = "store_true",
        help = "report what would drop without writing anything",
    )
    parser.add_argument(
        "--expect-kept", type = int, default = None,
        help = "fail if the kept row count differs (guards a silent "
        "predicate change between runs)",
    )
    parser.add_argument(
        "--no-archive", action = "store_true",
        help = "delete the pre-exclusion file instead of keeping it",
    )
    parser.add_argument(
        "--scope-override", default = None, metavar = "KEY=V1,V2;KEY=V3",
        help = "replace scoped_tags for this run only. Use with "
        "--report-only to measure a candidate scope before committing to it.",
    )
    args = parser.parse_args()

    target_path = Path(config.get_file_path("snapshot_osm", args.target))
    if not target_path.exists():
        raise SystemExit(f"No {args.target} at {target_path}")
    if not LANDUSE_VALUES:
        raise SystemExit(
            "download.osm.residential_exclusion.landuse_values is empty — "
            "the exclusion is disabled; nothing to do."
        )
    if not AREAS_PATH.exists():
        raise SystemExit(
            f"No landuse layer at {AREAS_PATH}.\n"
            "Build it first: python -u scripts/osm_snapshot/"
            "build_residential_areas.py"
        )

    scoped = SCOPED_TAGS
    if args.scope_override:
        scoped = {}
        for clause in args.scope_override.split(";"):
            key, _, values = clause.partition("=")
            scoped[key.strip()] = [v.strip() for v in values.split(",") if v.strip()]
        print(f"SCOPE OVERRIDE (this run only): {scoped}")
    if not scoped:
        raise SystemExit("scoped_tags is empty; nothing would be dropped.")

    areas = load_residential_areas(AREAS_PATH, LANDUSE_VALUES)
    print(f"Reading  {target_path}")

    if args.report_only:
        n_out, report = filter_parquet_by_residential(
            input_path = target_path,
            output_path = None,
            residential = areas,
            scoped_tags = scoped,
            filter_keys = FILTER_KEYS,
            verbose = False,
        )
        _print_report(target_path, n_out, report)
        return

    prefilter_path = target_path.with_suffix(".preresidential.parquet")
    target_path.rename(prefilter_path)
    try:
        n_out, report = filter_parquet_by_residential(
            input_path = prefilter_path,
            output_path = target_path,
            residential = areas,
            scoped_tags = scoped,
            filter_keys = FILTER_KEYS,
            verbose = True,
        )
    except Exception:
        # Restore the original so a failed run leaves nothing half-written.
        target_path.unlink(missing_ok = True)
        prefilter_path.rename(target_path)
        raise

    print(f"Wrote    {target_path}")
    _print_report(prefilter_path, n_out, report)

    if args.no_archive:
        prefilter_path.unlink()
        print(f"  removed {prefilter_path.name}")
    else:
        print(f"  pre-exclusion copy kept at {prefilter_path.name}")

    if args.expect_kept is not None and n_out != args.expect_kept:
        print(
            f"\nEXPECTED {args.expect_kept:,} kept rows, got {n_out:,}.",
            file = sys.stderr,
        )
        raise SystemExit(1)


def _print_report(source_path: Path, n_out: int, report: pd.DataFrame) -> None:
    import pyarrow.parquet as pq

    n_in = pq.read_metadata(source_path).num_rows
    dropped = n_in - n_out
    print(
        f"  {n_in:,} rows -> {n_out:,} kept "
        f"({dropped:,} dropped, {100 * dropped / max(n_in, 1):.2f}%)"
    )
    if report.empty:
        print("  nothing dropped.")
        return
    print("\n  dropped by primary tag:")
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(report.to_string(index = False))


if __name__ == "__main__":
    main()
