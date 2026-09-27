#!/usr/bin/env python
"""
QA: compare the OSM history's last state of each snapshot node with the
snapshot itself.

For every node in ``snapshot_osm``'s ``osm_snapshot.parquet`` last edited
before the history's ``coverage_end``, ``osm_data`` must hold a last version
with the same timestamp and name. A match rate below 99.5% means the history
missed edits (a skipped diff, a bad fold) and ghosts may be incomplete.
Works for full and incremental history builds alike.

Config keys used (config.yaml):
    versions.osm_data, versions.snapshot_osm
    directories.osm_data, directories.snapshot_osm.files.snapshot

Usage:
    python scripts/osm_data/check_history_vs_snapshot.py
"""
from __future__ import annotations

import sys

from config_versioned import Config

from openpois.io.osm_history_incremental import (
    check_history_vs_snapshot,
    read_coverage,
)

MIN_MATCH_RATE = 0.995


def main() -> int:
    config = Config("~/repos/openpois/config.yaml")
    osm_data_dir = config.get_dir_path("osm_data")
    snapshot_path = config.get_file_path("snapshot_osm", "snapshot")
    coverage = read_coverage(osm_data_dir)
    print(f"History:  {osm_data_dir} ({coverage.mode})")
    print(f"Snapshot: {snapshot_path}")
    print(f"Checking snapshot nodes last edited before {coverage.coverage_end}")

    res = check_history_vs_snapshot(
        osm_data_dir, snapshot_path, coverage.coverage_end,
    )
    print(f"  nodes checked:        {res['n_checked']:,}")
    print(f"  missing from history: {res['n_missing']:,}")
    print(f"  timestamp match:      {res['ts_match_rate']:.3%}")
    print(f"  name match (of those): {res['name_match_rate']:.3%}")
    ok = res["ts_match_rate"] >= MIN_MATCH_RATE
    print("PASS" if ok else f"FLAG: timestamp match below {MIN_MATCH_RATE:.1%}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
