#!/usr/bin/env python
"""
Build the ghost-OSM POI dataset from OSM history.

A ghost is a previous state of an OSM node that we believe no longer
reflects ground truth (primary tag deleted, lifecycle prefix added, or
substantial rename). The output Parquet feeds the change-detection
pass in ``scripts/conflation/conflate.py``.

Config keys used (config.yaml):
    versions.osm_data, versions.ghost_osm        — pinned together
    directories.osm_data.osm_versions
    directories.osm_data.osm_changes
    directories.ghost_osm.ghosts
    download.osm.filter_keys                     — POI tag keys
    conflation.change_detection.name_change_similarity_threshold

Usage:
    python scripts/conflation/build_ghosts.py
"""
from __future__ import annotations

import time

from config_versioned import Config

from openpois.conflation.ghost_osm import build_ghosts


def main() -> None:
    config = Config("~/repos/openpois/config.yaml")

    versions_path = config.get_file_path("osm_data", "osm_versions")
    changes_path = config.get_file_path("osm_data", "osm_changes")
    output_path = config.get_file_path("ghost_osm", "ghosts")

    filter_keys = config.get("download", "osm", "filter_keys")
    name_threshold = float(
        config.get(
            "conflation", "change_detection",
            "name_change_similarity_threshold",
        )
    )

    print(f"Versions path: {versions_path}")
    print(f"Changes path:  {changes_path}")
    print(f"Output path:   {output_path}")
    print(f"POI keys:      {filter_keys}")
    print(f"Name similarity threshold: {name_threshold}")

    t0 = time.time()
    ghosts = build_ghosts(
        versions_path = versions_path,
        changes_path = changes_path,
        poi_keys = filter_keys,
        name_change_similarity_threshold = name_threshold,
    )
    elapsed = time.time() - t0
    print(f"\nBuilt {len(ghosts):,} ghosts in {elapsed:.0f}s")

    if len(ghosts):
        event_counts = (
            ghosts["event_type"].value_counts().to_dict()
        )
        print("Event-type breakdown:")
        for et, n in sorted(event_counts.items(), key = lambda kv: -kv[1]):
            print(f"  {et}: {n:,}")

        sl_total = int((ghosts["shared_label"] != "").sum())
        print(
            f"shared_label assigned: {sl_total:,}/{len(ghosts):,} "
            f"({100 * sl_total / max(len(ghosts), 1):.1f}%)"
        )

    output_path.parent.mkdir(parents = True, exist_ok = True)
    ghosts.to_parquet(output_path, compression = "zstd")
    print(f"\nWrote {output_path}")
    config.write_self("ghost_osm")


if __name__ == "__main__":
    main()
