#!/usr/bin/env python
"""
Build the landuse=residential polygon layer used by the POI exclusion.

The POI ingest filter is derived from the taxonomy crosswalk and value-scopes
``landuse`` to ``cemetery,religious``, so residential polygons never reach the
filtered POI PBF. They need their own osmium pass over the *raw* Geofabrik
extracts — which ``scripts/osm_snapshot/download.py`` unlinks once it has a
snapshot, so a standalone run re-downloads them (~11 GB for the US extract,
resumable via ``download_resilient``).

The result is persisted next to the snapshot as ``landuse_residential.parquet``
so retuning the exclusion never costs another Geofabrik pull.

In the monthly flow this runs automatically inside ``download.py`` while the
raw PBFs are still on disk; use this script for a rebuild, or to build the
layer for a snapshot version that predates the wiring.

Config keys used (config.yaml):
    download.osm.pbf_url, pr_pbf_url, usvi_pbf_url, american_oceania_pbf_url
    download.osm.residential_exclusion.landuse_values
    directories.snapshot_osm.residential_areas — output
    directories.snapshot_osm.residential_*_filtered_pbf — intermediates

Usage:
    python -u scripts/osm_snapshot/build_residential_areas.py
    python -u scripts/osm_snapshot/build_residential_areas.py --overwrite
    python -u scripts/osm_snapshot/build_residential_areas.py --extracts us
"""
from __future__ import annotations

import argparse

from config_versioned import Config

from openpois.io.osm_snapshot import SnapshotExtract
from openpois.osm.residential import DEFAULT_MAX_AREA_NODES, build_residential_layer

config = Config("~/repos/openpois/config.yaml")

RESIDENTIAL = config.get(
    "download", "osm", "residential_exclusion", fail_if_none = False,
) or {}
LANDUSE_VALUES = RESIDENTIAL.get("landuse_values") or []
OUTPUT_PATH = config.get_file_path("snapshot_osm", "residential_areas")
SAVE_DIR = config.get_dir_path("snapshot_osm")

# Same four Geofabrik extracts as the POI snapshot, but pointed at the
# residential filtered-PBF paths so the two passes never collide.
EXTRACTS = [
    SnapshotExtract(
        name = "us",
        url = config.get("download", "osm", "pbf_url"),
        raw_pbf_path = config.get_file_path("snapshot_osm", "raw_pbf"),
        filtered_pbf_path = config.get_file_path(
            "snapshot_osm", "residential_filtered_pbf"
        ),
    ),
    SnapshotExtract(
        name = "pr",
        url = config.get("download", "osm", "pr_pbf_url"),
        raw_pbf_path = config.get_file_path("snapshot_osm", "raw_pr_pbf"),
        filtered_pbf_path = config.get_file_path(
            "snapshot_osm", "residential_pr_filtered_pbf"
        ),
    ),
    SnapshotExtract(
        name = "usvi",
        url = config.get("download", "osm", "usvi_pbf_url"),
        raw_pbf_path = config.get_file_path("snapshot_osm", "raw_usvi_pbf"),
        filtered_pbf_path = config.get_file_path(
            "snapshot_osm", "residential_usvi_filtered_pbf"
        ),
    ),
    SnapshotExtract(
        name = "american_oceania",
        url = config.get("download", "osm", "american_oceania_pbf_url"),
        raw_pbf_path = config.get_file_path(
            "snapshot_osm", "raw_american_oceania_pbf"
        ),
        filtered_pbf_path = config.get_file_path(
            "snapshot_osm", "residential_american_oceania_filtered_pbf"
        ),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--overwrite", action = "store_true",
        help = "rebuild even if the layer already exists",
    )
    parser.add_argument(
        "--extracts", nargs = "+", default = None,
        metavar = "NAME",
        help = "subset of extracts to build (default: all four). Only useful "
        "for debugging — a partial layer under-reports coverage.",
    )
    parser.add_argument(
        "--keep-intermediates", action = "store_true",
        help = "keep the filtered PBFs and per-extract parquets",
    )
    parser.add_argument(
        "--max-area-nodes", type = int, default = DEFAULT_MAX_AREA_NODES,
        help = "node ceiling for relation-derived areas "
        f"(default: {DEFAULT_MAX_AREA_NODES:,})",
    )
    args = parser.parse_args()

    if not LANDUSE_VALUES:
        raise SystemExit(
            "download.osm.residential_exclusion.landuse_values is empty — "
            "the exclusion is disabled, so there is no layer to build."
        )

    extracts = EXTRACTS
    if args.extracts:
        known = {e.name for e in EXTRACTS}
        unknown = set(args.extracts) - known
        if unknown:
            raise SystemExit(f"unknown extract(s): {sorted(unknown)}")
        extracts = [e for e in EXTRACTS if e.name in set(args.extracts)]
        print(f"WARNING: partial build ({', '.join(args.extracts)}).")

    SAVE_DIR.mkdir(parents = True, exist_ok = True)
    print(f"Landuse values: {LANDUSE_VALUES}")
    print(f"Output:         {OUTPUT_PATH}")

    build_residential_layer(
        extracts = extracts,
        output_path = OUTPUT_PATH,
        landuse_values = LANDUSE_VALUES,
        chunk_dir = SAVE_DIR,
        max_area_nodes = args.max_area_nodes,
        overwrite = args.overwrite,
        keep_intermediates = args.keep_intermediates,
        verbose = True,
    )


if __name__ == "__main__":
    main()
