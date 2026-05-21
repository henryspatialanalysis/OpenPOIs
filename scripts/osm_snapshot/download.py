"""
Download the current US + inhabited-territories OpenStreetMap POI snapshot
as a GeoParquet file.

Downloads four Geofabrik PBF extracts — the US-mainland extract (~11 GB,
covers all 50 states incl. AK + HI), Puerto Rico, US Virgin Islands, and
American Oceania (Guam, NMI, American Samoa, plus uninhabited US Pacific
possessions) — uses osmium tags-filter to extract nodes and ways matching
the configured tag keys, parses each result with pyosmium, concatenates the
per-extract parquets, and saves as GeoParquet. Incremental: skips any PBF
download or filter step whose output file already exists (controlled by
overwrite_download and overwrite_filter config flags).

Note: osmium is resolved from the conda env bin rather than the shell PATH;
no manual PATH modification is needed.

Config keys used (config.yaml):
    download.osm.pbf_url                      — Geofabrik US PBF URL (50 states)
    download.osm.pr_pbf_url                   — Geofabrik Puerto Rico PBF URL
    download.osm.usvi_pbf_url                 — Geofabrik US Virgin Islands PBF URL
    download.osm.american_oceania_pbf_url     — Geofabrik American Oceania PBF URL
    download.osm.filter_keys         — OSM tag keys to retain (e.g. amenity, shop)
    download.osm.extract_keys        — tag keys to include as output columns
    download.osm.overwrite_download  — re-download PBFs even if they already exist
    download.osm.overwrite_filter    — re-run osmium filter even if output exists
    download.osm.source_label        — value written to the "source" column
    download.osm.keep_all_keys       — retain all discovered tag columns in output
    download.osm.chunk_size          — number of elements per pyosmium parse chunk
    download.osm.max_area_nodes      — skip way geometries with more nodes than this
    download.osm.verbose             — print progress during PBF parsing
    directories.snapshot_osm         — output directory; also used for temp PBF files

Output file:
    osm_snapshot.parquet — GeoParquet with US + territories POIs (nodes + area centroids)
        Columns: osm_id, osm_type, name, geometry, last_edited, source,
        plus all extract_keys columns
"""
from config_versioned import Config
from openpois.io.osm_snapshot import SnapshotExtract, download_osm_snapshot

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

FILTER_KEYS = config.get("download", "osm", "filter_keys")
EXTRACT_KEYS = config.get("download", "osm", "extract_keys")
OVERWRITE_DOWNLOAD = config.get("download", "osm", "overwrite_download")
OVERWRITE_FILTER = config.get("download", "osm", "overwrite_filter")
SOURCE_LABEL = config.get("download", "osm", "source_label")
KEEP_ALL_KEYS = config.get("download", "osm", "keep_all_keys")
CHUNK_SIZE = config.get("download", "osm", "chunk_size")
MAX_AREA_NODES = config.get("download", "osm", "max_area_nodes", fail_if_none = False)
VERBOSE = config.get("download", "osm", "verbose")
SAVE_DIR = config.get_dir_path("snapshot_osm")
CHUNK_DIR = config.get_dir_path("snapshot_osm")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = config.get_file_path("snapshot_osm", "snapshot")

# One SnapshotExtract per Geofabrik PBF. Order is preserved through to the
# concat step; keep the US-mainland extract first since it dominates wall time.
EXTRACTS = [
    SnapshotExtract(
        name = "us",
        url = config.get("download", "osm", "pbf_url"),
        raw_pbf_path = config.get_file_path("snapshot_osm", "raw_pbf"),
        filtered_pbf_path = config.get_file_path("snapshot_osm", "filtered_pbf"),
    ),
    SnapshotExtract(
        name = "pr",
        url = config.get("download", "osm", "pr_pbf_url"),
        raw_pbf_path = config.get_file_path("snapshot_osm", "raw_pr_pbf"),
        filtered_pbf_path = config.get_file_path("snapshot_osm", "filtered_pr_pbf"),
    ),
    SnapshotExtract(
        name = "usvi",
        url = config.get("download", "osm", "usvi_pbf_url"),
        raw_pbf_path = config.get_file_path("snapshot_osm", "raw_usvi_pbf"),
        filtered_pbf_path = config.get_file_path("snapshot_osm", "filtered_usvi_pbf"),
    ),
    SnapshotExtract(
        name = "american_oceania",
        url = config.get("download", "osm", "american_oceania_pbf_url"),
        raw_pbf_path = config.get_file_path(
            "snapshot_osm", "raw_american_oceania_pbf"
        ),
        filtered_pbf_path = config.get_file_path(
            "snapshot_osm", "filtered_american_oceania_pbf"
        ),
    ),
]


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    download_osm_snapshot(
        extracts = EXTRACTS,
        output_path = OUTPUT_PATH,
        filter_keys = FILTER_KEYS,
        extract_keys = EXTRACT_KEYS,
        overwrite_download = OVERWRITE_DOWNLOAD,
        overwrite_filter = OVERWRITE_FILTER,
        source_label = SOURCE_LABEL,
        keep_all_keys = KEEP_ALL_KEYS,
        chunk_size = CHUNK_SIZE,
        max_area_nodes = MAX_AREA_NODES,
        chunk_dir = CHUNK_DIR,
        verbose = VERBOSE,
    )

    # -------------------------------------------------------------------------
    # Clean up intermediates
    # -------------------------------------------------------------------------
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        intermediates = [
            p for spec in EXTRACTS
            for p in (spec.raw_pbf_path, spec.filtered_pbf_path)
        ]
        for p in intermediates:
            if p.exists():
                print(f"Removing intermediate {p} ...")
                p.unlink()
