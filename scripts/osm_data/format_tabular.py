"""
Reformat raw OSM version histories into modelling-ready observations, tagged
with a shared taxonomy label plus the ``msa_code`` and ``urban_rural``
indicators.

Reads osm_versions.parquet and osm_changes.parquet (produced by
osm_data/download_history.py). DuckDB streams them into an observation-per-
version intermediate via ``format_observations_duckdb``. Each observation
records the value of ``osm_data.tag_key`` (the change event — the tag whose
add/change/delete fires ``changed=1``), timestamps of the previous tag
assignment and the current observation, and the current values of every
``download.osm.filter_keys`` tag.

The build then runs in a single streaming pass over the intermediate:

  1. ``msa_code`` / ``urban_rural`` are assigned **per element** from each
     element's most-recent location (snapshot geometry, with a node lat/lon
     fallback from osm_changes), then broadcast to all of that element's rows —
     one spatial join per unique element, not per observation row.
  2. Each batch is assigned zero or more ``shared_label`` values via the
     conflation taxonomy crosswalk and exploded so a POI version contributing to
     multiple taxonomy categories produces one row per category. Rows with no
     matching taxonomy category are dropped.

Streaming the label-assignment + explode (rather than loading the whole
observations file into pandas) keeps peak memory bounded, matching the
memory-boundedness goal of the DuckDB-stream observation builder.

Config keys used (config.yaml):
    directories.osm_data          — input and output files
    directories.snapshot_osm      — current snapshot (element coordinates)
    directories.census_areas      — cached CBSA / Place / population
    download.census_areas         — Census source URLs
    download.osm.filter_keys      — all tag keys collected (keep_keys AND used
                                    by taxonomy assignment)
    osm_data.tag_key              — single tag key whose changes define
                                    observation events (e.g. "name")
    osm_data.lifecycle_closure_prefixes
                                  — lifecycle namespaces (disused:, was:, …)
                                    whose appearance on a primary filter_keys
                                    tag is a closure (soft-delete) event; []
                                    disables (name-only signal, for ablation)

Prerequisites:
    Run osm_data/download_history.py first to produce osm_versions.parquet and
    osm_changes.parquet. The current OSM snapshot must also exist (used for
    element coordinates).

Output file (in osm_data directory):
    osm_observations.parquet — one row per (POI version, shared_label). Columns:
        id, osm_type, version, tag_key, last_tag_timestamp, obs_timestamp,
        changed, msa_code, urban_rural, shared_label, plus every filter_keys
        column for reference.
"""

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from config_versioned import Config

from openpois.conflation.taxonomy import (
    assign_osm_shared_label,
    load_match_radii,
    load_osm_crosswalk,
)
from openpois.io import census_areas, indicators
from openpois.osm.format_observations import format_observations_window


# ----------------------------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

SAVE_DIR = config.get_dir_path("osm_data")
OSM_KEYS = config.get("download", "osm", "filter_keys")
TAG_KEY = config.get("osm_data", "tag_key")
# Lifecycle prefixes whose appearance on a POI primary tag is a closure event.
# Optional; an empty list / missing key disables the behavior (ablation).
LIFECYCLE_PREFIXES = (
    config.get("osm_data", "lifecycle_closure_prefixes", fail_if_none = False)
    or []
)

CHANGES_PATH = config.get_file_path("osm_data", "osm_changes")
VERSIONS_PATH = config.get_file_path("osm_data", "osm_versions")
OUT_PATH = config.get_file_path("osm_data", "osm_observations")
# Raw (pre-label, pre-enrichment) intermediate. Transient; removed at the end.
RAW_PATH = SAVE_DIR / "osm_observations_raw.parquet"

SNAPSHOT_PATH = config.get_file_path("snapshot_osm", "snapshot")

# Census reference areas (downloaded + cached on first run).
CENSUS_DIR = config.get_dir_path("census_areas")
CBSA_URL = config.get("download", "census_areas", "cbsa_url")
PLACE_URL = config.get("download", "census_areas", "place_url")
CBSA_SHP_NAME = config.get("download", "census_areas", "cbsa_shp_name")
PLACE_SHP_NAME = config.get("download", "census_areas", "place_shp_name")
POPULATION_API_URL = config.get("download", "census_areas", "population_api_url")
CBSA_SHP = config.get_file_path("census_areas", "cbsa_shapefile")
PLACE_SHP = config.get_file_path("census_areas", "place_shapefile")
POPULATION_CSV = config.get_file_path("census_areas", "place_population")

# DuckDB execution limits. The sort operator spills past memory_limit so this
# caps peak RAM independent of input size. Threads default to os.cpu_count()
# when left as None.
DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_THREADS = None

BATCH_ROWS = 200_000


def ensure_census_layers():
    """Download (if needed) and load MSAs + classified places."""
    census_areas.download_census_zip(CBSA_URL, CENSUS_DIR, CBSA_SHP_NAME)
    census_areas.download_census_zip(PLACE_URL, CENSUS_DIR, PLACE_SHP_NAME)
    census_areas.fetch_place_population(
        POPULATION_CSV, api_url = POPULATION_API_URL
    )
    return indicators.load_classified_layers(CBSA_SHP, PLACE_SHP, POPULATION_CSV)


def build_element_indicator_map(msa_gdf, classified_places_gdf) -> pd.DataFrame:
    """One ``(msa_code, urban_rural)`` per distinct ``(osm_type, id)`` element."""
    con = duckdb.connect()
    try:
        elements = con.execute(
            f"SELECT DISTINCT osm_type, id "
            f"FROM read_parquet('{RAW_PATH.as_posix()}')"
        ).fetch_df()
    finally:
        con.close()
    print(f"Locating {len(elements):,} distinct elements for indicators ...")
    return indicators.element_indicator_map(
        elements,
        SNAPSHOT_PATH,
        msa_gdf,
        classified_places_gdf,
        changes_path = CHANGES_PATH,
    )


# ----------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Stage 2b: DuckDB window functions (bucketed for bounded memory), producing
    # observations byte-for-byte identical to the legacy state machine (gated by
    # tests/test_format_observations.py) but ~2-3x faster.
    n_written = format_observations_window(
        changes_path = CHANGES_PATH,
        versions_path = VERSIONS_PATH,
        output_path = RAW_PATH,
        tag_key = TAG_KEY,
        keep_keys = OSM_KEYS,
        lifecycle_prefixes = LIFECYCLE_PREFIXES,
        duckdb_memory_limit = DUCKDB_MEMORY_LIMIT,
        duckdb_threads = DUCKDB_THREADS,
    )
    print(f"DuckDB wrote {n_written:,} raw observations to {RAW_PATH}")

    print("Preparing Census indicator layers ...")
    msa_gdf, classified_places_gdf = ensure_census_layers()
    element_indicators = build_element_indicator_map(
        msa_gdf, classified_places_gdf
    )
    print(
        "Element indicator coverage: in-MSA "
        f"{100 * (element_indicators['msa_code'] != indicators.NO_MSA).mean():.1f}%; "
        "urban/suburban/rural = "
        + element_indicators["urban_rural"].value_counts().to_dict().__str__()
    )

    osm_crosswalk = load_osm_crosswalk()
    match_radii = load_match_radii()

    # Stream the raw observations: per batch, assign shared labels, explode, and
    # merge the per-element indicators. Memory stays bounded to one batch.
    print("Assigning shared labels + indicators (streaming, exploded) ...")
    pf = pq.ParquetFile(RAW_PATH)
    # Pin a stable output schema (raw columns + the three new string columns) and
    # cast every batch to it, so per-batch type inference (e.g. an all-null
    # column landing as null type) can't drift the ParquetWriter schema.
    target_schema = pa.schema(
        list(pf.schema_arrow)
        + [
            ("shared_label", pa.string()),
            ("msa_code", pa.string()),
            ("urban_rural", pa.string()),
        ]
    )
    n_in = 0
    n_out = 0
    writer = None
    OUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    try:
        for batch in pf.iter_batches(batch_size = BATCH_ROWS):
            df = batch.to_pandas()
            n_in += len(df)
            labels_per_row, _ = assign_osm_shared_label(
                df, osm_crosswalk, match_radii, OSM_KEYS, return_all = True,
            )
            df["shared_label"] = labels_per_row
            df = df.explode("shared_label", ignore_index = True)
            df = df.dropna(subset = ["shared_label"])
            df = df[df["shared_label"] != ""]
            if df.empty:
                continue
            df = df.merge(
                element_indicators, on = ["osm_type", "id"], how = "left"
            )
            df["msa_code"] = df["msa_code"].fillna(indicators.NO_MSA)
            df["urban_rural"] = df["urban_rural"].fillna(indicators.RURAL)
            table = (
                pa.Table.from_pandas(df, preserve_index = False)
                .select(target_schema.names)
                .cast(target_schema)
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    OUT_PATH, target_schema, compression = "zstd"
                )
            writer.write_table(table)
            n_out += len(df)
            print(f"  {n_in:,} raw rows in → {n_out:,} (POI, shared_label) rows out")
    finally:
        if writer is not None:
            writer.close()

    RAW_PATH.unlink(missing_ok = True)
    print(f"Saved {n_out:,} observations to {OUT_PATH}")
