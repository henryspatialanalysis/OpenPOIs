"""
Download the current US + territories Overture Maps Places snapshot as a
GeoParquet file.

Queries Overture Maps GeoParquet files on public S3 using DuckDB's httpfs and
spatial extensions. Iterates the release's ``part-*.parquet`` files, writing a
bounded-memory DuckDB COPY per part into a ``.parts/<release>/`` directory.
Once every part is present, a single DuckDB COPY applies the exact
US + territories polygon filter and writes the final GeoParquet without
materializing rows in Python. Interrupted runs resume by skipping parts whose
intermediates already exist. No authentication required — Overture Maps data
is publicly accessible.

Auto-detects the latest available Overture release from S3 unless a specific
release_date is pinned in config.yaml.

Config keys used (config.yaml):
    download.overture.release_date           — pinned release (null = auto-detect)
    download.overture.s3_bucket              — Overture Maps S3 bucket name
    download.overture.s3_region              — AWS region of the Overture bucket
    download.overture.taxonomy_allowlist     — list of [L0, L1] pairs; L1 null = any
    download.overture.duckdb.memory_limit    — per-connection DuckDB memory cap
    download.overture.duckdb.threads         — per-connection DuckDB thread count
    download.overture.duckdb.workers         — parallel part downloads (must be 1)
    download.general.boundary.source_url     — Census state-boundary zip URL
    download.general.boundary.coastline_buffer_m — outward coastline buffer (m)
    directories.boundary                     — cache directory for boundary file
    directories.snapshot_overture            — output directory

Output file:
    overture_snapshot.parquet — GeoParquet with US + territories POIs
        Columns: overture_id, overture_name, taxonomy_l0, taxonomy_l1,
        taxonomy_l2, brand_name, confidence, geometry, source
"""
import shutil

import pyarrow.parquet as pq
from config_versioned import Config
from openpois.io.boundary import get_us_pr_boundary
from openpois.io.overture import download_overture_snapshot

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

# None = auto-detect latest
RELEASE_DATE = config.get("download", "overture", "release_date", fail_if_none=False)
S3_BUCKET = config.get("download", "overture", "s3_bucket")
S3_REGION = config.get("download", "overture", "s3_region")
TAXONOMY_ALLOWLIST = config.get("download", "overture", "taxonomy_allowlist")
DUCKDB_MEMORY_LIMIT = config.get(
    "download", "overture", "duckdb", "memory_limit", fail_if_none=False
) or "4GB"
DUCKDB_THREADS = config.get(
    "download", "overture", "duckdb", "threads", fail_if_none=False
) or 2
DUCKDB_WORKERS = config.get(
    "download", "overture", "duckdb", "workers", fail_if_none=False
) or 2
BOUNDARY_URL = config.get("download", "general", "boundary", "source_url")
COASTLINE_BUFFER_M = config.get(
    "download", "general", "boundary", "coastline_buffer_m"
)
BOUNDARY_DIR = config.get_dir_path("boundary")
SAVE_DIR = config.get_dir_path("snapshot_overture")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = config.get_file_path("snapshot_overture", "snapshot")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    boundary_gdf, coarse_bboxes = get_us_pr_boundary(
        source_url = BOUNDARY_URL,
        cache_dir = BOUNDARY_DIR,
        coastline_buffer_m = COASTLINE_BUFFER_M,
    )
    output_path = download_overture_snapshot(
        output_path = OUTPUT_PATH,
        taxonomy_allowlist = TAXONOMY_ALLOWLIST,
        boundary_gdf = boundary_gdf,
        coarse_bboxes = coarse_bboxes,
        bucket = S3_BUCKET,
        s3_region = S3_REGION,
        release_date = RELEASE_DATE,
        duckdb_memory_limit = DUCKDB_MEMORY_LIMIT,
        duckdb_threads = DUCKDB_THREADS,
        workers = DUCKDB_WORKERS,
    )
    n_rows = pq.read_metadata(output_path).num_rows
    print(f"Saved {n_rows:,} Overture POIs to {output_path}")

    # -------------------------------------------------------------------------
    # Clean up intermediates
    # -------------------------------------------------------------------------
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        parts_dir = SAVE_DIR / ".parts"
        if parts_dir.is_dir():
            print(f"Removing intermediate {parts_dir} ...")
            shutil.rmtree(parts_dir)
