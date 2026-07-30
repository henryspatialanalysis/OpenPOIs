"""
Partition the conflated POI dataset by destination type for local queries.

Reads conflated.parquet, adds a geohash sort key from each POI's centroid,
and writes a Hive-style dataset partitioned by `shared_label`:

    conflated_partitioned/
        shared_label=Pharmacy/part-0.parquet
        shared_label=Restaurant/part-0.parquet
        ...

Rows within each partition are sorted by the `geohash` column so spatial
filters prune via Parquet row-group min/max stats. Queries like
``WHERE shared_label = 'Pharmacy'`` read a single partition file.
"""
from config_versioned import Config

from openpois.io.geohash_partition import (
    write_label_partitioned_from_parquet,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("conflation", "conflated")
OUTPUT_DIR = config.get_file_path("conflation", "partitioned")
OVERWRITE = True

PRECISION_SORT = config.get("publish", "geohash_precision_sort")
PARTITION_COL = "shared_label"

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Streamed one partition at a time: reading all 14.6M rows x 58 columns as
    # a single GeoDataFrame peaks around 21.5 GB and does not fit.
    print(f"Partitioning {INPUT_PATH} by {PARTITION_COL} ...")
    n_rows = write_label_partitioned_from_parquet(
        INPUT_PATH,
        output_dir = OUTPUT_DIR,
        partition_col = PARTITION_COL,
        geohash_precision = PRECISION_SORT,
        sort_col = "geohash",
        overwrite = OVERWRITE,
    )

    n_partitions = sum(1 for _ in OUTPUT_DIR.iterdir() if _.is_dir())
    print(
        f"Done. Wrote {n_rows:,} rows across {n_partitions} "
        f"{PARTITION_COL} partitions."
    )
    print(f"Output: {OUTPUT_DIR}")
