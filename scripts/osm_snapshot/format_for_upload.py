"""
Partition the rated OSM snapshot by top-level tag for local queries.

Reads osm_snapshot_rated.parquet, derives a `primary_tag` per POI via first-
non-null across the configured `download.osm.filter_keys` priority order
(shop > healthcare > leisure > amenity > tourism > office > craft > historic,
matching the priority in `openpois.conflation.taxonomy.assign_osm_shared_label`),
adds a geohash sort key from each POI's centroid, and writes a Hive-style
dataset:

    osm_snapshot_partitioned/
        primary_tag=amenity/part-0.parquet
        primary_tag=shop/part-0.parquet
        ...

Rows within each partition are sorted by the `geohash` column so spatial
filters prune via Parquet row-group min/max stats. Queries like
``WHERE primary_tag = 'shop' AND shop = 'bakery'`` read a single partition
file.
"""
from config_versioned import Config

import pyarrow.parquet as pq

from openpois.io.geohash_partition import (
    compute_primary_osm_tag,
    write_label_partitioned_from_parquet,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("snapshot_osm", "rated_snapshot")
OUTPUT_DIR = config.get_file_path("snapshot_osm", "partitioned")
OVERWRITE = True

FILTER_KEYS = config.get("download", "osm", "filter_keys")
PRECISION_SORT = config.get("publish", "geohash_precision_sort")
PARTITION_COL = "primary_tag"

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # primary_tag is DERIVED, not stored, so it is computed in a narrow pass
    # over just the filter-key columns; the wide read then happens one
    # partition at a time. Loading the whole rated snapshot as a single
    # GeoDataFrame does not fit in memory at national scale.
    print(f"Deriving {PARTITION_COL} from filter_keys {FILTER_KEYS} ...")
    keys = pq.read_table(str(INPUT_PATH), columns = FILTER_KEYS).to_pandas()
    labels = compute_primary_osm_tag(
        keys, filter_keys = FILTER_KEYS, out_col = PARTITION_COL
    )[PARTITION_COL]
    del keys
    print(f"  {labels.notna().sum():,} of {len(labels):,} rows carry a "
          f"{PARTITION_COL}")

    print(f"Partitioning {INPUT_PATH} by {PARTITION_COL} ...")
    n_rows = write_label_partitioned_from_parquet(
        INPUT_PATH,
        output_dir = OUTPUT_DIR,
        partition_col = PARTITION_COL,
        geohash_precision = PRECISION_SORT,
        sort_col = "geohash",
        overwrite = OVERWRITE,
        labels = labels,
    )

    n_partitions = sum(1 for _ in OUTPUT_DIR.iterdir() if _.is_dir())
    print(
        f"Done. Wrote {n_rows:,} rows across {n_partitions} "
        f"{PARTITION_COL} partitions."
    )
    print(f"Output: {OUTPUT_DIR}")
