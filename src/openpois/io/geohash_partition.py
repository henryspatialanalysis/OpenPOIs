"""Utilities for partitioning GeoDataFrames for downstream query workloads.

Two partition styles are supported:

* Geohash-based (``add_geohash_columns`` + ``write_partitioned_dataset``) —
  optimized for web map viewport queries, where clients fetch only the
  geohash cells covering a bbox.
* Label-based (``write_label_partitioned_dataset``) — optimized for
  nationwide local queries filtered by destination type (e.g., a
  ``shared_label`` on conflated POIs, or a derived ``primary_tag`` on
  OSM POIs). Row-group-level geohash sort is still used within each
  partition so spatial filters prune efficiently.
"""
import shutil
import urllib.parse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pygeohash
import shapely


def add_geohash_columns(
    gdf: gpd.GeoDataFrame,
    precision_partition: int,
    precision_sort: int,
) -> gpd.GeoDataFrame:
    """Add geohash_prefix (partition key) and geohash_sort columns from centroids.

    Rows with null or empty geometries are dropped before computing hashes.
    Both columns are derived from the geometry centroid, so Points, Polygons,
    and MultiPolygons are all handled uniformly.

    Geohash is a prefix code, so the partition hash equals the first
    ``precision_partition`` characters of the sort hash. We encode once at
    the higher precision and derive the shorter prefix by string slicing,
    avoiding a second pass over N Shapely Points.
    """
    mask = ~gdf.geometry.is_empty & gdf.geometry.notna()
    if not mask.all():
        gdf = gdf[mask].reset_index(drop = True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Geometry is in a geographic CRS", UserWarning
        )
        centroids = shapely.centroid(gdf.geometry.to_numpy())
    lats = shapely.get_y(centroids)
    lons = shapely.get_x(centroids)
    del centroids

    sort_hashes = [
        pygeohash.encode(float(lat), float(lon), precision = precision_sort)
        for lat, lon in zip(lats, lons)
    ]
    gdf["geohash_sort"] = sort_hashes
    gdf["geohash_prefix"] = [h[:precision_partition] for h in sort_hashes]
    return gdf


def write_partitioned_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir,
    overwrite: bool = True,
) -> None:
    """Sort gdf spatially and write as a geohash-partitioned parquet dataset.

    Writes one parquet file per geohash_prefix value into a Hive-style directory
    layout (geohash_prefix=9q/part-0.parquet). Converts and writes one partition
    at a time to avoid duplicating the full dataset in memory.

    The geohash_prefix column becomes the Hive partition directory name and is
    dropped from the stored parquet files. The geohash_sort column is used only
    for row ordering and is also dropped before writing.
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        if overwrite:
            print(f"Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass overwrite=True to replace it."
            )

    cols = [c for c in gdf.columns if c not in ("geohash_prefix", "geohash_sort")]
    output_dir.mkdir(parents = True, exist_ok = True)

    # Iterate without a global sort_values: that would double peak memory on
    # multi-GB frames. groupby(sort = False) hands us each partition as a view;
    # each small partition is sorted in-place before writing.
    groups = gdf.groupby("geohash_prefix", sort = False, observed = True)
    n_partitions = len(groups)
    print(f"Writing {n_partitions} partitions to {output_dir} ...")
    for i, (prefix, group) in enumerate(groups):
        partition_dir = output_dir / f"geohash_prefix={prefix}"
        partition_dir.mkdir()
        group.sort_values("geohash_sort")[cols].to_parquet(
            partition_dir / "part-0.parquet",
            write_covering_bbox = True,
            row_group_size = 100_000,
        )
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_partitions} partitions written...")


def add_geohash_column(
    gdf: gpd.GeoDataFrame,
    precision: int,
    out_col: str = "geohash",
) -> gpd.GeoDataFrame:
    """Add a single geohash column at the given precision from centroids.

    Thin variant of :func:`add_geohash_columns` for layouts that don't need
    a separate partition prefix. Used by the label-partitioned writer to
    place a sort key on the rows.
    """
    mask = ~gdf.geometry.is_empty & gdf.geometry.notna()
    if not mask.all():
        gdf = gdf[mask].reset_index(drop = True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Geometry is in a geographic CRS", UserWarning
        )
        centroids = shapely.centroid(gdf.geometry.to_numpy())
    lats = shapely.get_y(centroids)
    lons = shapely.get_x(centroids)
    del centroids

    gdf[out_col] = [
        pygeohash.encode(float(lat), float(lon), precision = precision)
        for lat, lon in zip(lats, lons)
    ]
    return gdf


def compute_primary_osm_tag(
    gdf: gpd.GeoDataFrame,
    filter_keys: list[str],
    out_col: str = "primary_tag",
) -> gpd.GeoDataFrame:
    """Assign each row the first non-null tag key from ``filter_keys``.

    Mirrors the first-match-wins priority in
    :func:`openpois.conflation.taxonomy.assign_osm_shared_label` so that
    OSM-only partitioning and conflation-time labeling agree on which tag
    is primary for multi-tagged POIs (~1.9% of the rated snapshot).
    """
    missing = [k for k in filter_keys if k not in gdf.columns]
    if missing:
        raise KeyError(f"filter_keys missing from gdf: {missing}")

    primary = pd.Series(pd.NA, index = gdf.index, dtype = "string")
    for key in filter_keys:
        unassigned = primary.isna() & gdf[key].notna()
        primary.loc[unassigned] = key
    gdf[out_col] = primary
    return gdf


def write_label_partitioned_dataset(
    gdf: gpd.GeoDataFrame,
    output_dir,
    partition_col: str,
    sort_col: str = "geohash",
    overwrite: bool = True,
) -> None:
    """Hive-partition a GeoDataFrame by ``partition_col``, writing one
    parquet file per distinct value.

    Rows within each partition are sorted by ``sort_col`` for spatial
    locality. ``partition_col`` is dropped from the stored files (it lives
    in the Hive directory name); ``sort_col`` is retained so downstream
    queries can filter on it directly and benefit from Parquet row-group
    min/max pruning.

    Partition values that are not alphanumeric (e.g., ``"Fast Food
    Restaurant"``) are URL-encoded in the directory name. DuckDB's
    ``hive_partitioning=1`` decodes these transparently at read time.
    """
    output_dir = Path(output_dir)

    if partition_col not in gdf.columns:
        raise KeyError(f"partition_col not in gdf: {partition_col!r}")
    if sort_col not in gdf.columns:
        raise KeyError(f"sort_col not in gdf: {sort_col!r}")

    if output_dir.exists():
        if overwrite:
            print(f"Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass overwrite=True to replace it."
            )

    # Drop rows with null partition value — they can't be addressed under any
    # partition key and would silently disappear into a `__HIVE_DEFAULT_PARTITION__`
    # bucket otherwise.
    null_mask = gdf[partition_col].isna()
    if null_mask.any():
        n_null = int(null_mask.sum())
        print(f"Skipping {n_null:,} rows with null {partition_col}")
        gdf = gdf[~null_mask]

    cols = [c for c in gdf.columns if c != partition_col]
    output_dir.mkdir(parents = True, exist_ok = True)

    groups = gdf.groupby(partition_col, sort = False, observed = True)
    n_partitions = len(groups)
    print(f"Writing {n_partitions} partitions to {output_dir} ...")
    for i, (value, group) in enumerate(groups):
        safe_value = urllib.parse.quote(str(value), safe = "")
        partition_dir = output_dir / f"{partition_col}={safe_value}"
        partition_dir.mkdir()
        group.sort_values(sort_col)[cols].to_parquet(
            partition_dir / "part-0.parquet",
            write_covering_bbox = True,
            row_group_size = 100_000,
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_partitions} partitions written...")
