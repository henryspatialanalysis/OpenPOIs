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
import gc
import shutil
import urllib.parse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pygeohash
import shapely
from geopandas.io.arrow import _geopandas_to_arrow


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
    chunk_rows: int = 1_000_000,
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

    Output files include a GeoParquet 1.1 ``covering.bbox`` struct column
    (``write_covering_bbox=True``) and use ``row_group_size=100_000`` so
    spatial bbox predicates can prune row groups. Gotchas for consumers:

    * Predicate pushdown only fires when filters reference the ``bbox``
      struct fields directly — e.g.,
      ``WHERE bbox.xmin <= ? AND bbox.xmax >= ? AND bbox.ymin <= ? AND
      bbox.ymax >= ?``. ``ST_Intersects(geometry, …)`` alone will not
      trigger bbox pruning in DuckDB; pass both predicates or rely on
      GeoPandas's ``read_parquet(..., bbox=...)`` which adds the struct
      filter automatically.
    * Minimum reader versions: DuckDB ≥ 0.10, PyArrow ≥ 15, GeoPandas
      ≥ 1.0, GDAL ≥ 3.8. Older readers see the ``bbox`` column as a
      plain struct and silently skip the pruning optimization.
    * The ``bbox`` column appears in ``DESCRIBE`` output and in
      ``SELECT *`` results. Downstream pipelines that materialize all
      columns will pick it up; drop it explicitly if it conflicts with a
      schema contract.
    * Writing requires ``geopandas >= 1.0``; calling this function from
      an older environment will fail with an unexpected-kwarg error on
      ``write_covering_bbox``.
    * ``row_group_size=100_000`` is fine for small partitions (they end
      up as a single row group) but it caps pruning granularity for very
      large partitions. Tune downward if a single file grows past ~5M
      rows and viewport queries still feel slow.

    Memory: large partitions are written via ``pyarrow.parquet.ParquetWriter``
    in row-group chunks of ``chunk_rows`` rows so the per-partition Arrow
    Table never coexists at full size with the parent GeoDataFrame. The
    sort step uses ``np.argsort`` + ``iloc`` rather than pandas
    ``sort_values``, dropping one full-partition copy from the peak.
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
        _write_one_partition(
            group, cols, output_dir, partition_col, value, sort_col,
            chunk_rows,
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_partitions} partitions written...")


def _write_one_partition(group, cols, output_dir: Path, partition_col: str,
                         value, sort_col: str, chunk_rows: int) -> int:
    """Write one Hive partition directory, geohash-sorted. Returns row count."""
    safe_value = urllib.parse.quote(str(value), safe = "")
    partition_dir = output_dir / f"{partition_col}={safe_value}"
    partition_dir.mkdir(parents = True, exist_ok = True)
    part_path = partition_dir / "part-0.parquet"

    # np.argsort + iloc avoids the full-partition copy that
    # pandas .sort_values() would make; on the 3.9M-row "Other
    # Amenity" partition that single dropped copy is ~3 GB.
    sort_keys = group[sort_col].to_numpy()
    sort_indices = np.argsort(sort_keys, kind = "mergesort")
    del sort_keys

    n_rows = len(group)
    if n_rows <= chunk_rows:
        # Small partitions: single write_table — same end state
        # as the old code path.
        sorted_slice = group.iloc[sort_indices][cols]
        table = _geopandas_to_arrow(
            sorted_slice, write_covering_bbox = True,
        )
        pq.write_table(
            table, str(part_path),
            row_group_size = 100_000,
        )
        del sorted_slice, table
    else:
        # Large partitions: stream via ParquetWriter in row-group
        # chunks so the Arrow Table never coexists at full size
        # with the parent GeoDataFrame.
        sample_slice = group.iloc[sort_indices[:1]][cols]
        sample_tbl = _geopandas_to_arrow(
            sample_slice, write_covering_bbox = True,
        )
        schema = sample_tbl.schema
        del sample_slice, sample_tbl

        with pq.ParquetWriter(str(part_path), schema) as writer:
            for chunk_start in range(0, n_rows, chunk_rows):
                chunk_end = min(chunk_start + chunk_rows, n_rows)
                chunk_indices = sort_indices[chunk_start:chunk_end]
                chunk_slice = group.iloc[chunk_indices][cols]
                chunk_tbl = _geopandas_to_arrow(
                    chunk_slice, write_covering_bbox = True,
                )
                writer.write_table(
                    chunk_tbl, row_group_size = 100_000,
                )
                del chunk_slice, chunk_tbl
                gc.collect()

    gc.collect()
    return n_rows


def _read_rows_by_position(input_path: Path, keep_mask: np.ndarray,
                           columns: list) -> gpd.GeoDataFrame:
    """Read only the rows flagged in ``keep_mask``, one row group at a time.

    Used when the partition label is *derived* rather than stored, so a dataset
    predicate cannot select the rows. Peak memory is one row group plus the
    accumulated matches for the single partition being built.
    """
    parquet_file = pq.ParquetFile(str(input_path))
    pieces = []
    offset = 0
    for group_index in range(parquet_file.metadata.num_row_groups):
        n_rows = parquet_file.metadata.row_group(group_index).num_rows
        group_mask = keep_mask[offset:offset + n_rows]
        offset += n_rows
        if not group_mask.any():
            continue
        table = parquet_file.read_row_group(group_index, columns = columns)
        pieces.append(table.filter(pa.array(group_mask)))
        del table
        gc.collect()
    if not pieces:
        return gpd.GeoDataFrame()
    combined = pa.concat_tables(pieces)
    del pieces
    frame = gpd.GeoDataFrame.from_arrow(combined)
    del combined
    gc.collect()
    return frame


def write_label_partitioned_from_parquet(
    input_path,
    output_dir,
    partition_col: str,
    geohash_precision: int,
    sort_col: str = "geohash",
    overwrite: bool = True,
    chunk_rows: int = 1_000_000,
    labels = None,
) -> int:
    """Label-partition a parquet file **without ever loading it whole**.

    Same on-disk result as :func:`write_label_partitioned_dataset` — identical
    columns, Hive directory names, geohash sort, GeoParquet 1.1 covering bbox,
    and ``row_group_size`` — but one partition is resident at a time instead of
    the entire dataset.

    This exists because the whole-frame path does not fit in memory at CONUS
    scale. Reading 14.6M rows x 58 columns as a GeoDataFrame peaked at 21.5 GB
    RSS against a 24 GB cap, spilling into swap; per-partition reads hold a few
    hundred MB. Prefer this for anything national.

    The pandas index is set to each row's **original file position**, so the
    stored ``__index_level_0__`` matches what the whole-frame path would have
    written rather than restarting at zero in every partition.

    ``labels`` supplies the partition value per row when it is *derived* rather
    than stored in the file (the OSM ``primary_tag`` case). It must be aligned
    to file order and the same length as the dataset. When given, rows are
    gathered by a row-group scan instead of a dataset predicate.

    Returns the number of rows written.
    """
    # Imported here: pyarrow.dataset pulls in a lot and only this path needs it.
    # pylint: disable-next=import-outside-toplevel
    import pyarrow.dataset as pads

    input_path = Path(input_path)
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
    output_dir.mkdir(parents = True, exist_ok = True)

    dataset = pads.dataset(str(input_path), format = "parquet")
    all_columns = list(dataset.schema.names)
    derived = labels is not None
    if not derived:
        if partition_col not in all_columns:
            raise KeyError(
                f"partition_col not in {input_path}: {partition_col!r}"
            )
        # One narrow pass for the partition values, so each label's original
        # row positions are known before any wide read happens.
        labels = dataset.to_table(columns = [partition_col])[
            partition_col
        ].to_pandas()
    else:
        labels = pd.Series(labels).reset_index(drop = True)
        if len(labels) != dataset.count_rows():
            raise ValueError(
                f"labels length {len(labels):,} does not match "
                f"{input_path} row count {dataset.count_rows():,}"
            )
    null_mask = labels.isna()
    if null_mask.any():
        print(f"Skipping {int(null_mask.sum()):,} rows with null "
              f"{partition_col}")
    values = pd.unique(labels[~null_mask])
    read_columns = [c for c in all_columns if c != partition_col]

    print(f"Writing {len(values)} partitions to {output_dir} ...")
    written = 0
    for i, value in enumerate(values):
        # fillna before to_numpy: on a nullable dtype (the derived primary_tag
        # is pandas "string") the comparison yields pd.NA for null rows, and a
        # mask carrying NA is neither indexable nor safe for flatnonzero.
        match_mask = (labels == value).fillna(False).to_numpy(dtype = bool)
        positions = np.flatnonzero(match_mask)
        if derived:
            group = _read_rows_by_position(input_path, match_mask,
                                           read_columns)
        else:
            # A dataset filter preserves file order, so the k-th returned row
            # is the k-th match — which makes `positions` the right index.
            table = dataset.to_table(
                columns = read_columns,
                filter = pads.field(partition_col) == value,
            )
            group = gpd.GeoDataFrame.from_arrow(table)
            del table
        # Carry the positions as a COLUMN, not the index: add_geohash_column
        # resets the index when it drops empty geometries, which would silently
        # renumber them.
        group["__row_position"] = positions
        group = add_geohash_column(group, precision = geohash_precision)
        group.index = group.pop("__row_position").to_numpy()
        written += _write_one_partition(
            group, list(group.columns), output_dir, partition_col, value,
            sort_col, chunk_rows,
        )
        del group
        gc.collect()
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(values)} partitions written "
                  f"({written:,} rows)...")
    print(f"  {len(values)}/{len(values)} partitions written "
          f"({written:,} rows)")
    return written
