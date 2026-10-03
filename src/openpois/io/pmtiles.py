#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Build a single-layer PMTiles archive from a GeoParquet POI snapshot.

Pipeline:
1. Stream the GeoParquet input through ``pyarrow.parquet.iter_batches`` so the
   full dataset never sits in memory (relevant for the ~20M-row conflated set).
2. For each batch, decode the WKB geometry column into shapely Points (via
   ``representative_point()``, which is null-safe and returns a guaranteed-inside
   point for Polygons/MultiPolygons).
3. Append the batch to a staged FlatGeobuf file via pyogrio. FlatGeobuf is the
   preferred tippecanoe input format — tippecanoe auto-parallelises on it
   without needing the ``-P`` flag.
4. Invoke ``tippecanoe`` to tile the FlatGeobuf into the target PMTiles archive.
5. Delete the intermediate FlatGeobuf on success (leave it on failure for debug).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pyogrio


def build_pmtiles(
    input_parquet: Path,
    output_pmtiles: Path,
    layer_name: str,
    properties: list[str],
    min_zoom: int,
    max_zoom: int,
    drop_strategy: str = "drop-densest-as-needed",
    batch_size: int = 1_000_000,
    tippecanoe_bin: str | None = None,
    batch_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    extend_zooms_if_still_dropping: bool = False,
) -> dict:
    """Build a PMTiles archive from a GeoParquet POI file.

    Arguments:
        input_parquet: path to GeoParquet file. Geometry column must be named
            ``geometry`` and stored as WKB.
        output_pmtiles: destination ``.pmtiles`` path. Will be overwritten.
        layer_name: vector-tile layer name (referenced from the frontend style).
        properties: list of column names to include as feature properties.
            Everything not in this list (and not ``geometry``) is dropped,
            which is critical for tile size.
        min_zoom, max_zoom: tippecanoe ``-Z`` / ``-z``. Pass equal values for a
            single-zoom archive (relies on client-side over-zoom above max_zoom).
        drop_strategy: one of ``drop-densest-as-needed`` (smallest files, silent
            feature drops) or ``cluster-densest-as-needed`` (adds cluster-count
            properties).
        batch_size: rows per Arrow batch. Smaller = lower peak RAM, more I/O.
        tippecanoe_bin: path to ``tippecanoe`` executable. If None, uses the
            one on PATH.
        batch_transform: optional function applied to each batch's attribute
            columns (everything but ``geometry``) before it is written. Use it
            to rename columns or flatten LIST columns, which FlatGeobuf cannot
            store. Its output columns become the tile properties.
        extend_zooms_if_still_dropping: pass tippecanoe's flag of the same
            name, which adds zoom levels past ``max_zoom`` until the top zoom
            drops no features. Without it, features dropped at ``max_zoom``
            are missing at every zoom, since deeper views over-zoom that tile.

    Returns a dict with: ``rows_written``, ``fgb_bytes``, ``pmtiles_bytes``.
    """
    input_parquet = Path(input_parquet).expanduser()
    output_pmtiles = Path(output_pmtiles).expanduser()
    output_pmtiles.parent.mkdir(parents = True, exist_ok = True)

    if not input_parquet.exists():
        raise FileNotFoundError(input_parquet)

    fgb_path = output_pmtiles.with_suffix(".fgb")
    # Reuse a pre-staged FlatGeobuf if it's newer than the input parquet — lets
    # us recover cheaply after a tippecanoe failure without re-streaming all
    # rows through pyogrio.
    if fgb_path.exists() and fgb_path.stat().st_mtime >= input_parquet.stat().st_mtime:
        rows = 0  # unknown; the FGB feature count is in its header
        print(f"  reusing existing FlatGeobuf {fgb_path} (newer than input)")
    else:
        if fgb_path.exists():
            fgb_path.unlink()
        rows = _stage_flatgeobuf(
            input_parquet = input_parquet,
            output_fgb = fgb_path,
            properties = properties,
            batch_size = batch_size,
            batch_transform = batch_transform,
        )

    fgb_bytes = fgb_path.stat().st_size
    print(
        f"  staged FlatGeobuf: {rows or '?'} rows, "
        f"{fgb_bytes / 1e9:.2f} GB at {fgb_path}"
    )

    _run_tippecanoe(
        fgb_path = fgb_path,
        output_pmtiles = output_pmtiles,
        layer_name = layer_name,
        min_zoom = min_zoom,
        max_zoom = max_zoom,
        drop_strategy = drop_strategy,
        tippecanoe_bin = tippecanoe_bin,
        extend_zooms_if_still_dropping = extend_zooms_if_still_dropping,
    )

    pmtiles_bytes = output_pmtiles.stat().st_size
    print(
        f"  built PMTiles: {pmtiles_bytes / 1e9:.2f} GB at {output_pmtiles}"
    )

    fgb_path.unlink()

    return {
        "rows_written": rows,
        "fgb_bytes": fgb_bytes,
        "pmtiles_bytes": pmtiles_bytes,
    }


def _stage_flatgeobuf(
    input_parquet: Path,
    output_fgb: Path,
    properties: list[str],
    batch_size: int,
    batch_transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> int:
    """Stream GeoParquet -> FlatGeobuf, projecting geometry to representative point.

    FlatGeobuf requires a homogeneous geometry type per file; representative_point
    collapses Polygons/MultiPolygons to Points up front, so the written file is
    uniformly Point-typed.
    """
    pq_file = pq.ParquetFile(input_parquet)
    total_rows = pq_file.metadata.num_rows
    read_cols = list(properties) + ["geometry"]
    n_written = 0
    first_batch = True

    print(
        f"  streaming {total_rows:,} rows from {input_parquet.name} "
        f"in batches of {batch_size:,}"
    )

    for batch in pq_file.iter_batches(batch_size = batch_size, columns = read_cols):
        df = batch.to_pandas()
        if batch_transform is not None:
            geometry = df.pop("geometry")
            df = batch_transform(df)
            df["geometry"] = geometry
        df["geometry"] = gpd.GeoSeries.from_wkb(df["geometry"])
        gdf = gpd.GeoDataFrame(df, geometry = "geometry", crs = "EPSG:4326")
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        gdf["geometry"] = gdf.geometry.representative_point()

        pyogrio.write_dataframe(
            gdf,
            output_fgb,
            driver = "FlatGeobuf",
            append = not first_batch,
        )
        first_batch = False
        n_written += len(gdf)
        print(f"    wrote {n_written:,}/{total_rows:,}")

    return n_written


def _run_tippecanoe(
    fgb_path: Path,
    output_pmtiles: Path,
    layer_name: str,
    min_zoom: int,
    max_zoom: int,
    drop_strategy: str,
    tippecanoe_bin: str | None,
    extend_zooms_if_still_dropping: bool = False,
) -> None:
    """Invoke tippecanoe. Stderr is streamed to our stderr so the caller sees
    the tile-count / drop summary in real time.
    """
    resolved_bin = tippecanoe_bin or shutil.which("tippecanoe")
    if not resolved_bin:
        # Conda env bins aren't always on the shell PATH; fall back to the
        # directory of the current python interpreter (same pattern we use for
        # osmium in io/osm_snapshot.py).
        env_bin = Path(sys.executable).parent / "tippecanoe"
        if env_bin.exists():
            resolved_bin = str(env_bin)
    if not resolved_bin:
        raise FileNotFoundError(
            "tippecanoe executable not found on PATH or in the conda env bin. "
            "Install via `mamba install -c conda-forge tippecanoe -n openpois`."
        )

    cmd = [
        resolved_bin,
        "-o", str(output_pmtiles),
        "-Z", str(min_zoom),
        "-z", str(max_zoom),
        f"--{drop_strategy}",
        "-l", layer_name,
        "--force",
        "--no-progress-indicator",   # -\r progress spam bloats captured logs
    ]
    if extend_zooms_if_still_dropping:
        cmd.append("--extend-zooms-if-still-dropping")
    cmd.append(str(fgb_path))
    print("  running: " + " ".join(cmd))
    subprocess.run(cmd, check = True)
