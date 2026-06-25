#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module downloads a current/latest POI snapshot for the US + inhabited
US territories from OpenStreetMap using Geofabrik PBF extracts, osmium-tool
CLI pre-filtering, and pyosmium parsing.

It is broken into the following functions:

- download_pbf: Downloads a PBF file from a URL via streaming HTTP.
- filter_pbf: Runs osmium tags-filter to produce a reduced POI-only PBF.
- parse_pbf_to_geodataframe: Parses the filtered PBF with pyosmium into a
    GeoDataFrame of nodes (Points) and ways (Polygons or Points).
- download_osm_snapshot: End-to-end orchestrator. Downloads and parses each
    Geofabrik extract in the provided list, then concatenates the results.

Data sources (Geofabrik extracts; full set passed via the ``extracts``
argument to ``download_osm_snapshot``):
    - US mainland (all 50 states incl. AK + HI, ~11 GB):
      https://download.geofabrik.de/north-america/us-latest.osm.pbf
    - Puerto Rico (separate extract, ~tens of MB):
      https://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf
    - US Virgin Islands (separate extract, small):
      https://download.geofabrik.de/north-america/us/us-virgin-islands-latest.osm.pbf
    - American Oceania (covers Guam, Northern Mariana Islands, American
      Samoa, and uninhabited US Pacific possessions; ~5 MB):
      https://download.geofabrik.de/australia-oceania/american-oceania-latest.osm.pbf

Geofabrik extracts are cut along administrative boundaries, so no polygon
post-filter is applied here — the listed extracts together cover the US +
inhabited territories footprint (with a small bonus of uninhabited Pacific
possessions, which contain near-zero POIs).

osmium-tool CLI must be installed (conda install -c conda-forge osmium-tool).

Note: This module is separate from openpois.io.osm_history_pbf, which fetches
full-history PBFs for change-rate modeling. This module downloads a current
snapshot only.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import osmium
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from openpois.io._osm_poi_handler import POIRecordBuilder


class SnapshotExtract(NamedTuple):
    """One Geofabrik extract to be downloaded, filtered, and parsed.

    Attributes:
        name: Short label used for logging and intermediate filenames
            (e.g. ``"us"``, ``"pr"``, ``"usvi"``, ``"american_oceania"``).
            Must be unique within the list passed to ``download_osm_snapshot``.
        url: Geofabrik ``*-latest.osm.pbf`` URL.
        raw_pbf_path: Local path to store the downloaded raw PBF.
        filtered_pbf_path: Local path to store the tags-filtered PBF.
    """
    name: str
    url: str
    raw_pbf_path: Path
    filtered_pbf_path: Path


# -----------------------------------------------------------------------------
# Download helper
# -----------------------------------------------------------------------------


def download_pbf(
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """
    Downloads a PBF file from the given URL to output_path via streaming HTTP.

    Args:
        url: URL of the PBF file to download (e.g., a Geofabrik extract).
        output_path: Local path to save the downloaded PBF.
        overwrite: If False and output_path already exists, skip the download.

    Returns:
        Path to the downloaded PBF file.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"PBF already exists at {output_path}; skipping download.")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PBF from {url} to {output_path}...")
    # Write to a temp file in the same directory, then rename atomically so
    # that a partial download never masquerades as a complete file.
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with requests.get(url, stream=True, timeout=(30, None)) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"  {pct:.1f}%", end="\r")
        tmp_path.rename(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"\nDownload complete: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# osmium-tool filtering
# -----------------------------------------------------------------------------


def filter_pbf(
    input_pbf: Path,
    output_pbf: Path,
    osm_keys: list[str],
    overwrite: bool = False,
    tag_filter_exprs: list[str] | None = None,
) -> Path:
    """
    Runs osmium tags-filter to extract nodes, ways, and relations matching the
    given keys.

    Constructs and runs a command of the form:
        osmium tags-filter -o {output_pbf} {input_pbf} nwr/{key1} nwr/{key2} ...

    The referenced nodes for matched ways are retained so that way geometries
    can be resolved by pyosmium in a subsequent step.

    Args:
        input_pbf: Path to the full PBF extract.
        output_pbf: Path to write the filtered output PBF.
        osm_keys: OSM tag keys to retain (e.g., ['amenity', 'shop']). Used to
            build key-level ``nwr/<key>`` expressions when ``tag_filter_exprs``
            is not supplied.
        overwrite: If False and output_pbf exists, skip filtering.
        tag_filter_exprs: Optional pre-built osmium filter expressions (e.g.
            ``["nwr/amenity", "nwr/landuse=cemetery,religious"]``). When given,
            these are used verbatim instead of ``osm_keys`` — letting callers
            value-scope individual keys to the taxonomy crosswalk values via
            ``taxonomy.build_osm_tag_filter_expressions``.

    Returns:
        Path to the filtered PBF file.

    Raises:
        subprocess.CalledProcessError: If osmium exits with non-zero status.
        FileNotFoundError: If osmium is not installed or not on PATH.
    """
    output_pbf = Path(output_pbf)
    if output_pbf.exists() and not overwrite:
        print(f"Filtered PBF already exists at {output_pbf}; skipping filter.")
        return output_pbf

    output_pbf.parent.mkdir(parents=True, exist_ok=True)
    # Look for osmium on PATH first, then in the same bin dir as Python
    _env_bin = Path(sys.executable).parent / "osmium"
    osmium_bin = (
        shutil.which("osmium") or (str(_env_bin) if _env_bin.exists() else "osmium")
    )
    key_args = (
        list(tag_filter_exprs)
        if tag_filter_exprs is not None
        else [f"nwr/{key}" for key in osm_keys]
    )
    cmd = [
        osmium_bin, "tags-filter", "--overwrite", "-o", str(output_pbf), str(input_pbf)
    ] + key_args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Filtered PBF written to {output_pbf}")
    return output_pbf


# -----------------------------------------------------------------------------
# pyosmium parsing
# -----------------------------------------------------------------------------


def _flush_chunk(
    records: list[dict],
    chunk_dir: Path,
    chunk_idx: int,
) -> Path:
    """Write a list of record dicts to a temporary GeoParquet chunk file."""
    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry = "geometry", crs = "EPSG:4326")
    chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
    gdf.to_parquet(chunk_path)
    return chunk_path


def _align_table_to_schema(
    table: pa.Table, unified_schema: pa.Schema,
) -> pa.Table:
    """Return a copy of `table` conforming to `unified_schema`, filling missing
    columns with nulls and casting type mismatches."""
    cols = {}
    for field in unified_schema:
        idx = table.schema.get_field_index(field.name)
        if idx == -1:
            cols[field.name] = pa.nulls(len(table), type = field.type)
        else:
            col = table.column(field.name)
            cols[field.name] = (
                col.cast(field.type) if col.type != field.type else col
            )
    return pa.table(cols, schema = unified_schema)


def _merge_parquets_streaming(
    input_paths: list[Path], output_path: Path,
) -> None:
    """Concatenate multiple parquets into `output_path` via a PyArrow
    streaming writer, unifying schemas across inputs. Peak memory is bounded
    to one row group at a time. Preserves the metadata (including GeoParquet
    "geo" key) from the first input."""
    schemas = [pq.read_schema(p) for p in input_paths]
    unified_schema = pa.unify_schemas(schemas).with_metadata(schemas[0].metadata)
    with pq.ParquetWriter(output_path, unified_schema) as writer:
        for p in input_paths:
            pf = pq.ParquetFile(p)
            for i in range(pf.num_row_groups):
                table = pf.read_row_group(i)
                writer.write_table(_align_table_to_schema(table, unified_schema))


def _empty_poi_geoparquet(
    out_path: Path, extract_keys: list[str] | None,
) -> None:
    """Write an empty POI GeoParquet with the standard column schema."""
    extra_cols = list(extract_keys) if extract_keys is not None else []
    empty = gpd.GeoDataFrame(
        columns = [
            "source", "osm_id", "osm_type", "name", "geometry",
        ] + extra_cols,
        geometry = "geometry",
        crs = "EPSG:4326",
    )
    out_path.parent.mkdir(parents = True, exist_ok = True)
    empty.to_parquet(out_path)


def parse_pbf_to_parquet(
    pbf_path: Path,
    out_path: Path,
    filter_keys: list[str] | None = None,
    extract_keys: list[str] | None = None,
    source_label: str = "osm",
    chunk_size: int = 500_000,
    max_area_nodes: int | None = None,
    chunk_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    Parses a filtered PBF file with pyosmium and writes the result as a
    single GeoParquet file at `out_path`.

    Memory-efficient alternative to parse_pbf_to_geodataframe: records are
    flushed to per-chunk parquet files on disk, then merged directly to
    out_path via a PyArrow streaming writer. A full GeoDataFrame is never
    materialised in memory. Peak memory is one chunk's worth of records.

    Args: see parse_pbf_to_geodataframe. out_path is written with the same
        schema that parse_pbf_to_geodataframe would produce in a GeoParquet
        round-trip (columns: source, osm_id, osm_type, name, geometry, plus
        any extract_keys tag columns).

    Returns:
        out_path.
    """
    out_path = Path(out_path)
    builder = POIRecordBuilder(
        source_label = source_label,
        filter_keys = filter_keys,
        extract_keys = extract_keys,
        max_area_nodes = max_area_nodes,
    )
    if verbose:
        print(
            f"Parsing {pbf_path} with pyosmium"
            f" (chunk_size={chunk_size:,})..."
        )

    base_dir = Path(chunk_dir) if chunk_dir is not None else Path(pbf_path).parent
    work_dir = base_dir / "parse_chunks"

    existing_chunks = (
        sorted(work_dir.glob("chunk_*.parquet")) if work_dir.exists() else []
    )
    if existing_chunks:
        if verbose:
            print(
                f"Found {len(existing_chunks)} existing chunk(s) in {work_dir};"
                " skipping parse, going straight to merge."
            )
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        chunk_idx = 0
        total_records = 0

        loc_cache = str(work_dir / "locations.dat")
        fp = osmium.FileProcessor(str(pbf_path)) \
            .with_locations(f"sparse_file_array,{loc_cache}") \
            .with_areas()

        for obj in fp:
            rec = None
            if obj.is_node():
                rec = builder.process_node(obj)
            elif obj.is_way():
                rec = builder.process_way(obj)
            elif obj.is_area():
                rec = builder.process_area(obj)

            if rec is not None:
                records.append(rec)
                if len(records) >= chunk_size:
                    _flush_chunk(records, work_dir, chunk_idx)
                    total_records += len(records)
                    if verbose:
                        print(
                            f"  Finished chunk {chunk_idx}"
                            f" ({total_records:,} records so far)"
                        )
                    records.clear()
                    chunk_idx += 1

        # Flush remaining records
        if records:
            _flush_chunk(records, work_dir, chunk_idx)
            total_records += len(records)

        if total_records == 0:
            shutil.rmtree(work_dir)
            _empty_poi_geoparquet(out_path, extract_keys)
            return out_path

        existing_chunks = sorted(work_dir.glob("chunk_*.parquet"))

    # Merge chunk files via PyArrow, writing directly to out_path. Different
    # chunks may have different tag columns (OSM tags are free-form), so we
    # unify schemas first then align each chunk's row groups.
    out_path.parent.mkdir(parents = True, exist_ok = True)
    _merge_parquets_streaming(existing_chunks, out_path)

    # Clean up only after a successful merge
    shutil.rmtree(work_dir)

    if verbose:
        n = pq.read_metadata(out_path).num_rows
        print(f"Parsed {n:,} OSM POIs -> {out_path}")
    return out_path


def parse_pbf_to_geodataframe(
    pbf_path: Path,
    filter_keys: list[str] | None = None,
    extract_keys: list[str] | None = None,
    source_label: str = "osm",
    chunk_size: int = 500_000,
    max_area_nodes: int | None = None,
    chunk_dir: Path | None = None,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Parses a filtered PBF file with pyosmium and returns a GeoDataFrame.

    Thin wrapper around parse_pbf_to_parquet that loads the written parquet
    into a GeoDataFrame. For very large extracts (e.g. a full US PBF), prefer
    parse_pbf_to_parquet and consume the parquet with PyArrow streaming to
    avoid holding all records in memory.

    See parse_pbf_to_parquet for parameter documentation.

    Returns:
        GeoDataFrame with columns:
            source, osm_id (int64), osm_type ('node'|'way'|'relation'),
            tag columns, name, geometry. CRS is EPSG:4326.
    """
    with tempfile.TemporaryDirectory(
        dir = Path(pbf_path).parent
    ) as tmp:
        tmp_parquet = Path(tmp) / "parsed.parquet"
        parse_pbf_to_parquet(
            pbf_path = pbf_path,
            out_path = tmp_parquet,
            filter_keys = filter_keys,
            extract_keys = extract_keys,
            source_label = source_label,
            chunk_size = chunk_size,
            max_area_nodes = max_area_nodes,
            chunk_dir = chunk_dir,
            verbose = verbose,
        )
        gdf = gpd.read_parquet(tmp_parquet)
    if verbose:
        print(
            f"Loaded {len(gdf):,} OSM POIs"
            f" ({gdf['osm_type'].value_counts().to_dict()})"
        )
    return gdf


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def _download_filter_parse_to_parquet(
    pbf_url: str,
    raw_pbf_path: Path,
    filtered_pbf_path: Path,
    parsed_parquet_path: Path,
    filter_keys: list[str],
    extract_keys: list[str] | None,
    overwrite_download: bool,
    overwrite_filter: bool,
    source_label: str,
    chunk_size: int,
    max_area_nodes: int | None,
    chunk_dir: Path | None,
    verbose: bool,
) -> Path:
    """Download a single Geofabrik PBF, filter it, and parse it to a parquet
    file at `parsed_parquet_path` without materialising a GeoDataFrame."""
    download_pbf(
        url = pbf_url,
        output_path = raw_pbf_path,
        overwrite = overwrite_download,
    )
    filter_pbf(
        input_pbf = raw_pbf_path,
        output_pbf = filtered_pbf_path,
        osm_keys = filter_keys,
        overwrite = overwrite_filter,
    )
    return parse_pbf_to_parquet(
        pbf_path = filtered_pbf_path,
        out_path = parsed_parquet_path,
        filter_keys = filter_keys,
        extract_keys = extract_keys,
        source_label = source_label,
        chunk_size = chunk_size,
        max_area_nodes = max_area_nodes,
        chunk_dir = chunk_dir,
        verbose = verbose,
    )


def download_osm_snapshot(
    extracts: list[SnapshotExtract],
    output_path: Path,
    filter_keys: list[str],
    extract_keys: list[str],
    overwrite_download: bool = False,
    overwrite_filter: bool = False,
    source_label: str = "osm",
    keep_all_keys: bool = False,
    chunk_size: int = 500_000,
    max_area_nodes: int | None = None,
    chunk_dir: Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    End-to-end orchestrator: download each Geofabrik extract in ``extracts``,
    filter each to POIs, parse each, concat, and save as GeoParquet.

    For each extract the steps are:

    1. download_pbf — streams the PBF to the raw_pbf path.
    2. filter_pbf — runs osmium tags-filter to produce a POI-only PBF.
    3. parse_pbf_to_parquet — parses with pyosmium into an intermediate
       parquet file.

    The intermediate parquets are concatenated and written to output_path.

    Steps 1 and 2 are skipped if the target files already exist unless
    overwrite_download / overwrite_filter are True.

    Args:
        extracts: List of ``SnapshotExtract`` specs (one per Geofabrik PBF).
            Each spec carries the URL plus the raw and filtered local paths.
            Names must be unique within the list (they are used as suffixes
            on the intermediate parquet filenames).
        output_path: Path to write the output GeoParquet file.
        filter_keys: OSM tag keys used to filter elements in the PBF. Elements
            lacking all of these keys are excluded.
        extract_keys: OSM tag keys to include as output columns. If None, all
            tags on accepted elements are extracted.
        overwrite_download: Re-download even if raw paths exist.
        overwrite_filter: Re-filter even if filtered paths exist.
        source_label: Value for the output 'source' column.
        keep_all_keys: If True, all OSM tags are retained as columns in the
            output GeoDataFrame, not just those in extract_keys. filter_keys
            is still used to filter which elements are included.
        chunk_size: Number of POI records per parquet chunk during parsing.
            Lower values reduce peak memory usage.
        max_area_nodes: If set, relation-derived areas with more than this
            many total coordinate nodes are skipped before any Shapely
            geometry is built. Useful for excluding large multipolygons
            (parks, admin boundaries) that can exhaust memory. None disables
            the check.
        chunk_dir: Directory under which a ``parse_chunks/`` subdirectory is
            created to hold intermediate chunk files. Defaults to the parent
            of each filtered PBF. See parse_pbf_to_geodataframe for details.
        verbose: If True, log progress after each chunk is flushed.

    Returns:
        Path to the written GeoParquet file (same as output_path).
    """
    if not extracts:
        raise ValueError("`extracts` must contain at least one SnapshotExtract.")
    names = [spec.name for spec in extracts]
    if len(set(names)) != len(names):
        raise ValueError(
            f"`extracts` names must be unique; got duplicates in {names}."
        )

    output_path = Path(output_path)
    resolved_extract_keys = None if keep_all_keys else extract_keys

    # One intermediate per-extract parquet, deleted after the final concat.
    # Written next to output_path so they live on the same filesystem (cheap
    # rename) and away from the PBF `parse_chunks/` work dirs.
    parsed_paths: list[Path] = []
    for spec in extracts:
        parsed_path = output_path.with_name(
            output_path.stem + f"._{spec.name}.parquet"
        )
        print(f"Processing {spec.name} extract...")
        _download_filter_parse_to_parquet(
            pbf_url = spec.url,
            raw_pbf_path = spec.raw_pbf_path,
            filtered_pbf_path = spec.filtered_pbf_path,
            parsed_parquet_path = parsed_path,
            filter_keys = filter_keys,
            extract_keys = resolved_extract_keys,
            overwrite_download = overwrite_download,
            overwrite_filter = overwrite_filter,
            source_label = source_label,
            chunk_size = chunk_size,
            max_area_nodes = max_area_nodes,
            chunk_dir = chunk_dir,
            verbose = verbose,
        )
        parsed_paths.append(parsed_path)

    per_extract_counts = [
        (spec.name, pq.read_metadata(p).num_rows)
        for spec, p in zip(extracts, parsed_paths)
    ]
    summary = " + ".join(f"{n} ({c:,})" for n, c in per_extract_counts)
    print(f"Concatenating {summary} POIs...")
    output_path.parent.mkdir(parents = True, exist_ok = True)
    _merge_parquets_streaming(parsed_paths, output_path)

    for p in parsed_paths:
        p.unlink()

    n_total = pq.read_metadata(output_path).num_rows
    print(f"Saved OSM snapshot ({n_total:,} POIs) to {output_path}")
    return output_path
