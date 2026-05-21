#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module downloads a current/latest Overture Maps Places snapshot for the
US + Puerto Rico, filtered to a set of taxonomy categories.

Download strategy: a wide single-query scan of the full US+PR footprint
crashed DuckDB on memory-constrained hosts (it materialized 6M+ rows before
the spatial filter). This module instead iterates the 16 ``part-*.parquet``
files that make up a release, queries each one with a bounded DuckDB session,
and writes a plain parquet intermediate per part. Intermediates survive across
invocations, so a crashed run can be resumed by re-running the script.

After every part is present on local disk, a single DuckDB ``COPY`` applies
the exact US+PR polygon filter (reading the boundary via the spatial
extension), builds the ``geometry`` column with ``ST_Point``, and writes the
final GeoParquet without ever materializing rows in Python. The output file
is valid GeoParquet (readable by ``gpd.read_parquet`` with CRS preserved).

Spatial filter strategy (two-stage, all inside DuckDB):

1. Per-part ``WHERE`` uses predicate pushdown on Overture's ``bbox`` struct
   column, OR-ing across one or more coarse bboxes. Multiple bboxes are
   required to capture the Alaskan Near Islands (+172 E) without scanning
   longitudes the main US bbox would miss.
2. The final ``COPY`` does an exact ``ST_Within`` check against the dissolved
   US+PR polygon to drop Canadian and Mexican border slivers.

Data source: s3://overturemaps-us-west-2/release/ (public, no auth required).

Category filtering uses the ``taxonomy.hierarchy`` array. The first element
(``taxonomy.hierarchy[1]`` in SQL 1-based indexing) is the L0 category. The
deprecated ``categories.primary`` field must NOT be used; it is removed in
June 2026.

Memory knobs: ``duckdb_memory_limit`` and ``duckdb_threads`` are per
DuckDB connection. ``workers`` parallelizes per-part downloads via a
``ThreadPoolExecutor``. Peak host RAM ≈ ``workers × duckdb_memory_limit``
and peak CPU ≈ ``workers × duckdb_threads`` — scale the per-worker knobs
down if raising ``workers`` beyond the default.
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import geopandas as gpd
import requests


# -----------------------------------------------------------------------------
# Release discovery
# -----------------------------------------------------------------------------


def get_latest_release_date(
    bucket: str,
) -> str:
    """
    Finds the most recent Overture Maps release date by listing the S3 bucket.

    Queries the public S3 HTTP API for prefix listings under the 'release/'
    key and returns the lexicographically largest date string found.

    Args:
        bucket: The S3 bucket name hosting Overture releases.

    Returns:
        Release date string in the format 'YYYY-MM-DD.N' as it appears in S3
        (e.g., '2026-02-18.0').

    Raises:
        requests.HTTPError: If the S3 list request fails.
        ValueError: If no release prefixes are found in the bucket.
    """
    s3_list_url = (
        f"https://{bucket}.s3.amazonaws.com"
        "/?list-type=2&prefix=release%2F&delimiter=%2F"
    )
    resp = requests.get(s3_list_url, timeout = 30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    prefixes = [
        el.text.rstrip("/").removeprefix("release/")
        for el in root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)
    ]

    if not prefixes:
        raise ValueError(
            f"No release prefixes found in s3://{bucket}/release/. "
            "Check that the bucket is accessible."
        )

    return sorted(prefixes)[-1]


def build_overture_s3_path(
    release_date: str,
    bucket: str,
) -> str:
    """
    Returns the S3 glob path for all Places Parquet files in a given release.

    Args:
        release_date: Release identifier as returned by get_latest_release_date
            (e.g., '2026-02-18.0').
        bucket: The S3 bucket name.

    Returns:
        S3 path string suitable for DuckDB ``read_parquet()``, e.g.
        ``s3://overturemaps-us-west-2/release/2026-02-18.0/theme=places/type=place/``
    """
    return (
        f"s3://{bucket}/release/{release_date}"
        "/theme=places/type=place/*.parquet"
    )


def _list_overture_part_keys(
    release_date: str,
    bucket: str,
) -> list[str]:
    """
    Lists the ``part-*.parquet`` keys for a given Overture places release.

    Uses the public S3 ``list-type=2`` HTTP API (no AWS SDK required). Asserts
    that the listing is not truncated — the ``theme=places/type=place/`` prefix
    has held well under the 1000-object page size historically, and crossing
    that threshold would silently cause missing parts.

    Args:
        release_date: Release identifier (e.g., '2026-02-18.0').
        bucket: S3 bucket name hosting Overture releases.

    Returns:
        Sorted list of S3 object keys ending in ``.parquet``, one per part.

    Raises:
        requests.HTTPError: If the S3 list request fails.
        ValueError: If the listing is empty, or if it's truncated (indicating
            Overture added enough parts to require pagination).
    """
    prefix = f"release/{release_date}/theme=places/type=place/"
    url = (
        f"https://{bucket}.s3.amazonaws.com"
        f"/?list-type=2&prefix={prefix.replace('/', '%2F')}"
    )
    resp = requests.get(url, timeout = 30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    truncated_el = root.find("s3:IsTruncated", ns)
    if truncated_el is not None and truncated_el.text == "true":
        raise ValueError(
            f"S3 listing for {prefix} is truncated; pagination not implemented. "
            "Overture has grown past the 1000-object page size."
        )
    keys = [
        el.text
        for el in root.findall(".//s3:Contents/s3:Key", ns)
        if el.text is not None and el.text.endswith(".parquet")
    ]
    if not keys:
        raise ValueError(
            f"No part Parquet files found under s3://{bucket}/{prefix}. "
            "Check the release date."
        )
    return sorted(keys)


# -----------------------------------------------------------------------------
# Query construction
# -----------------------------------------------------------------------------


def _build_bbox_predicate(coarse_bboxes: list[dict]) -> str:
    """Return a SQL fragment ORing Overture bbox-struct predicates together."""
    if not coarse_bboxes:
        raise ValueError("coarse_bboxes must contain at least one bbox.")
    terms = [
        (
            f"(bbox.xmin >= {b['xmin']} AND bbox.xmax <= {b['xmax']}"
            f" AND bbox.ymin >= {b['ymin']} AND bbox.ymax <= {b['ymax']})"
        )
        for b in coarse_bboxes
    ]
    return "(" + " OR ".join(terms) + ")"


def _build_taxonomy_predicate(
    allowlist: list[tuple[str, str | None]],
) -> str:
    """Return a SQL fragment ORing per-(L0, L1) taxonomy predicates together.

    If the L1 entry is ``None`` the predicate matches any L1 under that L0.
    """
    if not allowlist:
        raise ValueError("taxonomy allowlist must contain at least one entry.")
    terms = []
    for entry in allowlist:
        l0, l1 = entry[0], entry[1]
        if l1 is None:
            terms.append(f"taxonomy.hierarchy[1] = '{l0}'")
        else:
            terms.append(
                f"(taxonomy.hierarchy[1] = '{l0}' "
                f"AND taxonomy.hierarchy[2] = '{l1}')"
            )
    return "(" + " OR ".join(terms) + ")"


def _apply_duckdb_session_config(
    conn: duckdb.DuckDBPyConnection,
    s3_region: str,
    memory_limit: str,
    threads: int,
    temp_directory: Path,
) -> None:
    """
    Apply resource caps and extensions to a fresh DuckDB connection.

    ``enable_external_file_cache=false`` is retained as a workaround for a
    DuckDB 1.5.0 httpfs bug ("Information loss on integer cast") that fires on
    broad scans of the Overture S3 bucket. 1.5.2 may have fixed it, but the
    flag is cheap and defensive.
    """
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute(f"SET s3_region = '{s3_region}';")
    conn.execute(f"SET memory_limit = '{memory_limit}';")
    conn.execute(f"SET threads TO {int(threads)};")
    conn.execute(f"SET temp_directory = '{temp_directory.as_posix()}';")
    conn.execute("SET preserve_insertion_order = false;")
    conn.execute("SET enable_external_file_cache = false;")


# -----------------------------------------------------------------------------
# Per-part download
# -----------------------------------------------------------------------------


def _download_one_part(
    part_s3_uri: str,
    intermediate_path: Path,
    release_date: str,
    bbox_predicate: str,
    taxonomy_predicate: str,
    source_label: str,
    s3_region: str,
    memory_limit: str,
    threads: int,
    temp_directory: Path,
) -> None:
    """
    Stream a single Overture part file through DuckDB into a local parquet.

    Writes to a ``.tmp`` sibling and atomically renames on success so a
    partial file can never be mistaken for a completed intermediate on resume.
    The connection is opened fresh and closed afterwards so WSL2 reclaims RAM
    between parts.

    Output schema: source, overture_id, release_date, taxonomy_l0/l1/l2,
    overture_categories_alternate, overture_name, brand_name, confidence,
    overture_addr_street/city/state/postcode/country, overture_websites,
    overture_phones, overture_socials, longitude, latitude. No geometry
    column — geometry is built in the final merge step.
    """
    intermediate_path.parent.mkdir(parents = True, exist_ok = True)
    tmp_path = intermediate_path.with_suffix(intermediate_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    query = f"""
        COPY (
            SELECT
                '{source_label}' AS source,
                id AS overture_id,
                '{release_date}' AS release_date,
                taxonomy.hierarchy[1] AS taxonomy_l0,
                taxonomy.hierarchy[2] AS taxonomy_l1,
                taxonomy.hierarchy[3] AS taxonomy_l2,
                categories.alternate AS overture_categories_alternate,
                names.primary AS overture_name,
                brand.names.primary AS brand_name,
                confidence,
                addresses[1].freeform AS overture_addr_street,
                addresses[1].locality AS overture_addr_city,
                addresses[1].region   AS overture_addr_state,
                addresses[1].postcode AS overture_addr_postcode,
                addresses[1].country  AS overture_addr_country,
                websites AS overture_websites,
                phones   AS overture_phones,
                socials  AS overture_socials,
                ST_X(geometry) AS longitude,
                ST_Y(geometry) AS latitude
            FROM read_parquet('{part_s3_uri}', hive_partitioning = 1)
            WHERE
                {bbox_predicate}
                AND {taxonomy_predicate}
        ) TO '{tmp_path.as_posix()}' (FORMAT parquet, COMPRESSION 'zstd')
    """

    conn = duckdb.connect()
    try:
        _apply_duckdb_session_config(
            conn,
            s3_region = s3_region,
            memory_limit = memory_limit,
            threads = threads,
            temp_directory = temp_directory,
        )
        conn.execute(query)
    finally:
        conn.close()

    os.rename(tmp_path, intermediate_path)


# -----------------------------------------------------------------------------
# Final filter-and-write
# -----------------------------------------------------------------------------


def _finalize_snapshot_in_duckdb(
    parts_glob: str,
    boundary_path: Path,
    output_path: Path,
    s3_region: str,
    memory_limit: str,
    threads: int,
    temp_directory: Path,
) -> None:
    """
    Apply the exact US+PR polygon filter inside DuckDB and write the final
    GeoParquet.

    Reads all per-part intermediates, builds ``Point`` geometries from
    longitude/latitude, joins against the boundary polygon (read natively as
    GEOMETRY from a geopandas-written GeoParquet), and streams the surviving
    rows straight to ``output_path`` via ``COPY TO``. No Python-side
    materialization of rows occurs at any point.

    The output file is readable by ``gpd.read_parquet`` with CRS preserved
    (OGC:CRS84, equivalent to EPSG:4326).
    """
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    query = f"""
        CREATE TABLE boundary AS
            SELECT geometry AS geom
            FROM read_parquet('{boundary_path.as_posix()}');

        COPY (
            SELECT
                p.source,
                p.overture_id,
                p.release_date,
                p.taxonomy_l0,
                p.taxonomy_l1,
                p.taxonomy_l2,
                p.overture_categories_alternate,
                p.overture_name,
                p.brand_name,
                p.confidence,
                p.overture_addr_street,
                p.overture_addr_city,
                p.overture_addr_state,
                p.overture_addr_postcode,
                p.overture_addr_country,
                p.overture_websites,
                p.overture_phones,
                p.overture_socials,
                ST_Point(p.longitude, p.latitude) AS geometry
            FROM read_parquet('{parts_glob}', union_by_name = true) p, boundary b
            WHERE ST_Within(ST_Point(p.longitude, p.latitude), b.geom)
        ) TO '{tmp_output.as_posix()}' (FORMAT parquet, COMPRESSION 'zstd')
    """

    conn = duckdb.connect()
    try:
        _apply_duckdb_session_config(
            conn,
            s3_region = s3_region,
            memory_limit = memory_limit,
            threads = threads,
            temp_directory = temp_directory,
        )
        conn.execute(query)
    finally:
        conn.close()

    os.rename(tmp_output, output_path)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def download_overture_snapshot(
    output_path: Path,
    taxonomy_allowlist: list,
    boundary_gdf: gpd.GeoDataFrame,
    coarse_bboxes: list[dict],
    bucket: str,
    s3_region: str,
    release_date: str | None = None,
    source_label: str = "overture",
    duckdb_memory_limit: str = "4GB",
    duckdb_threads: int = 2,
    workers: int = 2,
) -> Path:
    """
    Downloads filtered Overture Maps Places data and writes it as GeoParquet.

    The full-CONUS scan is split across the release's ``part-*.parquet`` files.
    Each part streams through DuckDB into a plain parquet intermediate
    (under ``output_path.parent / ".parts" / <release_date> /``); the loop is
    resumable — if an intermediate already exists, the part is skipped on the
    next run. After every part is present, a single DuckDB ``COPY`` applies
    the exact US+PR polygon filter and writes the final GeoParquet without
    materializing rows in Python.

    Args:
        output_path: Path to write the output GeoParquet file.
        taxonomy_allowlist: List of (L0, L1) pairs specifying which taxonomy
            branches to retain. ``L1 = None`` means "all L1s under this L0".
            Accepts pairs as two-element tuples or lists (YAML).
            Valid L0 values (from S3 data as of 2026-02-18): 'food_and_drink',
            'shopping', 'arts_and_entertainment', 'sports_and_recreation',
            'health_care', 'services_and_business',
            'travel_and_transportation', 'lifestyle_services', 'education',
            'community_and_government', 'cultural_and_historic', 'lodging',
            'geographic_entities'.
            See: https://docs.overturemaps.org/guides/places/taxonomy/
        boundary_gdf: Single-row GeoDataFrame in EPSG:4326 containing the
            dissolved, buffered US+PR polygon. Used as the exact spatial
            filter; obtain it from ``openpois.io.boundary``.
        coarse_bboxes: List of bbox dicts (keys ``xmin, ymin, xmax, ymax``)
            used as the DuckDB predicate-pushdown prefilter. Typically
            obtained from ``openpois.io.boundary.us_pr_bboxes``.
        bucket: S3 bucket name hosting Overture releases.
        s3_region: AWS region of the S3 bucket.
        release_date: Overture release identifier (e.g., '2026-02-18.0').
            If None, the latest release is fetched automatically.
        source_label: Value for the output 'source' column.
        duckdb_memory_limit: Per-connection DuckDB memory cap (e.g., "4GB").
        duckdb_threads: Per-connection DuckDB thread count.
        workers: Number of parts to download in parallel via a
            ``ThreadPoolExecutor``. Peak host RAM is
            ``workers × duckdb_memory_limit`` and peak CPU is
            ``workers × duckdb_threads`` — scale down the per-worker knobs
            when increasing ``workers``. Must be >= 1.

    Returns:
        The ``output_path`` of the written GeoParquet file. The file is
        readable by ``gpd.read_parquet(path)`` (with ``columns=...`` support)
        with CRS preserved as OGC:CRS84 (equivalent to EPSG:4326).

    Raises:
        ValueError: If ``workers`` is less than 1, or if the S3 listing is
            truncated or empty.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    if release_date is None:
        print("Detecting latest Overture release...")
        release_date = get_latest_release_date(bucket = bucket)
        print(f"Using release: {release_date}")

    bbox_predicate = _build_bbox_predicate(coarse_bboxes)
    taxonomy_predicate = _build_taxonomy_predicate(taxonomy_allowlist)

    parts_dir = output_path.parent / ".parts" / release_date
    parts_dir.mkdir(parents = True, exist_ok = True)
    temp_directory = parts_dir / "duckdb_tmp"
    temp_directory.mkdir(parents = True, exist_ok = True)

    part_keys = _list_overture_part_keys(
        release_date = release_date, bucket = bucket
    )
    total = len(part_keys)
    print(
        f"Downloading {total} part files for release {release_date} "
        f"with {workers} worker(s)..."
    )

    pending: list[tuple[str, Path]] = []
    for key in part_keys:
        intermediate = parts_dir / (Path(key).stem + ".parquet")
        if intermediate.exists() and intermediate.stat().st_size > 0:
            print(f"  {intermediate.name} already present; skipping.")
            continue
        pending.append((key, intermediate))

    def _run_one(key: str, intermediate: Path) -> str:
        part_s3_uri = f"s3://{bucket}/{key}"
        _download_one_part(
            part_s3_uri = part_s3_uri,
            intermediate_path = intermediate,
            release_date = release_date,
            bbox_predicate = bbox_predicate,
            taxonomy_predicate = taxonomy_predicate,
            source_label = source_label,
            s3_region = s3_region,
            memory_limit = duckdb_memory_limit,
            threads = duckdb_threads,
            temp_directory = temp_directory,
        )
        return intermediate.name

    if pending:
        with ThreadPoolExecutor(max_workers = workers) as pool:
            futures = {
                pool.submit(_run_one, key, path): path
                for key, path in pending
            }
            done = 0
            for future in as_completed(futures):
                # Surface the first exception; the context manager will wait
                # for already-running futures to finish so we don't leave half-
                # written tmp files owned by DuckDB.
                name = future.result()
                done += 1
                print(f"  [{done}/{len(pending)}] finished {name}")

    # Write the boundary polygon to a temporary GeoParquet so DuckDB's spatial
    # extension can read it natively as GEOMETRY.
    boundary_tmp = parts_dir / "_boundary.parquet"
    boundary_gdf[["geometry"]].to_parquet(boundary_tmp)

    parts_glob = (parts_dir / "part-*.parquet").as_posix()
    print("Applying US+PR polygon filter and writing final GeoParquet...")
    _finalize_snapshot_in_duckdb(
        parts_glob = parts_glob,
        boundary_path = boundary_tmp,
        output_path = output_path,
        s3_region = s3_region,
        memory_limit = duckdb_memory_limit,
        threads = max(int(duckdb_threads), 4),
        temp_directory = temp_directory,
    )

    # Cleanup on success only. Leaving intermediates on failure is intentional
    # so the next run can resume.
    shutil.rmtree(parts_dir, ignore_errors = True)

    print(f"Saved Overture snapshot to {output_path}")
    return output_path
