#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module provides a single source of truth for the spatial extent used by
POI snapshot downloads: the 50 US states + DC + the 5 inhabited US
territories (Puerto Rico, US Virgin Islands, Guam, Northern Mariana Islands,
American Samoa).

It is broken into the following functions:

- download_us_pr_boundary: Downloads and unzips the Census cartographic
    boundary state shapefile. Skips the download if the file already exists.
- load_us_pr_boundary: Reads the shapefile and filters to the 56 US
    statistical areas (50 states + DC + 5 territories).
- us_pr_unary_polygon: Dissolves the filtered areas into a single
    (multi)polygon and applies an outward buffer to include near-coastal
    POIs. Internal borders disappear on dissolve, so the buffer only
    expands the coastline (plus tiny strips of the Canada/Mexico land
    border, which contain effectively no POIs).
- us_pr_bboxes: Returns coarse bounding boxes covering the buffered
    polygon. Splits at longitude 0 into a negative-longitude bbox (CONUS,
    AK mainland, HI, PR, USVI, American Samoa) and a positive-longitude
    bbox (Alaskan Near Islands ~+172 E, Guam ~+144 E, Northern Mariana
    Islands ~+145 E) so callers that can't filter on a polygon directly
    (e.g., DuckDB predicate pushdown on Overture's `bbox` struct) can
    still use bounding-box prefilters without missing the western-Pacific
    territories or the Aleutian Near Islands.

The Census 1:5M cartographic boundary file is used because it has clean
``STUSPS`` codes, includes all 5 inhabited territories, and is small
(~1.3 MB zipped). The 1:20M variant is too aggressive a generalisation to
include territories.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import requests

# EPSG:6933 = World Equal-Area Cylindrical. Preserves area globally with
# <1% distortion at latitudes relevant to the US + territories, so buffering by N
# metres in this CRS gives a true N-metre buffer everywhere we care about.
_EQUAL_AREA_CRS = "EPSG:6933"

# STUSPS codes for the 50 states + DC + 5 inhabited US territories:
# PR = Puerto Rico, VI = US Virgin Islands, GU = Guam,
# MP = Northern Mariana Islands, AS = American Samoa.
_US_STATE_CODES = frozenset({
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY", "PR", "VI", "GU", "MP", "AS",
})


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------


def download_us_pr_boundary(
    source_url: str,
    cache_dir: Path,
    zip_name: str = "cb_2023_us_state_5m.zip",
    shp_name: str = "cb_2023_us_state_5m.shp",
    overwrite: bool = False,
) -> Path:
    """
    Downloads the Census cartographic-boundary state shapefile and unzips it.

    The file is cached in ``cache_dir``. If the expected ``.shp`` already
    exists (and ``overwrite`` is False) this function is a no-op and simply
    returns the path.

    Args:
        source_url: URL of the Census cartographic boundary zip file.
        cache_dir: Directory where the zip file and unzipped shapefile
            components are stored.
        zip_name: Filename of the downloaded zip file within ``cache_dir``.
        shp_name: Filename of the target shapefile within ``cache_dir``.
        overwrite: If True, always re-download and re-unzip.

    Returns:
        Path to the unzipped ``.shp`` file in ``cache_dir``.

    Raises:
        requests.HTTPError: If the download fails.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents = True, exist_ok = True)
    zip_path = cache_dir / zip_name
    shp_path = cache_dir / shp_name

    if shp_path.exists() and not overwrite:
        return shp_path

    print(f"Downloading US + territories boundary from {source_url}...")
    resp = requests.get(source_url, stream = True, timeout = 120)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size = 1024 * 1024):
            f.write(chunk)

    print(f"Unzipping {zip_path} to {cache_dir}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)

    if not shp_path.exists():
        raise FileNotFoundError(
            f"Expected shapefile {shp_path} not found after unzipping "
            f"{zip_path}. Contents: {list(cache_dir.iterdir())}"
        )
    return shp_path


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_us_pr_boundary(
    shp_path: Path,
) -> gpd.GeoDataFrame:
    """
    Reads the state shapefile and returns the 50 states + DC + 5 inhabited
    US territories.

    Args:
        shp_path: Path to the unzipped Census state shapefile (``.shp``).

    Returns:
        GeoDataFrame in EPSG:4326 containing 56 rows (50 states + DC + PR
        + VI + GU + MP + AS), with the Census shapefile's columns preserved
        (``STUSPS``, ``NAME``, ``GEOID``, etc.) plus the ``geometry`` column.

    Raises:
        ValueError: If fewer than 56 rows match ``_US_STATE_CODES`` (suggests
            the source file is wrong or corrupted, e.g. the 1:20M variant
            that omits territories).
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.loc[gdf["STUSPS"].isin(_US_STATE_CODES)].copy()
    if len(gdf) != len(_US_STATE_CODES):
        raise ValueError(
            f"Expected {len(_US_STATE_CODES)} rows after filtering to "
            f"US + DC + territories, got {len(gdf)}. Missing codes: "
            f"{sorted(_US_STATE_CODES - set(gdf['STUSPS']))}"
        )
    return gdf.reset_index(drop = True)


def us_pr_unary_polygon(
    shp_path: Path,
    coastline_buffer_m: float = 100.0,
) -> gpd.GeoDataFrame:
    """
    Returns a single-row GeoDataFrame containing the dissolved, buffered
    US + DC + territories polygon in EPSG:4326.

    Dissolving removes all internal borders, so the resulting polygon's
    exterior is either coastline/water or international land border (Canada,
    Mexico). A uniform outward buffer therefore effectively buffers only the
    coastline — the Canada/Mexico land border is a tiny fraction of the total
    exterior and captures essentially no POIs within 100 m.

    The buffer is applied in EPSG:6933 (World Equal-Area Cylindrical) to keep
    the distance metric accurate across the very different latitudes of
    CONUS, Alaska, Hawaii, the Caribbean territories, and the western
    Pacific territories.

    Args:
        shp_path: Path to the unzipped Census state shapefile.
        coastline_buffer_m: Outward buffer distance in metres. Set to 0 to
            disable the buffer.

    Returns:
        Single-row GeoDataFrame in EPSG:4326 with a single ``geometry`` column.
    """
    states = load_us_pr_boundary(shp_path = shp_path)
    dissolved = states.dissolve()[["geometry"]]
    if coastline_buffer_m > 0:
        dissolved = dissolved.to_crs(_EQUAL_AREA_CRS)
        dissolved["geometry"] = dissolved.buffer(coastline_buffer_m)
        dissolved = dissolved.to_crs("EPSG:4326")
    return dissolved.reset_index(drop = True)


# -----------------------------------------------------------------------------
# Bounding boxes (coarse pre-filters)
# -----------------------------------------------------------------------------


def us_pr_bboxes(
    shp_path: Path,
    coastline_buffer_m: float = 100.0,
) -> list[dict]:
    """
    Returns coarse bounding boxes covering the buffered US + territories
    polygon.

    Two bboxes are returned to handle the antimeridian. The negative-longitude
    bbox covers CONUS, AK mainland, HI, PR, USVI, and American Samoa
    (~-170 W). The positive-longitude bbox covers the western Aleutian "Near
    Islands" (~+172 E), Guam (~+144 E), and the Northern Mariana Islands
    (~+145 E). ``gpd.total_bounds`` on a multipolygon that crosses the
    antimeridian reports ``(~-180, ..., ~180)`` which is useless as a coarse
    prefilter.

    This function separates the positive-longitude parts from the rest of
    the polygon using an x-split at longitude 0 (via per-part centroid) and
    computes a bbox for each side. Callers that can only apply bbox
    predicates (e.g., DuckDB predicate pushdown on Overture's ``bbox``
    struct) should OR the returned bboxes together.

    Args:
        shp_path: Path to the unzipped Census state shapefile.
        coastline_buffer_m: Outward buffer distance applied before computing
            the bboxes. Matches the buffer used by
            :func:`us_pr_unary_polygon`.

    Returns:
        List of one or two dicts, each with keys ``xmin, ymin, xmax, ymax``
        in WGS-84 degrees.
    """
    polygon_gdf = us_pr_unary_polygon(
        shp_path = shp_path,
        coastline_buffer_m = coastline_buffer_m,
    )
    geom = polygon_gdf.geometry.iloc[0]

    # Split the multipolygon at longitude 0: negative-x parts go into the
    # "main" bbox (CONUS, AK mainland, HI, PR), positive-x parts into the
    # "Near Islands" bbox.
    neg_parts = []
    pos_parts = []
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for part in parts:
        # A single polygon's .bounds is (minx, miny, maxx, maxy). We split on
        # the x-coordinate of the polygon's centroid; that cleanly separates
        # the eastern Aleutians (negative longitudes) from the Near Islands
        # (positive longitudes) without needing to clip at the antimeridian.
        if part.centroid.x < 0:
            neg_parts.append(part)
        else:
            pos_parts.append(part)

    bboxes: list[dict] = []
    for group in (neg_parts, pos_parts):
        if not group:
            continue
        xmins = [p.bounds[0] for p in group]
        ymins = [p.bounds[1] for p in group]
        xmaxs = [p.bounds[2] for p in group]
        ymaxs = [p.bounds[3] for p in group]
        bboxes.append({
            "xmin": min(xmins),
            "ymin": min(ymins),
            "xmax": max(xmaxs),
            "ymax": max(ymaxs),
        })
    return bboxes


# -----------------------------------------------------------------------------
# Convenience orchestrator
# -----------------------------------------------------------------------------


def get_us_pr_boundary(
    source_url: str,
    cache_dir: Path,
    coastline_buffer_m: float = 100.0,
    zip_name: str = "cb_2023_us_state_5m.zip",
    shp_name: str = "cb_2023_us_state_5m.shp",
) -> tuple[gpd.GeoDataFrame, list[dict]]:
    """
    Convenience wrapper: downloads the boundary file if needed and returns
    both the buffered polygon and the coarse bboxes.

    Args:
        source_url: URL of the Census cartographic boundary zip file.
        cache_dir: Directory for the cached shapefile.
        coastline_buffer_m: Outward buffer distance in metres.
        zip_name: Filename of the downloaded zip within ``cache_dir``.
        shp_name: Filename of the target shapefile within ``cache_dir``.

    Returns:
        Tuple of (single-row buffered polygon GeoDataFrame in EPSG:4326,
        list of coarse bbox dicts).
    """
    shp_path = download_us_pr_boundary(
        source_url = source_url,
        cache_dir = cache_dir,
        zip_name = zip_name,
        shp_name = shp_name,
    )
    polygon_gdf = us_pr_unary_polygon(
        shp_path = shp_path,
        coastline_buffer_m = coastline_buffer_m,
    )
    bboxes = us_pr_bboxes(
        shp_path = shp_path,
        coastline_buffer_m = coastline_buffer_m,
    )
    return polygon_gdf, bboxes
