#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Census reference areas used to enrich OSM observations and snapshots with two
indicators: ``msa_code`` (Metropolitan Statistical Area) and ``urban_rural``
(urban / suburban / rural). All three sources mirror the download/cache pattern
of :mod:`openpois.io.boundary` (stream-download, unzip, skip-if-exists, return
EPSG:4326).

It is broken into the following functions:

- download_census_zip: Generic streaming download + unzip of a Census
    cartographic-boundary shapefile zip, returning the ``.shp`` path.
- load_msa_boundary: Reads the national CBSA cartographic-boundary shapefile
    and filters to Metropolitan Statistical Areas (``LSAD == "M1"``), dropping
    Micropolitan areas. Returns ``[msa_code, msa_name, geometry]``.
- load_places: Reads the national Place cartographic-boundary shapefile
    (incorporated cities + Census-Designated Places). Returns
    ``[place_geoid, place_name, aland_m2, geometry]``.
- fetch_place_population: Queries the 2020 Decennial Census (PL, table
    ``P1_001N``) for total population of every place (incorporated AND CDP) via
    the public Census API, and caches the result as a static CSV. The Decennial
    is used rather than the Population Estimates Program because PEP excludes
    CDPs, which the urban/suburban classification needs.
- load_place_population: Reads the cached population CSV, returning
    ``[place_geoid, population]``.

The cartographic-boundary (``cb_*_500k``) line is used rather than the raw
TIGER line because Census publishes single national ``us`` files for CBSA and
Place there, whereas raw TIGER publishes Place only per-state.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

# 2020 Decennial PL place query: total population (P1_001N) for every place in
# every state. Covers incorporated places and CDPs. Requires a (free) Census
# API key, appended by ``fetch_place_population``.
DEFAULT_POPULATION_API_URL = (
    "https://api.census.gov/data/2020/dec/pl"
    "?get=NAME,P1_001N&for=place:*&in=state:*"
)

# CENSUS_API_KEY is read from the environment, falling back to ~/.Renviron
# (where the maintainer defines it for R sessions).
_RENVIRON_PATH = Path("~/.Renviron").expanduser()

# CBSA LSAD code for Metropolitan Statistical Areas. "M2" = Micropolitan.
_METRO_LSAD = "M1"


# -----------------------------------------------------------------------------
# Shapefile download
# -----------------------------------------------------------------------------


def download_census_zip(
    source_url: str,
    cache_dir: Path,
    shp_name: str,
    overwrite: bool = False,
    timeout: int = 120,
) -> Path:
    """
    Download and unzip a Census cartographic-boundary shapefile zip.

    Caches in ``cache_dir``. If the expected ``.shp`` already exists (and
    ``overwrite`` is False) this is a no-op and returns the path.

    Args:
        source_url: URL of the Census cartographic-boundary zip file.
        cache_dir: Directory where the zip + unzipped components are stored.
        shp_name: Filename of the target shapefile within ``cache_dir`` (e.g.
            ``cb_2023_us_cbsa_500k.shp``).
        overwrite: If True, always re-download and re-unzip.
        timeout: Per-request timeout in seconds.

    Returns:
        Path to the unzipped ``.shp`` file in ``cache_dir``.

    Raises:
        requests.HTTPError: If the download fails.
        FileNotFoundError: If the expected shapefile is missing after unzip.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents = True, exist_ok = True)
    shp_path = cache_dir / shp_name
    if shp_path.exists() and not overwrite:
        return shp_path

    print(f"Downloading Census boundary from {source_url}...")
    resp = requests.get(source_url, stream = True, timeout = timeout)
    resp.raise_for_status()
    zip_path = cache_dir / Path(source_url).name
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size = 1024 * 1024):
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)

    if not shp_path.exists():
        raise FileNotFoundError(
            f"Expected shapefile {shp_path} not found after unzipping "
            f"{zip_path}. Contents: {sorted(p.name for p in cache_dir.iterdir())}"
        )
    return shp_path


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


def load_msa_boundary(shp_path: Path) -> gpd.GeoDataFrame:
    """
    Read the national CBSA shapefile and return only Metropolitan Statistical
    Areas, in EPSG:4326.

    Micropolitan areas (``LSAD == "M2"``) are dropped — observations inside a
    micropolitan CBSA, or outside any CBSA, are treated as ``NO_MSA`` by the
    downstream spatial join (see :mod:`openpois.io.indicators`).

    Args:
        shp_path: Path to the unzipped CBSA cartographic-boundary ``.shp``.

    Returns:
        GeoDataFrame with columns ``[msa_code, msa_name, geometry]`` in
        EPSG:4326, one row per Metropolitan Statistical Area.

    Raises:
        ValueError: If no metropolitan rows are found (wrong/empty file).
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.loc[gdf["LSAD"] == _METRO_LSAD].copy()
    if gdf.empty:
        raise ValueError(
            f"No Metropolitan Statistical Areas (LSAD == {_METRO_LSAD!r}) found "
            f"in {shp_path}; check the source file."
        )
    gdf = gdf.rename(columns = {"GEOID": "msa_code", "NAME": "msa_name"})
    return gdf.loc[:, ["msa_code", "msa_name", "geometry"]].reset_index(drop = True)


def load_places(shp_path: Path) -> gpd.GeoDataFrame:
    """
    Read the national Place shapefile (incorporated cities + CDPs), EPSG:4326.

    Args:
        shp_path: Path to the unzipped Place cartographic-boundary ``.shp``.

    Returns:
        GeoDataFrame with columns ``[place_geoid, place_name, aland_m2,
        geometry]`` in EPSG:4326. ``place_geoid`` = STATEFP + PLACEFP (the
        7-character Census place GEOID); ``aland_m2`` is land area in m².
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.rename(
        columns = {"GEOID": "place_geoid", "NAME": "place_name", "ALAND": "aland_m2"}
    )
    return gdf.loc[
        :, ["place_geoid", "place_name", "aland_m2", "geometry"]
    ].reset_index(drop = True)


# -----------------------------------------------------------------------------
# Place population (2020 Decennial, cached to CSV)
# -----------------------------------------------------------------------------


def _resolve_census_api_key(api_key: str | None = None) -> str:
    """
    Resolve the Census API key from (in order) the explicit argument, the
    ``CENSUS_API_KEY`` environment variable, or a ``CENSUS_API_KEY=...`` line in
    ``~/.Renviron``.

    Raises:
        RuntimeError: If no key can be found.
    """
    if api_key:
        return api_key
    env_key = os.environ.get("CENSUS_API_KEY")
    if env_key:
        return env_key
    if _RENVIRON_PATH.exists():
        for line in _RENVIRON_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("CENSUS_API_KEY"):
                _, _, value = line.partition("=")
                return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "No Census API key found. Set CENSUS_API_KEY in the environment or in "
        "~/.Renviron, or pass api_key=... (free key: "
        "https://api.census.gov/data/key_signup.html)."
    )


def fetch_place_population(
    cache_csv: Path,
    api_url: str = DEFAULT_POPULATION_API_URL,
    api_key: str | None = None,
    overwrite: bool = False,
    timeout: int = 120,
) -> Path:
    """
    Fetch 2020 Decennial total population for every place and cache it as CSV.

    The query (``2020/dec/pl``, variable ``P1_001N``) returns one row per place
    across all states, covering both incorporated places and CDPs. The result
    is written once to ``cache_csv`` with columns ``[place_geoid, population]``
    and reused thereafter, so subsequent runs are a pure static-file read.

    Args:
        cache_csv: Destination CSV path.
        api_url: Census API query URL (default: 2020 Decennial PL, P1_001N).
        api_key: Census API key. If None, resolved from ``CENSUS_API_KEY`` in
            the environment or ``~/.Renviron`` (see
            :func:`_resolve_census_api_key`).
        overwrite: If True, always re-query and overwrite the cache.
        timeout: Per-request timeout in seconds.

    Returns:
        Path to the cached CSV.

    Raises:
        requests.HTTPError: If the API request fails.
        RuntimeError: If no Census API key can be resolved.
    """
    cache_csv = Path(cache_csv)
    if cache_csv.exists() and not overwrite:
        return cache_csv

    key = _resolve_census_api_key(api_key)
    print("Fetching 2020 Decennial place population from the Census API...")
    resp = requests.get(api_url, params = {"key": key}, timeout = timeout)
    resp.raise_for_status()
    rows = resp.json()
    # First row is the header: e.g. ["NAME", "P1_001N", "state", "place"].
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns = header)
    df["place_geoid"] = df["state"].str.zfill(2) + df["place"].str.zfill(5)
    df["population"] = pd.to_numeric(df["P1_001N"], errors = "coerce").astype("Int64")
    out = df.loc[:, ["place_geoid", "population"]].dropna(subset = ["population"])
    cache_csv.parent.mkdir(parents = True, exist_ok = True)
    out.to_csv(cache_csv, index = False)
    print(f"Cached {len(out):,} place populations to {cache_csv}")
    return cache_csv


def load_place_population(cache_csv: Path) -> pd.DataFrame:
    """
    Read the cached place-population CSV.

    Args:
        cache_csv: Path to the CSV written by :func:`fetch_place_population`.

    Returns:
        DataFrame with columns ``[place_geoid, population]`` (``place_geoid``
        as a zero-padded 7-character string).
    """
    df = pd.read_csv(cache_csv, dtype = {"place_geoid": str})
    df["place_geoid"] = df["place_geoid"].str.zfill(7)
    return df.loc[:, ["place_geoid", "population"]]
