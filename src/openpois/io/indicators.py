#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Assign the ``msa_code`` and ``urban_rural`` indicators to a set of points.

The indicators are computed **per element** from a single representative
location, then broadcast to all of that element's observation rows. The same
:func:`assign_indicators` is reused at model-apply time on the POI snapshot
(which already carries geometry).

It is broken into the following functions:

- classify_places: Adds a per-place ``urban_rural`` class from population +
    land area, per the project definition:
      * urban    = population > 100k AND density > 1000 / sq mi
      * suburban = otherwise, density > 500 / sq mi
      * rural    = otherwise (and, downstream, any point in no place)
- assign_indicators: Spatial-joins points to Metropolitan Statistical Areas
    (``msa_code``, ``NO_MSA`` on no match) and to classified places
    (``urban_rural``, ``rural`` on no match).
- element_coordinates: Builds one representative point per ``(osm_type, id)``
    OSM element, primarily from the current snapshot geometry and falling back
    to the last-known node lat/lon recovered from ``osm_changes``.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

from openpois.io import census_areas

# Square metres per square mile (international mile).
_SQ_M_PER_SQ_MI = 2_589_988.110336

URBAN = "urban"
SUBURBAN = "suburban"
RURAL = "rural"
NO_MSA = "NO_MSA"


def load_classified_layers(
    cbsa_shp: Path,
    place_shp: Path,
    population_csv: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load and prepare the two spatial layers needed by :func:`assign_indicators`.

    Convenience wrapper shared by the observation build and the model-apply
    step so both resolve MSAs + classified places identically.

    Args:
        cbsa_shp: Path to the CBSA cartographic-boundary shapefile.
        place_shp: Path to the Place cartographic-boundary shapefile.
        population_csv: Path to the cached place-population CSV.

    Returns:
        ``(msa_gdf, classified_places_gdf)`` — both EPSG:4326.
    """
    msa_gdf = census_areas.load_msa_boundary(cbsa_shp)
    classified = classify_places(
        census_areas.load_places(place_shp),
        census_areas.load_place_population(population_csv),
    )
    return msa_gdf, classified


def classify_places(
    places_gdf: gpd.GeoDataFrame,
    population_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Attach population, density, and an ``urban_rural`` class to each place.

    Args:
        places_gdf: Places from :func:`openpois.io.census_areas.load_places`
            with columns ``[place_geoid, place_name, aland_m2, geometry]``.
        population_df: Population from
            :func:`openpois.io.census_areas.load_place_population` with columns
            ``[place_geoid, population]``.

    Returns:
        Copy of ``places_gdf`` with added ``population``, ``density_sq_mi``, and
        ``urban_rural`` columns. Places with no population record default to
        ``rural`` (they cannot meet the density thresholds).
    """
    gdf = places_gdf.merge(population_df, on = "place_geoid", how = "left")
    land_sq_mi = gdf["aland_m2"] / _SQ_M_PER_SQ_MI
    gdf["density_sq_mi"] = gdf["population"] / land_sq_mi.where(land_sq_mi > 0)
    is_urban = (gdf["population"] > 100_000) & (gdf["density_sq_mi"] > 1_000)
    is_suburban = (~is_urban) & (gdf["density_sq_mi"] > 500)
    gdf["urban_rural"] = RURAL
    gdf.loc[is_suburban, "urban_rural"] = SUBURBAN
    gdf.loc[is_urban, "urban_rural"] = URBAN
    return gdf


def assign_indicators(
    points_gdf: gpd.GeoDataFrame,
    msa_gdf: gpd.GeoDataFrame,
    classified_places_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add ``msa_code`` and ``urban_rural`` to a points GeoDataFrame via two
    point-in-polygon spatial joins.

    Args:
        points_gdf: Points (EPSG:4326) to enrich. Any non-geometry columns are
            preserved. Must not already contain ``msa_code``/``urban_rural``.
        msa_gdf: Metropolitan Statistical Areas with ``[msa_code, geometry]``.
        classified_places_gdf: Places with ``[urban_rural, geometry]`` from
            :func:`classify_places`.

    Returns:
        Copy of ``points_gdf`` with ``msa_code`` (``NO_MSA`` outside any MSA)
        and ``urban_rural`` (``rural`` outside any place) columns added. Row
        count and order are preserved.
    """
    if points_gdf.crs is None or points_gdf.crs.to_epsg() != 4326:
        points_gdf = points_gdf.to_crs("EPSG:4326")
    out = points_gdf.copy()

    msa_join = gpd.sjoin(
        out, msa_gdf.loc[:, ["msa_code", "geometry"]], how = "left", predicate = "within"
    )
    # Overlapping polygons or boundary touches could duplicate rows; keep the
    # first match per original row so the join is row-preserving.
    msa_join = msa_join[~msa_join.index.duplicated(keep = "first")]
    out["msa_code"] = msa_join["msa_code"].fillna(NO_MSA).to_numpy()

    place_join = gpd.sjoin(
        out.drop(columns = ["msa_code"]),
        classified_places_gdf.loc[:, ["urban_rural", "geometry"]],
        how = "left",
        predicate = "within",
    )
    place_join = place_join[~place_join.index.duplicated(keep = "first")]
    out["urban_rural"] = place_join["urban_rural"].fillna(RURAL).to_numpy()
    return out


def element_indicator_map(
    unique_type_ids: pd.DataFrame,
    snapshot_path: Path,
    msa_gdf: gpd.GeoDataFrame,
    classified_places_gdf: gpd.GeoDataFrame,
    changes_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compute one ``(msa_code, urban_rural)`` per OSM element.

    Locates each ``(osm_type, id)`` via :func:`element_coordinates` and assigns
    indicators via :func:`assign_indicators`. The result is a small lookup the
    caller merges back onto the (much larger) observation rows.

    Args:
        unique_type_ids: DataFrame with ``osm_type`` and ``id`` columns.
        snapshot_path: Path to ``osm_snapshot.parquet``.
        msa_gdf: MSAs from :func:`openpois.io.census_areas.load_msa_boundary`.
        classified_places_gdf: Classified places from :func:`classify_places`.
        changes_path: Optional ``osm_changes.parquet`` for the node fallback.

    Returns:
        DataFrame with columns ``[osm_type, id, msa_code, urban_rural]``.
        Elements that could not be located get ``NO_MSA`` / ``rural``.
    """
    points = element_coordinates(
        unique_type_ids, snapshot_path, changes_path = changes_path
    )
    located = points.dropna(subset = ["geometry"]).copy()
    if not located.empty:
        enriched = assign_indicators(located, msa_gdf, classified_places_gdf)
        enriched = enriched.loc[:, ["osm_type", "id", "msa_code", "urban_rural"]]
    else:
        enriched = pd.DataFrame(
            columns = ["osm_type", "id", "msa_code", "urban_rural"]
        )
    # Left-join back so unlocated elements still appear, defaulted.
    out = points.loc[:, ["osm_type", "id"]].merge(
        enriched, on = ["osm_type", "id"], how = "left"
    )
    out["msa_code"] = out["msa_code"].fillna(NO_MSA)
    out["urban_rural"] = out["urban_rural"].fillna(RURAL)
    return out


def element_coordinates(
    unique_type_ids: pd.DataFrame,
    snapshot_path: Path,
    changes_path: Path | None = None,
    duckdb_memory_limit: str = "4GB",
) -> gpd.GeoDataFrame:
    """
    Build one representative point per ``(osm_type, id)`` OSM element.

    The primary source is the current snapshot geometry, joined by
    ``(osm_type, osm_id)`` and reduced to a centroid. Elements absent from the
    snapshot (deleted "ghosts") fall back to their last-known node lat/lon,
    recovered from the ``lat``/``lon`` pseudo-tags in ``osm_changes`` (nodes
    only). Elements that resolve to neither are returned with a null geometry;
    the downstream :func:`assign_indicators` will give them ``NO_MSA`` / rural.

    Args:
        unique_type_ids: DataFrame with ``osm_type`` (``node``/``way``/
            ``relation``) and ``id`` columns — the distinct elements to locate.
        snapshot_path: Path to ``osm_snapshot.parquet`` (GeoParquet, EPSG:4326,
            with ``osm_type``, ``osm_id``, ``geometry``).
        changes_path: Optional ``osm_changes.parquet`` for the node fallback.
        duckdb_memory_limit: DuckDB memory cap for the changes fallback query.

    Returns:
        GeoDataFrame keyed on ``[osm_type, id]`` with a point ``geometry`` in
        EPSG:4326 (null where the element could not be located).
    """
    keys = (
        unique_type_ids.loc[:, ["osm_type", "id"]]
        .drop_duplicates()
        .reset_index(drop = True)
    )

    snap = gpd.read_parquet(
        snapshot_path, columns = ["osm_type", "osm_id", "geometry"]
    )
    if snap.crs is None or snap.crs.to_epsg() != 4326:
        snap = snap.to_crs("EPSG:4326")
    snap = snap.rename(columns = {"osm_id": "id"})
    # Coarse indicators only need a representative interior point.
    snap["geometry"] = snap.geometry.representative_point()
    snap = snap.loc[:, ["osm_type", "id", "geometry"]]

    located = keys.merge(snap, on = ["osm_type", "id"], how = "left")
    located = gpd.GeoDataFrame(located, geometry = "geometry", crs = "EPSG:4326")

    missing = located[located.geometry.isna()]
    if changes_path is not None and not missing.empty:
        fallback = _node_coords_from_changes(
            missing.loc[missing["osm_type"] == "node", "id"],
            changes_path,
            duckdb_memory_limit,
        )
        if not fallback.empty:
            located = located.merge(
                fallback, on = "id", how = "left", suffixes = ("", "_fb")
            )
            fill = located.geometry.isna() & located["geometry_fb"].notna()
            located.loc[fill, "geometry"] = located.loc[fill, "geometry_fb"]
            located = located.drop(columns = ["geometry_fb"])
            located = gpd.GeoDataFrame(
                located, geometry = "geometry", crs = "EPSG:4326"
            )

    return located.loc[:, ["osm_type", "id", "geometry"]]


def _node_coords_from_changes(
    node_ids: pd.Series,
    changes_path: Path,
    duckdb_memory_limit: str,
) -> gpd.GeoDataFrame:
    """
    Recover the last-known lat/lon of the given node ids from ``osm_changes``.

    The history parser stores ``lat``/``lon`` as pseudo-tags (one change row per
    coordinate edit). For each node we take the value at its highest version.
    Returns an empty frame if no coordinates are found.
    """
    ids = pd.unique(node_ids.dropna()).tolist()
    if not ids:
        return gpd.GeoDataFrame(
            columns = ["id", "geometry"], geometry = "geometry", crs = "EPSG:4326"
        )
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.register("wanted", pd.DataFrame({"id": ids}))
        # Pick each node's latest lat and lon via argmax over version.
        df = con.execute(
            f"""
            SELECT c.id,
                   arg_max(CASE WHEN c.key = 'lat' THEN c.value END, c.version)
                       AS lat,
                   arg_max(CASE WHEN c.key = 'lon' THEN c.value END, c.version)
                       AS lon
            FROM read_parquet('{Path(changes_path).as_posix()}') c
            SEMI JOIN wanted w ON c.id = w.id
            WHERE c.type = 'node' AND c.key IN ('lat', 'lon')
            GROUP BY c.id
            """
        ).fetch_df()
    finally:
        con.close()
    df["lat"] = pd.to_numeric(df["lat"], errors = "coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors = "coerce")
    df = df.dropna(subset = ["lat", "lon"])
    return gpd.GeoDataFrame(
        df.loc[:, ["id"]],
        geometry = gpd.points_from_xy(df["lon"], df["lat"]),
        crs = "EPSG:4326",
    )
