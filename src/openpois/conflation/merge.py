#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Merge matched and unmatched POIs into a unified conflated dataset.

Produces a GeoDataFrame superset:
  - Matched pairs (OSM + Overture) with blended confidence.
  - Unmatched OSM POIs with their original confidence.
  - Unmatched Overture POIs with downweighted confidence.

Three entry points:
  - ``merge_matched_pois``: in-memory, for tests/small datasets.
  - ``build_merge_parts``: disk-backed, row-sliced. Writes multiple
    part parquets so peak memory is bounded by slice size.
  - ``build_merge_parts_chunked``: disk-backed, spatial-chunk-sliced.
    Reuses the ``osm_primary`` / ``overture_primary`` arrays produced
    by the chunked matching driver so each per-chunk part is small
    and independent.
"""
from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely


def _pick_geometries(
    osm_geoms: np.ndarray,
    overture_geoms: np.ndarray,
) -> np.ndarray:
    """
    Vectorized geometry selection: prefer higher-level geometry type,
    OSM on ties.
    """
    osm_types = shapely.get_type_id(osm_geoms)
    ov_types = shapely.get_type_id(overture_geoms)
    rank_table = np.ones(8, dtype = np.int8)
    rank_table[0] = 1  # Point
    rank_table[1] = 2  # LineString
    rank_table[3] = 3  # Polygon
    rank_table[6] = 4  # MultiPolygon
    osm_ranks = rank_table[osm_types]
    ov_ranks = rank_table[ov_types]
    use_overture = ov_ranks > osm_ranks
    result = osm_geoms.copy()
    result[use_overture] = overture_geoms[use_overture]
    return result


def _present(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: True where the entry is non-null and, if a
    string, non-empty. Overture occasionally emits ``""`` for missing
    address subfields; treat that as missing so backfill kicks in.
    """
    notna = pd.notna(arr)
    if arr.dtype != object:
        return notna
    nonempty = np.array(
        [v != "" for v in arr], dtype = bool,
    )
    return notna & nonempty


def _blend_with_backfill(
    osm_vals: np.ndarray,
    ov_vals: np.ndarray,
    osm_higher: np.ndarray,
) -> np.ndarray:
    """Pick from the higher-confidence source per row, backfilling
    from the other side whenever the preferred value is missing
    (None, NaN, or empty string).

    Bidirectional backfill — fixes the prior one-sided brand blend
    that dropped Overture's brand whenever OSM had higher confidence
    but a null brand. Used for ``name``, ``brand``, every ``addr_*``
    field, and the unwrapped ``phone`` / ``website`` primaries.
    """
    osm_ok = _present(osm_vals)
    ov_ok = _present(ov_vals)
    out = np.where(osm_higher, osm_vals, ov_vals)
    use_ov = osm_higher & ~osm_ok & ov_ok
    use_osm = (~osm_higher) & ~ov_ok & osm_ok
    out = np.where(use_ov, ov_vals, out)
    out = np.where(use_osm, osm_vals, out)
    return out


def _col_or_null(
    gdf: gpd.GeoDataFrame,
    col: str,
    idx: np.ndarray,
) -> np.ndarray:
    """Return ``gdf[col].to_numpy()[idx]`` if column exists, else an
    object array of ``None`` of length ``len(idx)``. Mirrors the
    inline ``"brand" in osm_gdf.columns`` guard that previously
    appeared in three places.
    """
    if col in gdf.columns:
        return gdf[col].to_numpy()[idx]
    return np.full(len(idx), None, dtype = object)


def _unwrap_first_element(arr: np.ndarray) -> np.ndarray:
    """Flatten an array of LIST<STRING> values (Overture
    ``websites`` / ``phones``) to an object array of the first
    element or ``None``. Empty strings become ``None`` so backfill
    treats them as missing.
    """
    out = np.empty(len(arr), dtype = object)
    for i, x in enumerate(arr):
        if x is None:
            out[i] = None
            continue
        if isinstance(x, float) and np.isnan(x):
            out[i] = None
            continue
        try:
            if len(x) > 0:
                v = x[0]
                out[i] = v if v != "" else None
            else:
                out[i] = None
        except TypeError:
            out[i] = None
    return out


def _build_matched_gdf(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    osm_w: float,
    ov_w: float,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame for matched pairs."""
    oi = matches["osm_idx"].to_numpy()
    vi = matches["overture_idx"].to_numpy()

    osm_conf = osm_gdf["conf_mean"].to_numpy()[oi].astype(float)
    ov_conf_raw = overture_gdf["confidence"].to_numpy()[vi]
    ov_conf = pd.to_numeric(
        ov_conf_raw, errors = "coerce"
    ).astype(float)
    ov_conf = np.where(np.isnan(ov_conf), 0.5, ov_conf)
    osm_higher = osm_conf >= ov_conf

    # Dual-source string fields — blended primary + per-source
    # traceability arrays. ``_blend_with_backfill`` handles the
    # higher-confidence pick AND bidirectional null/empty backfill.
    osm_names = osm_gdf["name"].to_numpy()[oi]
    ov_names = overture_gdf["overture_name"].to_numpy()[vi]
    names = _blend_with_backfill(osm_names, ov_names, osm_higher)

    osm_brands = _col_or_null(osm_gdf, "brand", oi)
    ov_brands = _col_or_null(overture_gdf, "brand_name", vi)
    brands = _blend_with_backfill(osm_brands, ov_brands, osm_higher)

    # Address subfields. ``housenumber`` and ``unit`` are OSM-only;
    # the remaining five exist on both sides.
    osm_addr_hn = _col_or_null(osm_gdf, "addr:housenumber", oi)
    osm_addr_st = _col_or_null(osm_gdf, "addr:street", oi)
    osm_addr_unit = _col_or_null(osm_gdf, "addr:unit", oi)
    osm_addr_city = _col_or_null(osm_gdf, "addr:city", oi)
    osm_addr_state = _col_or_null(osm_gdf, "addr:state", oi)
    osm_addr_pc = _col_or_null(osm_gdf, "addr:postcode", oi)
    osm_addr_country = _col_or_null(osm_gdf, "addr:country", oi)
    ov_addr_st = _col_or_null(
        overture_gdf, "overture_addr_street", vi,
    )
    ov_addr_city = _col_or_null(
        overture_gdf, "overture_addr_city", vi,
    )
    ov_addr_state = _col_or_null(
        overture_gdf, "overture_addr_state", vi,
    )
    ov_addr_pc = _col_or_null(
        overture_gdf, "overture_addr_postcode", vi,
    )
    ov_addr_country = _col_or_null(
        overture_gdf, "overture_addr_country", vi,
    )
    addr_street = _blend_with_backfill(
        osm_addr_st, ov_addr_st, osm_higher,
    )
    addr_city = _blend_with_backfill(
        osm_addr_city, ov_addr_city, osm_higher,
    )
    addr_state = _blend_with_backfill(
        osm_addr_state, ov_addr_state, osm_higher,
    )
    addr_postcode = _blend_with_backfill(
        osm_addr_pc, ov_addr_pc, osm_higher,
    )
    addr_country = _blend_with_backfill(
        osm_addr_country, ov_addr_country, osm_higher,
    )

    # Contact arrays. Overture sides are LIST<STRING>; unwrap to the
    # first element for the blended primary, preserve the full list
    # in the traceability column.
    osm_phone = _col_or_null(osm_gdf, "phone", oi)
    osm_website = _col_or_null(osm_gdf, "website", oi)
    ov_phones = _col_or_null(overture_gdf, "overture_phones", vi)
    ov_websites = _col_or_null(overture_gdf, "overture_websites", vi)
    phone = _blend_with_backfill(
        osm_phone, _unwrap_first_element(ov_phones), osm_higher,
    )
    website = _blend_with_backfill(
        osm_website, _unwrap_first_element(ov_websites), osm_higher,
    )

    # OSM-only fields
    opening_hours = _col_or_null(osm_gdf, "opening_hours", oi)
    access = _col_or_null(osm_gdf, "access", oi)

    # Overture-only LIST<STRING> fields — preserved as-is.
    overture_socials = _col_or_null(
        overture_gdf, "overture_socials", vi,
    )
    overture_categories_alternate = _col_or_null(
        overture_gdf, "overture_categories_alternate", vi,
    )

    merged_conf = osm_conf * osm_w + ov_conf * ov_w

    osm_conf_lower = osm_gdf["conf_lower"].to_numpy()[oi].astype(
        float
    )
    osm_conf_upper = osm_gdf["conf_upper"].to_numpy()[oi].astype(
        float
    )
    conf_lower = osm_conf_lower * osm_w + ov_conf * ov_w
    conf_upper = osm_conf_upper * osm_w + ov_conf * ov_w

    osm_geoms = osm_gdf.geometry.to_numpy()[oi]
    ov_geoms = overture_gdf.geometry.to_numpy()[vi]
    geoms = _pick_geometries(osm_geoms, ov_geoms)

    osm_ids = osm_gdf["osm_id"].to_numpy()[oi]
    osm_types = osm_gdf["osm_type"].to_numpy()[oi]
    ov_ids = overture_gdf["overture_id"].to_numpy()[vi]

    unified_ids = np.array(
        [
            f"matched:{o}_{v}"
            for o, v in zip(osm_ids, ov_ids)
        ],
        dtype = object,
    )

    return gpd.GeoDataFrame(
        {
            "unified_id": unified_ids,
            "source": "matched",
            "osm_id": osm_ids,
            "osm_type": osm_types,
            "overture_id": ov_ids,
            "name": names,
            "brand": brands,
            "shared_label": osm_shared_labels[oi],
            "conf_mean": merged_conf,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "match_score": matches["composite_score"].to_numpy(),
            "match_distance_m": matches["distance_m"].to_numpy(),
            "addr_housenumber": osm_addr_hn,
            "addr_street": addr_street,
            "addr_unit": osm_addr_unit,
            "addr_city": addr_city,
            "addr_state": addr_state,
            "addr_postcode": addr_postcode,
            "addr_country": addr_country,
            "phone": phone,
            "website": website,
            "opening_hours": opening_hours,
            "access": access,
            "overture_socials": overture_socials,
            "overture_categories_alternate":
                overture_categories_alternate,
            "osm_name": osm_names,
            "overture_name": ov_names,
            "osm_brand": osm_brands,
            "overture_brand": ov_brands,
            "osm_addr_housenumber": osm_addr_hn,
            "osm_addr_street": osm_addr_st,
            "osm_addr_unit": osm_addr_unit,
            "osm_addr_city": osm_addr_city,
            "osm_addr_state": osm_addr_state,
            "osm_addr_postcode": osm_addr_pc,
            "osm_addr_country": osm_addr_country,
            "overture_addr_street": ov_addr_st,
            "overture_addr_city": ov_addr_city,
            "overture_addr_state": ov_addr_state,
            "overture_addr_postcode": ov_addr_pc,
            "overture_addr_country": ov_addr_country,
            "osm_phone": osm_phone,
            "overture_phones": ov_phones,
            "osm_website": osm_website,
            "overture_websites": ov_websites,
            "osm_conf_mean": osm_conf,
            "overture_confidence": ov_conf,
        },
        geometry = geoms,
        crs = osm_gdf.crs,
    )


def _build_unmatched_osm_gdf(
    osm_gdf: gpd.GeoDataFrame,
    idx: np.ndarray,
    osm_shared_labels: np.ndarray,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame for unmatched OSM POIs at the given indices.

    Uses column-wise ``to_numpy()[idx]`` to avoid a full ``.iloc[idx]``
    copy — the old implementation held both the source frame and the
    full iloc copy in memory simultaneously.
    """
    n = len(idx)

    osm_ids = osm_gdf["osm_id"].to_numpy()[idx]
    osm_types = osm_gdf["osm_type"].to_numpy()[idx]
    names = osm_gdf["name"].to_numpy()[idx]
    brand_arr = _col_or_null(osm_gdf, "brand", idx)
    conf_mean = osm_gdf["conf_mean"].to_numpy()[idx].astype(float)
    conf_lower = osm_gdf["conf_lower"].to_numpy()[idx].astype(float)
    conf_upper = osm_gdf["conf_upper"].to_numpy()[idx].astype(float)
    geoms = osm_gdf.geometry.to_numpy()[idx]

    osm_addr_hn = _col_or_null(osm_gdf, "addr:housenumber", idx)
    osm_addr_st = _col_or_null(osm_gdf, "addr:street", idx)
    osm_addr_unit = _col_or_null(osm_gdf, "addr:unit", idx)
    osm_addr_city = _col_or_null(osm_gdf, "addr:city", idx)
    osm_addr_state = _col_or_null(osm_gdf, "addr:state", idx)
    osm_addr_pc = _col_or_null(osm_gdf, "addr:postcode", idx)
    osm_addr_country = _col_or_null(osm_gdf, "addr:country", idx)
    osm_phone = _col_or_null(osm_gdf, "phone", idx)
    osm_website = _col_or_null(osm_gdf, "website", idx)
    opening_hours = _col_or_null(osm_gdf, "opening_hours", idx)
    access = _col_or_null(osm_gdf, "access", idx)

    nulls = lambda: np.full(n, None, dtype = object)

    unified_ids = np.array(
        [f"osm:{x}" for x in osm_ids], dtype = object,
    )

    return gpd.GeoDataFrame(
        {
            "unified_id": unified_ids,
            "source": "osm",
            "osm_id": osm_ids,
            "osm_type": osm_types,
            "overture_id": nulls(),
            "name": names,
            "brand": brand_arr,
            "shared_label": osm_shared_labels[idx],
            "conf_mean": conf_mean,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "match_score": np.full(n, np.nan),
            "match_distance_m": np.full(n, np.nan),
            "addr_housenumber": osm_addr_hn,
            "addr_street": osm_addr_st,
            "addr_unit": osm_addr_unit,
            "addr_city": osm_addr_city,
            "addr_state": osm_addr_state,
            "addr_postcode": osm_addr_pc,
            "addr_country": osm_addr_country,
            "phone": osm_phone,
            "website": osm_website,
            "opening_hours": opening_hours,
            "access": access,
            "overture_socials": nulls(),
            "overture_categories_alternate": nulls(),
            "osm_name": names,
            "overture_name": nulls(),
            "osm_brand": brand_arr,
            "overture_brand": nulls(),
            "osm_addr_housenumber": osm_addr_hn,
            "osm_addr_street": osm_addr_st,
            "osm_addr_unit": osm_addr_unit,
            "osm_addr_city": osm_addr_city,
            "osm_addr_state": osm_addr_state,
            "osm_addr_postcode": osm_addr_pc,
            "osm_addr_country": osm_addr_country,
            "overture_addr_street": nulls(),
            "overture_addr_city": nulls(),
            "overture_addr_state": nulls(),
            "overture_addr_postcode": nulls(),
            "overture_addr_country": nulls(),
            "osm_phone": osm_phone,
            "overture_phones": nulls(),
            "osm_website": osm_website,
            "overture_websites": nulls(),
            "osm_conf_mean": conf_mean,
            "overture_confidence": np.full(n, np.nan),
        },
        geometry = geoms,
        crs = osm_gdf.crs,
    )


def _build_unmatched_overture_gdf(
    overture_gdf: gpd.GeoDataFrame,
    idx: np.ndarray,
    overture_shared_labels: np.ndarray,
    w: float,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame for unmatched Overture POIs at the given
    indices.
    """
    n = len(idx)

    ov_ids = overture_gdf["overture_id"].to_numpy()[idx]
    names = overture_gdf["overture_name"].to_numpy()[idx]
    brand_arr = _col_or_null(overture_gdf, "brand_name", idx)
    ov_conf_raw = overture_gdf["confidence"].to_numpy()[idx]
    ov_conf = pd.to_numeric(
        ov_conf_raw, errors = "coerce"
    ).astype(float)
    ov_conf = np.where(np.isnan(ov_conf), 0.5, ov_conf)
    geoms = overture_gdf.geometry.to_numpy()[idx]

    ov_addr_st = _col_or_null(
        overture_gdf, "overture_addr_street", idx,
    )
    ov_addr_city = _col_or_null(
        overture_gdf, "overture_addr_city", idx,
    )
    ov_addr_state = _col_or_null(
        overture_gdf, "overture_addr_state", idx,
    )
    ov_addr_pc = _col_or_null(
        overture_gdf, "overture_addr_postcode", idx,
    )
    ov_addr_country = _col_or_null(
        overture_gdf, "overture_addr_country", idx,
    )
    ov_phones = _col_or_null(overture_gdf, "overture_phones", idx)
    ov_websites = _col_or_null(overture_gdf, "overture_websites", idx)
    ov_socials = _col_or_null(overture_gdf, "overture_socials", idx)
    ov_categories_alt = _col_or_null(
        overture_gdf, "overture_categories_alternate", idx,
    )
    phone_primary = _unwrap_first_element(ov_phones)
    website_primary = _unwrap_first_element(ov_websites)

    nulls = lambda: np.full(n, None, dtype = object)

    unified_ids = np.array(
        [f"overture:{x}" for x in ov_ids], dtype = object,
    )

    return gpd.GeoDataFrame(
        {
            "unified_id": unified_ids,
            "source": "overture",
            "osm_id": nulls(),
            "osm_type": nulls(),
            "overture_id": ov_ids,
            "name": names,
            "brand": brand_arr,
            "shared_label": overture_shared_labels[idx],
            "conf_mean": ov_conf * w,
            "conf_lower": np.full(n, np.nan),
            "conf_upper": np.full(n, np.nan),
            "match_score": np.full(n, np.nan),
            "match_distance_m": np.full(n, np.nan),
            "addr_housenumber": nulls(),
            "addr_street": ov_addr_st,
            "addr_unit": nulls(),
            "addr_city": ov_addr_city,
            "addr_state": ov_addr_state,
            "addr_postcode": ov_addr_pc,
            "addr_country": ov_addr_country,
            "phone": phone_primary,
            "website": website_primary,
            "opening_hours": nulls(),
            "access": nulls(),
            "overture_socials": ov_socials,
            "overture_categories_alternate": ov_categories_alt,
            "osm_name": nulls(),
            "overture_name": names,
            "osm_brand": nulls(),
            "overture_brand": brand_arr,
            "osm_addr_housenumber": nulls(),
            "osm_addr_street": nulls(),
            "osm_addr_unit": nulls(),
            "osm_addr_city": nulls(),
            "osm_addr_state": nulls(),
            "osm_addr_postcode": nulls(),
            "osm_addr_country": nulls(),
            "overture_addr_street": ov_addr_st,
            "overture_addr_city": ov_addr_city,
            "overture_addr_state": ov_addr_state,
            "overture_addr_postcode": ov_addr_pc,
            "overture_addr_country": ov_addr_country,
            "osm_phone": nulls(),
            "overture_phones": ov_phones,
            "osm_website": nulls(),
            "overture_websites": ov_websites,
            "osm_conf_mean": np.full(n, np.nan),
            "overture_confidence": ov_conf,
        },
        geometry = geoms,
        crs = overture_gdf.crs,
    )


def _unmatched_idx(
    n: int, matched_idx: np.ndarray,
) -> np.ndarray:
    """Return the sorted indices in ``[0, n)`` not present in
    ``matched_idx``.
    """
    mask = np.ones(n, dtype = bool)
    if len(matched_idx) > 0:
        mask[matched_idx] = False
    return np.where(mask)[0]


# -----------------------------------------------------------------
# In-memory merge (for tests and small datasets)
# -----------------------------------------------------------------


def merge_matched_pois(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    overture_confidence_weight: float = 0.7,
) -> gpd.GeoDataFrame:
    """
    Build the unified conflated dataset from matches + unmatched.

    This in-memory version is suitable for tests and small datasets.
    For large datasets, use ``build_merge_parts`` (row-sliced) or
    ``build_merge_parts_chunked`` (spatial-chunk-sliced) +
    ``save_conflated_from_parts``.

    Returns:
        Conflated GeoDataFrame with unified schema.
    """
    w = overture_confidence_weight
    osm_w = 1.0 / (1.0 + w)
    ov_w = w / (1.0 + w)

    matched_osm_idx = matches["osm_idx"].to_numpy()
    matched_ov_idx = matches["overture_idx"].to_numpy()

    parts = []

    if len(matches) > 0:
        parts.append(
            _build_matched_gdf(
                osm_gdf, overture_gdf, matches,
                osm_shared_labels, osm_w, ov_w,
            )
        )

    parts.append(
        _build_unmatched_osm_gdf(
            osm_gdf,
            _unmatched_idx(len(osm_gdf), matched_osm_idx),
            osm_shared_labels,
        )
    )

    parts.append(
        _build_unmatched_overture_gdf(
            overture_gdf,
            _unmatched_idx(len(overture_gdf), matched_ov_idx),
            overture_shared_labels, w,
        )
    )

    # Normalize CRS across parts — OSM and Overture may declare the
    # same WGS84 lon/lat system with different authority strings
    # (e.g. "WGS 84" vs "WGS 84 (CRS84)"), which breaks pd.concat.
    target_crs = osm_gdf.crs
    for part in parts:
        if part.crs != target_crs:
            part.set_crs(
                target_crs, allow_override = True, inplace = True,
            )

    result = pd.concat(parts, ignore_index = True)
    return gpd.GeoDataFrame(result, crs = target_crs)


# -----------------------------------------------------------------
# Disk-backed merge (for large datasets)
# -----------------------------------------------------------------


def _write_part(
    gdf: gpd.GeoDataFrame, path: Path,
) -> None:
    gdf.to_parquet(path, compression = "zstd")


def _split_indices(
    idx: np.ndarray, n_slices: int,
) -> list[np.ndarray]:
    """Split an index array into ``n_slices`` roughly-equal contiguous
    ranges. Preserves order so downstream concat stays deterministic.
    """
    if n_slices <= 1 or len(idx) == 0:
        return [idx]
    return [s for s in np.array_split(idx, n_slices) if len(s) > 0]


def build_merge_parts(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    overture_confidence_weight: float = 0.7,
    n_slices: int = 4,
) -> list[Path]:
    """
    Build each merge subset, writing to temp parquet files.

    Unmatched OSM and Overture rows are split into ``n_slices``
    contiguous row ranges each, and each slice is built and written
    independently. This caps peak memory at roughly
    ``(1 / n_slices)`` of the full-dataset footprint for unmatched
    parts. The matched part is written as a single file (it is the
    smallest and already bounded by the number of matches).

    Returns:
        List of temp parquet file paths in concat order.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix = "openpois_merge_"))

    w = overture_confidence_weight
    osm_w = 1.0 / (1.0 + w)
    ov_w = w / (1.0 + w)

    matched_osm_idx = matches["osm_idx"].to_numpy()
    matched_ov_idx = matches["overture_idx"].to_numpy()
    part_paths: list[Path] = []

    # Part 1: matched pairs (single file)
    if len(matches) > 0:
        print(f"  Building {len(matches):,} matched pairs ...")
        part = _build_matched_gdf(
            osm_gdf, overture_gdf, matches,
            osm_shared_labels, osm_w, ov_w,
        )
        p = tmp_dir / "1_matched.parquet"
        _write_part(part, p)
        part_paths.append(p)
        del part
        gc.collect()

    # Part 2: unmatched OSM (sliced)
    unmatched_osm = _unmatched_idx(
        len(osm_gdf), matched_osm_idx,
    )
    osm_slices = _split_indices(unmatched_osm, n_slices)
    print(
        f"  Building {len(unmatched_osm):,} unmatched OSM POIs "
        f"in {len(osm_slices)} slice(s) ..."
    )
    for i, sl in enumerate(osm_slices):
        part = _build_unmatched_osm_gdf(
            osm_gdf, sl, osm_shared_labels,
        )
        p = tmp_dir / f"2_unmatched_osm_{i:02d}.parquet"
        _write_part(part, p)
        part_paths.append(p)
        del part
        gc.collect()

    # Part 3: unmatched Overture (sliced)
    unmatched_ov = _unmatched_idx(
        len(overture_gdf), matched_ov_idx,
    )
    ov_slices = _split_indices(unmatched_ov, n_slices)
    print(
        f"  Building {len(unmatched_ov):,} unmatched Overture "
        f"POIs in {len(ov_slices)} slice(s) ..."
    )
    for i, sl in enumerate(ov_slices):
        part = _build_unmatched_overture_gdf(
            overture_gdf, sl,
            overture_shared_labels, w,
        )
        p = tmp_dir / f"3_unmatched_overture_{i:02d}.parquet"
        _write_part(part, p)
        part_paths.append(p)
        del part
        gc.collect()

    return part_paths


def _group_idx_by_chunk(
    idx: np.ndarray,
    primary: np.ndarray,
    n_chunks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a subset of indices and a global ``primary`` array,
    return ``(idx_sorted_by_chunk, chunk_offsets)`` where
    ``chunk_offsets[c:c+2]`` slices out the indices for chunk ``c``.
    """
    if len(idx) == 0:
        return (
            np.empty(0, dtype = np.int64),
            np.zeros(n_chunks + 1, dtype = np.int64),
        )
    chunks = primary[idx]
    order = np.argsort(chunks, kind = "stable")
    idx_sorted = idx[order]
    chunks_sorted = chunks[order]
    offsets = np.searchsorted(
        chunks_sorted, np.arange(n_chunks + 1),
    ).astype(np.int64)
    return idx_sorted, offsets


def build_merge_parts_chunked(
    osm_gdf: gpd.GeoDataFrame,
    overture_gdf: gpd.GeoDataFrame,
    matches: pd.DataFrame,
    osm_shared_labels: np.ndarray,
    overture_shared_labels: np.ndarray,
    osm_primary: np.ndarray,
    overture_primary: np.ndarray,
    n_chunks: int,
    overture_confidence_weight: float = 0.7,
) -> list[Path]:
    """
    Build per-spatial-chunk merge parts, writing one parquet per chunk.

    Reuses the KD-bisected chunks produced by the chunked matching
    driver: for each chunk ``c`` we emit matched pairs whose OSM POI
    has ``osm_primary == c`` (the same OSM-anchored emit rule used
    during matching), unmatched OSM POIs with ``osm_primary == c``,
    and unmatched Overture POIs with ``overture_primary == c``.

    Peak memory per chunk is bounded by chunk size × 18-column
    schema, so this stays within a few hundred MB for ~200k-POI
    chunks regardless of total dataset size.

    Args:
        osm_gdf, overture_gdf: Full source frames.
        matches: Post-dedup match DataFrame (osm_idx unique).
        osm_shared_labels, overture_shared_labels: Parallel to source
            frames.
        osm_primary, overture_primary: ``(n,)`` int arrays assigning
            each row to its primary chunk. Produced by
            ``chunking.assign_primary_chunk``.
        n_chunks: Total number of chunks; used for offset arrays.
        overture_confidence_weight: Blend weight ``w`` (see
            ``_build_matched_gdf``).

    Returns:
        List of per-chunk part file paths, in ascending chunk order.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix = "openpois_merge_"))

    w = overture_confidence_weight
    osm_w = 1.0 / (1.0 + w)
    ov_w = w / (1.0 + w)

    # Sort matches by the OSM POI's primary chunk. The OSM-anchored
    # emit rule guarantees osm_idx is unique, so each match belongs to
    # exactly one chunk.
    if len(matches) > 0:
        matched_osm_idx = matches["osm_idx"].to_numpy()
        matched_ov_idx = matches["overture_idx"].to_numpy()
        match_chunk = osm_primary[matched_osm_idx]
        match_order = np.argsort(match_chunk, kind = "stable")
        matches_sorted = matches.iloc[match_order].reset_index(
            drop = True,
        )
        match_chunk_sorted = match_chunk[match_order]
        match_offsets = np.searchsorted(
            match_chunk_sorted, np.arange(n_chunks + 1),
        ).astype(np.int64)
    else:
        matched_osm_idx = np.empty(0, dtype = np.int64)
        matched_ov_idx = np.empty(0, dtype = np.int64)
        matches_sorted = matches
        match_offsets = np.zeros(
            n_chunks + 1, dtype = np.int64,
        )

    unmatched_osm = _unmatched_idx(len(osm_gdf), matched_osm_idx)
    osm_by_chunk, osm_offsets = _group_idx_by_chunk(
        unmatched_osm, osm_primary, n_chunks,
    )
    del unmatched_osm
    gc.collect()

    unmatched_ov = _unmatched_idx(
        len(overture_gdf), matched_ov_idx,
    )
    ov_by_chunk, ov_offsets = _group_idx_by_chunk(
        unmatched_ov, overture_primary, n_chunks,
    )
    del unmatched_ov
    gc.collect()

    part_paths: list[Path] = []
    total_matched = 0
    total_unmatched_osm = 0
    total_unmatched_ov = 0

    print(
        f"  Building {n_chunks} per-chunk merge parts ..."
    )
    for c in range(n_chunks):
        subparts: list[gpd.GeoDataFrame] = []

        m_start, m_end = (
            int(match_offsets[c]), int(match_offsets[c + 1]),
        )
        if m_end > m_start:
            matched_c = matches_sorted.iloc[m_start:m_end]
            subparts.append(
                _build_matched_gdf(
                    osm_gdf, overture_gdf, matched_c,
                    osm_shared_labels, osm_w, ov_w,
                )
            )
            total_matched += m_end - m_start

        o_start, o_end = (
            int(osm_offsets[c]), int(osm_offsets[c + 1]),
        )
        if o_end > o_start:
            osm_idx_c = osm_by_chunk[o_start:o_end]
            subparts.append(
                _build_unmatched_osm_gdf(
                    osm_gdf, osm_idx_c, osm_shared_labels,
                )
            )
            total_unmatched_osm += o_end - o_start

        v_start, v_end = (
            int(ov_offsets[c]), int(ov_offsets[c + 1]),
        )
        if v_end > v_start:
            ov_idx_c = ov_by_chunk[v_start:v_end]
            subparts.append(
                _build_unmatched_overture_gdf(
                    overture_gdf, ov_idx_c,
                    overture_shared_labels, w,
                )
            )
            total_unmatched_ov += v_end - v_start

        if not subparts:
            continue

        # OSM and Overture may be loaded with different CRS
        # representations for the same WGS84 lon/lat system
        # (e.g. "WGS 84" vs "WGS 84 (CRS84)"). Geometries are
        # already in lon/lat on both sides, so force a common CRS
        # without reprojecting.
        target_crs = osm_gdf.crs
        for sp in subparts:
            if sp.crs != target_crs:
                sp.set_crs(
                    target_crs, allow_override = True,
                    inplace = True,
                )

        chunk_gdf = pd.concat(subparts, ignore_index = True)
        p = tmp_dir / f"chunk_{c:04d}.parquet"
        _write_part(
            gpd.GeoDataFrame(chunk_gdf, crs = target_crs), p,
        )
        part_paths.append(p)
        del subparts, chunk_gdf
        gc.collect()

        done = c + 1
        if done % 25 == 0 or done == n_chunks:
            print(
                f"    {done}/{n_chunks} chunks written "
                f"(matched: {total_matched:,}, "
                f"unmatched OSM: {total_unmatched_osm:,}, "
                f"unmatched Overture: {total_unmatched_ov:,})"
            )

    return part_paths


def save_conflated_from_parts(
    part_paths: list[Path],
    output_path: Path,
) -> int:
    """
    Stream temp parquet parts into the final output file.

    Opens each part sequentially, unifies its schema against the
    writer, and appends its row groups. Only one part is held in
    memory at a time, so peak memory is bounded by the largest
    part — independent of the number of parts or the total dataset
    size. Skips Hilbert sorting to stay within memory limits.

    Returns:
        Number of POIs written.
    """
    if not part_paths:
        raise ValueError("No part paths provided.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    # First pass: read schemas and compute a promoted schema so the
    # writer can accept all parts even if some are missing optional
    # columns or have slightly different null-vs-typed fields.
    schemas = [pq.read_schema(p) for p in part_paths]
    unified_schema = pa.unify_schemas(
        schemas, promote_options = "permissive",
    )

    print(
        f"  Streaming {len(part_paths)} parts into "
        f"{output_path} ..."
    )
    n = 0
    writer = pq.ParquetWriter(
        output_path,
        unified_schema,
        compression = "zstd",
    )
    try:
        for i, p in enumerate(part_paths):
            table = pq.read_table(p)
            # Re-cast columns to the unified schema so row groups
            # written sequentially stay compatible.
            table = table.cast(unified_schema, safe = False)
            writer.write_table(table, row_group_size = 50_000)
            n += table.num_rows
            del table
            gc.collect()
            if (i + 1) % 25 == 0 or (i + 1) == len(part_paths):
                print(
                    f"    {i + 1}/{len(part_paths)} parts written "
                    f"({n:,} rows)"
                )
    finally:
        writer.close()

    # Clean up temp files
    for p in part_paths:
        p.unlink()
    part_paths[0].parent.rmdir()

    print("  Done.")
    return n


def save_conflated(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Hilbert-sort and save as GeoParquet (zstd, 50k row groups)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    print("Sorting by Hilbert curve index ...")
    hilbert_order = gdf.hilbert_distance()
    gdf = gdf.iloc[hilbert_order.argsort()].reset_index(drop = True)

    print(f"Saving conflated dataset to {output_path} ...")
    gdf.to_parquet(
        output_path,
        compression = "zstd",
        row_group_size = 50_000,
    )
    print(f"Done. Saved {len(gdf):,} POIs.")
