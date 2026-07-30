#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""
Residential-landuse exclusion for unnamed OSM POIs.

``access=private`` only catches features whose mapper bothered to tag them. The
2026-07 validation round found the OSM-only segment's unverifiable rate is still
dominated by unnamed, imagery-traced features on private property — backyard
pools, HOA gardens, unnamed pitches and playgrounds. None of those can be desk-
or phone-verified, and none of them belong in a public POI dataset.

``landuse=residential`` polygons are a mapper-independent proxy for private
property. This module extracts them from the raw Geofabrik PBFs and drops
unnamed POIs of private-prone types whose centroid falls inside one.

Two properties of the rule matter:

- **Tag-scoped.** In the US ``landuse=residential`` is routinely drawn around a
  whole subdivision or neighbourhood block, so an unscoped rule would also
  remove unnamed convenience stores, places of worship, fuel stations and
  schools that legitimately sit inside those blocks. Only the values in
  ``download.osm.residential_exclusion.scoped_tags`` are eligible.
- **Named features are always kept**, however they are tagged and wherever they
  sit. A name is the single strongest signal that somebody verified the feature
  on the ground.

Scoping is applied to a POI's *primary* tag — the first non-null key in
``download.osm.filter_keys`` order — so a POI is never scoped by an incidental
secondary tag.

This is deliberately not identical to ``assign_osm_shared_label``, which skips
a key whose value the crosswalk does not map and lets a lower-priority key
label the row instead. The divergence only bites when the highest-priority key
carries an *unmapped* value: such a row is scoped on that unmapped tag, finds
it out of scope, and is kept. The exclusion therefore errs towards
under-dropping, which is the right direction for a rule that deletes data.

See the "Exclusion" section of .claude/docs/data-sources.md for the measured
drop rate and the false-exclusion estimate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from openpois.io.osm_snapshot import (
    SnapshotExtract,
    _merge_parquets_streaming,
    download_pbf,
    filter_pbf,
    parse_pbf_to_parquet,
)

# Only these two geometry types can contain a point. Open ways come out of
# POIRecordBuilder.process_way as centroid Points and are useless as a
# containment surface, so they are dropped on load rather than silently
# contributing nothing.
_AREA_GEOM_TYPES = ("Polygon", "MultiPolygon")

# Residential blocks are ordinary closed ways or modest multipolygons, but a few
# are drawn as very large relations. The POI-side default of 1_000 nodes would
# clip exactly the biggest neighbourhoods, so the layer build uses its own,
# much higher, ceiling.
DEFAULT_MAX_AREA_NODES = 50_000


def build_landuse_filter_exprs(landuse_values: Sequence[str]) -> list[str]:
    """Build the osmium ``tags-filter`` expressions for the landuse layer.

    Ways and relations only — a ``landuse`` node carries no area and cannot
    contain anything.

    Args:
        landuse_values: e.g. ``["residential"]``.

    Returns:
        e.g. ``["wr/landuse=residential"]``. Empty if no values are given,
        which callers treat as "the exclusion is switched off".
    """
    values = [str(v) for v in landuse_values if v]
    if not values:
        return []
    return [f"wr/landuse={','.join(sorted(values))}"]


def build_residential_areas(
    raw_pbf: Path,
    filtered_pbf: Path,
    output_path: Path,
    landuse_values: Sequence[str] = ("residential",),
    chunk_dir: Path | None = None,
    max_area_nodes: int | None = DEFAULT_MAX_AREA_NODES,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Extract landuse polygons from a raw PBF into a GeoParquet layer.

    Runs ``osmium tags-filter`` for the landuse values, then reuses the POI
    parser with ``filter_keys = ["landuse"]``. No bespoke parsing code is
    needed: ``POIRecordBuilder.process_way`` already emits ``Polygon`` for
    closed ways and ``process_area`` handles multipolygon relations.

    Args:
        raw_pbf: The unfiltered Geofabrik extract. Must still exist —
            ``scripts/osm_snapshot/download.py`` unlinks these after a
            successful run, so the layer has to be built before that cleanup.
        filtered_pbf: Where to write the landuse-only PBF.
        output_path: Where to write the GeoParquet layer.
        landuse_values: Landuse values to extract.
        chunk_dir: Parent for the parser's ``parse_chunks/`` work directory.
            **Must differ from the POI parse's chunk_dir** — the parser
            short-circuits to a merge when it finds existing chunks, so a
            shared directory would silently merge the wrong elements.
        max_area_nodes: Node ceiling for relation-derived areas.
        overwrite: Rebuild even if ``output_path`` already exists.
        verbose: Print progress.

    Returns:
        ``output_path``.

    Raises:
        FileNotFoundError: If ``raw_pbf`` does not exist.
        ValueError: If ``landuse_values`` is empty.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        if verbose:
            print(f"Landuse layer already exists at {output_path}; skipping.")
        return output_path

    exprs = build_landuse_filter_exprs(landuse_values)
    if not exprs:
        raise ValueError("landuse_values is empty; nothing to extract.")
    raw_pbf = Path(raw_pbf)
    if not raw_pbf.exists():
        raise FileNotFoundError(
            f"{raw_pbf} not found. The landuse layer must be built while the "
            "raw PBF is still on disk, before download.py's cleanup step."
        )

    filter_pbf(
        input_pbf = raw_pbf,
        output_pbf = Path(filtered_pbf),
        osm_keys = ["landuse"],
        overwrite = overwrite,
        tag_filter_exprs = exprs,
    )
    parse_pbf_to_parquet(
        pbf_path = Path(filtered_pbf),
        out_path = output_path,
        filter_keys = ["landuse"],
        extract_keys = ["landuse"],
        source_label = "osm",
        max_area_nodes = max_area_nodes,
        chunk_dir = chunk_dir,
        verbose = verbose,
    )
    return output_path


def build_residential_layer(
    extracts: Sequence[SnapshotExtract],
    output_path: Path,
    landuse_values: Sequence[str] = ("residential",),
    chunk_dir: Path | None = None,
    max_area_nodes: int | None = DEFAULT_MAX_AREA_NODES,
    overwrite: bool = False,
    keep_intermediates: bool = False,
    verbose: bool = True,
) -> Path:
    """Build one landuse layer covering every Geofabrik extract.

    Each extract's raw PBF is downloaded only if absent, so the monthly caller
    (which runs while ``download_osm_snapshot``'s PBFs are still on disk) does
    no network I/O, while a standalone rebuild re-fetches what it needs.

    Args:
        extracts: One entry per Geofabrik extract. ``filtered_pbf_path`` must
            point at the *residential* filtered PBF, not the POI one.
        output_path: Where to write the merged GeoParquet layer.
        landuse_values: Landuse values to extract.
        chunk_dir: Parent for per-extract ``parse_chunks/`` work directories.
            Defaults to ``output_path.parent``. A per-extract subdirectory is
            used so an interrupted run cannot merge one extract's chunks into
            the next extract's output.
        max_area_nodes: Node ceiling for relation-derived areas.
        overwrite: Rebuild even if ``output_path`` exists.
        keep_intermediates: Keep the filtered PBFs and per-extract parquets.
        verbose: Print progress.

    Returns:
        ``output_path``.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        if verbose:
            print(f"Landuse layer already exists at {output_path}; skipping.")
        return output_path
    if not build_landuse_filter_exprs(landuse_values):
        raise ValueError("landuse_values is empty; nothing to extract.")

    base_chunk_dir = Path(chunk_dir) if chunk_dir else output_path.parent
    parts: list[Path] = []
    for spec in extracts:
        part = output_path.with_name(
            f"{output_path.stem}_{spec.name}{output_path.suffix}"
        )
        if verbose:
            print(f"\n=== landuse layer: {spec.name} ===")
        # overwrite=False: only fetch what is missing. The monthly caller runs
        # before download.py unlinks the raw PBFs, so this is a no-op there.
        download_pbf(spec.url, Path(spec.raw_pbf_path), overwrite = False)
        build_residential_areas(
            raw_pbf = Path(spec.raw_pbf_path),
            filtered_pbf = Path(spec.filtered_pbf_path),
            output_path = part,
            landuse_values = landuse_values,
            chunk_dir = base_chunk_dir / f"residential_{spec.name}",
            max_area_nodes = max_area_nodes,
            overwrite = overwrite,
            verbose = verbose,
        )
        parts.append(part)

    output_path.parent.mkdir(parents = True, exist_ok = True)
    _merge_parquets_streaming(parts, output_path)
    if verbose:
        n = pq.read_metadata(output_path).num_rows
        print(f"\nMerged {len(parts)} extracts -> {n:,} polygons at {output_path}")

    if not keep_intermediates:
        for spec in extracts:
            Path(spec.filtered_pbf_path).unlink(missing_ok = True)
        for part in parts:
            part.unlink(missing_ok = True)
    return output_path


def load_residential_areas(
    path: Path,
    landuse_values: Sequence[str] = ("residential",),
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Load a landuse layer as containment polygons, in EPSG:4326.

    Keeps only the requested landuse values and only polygonal geometries, and
    returns a minimal frame — the caller only ever needs the geometry column,
    and a national layer is large enough that dropping the rest matters.

    Args:
        path: GeoParquet written by :func:`build_residential_areas`.
        landuse_values: Values to retain.
        verbose: Print the retained polygon count.

    Returns:
        A ``GeoDataFrame`` with a single ``geometry`` column.
    """
    gdf = gpd.read_parquet(path, columns = ["landuse", "geometry"])
    wanted = {str(v) for v in landuse_values if v}
    if wanted and "landuse" in gdf.columns:
        gdf = gdf.loc[gdf["landuse"].isin(wanted)]
    gdf = gdf.loc[gdf.geometry.geom_type.isin(_AREA_GEOM_TYPES)]
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame(geometry = gdf.geometry.to_numpy(), crs = "EPSG:4326")
    if verbose:
        print(f"Loaded {len(gdf):,} landuse polygons from {path}")
    return gdf


def primary_tag(
    df: pd.DataFrame, filter_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve each row's primary ``(key, value)``.

    The primary tag is the first non-null, non-empty key in ``filter_keys``
    order, so a POI is scoped by its dominant tag rather than by an incidental
    secondary one.

    Note this is *not* the same walk as ``assign_osm_shared_label``, which
    skips a key whose value the crosswalk does not map. See the module
    docstring for why the difference is safe.

    Missing columns are skipped rather than raising: a caller filtering a
    partially-loaded frame gets a correct answer for the keys it did load.

    Args:
        df: Frame with zero or more of the ``filter_keys`` as columns.
        filter_keys: Keys in priority order.

    Returns:
        ``(keys, values)`` object arrays of length ``len(df)``. Rows with no
        tag among ``filter_keys`` get ``None`` in both.
    """
    n = len(df)
    keys = np.full(n, None, dtype = object)
    values = np.full(n, None, dtype = object)
    unset = np.ones(n, dtype = bool)
    for key in filter_keys:
        if not unset.any():
            break
        if key not in df.columns:
            continue
        col = df[key]
        # Nullable dtypes make `col != ""` return pd.NA rather than False, and
        # a nullable-boolean .to_numpy() yields object dtype; pin both.
        present = (col.notna() & (col.astype("string") != "")).to_numpy(
            dtype = bool, na_value = False,
        )
        take = unset & present
        if not take.any():
            continue
        keys[take] = key
        values[take] = col.to_numpy(dtype = object)[take]
        unset &= ~take
    return keys, values


def scope_mask(
    df: pd.DataFrame,
    scoped_tags: Mapping[str, Sequence[str]] | None,
    filter_keys: Sequence[str],
) -> np.ndarray:
    """Rows eligible for the residential test: unnamed AND primary tag in scope.

    This is the cheap half of the predicate and is evaluated first so that only
    eligible rows ever have their geometry decoded.

    Args:
        df: Frame with a ``name`` column and the ``filter_keys`` tag columns.
        scoped_tags: ``{osm_key: [osm_value, ...]}``. ``None`` or empty means
            nothing is eligible — the exclusion is off.
        filter_keys: Keys in priority order.

    Returns:
        Null-free boolean array of length ``len(df)``.
    """
    n = len(df)
    if not scoped_tags:
        return np.zeros(n, dtype = bool)
    if "name" not in df.columns:
        return np.zeros(n, dtype = bool)

    name = df["name"]
    unnamed = (name.isna() | (name.astype("string") == "")).to_numpy()
    unnamed = np.nan_to_num(unnamed, nan = False).astype(bool)
    if not unnamed.any():
        return np.zeros(n, dtype = bool)

    keys, values = primary_tag(df, filter_keys)
    in_scope = np.zeros(n, dtype = bool)
    for key, vals in scoped_tags.items():
        wanted = {str(v) for v in vals if v}
        if not wanted:
            continue
        in_scope |= (keys == key) & np.fromiter(
            (v in wanted for v in values), dtype = bool, count = n,
        )
    return unnamed & in_scope


def build_index(
    residential: gpd.GeoDataFrame | shapely.STRtree,
) -> shapely.STRtree:
    """Build (or pass through) the containment index.

    Build this **once** and reuse it: the national layer runs to millions of
    polygons, and rebuilding the tree per batch dominates the runtime of a
    streaming filter.

    Args:
        residential: Polygons in EPSG:4326, or an already-built tree.

    Returns:
        A ``shapely.STRtree`` over the polygons.
    """
    if isinstance(residential, shapely.STRtree):
        return residential
    return shapely.STRtree(np.asarray(residential.geometry.to_numpy()))


def points_within(
    geometry: np.ndarray,
    residential: gpd.GeoDataFrame | shapely.STRtree,
) -> np.ndarray:
    """Test whether each geometry's representative point falls in a polygon.

    Snapshot geometry is a mix of ``Point`` (nodes), ``Polygon`` (closed ways)
    and ``MultiPolygon`` (relations), so containment is evaluated on a
    representative point rather than on the footprint. ``point_on_surface`` is
    used over ``centroid`` because it is guaranteed to lie inside a concave or
    holed footprint — a C-shaped pitch's centroid can sit in the notch.

    Args:
        geometry: Shapely geometries.
        residential: Containment polygons in EPSG:4326, or a prebuilt tree
            from :func:`build_index`. Pass a tree when calling in a loop.

    Returns:
        Null-free boolean array of length ``len(geometry)``.
    """
    n = len(geometry)
    inside = np.zeros(n, dtype = bool)
    if n == 0:
        return inside
    tree = build_index(residential)
    if len(tree) == 0:
        return inside

    valid = shapely.is_valid_input(geometry) & ~shapely.is_missing(geometry)
    if not valid.any():
        return inside
    reps = shapely.point_on_surface(geometry[valid])

    # query() returns (input_position, tree_position) pairs; a point inside
    # two abutting subdivisions appears twice, so take the unique inputs
    # rather than assuming a 1:1 join.
    hits = tree.query(reps, predicate = "within")[0]
    matched = np.zeros(int(valid.sum()), dtype = bool)
    matched[np.unique(hits)] = True
    inside[valid] = matched
    return inside


def residential_drop_mask(
    df: pd.DataFrame,
    residential: gpd.GeoDataFrame | shapely.STRtree,
    scoped_tags: Mapping[str, Sequence[str]] | None,
    filter_keys: Sequence[str],
    geometry: np.ndarray | None = None,
) -> np.ndarray:
    """Rows to DROP: unnamed, in scope, and inside a residential polygon.

    All three conditions must hold, so a named feature is never dropped, an
    out-of-scope tag is never dropped, and a POI outside every residential
    polygon is never dropped.

    Args:
        df: Frame with ``name`` and the ``filter_keys`` tag columns.
        residential: Containment polygons in EPSG:4326, or a prebuilt tree
            from :func:`build_index`. Pass a tree when calling in a loop.
        scoped_tags: ``{osm_key: [osm_value, ...]}``; empty disables the rule.
        filter_keys: Keys in priority order.
        geometry: Shapely geometries aligned to ``df``. Defaults to
            ``df["geometry"]``.

    Returns:
        Null-free boolean array of length ``len(df)``.
    """
    eligible = scope_mask(df, scoped_tags, filter_keys)
    drop = np.zeros(len(df), dtype = bool)
    if not eligible.any():
        return drop
    if geometry is None:
        geometry = df["geometry"].to_numpy()
    drop[eligible] = points_within(np.asarray(geometry)[eligible], residential)
    return drop


def _accumulate_drops(
    dropped: pd.DataFrame,
    filter_keys: Sequence[str],
    has_conf: bool,
    counts: dict[tuple[str, str], int],
    conf_sums: dict[tuple[str, str], float],
) -> None:
    """Fold one batch's dropped rows into the running per-tag tallies."""
    keys, values = primary_tag(dropped, filter_keys)
    confs = (
        dropped["conf_mean"].to_numpy(dtype = float, na_value = np.nan)
        if has_conf else np.full(len(dropped), np.nan)
    )
    for key, value, conf in zip(keys, values, confs):
        counts[(key, value)] = counts.get((key, value), 0) + 1
        if not np.isnan(conf):
            conf_sums[(key, value)] = conf_sums.get((key, value), 0.0) + conf


def _build_drop_report(
    counts: Mapping[tuple[str, str], int],
    conf_sums: Mapping[tuple[str, str], float],
    has_conf: bool,
) -> pd.DataFrame:
    """Per-tag drop counts, most-dropped first, with mean model confidence."""
    columns = ["osm_key", "osm_value", "n_dropped"]
    if has_conf:
        columns.append("conf_mean")
    rows = []
    for (key, value), n in sorted(counts.items(), key = lambda kv: -kv[1]):
        row = {"osm_key": key, "osm_value": value, "n_dropped": n}
        if has_conf:
            row["conf_mean"] = round(conf_sums.get((key, value), float("nan")) / n, 3)
        rows.append(row)
    return pd.DataFrame(rows, columns = columns)


def filter_parquet_by_residential(
    input_path: Path,
    output_path: Path | None,
    residential: gpd.GeoDataFrame | shapely.STRtree,
    scoped_tags: Mapping[str, Sequence[str]] | None,
    filter_keys: Sequence[str],
    batch_size: int = 250_000,
    verbose: bool = True,
) -> tuple[int, pd.DataFrame]:
    """Stream a POI parquet through the residential exclusion.

    Row groups stream through pyarrow so peak memory stays well under the file
    size; only rows passing the cheap unnamed+scope predicate have their WKB
    decoded. The input schema and metadata (including the GeoParquet ``geo``
    key) are preserved.

    Args:
        input_path: POI GeoParquet to read.
        output_path: Where to write the filtered copy. Must differ from
            ``input_path`` — callers archive the original first. Pass ``None``
            for a dry run that reports what would drop without writing.
        residential: Containment polygons in EPSG:4326, or a prebuilt tree.
            The tree is built once here and reused across every batch.
        scoped_tags: ``{osm_key: [osm_value, ...]}``; empty makes this a copy.
        filter_keys: Keys in priority order.
        batch_size: Rows per pyarrow batch.
        verbose: Print per-batch progress.

    Returns:
        ``(rows_kept, dropped_by_tag)``. ``dropped_by_tag`` has columns
        ``osm_key, osm_value, n_dropped`` plus ``conf_mean`` when the input
        carries a ``conf_mean`` column, so a drop can be read against how
        confident the model was about those rows.

    Raises:
        ValueError: If ``input_path`` and ``output_path`` are the same file.
    """
    input_path = Path(input_path)
    dry_run = output_path is None
    if not dry_run:
        output_path = Path(output_path)
        if input_path.resolve() == output_path.resolve():
            raise ValueError("input_path and output_path must differ")

    # Built once: an STRtree over a national landuse layer is millions of
    # polygons, and rebuilding it per batch would dominate the runtime.
    tree = build_index(residential)

    source = pq.ParquetFile(input_path)
    schema = source.schema_arrow
    needed = [k for k in filter_keys if k in schema.names]
    has_conf = "conf_mean" in schema.names
    cols = ["name", "geometry"] + needed + (["conf_mean"] if has_conf else [])
    n_in = source.metadata.num_rows
    n_out = 0
    counts: dict[tuple[str, str], int] = {}
    conf_sums: dict[tuple[str, str], float] = {}

    writer = (
        None if dry_run
        else pq.ParquetWriter(output_path, schema, compression = "snappy")
    )
    try:
        for i, batch in enumerate(source.iter_batches(batch_size = batch_size)):
            table = pa.Table.from_batches([batch], schema = schema)
            df = table.select(cols).to_pandas()
            geom = shapely.from_wkb(df["geometry"].to_numpy())
            drop = residential_drop_mask(
                df, tree, scoped_tags, filter_keys, geometry = geom,
            )
            if drop.any():
                _accumulate_drops(
                    df.loc[drop], filter_keys, has_conf, counts, conf_sums,
                )
            keep = pa.array(~drop)
            if keep.null_count:
                raise ValueError(
                    f"keep mask has {keep.null_count} nulls; filter() would "
                    "drop those rows silently."
                )
            if dry_run:
                n_out += int((~drop).sum())
            else:
                kept = table.filter(keep)
                n_out += kept.num_rows
                writer.write_table(kept)
            if verbose:
                print(f"  batch {i}: {n_out:,} kept / {n_in:,} read", flush = True)
    except Exception:
        if writer is not None:
            writer.close()
            output_path.unlink(missing_ok = True)
        raise
    finally:
        if writer is not None and writer.is_open:
            writer.close()

    return n_out, _build_drop_report(counts, conf_sums, has_conf)
