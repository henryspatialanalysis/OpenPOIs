#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Reconstruct a "ghost OSM" POI dataset from OSM history.

A ghost is a *previous* state of an OSM node that we believe no longer
reflects ground truth — for example, an element whose POI tag was
deleted, whose POI tag was moved into a lifecycle namespace
(``disused:*``, ``was:*``, ``demolished:*``, ``abandoned:*``,
``removed:*``, ``razed:*``), or whose name was substantially rewritten.
Each detected event produces one ghost row carrying the element's
prior-version geometry, name, brand, and POI tag dict.

Used downstream by ``openpois.conflation.change_detection`` to penalize
Overture POIs that co-locate with a ghost — the OSM editor's removal /
rename signal is Bayesian evidence that the Overture record is stale.

Only **nodes** are emitted in v1: the per-version Parquet captures node
``lat``/``lon`` as pseudo-tags so prior-state geometry is fully
recoverable; way and relation centroids would require replaying the
PBF.

Event detection rule, per version of one element (priority order):

1. ``hard_delete`` — the element's ``visible`` pseudo-tag changed
   from ``true`` to ``false`` (full deletion of the OSM element).
   Strongest signal; fires regardless of whether the prior state
   carried a name.
2. ``lifecycle_prefix_added`` — a ``disused:*`` / ``was:*`` /
   ``demolished:*`` / ``abandoned:*`` / ``removed:*`` / ``razed:*``
   key appeared (Added or Changed). Only fires when the prior state
   was un-named (named lifecycle changes are usually retagging
   cleanup, not real removals).
3. ``primary_tag_deleted`` — a POI tag key (the configured
   ``filter_keys``) was Deleted this version. Same no-prior-name
   gate as ``lifecycle_prefix_added``.
4. ``substantial_rename`` — ``name`` changed and the
   ``rapidfuzz.fuzz.token_set_ratio`` between the prior and new name
   falls below ``name_change_similarity_threshold``, and neither
   name is a token-level subset/superset of the other.

The four signals are checked in priority order; at most one ghost is
emitted per (element, version).
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from shapely.geometry import Point

from openpois.conflation.taxonomy import (
    assign_osm_shared_label,
    load_match_radii,
    load_osm_crosswalk,
)
from openpois.osm.lifecycle import LIFECYCLE_PREFIXES, is_lifecycle_key


# Keys we track in the rolling tag dict per element. POI keys come from
# the caller's ``filter_keys`` (matches ``download.osm.filter_keys`` in
# config.yaml). Geometry pseudo-tags, identity tags, and the
# ``visible`` pseudo-tag are appended.
_GEOMETRY_KEYS = ("lat", "lon")
_IDENTITY_KEYS = ("name", "brand")
_LIFECYCLE_PSEUDO_KEYS = ("visible",)


_NAME_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokenize_name(name: str | None) -> set[str]:
    """Lowercase, split on non-alphanumeric, drop empty tokens."""
    if not name:
        return set()
    return {t for t in _NAME_TOKEN_SPLIT.split(name.lower()) if t}


def _is_token_subset_or_superset(a: str, b: str) -> bool:
    """True when one name's tokens are fully contained in the other's.

    Used to suppress ``substantial_rename`` ghosts on cases like
    "Walgreens" ↔ "Walgreens Pharmacy" or "CVS" ↔ "CVS/pharmacy" —
    both are obviously the same entity even though
    rapidfuzz.token_set_ratio can dip below 50 for very short names.
    """
    ta = _tokenize_name(a)
    tb = _tokenize_name(b)
    if not ta or not tb:
        return False
    return ta.issubset(tb) or tb.issubset(ta)


def _load_filtered_changes(
    changes_path: Path,
    poi_keys: tuple[str, ...],
    duckdb_memory_limit: str = "6GB",
) -> pd.DataFrame:
    """Load node-only osm_changes filtered to keys we need to track.

    Sorted by (id, version, key) so the per-element walk can rely on
    contiguous version slices.
    """
    poi_list = ",".join(f"'{k}'" for k in poi_keys)
    identity_list = ",".join(f"'{k}'" for k in _IDENTITY_KEYS)
    geometry_list = ",".join(f"'{k}'" for k in _GEOMETRY_KEYS)
    pseudo_list = ",".join(f"'{k}'" for k in _LIFECYCLE_PSEUDO_KEYS)
    lifecycle_clauses = " OR ".join(
        [f"key LIKE '{p}%'" for p in LIFECYCLE_PREFIXES]
    )
    sql = f"""
        SELECT id, version, key, value, change
        FROM read_parquet('{changes_path}')
        WHERE type = 'node'
          AND (
              key IN ({poi_list})
              OR key IN ({identity_list})
              OR key IN ({geometry_list})
              OR key IN ({pseudo_list})
              OR {lifecycle_clauses}
          )
        ORDER BY id, version, key
    """
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
        return con.execute(sql).fetch_df()
    finally:
        con.close()


def _load_version_timestamps(
    versions_path: Path,
    duckdb_memory_limit: str = "4GB",
) -> pd.DataFrame:
    """Load (id, version, timestamp) for nodes only."""
    sql = f"""
        SELECT id, version, timestamp
        FROM read_parquet('{versions_path}')
        WHERE type = 'node'
    """
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
        return con.execute(sql).fetch_df()
    finally:
        con.close()


def _scan_all_changes(
    changes: pd.DataFrame,
    poi_keys: frozenset[str],
    name_threshold: float,
) -> list[dict]:
    """
    Flat sequential scan over the full sorted changes DataFrame, emitting
    one ghost per detected (element, version) event.

    A single scan replaces the per-element ``groupby`` + Python loop in
    the obvious nested implementation. Element and version boundaries
    are detected as the iterator advances; rolling per-element tag
    state plus per-version event flags are kept in local variables.

    Args:
        changes: Full changes DataFrame, columns ``id, version, key,
            value, change``, sorted by ``(id, version, key)``.
        poi_keys: POI tag keys (e.g. amenity, shop, leisure, …).
        name_threshold: Below this rapidfuzz.token_set_ratio (0-100),
            a Changed ``name`` triggers a ``substantial_rename``.

    Returns:
        List of ghost dicts (one per detected event).
    """
    ids = changes["id"].to_numpy()
    versions = changes["version"].to_numpy()
    keys = changes["key"].to_numpy()
    values = changes["value"].to_numpy()
    changes_arr = changes["change"].to_numpy()
    n = len(changes)

    ghosts: list[dict] = []

    state: dict[str, str] = {}
    prior_snapshot: dict[str, str] = {}
    prior_name_snapshot: str | None = None

    cur_id = ids[0] if n else None
    cur_ver = versions[0] if n else None

    # Per-version event accumulators (reset on each version boundary).
    deleted_poi_keys: list[str] = []
    added_lifecycle_keys: list[str] = []
    new_name: str | None = None
    name_changed = False
    visibility_deleted = False  # set when visible: true → false

    def _flush() -> None:
        """Emit a ghost for the just-finished (cur_id, cur_ver) if any
        event triggered.

        Detection rules:

        - ``hard_delete`` fires whenever ``visible`` transitions to
          ``false`` — the OSM element was deleted outright. Strongest
          signal; fires regardless of name presence.
        - ``lifecycle_prefix_added`` / ``primary_tag_deleted`` are
          emitted only when the prior state had **no name**. A type
          change on a named POI is noisy (often just retagging /
          cleanup); for un-named POIs (playgrounds, ATMs, benches)
          the type change is the only available "this is gone"
          signal, so we keep those.
        - ``substantial_rename`` requires both names to be present,
          token_set_ratio below ``name_threshold``, AND that neither
          name is a token-level subset/superset of the other —
          guards "Walgreens" ↔ "Walgreens Pharmacy" type cases.
        """
        event_type: str | None = None
        has_prior_name = bool(prior_name_snapshot)

        if visibility_deleted:
            event_type = "hard_delete"
        elif added_lifecycle_keys:
            if not has_prior_name:
                event_type = "lifecycle_prefix_added"
        elif deleted_poi_keys:
            if not has_prior_name:
                event_type = "primary_tag_deleted"
        elif (
            name_changed
            and prior_name_snapshot
            and new_name
        ):
            sim = fuzz.token_set_ratio(prior_name_snapshot, new_name)
            if sim < name_threshold and not _is_token_subset_or_superset(
                prior_name_snapshot, new_name,
            ):
                event_type = "substantial_rename"

        if event_type is None:
            return

        lat = prior_snapshot.get("lat")
        lon = prior_snapshot.get("lon")
        if lat is None or lon is None:
            return
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            return

        ghosts.append(
            {
                "osm_id": int(cur_id),
                "osm_type": "node",
                "osm_version_after": int(cur_ver),
                "event_type": event_type,
                "prior_name": prior_name_snapshot,
                "prior_brand": prior_snapshot.get("brand"),
                "ghost_lat": lat_f,
                "ghost_lon": lon_f,
                **{k: prior_snapshot.get(k) for k in poi_keys},
            }
        )

    for i in range(n):
        rid = ids[i]
        rver = versions[i]

        if rid != cur_id or rver != cur_ver:
            # Boundary: flush the just-finished version and reset.
            _flush()

            if rid != cur_id:
                cur_id = rid
                state = {}

            cur_ver = rver
            prior_snapshot = state.copy()
            prior_name_snapshot = prior_snapshot.get("name")
            deleted_poi_keys = []
            added_lifecycle_keys = []
            new_name = prior_name_snapshot
            name_changed = False
            visibility_deleted = False

        key = keys[i]
        value = values[i]
        change = changes_arr[i]

        if change == "Added" or change == "Changed":
            state[key] = value
            if is_lifecycle_key(key):
                added_lifecycle_keys.append(key)
            if key == "name":
                new_name = value
                if change == "Changed":
                    name_changed = True
            if key == "visible" and value == "false":
                # ``visible: true → false`` (a Changed row) is the
                # element-deletion event. Tracking ``visible: false``
                # as Added is a defensive no-op — the parser only
                # writes Added on the first version of an element,
                # which would never be visible=false.
                visibility_deleted = True
        elif change == "Deleted":
            state.pop(key, None)
            if key in poi_keys:
                deleted_poi_keys.append(key)
            if key == "name":
                new_name = None

    # Final flush after the loop.
    if n:
        _flush()

    return ghosts


def build_ghosts(
    versions_path: Path,
    changes_path: Path,
    poi_keys: list[str] | tuple[str, ...],
    name_change_similarity_threshold: float = 50.0,
    duckdb_memory_limit: str = "6GB",
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Reconstruct ghost POIs from OSM history.

    Args:
        versions_path: Path to ``osm_versions.parquet`` (one row per
            element-version with timestamp).
        changes_path: Path to ``osm_changes.parquet`` (one row per tag
            diff with key/value/change).
        poi_keys: POI tag keys to track and trigger
            ``primary_tag_deleted`` events on.  Should match
            ``download.osm.filter_keys`` in config.yaml.
        name_change_similarity_threshold: Below this
            ``rapidfuzz.fuzz.token_set_ratio`` (0–100), a Changed
            ``name`` row produces a ``substantial_rename`` ghost.
        duckdb_memory_limit: Per-connection DuckDB memory limit for the
            two initial scans.
        verbose: Print progress.

    Returns:
        GeoDataFrame with one row per ghost, columns:
            ghost_id, osm_id, osm_type, osm_version_after, event_type,
            event_timestamp, prior_name, prior_brand, shared_label,
            geometry, plus one column per ``poi_keys`` entry carrying
            the prior tag value (used by
            ``assign_osm_shared_label``).
    """
    poi_keys_t = tuple(poi_keys)
    poi_keys_set: frozenset[str] = frozenset(poi_keys_t)

    if verbose:
        print(f"Loading filtered changes from {changes_path} ...")
    changes = _load_filtered_changes(
        changes_path, poi_keys_t,
        duckdb_memory_limit = duckdb_memory_limit,
    )
    if verbose:
        print(
            f"  {len(changes):,} change rows across "
            f"{changes['id'].nunique():,} nodes"
        )

    if verbose:
        print(f"Loading version timestamps from {versions_path} ...")
    versions = _load_version_timestamps(
        versions_path,
        duckdb_memory_limit = duckdb_memory_limit,
    )

    if verbose:
        print(
            "Scanning changes for ghost events (flat single-pass) ..."
        )
    ghost_rows = _scan_all_changes(
        changes,
        poi_keys_set,
        name_change_similarity_threshold,
    )

    if verbose:
        print(f"  Total ghosts emitted: {len(ghost_rows):,}")

    if not ghost_rows:
        # Return an empty GeoDataFrame with the expected schema.
        empty_cols = {
            "ghost_id": pd.Series(dtype = object),
            "osm_id": pd.Series(dtype = "int64"),
            "osm_type": pd.Series(dtype = object),
            "osm_version_after": pd.Series(dtype = "int64"),
            "event_type": pd.Series(dtype = object),
            "event_timestamp": pd.Series(dtype = "datetime64[ns, UTC]"),
            "prior_name": pd.Series(dtype = object),
            "prior_brand": pd.Series(dtype = object),
            "shared_label": pd.Series(dtype = object),
        }
        for k in poi_keys_t:
            empty_cols[k] = pd.Series(dtype = object)
        return gpd.GeoDataFrame(
            empty_cols,
            geometry = gpd.GeoSeries([], crs = "EPSG:4326"),
            crs = "EPSG:4326",
        )

    df = pd.DataFrame(ghost_rows)
    df["ghost_id"] = (
        df["osm_type"].astype(str)
        + "/"
        + df["osm_id"].astype(str)
        + "/v"
        + df["osm_version_after"].astype(str)
        + ":"
        + df["event_type"].astype(str)
    )

    # Join event timestamps.
    df = df.merge(
        versions.rename(columns = {
            "id": "osm_id",
            "version": "osm_version_after",
            "timestamp": "event_timestamp",
        }),
        on = ["osm_id", "osm_version_after"],
        how = "left",
    )
    df["event_timestamp"] = pd.to_datetime(
        df["event_timestamp"], utc = True, errors = "coerce",
    )

    if verbose:
        print("Assigning shared_label to ghosts ...")
    osm_crosswalk = load_osm_crosswalk()
    match_radii = load_match_radii()
    shared_labels, _ = assign_osm_shared_label(
        df, osm_crosswalk, match_radii, list(poi_keys_t),
    )
    df["shared_label"] = shared_labels

    geometry = gpd.points_from_xy(
        df["ghost_lon"].to_numpy(),
        df["ghost_lat"].to_numpy(),
        crs = "EPSG:4326",
    )

    column_order = [
        "ghost_id", "osm_id", "osm_type",
        "osm_version_after", "event_type", "event_timestamp",
        "prior_name", "prior_brand", "shared_label",
        *poi_keys_t,
    ]
    return gpd.GeoDataFrame(
        df[column_order],
        geometry = geometry,
        crs = "EPSG:4326",
    )
