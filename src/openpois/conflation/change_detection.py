#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Penalize unmatched Overture POIs that shadow-match a "ghost" OSM POI.

A ghost is a previous state of an OSM node we believe no longer
reflects ground truth (built by ``openpois.conflation.ghost_osm``).
When an unmatched Overture POI from the baseline conflation lies near
a ghost with a compatible taxonomy and name, we treat OSM's removal /
rename edit as Bayesian evidence that the Overture record is stale,
and multiply its ``conf_mean`` by the per-``shared_label`` delta from
the fitted turnover model (``δ_group``). ``δ`` is the model's
estimate of P(user-edit is spurious) for that category, so the new
confidence is ≈ ``old_conf × δ`` = P(POI still exists | OSM edit).

Operates as a post-processing pass on the baseline ``conflated.parquet``
so the no-change-detection output is bit-identical and only the
penalized output picks up the new audit columns.
"""
from __future__ import annotations

import gc
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from sklearn.neighbors import BallTree

from openpois.conflation.ghost_osm import _is_token_subset_or_superset
from openpois.conflation.match import (
    compute_match_scores,
    find_spatial_candidates,
    select_best_matches,
)


SHADOW_AUDIT_COLUMNS = [
    "shadow_matched",
    "shadow_ghost_id",
    "shadow_event_type",
    "shadow_event_timestamp",
    "shadow_score",
    "shadow_distance_m",
    "original_conf_mean",
]


def load_delta_lookup(
    fitted_params_path: Path,
    default_delta: float,
) -> tuple[dict[str, float], float]:
    """Load per-shared_label delta from fitted_params.csv.

    Returns:
        (lookup, default) where ``lookup[shared_label]`` is the
        fitted ``delta`` posterior mean for that group, and
        ``default`` is the configured fallback for groups absent
        from the fit.
    """
    df = pd.read_csv(fitted_params_path)
    delta_rows = df[df["param_name"] == "delta"]
    lookup = {
        str(row["group_name"]): float(row["mean"])
        for _, row in delta_rows.iterrows()
        if pd.notna(row["group_name"])
    }
    return lookup, float(default_delta)


def _to_str_array(s: pd.Series) -> np.ndarray:
    """Object array of strings, NaN/None as empty string."""
    return s.fillna("").astype(str).to_numpy()


def find_shadow_matches(
    unmatched_overture: gpd.GeoDataFrame,
    ghosts: gpd.GeoDataFrame,
    *,
    min_match_score: float,
    max_radius_m: float,
    default_radius_m: float,
    distance_weight: float,
    name_weight: float,
    type_weight: float,
    identifier_weight: float,
    min_prior_name_match_score: float = 0.0,
) -> pd.DataFrame:
    """Run a single-pass match between Overture rows and ghost rows.

    Reuses the BallTree spatial candidate search + composite-score
    selection from ``match.py``. Returned columns:
    ``osm_idx`` (== ghost row index), ``overture_idx`` (== Overture
    row index), ``composite_score``, ``distance_m``.

    L0 bitmask scoring is intentionally skipped (passing all-zero
    bit arrays) so type_score is binary on exact ``shared_label``
    equality — the change-detection penalty is conservative and
    should only fire when taxonomy genuinely matches.

    ``min_prior_name_match_score`` is an additional hard gate on the
    Overture-name vs ghost-prior-name token_set_ratio (0–100). When
    > 0, candidate pairs below that threshold are dropped *before*
    the composite-score-based selection runs. Subset/superset pairs
    pass regardless. Set this to require a strong direct name match
    (e.g. 70) and you'll trade most of the recall for much higher
    precision. Default 0 disables the gate.
    """
    if len(unmatched_overture) == 0 or len(ghosts) == 0:
        return pd.DataFrame(
            columns = [
                "osm_idx", "overture_idx",
                "composite_score", "distance_m",
            ]
        )

    # Per-ghost match radius via shared_label; falls back to the
    # default for ghosts with no recognized label.
    from openpois.conflation.taxonomy import load_match_radii

    radii_df = load_match_radii()
    radii_dict = {
        row["shared_label"]: float(row["match_radius_m"])
        for _, row in radii_df.iterrows()
    }
    ghost_labels = _to_str_array(ghosts["shared_label"])
    ghost_radii = np.array(
        [radii_dict.get(lb, default_radius_m) for lb in ghost_labels],
        dtype = np.float64,
    )

    candidates = find_spatial_candidates(
        osm_geom = ghosts.geometry.values,
        overture_geom = unmatched_overture.geometry.values,
        osm_radii_m = ghost_radii,
        max_radius_m = max_radius_m,
    )
    if candidates.empty:
        return pd.DataFrame(
            columns = [
                "osm_idx", "overture_idx",
                "composite_score", "distance_m",
            ]
        )

    ov_names = _to_str_array(unmatched_overture["name"])
    ov_brands = _to_str_array(unmatched_overture["brand"])
    ghost_names = _to_str_array(ghosts["prior_name"])
    ghost_brands = _to_str_array(ghosts["prior_brand"])

    ov_labels = _to_str_array(unmatched_overture["shared_label"])

    # Optional pre-gate: drop candidate pairs whose Overture-name vs
    # ghost-prior-name token_set_ratio is below the configured floor.
    # Subset/superset pairs pass regardless (a short subset like
    # "CVS" vs "CVS Pharmacy" can dip below threshold on token-set
    # ratio but is obviously the same business). This is the "tighten
    # matcher" alternative — when set high (e.g. 70) it trades most
    # recall for high precision and removes the need for downstream
    # suppression rules.
    if min_prior_name_match_score > 0 and not candidates.empty:
        cand_osm_idx = candidates["osm_idx"].to_numpy()
        cand_ov_idx = candidates["overture_idx"].to_numpy()
        keep = np.zeros(len(candidates), dtype = bool)
        for i in range(len(candidates)):
            gname = ghost_names[cand_osm_idx[i]]
            oname = ov_names[cand_ov_idx[i]]
            if not gname or not oname:
                continue
            if _is_token_subset_or_superset(gname, oname):
                keep[i] = True
                continue
            sim = fuzz.token_set_ratio(gname, oname)
            if sim >= min_prior_name_match_score:
                keep[i] = True
        candidates = candidates.loc[keep].reset_index(drop = True)
        if candidates.empty:
            return pd.DataFrame(
                columns = [
                    "osm_idx", "overture_idx",
                    "composite_score", "distance_m",
                ]
            )

    # All-zero L0 bits → only exact shared_label match scores 1.0
    # (broad-group bitmask overlap collapses to 0 because all bits
    # are 0). Keeps the secondary pass conservative.
    n_ghost = len(ghosts)
    n_ov = len(unmatched_overture)
    zero_bits_ghost = np.zeros(n_ghost, dtype = np.uint16)
    zero_bits_ov = np.zeros(n_ov, dtype = np.uint16)

    scored = compute_match_scores(
        candidates = candidates,
        osm_names = ghost_names,
        osm_brands = ghost_brands,
        overture_names = ov_names,
        overture_brands = ov_brands,
        osm_shared_labels = ghost_labels,
        overture_shared_labels = ov_labels,
        osm_radii_m = ghost_radii,
        osm_l0_bits = zero_bits_ghost,
        overture_l0_bits = zero_bits_ov,
        distance_weight = distance_weight,
        name_weight = name_weight,
        type_weight = type_weight,
        identifier_weight = identifier_weight,
    )

    matches = select_best_matches(scored, min_score = min_match_score)
    if matches.empty:
        return matches[
            ["osm_idx", "overture_idx", "composite_score", "distance_m"]
        ].reset_index(drop = True)

    # Second-stage subset/superset filter: drop matches where the
    # Overture name is just a token-level subset/superset of the
    # ghost's prior name (e.g. "Walgreens" ↔ "Walgreens Pharmacy",
    # "CVS" ↔ "CVS Pharmacy"). These are obviously the same entity
    # even when token_set_ratio dips below the threshold on short
    # names, so we don't want to penalize Overture for them.
    osm_idx_arr = matches["osm_idx"].to_numpy().astype(int)
    ov_idx_arr = matches["overture_idx"].to_numpy().astype(int)
    keep = np.ones(len(matches), dtype = bool)
    for i in range(len(matches)):
        gname = ghost_names[osm_idx_arr[i]]
        oname = ov_names[ov_idx_arr[i]]
        if gname and oname and _is_token_subset_or_superset(gname, oname):
            keep[i] = False

    if not keep.all():
        matches = matches.iloc[keep].reset_index(drop = True)

    return matches[
        ["osm_idx", "overture_idx", "composite_score", "distance_m"]
    ].reset_index(drop = True)


_EARTH_RADIUS_M = 6_371_000.0


def apply_current_survivor_filter(
    matches: pd.DataFrame,
    unmatched_overture: gpd.GeoDataFrame,
    *,
    rated_snapshot_path: Path,
    radius_m: float,
    name_similarity_threshold: float,
    test_bbox: dict | None = None,
    query_chunk_size: int = 50_000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Drop shadow matches where the POI is still present in the live
    OSM snapshot under a different geometry / spelling.

    For each Overture row in ``matches``, find the live rated-snapshot
    POIs within ``radius_m`` and check whether any name token-set-
    matches the Overture name at ≥ ``name_similarity_threshold``. If
    so, drop the match — the primary matcher just missed it.

    Scales to nationwide via:

    - A single BallTree over rated-snapshot centroids (haversine
      metric) instead of a DuckDB cross-join. Build is O(M log M),
      query is O(log M) per match. The earlier DuckDB
      ``ST_Distance_Sphere`` implementation also returned distances
      that were ~50% off in the v1.4.1 pin we use today (the
      bundled formula is buggy — NYC → LA registers as ~4,900 km
      versus the correct ~3,940 km), which silently broke the 50 m
      gate. BallTree haversine here is the reference implementation.
    - DuckDB is still used for the *load*: it computes centroids in
      SQL and emits just ``(lon, lat, name)``, avoiding the ~8 GB
      peak from materializing 8.7 M shapely Points via GeoPandas.
      The optional ``test_bbox`` pushes down into that SQL for the
      Seattle path.
    - Snapshot rows with empty names are dropped before the tree
      build — they can't satisfy the name gate anyway, and this
      typically halves the tree size on real OSM data.
    - The match side is queried in chunks of ``query_chunk_size`` so
      peak memory stays bounded regardless of match count.

    Returns ``(kept_matches, n_dropped)``.
    """
    if matches.empty:
        return matches.copy(), 0

    ov_idx_arr = matches["overture_idx"].to_numpy().astype(int)

    if verbose:
        print(
            f"  R1 (current-OSM-survivor): radius="
            f"{radius_m}m, name>={name_similarity_threshold}"
        )

    # ---------------- snapshot load + centroid extraction ----------
    # DuckDB extracts (lon, lat, name) directly from the parquet,
    # computing centroids server-side in C and emitting three thin
    # columns. This avoids materializing 8.7M shapely Point objects
    # (which costs ~8 GB peak when going through GeoPandas) and lets
    # us push the bbox filter into the SQL for the Seattle path.
    # Filters out unnamed rows up front — they can't satisfy the
    # name gate anyway, typically halves the tree size.
    if verbose:
        print(
            f"    Loading rated snapshot centroids from "
            f"{rated_snapshot_path} ..."
        )
    bbox_clause = ""
    if test_bbox is not None:
        bbox_clause = (
            "AND lon BETWEEN "
            f"{test_bbox['xmin']} AND {test_bbox['xmax']} "
            "AND lat BETWEEN "
            f"{test_bbox['ymin']} AND {test_bbox['ymax']}"
        )
    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
        snap = con.execute(f"""
            WITH centroids AS (
                SELECT
                    ST_X(ST_Centroid(geometry)) AS lon,
                    ST_Y(ST_Centroid(geometry)) AS lat,
                    COALESCE(name, '') AS name
                FROM read_parquet('{rated_snapshot_path}')
            )
            SELECT lon, lat, name
            FROM centroids
            WHERE name <> ''
              AND lon IS NOT NULL
              AND lat IS NOT NULL
              {bbox_clause}
        """).fetch_df()
    finally:
        con.close()

    snap_lons = snap["lon"].to_numpy()
    snap_lats = snap["lat"].to_numpy()
    snap_names = snap["name"].to_numpy()
    del snap
    gc.collect()

    if verbose:
        print(
            f"    Snapshot rows in scope (named, non-NaN, in bbox): "
            f"{len(snap_lons):,}"
        )

    if len(snap_lons) == 0:
        return matches.copy(), 0

    # ---------------- BallTree build ------------------------------
    if verbose:
        print(
            f"    Building BallTree on {len(snap_lons):,} points "
            f"(haversine, ~140 MB at 8.7 M points) ..."
        )
    snap_coords_rad = np.column_stack([
        np.deg2rad(snap_lats), np.deg2rad(snap_lons),
    ])
    tree = BallTree(snap_coords_rad, metric = "haversine")
    del snap_coords_rad
    gc.collect()

    # ---------------- per-match query + name check ----------------
    ov_lons = unmatched_overture.geometry.x.to_numpy()[ov_idx_arr]
    ov_lats = unmatched_overture.geometry.y.to_numpy()[ov_idx_arr]
    ov_names = _to_str_array(unmatched_overture["name"])[ov_idx_arr]
    radius_rad = radius_m / _EARTH_RADIUS_M

    suppress_idx_set: set[int] = set()
    n_pairs_scored = 0
    n_chunks = (len(matches) + query_chunk_size - 1) // query_chunk_size
    if verbose:
        print(
            f"    Querying for {len(matches):,} matches in "
            f"{n_chunks} chunk(s) of up to {query_chunk_size:,} ..."
        )

    for start in range(0, len(matches), query_chunk_size):
        end = min(start + query_chunk_size, len(matches))
        chunk_coords = np.column_stack([
            np.deg2rad(ov_lats[start:end]),
            np.deg2rad(ov_lons[start:end]),
        ])
        neighbor_idx_per_match = tree.query_radius(
            chunk_coords, r = radius_rad,
        )

        for local_i, idx_arr in enumerate(neighbor_idx_per_match):
            if len(idx_arr) == 0:
                continue
            ov_name = ov_names[start + local_i]
            if not ov_name:
                continue
            global_i = start + local_i
            for snap_i in idx_arr:
                snap_name = snap_names[snap_i]
                if not snap_name:
                    continue
                n_pairs_scored += 1
                sim = fuzz.token_set_ratio(ov_name, snap_name)
                if sim >= name_similarity_threshold:
                    suppress_idx_set.add(global_i)
                    break  # short-circuit; one hit is enough

    if verbose:
        print(
            f"    Scored {n_pairs_scored:,} name-similarity pairs; "
            f"R1 suppressed: {len(suppress_idx_set)}"
        )

    if not suppress_idx_set:
        return matches.copy(), 0

    suppress_idx_arr = np.fromiter(suppress_idx_set, dtype = int)
    keep_mask = np.ones(len(matches), dtype = bool)
    keep_mask[suppress_idx_arr] = False
    kept = matches.loc[keep_mask].reset_index(drop = True)
    return kept, int(len(suppress_idx_set))


def apply_shadow_match(
    conflated_path: Path,
    ghosts_path: Path,
    fitted_params_path: Path,
    output_path: Path,
    *,
    min_match_score: float,
    max_radius_m: float,
    default_radius_m: float,
    distance_weight: float,
    name_weight: float,
    type_weight: float,
    identifier_weight: float,
    default_delta: float,
    test_bbox: dict | None = None,
    rated_snapshot_path: Path | None = None,
    survivor_filter: dict | None = None,
    min_prior_name_match_score: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Post-process a conflated dataset with the change-detection penalty.

    Reads the baseline ``conflated.parquet``, runs a secondary match
    between its unmatched-Overture rows and the ghost dataset, applies
    ``conf_mean *= delta_group`` for each shadow match, attaches audit
    columns, and writes a new parquet at ``output_path``.

    Args:
        conflated_path: Existing baseline conflated parquet.
        ghosts_path: Output of ``scripts/conflation/build_ghosts.py``.
        fitted_params_path: ``fitted_params.csv`` from the chosen
            ``model_output`` run; supplies per-group δ.
        output_path: Where to write the change-detection conflated
            parquet.
        min_match_score, max_radius_m, default_radius_m,
        distance_weight, name_weight, type_weight, identifier_weight:
            Match-scoring controls; mirror the main conflation knobs.
        default_delta: Fallback δ for ghosts whose ``shared_label``
            isn't in ``fitted_params.csv``.
        test_bbox: If set, filter ghosts to this bbox before matching
            (useful for the Seattle A/B test so the matcher's
            candidate search isn't dominated by national-scale ghosts).
        verbose: Print progress.

    Returns:
        Summary dict with shadow-match counts and elapsed timings.
    """
    if verbose:
        print(f"Reading conflated parquet from {conflated_path} ...")
    conflated = gpd.read_parquet(conflated_path)
    if verbose:
        print(f"  {len(conflated):,} rows")

    if verbose:
        print(f"Reading ghosts from {ghosts_path} ...")
    ghosts = gpd.read_parquet(ghosts_path)
    if verbose:
        print(f"  {len(ghosts):,} ghosts")

    if test_bbox is not None:
        from shapely.geometry import box
        bbox_geom = box(
            test_bbox["xmin"], test_bbox["ymin"],
            test_bbox["xmax"], test_bbox["ymax"],
        )
        ghosts = ghosts[
            ghosts.geometry.within(bbox_geom)
        ].reset_index(drop = True)
        if verbose:
            print(
                f"  Filtered to test bbox: "
                f"{len(ghosts):,} ghosts"
            )

    delta_lookup, delta_default = load_delta_lookup(
        fitted_params_path, default_delta,
    )
    if verbose:
        print(
            f"Loaded delta lookup: {len(delta_lookup)} groups "
            f"(default = {delta_default:.4f})"
        )

    # Isolate the unmatched-Overture subset; this is the only segment
    # the change-detection pass touches.
    is_ov = conflated["source"].to_numpy() == "overture"
    ov_global_idx = np.where(is_ov)[0]
    if verbose:
        print(
            f"Unmatched Overture rows in baseline: "
            f"{len(ov_global_idx):,}"
        )

    if len(ov_global_idx) == 0 or len(ghosts) == 0:
        if verbose:
            print(
                "  Nothing to shadow-match; writing pass-through "
                "with audit columns."
            )
        matches = pd.DataFrame(
            columns = [
                "osm_idx", "overture_idx",
                "composite_score", "distance_m",
            ]
        )
    else:
        # ``find_shadow_matches`` expects an Overture-shaped GDF with
        # ``name``, ``brand``, ``shared_label``, geometry.
        unmatched_ov = conflated.iloc[ov_global_idx].reset_index(
            drop = True
        )
        if verbose:
            print("Running shadow match ...")
        matches = find_shadow_matches(
            unmatched_ov, ghosts,
            min_match_score = min_match_score,
            max_radius_m = max_radius_m,
            default_radius_m = default_radius_m,
            distance_weight = distance_weight,
            name_weight = name_weight,
            type_weight = type_weight,
            identifier_weight = identifier_weight,
            min_prior_name_match_score = min_prior_name_match_score,
        )
        if verbose:
            print(
                f"  Shadow matches (pre-survivor-filter): "
                f"{len(matches):,}"
            )

    # -- Current-OSM-survivor filter ----------------------------------
    n_survivor_dropped = 0
    if (
        survivor_filter
        and bool(survivor_filter.get("enabled", False))
        and rated_snapshot_path is not None
        and len(matches) > 0
    ):
        if verbose:
            print("Applying current-OSM-survivor filter ...")
        matches, n_survivor_dropped = apply_current_survivor_filter(
            matches = matches,
            unmatched_overture = unmatched_ov,
            rated_snapshot_path = rated_snapshot_path,
            radius_m = float(survivor_filter.get("radius_m", 50)),
            name_similarity_threshold = float(
                survivor_filter.get("name_similarity_threshold", 70)
            ),
            test_bbox = test_bbox,
            verbose = verbose,
        )
        if verbose:
            print(
                f"  Shadow matches (post-survivor-filter): "
                f"{len(matches):,} (dropped {n_survivor_dropped})"
            )

    # -- Build audit columns -------------------------------------------
    n = len(conflated)
    shadow_matched = np.zeros(n, dtype = bool)
    shadow_ghost_id = np.full(n, None, dtype = object)
    shadow_event_type = np.full(n, None, dtype = object)
    shadow_event_timestamp = pd.Series(
        [pd.NaT] * n, dtype = "datetime64[ns, UTC]",
    )
    shadow_score = np.full(n, np.nan, dtype = np.float64)
    shadow_distance_m = np.full(n, np.nan, dtype = np.float64)
    original_conf_mean = conflated["conf_mean"].to_numpy().astype(
        np.float64
    ).copy()

    new_conf_mean = original_conf_mean.copy()
    new_conf_lower = conflated["conf_lower"].to_numpy().astype(
        np.float64
    ).copy()
    new_conf_upper = conflated["conf_upper"].to_numpy().astype(
        np.float64
    ).copy()

    if len(matches) > 0:
        # Map secondary-pass "osm_idx" (ghost row) and "overture_idx"
        # (row in the unmatched-Overture subset) back to global rows.
        ghost_row = matches["osm_idx"].to_numpy().astype(int)
        ov_sub_row = matches["overture_idx"].to_numpy().astype(int)
        target_global = ov_global_idx[ov_sub_row]

        ghost_id_arr = ghosts["ghost_id"].to_numpy()[ghost_row]
        event_type_arr = ghosts["event_type"].to_numpy()[ghost_row]
        event_ts_arr = ghosts["event_timestamp"].to_numpy()[ghost_row]
        ghost_label_arr = (
            ghosts["shared_label"].fillna("").astype(str).to_numpy()
        )[ghost_row]

        shadow_matched[target_global] = True
        shadow_ghost_id[target_global] = ghost_id_arr
        shadow_event_type[target_global] = event_type_arr
        shadow_event_timestamp.iloc[target_global] = (
            pd.to_datetime(event_ts_arr, utc = True, errors = "coerce")
        )
        shadow_score[target_global] = matches[
            "composite_score"
        ].to_numpy().astype(np.float64)
        shadow_distance_m[target_global] = matches[
            "distance_m"
        ].to_numpy().astype(np.float64)

        # Apply per-group penalty.
        deltas = np.array(
            [
                delta_lookup.get(lb, delta_default)
                if lb else delta_default
                for lb in ghost_label_arr
            ],
            dtype = np.float64,
        )
        new_conf_mean[target_global] = (
            original_conf_mean[target_global] * deltas
        )
        # Re-weighted probabilities lose their CI semantics.
        new_conf_lower[target_global] = np.nan
        new_conf_upper[target_global] = np.nan

    # -- Stitch into output --------------------------------------------
    out = conflated.copy()
    out["conf_mean"] = new_conf_mean
    out["conf_lower"] = new_conf_lower
    out["conf_upper"] = new_conf_upper
    out["shadow_matched"] = shadow_matched
    out["shadow_ghost_id"] = shadow_ghost_id
    out["shadow_event_type"] = shadow_event_type
    out["shadow_event_timestamp"] = shadow_event_timestamp.values
    out["shadow_score"] = shadow_score
    out["shadow_distance_m"] = shadow_distance_m
    out["original_conf_mean"] = original_conf_mean

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    if verbose:
        print(f"Writing {output_path} ...")
    out.to_parquet(output_path, compression = "zstd")

    summary = {
        "n_total": int(n),
        "n_unmatched_overture": int(len(ov_global_idx)),
        "n_ghosts": int(len(ghosts)),
        "n_shadow_matches": int(len(matches)),
        "n_survivor_dropped": int(n_survivor_dropped),
        "mean_penalty_factor": (
            float(
                (new_conf_mean[shadow_matched]
                 / np.where(
                     original_conf_mean[shadow_matched] == 0, 1,
                     original_conf_mean[shadow_matched],
                 )
                ).mean()
            )
            if shadow_matched.any() else float("nan")
        ),
    }

    # Confirm read-back schema integrity.
    if verbose:
        sch = pq.read_schema(output_path)
        missing = [c for c in SHADOW_AUDIT_COLUMNS if c not in sch.names]
        if missing:
            print(
                f"  WARNING: audit columns missing from output: "
                f"{missing}"
            )

    gc.collect()
    return summary
