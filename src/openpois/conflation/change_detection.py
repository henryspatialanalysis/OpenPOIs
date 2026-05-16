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

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
        )
        if verbose:
            print(f"  Shadow matches: {len(matches):,}")

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
