#!/usr/bin/env python
"""
Conflate rated OSM POIs with Overture Maps POIs into a unified dataset.

Reads both snapshots, assigns each POI a shared taxonomy label via CSV
crosswalk files, finds spatial candidates within per-category radii using a
BallTree, scores candidate pairs on distance, name similarity, type agreement,
and shared identifiers, performs greedy one-to-one matching, and merges all
POIs (matched and unmatched) into a single GeoParquet output.

Config keys used (config.yaml):
    snapshot_osm.rated_snapshot            — rated OSM GeoParquet input path
    snapshot_overture.snapshot             — Overture GeoParquet input path
    conflation.conflated                   — output GeoParquet path
    download.osm.filter_keys               — tag keys used for taxonomy assignment
    conflation.overture_confidence_weight  — weight on Overture confidence in scoring
    conflation.min_match_score             — minimum composite score to accept a match
    conflation.max_radius_m                — maximum candidate search radius in meters
    conflation.default_radius_m            — fallback radius for unclassified POIs
    conflation.distance_weight             — scoring weight for spatial distance
    conflation.name_weight                 — scoring weight for name similarity
    conflation.type_weight                 — scoring weight for taxonomy agreement
    conflation.identifier_weight           — scoring weight for shared identifiers
    conflation.chunk_size                  — BallTree chunk size for memory management
    conflation.test_bbox                   — small bbox used with --test flag

Usage:
    python scripts/conflation/conflate.py           # full CONUS run
    python scripts/conflation/conflate.py --test    # Seattle test bbox

Output file:
    conflated.parquet — GeoParquet with all OSM + Overture POIs, columns:
        shared_label, source (matched/osm/overture), match_score,
        osm_id, overture_id, name, conf_mean/lower/upper, geometry, ...
"""
from __future__ import annotations

import argparse
import gc
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
from config_versioned import Config
from shapely.geometry import box

from openpois.conflation.chunking import extract_centroids_lonlat
from openpois.conflation.dedup_overture import mark_no_conflate
from openpois.conflation.match import (
    compute_match_scores,
    find_and_score_matches_chunked,
    find_spatial_candidates,
    select_best_matches,
)
from openpois.conflation.merge import (
    build_merge_parts,
    build_merge_parts_chunked,
    save_conflated_from_parts,
)
from openpois.conflation.taxonomy import (
    assign_osm_shared_label,
    assign_overture_shared_label,
    compute_osm_l0_bits,
    compute_overture_l0_bits,
    load_match_radii,
    load_osm_crosswalk,
    load_overture_crosswalk,
    load_top_level_matches,
)

CHECKPOINT_SUBDIR = "chunk_matches"
DEDUP_CHECKPOINT_SUBDIR = "chunk_selfdedup"
DEDUP_DROPPED_FILE = "overture_dedup_dropped.parquet"
DEDUP_POST_FILTER_FILE = "overture_post_dedup.parquet"


# -----------------------------------------------------------------
# Memory instrumentation
# -----------------------------------------------------------------

_RSS_T0 = time.time()


def _read_proc_status() -> dict[str, int]:
    """Return VmRSS and VmHWM in bytes from /proc/self/status (Linux)."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, val, unit = line.split()
                    out[key.rstrip(":")] = int(val) * 1024
    except FileNotFoundError:
        pass
    return out


def log_rss(label: str) -> None:
    """Print current RSS and peak RSS at a phase boundary.

    Forces a gc.collect() first so the report reflects retained
    memory, not pending garbage. Cheap to call (~few ms).
    """
    gc.collect()
    info = _read_proc_status()
    rss = info.get("VmRSS", 0) / 2**30
    hwm = info.get("VmHWM", 0) / 2**30
    elapsed = time.time() - _RSS_T0
    print(
        f"  [RSS {rss:5.2f} GB | peak {hwm:5.2f} GB | "
        f"+{elapsed:6.0f}s] {label}",
        flush = True,
    )


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

OSM_PATH = config.get_file_path("snapshot_osm", "rated_snapshot")
OVERTURE_PATH = config.get_file_path("snapshot_overture", "snapshot")
OUTPUT_PATH = config.get_file_path("conflation", "conflated")

FILTER_KEYS = config.get("download", "osm", "filter_keys")

OVERTURE_CONF_WEIGHT = config.get(
    "conflation", "overture_confidence_weight"
)
MIN_MATCH_SCORE = config.get("conflation", "min_match_score")
MAX_RADIUS_M = config.get("conflation", "max_radius_m")
DEFAULT_RADIUS_M = config.get("conflation", "default_radius_m")
DISTANCE_WEIGHT = config.get("conflation", "distance_weight")
NAME_WEIGHT = config.get("conflation", "name_weight")
TYPE_WEIGHT = config.get("conflation", "type_weight")
IDENTIFIER_WEIGHT = config.get("conflation", "identifier_weight")
CHUNK_SIZE = config.get("conflation", "chunk_size")
CHUNK_TARGET_POIS = config.get(
    "conflation", "chunk_target_pois",
)
TEST_BBOX = config.get("conflation", "test_bbox")

DEDUP_CFG = config.get(
    "conflation", "overture_internal_dedup",
)
DEDUP_ENABLED = bool(DEDUP_CFG.get("enabled", True))
DEDUP_MIN_SCORE = float(DEDUP_CFG.get("min_match_score", 0.75))
DEDUP_MAX_RADIUS_M = float(DEDUP_CFG.get("max_radius_m", 100))
DEDUP_CHUNK_TARGET = int(
    DEDUP_CFG.get("chunk_target_pois", CHUNK_TARGET_POIS)
)
DEDUP_DUCKDB_MEM = str(
    DEDUP_CFG.get("duckdb_memory_limit", "4GB")
)

# Columns needed for matching (memory optimization)
OSM_MATCH_COLS = [
    "osm_id", "osm_type", "name", "brand", "brand:wikidata",
    "website", "phone", "amenity", "shop", "healthcare", "leisure",
    "conf_mean", "conf_lower", "conf_upper", "geometry",
]
OVERTURE_MATCH_COLS = [
    "overture_id", "taxonomy_l0", "taxonomy_l1", "taxonomy_l2",
    "overture_name", "brand_name", "confidence", "geometry",
]
# Columns needed downstream by ``build_merge_parts*``. Reloaded into
# memory after the chunked matcher returns so the matching phase can
# run without the full source GeoDataFrames resident.
OSM_MERGE_COLS = [
    "osm_id", "osm_type", "name", "brand",
    "conf_mean", "conf_lower", "conf_upper", "geometry",
    # Address subfields
    "addr:housenumber", "addr:street", "addr:unit",
    "addr:city", "addr:state", "addr:postcode", "addr:country",
    # Contact + hours + access
    "phone", "website", "opening_hours", "access",
]
OVERTURE_MERGE_COLS = [
    "overture_id", "overture_name", "brand_name",
    "confidence", "geometry",
    # Address subfields (no housenumber / unit on the Overture side)
    "overture_addr_street", "overture_addr_city",
    "overture_addr_state", "overture_addr_postcode",
    "overture_addr_country",
    # Contact arrays + alternates
    "overture_websites", "overture_phones", "overture_socials",
    "overture_categories_alternate",
]


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


def _write_dedup_audit(
    overture_gdf,
    no_conflate: np.ndarray,
    components: np.ndarray,
    audit_path: Path,
) -> None:
    """Write one row per dropped Overture POI with its cluster winner.

    Columns: overture_id, cluster_id, winner_overture_id,
    confidence, geometry. Small file (~1 row per dropped POI); lets
    us spot-check dedup decisions in a GIS or with DuckDB.
    """
    losers_idx = np.where(no_conflate)[0]
    if len(losers_idx) == 0:
        return

    n_comps = int(components.max()) + 1
    winner_of = np.full(n_comps, -1, dtype = np.int64)
    not_loser_idx = np.where(~no_conflate)[0]
    winner_of[components[not_loser_idx]] = not_loser_idx
    winner_idx = winner_of[components[losers_idx]]

    ids = overture_gdf["overture_id"].to_numpy()
    confidence = (
        overture_gdf["confidence"].to_numpy()
        if "confidence" in overture_gdf.columns
        else np.full(len(overture_gdf), None, dtype = object)
    )
    audit = gpd.GeoDataFrame(
        {
            "overture_id": ids[losers_idx],
            "cluster_id": components[losers_idx],
            "winner_overture_id": ids[winner_idx],
            "confidence": confidence[losers_idx],
            "geometry": overture_gdf.geometry.values[losers_idx],
        },
        crs = overture_gdf.crs,
    )
    audit_path.parent.mkdir(parents = True, exist_ok = True)
    audit.to_parquet(audit_path, compression = "zstd")


def _load_gdf(
    path, columns, test_bbox = None, label = "dataset"
):
    """Load a GeoParquet, optionally filtering to a test bbox."""
    print(f"Loading {label} from {path} ...")
    # Read only needed columns
    avail_cols = pq.read_schema(path).names
    cols = [c for c in columns if c in avail_cols]
    gdf = gpd.read_parquet(path, columns = cols)
    print(f"  Loaded {len(gdf):,} rows ({len(cols)} columns)")

    if test_bbox is not None:
        bbox_geom = box(
            test_bbox["xmin"], test_bbox["ymin"],
            test_bbox["xmax"], test_bbox["ymax"],
        )
        gdf = gdf[gdf.geometry.within(bbox_geom)].reset_index(
            drop = True
        )
        print(
            f"  Filtered to test bbox: {len(gdf):,} rows"
        )

    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Conflate OSM and Overture Maps POIs."
    )
    parser.add_argument(
        "--test",
        action = "store_true",
        help = (
            "Filter both datasets to a small bbox "
            "(Seattle area) for testing."
        ),
    )
    parser.add_argument(
        "--no-chunk",
        action = "store_true",
        help = (
            "Disable spatial chunking and run matching over the "
            "full dataset in one pass. Uses more memory; kept as "
            "a debug/baseline option."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default = "",
        help = (
            "If set, inserts ``_<suffix>`` before .parquet in the "
            "output filename (e.g. --output-suffix=baseline writes "
            "conflated_baseline.parquet). Used by the change-detection "
            "A/B testing workflow."
        ),
    )
    args = parser.parse_args()

    if args.output_suffix:
        OUTPUT_PATH = OUTPUT_PATH.with_name(
            f"{OUTPUT_PATH.stem}_{args.output_suffix}"
            f"{OUTPUT_PATH.suffix}"
        )
    t0 = time.time()

    test_bbox = TEST_BBOX if args.test else None

    log_rss("startup")

    # -- Load data -------------------------------------------------
    osm_gdf = _load_gdf(
        OSM_PATH, OSM_MATCH_COLS,
        test_bbox = test_bbox, label = "OSM rated",
    )
    log_rss("after OSM load")
    overture_gdf = _load_gdf(
        OVERTURE_PATH, OVERTURE_MATCH_COLS,
        test_bbox = test_bbox, label = "Overture",
    )
    log_rss("after Overture load")

    # -- Taxonomy assignment ---------------------------------------
    # Overture taxonomy is assigned first so the internal-dedup pass
    # (below) can reuse shared_label + L0 bits for its type scoring.
    # OSM taxonomy follows after dedup has filtered Overture rows.
    print("\nAssigning Overture shared labels ...")
    overture_crosswalk = load_overture_crosswalk()
    match_radii = load_match_radii()
    top_level_matches = load_top_level_matches()

    overture_shared_labels, overture_radii = (
        assign_overture_shared_label(
            overture_gdf, overture_crosswalk, match_radii,
            default_radius_m = DEFAULT_RADIUS_M,
        )
    )
    ov_assigned = np.sum(overture_shared_labels != "")
    print(
        f"  Overture: {ov_assigned:,}/{len(overture_gdf):,}"
        f" assigned"
    )
    overture_l0_bits = compute_overture_l0_bits(
        overture_gdf["taxonomy_l0"].fillna("").to_numpy(),
    )
    del overture_crosswalk
    log_rss("after Overture taxonomy assignment")

    # -- Overture internal deduplication ---------------------------
    conflation_dir = Path(OUTPUT_PATH).parent
    dedup_summary: dict | None = None
    # Path used for the Overture reload at merge time. Overridden to
    # a post-dedup temp parquet when dedup runs so the reload's row
    # indices match the match-phase output.
    overture_merge_source_path = OVERTURE_PATH
    overture_merge_needs_test_bbox = True
    post_dedup_resume_path = conflation_dir / DEDUP_POST_FILTER_FILE
    if DEDUP_ENABLED and post_dedup_resume_path.exists():
        print(
            f"\nReusing post-dedup Overture from "
            f"{post_dedup_resume_path} (skipping dedup pass)."
        )
        overture_gdf = (
            gpd.read_parquet(post_dedup_resume_path)
            .reset_index(drop = True)
        )
        overture_shared_labels, overture_radii = (
            assign_overture_shared_label(
                overture_gdf, load_overture_crosswalk(), match_radii,
                default_radius_m = DEFAULT_RADIUS_M,
            )
        )
        overture_l0_bits = compute_overture_l0_bits(
            overture_gdf["taxonomy_l0"].fillna("").to_numpy(),
        )
        overture_merge_source_path = post_dedup_resume_path
        overture_merge_needs_test_bbox = False
        log_rss("after Overture post-dedup reload")
    elif DEDUP_ENABLED:
        dedup_checkpoint_dir = (
            conflation_dir / DEDUP_CHECKPOINT_SUBDIR
        )
        (
            no_conflate,
            dedup_components,
            dedup_pairs,
            dedup_summary,
        ) = mark_no_conflate(
            overture_gdf,
            overture_shared_labels,
            overture_l0_bits,
            min_match_score = DEDUP_MIN_SCORE,
            max_radius_m = DEDUP_MAX_RADIUS_M,
            chunk_target_pois = DEDUP_CHUNK_TARGET,
            chunk_size = CHUNK_SIZE,
            checkpoint_dir = dedup_checkpoint_dir,
            duckdb_memory_limit = DEDUP_DUCKDB_MEM,
        )
        log_rss(
            f"after Overture dedup "
            f"({dedup_summary['n_dropped']:,} marked)"
        )

        if no_conflate.any():
            audit_path = conflation_dir / DEDUP_DROPPED_FILE
            _write_dedup_audit(
                overture_gdf, no_conflate,
                dedup_components, audit_path,
            )
            print(
                f"  Wrote {dedup_summary['n_dropped']:,} dropped "
                f"rows to {audit_path}"
            )

        # Filter overture rows + parallel arrays. A boolean-mask
        # copy allocates a fresh frame; ``del`` + ``gc.collect`` on
        # the old one releases the dropped-rows' memory before the
        # OSM taxonomy + matching phase begins.
        keep_mask = ~no_conflate
        n_before = len(overture_gdf)
        overture_gdf = (
            overture_gdf.loc[keep_mask].reset_index(drop = True)
        )
        overture_shared_labels = overture_shared_labels[keep_mask]
        overture_radii = overture_radii[keep_mask]
        overture_l0_bits = overture_l0_bits[keep_mask]
        del (
            no_conflate, dedup_components,
            dedup_pairs, keep_mask,
        )
        gc.collect()
        if dedup_checkpoint_dir.exists():
            shutil.rmtree(dedup_checkpoint_dir)
        print(
            f"  Overture rows after dedup: {len(overture_gdf):,} "
            f"(dropped {n_before - len(overture_gdf):,})"
        )

        # Spill post-dedup Overture to disk so the later merge-phase
        # reload sees rows whose indices match the match-phase output
        # (the pre-dedup snapshot on disk would misalign against
        # ``matches.overture_idx``). Minimal column set — merge adds
        # no metadata beyond what's already here.
        overture_merge_source_path = (
            conflation_dir / DEDUP_POST_FILTER_FILE
        )
        overture_merge_source_path.parent.mkdir(
            parents = True, exist_ok = True,
        )
        overture_gdf.to_parquet(
            overture_merge_source_path, compression = "zstd",
        )
        overture_merge_needs_test_bbox = False
        log_rss("after Overture dedup filter")
    else:
        print("\nOverture internal deduplication disabled.")

    # -- OSM taxonomy assignment -----------------------------------
    print("\nAssigning OSM shared labels ...")
    osm_crosswalk = load_osm_crosswalk()
    osm_shared_labels, osm_radii = assign_osm_shared_label(
        osm_gdf, osm_crosswalk, match_radii, FILTER_KEYS,
        default_radius_m = DEFAULT_RADIUS_M,
    )
    osm_assigned = np.sum(osm_shared_labels != "")
    print(
        f"  OSM: {osm_assigned:,}/{len(osm_gdf):,} assigned"
    )
    osm_l0_bits = compute_osm_l0_bits(
        osm_gdf, top_level_matches,
    )

    # Drop columns only needed for taxonomy assignment
    for col in [
        "amenity", "shop", "healthcare", "leisure",
        "osm_type", "brand:wikidata", "website", "phone",
    ]:
        if col in osm_gdf.columns:
            osm_gdf.drop(columns = col, inplace = True)
    for col in ["taxonomy_l0", "taxonomy_l1", "taxonomy_l2"]:
        if col in overture_gdf.columns:
            overture_gdf.drop(columns = col, inplace = True)
    del osm_crosswalk, top_level_matches, match_radii
    log_rss("after taxonomy assignment + tag-col drop")

    # -- Matching --------------------------------------------------
    # Prepare name/brand arrays once (used by both code paths).
    osm_names = osm_gdf["name"].to_numpy()
    osm_brands = (
        osm_gdf["brand"].to_numpy()
        if "brand" in osm_gdf.columns
        else np.full(len(osm_gdf), None, dtype = object)
    )
    overture_names = overture_gdf["overture_name"].to_numpy()
    overture_brands = (
        overture_gdf["brand_name"].to_numpy()
        if "brand_name" in overture_gdf.columns
        else np.full(
            len(overture_gdf), None, dtype = object
        )
    )

    chunk_summary: dict | None = None
    checkpoint_dir: Path | None = None

    if args.no_chunk:
        # -- Non-chunked baseline (full-dataset pipeline) ----------
        print(
            f"\nFinding spatial candidates (max {MAX_RADIUS_M}m) ..."
        )
        candidates = find_spatial_candidates(
            osm_geom = osm_gdf.geometry.values,
            overture_geom = overture_gdf.geometry.values,
            osm_radii_m = osm_radii,
            max_radius_m = MAX_RADIUS_M,
            chunk_size = CHUNK_SIZE,
        )
        print(f"  Found {len(candidates):,} candidate pairs")
        gc.collect()

        if candidates.empty:
            print(
                "No spatial candidates found. "
                "Merging all as unmatched."
            )
            matches = candidates
        else:
            print("\nScoring candidates ...")
            scored = compute_match_scores(
                candidates = candidates,
                osm_names = osm_names,
                osm_brands = osm_brands,
                overture_names = overture_names,
                overture_brands = overture_brands,
                osm_shared_labels = osm_shared_labels,
                overture_shared_labels = overture_shared_labels,
                osm_radii_m = osm_radii,
                osm_l0_bits = osm_l0_bits,
                overture_l0_bits = overture_l0_bits,
                distance_weight = DISTANCE_WEIGHT,
                name_weight = NAME_WEIGHT,
                type_weight = TYPE_WEIGHT,
                identifier_weight = IDENTIFIER_WEIGHT,
            )
            print(
                f"  Mean composite score: "
                f"{scored['composite_score'].mean():.3f}"
            )

            print(
                f"\nSelecting best matches "
                f"(min_score={MIN_MATCH_SCORE}) ..."
            )
            matches = select_best_matches(
                scored, min_score = MIN_MATCH_SCORE,
            )
            print(
                f"  Selected {len(matches):,} one-to-one matches"
            )

            del scored, candidates
            gc.collect()
    else:
        # -- Chunked driver (default) ------------------------------
        checkpoint_dir = conflation_dir / CHECKPOINT_SUBDIR
        print(
            f"\nRunning chunked matching "
            f"(target ~{CHUNK_TARGET_POIS:,} POIs/chunk, "
            f"max {MAX_RADIUS_M}m) ..."
        )

        # Precompute centroids before freeing the source frames so
        # the matching phase never holds the full GeoDataFrames in
        # memory. Geometries are reloaded from disk for the merge.
        print("  Precomputing centroids ...")
        osm_centroids_lonlat = extract_centroids_lonlat(
            np.asarray(osm_gdf.geometry.values)
        )
        overture_centroids_lonlat = extract_centroids_lonlat(
            np.asarray(overture_gdf.geometry.values)
        )
        del osm_gdf, overture_gdf
        log_rss("after dropping gdfs (centroids extracted)")

        matches, chunk_summary = find_and_score_matches_chunked(
            osm_centroids_lonlat = osm_centroids_lonlat,
            overture_centroids_lonlat = overture_centroids_lonlat,
            osm_radii_m = osm_radii,
            osm_shared_labels = osm_shared_labels,
            overture_shared_labels = overture_shared_labels,
            osm_l0_bits = osm_l0_bits,
            overture_l0_bits = overture_l0_bits,
            osm_names = osm_names,
            osm_brands = osm_brands,
            overture_names = overture_names,
            overture_brands = overture_brands,
            distance_weight = DISTANCE_WEIGHT,
            name_weight = NAME_WEIGHT,
            type_weight = TYPE_WEIGHT,
            identifier_weight = IDENTIFIER_WEIGHT,
            min_match_score = MIN_MATCH_SCORE,
            max_radius_m = MAX_RADIUS_M,
            chunk_target_pois = CHUNK_TARGET_POIS,
            chunk_size = CHUNK_SIZE,
            checkpoint_dir = checkpoint_dir,
        )
        del osm_centroids_lonlat, overture_centroids_lonlat
        print(
            f"  Selected {len(matches):,} one-to-one matches "
            f"across {chunk_summary['n_chunks']} chunks "
            f"(Overture dedup drops: "
            f"{chunk_summary['n_overture_dedup_drops']:,})"
        )
        log_rss("after chunked matching + dedup")

    del osm_names, osm_brands
    del overture_names, overture_brands
    gc.collect()

    # Drop scoring intermediates the merge doesn't need. Keeps only
    # the four columns read by ``_build_matched_gdf``.
    keep_cols = [
        "osm_idx", "overture_idx",
        "composite_score", "distance_m",
    ]
    matches = matches[
        [c for c in keep_cols if c in matches.columns]
    ].reset_index(drop = True)
    gc.collect()

    # -- Merge (disk-backed to limit memory) -----------------------
    print("\nMerging into unified dataset ...")
    match_score_mean = (
        matches["composite_score"].mean() if len(matches) > 0
        else float("nan")
    )
    match_dist_mean = (
        matches["distance_m"].mean() if len(matches) > 0
        else float("nan")
    )
    n_matches = len(matches)

    if chunk_summary is not None:
        # Reload only the columns the merge needs — the matching
        # phase has already returned the dedup-resolved matches, so
        # we no longer need the wide load schema.
        print("  Reloading source frames for merge ...")
        osm_gdf = _load_gdf(
            OSM_PATH, OSM_MERGE_COLS,
            test_bbox = test_bbox, label = "OSM (merge cols)",
        )
        overture_gdf = _load_gdf(
            overture_merge_source_path,
            OVERTURE_MERGE_COLS,
            test_bbox = (
                test_bbox if overture_merge_needs_test_bbox
                else None
            ),
            label = "Overture (merge cols)",
        )
        log_rss("after reload for merge")

        part_paths = build_merge_parts_chunked(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_shared_labels,
            overture_shared_labels = overture_shared_labels,
            osm_primary = chunk_summary["osm_primary"],
            overture_primary = chunk_summary["overture_primary"],
            n_chunks = chunk_summary["n_chunks"],
            overture_confidence_weight = OVERTURE_CONF_WEIGHT,
        )
    else:
        part_paths = build_merge_parts(
            osm_gdf = osm_gdf,
            overture_gdf = overture_gdf,
            matches = matches,
            osm_shared_labels = osm_shared_labels,
            overture_shared_labels = overture_shared_labels,
            overture_confidence_weight = OVERTURE_CONF_WEIGHT,
        )

    # Free ALL source data before concat+save
    del osm_gdf, overture_gdf, matches
    del osm_shared_labels, overture_shared_labels, osm_radii
    del osm_l0_bits, overture_l0_bits
    log_rss("after merge parts written")

    # -- Save ------------------------------------------------------
    print("\nSaving conflated dataset ...")
    n_total = save_conflated_from_parts(part_paths, OUTPUT_PATH)
    log_rss("after final parquet stream")
    config.write_self("conflation")

    # Clear chunk checkpoints after a successful save.
    if checkpoint_dir is not None and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    # Clear the post-dedup Overture temp parquet (kept only to back
    # the merge-phase reload).
    if (
        overture_merge_source_path != OVERTURE_PATH
        and overture_merge_source_path.exists()
    ):
        overture_merge_source_path.unlink()

    # -- Summary ---------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Conflation complete in {elapsed:.0f}s")
    print(f"  Total POIs:     {n_total:,}")
    if n_matches > 0:
        print(f"  Matched:        {n_matches:,}")
        print(
            f"  Mean match score: {match_score_mean:.3f}"
        )
        print(
            f"  Mean match distance: {match_dist_mean:.1f}m"
        )
    if chunk_summary is not None:
        print(
            f"  Chunks:         {chunk_summary['n_chunks']} "
            f"(OSM/chunk: "
            f"{chunk_summary['min_chunk_pois']:,}"
            f"–{chunk_summary['max_chunk_pois']:,})"
        )
        print(
            f"  Overture dedup drops: "
            f"{chunk_summary['n_overture_dedup_drops']:,}"
        )
    if dedup_summary is not None:
        print(
            f"  Overture internal dedup: "
            f"{dedup_summary['n_dropped']:,} dropped across "
            f"{dedup_summary['n_multi_clusters']:,} clusters"
        )
    print(f"  Output: {OUTPUT_PATH}")
