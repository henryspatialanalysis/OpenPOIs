"""Tests for openpois.conflation.match."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from openpois.conflation.match import (
    _normalize_name,
    compute_match_scores,
    compute_name_scores,
    compute_type_scores,
    find_and_score_matches_chunked,
    find_spatial_candidates,
    select_best_matches,
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def _make_geom_array(coords: list[tuple[float, float]]):
    """Create a numpy array of shapely Points from (lon, lat) pairs."""
    return np.array([Point(lon, lat) for lon, lat in coords])


# -----------------------------------------------------------------
# Spatial candidates
# -----------------------------------------------------------------


class TestFindSpatialCandidates:
    def test_nearby_points_match(self):
        """Two points ~11m apart should match at 50m radius."""
        osm_geoms = _make_geom_array([(-122.335, 47.608)])
        ov_geoms = _make_geom_array([(-122.335, 47.6081)])
        radii = np.array([50.0])

        result = find_spatial_candidates(
            osm_geoms, ov_geoms, radii, max_radius_m = 200.0,
        )
        assert len(result) == 1
        assert result.iloc[0]["osm_idx"] == 0
        assert result.iloc[0]["overture_idx"] == 0
        assert result.iloc[0]["distance_m"] < 50.0

    def test_far_points_no_match(self):
        """Two points ~1km apart should not match at 50m radius."""
        osm_geoms = _make_geom_array([(-122.335, 47.608)])
        ov_geoms = _make_geom_array([(-122.335, 47.618)])
        radii = np.array([50.0])

        result = find_spatial_candidates(
            osm_geoms, ov_geoms, radii, max_radius_m = 200.0,
        )
        assert len(result) == 0

    def test_per_poi_radius_filtering(self):
        """A point within 200m but beyond 50m should only match
        if the radius is large enough."""
        osm_geoms = _make_geom_array([
            (-122.335, 47.608),   # 50m radius
            (-122.335, 47.6088),  # 200m radius
        ])
        # This point is ~100m from both OSM points
        ov_geoms = _make_geom_array([(-122.335, 47.6089)])
        radii = np.array([50.0, 200.0])

        result = find_spatial_candidates(
            osm_geoms, ov_geoms, radii, max_radius_m = 200.0,
        )
        # Only the second OSM point (200m radius) should match
        assert len(result) >= 1
        matched_osm = set(result["osm_idx"].tolist())
        assert 1 in matched_osm

    def test_empty_input(self):
        osm_geoms = _make_geom_array([])
        ov_geoms = _make_geom_array([(-122.335, 47.608)])
        radii = np.array([], dtype = float)

        result = find_spatial_candidates(
            osm_geoms, ov_geoms, radii, max_radius_m = 200.0,
        )
        assert len(result) == 0

    def test_chunking(self):
        """Verify chunking produces same results as no chunking."""
        np.random.seed(42)
        n = 100
        osm_lons = -122.3 + np.random.randn(n) * 0.01
        osm_lats = 47.6 + np.random.randn(n) * 0.01
        ov_lons = -122.3 + np.random.randn(n) * 0.01
        ov_lats = 47.6 + np.random.randn(n) * 0.01

        osm_geoms = _make_geom_array(
            list(zip(osm_lons, osm_lats))
        )
        ov_geoms = _make_geom_array(
            list(zip(ov_lons, ov_lats))
        )
        radii = np.full(n, 200.0)

        r1 = find_spatial_candidates(
            osm_geoms, ov_geoms, radii,
            max_radius_m = 200.0, chunk_size = 10,
        )
        r2 = find_spatial_candidates(
            osm_geoms, ov_geoms, radii,
            max_radius_m = 200.0, chunk_size = 1000,
        )
        assert len(r1) == len(r2)


# -----------------------------------------------------------------
# Name scoring
# -----------------------------------------------------------------


class TestNameScoring:
    def test_identical_names(self):
        scores = compute_name_scores(
            osm_names = np.array(["Starbucks"]),
            osm_brands = np.array([None]),
            overture_names = np.array(["Starbucks"]),
            overture_brands = np.array([None]),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] == pytest.approx(1.0)

    def test_brand_vs_name(self):
        """Brand-to-name cross-compare should still score high."""
        scores = compute_name_scores(
            osm_names = np.array([None]),
            osm_brands = np.array(["Walmart"]),
            overture_names = np.array(["Walmart Supercenter"]),
            overture_brands = np.array([None]),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] > 0.8

    def test_all_null_neutral(self):
        """All null names/brands should give neutral 0.5."""
        scores = compute_name_scores(
            osm_names = np.array([None]),
            osm_brands = np.array([None]),
            overture_names = np.array([None]),
            overture_brands = np.array([None]),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] == pytest.approx(0.5)

    def test_completely_different(self):
        scores = compute_name_scores(
            osm_names = np.array(["Alpha Restaurant"]),
            osm_brands = np.array([None]),
            overture_names = np.array(["Zephyr Grocery"]),
            overture_brands = np.array([None]),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] < 0.5


class TestNormalizeName:
    def test_basic(self):
        assert _normalize_name("  Hello  World ") == "hello world"

    def test_none(self):
        assert _normalize_name(None) == ""

    def test_nan(self):
        assert _normalize_name(float("nan")) == ""


# -----------------------------------------------------------------
# Type taxonomy scoring
# -----------------------------------------------------------------


class TestTypeScoring:
    def test_exact_shared_label(self):
        scores = compute_type_scores(
            osm_shared_labels = np.array(["Restaurant"]),
            overture_shared_labels = np.array(["Restaurant"]),
            osm_l0_bits = np.array([3], dtype = np.uint16),
            overture_l0_bits = np.array([2], dtype = np.uint16),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] == 1.0

    def test_l0_overlap(self):
        """Different labels, but L0 bits overlap -> 0.5."""
        scores = compute_type_scores(
            osm_shared_labels = np.array(["Restaurant"]),
            overture_shared_labels = np.array(["Cafe"]),
            osm_l0_bits = np.array([3], dtype = np.uint16),
            overture_l0_bits = np.array([2], dtype = np.uint16),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] == 0.5

    def test_no_l0_overlap(self):
        """Different labels, no L0 overlap -> 0.0."""
        scores = compute_type_scores(
            osm_shared_labels = np.array(["Restaurant"]),
            overture_shared_labels = np.array(["Park"]),
            osm_l0_bits = np.array([3], dtype = np.uint16),
            overture_l0_bits = np.array([16], dtype = np.uint16),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] == 0.0

    def test_unmapped_zero(self):
        """Empty label + zero bits -> 0.0."""
        scores = compute_type_scores(
            osm_shared_labels = np.array([""]),
            overture_shared_labels = np.array(["Restaurant"]),
            osm_l0_bits = np.array([0], dtype = np.uint16),
            overture_l0_bits = np.array([2], dtype = np.uint16),
            osm_idx = np.array([0]),
            overture_idx = np.array([0]),
        )
        assert scores[0] == 0.0


# -----------------------------------------------------------------
# Composite scoring and selection
# -----------------------------------------------------------------


class TestSelectBestMatches:
    def test_greedy_one_to_one(self):
        scored = pd.DataFrame(
            {
                "osm_idx": [0, 0, 1],
                "overture_idx": [0, 1, 0],
                "distance_m": [10, 20, 15],
                "composite_score": [0.9, 0.7, 0.8],
            }
        )
        result = select_best_matches(scored, min_score = 0.67)
        # OSM 0 -> Overture 0 (score 0.9, highest)
        # OSM 1 -> Overture 0 is blocked, so OSM 1 unmatched
        # OSM 0 -> Overture 1 is blocked, so only 1 match
        assert len(result) == 1
        assert result.iloc[0]["osm_idx"] == 0
        assert result.iloc[0]["overture_idx"] == 0

    def test_min_score_filter(self):
        scored = pd.DataFrame(
            {
                "osm_idx": [0, 1],
                "overture_idx": [0, 1],
                "distance_m": [10, 10],
                "composite_score": [0.8, 0.5],
            }
        )
        result = select_best_matches(scored, min_score = 0.67)
        assert len(result) == 1

    def test_empty_input(self):
        scored = pd.DataFrame(
            columns = [
                "osm_idx", "overture_idx",
                "distance_m", "composite_score",
            ]
        )
        result = select_best_matches(scored, min_score = 0.67)
        assert len(result) == 0

    def test_exact_tie_breaks_on_source_id_order(self):
        # Two OSM POIs tie on score and distance for one Overture POI. The
        # winner follows source-id order, not row order.
        scored = pd.DataFrame(
            {
                "osm_idx": [0, 1],
                "overture_idx": [0, 0],
                "distance_m": [12.0, 12.0],
                "composite_score": [0.85, 0.85],
            }
        )
        osm_order = np.array([1, 0])  # OSM 1 has the smaller source id
        for frame in (scored, scored.iloc[::-1]):
            result = select_best_matches(
                frame, min_score = 0.67, osm_order = osm_order,
            )
            assert result["osm_idx"].tolist() == [1]

    def test_distance_breaks_score_ties(self):
        scored = pd.DataFrame(
            {
                "osm_idx": [0, 1],
                "overture_idx": [0, 0],
                "distance_m": [30.0, 5.0],
                "composite_score": [0.85, 0.85],
            }
        )
        result = select_best_matches(scored, min_score = 0.67)
        assert result["osm_idx"].tolist() == [1]


# -----------------------------------------------------------------
# Chunked driver
# -----------------------------------------------------------------


def _build_synthetic_datasets(
    n_clusters: int = 10,
    poi_per_cluster: int = 5,
    cluster_spacing_deg: float = 0.05,
    center_lon: float = -122.3,
    center_lat: float = 47.6,
    seed: int = 0,
):
    """
    Build OSM + Overture POI arrays arranged in well-separated
    clusters. Each cluster holds ``poi_per_cluster`` OSM and
    ``poi_per_cluster`` Overture points with identical coords (so
    matching is deterministic), unique-ish names so scoring is
    decisive, and clusters far enough apart (~5 km at ``0.05°``) that
    no chunk boundary should bisect a cluster.
    """
    rng = np.random.default_rng(seed)
    osm_geoms = []
    ov_geoms = []
    osm_names = []
    ov_names = []
    osm_radii = []

    for ci in range(n_clusters):
        base_lon = center_lon + ci * cluster_spacing_deg
        base_lat = center_lat
        for pi in range(poi_per_cluster):
            # Tiny jitter within a cluster (~1 m)
            lon = base_lon + rng.uniform(-1e-5, 1e-5)
            lat = base_lat + rng.uniform(-1e-5, 1e-5)
            osm_geoms.append(Point(lon, lat))
            ov_geoms.append(Point(lon, lat))
            name = f"Cluster{ci}_POI{pi}"
            osm_names.append(name)
            ov_names.append(name)
            osm_radii.append(200.0)

    osm_geom_arr = np.array(osm_geoms)
    ov_geom_arr = np.array(ov_geoms)
    osm_names_arr = np.array(osm_names, dtype = object)
    ov_names_arr = np.array(ov_names, dtype = object)
    none_brand = np.full(len(osm_geoms), None, dtype = object)
    empty_label = np.full(len(osm_geoms), "", dtype = object)
    l0_bits = np.zeros(len(osm_geoms), dtype = np.uint16)
    return {
        "osm_geom": osm_geom_arr,
        "overture_geom": ov_geom_arr,
        "osm_names": osm_names_arr,
        "overture_names": ov_names_arr,
        "osm_brands": none_brand,
        "overture_brands": none_brand.copy(),
        "osm_shared_labels": empty_label,
        "overture_shared_labels": empty_label.copy(),
        "osm_l0_bits": l0_bits,
        "overture_l0_bits": l0_bits.copy(),
        "osm_radii_m": np.array(osm_radii, dtype = np.float64),
    }


class TestChunkedDriver:
    def _run_nonchunked(self, data, **kwargs):
        candidates = find_spatial_candidates(
            osm_geom = data["osm_geom"],
            overture_geom = data["overture_geom"],
            osm_radii_m = data["osm_radii_m"],
            max_radius_m = kwargs["max_radius_m"],
        )
        scored = compute_match_scores(
            candidates = candidates,
            osm_names = data["osm_names"],
            osm_brands = data["osm_brands"],
            overture_names = data["overture_names"],
            overture_brands = data["overture_brands"],
            osm_shared_labels = data["osm_shared_labels"],
            overture_shared_labels = data[
                "overture_shared_labels"
            ],
            osm_radii_m = data["osm_radii_m"],
            osm_l0_bits = data["osm_l0_bits"],
            overture_l0_bits = data["overture_l0_bits"],
            distance_weight = kwargs["distance_weight"],
            name_weight = kwargs["name_weight"],
            type_weight = kwargs["type_weight"],
            identifier_weight = kwargs["identifier_weight"],
        )
        return select_best_matches(
            scored, min_score = kwargs["min_match_score"],
        )

    def test_matches_nonchunked_on_separated_clusters(self):
        """Clusters spaced far apart so no chunk boundary bisects
        one — chunked and non-chunked must produce identical matched
        sets (same (osm_idx, overture_idx) pairs)."""
        data = _build_synthetic_datasets(
            n_clusters = 10,
            poi_per_cluster = 5,
            cluster_spacing_deg = 0.05,
            seed = 42,
        )
        kwargs = dict(
            distance_weight = 0.0,
            name_weight = 0.50,
            type_weight = 0.30,
            identifier_weight = 0.20,
            min_match_score = 0.50,
            max_radius_m = 200.0,
        )

        ref = self._run_nonchunked(data, **kwargs)

        chunked, summary = find_and_score_matches_chunked(
            osm_geom = data["osm_geom"],
            overture_geom = data["overture_geom"],
            osm_radii_m = data["osm_radii_m"],
            osm_shared_labels = data["osm_shared_labels"],
            overture_shared_labels = data[
                "overture_shared_labels"
            ],
            osm_l0_bits = data["osm_l0_bits"],
            overture_l0_bits = data["overture_l0_bits"],
            osm_names = data["osm_names"],
            osm_brands = data["osm_brands"],
            overture_names = data["overture_names"],
            overture_brands = data["overture_brands"],
            chunk_target_pois = 20,
            **kwargs,
        )

        ref_pairs = set(
            zip(
                ref["osm_idx"].astype(int).tolist(),
                ref["overture_idx"].astype(int).tolist(),
            )
        )
        chunked_pairs = set(
            zip(
                chunked["osm_idx"].astype(int).tolist(),
                chunked["overture_idx"].astype(int).tolist(),
            )
        )
        assert ref_pairs == chunked_pairs
        assert summary["n_chunks"] > 1

    def test_single_chunk_when_small(self):
        """N well below target: collapses to one chunk; output equals
        non-chunked."""
        data = _build_synthetic_datasets(
            n_clusters = 4, poi_per_cluster = 3, seed = 7,
        )
        kwargs = dict(
            distance_weight = 0.0,
            name_weight = 0.50,
            type_weight = 0.30,
            identifier_weight = 0.20,
            min_match_score = 0.50,
            max_radius_m = 200.0,
        )
        ref = self._run_nonchunked(data, **kwargs)
        chunked, summary = find_and_score_matches_chunked(
            osm_geom = data["osm_geom"],
            overture_geom = data["overture_geom"],
            osm_radii_m = data["osm_radii_m"],
            osm_shared_labels = data["osm_shared_labels"],
            overture_shared_labels = data[
                "overture_shared_labels"
            ],
            osm_l0_bits = data["osm_l0_bits"],
            overture_l0_bits = data["overture_l0_bits"],
            osm_names = data["osm_names"],
            osm_brands = data["osm_brands"],
            overture_names = data["overture_names"],
            overture_brands = data["overture_brands"],
            chunk_target_pois = 10_000,
            **kwargs,
        )
        assert summary["n_chunks"] == 1
        assert len(ref) == len(chunked)

    def test_checkpoint_resume(self, tmp_path):
        """Second run picks up per-chunk parquets from disk."""
        data = _build_synthetic_datasets(
            n_clusters = 6, poi_per_cluster = 4, seed = 3,
        )
        kwargs = dict(
            distance_weight = 0.0,
            name_weight = 0.50,
            type_weight = 0.30,
            identifier_weight = 0.20,
            min_match_score = 0.50,
            max_radius_m = 200.0,
        )
        first, _ = find_and_score_matches_chunked(
            osm_geom = data["osm_geom"],
            overture_geom = data["overture_geom"],
            osm_radii_m = data["osm_radii_m"],
            osm_shared_labels = data["osm_shared_labels"],
            overture_shared_labels = data[
                "overture_shared_labels"
            ],
            osm_l0_bits = data["osm_l0_bits"],
            overture_l0_bits = data["overture_l0_bits"],
            osm_names = data["osm_names"],
            osm_brands = data["osm_brands"],
            overture_names = data["overture_names"],
            overture_brands = data["overture_brands"],
            chunk_target_pois = 10,
            checkpoint_dir = tmp_path,
            **kwargs,
        )
        # Part files should have been written
        parts = sorted(tmp_path.glob("chunk_*.parquet"))
        assert len(parts) >= 2

        # Second run should reload rather than recompute and match
        second, _ = find_and_score_matches_chunked(
            osm_geom = data["osm_geom"],
            overture_geom = data["overture_geom"],
            osm_radii_m = data["osm_radii_m"],
            osm_shared_labels = data["osm_shared_labels"],
            overture_shared_labels = data[
                "overture_shared_labels"
            ],
            osm_l0_bits = data["osm_l0_bits"],
            overture_l0_bits = data["overture_l0_bits"],
            osm_names = data["osm_names"],
            osm_brands = data["osm_brands"],
            overture_names = data["overture_names"],
            overture_brands = data["overture_brands"],
            chunk_target_pois = 10,
            checkpoint_dir = tmp_path,
            **kwargs,
        )
        # Match sets identical
        first_pairs = set(
            zip(
                first["osm_idx"].astype(int).tolist(),
                first["overture_idx"].astype(int).tolist(),
            )
        )
        second_pairs = set(
            zip(
                second["osm_idx"].astype(int).tolist(),
                second["overture_idx"].astype(int).tolist(),
            )
        )
        assert first_pairs == second_pairs
