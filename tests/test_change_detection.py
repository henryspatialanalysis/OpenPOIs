"""Tests for openpois.conflation.change_detection (same-entity rule).

Synthetic pairs only: a ghost and an Overture row 3 m apart, both ``Cafe``,
scored with the production weights (0.20 distance / 0.40 name / 0.40 type,
``min_shadow_match_score`` 0.70, ``min_prior_name_match_score`` 70). The
rule under test (2026-09-24): demote an Overture-only POI only when OSM
closed the *same* business, with the six guards from
``.claude/plans/location-report-propagation.md``.
"""
from __future__ import annotations

from math import cos, radians

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from openpois.conflation.change_detection import (
    apply_current_survivor_filter,
    apply_shadow_match,
    filter_ghosts_by_age,
    find_shadow_matches,
)


LON, LAT = -122.335, 47.608
M_PER_DEG_LAT = 111_320.0
NOW = pd.Timestamp("2026-09-24", tz = "UTC")

PRODUCTION = dict(
    min_match_score = 0.70,
    max_radius_m = 200.0,
    default_radius_m = 100.0,
    distance_weight = 0.20,
    name_weight = 0.40,
    type_weight = 0.40,
    identifier_weight = 0.0,
    min_prior_name_match_score = 70.0,
)


def _pt(dx_m: float = 0.0, dy_m: float = 0.0) -> Point:
    """A point offset from (LON, LAT) by metres east / north."""
    return Point(
        LON + dx_m / (M_PER_DEG_LAT * cos(radians(LAT))),
        LAT + dy_m / M_PER_DEG_LAT,
    )


def _ghosts(rows: list[dict]) -> gpd.GeoDataFrame:
    defaults = {
        "event_type": "hard_delete",
        "event_timestamp": NOW - pd.Timedelta(days = 365),
        "prior_name": None,
        "prior_brand": None,
        "new_name": None,
        "shared_label": "Cafe",
        "geometry": _pt(),
    }
    full = []
    for i, row in enumerate(rows):
        r = {**defaults, **row}
        r.setdefault("ghost_id", f"node/{i + 1}/v2:{r['event_type']}")
        full.append(r)
    df = pd.DataFrame(full)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc = True)
    return gpd.GeoDataFrame(df, geometry = "geometry", crs = "EPSG:4326")


def _overture(rows: list[dict]) -> gpd.GeoDataFrame:
    defaults = {
        "name": None,
        "brand": None,
        "shared_label": "Cafe",
        "geometry": _pt(3.0, 0.0),
    }
    df = pd.DataFrame([{**defaults, **row} for row in rows])
    return gpd.GeoDataFrame(df, geometry = "geometry", crs = "EPSG:4326")


def _match(ov: gpd.GeoDataFrame, gh: gpd.GeoDataFrame, **overrides):
    return find_shadow_matches(ov, gh, **{**PRODUCTION, **overrides})


# --- Same-entity name gate --------------------------------------------------

@pytest.mark.parametrize(
    "ghost_name, ghost_brand, ov_name, ov_brand, expect",
    [
        ("Joe's Cafe", None, "Joe's Cafe", None, True),      # same name
        ("Walgreens", None, "Walgreens Pharmacy", None, True),  # subset
        ("Starbucks", None, "Pike Place Roastery", "Starbucks", True),  # brand
        ("Joe's Coffee House", None, "Joe's Cafe", None, False),  # different
        (None, None, "Joe's Cafe", None, False),               # unnamed ghost
        ("Random Books", None, "Joe's Cafe", None, False),     # unrelated
        ("La Madelaine Bakery", None, "la Madeleine", None, True),  # accents
    ],
    ids = [
        "same-name", "walgreens-subset", "starbucks-brand-only",
        "joes-coffee-house-vs-cafe", "unnamed-ghost", "unrelated",
        "la-madeleine-normalised",
    ],
)
def test_name_gate_decides_the_match(
    ghost_name, ghost_brand, ov_name, ov_brand, expect,
):
    gh = _ghosts([{"prior_name": ghost_name, "prior_brand": ghost_brand}])
    ov = _overture([{"name": ov_name, "brand": ov_brand}])
    matches = _match(ov, gh)
    assert (len(matches) == 1) is expect
    if expect:
        assert matches.iloc[0]["composite_score"] >= 0.70
        assert matches.iloc[0]["distance_m"] < 5


def test_identical_names_are_no_longer_dropped_as_same_entity():
    """The pre-2026-09-24 second-stage subset/superset drop is gone: an
    identical name is now the strongest reason to demote, not a reason to
    skip."""
    gh = _ghosts([{"prior_name": "Blue Harbor Bank"}])
    ov = _overture([{"name": "Blue Harbor Bank"}])
    assert len(_match(ov, gh)) == 1


def test_unnamed_overture_row_never_matches_even_on_brand():
    gh = _ghosts([{"prior_name": "Starbucks", "prior_brand": "Starbucks"}])
    ov = _overture([{"name": None, "brand": "Starbucks"}])
    assert _match(ov, gh).empty


def test_unnamed_ghost_with_brand_never_matches():
    gh = _ghosts([{"prior_name": None, "prior_brand": "Starbucks"}])
    ov = _overture([{"name": "Starbucks"}])
    assert _match(ov, gh).empty


def test_shared_generic_token_alone_is_not_a_match():
    """"Plaza Pharmacy" / "CVS Pharmacy" score 80 on the raw token-set ratio
    purely through "pharmacy"; the category-token rule strips it."""
    gh = _ghosts([{"prior_name": "Plaza Pharmacy", "shared_label": "Pharmacy"}])
    ov = _overture([{"name": "CVS Pharmacy", "shared_label": "Pharmacy"}])
    assert _match(ov, gh).empty


def test_type_mismatch_still_blocks_a_same_name_pair():
    gh = _ghosts([{"prior_name": "Joe's Cafe", "shared_label": "Bank"}])
    ov = _overture([{"name": "Joe's Cafe"}])
    assert _match(ov, gh).empty


# --- Rename direction -------------------------------------------------------

def _rename_ghost():
    return _ghosts([{
        "event_type": "substantial_rename",
        "prior_name": "Plaza Pharmacy",
        "new_name": "CVS Pharmacy",
        "shared_label": "Pharmacy",
    }])


def test_rename_already_carried_by_overture_is_skipped():
    ov = _overture([{"name": "CVS Pharmacy", "shared_label": "Pharmacy"}])
    assert _match(ov, _rename_ghost()).empty


def test_rename_demotes_when_overture_still_shows_the_prior_name():
    ov = _overture([{"name": "Plaza Pharmacy", "shared_label": "Pharmacy"}])
    assert len(_match(ov, _rename_ghost())) == 1


def test_rename_guard_only_applies_to_substantial_rename_ghosts():
    """A lifecycle ghost keeps its name, so new_name == prior_name; the
    guard must not turn every named lifecycle ghost into a skip."""
    gh = _ghosts([{
        "event_type": "lifecycle_prefix_added",
        "prior_name": "Joe's Cafe",
        "new_name": "Joe's Cafe",
    }])
    ov = _overture([{"name": "Joe's Cafe"}])
    assert len(_match(ov, gh)) == 1


def test_ghosts_without_new_name_column_still_match():
    gh = _ghosts([{"prior_name": "Joe's Cafe"}]).drop(columns = ["new_name"])
    ov = _overture([{"name": "Joe's Cafe"}])
    assert len(_match(ov, gh)) == 1


# --- Current-OSM-survivor filter -------------------------------------------

def _snapshot(tmp_path, name = "Blue Harbor Bank", offset_m = 120.0):
    """A full-snapshot GeoParquet with one named building way whose
    centroid is ``offset_m`` north of the Overture point, plus an unnamed
    node."""
    c = _pt(0.0, offset_m)
    dlat = 0.0001
    dlon = 0.0002
    way = box(c.x - dlon, c.y - dlat, c.x + dlon, c.y + dlat)
    gdf = gpd.GeoDataFrame(
        {
            "source": ["osm", "osm"],
            "osm_id": [10, 11],
            "osm_type": ["way", "node"],
            "name": [name, None],
        },
        geometry = [way, _pt(500.0, 0.0)],
        crs = "EPSG:4326",
    )
    path = tmp_path / "osm_snapshot.parquet"
    gdf.to_parquet(path)
    return path


def _one_match():
    return pd.DataFrame({
        "osm_idx": [0], "overture_idx": [0],
        "composite_score": [0.95], "distance_m": [3.0],
    })


def test_survivor_way_centroid_120m_away_suppresses_at_150m(tmp_path):
    ov = _overture([{"name": "Blue Harbor Bank", "shared_label": "Bank"}])
    kept, n_dropped = apply_current_survivor_filter(
        _one_match(), ov,
        snapshot_path = _snapshot(tmp_path),
        radius_m = 150.0, name_similarity_threshold = 70.0,
        verbose = False,
    )
    assert n_dropped == 1
    assert kept.empty


def test_survivor_way_centroid_120m_away_is_missed_at_50m(tmp_path):
    """The pre-2026-09-24 radius: documents why it was widened."""
    ov = _overture([{"name": "Blue Harbor Bank", "shared_label": "Bank"}])
    kept, n_dropped = apply_current_survivor_filter(
        _one_match(), ov,
        snapshot_path = _snapshot(tmp_path),
        radius_m = 50.0, name_similarity_threshold = 70.0,
        verbose = False,
    )
    assert n_dropped == 0
    assert len(kept) == 1


def test_survivor_names_are_normalised_before_comparison(tmp_path):
    ov = _overture([{"name": "Blue Harbor Bank, Inc.", "shared_label": "Bank"}])
    kept, n_dropped = apply_current_survivor_filter(
        _one_match(), ov,
        snapshot_path = _snapshot(tmp_path, name = "Blue Harbór Bank LLC"),
        radius_m = 150.0, name_similarity_threshold = 70.0,
        verbose = False,
    )
    assert n_dropped == 1


def test_survivor_with_a_different_name_does_not_suppress(tmp_path):
    ov = _overture([{"name": "Blue Harbor Bank", "shared_label": "Bank"}])
    kept, n_dropped = apply_current_survivor_filter(
        _one_match(), ov,
        snapshot_path = _snapshot(tmp_path, name = "Mendocino Masonic Hall"),
        radius_m = 150.0, name_similarity_threshold = 70.0,
        verbose = False,
    )
    assert n_dropped == 0


# --- Ghost age --------------------------------------------------------------

def test_filter_ghosts_by_age_drops_a_four_year_old_ghost():
    gh = _ghosts([
        {"prior_name": "Old", "event_timestamp": NOW - pd.Timedelta(days = 4 * 365)},
        {"prior_name": "Recent", "event_timestamp": NOW - pd.Timedelta(days = 365)},
        {"prior_name": "Undated", "event_timestamp": pd.NaT},
    ])
    kept = filter_ghosts_by_age(gh, 3, now = NOW)
    assert list(kept["prior_name"]) == ["Recent", "Undated"]
    # Disabled filter keeps everything.
    assert len(filter_ghosts_by_age(gh, None, now = NOW)) == 3
    assert len(filter_ghosts_by_age(gh, 0, now = NOW)) == 3


# --- End to end through apply_shadow_match ---------------------------------

def _baseline(tmp_path, rows: list[dict]):
    defaults = {
        "unified_id": None, "source": "overture", "shared_label": "Cafe",
        "name": None, "brand": None,
        "conf_mean": 0.8, "conf_lower": 0.6, "conf_upper": 0.9,
        "geometry": _pt(3.0, 0.0),
    }
    full = []
    for i, row in enumerate(rows):
        r = {**defaults, **row}
        r["unified_id"] = r["unified_id"] or f"ov{i}"
        full.append(r)
    gdf = gpd.GeoDataFrame(pd.DataFrame(full), geometry = "geometry",
                           crs = "EPSG:4326")
    path = tmp_path / "conflated_baseline.parquet"
    gdf.to_parquet(path)
    return path


def _run(tmp_path, ghosts, baseline_rows, **overrides):
    ghosts_path = tmp_path / "ghosts.parquet"
    ghosts.to_parquet(ghosts_path)
    params = tmp_path / "fitted_params.csv"
    params.write_text("param_name,mean\nlambda_0,0.1\n")
    out = tmp_path / "conflated_cd.parquet"
    kwargs = {
        **{k: v for k, v in PRODUCTION.items()},
        "default_delta": 0.141,
        "survivor_filter": {"enabled": False},
        "verbose": False,
    }
    kwargs.update(overrides)
    summary = apply_shadow_match(
        _baseline(tmp_path, baseline_rows), ghosts_path, params, out, **kwargs,
    )
    return summary, gpd.read_parquet(out)


def test_apply_shadow_match_demotes_a_recent_same_name_ghost(tmp_path):
    gh = _ghosts([{"prior_name": "Joe's Cafe"}])
    summary, out = _run(
        tmp_path, gh, [{"name": "Joe's Cafe"}], max_ghost_age_years = 3,
    )
    assert summary["n_shadow_matches"] == 1
    assert bool(out["shadow_matched"].iloc[0])
    assert out["conf_mean"].iloc[0] == pytest.approx(0.8 * 0.141)
    assert out["shadow_event_type"].iloc[0] == "hard_delete"


def test_apply_shadow_match_drops_a_four_year_old_ghost(tmp_path):
    gh = _ghosts([{
        "prior_name": "Joe's Cafe",
        "event_timestamp": pd.Timestamp.now(tz = "UTC") - pd.Timedelta(days = 4 * 365),
    }])
    summary, out = _run(
        tmp_path, gh, [{"name": "Joe's Cafe"}], max_ghost_age_years = 3,
    )
    assert summary["n_ghosts"] == 0
    assert summary["n_shadow_matches"] == 0
    assert not out["shadow_matched"].iloc[0]
    assert out["conf_mean"].iloc[0] == pytest.approx(0.8)


def test_apply_shadow_match_uses_full_snapshot_for_survivors(tmp_path):
    gh = _ghosts([{"prior_name": "Blue Harbor Bank", "shared_label": "Bank"}])
    summary, out = _run(
        tmp_path, gh,
        [{"name": "Blue Harbor Bank", "shared_label": "Bank"}],
        full_snapshot_path = _snapshot(tmp_path),
        rated_snapshot_path = None,
        survivor_filter = {
            "enabled": True, "radius_m": 150, "name_similarity_threshold": 70,
            "use_full_snapshot": True,
        },
    )
    assert summary["n_survivor_dropped"] == 1
    assert summary["n_shadow_matches"] == 0
    assert out["conf_mean"].iloc[0] == pytest.approx(0.8)


def test_apply_shadow_match_falls_back_to_rated_snapshot(tmp_path):
    """use_full_snapshot false → the rated snapshot is the candidate set."""
    gh = _ghosts([{"prior_name": "Blue Harbor Bank", "shared_label": "Bank"}])
    rated = _snapshot(tmp_path)
    summary, _ = _run(
        tmp_path, gh,
        [{"name": "Blue Harbor Bank", "shared_label": "Bank"}],
        full_snapshot_path = None,
        rated_snapshot_path = rated,
        survivor_filter = {
            "enabled": True, "radius_m": 150, "name_similarity_threshold": 70,
            "use_full_snapshot": False,
        },
    )
    assert summary["n_survivor_dropped"] == 1
