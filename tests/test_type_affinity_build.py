"""Tests for the label-change helpers in scripts/conflation/build_type_affinity.py.

- an OSM-only label takes its parent's prior row, and bad config is refused
- elements under a label new since the prior run are relabelled; others are not
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "conflation" / "build_type_affinity.py"
)
_spec = importlib.util.spec_from_file_location("build_type_affinity", PATH)
bta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bta)

PRIOR = {("Super", "Super"): 1.0, ("Super", "Cafe"): 0.4, ("Cafe", "Cafe"): 1.0}


def test_inherit_prior_copies_the_parent_row_to_the_osm_side():
    out = bta.inherit_prior(PRIOR, {"Small": "Super"})
    assert out[("Small", "Super")] == 1.0 and out[("Small", "Cafe")] == 0.4
    assert ("Super", "Small") not in out
    assert bta.inherit_prior(PRIOR, {}) == PRIOR


def test_inherit_prior_refuses_bad_config():
    with pytest.raises(ValueError, match = "has Overture categories"):
        bta.inherit_prior(PRIOR, {"Cafe": "Super"})
    with pytest.raises(ValueError, match = "no Overture category"):
        bta.inherit_prior(PRIOR, {"Small": "Nowhere"})


def test_relabel_new_only_touches_new_labels(tmp_path):
    snapshot = tmp_path / "snap.parquet"
    frame = pd.DataFrame({
        "osm_id": [1, 2, 3, 4], "osm_type": ["node", "way", "node", "node"],
        "shop": ["grocery", "bakery", None, "grocery"],
        "amenity": [None, None, "cafe", "cafe"],
    })
    con = duckdb.connect()
    frame.to_parquet(snapshot)
    crosswalk = pd.DataFrame({
        "osm_key": ["shop", "shop", "shop", "amenity"],
        "osm_value": ["grocery", "bakery", "*", "cafe"],
        "shared_label": ["Small Grocery", "Bakery", "Other Shop", "Cafe"],
    })
    radii = pd.DataFrame({
        "shared_label": ["Small Grocery", "Bakery", "Other Shop", "Cafe"],
        "match_radius_m": [50, 50, 50, 50],
    })
    out = bta.relabel_new(
        con, snapshot, ["shop", "amenity"], crosswalk, radii,
        {"Bakery", "Other Shop", "Cafe"},
    )
    assert sorted(zip(out["osm_id"], out["label"])) == [
        (1, "Small Grocery"), (4, "Small Grocery"),
    ]
    none = bta.relabel_new(
        con, snapshot, ["shop", "amenity"], crosswalk, radii,
        {"Bakery", "Other Shop", "Cafe", "Small Grocery"},
    )
    assert none.empty
