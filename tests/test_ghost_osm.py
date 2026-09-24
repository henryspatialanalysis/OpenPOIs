"""Tests for openpois.conflation.ghost_osm: name normalisation and the
ghost event rules (named lifecycle ghosts, ``new_name``)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpois.conflation.ghost_osm import (
    _scan_all_changes,
    names_match,
    normalise_name,
    strip_trailing_category_token,
)


# --- normalise_name ---------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Wapa Café Boutique", "wapa cafe boutique"),
        ("La Madelaine Bakery", "la madelaine bakery"),
        ("The Hair Co.", "the hair"),            # co. → company → dropped
        ("Hair Company", "hair"),
        ("Superchef Brands Llc", "superchef brands"),
        ("Acme, Inc.", "acme"),
        ("Main St. Market", "main st market"),   # st. left alone
        ("Company", "company"),                  # never empties a name
        ("  ", ""),
        (None, ""),
        (np.nan, ""),
    ],
)
def test_normalise_name(raw, expected):
    assert normalise_name(raw) == expected


def test_strip_trailing_category_token():
    assert strip_trailing_category_token("walgreens pharmacy") == "walgreens"
    assert strip_trailing_category_token("pharmacy") == "pharmacy"
    assert strip_trailing_category_token("joe s coffee house") == "joe s coffee house"


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("Joe's Cafe", "Joe's Cafe", True),
        ("Walgreens", "Walgreens Pharmacy", True),
        ("Joe's Coffee House", "Joe's Cafe", False),
        ("La Madelaine Bakery", "la Madeleine", True),
        ("WAPA cafe", "Wapa Café Boutique", True),
        ("The Hair Co.", "Hair Company", True),
        ("Plaza Pharmacy", "CVS Pharmacy", False),
        ("Kangam Cafe", "Kangnam Cafe", True),
        ("First Bank", "Chase Bank", False),
        ("Blue Harbor Bank", "Blue Harbor", True),
        (None, "Anything", False),
        ("", "", False),
    ],
)
def test_names_match(a, b, expected):
    assert names_match(a, b, 70) is expected


# --- ghost events -----------------------------------------------------------

POI_KEYS = frozenset({"amenity", "shop"})


def _changes(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns = ["id", "version", "key", "value", "change"])
    return df.sort_values(["id", "version", "key"]).reset_index(drop = True)


def _v1(node_id: int, name: str | None, tag = ("amenity", "cafe")):
    rows = [
        (node_id, 1, tag[0], tag[1], "Added"),
        (node_id, 1, "lat", "47.608", "Added"),
        (node_id, 1, "lon", "-122.335", "Added"),
        (node_id, 1, "visible", "true", "Added"),
    ]
    if name:
        rows.append((node_id, 1, "name", name, "Added"))
    return rows


def test_named_disused_retag_emits_lifecycle_ghost_with_new_name():
    changes = _changes(
        _v1(1, "Joe's Cafe") + [
            (1, 2, "amenity", "cafe", "Deleted"),
            (1, 2, "disused:amenity", "cafe", "Added"),
        ]
    )
    ghosts = _scan_all_changes(changes, POI_KEYS, 50.0)
    assert len(ghosts) == 1
    g = ghosts[0]
    assert g["event_type"] == "lifecycle_prefix_added"
    assert g["prior_name"] == "Joe's Cafe"
    assert g["new_name"] == "Joe's Cafe"
    assert g["osm_version_after"] == 2
    assert g["amenity"] == "cafe"


def test_named_primary_tag_deletion_emits_a_ghost():
    changes = _changes(
        _v1(2, "Joe's Cafe") + [(2, 2, "amenity", "cafe", "Deleted")]
    )
    ghosts = _scan_all_changes(changes, POI_KEYS, 50.0)
    assert [g["event_type"] for g in ghosts] == ["primary_tag_deleted"]
    assert ghosts[0]["prior_name"] == "Joe's Cafe"


def test_unnamed_lifecycle_ghost_still_emitted():
    changes = _changes(
        _v1(3, None) + [(3, 2, "disused:amenity", "cafe", "Added")]
    )
    ghosts = _scan_all_changes(changes, POI_KEYS, 50.0)
    assert [g["event_type"] for g in ghosts] == ["lifecycle_prefix_added"]
    assert ghosts[0]["prior_name"] is None


def test_substantial_rename_records_prior_and_new_name():
    changes = _changes(
        _v1(4, "Kangam Sushi") + [(4, 2, "name", "Bella Pizza", "Changed")]
    )
    ghosts = _scan_all_changes(changes, POI_KEYS, 50.0)
    assert len(ghosts) == 1
    assert ghosts[0]["event_type"] == "substantial_rename"
    assert ghosts[0]["prior_name"] == "Kangam Sushi"
    assert ghosts[0]["new_name"] == "Bella Pizza"


def test_minor_rename_is_not_a_ghost():
    changes = _changes(
        _v1(5, "Walgreens") + [(5, 2, "name", "Walgreens Pharmacy", "Changed")]
    )
    assert _scan_all_changes(changes, POI_KEYS, 50.0) == []


def test_hard_delete_has_no_new_name():
    changes = _changes(
        _v1(6, "Joe's Cafe") + [
            (6, 2, "amenity", "cafe", "Deleted"),
            (6, 2, "name", "Joe's Cafe", "Deleted"),
            (6, 2, "visible", "false", "Changed"),
        ]
    )
    ghosts = _scan_all_changes(changes, POI_KEYS, 50.0)
    assert [g["event_type"] for g in ghosts] == ["hard_delete"]
    assert ghosts[0]["new_name"] is None
