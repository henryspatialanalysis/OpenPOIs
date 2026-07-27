"""Tests for the type-affinity and identifier components of match scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openpois.conflation.match import (
    compute_identifier_scores,
    compute_match_scores,
    compute_type_scores,
    normalize_identifier_array,
    normalize_phone,
    normalize_website,
    normalize_wikidata,
)
from openpois.conflation.taxonomy import build_affinity_matrix


@pytest.fixture
def mini_affinity():
    """Cafe/Restaurant related; Cafe/Dentist unrelated."""
    return pd.DataFrame(
        {
            "osm_label": ["Cafe", "Cafe", "Restaurant", "Tire Store"],
            "overture_label": [
                "Cafe", "Restaurant", "Restaurant", "Car Repair",
            ],
            "affinity": ["1.0", "0.6", "1.0", "0.979"],
            "s_lin": ["1.0", "0.6", "1.0", "0.648"],
            "s_emp": ["1.0", "0.0", "1.0", "1.0"],
            "n_confirmed": ["10", "0", "10", "1556"],
        }
    )


class TestAffinityMatrix:
    def test_identity_forced_to_one(self, mini_affinity):
        idx, mat = build_affinity_matrix(mini_affinity)
        for label in idx:
            assert mat[idx[label], idx[label]] == pytest.approx(1.0)

    def test_asymmetry_preserved(self, mini_affinity):
        idx, mat = build_affinity_matrix(mini_affinity)
        # Cafe -> Restaurant is populated; the reverse is not.
        assert mat[idx["Cafe"], idx["Restaurant"]] == pytest.approx(0.6)
        assert mat[idx["Restaurant"], idx["Cafe"]] == pytest.approx(0.0)

    def test_empty_table_is_safe(self):
        idx, mat = build_affinity_matrix(
            pd.DataFrame(
                {c: pd.Series(dtype = str) for c in
                 ["osm_label", "overture_label", "affinity"]}
            )
        )
        assert idx == {}
        assert mat.shape == (0, 0)


class TestTypeScores:
    def _score(self, osm, ovt, affinity = None):
        idx, mat = (
            build_affinity_matrix(affinity)
            if affinity is not None else ({}, None)
        )
        return compute_type_scores(
            np.array(osm, dtype = object), np.array(ovt, dtype = object),
            np.zeros(len(osm), dtype = np.uint16),
            np.zeros(len(ovt), dtype = np.uint16),
            np.arange(len(osm)), np.arange(len(ovt)),
            affinity_index = idx or None, affinity_matrix = mat,
        )

    def test_uses_affinity_values(self, mini_affinity):
        s = self._score(
            ["Cafe", "Cafe", "Tire Store"],
            ["Restaurant", "Cafe", "Car Repair"],
            mini_affinity,
        )
        assert s[0] == pytest.approx(0.6)
        assert s[1] == pytest.approx(1.0)
        assert s[2] == pytest.approx(0.979)

    def test_unrelated_pair_scores_zero(self, mini_affinity):
        s = self._score(["Cafe"], ["Dentist"], mini_affinity)
        assert s[0] == pytest.approx(0.0)

    def test_label_absent_from_table_still_matches_itself(
        self, mini_affinity,
    ):
        # Car Rental is OSM-only and has no Overture row anywhere.
        s = self._score(["Car Rental"], ["Car Rental"], mini_affinity)
        assert s[0] == pytest.approx(1.0)

    def test_legacy_fallback_without_table(self):
        s = self._score(["Cafe", "Cafe"], ["Cafe", "Dentist"])
        assert s[0] == pytest.approx(1.0)
        assert s[1] == pytest.approx(0.0)

    def test_empty_label_scores_zero(self, mini_affinity):
        s = self._score([""], ["Cafe"], mini_affinity)
        assert s[0] == pytest.approx(0.0)


class TestIdentifierNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://www.Example.com/", "example.com"),
            ("http://example.com", "example.com"),
            ("EXAMPLE.COM", "example.com"),
            # Campaign-tagged URLs must reduce to the same host as a plain
            # one: Overture ships these on tens of thousands of rows.
            ("https://westernunion.com/?utm_source=bingmaps", "westernunion.com"),
            ("http://www.example.com/store#hours", "example.com/store"),
            ("https://example.com/a/b/", "example.com/a/b"),
            (None, ""),
        ],
    )
    def test_website(self, raw, expected):
        assert normalize_website(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("+1 (206) 555-0143", "2065550143"),
            ("206-555-0143", "2065550143"),
            ("12065550143", "2065550143"),   # leading country code dropped
            ("555-0143", ""),                # too short
            (None, ""),
        ],
    )
    def test_phone(self, raw, expected):
        assert normalize_phone(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Q38076", "Q38076"),
            ("  q38076 ", "Q38076"),
            ("38076", ""),        # missing the Q prefix
            ("Q0", ""),           # not a valid entity id
            ("QQ123", ""),
            ("", ""),
            (None, ""),
        ],
    )
    def test_wikidata(self, raw, expected):
        assert normalize_wikidata(raw) == expected


class TestIdentifierScores:
    def test_agreement_and_comparability(self):
        osm_w = normalize_identifier_array(
            np.array(["https://a.com", "https://b.com", None, None],
                     dtype = object), "website",
        )
        ovt_w = normalize_identifier_array(
            np.array(["http://www.a.com/", "https://zzz.com", "https://c.com",
                      None], dtype = object), "website",
        )
        none = np.array([""] * 4, dtype = object)
        scores, comparable = compute_identifier_scores(
            osm_w, none, none, ovt_w, none, none,
            np.arange(4), np.arange(4),
        )
        assert list(comparable) == [True, True, False, False]
        assert scores[0] == pytest.approx(1.0)   # agree
        assert scores[1] == pytest.approx(0.0)   # both present, disagree
        assert scores[2] == pytest.approx(0.0)   # OSM side missing
        assert scores[3] == pytest.approx(0.0)   # neither side

    def test_wikidata_agreement_when_website_absent(self):
        empty = np.array(["", ""], dtype = object)
        osm_wd = normalize_identifier_array(
            np.array(["Q38076", "Q12345"], dtype = object), "wikidata",
        )
        ovt_wd = normalize_identifier_array(
            np.array(["q38076", "Q99999"], dtype = object), "wikidata",
        )
        scores, comparable = compute_identifier_scores(
            empty, empty, osm_wd, empty, empty, ovt_wd,
            np.arange(2), np.arange(2),
        )
        assert list(comparable) == [True, True]
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)

    def test_either_identifier_agreeing_scores_one(self):
        """Websites disagree but Wikidata ids agree -> 1.0."""
        osm_w = normalize_identifier_array(
            np.array(["https://franchise-a.com"], dtype = object), "website",
        )
        ovt_w = normalize_identifier_array(
            np.array(["https://corporate.com"], dtype = object), "website",
        )
        osm_wd = normalize_identifier_array(
            np.array(["Q38076"], dtype = object), "wikidata",
        )
        ovt_wd = normalize_identifier_array(
            np.array(["Q38076"], dtype = object), "wikidata",
        )
        scores, comparable = compute_identifier_scores(
            osm_w, np.array([""], dtype = object), osm_wd,
            ovt_w, np.array([""], dtype = object), ovt_wd,
            np.arange(1), np.arange(1),
        )
        assert bool(comparable[0]) is True
        assert scores[0] == pytest.approx(1.0)

    def test_phone_alone_makes_a_pair_comparable(self):
        """Phone is a valid identifier kind alongside website/wikidata."""
        empty = np.array(["", ""], dtype = object)
        osm_p = normalize_identifier_array(
            np.array(["+1 206-555-0143", "206-555-0000"], dtype = object),
            "phone",
        )
        ovt_p = normalize_identifier_array(
            np.array(["(206) 555-0143", "206-555-9999"], dtype = object),
            "phone",
        )
        scores, comparable = compute_identifier_scores(
            empty, osm_p, empty, empty, ovt_p, empty,
            np.arange(2), np.arange(2),
        )
        assert list(comparable) == [True, True]
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)

    def test_no_identifiers_anywhere_is_not_comparable(self):
        empty = np.array(["", ""], dtype = object)
        scores, comparable = compute_identifier_scores(
            empty, empty, empty, empty, empty, empty,
            np.arange(2), np.arange(2),
        )
        assert not comparable.any()
        assert not scores.any()

    def test_missing_wikidata_arrays_degrade_to_websites(self):
        """A snapshot without brand_wikidata still scores on websites."""
        osm_w = normalize_identifier_array(
            np.array(["https://a.com"], dtype = object), "website",
        )
        ovt_w = normalize_identifier_array(
            np.array(["http://www.a.com/"], dtype = object), "website",
        )
        scores, comparable = compute_identifier_scores(
            osm_w, None, None, ovt_w, None, None,
            np.arange(1), np.arange(1),
        )
        assert bool(comparable[0]) is True
        assert scores[0] == pytest.approx(1.0)


class TestConditionalWeights:
    """Pairs with identifiers use one weight set, pairs without another."""

    WITH_ID = (0.1667, 0.3333, 0.3333, 0.1667)
    NO_ID = (0.20, 0.40, 0.40, 0.0)

    def _run(self, osm_web, ovt_web):
        candidates = pd.DataFrame(
            {
                "osm_idx": [0, 1], "overture_idx": [0, 1],
                "distance_m": [0.0, 0.0],
            }
        )
        names = np.array(["Same Name", "Same Name"], dtype = object)
        labels = np.array(["Cafe", "Cafe"], dtype = object)
        empty = np.array(["", ""], dtype = object)
        return compute_match_scores(
            candidates = candidates,
            osm_names = names, osm_brands = names,
            overture_names = names, overture_brands = names,
            osm_shared_labels = labels, overture_shared_labels = labels,
            osm_radii_m = np.array([100.0, 100.0]),
            osm_l0_bits = np.zeros(2, dtype = np.uint16),
            overture_l0_bits = np.zeros(2, dtype = np.uint16),
            osm_websites = normalize_identifier_array(osm_web, "website"),
            osm_phones = empty,
            osm_wikidata = empty,
            overture_websites = normalize_identifier_array(
                ovt_web, "website",
            ),
            overture_phones = empty,
            overture_wikidata = empty,
            id_weights = self.WITH_ID, no_id_weights = self.NO_ID,
        )

    def test_identifier_pair_uses_identifier_weights(self):
        # Row 0 has matching websites, row 1 has none. Both agree on
        # name (1.0), type (1.0) and distance (1.0), so the composite is
        # just the weight sum — 1.0 for row 0 (identifier also 1.0) and
        # 1.0 - 0.0 identifier share for row 1.
        out = self._run(
            np.array(["https://a.com", None], dtype = object),
            np.array(["https://a.com", None], dtype = object),
        )
        assert out.identifier_score.iloc[0] == pytest.approx(1.0)
        assert out.composite_score.iloc[0] == pytest.approx(1.0, abs = 1e-3)
        # Row 1: identifier weight is 0, other three sum to 1.0.
        assert out.composite_score.iloc[1] == pytest.approx(1.0, abs = 1e-3)

    def test_disagreeing_identifier_costs_its_share(self):
        out = self._run(
            np.array(["https://a.com", None], dtype = object),
            np.array(["https://different.com", None], dtype = object),
        )
        # Row 0 loses the 0.1667 identifier share; row 1 is unaffected.
        assert out.identifier_score.iloc[0] == pytest.approx(0.0)
        assert out.composite_score.iloc[0] == pytest.approx(
            1.0 - self.WITH_ID[3], abs = 1e-3,
        )
        assert out.composite_score.iloc[1] == pytest.approx(1.0, abs = 1e-3)
