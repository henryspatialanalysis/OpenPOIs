#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.osm.format_observations.
"""
from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import pytest

from openpois.osm.format_observations import (
    format_observations_duckdb,
    format_observations_window,
)


VERSIONS_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("version", pa.int64()),
    ("changeset", pa.int64()),
    ("timestamp", pa.string()),
    ("user", pa.string()),
    ("uid", pa.int64()),
    ("type", pa.string()),
])

CHANGES_SCHEMA = pa.schema([
    ("key", pa.string()),
    ("value", pa.string()),
    ("change", pa.string()),
    ("id", pa.int64()),
    ("version", pa.int64()),
    ("type", pa.string()),
])


def _make_versions(rows):
    return pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in VERSIONS_SCHEMA},
        schema=VERSIONS_SCHEMA,
    )


def _make_changes(rows):
    return pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in CHANGES_SCHEMA},
        schema=CHANGES_SCHEMA,
    )


def _synthetic_inputs():
    """
    Three POIs with name tags. Each has an Added, then a Changed on the
    tag_key, so each should yield 2 observations (the second with changed=1).
    """
    versions = []
    changes = []
    for elem_id in [100, 200, 300]:
        for ver in [1, 2]:
            versions.append({
                "id": elem_id, "version": ver, "changeset": 1000 + ver,
                "timestamp": f"2024-01-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            changes.append({
                "key": "name", "value": f"n{elem_id}.v{ver}",
                "change": "Added" if ver == 1 else "Changed",
                "id": elem_id, "version": ver, "type": "node",
            })
            changes.append({
                "key": "amenity", "value": "cafe",
                "change": "Added" if ver == 1 else "Changed",
                "id": elem_id, "version": ver, "type": "node",
            })
    return versions, changes


def _write_parquets(tmp_path, versions, changes):
    v_path = tmp_path / "versions.parquet"
    c_path = tmp_path / "changes.parquet"
    pq.write_table(_make_versions(versions), v_path)
    pq.write_table(_make_changes(changes), c_path)
    return v_path, c_path


class TestFormatObservationsDuckdb:
    """``format_observations_duckdb`` is the production entry point for the
    OSM history → observations pipeline; these tests pin its row count,
    column set, and state-machine semantics."""

    def test_synthetic_inputs_produce_expected_rows(self, tmp_path):
        versions, changes = _synthetic_inputs()
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        total = format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = ["amenity"],
            verbose = False,
        )
        # Two observations per POI (Added + Changed), three POIs
        assert total == 6
        out = pd.read_parquet(out_path)
        assert len(out) == 6
        assert set(out["id"]) == {100, 200, 300}
        assert set(out["version"]) == {1, 2}
        assert out["changed"].sum() == 6
        # Expected output columns
        expected = {
            "id", "osm_type", "version", "changeset", "obs_timestamp",
            "last_obs_timestamp", "last_tag_timestamp", "user", "last_tag_user",
            "tag_value", "last_tag_value", "changed", "deleted",
            "amenity", "amenity_last_value", "tag_key",
        }
        assert set(out.columns) == expected
        assert set(out["osm_type"]) == {"node"}
        # last_tag_value reflects the PRE-update state: None on v1, v1's value on v2.
        out = out.sort_values(["id", "version"]).reset_index(drop = True)
        for poi_id in [100, 200, 300]:
            rows = out[out["id"] == poi_id].reset_index(drop = True)
            assert pd.isna(rows.loc[0, "last_tag_value"])
            assert rows.loc[1, "last_tag_value"] == f"n{poi_id}.v1"
            assert rows.loc[0, "tag_value"] == f"n{poi_id}.v1"
            assert rows.loc[1, "tag_value"] == f"n{poi_id}.v2"

    def test_tag_state_machine(self, tmp_path):
        """Added → Changed → visible=false → visible=true sequence.

        The re-added version should restore ``tag_value`` to the last SET
        value (``"bar"``), matching the original state-machine semantics.
        """
        versions = []
        changes = []
        seq = [
            ("Added",   "foo", None,    None),
            ("Changed", "bar", None,    None),
            (None,      None,  "false", "Added"),
            (None,      None,  "true",  "Changed"),
        ]
        for ver, (tag_ch, tag_val, vis_val, vis_ch) in enumerate(seq, start = 1):
            versions.append({
                "id": 42, "version": ver, "changeset": 1000 + ver,
                "timestamp": f"2024-01-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            if tag_ch is not None:
                changes.append({
                    "key": "name", "value": tag_val, "change": tag_ch,
                    "id": 42, "version": ver, "type": "node",
                })
            if vis_ch is not None:
                changes.append({
                    "key": "visible", "value": vis_val, "change": vis_ch,
                    "id": 42, "version": ver, "type": "node",
                })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = [],
            verbose = False,
        )
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        assert list(out["version"]) == [1, 2, 3, 4]
        assert list(out["tag_value"].fillna("")) == ["foo", "bar", "", "bar"]
        assert list(out["changed"]) == [1, 1, 1, 1]

    def test_keep_key_stickiness(self, tmp_path):
        """``{k}_last_value`` must persist across versions that don't touch ``k``."""
        versions = []
        changes = []
        seq = [
            ("Added",   "foo",  "Added",   "restaurant"),
            ("Changed", "foo2", None,      None),
            ("Changed", "foo3", "Changed", "bar"),
        ]
        for ver, (tag_ch, tag_val, kk_ch, kk_val) in enumerate(seq, start = 1):
            versions.append({
                "id": 7, "version": ver, "changeset": 1000 + ver,
                "timestamp": f"2024-02-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            changes.append({
                "key": "name", "value": tag_val, "change": tag_ch,
                "id": 7, "version": ver, "type": "node",
            })
            if kk_ch is not None:
                changes.append({
                    "key": "amenity", "value": kk_val, "change": kk_ch,
                    "id": 7, "version": ver, "type": "node",
                })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = ["amenity"],
            verbose = False,
        )
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        amenities = list(out["amenity"].fillna(""))
        lasts = list(out["amenity_last_value"].fillna(""))
        assert amenities == ["restaurant", "restaurant", "bar"]
        # v1: pre-change was None; v2: no change → last stays empty;
        # v3: last = "restaurant".
        assert lasts == ["", "", "restaurant"]

    def test_pre_addition_rows_are_gated_out(self, tmp_path):
        """Versions before the tag_key is first Added must NOT be emitted.

        The ``add_to_list`` gate only opens once ``tag_key`` is Added; a POI
        carrying only a keep_key in early versions yields no observations until
        the name appears.
        """
        versions = []
        changes = []
        # v1, v2: only amenity present (no name yet). v3: name Added. v4: name Changed.
        seq = [
            (None,      None,    "Added",   "shop"),    # v1: no name
            (None,      None,    "Changed", "shop2"),   # v2: still no name
            ("Added",   "cafe",  None,      None),       # v3: name appears
            ("Changed", "cafe2", None,      None),       # v4: name changes
        ]
        for ver, (tag_ch, tag_val, kk_ch, kk_val) in enumerate(seq, start = 1):
            versions.append({
                "id": 55, "version": ver, "changeset": 3000 + ver,
                "timestamp": f"2024-04-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            if tag_ch is not None:
                changes.append({
                    "key": "name", "value": tag_val, "change": tag_ch,
                    "id": 55, "version": ver, "type": "node",
                })
            if kk_ch is not None:
                changes.append({
                    "key": "amenity", "value": kk_val, "change": kk_ch,
                    "id": 55, "version": ver, "type": "node",
                })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        total = format_observations_duckdb(
            changes_path = c_path, versions_path = v_path, output_path = out_path,
            tag_key = "name", keep_keys = ["amenity"], verbose = False,
        )
        # Only v3 (Added) and v4 (Changed) survive the gate.
        assert total == 2
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        assert list(out["version"]) == [3, 4]
        assert list(out["tag_value"]) == ["cafe", "cafe2"]

    def test_multi_cycle_delete_readd(self, tmp_path):
        """Two delete→re-add cycles restore tag_value each time."""
        versions = []
        changes = []
        seq = [
            ("Added",   "foo", None,    None),     # v1
            (None,      None,  "false", "Added"),  # v2 delete
            (None,      None,  "true",  "Changed"),# v3 re-add → restore "foo"
            ("Changed", "baz", None,    None),     # v4 rename
            (None,      None,  "false", "Added"),  # v5 delete
            (None,      None,  "true",  "Changed"),# v6 re-add → restore "baz"
        ]
        for ver, (tag_ch, tag_val, vis_val, vis_ch) in enumerate(seq, start = 1):
            versions.append({
                "id": 71, "version": ver, "changeset": 4000 + ver,
                "timestamp": f"2024-05-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
            if tag_ch is not None:
                changes.append({
                    "key": "name", "value": tag_val, "change": tag_ch,
                    "id": 71, "version": ver, "type": "node",
                })
            if vis_ch is not None:
                changes.append({
                    "key": "visible", "value": vis_val, "change": vis_ch,
                    "id": 71, "version": ver, "type": "node",
                })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        format_observations_duckdb(
            changes_path = c_path, versions_path = v_path, output_path = out_path,
            tag_key = "name", keep_keys = [], verbose = False,
        )
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        assert list(out["version"]) == [1, 2, 3, 4, 5, 6]
        assert list(out["tag_value"].fillna("")) == ["foo", "", "foo", "baz", "", "baz"]
        assert list(out["changed"]) == [1, 1, 1, 1, 1, 1]

    def test_left_join_null_inheritance(self, tmp_path):
        """Versions with no relevant changes (LEFT-JOIN produces NULLs) should
        inherit prior state without crashing."""
        versions = []
        changes = []
        for ver in [1, 2, 3]:
            versions.append({
                "id": 99, "version": ver, "changeset": 2000 + ver,
                "timestamp": f"2024-03-{ver:02d}T00:00:00+00:00",
                "user": "u", "uid": 1, "type": "node",
            })
        # Only v1 has tag/keep-key changes; v2 and v3 have no rows at all.
        changes.append({
            "key": "name", "value": "cafe", "change": "Added",
            "id": 99, "version": 1, "type": "node",
        })
        changes.append({
            "key": "amenity", "value": "cafe", "change": "Added",
            "id": 99, "version": 1, "type": "node",
        })
        v_path, c_path = _write_parquets(tmp_path, versions, changes)
        out_path = tmp_path / "obs.parquet"
        total = format_observations_duckdb(
            changes_path = c_path,
            versions_path = v_path,
            output_path = out_path,
            tag_key = "name",
            keep_keys = ["amenity"],
            verbose = False,
        )
        assert total == 3
        out = pd.read_parquet(out_path).sort_values("version").reset_index(drop = True)
        assert list(out["tag_value"].fillna("")) == ["cafe", "cafe", "cafe"]
        assert list(out["amenity"].fillna("")) == ["cafe", "cafe", "cafe"]
        assert list(out["changed"]) == [1, 0, 0]


# Stage 2b golden test: DuckDB window functions must reproduce the per-POI
# state machine byte-for-byte on every fixture scenario. -------------------->


def _versions_changes_from_seq(elem_id, seq, keep_key = None):
    """Build (versions, changes) from a per-version sequence.

    Each ``seq`` entry is ``(tag_change, tag_value, other_change, other_value)``
    where ``other`` is a visibility toggle (``visible``) or a keep-key change,
    depending on ``keep_key``.
    """
    versions, changes = [], []
    for ver, (tag_ch, tag_val, oth_ch, oth_val) in enumerate(seq, start = 1):
        versions.append({
            "id": elem_id, "version": ver, "changeset": 1000 + ver,
            "timestamp": f"2024-06-{ver:02d}T00:00:00+00:00",
            "user": f"u{ver}", "uid": ver, "type": "node",
        })
        if tag_ch is not None:
            changes.append({
                "key": "name", "value": tag_val, "change": tag_ch,
                "id": elem_id, "version": ver, "type": "node",
            })
        if oth_ch is not None:
            key = keep_key if keep_key else "visible"
            changes.append({
                "key": key, "value": oth_val, "change": oth_ch,
                "id": elem_id, "version": ver, "type": "node",
            })
    return versions, changes


def _scenario_synthetic():
    v, c = _synthetic_inputs()
    return v, c, "name", ["amenity"]


def _scenario_delete_readd():
    v, c = _versions_changes_from_seq(42, [
        ("Added", "foo", None, None),
        ("Changed", "bar", None, None),
        (None, None, "Added", "false"),
        (None, None, "Changed", "true"),
    ])
    return v, c, "name", []


def _scenario_multi_cycle():
    v, c = _versions_changes_from_seq(71, [
        ("Added", "foo", None, None),
        (None, None, "Added", "false"),
        (None, None, "Changed", "true"),
        ("Changed", "baz", None, None),
        (None, None, "Added", "false"),
        (None, None, "Changed", "true"),
    ])
    return v, c, "name", []


def _scenario_keep_sticky():
    v, c = _versions_changes_from_seq(7, [
        ("Added", "foo", "Added", "restaurant"),
        ("Changed", "foo2", None, None),
        ("Changed", "foo3", "Changed", "bar"),
    ], keep_key = "amenity")
    return v, c, "name", ["amenity"]


def _scenario_pre_addition():
    v, c = _versions_changes_from_seq(55, [
        (None, None, "Added", "shop"),
        (None, None, "Changed", "shop2"),
        ("Added", "cafe", None, None),
        ("Changed", "cafe2", None, None),
    ], keep_key = "amenity")
    return v, c, "name", ["amenity"]


def _scenario_multi_poi():
    """Two POIs interleaved + a keep-key delete, to exercise partitioning."""
    v1, c1 = _versions_changes_from_seq(1, [
        ("Added", "a", "Added", "cafe"),
        ("Changed", "a2", "Deleted", "cafe"),
    ], keep_key = "amenity")
    v2, c2 = _versions_changes_from_seq(2, [
        ("Added", "b", None, None),
        (None, None, "Added", "false"),
    ])
    return v1 + v2, c1 + c2, "name", ["amenity"]


_SCENARIOS = {
    "synthetic": _scenario_synthetic,
    "delete_readd": _scenario_delete_readd,
    "multi_cycle": _scenario_multi_cycle,
    "keep_sticky": _scenario_keep_sticky,
    "pre_addition": _scenario_pre_addition,
    "multi_poi": _scenario_multi_poi,
}


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
@pytest.mark.parametrize("num_buckets", [1, 3])
def test_window_matches_state_machine(tmp_path, scenario, num_buckets):
    """The Stage 2b window-function build must equal the state-machine build,
    both single-pass and bucketed (bucketing must never split a POI's rows)."""
    versions, changes, tag_key, keep_keys = _SCENARIOS[scenario]()
    v_path, c_path = _write_parquets(tmp_path, versions, changes)

    sm_path = tmp_path / "state_machine.parquet"
    win_path = tmp_path / "window.parquet"
    format_observations_duckdb(
        c_path, v_path, sm_path, tag_key, keep_keys, verbose = False,
    )
    format_observations_window(
        c_path, v_path, win_path, tag_key, keep_keys,
        num_buckets = num_buckets, verbose = False,
    )

    sort_cols = ["osm_type", "id", "version"]
    sm = (
        pd.read_parquet(sm_path).sort_values(sort_cols).reset_index(drop = True)
    )
    win = (
        pd.read_parquet(win_path).sort_values(sort_cols).reset_index(drop = True)
    )
    sm = sm[sorted(sm.columns)]
    win = win[sorted(win.columns)]
    pd.testing.assert_frame_equal(sm, win, check_dtype = False)
