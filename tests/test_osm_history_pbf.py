#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.io.osm_history_pbf.

All external I/O (requests.get, subprocess.run, osmium.FileProcessor)
is mocked so tests run in milliseconds without network or filesystem access.
"""
from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

import requests

from openpois.io.osm_history_pbf import (
    CHANGES_SCHEMA,
    VERSIONS_SCHEMA,
    HistoryExtract,
    _concat_history,
    _diff_tag_sets,
    _tag_set_for_version,
    download_history_pbf,
    download_osm_history,
    filter_history_pbf,
    parse_history_pbf,
    time_filter_history_pbf,
)
import pyarrow as pa


# ---------------------------------------------------------------------------
# Helpers: fake pyosmium objects
# ---------------------------------------------------------------------------


def _make_tag(k: str, v: str) -> MagicMock:
    tag = MagicMock()
    tag.k = k
    tag.v = v
    return tag


def _make_tags(tag_dict: dict) -> list[MagicMock]:
    """Return an iterable of mock pyosmium Tag objects from a dict."""
    return [_make_tag(k, v) for k, v in tag_dict.items()]


def _make_location(lon: float | None, lat: float | None) -> MagicMock:
    loc = MagicMock()
    if lon is None or lat is None:
        loc.valid = MagicMock(return_value=False)
    else:
        loc.valid = MagicMock(return_value=True)
        loc.lon = lon
        loc.lat = lat
    return loc


def _make_version(
    kind: str,
    osm_id: int,
    version: int,
    changeset: int,
    tag_dict: dict,
    user: str = "alice",
    uid: int = 1,
    visible: bool = True,
    timestamp: datetime.datetime | None = None,
    lon: float | None = None,
    lat: float | None = None,
) -> MagicMock:
    obj = MagicMock()
    obj.id = osm_id
    obj.version = version
    obj.changeset = changeset
    obj.user = user
    obj.uid = uid
    obj.visible = visible
    obj.timestamp = (
        timestamp if timestamp is not None
        else datetime.datetime(2020, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    )
    obj.tags = _make_tags(tag_dict)
    obj.is_node = MagicMock(return_value=kind == "node")
    obj.is_way = MagicMock(return_value=kind == "way")
    obj.is_relation = MagicMock(return_value=kind == "relation")
    obj.location = _make_location(lon, lat)
    return obj


def _make_file_processor(objects: list) -> MagicMock:
    fp = MagicMock()
    fp.__iter__ = MagicMock(return_value=iter(objects))
    return fp


# ---------------------------------------------------------------------------
# _diff_tag_sets
# ---------------------------------------------------------------------------


class TestDiffTagSets:
    def test_added_only(self):
        rows = _diff_tag_sets(set(), {("amenity", "cafe")})
        assert rows == [{"key": "amenity", "value": "cafe", "change": "Added"}]

    def test_deleted_only(self):
        rows = _diff_tag_sets({("amenity", "cafe")}, set())
        assert rows == [{"key": "amenity", "value": "cafe", "change": "Deleted"}]

    def test_changed_key(self):
        rows = _diff_tag_sets(
            {("amenity", "cafe")},
            {("amenity", "restaurant")},
        )
        assert rows == [
            {"key": "amenity", "value": "restaurant", "change": "Changed"},
        ]

    def test_unchanged_tags_produce_no_rows(self):
        tags = {("amenity", "cafe"), ("name", "X")}
        assert _diff_tag_sets(tags, tags) == []

    def test_mixed_diff(self):
        prev = {("amenity", "cafe"), ("name", "Old")}
        curr = {("amenity", "restaurant"), ("cuisine", "italian")}
        rows = _diff_tag_sets(prev, curr)
        by_key = {r["key"]: r for r in rows}
        assert by_key["amenity"]["change"] == "Changed"
        assert by_key["amenity"]["value"] == "restaurant"
        assert by_key["cuisine"]["change"] == "Added"
        assert by_key["name"]["change"] == "Deleted"
        assert by_key["name"]["value"] == "Old"


# ---------------------------------------------------------------------------
# _tag_set_for_version
# ---------------------------------------------------------------------------


class TestTagSetForVersion:
    def test_node_includes_lat_lon_and_visible(self):
        obj = _make_version(
            kind="node",
            osm_id=1, version=1, changeset=100,
            tag_dict={"amenity": "cafe"},
            lon=-122.0, lat=47.0,
        )
        tags = _tag_set_for_version(obj)
        assert ("amenity", "cafe") in tags
        assert ("visible", "true") in tags
        assert ("lat", "47.0") in tags
        assert ("lon", "-122.0") in tags

    def test_way_excludes_lat_lon(self):
        obj = _make_version(
            kind="way",
            osm_id=2, version=1, changeset=100,
            tag_dict={"building": "yes"},
        )
        tags = _tag_set_for_version(obj)
        assert ("building", "yes") in tags
        assert ("visible", "true") in tags
        assert not any(k == "lat" for k, _ in tags)
        assert not any(k == "lon" for k, _ in tags)

    def test_invisible_flag(self):
        obj = _make_version(
            kind="node",
            osm_id=1, version=2, changeset=101,
            tag_dict={}, visible=False,
        )
        tags = _tag_set_for_version(obj)
        assert ("visible", "false") in tags


# ---------------------------------------------------------------------------
# download_history_pbf
# ---------------------------------------------------------------------------


class TestDownloadHistoryPbf:
    def test_skips_if_exists_and_no_overwrite(self, tmp_path):
        output = tmp_path / "out.osh.pbf"
        output.write_bytes(b"fake")
        with patch(
            "openpois.io.osm_history_pbf._load_cookie_session"
        ) as mock_session:
            result = download_history_pbf(
                url="http://example.com/x.osh.pbf",
                output_path=output,
                overwrite=False,
            )
        mock_session.assert_not_called()
        assert result == output

    def test_raises_if_cookie_file_missing(self, tmp_path):
        output = tmp_path / "out.osh.pbf"
        missing_cookie = tmp_path / "nope.txt"
        with pytest.raises(FileNotFoundError, match="cookie file not found"):
            download_history_pbf(
                url="http://example.com/x.osh.pbf",
                output_path=output,
                cookie_file=missing_cookie,
                overwrite=False,
            )

    def test_downloads_via_streaming_session(self, tmp_path):
        output = tmp_path / "subdir" / "out.osh.pbf"

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"content-length": "5"}
        mock_resp.iter_content = MagicMock(return_value=[b"hello"])

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch(
            "openpois.io.osm_history_pbf._load_cookie_session",
            return_value=mock_session,
        ):
            result = download_history_pbf(
                url="http://example.com/x.osh.pbf",
                output_path=output,
                overwrite=False,
            )

        mock_session.get.assert_called_once_with(
            "http://example.com/x.osh.pbf", stream=True, timeout=(30, None)
        )
        assert result == output
        assert output.exists()


# ---------------------------------------------------------------------------
# filter_history_pbf
# ---------------------------------------------------------------------------


class TestFilterHistoryPbf:
    def test_skips_if_output_exists_and_no_overwrite(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")
        output_pbf.write_bytes(b"fake")
        with patch(
            "openpois.io.osm_history_pbf.subprocess.run"
        ) as mock_run:
            result = filter_history_pbf(
                input_pbf, output_pbf, ["amenity"], overwrite=False
            )
        mock_run.assert_not_called()
        assert result == output_pbf

    def test_command_uses_omit_referenced_and_osh_format(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")

        with (
            patch("openpois.io.osm_history_pbf.subprocess.run") as mock_run,
            patch(
                "openpois.io.osm_history_pbf._resolve_osmium",
                return_value="/usr/bin/osmium",
            ),
        ):
            filter_history_pbf(
                input_pbf, output_pbf, ["amenity", "shop"], overwrite=False
            )

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "tags-filter"
        assert "--omit-referenced" in cmd
        assert "--output-format=osh.pbf" in cmd
        assert "nwr/amenity" in cmd
        assert "nwr/shop" in cmd
        assert mock_run.call_args[1].get("check") is True


# ---------------------------------------------------------------------------
# time_filter_history_pbf
# ---------------------------------------------------------------------------


class TestTimeFilterHistoryPbf:
    def test_skips_if_output_exists_and_no_overwrite(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")
        output_pbf.write_bytes(b"fake")
        with patch(
            "openpois.io.osm_history_pbf.subprocess.run"
        ) as mock_run:
            result = time_filter_history_pbf(
                input_pbf, output_pbf,
                datetime.date(2016, 1, 1),
                datetime.date(2025, 12, 31),
                overwrite=False,
            )
        mock_run.assert_not_called()
        assert result == output_pbf

    def test_passes_iso_formatted_timestamps(self, tmp_path):
        input_pbf = tmp_path / "in.osh.pbf"
        output_pbf = tmp_path / "out.osh.pbf"
        input_pbf.write_bytes(b"fake")

        with (
            patch("openpois.io.osm_history_pbf.subprocess.run") as mock_run,
            patch(
                "openpois.io.osm_history_pbf._resolve_osmium",
                return_value="/usr/bin/osmium",
            ),
        ):
            time_filter_history_pbf(
                input_pbf, output_pbf,
                datetime.date(2016, 1, 1),
                datetime.date(2025, 12, 31),
                overwrite=False,
            )

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "time-filter"
        assert "2016-01-01T00:00:00Z" in cmd
        assert "2025-12-31T00:00:00Z" in cmd
        # Range must come AFTER the input path for osmium
        input_idx = cmd.index(str(input_pbf))
        assert cmd.index("2016-01-01T00:00:00Z") > input_idx
        assert cmd.index("2025-12-31T00:00:00Z") > input_idx


# ---------------------------------------------------------------------------
# parse_history_pbf
# ---------------------------------------------------------------------------


class TestParseHistoryPbf:
    def test_single_element_multiple_versions_emits_diffs(self, tmp_path):
        """Versions of the same element should diff against the previous one."""
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"

        objs = [
            _make_version(
                kind="node", osm_id=1, version=1, changeset=100,
                tag_dict={"amenity": "cafe"}, lon=-122.0, lat=47.0,
            ),
            _make_version(
                kind="node", osm_id=1, version=2, changeset=101,
                tag_dict={"amenity": "restaurant"}, lon=-122.0, lat=47.0,
            ),
            _make_version(
                kind="node", osm_id=1, version=3, changeset=102,
                tag_dict={}, visible=False,
                lon=-122.0, lat=47.0,
            ),
        ]

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor",
            return_value=_make_file_processor(objs),
        ):
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                chunk_size=10,
                overwrite=True,
                verbose=False,
            )

        versions = pd.read_parquet(v_path)
        changes = pd.read_parquet(c_path)

        assert len(versions) == 3
        assert list(versions["version"]) == [1, 2, 3]
        assert list(versions["changeset"]) == [100, 101, 102]
        assert (versions["type"] == "node").all()

        # Version 1 must produce Added rows (no prior state)
        v1_changes = changes.query("version == 1")
        assert (v1_changes["change"] == "Added").all()
        assert "amenity" in set(v1_changes["key"])

        # Version 2 changes amenity value (Changed)
        v2_changes = changes.query("version == 2")
        amenity_rows = v2_changes.query('key == "amenity"')
        assert len(amenity_rows) == 1
        assert amenity_rows.iloc[0]["change"] == "Changed"
        assert amenity_rows.iloc[0]["value"] == "restaurant"

        # Version 3 marks visible=false (Changed) and removes amenity (Deleted)
        v3_changes = changes.query("version == 3")
        visible_rows = v3_changes.query('key == "visible"')
        assert len(visible_rows) == 1
        assert visible_rows.iloc[0]["value"] == "false"
        assert visible_rows.iloc[0]["change"] == "Changed"
        amenity_del = v3_changes.query(
            'key == "amenity" and change == "Deleted"'
        )
        assert len(amenity_del) == 1

    def test_element_boundary_resets_prev_tags(self, tmp_path):
        """When (type, id) changes, the next version's diff is against empty."""
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"

        objs = [
            _make_version(
                kind="node", osm_id=1, version=1, changeset=100,
                tag_dict={"amenity": "cafe"}, lon=-122.0, lat=47.0,
            ),
            _make_version(
                kind="node", osm_id=2, version=1, changeset=101,
                tag_dict={"shop": "bakery"}, lon=-122.1, lat=47.1,
            ),
        ]

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor",
            return_value=_make_file_processor(objs),
        ):
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                chunk_size=10,
                overwrite=True,
                verbose=False,
            )

        changes = pd.read_parquet(c_path)
        # Every row for node id=2 v=1 must be "Added", never "Deleted"
        n2_rows = changes.query("id == 2 and version == 1")
        assert (n2_rows["change"] == "Added").all()
        # It should NOT inherit a "Deleted amenity=cafe" row from node 1
        assert not (
            (n2_rows["key"] == "amenity") & (n2_rows["change"] == "Deleted")
        ).any()

    def test_empty_input_writes_empty_parquets(self, tmp_path):
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor",
            return_value=_make_file_processor([]),
        ):
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                chunk_size=10,
                overwrite=True,
                verbose=False,
            )

        # Both files should exist and be readable
        assert v_path.exists()
        assert c_path.exists()
        assert pq.ParquetFile(str(v_path)).metadata.num_rows == 0
        assert pq.ParquetFile(str(c_path)).metadata.num_rows == 0

    def test_skips_if_both_outputs_exist_and_no_overwrite(self, tmp_path):
        pbf_path = tmp_path / "in.osh.pbf"
        pbf_path.write_bytes(b"fake")
        v_path = tmp_path / "versions.parquet"
        c_path = tmp_path / "changes.parquet"
        v_path.write_bytes(b"fake")
        c_path.write_bytes(b"fake")

        with patch(
            "openpois.io.osm_history_pbf.osmium.FileProcessor"
        ) as mock_fp:
            parse_history_pbf(
                pbf_path=pbf_path,
                versions_path=v_path,
                changes_path=c_path,
                overwrite=False,
                verbose=False,
            )
        mock_fp.assert_not_called()


# ---------------------------------------------------------------------------
# _concat_history cross-extract dedup
# ---------------------------------------------------------------------------


def _write_versions(path, rows):
    tbl = pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in VERSIONS_SCHEMA},
        schema=VERSIONS_SCHEMA,
    )
    pq.write_table(tbl, path)


def _write_changes(path, rows):
    tbl = pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in CHANGES_SCHEMA},
        schema=CHANGES_SCHEMA,
    )
    pq.write_table(tbl, path)


class TestConcatHistory:
    """Cover US+PR concat with dropping of cross-extract duplicates."""

    def test_drops_pr_copy_of_shared_type_id(self, tmp_path):
        # Same (type, id) = ('node', 10) appears in both US and PR
        us_v = tmp_path / "us_v.parquet"
        pr_v = tmp_path / "pr_v.parquet"
        out_v = tmp_path / "out_v.parquet"
        us_c = tmp_path / "us_c.parquet"
        pr_c = tmp_path / "pr_c.parquet"
        out_c = tmp_path / "out_c.parquet"

        _write_versions(us_v, [
            {"id": 10, "version": 1, "changeset": 1, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
            {"id": 20, "version": 1, "changeset": 2, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
        ])
        _write_versions(pr_v, [
            {"id": 10, "version": 1, "changeset": 1, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},  # duplicate of US
            {"id": 30, "version": 1, "changeset": 3, "timestamp": "t",
             "user": "u", "uid": 1, "type": "way"},
        ])
        _write_changes(us_c, [
            {"key": "amenity", "value": "cafe", "change": "Added",
             "id": 10, "version": 1, "type": "node"},
            {"key": "amenity", "value": "bar", "change": "Added",
             "id": 20, "version": 1, "type": "node"},
        ])
        _write_changes(pr_c, [
            {"key": "amenity", "value": "cafe", "change": "Added",
             "id": 10, "version": 1, "type": "node"},  # duplicate
            {"key": "shop", "value": "gift", "change": "Added",
             "id": 30, "version": 1, "type": "way"},
        ])

        _concat_history(
            intermediates=[(us_v, us_c), (pr_v, pr_c)],
            out_versions_path=out_v,
            out_changes_path=out_c,
        )

        v = pd.read_parquet(out_v)
        c = pd.read_parquet(out_c)
        assert sorted(zip(v["type"], v["id"])) == [
            ("node", 10), ("node", 20), ("way", 30)
        ]
        # (type, id, version, key) must be unique — the dedup invariant that
        # format_observations.py depends on
        assert c.groupby(["type", "id", "version", "key"]).size().max() == 1
        assert sorted(zip(c["type"], c["id"])) == [
            ("node", 10), ("node", 20), ("way", 30)
        ]

    def test_keeps_node_and_way_with_same_integer_id(self, tmp_path):
        # Regression: OSM ids are type-scoped. A node and a way both with
        # id=100 must be kept as separate POIs, not deduped together.
        us_v = tmp_path / "us_v.parquet"
        pr_v = tmp_path / "pr_v.parquet"
        out_v = tmp_path / "out_v.parquet"
        us_c = tmp_path / "us_c.parquet"
        pr_c = tmp_path / "pr_c.parquet"
        out_c = tmp_path / "out_c.parquet"
        _write_versions(us_v, [
            {"id": 100, "version": 1, "changeset": 1, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
            {"id": 100, "version": 1, "changeset": 2, "timestamp": "t",
             "user": "u", "uid": 1, "type": "way"},
        ])
        _write_versions(pr_v, [])  # PR has no rows at all
        _write_changes(us_c, [
            {"key": "amenity", "value": "cafe", "change": "Added",
             "id": 100, "version": 1, "type": "node"},
            {"key": "leisure", "value": "park", "change": "Added",
             "id": 100, "version": 1, "type": "way"},
        ])
        _write_changes(pr_c, [])
        _concat_history(
            intermediates=[(us_v, us_c), (pr_v, pr_c)],
            out_versions_path=out_v,
            out_changes_path=out_c,
        )
        v = pd.read_parquet(out_v)
        c = pd.read_parquet(out_c)
        assert sorted(zip(v["type"], v["id"])) == [("node", 100), ("way", 100)]
        assert sorted(zip(c["type"], c["key"])) == [
            ("node", "amenity"), ("way", "leisure")
        ]

    def test_no_overlap_is_pure_concat(self, tmp_path):
        us_v = tmp_path / "us_v.parquet"
        pr_v = tmp_path / "pr_v.parquet"
        out_v = tmp_path / "out_v.parquet"
        us_c = tmp_path / "us_c.parquet"
        pr_c = tmp_path / "pr_c.parquet"
        out_c = tmp_path / "out_c.parquet"
        _write_versions(us_v, [
            {"id": 1, "version": 1, "changeset": 1, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
        ])
        _write_versions(pr_v, [
            {"id": 2, "version": 1, "changeset": 2, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
        ])
        _write_changes(us_c, [
            {"key": "amenity", "value": "a", "change": "Added",
             "id": 1, "version": 1, "type": "node"},
        ])
        _write_changes(pr_c, [
            {"key": "amenity", "value": "b", "change": "Added",
             "id": 2, "version": 1, "type": "node"},
        ])
        _concat_history(
            intermediates=[(us_v, us_c), (pr_v, pr_c)],
            out_versions_path=out_v,
            out_changes_path=out_c,
        )
        assert len(pd.read_parquet(out_v)) == 2
        assert len(pd.read_parquet(out_c)) == 2

    def test_n_way_dedup_drops_against_any_prior_extract(self, tmp_path):
        # Regression: with N >= 3 extracts, each subsequent extract must
        # dedup against ALL prior extracts, not just the first. Here the
        # third extract carries an id seen in extract #2 (not #1).
        us_v, pr_v, ter_v = (tmp_path / f"{n}_v.parquet" for n in ("us", "pr", "ter"))
        us_c, pr_c, ter_c = (tmp_path / f"{n}_c.parquet" for n in ("us", "pr", "ter"))
        out_v = tmp_path / "out_v.parquet"
        out_c = tmp_path / "out_c.parquet"
        _write_versions(us_v, [
            {"id": 1, "version": 1, "changeset": 1, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
        ])
        _write_versions(pr_v, [
            {"id": 2, "version": 1, "changeset": 2, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
        ])
        # Territory extract: id=2 dupes PR (must be dropped), id=3 is new.
        _write_versions(ter_v, [
            {"id": 2, "version": 1, "changeset": 2, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
            {"id": 3, "version": 1, "changeset": 3, "timestamp": "t",
             "user": "u", "uid": 1, "type": "node"},
        ])
        _write_changes(us_c, [
            {"key": "amenity", "value": "a", "change": "Added",
             "id": 1, "version": 1, "type": "node"},
        ])
        _write_changes(pr_c, [
            {"key": "amenity", "value": "b", "change": "Added",
             "id": 2, "version": 1, "type": "node"},
        ])
        _write_changes(ter_c, [
            {"key": "amenity", "value": "b", "change": "Added",
             "id": 2, "version": 1, "type": "node"},  # dropped
            {"key": "shop", "value": "c", "change": "Added",
             "id": 3, "version": 1, "type": "node"},  # kept
        ])
        _concat_history(
            intermediates=[(us_v, us_c), (pr_v, pr_c), (ter_v, ter_c)],
            out_versions_path=out_v,
            out_changes_path=out_c,
        )
        v = pd.read_parquet(out_v)
        c = pd.read_parquet(out_c)
        assert sorted(zip(v["type"], v["id"])) == [
            ("node", 1), ("node", 2), ("node", 3)
        ]
        # The id=2 row must NOT be duplicated; it was kept from PR and
        # dropped from the territory extract.
        assert c.groupby(["type", "id", "version", "key"]).size().max() == 1
        assert sorted(zip(c["type"], c["id"])) == [
            ("node", 1), ("node", 2), ("node", 3)
        ]


def _make_history_extract(name, tmp_path, url="https://example.invalid/x.osh.pbf"):
    """Build a HistoryExtract with unique scratch paths under tmp_path."""
    return HistoryExtract(
        name=name,
        url=url,
        raw_pbf_path=tmp_path / f"{name}.osh.pbf",
        filtered_pbf_path=tmp_path / f"{name}-pois.osh.pbf",
        time_filtered_pbf_path=tmp_path / f"{name}-pois-timefilt.osh.pbf",
        versions_path=tmp_path / f"{name}_versions.parquet",
        changes_path=tmp_path / f"{name}_changes.parquet",
    )


def _fake_intermediate_writer(rows_by_name):
    """Return a side_effect for _download_filter_timefilter_parse that writes
    minimal versions/changes parquets for each named extract."""
    def _side_effect(**kwargs):
        # Recover the extract name from the parquet path stem.
        versions_path = kwargs["versions_path"]
        changes_path = kwargs["changes_path"]
        name = versions_path.stem.removesuffix("_versions")
        rows = rows_by_name.get(name, [])
        _write_versions(versions_path, rows)
        _write_changes(changes_path, [
            {"key": "amenity", "value": "x", "change": "Added",
             "id": r["id"], "version": r["version"], "type": r["type"]}
            for r in rows
        ])
        return (versions_path, changes_path)
    return _side_effect


class TestDownloadOsmHistoryOrchestrator:
    """Cover the orchestrator's input validation and per-extract 404
    tolerance — the new opinionated behavior on the territory branch."""

    def test_rejects_duplicate_extract_names(self, tmp_path):
        extracts = [
            _make_history_extract("us", tmp_path),
            _make_history_extract("us", tmp_path / "dup"),
        ]
        with pytest.raises(ValueError, match="unique"):
            download_osm_history(
                extracts=extracts,
                output_versions_path=tmp_path / "out_v.parquet",
                output_changes_path=tmp_path / "out_c.parquet",
                filter_keys=["amenity"],
                start_date=datetime.date(2024, 1, 1),
                end_date=datetime.date(2024, 6, 30),
            )

    def test_rejects_empty_extracts_list(self, tmp_path):
        with pytest.raises(ValueError, match="at least one"):
            download_osm_history(
                extracts=[],
                output_versions_path=tmp_path / "out_v.parquet",
                output_changes_path=tmp_path / "out_c.parquet",
                filter_keys=["amenity"],
                start_date=datetime.date(2024, 1, 1),
                end_date=datetime.date(2024, 6, 30),
            )

    @patch("openpois.io.osm_history_pbf._download_filter_timefilter_parse")
    def test_skips_one_extract_on_404_and_concats_rest(self, mock_proc, tmp_path):
        # Three extracts; the middle one 404s. Other two should succeed.
        extracts = [
            _make_history_extract("us", tmp_path),
            _make_history_extract("missing", tmp_path),
            _make_history_extract("usvi", tmp_path),
        ]
        rows_by_name = {
            "us": [{"id": 1, "version": 1, "changeset": 1, "timestamp": "t",
                    "user": "u", "uid": 1, "type": "node"}],
            "usvi": [{"id": 2, "version": 1, "changeset": 2, "timestamp": "t",
                      "user": "u", "uid": 1, "type": "node"}],
        }
        writer = _fake_intermediate_writer(rows_by_name)

        # 404 only for "missing"; otherwise write parquets.
        not_found_response = MagicMock()
        not_found_response.status_code = 404
        not_found_err = requests.HTTPError(response=not_found_response)

        def side_effect(**kwargs):
            if "missing" in str(kwargs["raw_pbf_path"]):
                raise not_found_err
            return writer(**kwargs)
        mock_proc.side_effect = side_effect

        out_v = tmp_path / "out_v.parquet"
        out_c = tmp_path / "out_c.parquet"
        download_osm_history(
            extracts=extracts,
            output_versions_path=out_v,
            output_changes_path=out_c,
            filter_keys=["amenity"],
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 6, 30),
        )
        # Only us + usvi rows should reach the final concat.
        v = pd.read_parquet(out_v)
        assert sorted(zip(v["type"], v["id"])) == [("node", 1), ("node", 2)]

    @patch("openpois.io.osm_history_pbf._download_filter_timefilter_parse")
    def test_non_404_http_error_propagates(self, mock_proc, tmp_path):
        extracts = [_make_history_extract("us", tmp_path)]
        server_err_response = MagicMock()
        server_err_response.status_code = 503
        mock_proc.side_effect = requests.HTTPError(response=server_err_response)
        with pytest.raises(requests.HTTPError):
            download_osm_history(
                extracts=extracts,
                output_versions_path=tmp_path / "out_v.parquet",
                output_changes_path=tmp_path / "out_c.parquet",
                filter_keys=["amenity"],
                start_date=datetime.date(2024, 1, 1),
                end_date=datetime.date(2024, 6, 30),
            )

    @patch("openpois.io.osm_history_pbf._download_filter_timefilter_parse")
    def test_all_extracts_404_raises_value_error(self, mock_proc, tmp_path):
        extracts = [
            _make_history_extract("us", tmp_path),
            _make_history_extract("pr", tmp_path),
        ]
        not_found_response = MagicMock()
        not_found_response.status_code = 404
        mock_proc.side_effect = requests.HTTPError(response=not_found_response)
        with pytest.raises(ValueError, match="All history extracts failed"):
            download_osm_history(
                extracts=extracts,
                output_versions_path=tmp_path / "out_v.parquet",
                output_changes_path=tmp_path / "out_c.parquet",
                filter_keys=["amenity"],
                start_date=datetime.date(2024, 1, 1),
                end_date=datetime.date(2024, 6, 30),
            )
