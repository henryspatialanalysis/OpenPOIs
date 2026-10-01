#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.io.osm_snapshot.

All external I/O (requests.get, subprocess.run, osmium.FileProcessor)
is mocked so tests run in milliseconds without network or filesystem access.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import osmium
import pytest
from shapely.geometry import Point, Polygon

from openpois.io._osm_poi_handler import POIRecordBuilder
from openpois.io.osm_snapshot import (
    download_pbf,
    filter_pbf,
    parse_pbf_to_geodataframe,
)


# ---------------------------------------------------------------------------
# Helpers: fake osmium objects
# ---------------------------------------------------------------------------


def _make_tags(tag_dict: dict) -> MagicMock:
    """Return a mock osmium TagList from a plain dict."""
    tags = MagicMock()
    tags.__iter__ = MagicMock(return_value=iter(tag_dict.keys()))
    tags.__contains__ = MagicMock(side_effect=lambda k: k in tag_dict)
    tags.get = MagicMock(side_effect=lambda k, default=None: tag_dict.get(k, default))
    return tags


def _make_node(osm_id: int, lon: float, lat: float, tag_dict: dict) -> MagicMock:
    node = MagicMock()
    node.id = osm_id
    node.location.lon = lon
    node.location.lat = lat
    node.tags = _make_tags(tag_dict)
    node.timestamp = None
    node.is_node = MagicMock(return_value=True)
    node.is_way = MagicMock(return_value=False)
    node.is_area = MagicMock(return_value=False)
    return node


def _make_way_nodes(coords: list[tuple[float, float]]) -> list[MagicMock]:
    way_nodes = []
    for lon, lat in coords:
        nd = MagicMock()
        nd.lon = lon
        nd.lat = lat
        way_nodes.append(nd)
    return way_nodes


def _make_way(
    osm_id: int, coords: list[tuple[float, float]], tag_dict: dict
) -> MagicMock:
    way = MagicMock()
    way.id = osm_id
    way.nodes = _make_way_nodes(coords)
    way.tags = _make_tags(tag_dict)
    return way


def _make_file_processor(objects: list) -> MagicMock:
    """Return a mock osmium.FileProcessor that yields the given objects."""
    fp = MagicMock()
    fp.with_locations.return_value = fp
    fp.with_areas.return_value = fp
    fp.__iter__ = MagicMock(return_value=iter(objects))
    return fp


# ---------------------------------------------------------------------------
# download_pbf
# ---------------------------------------------------------------------------


class TestDownloadPbf:
    def test_skips_if_exists_and_no_overwrite(self, tmp_path):
        """Should return early without calling requests.get when file exists."""
        output = tmp_path / "out.pbf"
        output.write_bytes(b"fake")

        with patch("openpois.io.osm_snapshot.requests.get") as mock_get:
            result = download_pbf("http://example.com/file.pbf", output, overwrite=False)

        mock_get.assert_not_called()
        assert result == output

    def test_downloads_when_no_file(self, tmp_path):
        """Should call requests.get with stream=True and write content."""
        output = tmp_path / "subdir" / "out.pbf"

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"content-length": "10"}
        mock_resp.iter_content = MagicMock(return_value=[b"hellworld!"])

        with patch(
            "openpois.io.osm_snapshot.requests.get", return_value=mock_resp
        ) as mock_get:
            result = download_pbf(
                "http://example.com/file.pbf", output, overwrite=False
            )

        mock_get.assert_called_once_with(
            "http://example.com/file.pbf", stream=True, timeout=(30, None)
        )
        assert result == output
        assert output.exists()

    def test_overwrites_existing_file(self, tmp_path):
        """Should call requests.get even if file already exists when overwrite=True."""
        output = tmp_path / "out.pbf"
        output.write_bytes(b"old content")

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {}
        mock_resp.iter_content = MagicMock(return_value=[b"new"])

        with patch(
            "openpois.io.osm_snapshot.requests.get", return_value=mock_resp
        ) as mock_get:
            download_pbf("http://example.com/file.pbf", output, overwrite=True)

        mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# filter_pbf
# ---------------------------------------------------------------------------


class TestFilterPbf:
    def test_skips_if_output_exists_and_no_overwrite(self, tmp_path):
        """Should return early without calling subprocess.run when output exists."""
        input_pbf = tmp_path / "in.pbf"
        output_pbf = tmp_path / "out.pbf"
        input_pbf.write_bytes(b"fake")
        output_pbf.write_bytes(b"fake")

        with patch("openpois.io.osm_snapshot.subprocess.run") as mock_run:
            result = filter_pbf(input_pbf, output_pbf, ["amenity"], overwrite=False)

        mock_run.assert_not_called()
        assert result == output_pbf

    def test_runs_osmium_command_with_correct_args(self, tmp_path):
        """Should construct osmium tags-filter command with nwr/ prefixed keys."""
        input_pbf = tmp_path / "in.pbf"
        output_pbf = tmp_path / "out.pbf"
        input_pbf.write_bytes(b"fake")
        osm_keys = ["amenity", "shop", "tourism"]

        with (
            patch("openpois.io.osm_snapshot.subprocess.run") as mock_run,
            patch(
                "openpois.io.osm_snapshot.shutil.which",
                return_value = "/usr/bin/osmium",
            ),
        ):
            filter_pbf(input_pbf, output_pbf, osm_keys, overwrite=False)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "tags-filter"
        assert "-o" in cmd
        for key in osm_keys:
            assert f"nwr/{key}" in cmd
        assert mock_run.call_args[1].get("check") is True

    def test_overwrites_when_flag_set(self, tmp_path):
        """Should call subprocess.run even if output exists when overwrite=True."""
        input_pbf = tmp_path / "in.pbf"
        output_pbf = tmp_path / "out.pbf"
        input_pbf.write_bytes(b"fake")
        output_pbf.write_bytes(b"existing")

        with (
            patch("openpois.io.osm_snapshot.subprocess.run") as mock_run,
            patch(
                "openpois.io.osm_snapshot.shutil.which",
                return_value = "/usr/bin/osmium",
            ),
        ):
            filter_pbf(input_pbf, output_pbf, ["amenity"], overwrite=True)

        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# POIRecordBuilder.process_node
# ---------------------------------------------------------------------------


class TestPOIRecordBuilderNode:
    def test_returns_record_when_tag_matches(self):
        """process_node() should return a record dict when a filter key matches."""
        builder = POIRecordBuilder(
            filter_keys=["amenity", "shop"],
            extract_keys=["amenity", "shop"],
            source_label="osm_test",
        )
        node = _make_node(
            osm_id=42,
            lon=-122.3,
            lat=47.6,
            tag_dict={"amenity": "restaurant", "name": "Good Eats"},
        )
        rec = builder.process_node(node)

        assert rec is not None
        assert rec["osm_id"] == 42
        assert rec["osm_type"] == "node"
        assert rec["source"] == "osm_test"
        assert rec["amenity"] == "restaurant"
        assert rec["shop"] is None  # in extract_keys but absent from tags
        assert rec["name"] == "Good Eats"
        assert isinstance(rec["geometry"], Point)
        assert rec["geometry"].x == pytest.approx(-122.3)
        assert rec["geometry"].y == pytest.approx(47.6)

    def test_returns_none_when_no_matching_tag(self):
        """process_node() should return None when no filter key is present."""
        builder = POIRecordBuilder(filter_keys=["amenity"], source_label="osm")
        node = _make_node(osm_id=99, lon=0.0, lat=0.0, tag_dict={"highway": "crossing"})
        assert builder.process_node(node) is None


# ---------------------------------------------------------------------------
# POIRecordBuilder.process_way
# ---------------------------------------------------------------------------


class TestPOIRecordBuilderWay:
    def test_closed_way_produces_polygon(self):
        """A closed way with >= 4 coords should produce a Polygon geometry."""
        builder = POIRecordBuilder(
            filter_keys=["amenity"], extract_keys=["amenity"], source_label="osm"
        )
        coords = [(-122.0, 47.0), (-121.9, 47.0), (-121.9, 47.1), (-122.0, 47.0)]
        way = _make_way(osm_id=10, coords=coords, tag_dict={"amenity": "school"})
        rec = builder.process_way(way)

        assert rec is not None
        assert isinstance(rec["geometry"], Polygon)
        assert rec["osm_type"] == "way"

    def test_open_way_produces_centroid_point(self):
        """An open way should produce a Point (centroid of LineString)."""
        builder = POIRecordBuilder(
            filter_keys=["amenity"], extract_keys=["amenity"], source_label="osm"
        )
        coords = [(-122.0, 47.0), (-121.9, 47.1), (-121.8, 47.2)]
        way = _make_way(osm_id=20, coords=coords, tag_dict={"amenity": "parking"})
        rec = builder.process_way(way)

        assert rec is not None
        assert isinstance(rec["geometry"], Point)

    def test_invalid_location_error_returns_none(self):
        """process_way() should return None on InvalidLocationError."""
        builder = POIRecordBuilder(filter_keys=["amenity"], source_label="osm")
        way = MagicMock()
        way.id = 30
        way.tags = _make_tags({"amenity": "cafe"})

        def _raising_iter():
            raise osmium.InvalidLocationError("no location")

        way.nodes.__iter__ = MagicMock(side_effect=_raising_iter)
        assert builder.process_way(way) is None

    def test_returns_none_when_no_matching_tag(self):
        """process_way() should return None when no filter key is present."""
        builder = POIRecordBuilder(filter_keys=["amenity"], source_label="osm")
        coords = [(-122.0, 47.0), (-121.9, 47.1)]
        way = _make_way(osm_id=40, coords=coords, tag_dict={"highway": "path"})
        assert builder.process_way(way) is None

    def test_way_with_fewer_than_two_coords_returns_none(self):
        """process_way() should return None for ways with fewer than 2 coords."""
        builder = POIRecordBuilder(filter_keys=["amenity"], source_label="osm")
        coords = [(-122.0, 47.0)]
        way = _make_way(osm_id=50, coords=coords, tag_dict={"amenity": "bench"})
        assert builder.process_way(way) is None


# ---------------------------------------------------------------------------
# parse_pbf_to_geodataframe
# ---------------------------------------------------------------------------


class TestParsePbfToGeoDataFrame:
    def test_returns_empty_geodataframe_with_correct_schema_when_no_records(
        self, tmp_path
    ):
        """Should return an empty GeoDataFrame with the expected columns."""
        pbf_path = tmp_path / "empty.pbf"
        pbf_path.write_bytes(b"fake")
        extract_keys = ["amenity", "shop"]

        with patch(
            "openpois.io.osm_snapshot.osmium.FileProcessor",
            return_value=_make_file_processor([]),
        ):
            gdf = parse_pbf_to_geodataframe(
                pbf_path,
                filter_keys=extract_keys,
                extract_keys=extract_keys,
                verbose=False,
            )

        assert len(gdf) == 0
        assert gdf.crs.to_epsg() == 4326
        for col in ["source", "osm_id", "osm_type", "name", "geometry"] + extract_keys:
            assert col in gdf.columns

    def test_returns_geodataframe_from_handler_records(self, tmp_path):
        """Should build a GeoDataFrame from records produced by POIRecordBuilder."""
        pbf_path = tmp_path / "pois.pbf"
        pbf_path.write_bytes(b"fake")

        fake_node = _make_node(
            osm_id=1,
            lon=-122.0,
            lat=47.0,
            tag_dict={"amenity": "cafe", "name": "A Cafe"},
        )

        with patch(
            "openpois.io.osm_snapshot.osmium.FileProcessor",
            return_value=_make_file_processor([fake_node]),
        ):
            gdf = parse_pbf_to_geodataframe(
                pbf_path,
                filter_keys=["amenity"],
                extract_keys=["amenity"],
                source_label="osm",
                verbose=False,
            )

        assert len(gdf) == 1
        assert gdf.iloc[0]["amenity"] == "cafe"
        assert gdf.crs.to_epsg() == 4326


class TestResolveGeofabrikLatest:
    LISTING = (
        '<a href="us-260101.osm.pbf">x</a> <a href="us-260928.osm.pbf">x</a> '
        '<a href="us-260929.osm.pbf">x</a> <a href="us-260929.osm.pbf.md5">x</a> '
        '<a href="us-virgin-islands-260930.osm.pbf">x</a>'
    )

    def _response(self, text):
        resp = MagicMock()
        resp.text = text
        resp.raise_for_status.return_value = None
        return resp

    def test_picks_the_newest_dated_extract_of_that_name(self):
        from openpois.io.osm_snapshot import resolve_geofabrik_latest
        url = "https://download.geofabrik.de/north-america/us-latest.osm.pbf"
        with patch("openpois.io.osm_snapshot.requests.get",
                   return_value = self._response(self.LISTING)):
            assert resolve_geofabrik_latest(url) == (
                "https://download.geofabrik.de/north-america/us-260929.osm.pbf"
            )

    def test_other_urls_and_failures_pass_through(self):
        import requests as req
        from openpois.io.osm_snapshot import resolve_geofabrik_latest
        dated = "https://download.geofabrik.de/north-america/us-260929.osm.pbf"
        assert resolve_geofabrik_latest(dated) == dated
        url = "https://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf"
        with patch("openpois.io.osm_snapshot.requests.get",
                   return_value = self._response("<html>no files</html>")):
            assert resolve_geofabrik_latest(url) == url
        with patch("openpois.io.osm_snapshot.requests.get",
                   side_effect = req.ConnectionError("down")), \
                patch("openpois.io.osm_snapshot.time.sleep"):
            assert resolve_geofabrik_latest(url) == url
