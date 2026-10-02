#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Unit tests for openpois.io.overture.

All external calls (requests.get for S3 listing, duckdb queries) are mocked
so tests run without network access.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from openpois.io import overture as overture_module
from openpois.io.overture import (
    _list_overture_part_keys,
    build_overture_s3_path,
    check_overture_confidence,
    download_overture_snapshot,
    get_latest_release_date,
)


def _rect_boundary_gdf(
    xmin: float, ymin: float, xmax: float, ymax: float
) -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame containing a WGS-84 rectangle."""
    return gpd.GeoDataFrame(
        geometry = [box(xmin, ymin, xmax, ymax)],
        crs = "EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Minimal S3 XML list-type=2 response with two release prefixes.
_S3_XML_TWO_RELEASES = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Name>overturemaps-us-west-2</Name>
      <CommonPrefixes><Prefix>release/2025-11-13.0/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>release/2026-02-18.0/</Prefix></CommonPrefixes>
    </ListBucketResult>
""")

_S3_XML_EMPTY_RELEASES = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Name>overturemaps-us-west-2</Name>
    </ListBucketResult>
""")


def _parts_xml(keys: list[str], truncated: bool = False) -> str:
    """Build a minimal list-type=2 XML response listing the given keys."""
    contents = "\n".join(
        f"<Contents><Key>{k}</Key><Size>123</Size></Contents>"
        for k in keys
    )
    truncated_str = "true" if truncated else "false"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">\n'
        "<Name>overturemaps-us-west-2</Name>\n"
        f"<IsTruncated>{truncated_str}</IsTruncated>\n"
        f"{contents}\n"
        "</ListBucketResult>\n"
    )


def _mock_requests_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# get_latest_release_date
# ---------------------------------------------------------------------------


class TestGetLatestReleaseDate:
    def test_returns_lexicographically_latest_date(self):
        """Should return the largest date string when multiple releases exist."""
        mock_resp = _mock_requests_response(_S3_XML_TWO_RELEASES)

        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ):
            result = get_latest_release_date("overturemaps-us-west-2")

        assert result == "2026-02-18.0"

    def test_raises_value_error_when_no_prefixes(self):
        """Should raise ValueError when the S3 bucket lists no release prefixes."""
        mock_resp = _mock_requests_response(_S3_XML_EMPTY_RELEASES)

        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ):
            with pytest.raises(ValueError, match = "No release prefixes found"):
                get_latest_release_date("overturemaps-us-west-2")

    def test_raises_on_http_error(self):
        """Should propagate HTTPError from raise_for_status."""
        import requests as _req

        mock_resp = _mock_requests_response("", status_code = 403)
        mock_resp.raise_for_status = MagicMock(
            side_effect = _req.HTTPError("403 Forbidden")
        )

        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ):
            with pytest.raises(_req.HTTPError):
                get_latest_release_date("overturemaps-us-west-2")

    def test_queries_correct_s3_url(self):
        """Should construct the S3 list-type=2 URL with the given bucket name."""
        mock_resp = _mock_requests_response(_S3_XML_TWO_RELEASES)

        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ) as mock_get:
            get_latest_release_date("my-bucket")

        url = mock_get.call_args[0][0]
        assert "my-bucket.s3.amazonaws.com" in url
        assert "list-type=2" in url
        assert "prefix=release" in url


# ---------------------------------------------------------------------------
# build_overture_s3_path
# ---------------------------------------------------------------------------


class TestBuildOvertureS3Path:
    def test_returns_expected_path_format(self):
        """Should embed bucket, release date, and place paths correctly."""
        result = build_overture_s3_path(
            release_date = "2026-02-18.0",
            bucket = "overturemaps-us-west-2",
        )
        assert result == (
            "s3://overturemaps-us-west-2/release/2026-02-18.0"
            "/theme=places/type=place/*.parquet"
        )

    def test_different_bucket(self):
        """Should use whatever bucket name is supplied."""
        result = build_overture_s3_path(
            release_date = "2025-01-01.0",
            bucket = "custom-bucket",
        )
        assert result.startswith("s3://custom-bucket/")
        assert "2025-01-01.0" in result


# ---------------------------------------------------------------------------
# _list_overture_part_keys
# ---------------------------------------------------------------------------


class TestListOvertureParts:
    _SAMPLE_KEYS = [
        f"release/2026-02-18.0/theme=places/type=place/part-{i:05d}.parquet"
        for i in range(3)
    ]

    def test_returns_sorted_parquet_keys(self):
        """Should parse Contents/Key entries ending in .parquet, sorted."""
        mock_resp = _mock_requests_response(
            _parts_xml(self._SAMPLE_KEYS, truncated = False)
        )
        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ):
            keys = _list_overture_part_keys(
                release_date = "2026-02-18.0",
                bucket = "overturemaps-us-west-2",
            )
        assert keys == sorted(self._SAMPLE_KEYS)

    def test_filters_non_parquet_keys(self):
        """Non-.parquet keys in the listing should be dropped."""
        mixed = self._SAMPLE_KEYS + [
            "release/2026-02-18.0/theme=places/type=place/_SUCCESS"
        ]
        mock_resp = _mock_requests_response(
            _parts_xml(mixed, truncated = False)
        )
        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ):
            keys = _list_overture_part_keys(
                release_date = "2026-02-18.0",
                bucket = "overturemaps-us-west-2",
            )
        assert all(k.endswith(".parquet") for k in keys)
        assert len(keys) == len(self._SAMPLE_KEYS)

    def test_raises_when_empty(self):
        """Empty listing should raise ValueError."""
        mock_resp = _mock_requests_response(_parts_xml([], truncated = False))
        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ):
            with pytest.raises(ValueError, match = "No part Parquet files"):
                _list_overture_part_keys(
                    release_date = "2026-02-18.0",
                    bucket = "overturemaps-us-west-2",
                )

    def test_raises_when_truncated(self):
        """Truncated listing should raise (pagination not implemented)."""
        mock_resp = _mock_requests_response(
            _parts_xml(self._SAMPLE_KEYS, truncated = True)
        )
        with patch(
            "openpois.io.overture.requests.get", return_value = mock_resp
        ):
            with pytest.raises(ValueError, match = "truncated"):
                _list_overture_part_keys(
                    release_date = "2026-02-18.0",
                    bucket = "overturemaps-us-west-2",
                )


# ---------------------------------------------------------------------------
# download_overture_snapshot
# ---------------------------------------------------------------------------


def _write_fake_snapshot(path, confidence = (0.9, 0.5)) -> None:
    """Stand-in for the final DuckDB COPY: just the column the check reads."""
    table = pa.table({"confidence": pa.array(confidence, type = pa.float64())})
    pq.write_table(table, path)


def _fake_part_key(i: int, release: str = "2026-02-18.0") -> str:
    return f"release/{release}/theme=places/type=place/part-{i:05d}.parquet"


class TestDownloadOvertureSnapshot:
    _PART_KEYS = [_fake_part_key(0), _fake_part_key(1)]

    def _write_dummy_intermediate(self, path: Path) -> None:
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"dummy")

    def test_calls_download_one_part_for_each_key(self, tmp_path):
        """Should call _download_one_part once per listed part key."""
        output = tmp_path / "overture.parquet"
        coarse_bboxes = [
            {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
        ]
        boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0)

        def fake_download_one_part(intermediate_path, **_kwargs):
            self._write_dummy_intermediate(intermediate_path)

        def fake_finalize(output_path, **_kwargs):
            _write_fake_snapshot(output_path)

        with patch.object(
            overture_module,
            "_list_overture_part_keys",
            return_value = self._PART_KEYS,
        ) as mock_list, patch.object(
            overture_module,
            "_download_one_part",
            side_effect = fake_download_one_part,
        ) as mock_download, patch.object(
            overture_module,
            "_finalize_snapshot_in_duckdb",
            side_effect = fake_finalize,
        ) as mock_finalize:
            result = download_overture_snapshot(
                output_path = output,
                taxonomy_allowlist = [("eat_and_drink", None)],
                boundary_gdf = boundary_gdf,
                coarse_bboxes = coarse_bboxes,
                bucket = "overturemaps-us-west-2",
                s3_region = "us-west-2",
                release_date = "2026-02-18.0",
            )

        assert result == output
        mock_list.assert_called_once()
        assert mock_download.call_count == len(self._PART_KEYS)
        mock_finalize.assert_called_once()
        assert output.exists()

    def test_skips_existing_intermediates_on_resume(self, tmp_path):
        """Pre-existing, non-empty intermediate should be skipped."""
        output = tmp_path / "overture.parquet"
        coarse_bboxes = [
            {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
        ]
        boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0)

        release = "2026-02-18.0"
        parts_dir = output.parent / ".parts" / release
        parts_dir.mkdir(parents = True)
        # Pre-populate the intermediate for the first part.
        (parts_dir / "part-00000.parquet").write_bytes(b"already-here")

        def fake_download_one_part(intermediate_path, **_kwargs):
            self._write_dummy_intermediate(intermediate_path)

        def fake_finalize(output_path, **_kwargs):
            _write_fake_snapshot(output_path)

        with patch.object(
            overture_module,
            "_list_overture_part_keys",
            return_value = self._PART_KEYS,
        ), patch.object(
            overture_module,
            "_download_one_part",
            side_effect = fake_download_one_part,
        ) as mock_download, patch.object(
            overture_module,
            "_finalize_snapshot_in_duckdb",
            side_effect = fake_finalize,
        ):
            download_overture_snapshot(
                output_path = output,
                taxonomy_allowlist = [("eat_and_drink", None)],
                boundary_gdf = boundary_gdf,
                coarse_bboxes = coarse_bboxes,
                bucket = "overturemaps-us-west-2",
                s3_region = "us-west-2",
                release_date = release,
            )

        # Only the second part should have been downloaded.
        assert mock_download.call_count == 1
        downloaded_path = mock_download.call_args.kwargs["intermediate_path"]
        assert downloaded_path.name == "part-00001.parquet"

    def test_cleans_up_parts_dir_on_success(self, tmp_path):
        """On successful finalize, .parts/<release>/ should be removed."""
        output = tmp_path / "overture.parquet"
        release = "2026-02-18.0"
        parts_dir = output.parent / ".parts" / release

        def fake_download_one_part(intermediate_path, **_kwargs):
            self._write_dummy_intermediate(intermediate_path)

        def fake_finalize(output_path, **_kwargs):
            _write_fake_snapshot(output_path)

        with patch.object(
            overture_module,
            "_list_overture_part_keys",
            return_value = self._PART_KEYS,
        ), patch.object(
            overture_module,
            "_download_one_part",
            side_effect = fake_download_one_part,
        ), patch.object(
            overture_module,
            "_finalize_snapshot_in_duckdb",
            side_effect = fake_finalize,
        ):
            download_overture_snapshot(
                output_path = output,
                taxonomy_allowlist = [("eat_and_drink", None)],
                boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0),
                coarse_bboxes = [
                    {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
                ],
                bucket = "overturemaps-us-west-2",
                s3_region = "us-west-2",
                release_date = release,
            )

        assert not parts_dir.exists()

    def test_leaves_intermediates_on_finalize_failure(self, tmp_path):
        """If finalize raises, intermediates should remain for resume."""
        output = tmp_path / "overture.parquet"
        release = "2026-02-18.0"
        parts_dir = output.parent / ".parts" / release

        def fake_download_one_part(intermediate_path, **_kwargs):
            self._write_dummy_intermediate(intermediate_path)

        def failing_finalize(**_kwargs):
            raise RuntimeError("boom")

        with patch.object(
            overture_module,
            "_list_overture_part_keys",
            return_value = self._PART_KEYS,
        ), patch.object(
            overture_module,
            "_download_one_part",
            side_effect = fake_download_one_part,
        ), patch.object(
            overture_module,
            "_finalize_snapshot_in_duckdb",
            side_effect = failing_finalize,
        ):
            with pytest.raises(RuntimeError, match = "boom"):
                download_overture_snapshot(
                    output_path = output,
                    taxonomy_allowlist = [("eat_and_drink", None)],
                    boundary_gdf = _rect_boundary_gdf(
                        -125.0, 24.0, -66.0, 50.0
                    ),
                    coarse_bboxes = [
                        {"xmin": -125.0, "ymin": 24.0,
                         "xmax": -66.0, "ymax": 50.0}
                    ],
                    bucket = "overturemaps-us-west-2",
                    s3_region = "us-west-2",
                    release_date = release,
                )

        assert parts_dir.exists()
        intermediates = sorted(p.name for p in parts_dir.glob("part-*.parquet"))
        assert intermediates == ["part-00000.parquet", "part-00001.parquet"]

    def test_fetches_latest_release_when_not_provided(self, tmp_path):
        """Should call get_latest_release_date when release_date is None."""
        output = tmp_path / "overture.parquet"

        def fake_download_one_part(intermediate_path, **_kwargs):
            self._write_dummy_intermediate(intermediate_path)

        def fake_finalize(output_path, **_kwargs):
            _write_fake_snapshot(output_path)

        with patch.object(
            overture_module,
            "get_latest_release_date",
            return_value = "2026-02-18.0",
        ) as mock_latest, patch.object(
            overture_module,
            "_list_overture_part_keys",
            return_value = self._PART_KEYS,
        ), patch.object(
            overture_module,
            "_download_one_part",
            side_effect = fake_download_one_part,
        ), patch.object(
            overture_module,
            "_finalize_snapshot_in_duckdb",
            side_effect = fake_finalize,
        ):
            download_overture_snapshot(
                output_path = output,
                taxonomy_allowlist = [("eat_and_drink", None)],
                boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0),
                coarse_bboxes = [
                    {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
                ],
                bucket = "overturemaps-us-west-2",
                s3_region = "us-west-2",
                release_date = None,
            )

        mock_latest.assert_called_once_with(bucket = "overturemaps-us-west-2")

    def test_per_part_sql_uses_single_file_s3_uri(self, tmp_path):
        """Per-part SQL should reference a single S3 URI, not the glob."""
        output = tmp_path / "overture.parquet"
        captured_uris: list[str] = []

        def capture_download_one_part(part_s3_uri, intermediate_path, **_kwargs):
            captured_uris.append(part_s3_uri)
            self._write_dummy_intermediate(intermediate_path)

        def fake_finalize(output_path, **_kwargs):
            _write_fake_snapshot(output_path)

        with patch.object(
            overture_module,
            "_list_overture_part_keys",
            return_value = self._PART_KEYS,
        ), patch.object(
            overture_module,
            "_download_one_part",
            side_effect = capture_download_one_part,
        ), patch.object(
            overture_module,
            "_finalize_snapshot_in_duckdb",
            side_effect = fake_finalize,
        ):
            download_overture_snapshot(
                output_path = output,
                taxonomy_allowlist = [("eat_and_drink", None)],
                boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0),
                coarse_bboxes = [
                    {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
                ],
                bucket = "overturemaps-us-west-2",
                s3_region = "us-west-2",
                release_date = "2026-02-18.0",
            )

        assert captured_uris == [
            f"s3://overturemaps-us-west-2/{k}" for k in self._PART_KEYS
        ]
        for uri in captured_uris:
            assert "*" not in uri

    def test_rejects_workers_lt_one(self, tmp_path):
        """workers must be >= 1."""
        with pytest.raises(ValueError, match = "workers must be >= 1"):
            download_overture_snapshot(
                output_path = tmp_path / "overture.parquet",
                taxonomy_allowlist = [("eat_and_drink", None)],
                boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0),
                coarse_bboxes = [
                    {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
                ],
                bucket = "overturemaps-us-west-2",
                s3_region = "us-west-2",
                release_date = "2026-02-18.0",
                workers = 0,
            )

    def test_parallel_workers_download_all_parts(self, tmp_path):
        """workers > 1 should still result in every part being downloaded."""
        output = tmp_path / "overture.parquet"
        many_keys = [_fake_part_key(i) for i in range(4)]

        def fake_download_one_part(intermediate_path, **_kwargs):
            self._write_dummy_intermediate(intermediate_path)

        def fake_finalize(output_path, **_kwargs):
            _write_fake_snapshot(output_path)

        with patch.object(
            overture_module,
            "_list_overture_part_keys",
            return_value = many_keys,
        ), patch.object(
            overture_module,
            "_download_one_part",
            side_effect = fake_download_one_part,
        ) as mock_download, patch.object(
            overture_module,
            "_finalize_snapshot_in_duckdb",
            side_effect = fake_finalize,
        ):
            download_overture_snapshot(
                output_path = output,
                taxonomy_allowlist = [("eat_and_drink", None)],
                boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0),
                coarse_bboxes = [
                    {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
                ],
                bucket = "overturemaps-us-west-2",
                s3_region = "us-west-2",
                release_date = "2026-02-18.0",
                workers = 3,
            )

        assert mock_download.call_count == len(many_keys)
        assert output.exists()


class TestCheckOvertureConfidence:
    def test_accepts_valid_scores_including_one_half(self):
        check_overture_confidence(np.array([0.0, 0.5, 0.91, 1.0]))

    @pytest.mark.parametrize(
        "values, missing, out_of_range",
        [
            ([0.9, np.nan], 1, 0),
            ([0.9, None], 1, 0),
            (["0.9", "high"], 1, 0),
            ([0.9, 1.2, -0.1], 0, 2),
        ],
    )
    def test_rejects_missing_and_out_of_range(self, values, missing, out_of_range):
        with pytest.raises(ValueError) as err:
            check_overture_confidence(values, label = "test")
        assert f"{missing:,} of {len(values):,} rows have no confidence" in str(err.value)
        assert f"{out_of_range:,} are outside [0, 1]" in str(err.value)

    def test_download_fails_on_a_snapshot_with_missing_confidence(self, tmp_path):
        """The check runs before cleanup, so the parts stay for inspection."""
        output = tmp_path / "overture.parquet"

        def fake_download_one_part(intermediate_path, **_kwargs):
            intermediate_path.parent.mkdir(parents = True, exist_ok = True)
            intermediate_path.write_bytes(b"dummy")

        def fake_finalize(output_path, **_kwargs):
            _write_fake_snapshot(output_path, confidence = (0.9, None))

        with patch.object(
            overture_module, "_list_overture_part_keys",
            return_value = [_fake_part_key(0)],
        ), patch.object(
            overture_module, "_download_one_part",
            side_effect = fake_download_one_part,
        ), patch.object(
            overture_module, "_finalize_snapshot_in_duckdb",
            side_effect = fake_finalize,
        ):
            with pytest.raises(ValueError, match = "no confidence"):
                download_overture_snapshot(
                    output_path = output,
                    taxonomy_allowlist = [("eat_and_drink", None)],
                    boundary_gdf = _rect_boundary_gdf(-125.0, 24.0, -66.0, 50.0),
                    coarse_bboxes = [
                        {"xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0}
                    ],
                    bucket = "overturemaps-us-west-2",
                    s3_region = "us-west-2",
                    release_date = "2026-02-18.0",
                )
        assert any(tmp_path.rglob("part-00000.parquet"))
