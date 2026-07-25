"""Tests for openpois.conflation.spill."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
from shapely.geometry import Point

from openpois.conflation.spill import spill_rows

N_ROWS = 10


@pytest.fixture
def source_gdf():
    return gpd.GeoDataFrame(
        {
            "overture_id": [f"id{i}" for i in range(N_ROWS)],
            "overture_name": [f"name{i}" for i in range(N_ROWS)],
            "confidence": np.linspace(0.1, 1.0, N_ROWS),
            "overture_phones": [
                ["+1-206-555-0100", "+1-206-555-0101"]
                if i % 2 == 0 else None
                for i in range(N_ROWS)
            ],
        },
        geometry = [Point(i, i) for i in range(N_ROWS)],
        crs = "EPSG:4326",
    )


@pytest.fixture
def source_path(source_gdf, tmp_path):
    path = tmp_path / "source.parquet"
    # Small row groups so the streaming loop crosses group boundaries
    source_gdf.to_parquet(path, row_group_size = 3)
    return path


def test_subset_preserves_order_and_values(
    source_gdf, source_path, tmp_path,
):
    dest = tmp_path / "spill.parquet"
    # Unsorted keep_rows: output must come back in SOURCE order
    keep = np.array([7, 1, 8, 4])
    n = spill_rows(
        source_path, dest, keep,
        ["overture_id", "confidence", "overture_phones", "geometry"],
    )
    assert n == 4

    out = gpd.read_parquet(dest)
    expected = source_gdf.iloc[np.sort(keep)].reset_index(
        drop = True
    )
    assert list(out["overture_id"]) == list(expected["overture_id"])
    np.testing.assert_allclose(
        out["confidence"], expected["confidence"],
    )
    assert out.crs == expected.crs
    assert list(out.geometry.geom_equals(expected.geometry)) == (
        [True] * 4
    )
    # List column round-trips (row 4 and 8 populated, 1 and 7 null)
    phones = [
        None if p is None else list(p)
        for p in out["overture_phones"]
    ]
    assert phones == [
        None,
        ["+1-206-555-0100", "+1-206-555-0101"],
        None,
        ["+1-206-555-0100", "+1-206-555-0101"],
    ]


def test_readable_with_column_subset(source_path, tmp_path):
    """The merge-phase reload reads the spill with a column subset."""
    dest = tmp_path / "spill.parquet"
    spill_rows(
        source_path, dest, np.arange(N_ROWS),
        ["overture_id", "overture_name", "geometry"],
    )
    out = gpd.read_parquet(
        dest, columns = ["overture_id", "geometry"],
    )
    assert len(out) == N_ROWS
    assert list(out.columns) == ["overture_id", "geometry"]


def test_missing_column_raises(source_path, tmp_path):
    with pytest.raises(ValueError, match = "not_a_column"):
        spill_rows(
            source_path, tmp_path / "spill.parquet",
            np.array([0]),
            ["overture_id", "not_a_column", "geometry"],
        )


def test_out_of_range_rows_raise(source_path, tmp_path):
    with pytest.raises(ValueError, match = "out of range"):
        spill_rows(
            source_path, tmp_path / "spill.parquet",
            np.array([0, N_ROWS]),
            ["overture_id", "geometry"],
        )


def test_empty_keep_rows_writes_empty_file(source_path, tmp_path):
    dest = tmp_path / "spill.parquet"
    n = spill_rows(
        source_path, dest,
        np.array([], dtype = np.int64),
        ["overture_id", "geometry"],
    )
    assert n == 0
    out = gpd.read_parquet(dest)
    assert len(out) == 0
    assert "overture_id" in out.columns


def test_zero_row_group_source(source_gdf, tmp_path):
    """A source with no row groups still yields a valid empty file."""
    src = tmp_path / "empty_source.parquet"
    schema = pq.read_schema(
        _write_gdf(source_gdf, tmp_path / "schema_donor.parquet")
    )
    pq.ParquetWriter(src, schema).close()

    dest = tmp_path / "spill.parquet"
    n = spill_rows(
        src, dest,
        np.array([], dtype = np.int64),
        ["overture_id", "geometry"],
    )
    assert n == 0
    out = gpd.read_parquet(dest)
    assert len(out) == 0


def _write_gdf(gdf, path):
    gdf.to_parquet(path)
    return path


def test_metadata_keeps_geo_drops_pandas(source_path, tmp_path):
    dest = tmp_path / "spill.parquet"
    spill_rows(
        source_path, dest, np.array([2, 3]),
        ["overture_id", "geometry"],
    )
    meta = pq.read_schema(dest).metadata or {}
    assert b"geo" in meta
    assert b"pandas" not in meta


def test_driver_alignment(source_gdf, source_path, tmp_path):
    """Mimic the conflate.py flow: narrow match-column load, dedup
    mask, spill from source, merge-column reload. Row ``i`` of the
    reload must be the same POI as row ``i`` of the in-memory
    post-dedup frame.
    """
    match_cols = ["overture_id", "confidence", "geometry"]
    merge_cols = ["overture_id", "overture_phones", "geometry"]
    narrow = gpd.read_parquet(source_path, columns = match_cols)
    source_rows = np.arange(len(narrow), dtype = np.int64)

    keep_mask = np.ones(len(narrow), dtype = bool)
    keep_mask[[2, 5, 6]] = False
    in_memory = narrow.loc[keep_mask].reset_index(drop = True)

    dest = tmp_path / "spill.parquet"
    spill_cols = match_cols + [
        c for c in merge_cols if c not in match_cols
    ]
    n = spill_rows(
        source_path, dest, source_rows[keep_mask], spill_cols,
    )
    assert n == len(in_memory)

    reloaded = gpd.read_parquet(dest, columns = merge_cols)
    assert list(reloaded["overture_id"]) == list(
        in_memory["overture_id"]
    )
    # Merge-only column present with real (non-null) values
    src_phones = source_gdf.loc[keep_mask, "overture_phones"]
    assert [
        p is not None for p in reloaded["overture_phones"]
    ] == [p is not None for p in src_phones]
