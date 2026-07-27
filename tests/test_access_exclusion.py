"""Tests for the unnamed private/no-access exclusion mask."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "osm_snapshot" / "apply_access_exclusion.py"
)


@pytest.fixture(scope = "module")
def module():
    spec = importlib.util.spec_from_file_location("aae", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch(names, accesses):
    return pa.RecordBatch.from_arrays(
        [
            pa.array(names, type = pa.large_string()),
            pa.array(accesses, type = pa.large_string()),
        ],
        names = ["name", "access"],
    )


class TestKeepMask:
    """Only unnamed AND private/no rows are dropped."""

    @pytest.mark.parametrize(
        "name,access,expected_keep",
        [
            (None, "private", False),   # unnamed + private -> drop
            ("", "private", False),      # empty name counts as unnamed
            (None, "no", False),
            ("Pool", "private", True),   # named -> keep
            (None, "restricted", True),  # restricted is deliberately kept
            (None, "yes", True),
            (None, None, True),          # no access tag -> keep
            ("Pool", None, True),
            ("", "restricted", True),
        ],
    )
    def test_single_rows(self, module, name, access, expected_keep):
        keep = module._keep_mask(_batch([name], [access]))
        assert keep.to_pylist() == [expected_keep]

    def test_mask_is_never_null(self, module):
        """A null in the mask would make filter() drop the row silently.

        ``pc.or_`` propagates nulls rather than applying Kleene logic, so
        a null name used to yield a null mask value and every unnamed POI
        was dropped regardless of access — 2.44M rows instead of 477k.
        """
        batch = _batch(
            [None, "", "Named", None, "X"],
            [None, "private", None, "yes", "no"],
        )
        keep = module._keep_mask(batch)
        assert keep.null_count == 0

    def test_null_name_with_null_access_is_kept(self, module):
        # The regression case: unnamed but no access tag at all.
        keep = module._keep_mask(_batch([None] * 3, [None] * 3))
        assert keep.to_pylist() == [True, True, True]

    def test_counts_on_a_mixed_batch(self, module):
        names = [None, None, "A", "", "B", None]
        access = ["private", "no", "private", "yes", None, "restricted"]
        keep = module._keep_mask(_batch(names, access))
        kept = pc.sum(pc.cast(keep, pa.int64())).as_py()
        assert kept == 4  # only the first two drop


class TestIngestSideFilter:
    """The same predicate applied at snapshot build time.

    From the 2026-08 OSM pull onward the snapshot arrives already
    filtered (``download.osm.excluded_access``), making the post-hoc
    script a no-op. The two implementations must agree exactly, or a
    month's snapshot would silently differ from the one before it.
    """

    @staticmethod
    def _gdf(names, accesses):
        import geopandas as gpd
        from shapely.geometry import Point

        return gpd.GeoDataFrame(
            {
                "name": names,
                "access": accesses,
                "geometry": [Point(0, i) for i in range(len(names))],
            },
            crs = "EPSG:4326",
        )

    def test_drops_only_unnamed_and_blocked(self):
        from openpois.io.osm_snapshot import _drop_unnamed_private_rows

        gdf = self._gdf(
            [None, "", "Named", None, None, "Named2"],
            ["private", "no", "private", "restricted", None, None],
        )
        out = _drop_unnamed_private_rows(gdf, ["private", "no"])
        # Rows 0 (unnamed+private) and 1 (empty name + no) drop; the named
        # private one, the restricted one and the untagged ones all stay.
        assert out.index.tolist() == [2, 3, 4, 5]

    def test_empty_config_is_a_noop(self):
        from openpois.io.osm_snapshot import _drop_unnamed_private_rows

        gdf = self._gdf([None, None], ["private", "no"])
        assert len(_drop_unnamed_private_rows(gdf, [])) == 2
        assert len(_drop_unnamed_private_rows(gdf, None)) == 2

    def test_missing_columns_are_a_noop(self):
        import geopandas as gpd
        from shapely.geometry import Point

        from openpois.io.osm_snapshot import _drop_unnamed_private_rows

        gdf = gpd.GeoDataFrame(
            {"amenity": ["cafe"], "geometry": [Point(0, 0)]}, crs = "EPSG:4326",
        )
        assert len(_drop_unnamed_private_rows(gdf, ["private"])) == 1

    def test_agrees_with_the_post_hoc_script(self, module):
        """Ingest-side and post-hoc filters must select identical rows."""
        import pyarrow as pa

        names = [None, "", "A", None, None, "B", "", None]
        access = [
            "private", "private", "private", "no",
            "restricted", None, "yes", None,
        ]
        excluded = ["private", "no"]

        ingest = self._gdf(names, access)
        from openpois.io.osm_snapshot import _drop_unnamed_private_rows

        kept_ingest = _drop_unnamed_private_rows(ingest, excluded).index.tolist()

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(names, type = pa.large_string()),
                pa.array(access, type = pa.large_string()),
            ],
            names = ["name", "access"],
        )
        mask = module._keep_mask(batch).to_pylist()
        kept_script = [i for i, keep in enumerate(mask) if keep]

        assert kept_ingest == kept_script
