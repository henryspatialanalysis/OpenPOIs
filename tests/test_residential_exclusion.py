"""Tests for the residential-landuse exclusion of unnamed OSM POIs."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from openpois.osm.residential import (
    build_landuse_filter_exprs,
    filter_parquet_by_residential,
    load_residential_areas,
    primary_tag,
    residential_drop_mask,
    scope_mask,
)

FILTER_KEYS = [
    "shop", "healthcare", "leisure", "amenity", "tourism", "office",
    "craft", "historic", "landuse",
]
SCOPED = {"leisure": ["swimming_pool", "pitch"], "amenity": ["fountain"]}

# One unit square at the origin; (0.5, 0.5) is inside, (5, 5) is outside.
BLOCK = gpd.GeoDataFrame(geometry = [box(0, 0, 1, 1)], crs = "EPSG:4326")


def _df(rows):
    """Build a POI frame from (name, key, value, geometry) tuples."""
    cols = {k: [] for k in FILTER_KEYS}
    names, geoms = [], []
    for name, key, value, geom in rows:
        names.append(name)
        geoms.append(geom)
        for k in FILTER_KEYS:
            cols[k].append(value if k == key else None)
    return pd.DataFrame({"name": names, **cols, "geometry": geoms})


class TestBuildLanduseFilterExprs:
    def test_ways_and_relations_only(self):
        # A landuse *node* carries no area and cannot contain anything.
        assert build_landuse_filter_exprs(["residential"]) == [
            "wr/landuse=residential"
        ]

    def test_multiple_values_are_sorted_into_one_expression(self):
        assert build_landuse_filter_exprs(["retail", "residential"]) == [
            "wr/landuse=residential,retail"
        ]

    def test_empty_disables_the_step(self):
        assert build_landuse_filter_exprs([]) == []
        assert build_landuse_filter_exprs([None, ""]) == []


class TestPrimaryTag:
    """Precedence must match assign_osm_shared_label's filter_keys order."""

    def test_first_non_null_key_in_priority_order_wins(self):
        df = pd.DataFrame({
            "shop": [None, None, "bakery"],
            "leisure": [None, "pitch", "pitch"],
            "amenity": ["cafe", "cafe", "cafe"],
        })
        keys, values = primary_tag(df, ["shop", "leisure", "amenity"])
        assert list(keys) == ["amenity", "leisure", "shop"]
        assert list(values) == ["cafe", "pitch", "bakery"]

    def test_empty_string_is_not_a_tag(self):
        df = pd.DataFrame({"leisure": [""], "amenity": ["cafe"]})
        keys, _ = primary_tag(df, ["leisure", "amenity"])
        assert list(keys) == ["amenity"]

    def test_untagged_row_yields_none(self):
        df = pd.DataFrame({"leisure": [None], "amenity": [None]})
        keys, values = primary_tag(df, ["leisure", "amenity"])
        assert keys[0] is None and values[0] is None

    def test_absent_columns_are_skipped_not_raised(self):
        df = pd.DataFrame({"amenity": ["cafe"]})
        keys, _ = primary_tag(df, ["shop", "leisure", "amenity"])
        assert list(keys) == ["amenity"]


class TestScopeMask:
    def test_unnamed_and_in_scope_is_eligible(self):
        df = _df([(None, "leisure", "swimming_pool", Point(0.5, 0.5))])
        assert scope_mask(df, SCOPED, FILTER_KEYS).tolist() == [True]

    def test_named_is_never_eligible(self):
        df = _df([("Lido", "leisure", "swimming_pool", Point(0.5, 0.5))])
        assert scope_mask(df, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_empty_name_counts_as_unnamed(self):
        df = _df([("", "leisure", "swimming_pool", Point(0.5, 0.5))])
        assert scope_mask(df, SCOPED, FILTER_KEYS).tolist() == [True]

    def test_out_of_scope_value_is_not_eligible(self):
        df = _df([(None, "leisure", "park", Point(0.5, 0.5))])
        assert scope_mask(df, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_scope_applies_to_the_primary_tag_only(self):
        """A scoped value on a lower-priority key must not pull a row in.

        `leisure` outranks `amenity`, so this row's primary tag is
        leisure=park -- out of scope -- even though it also carries the
        scoped amenity=fountain.
        """
        df = pd.DataFrame({
            "name": [None], "leisure": ["park"], "amenity": ["fountain"],
        })
        assert scope_mask(df, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_empty_scoped_tags_disables_the_rule(self):
        df = _df([(None, "leisure", "swimming_pool", Point(0.5, 0.5))])
        assert scope_mask(df, {}, FILTER_KEYS).tolist() == [False]
        assert scope_mask(df, None, FILTER_KEYS).tolist() == [False]

    def test_mask_is_a_plain_null_free_bool_array(self):
        df = _df([
            (None, "leisure", "swimming_pool", Point(0.5, 0.5)),
            ("A", "amenity", "fountain", Point(0.5, 0.5)),
        ])
        m = scope_mask(df, SCOPED, FILTER_KEYS)
        assert m.dtype == np.bool_
        assert not pd.isna(m).any()


class TestResidentialDropMask:
    def test_drops_unnamed_scoped_poi_inside_a_block(self):
        df = _df([(None, "leisure", "swimming_pool", Point(0.5, 0.5))])
        assert residential_drop_mask(
            df, BLOCK, SCOPED, FILTER_KEYS).tolist() == [True]

    def test_keeps_the_same_poi_outside_every_block(self):
        df = _df([(None, "leisure", "swimming_pool", Point(5, 5))])
        assert residential_drop_mask(
            df, BLOCK, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_keeps_a_named_poi_inside_a_block(self):
        df = _df([("Lido", "leisure", "swimming_pool", Point(0.5, 0.5))])
        assert residential_drop_mask(
            df, BLOCK, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_keeps_an_out_of_scope_poi_inside_a_block(self):
        df = _df([(None, "amenity", "restaurant", Point(0.5, 0.5))])
        assert residential_drop_mask(
            df, BLOCK, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_polygon_poi_is_tested_on_its_representative_point(self):
        """Way-derived POIs are Polygons; containment uses a point inside."""
        inside = Polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
        outside = Polygon([(4, 4), (5, 4), (5, 5), (4, 5)])
        df = _df([
            (None, "leisure", "pitch", inside),
            (None, "leisure", "pitch", outside),
        ])
        assert residential_drop_mask(
            df, BLOCK, SCOPED, FILTER_KEYS).tolist() == [True, False]

    def test_representative_point_of_a_c_shape_stays_inside(self):
        """centroid would fall in the notch; point_on_surface must not."""
        c_shape = Polygon([
            (0.1, 0.1), (0.9, 0.1), (0.9, 0.3), (0.3, 0.3),
            (0.3, 0.7), (0.9, 0.7), (0.9, 0.9), (0.1, 0.9),
        ])
        rep = shapely.point_on_surface(c_shape)
        assert c_shape.contains(rep)
        df = _df([(None, "leisure", "pitch", c_shape)])
        assert residential_drop_mask(
            df, BLOCK, SCOPED, FILTER_KEYS).tolist() == [True]

    def test_overlapping_blocks_do_not_duplicate_or_break_alignment(self):
        """Abutting subdivisions overlap; the mask must stay 1:1 with rows."""
        blocks = gpd.GeoDataFrame(
            geometry = [box(0, 0, 1, 1), box(0.4, 0.4, 2, 2)],
            crs = "EPSG:4326",
        )
        df = _df([
            (None, "leisure", "pitch", Point(0.5, 0.5)),   # in both
            (None, "leisure", "pitch", Point(5, 5)),       # in neither
        ])
        m = residential_drop_mask(df, blocks, SCOPED, FILTER_KEYS)
        assert m.tolist() == [True, False]

    def test_empty_polygon_layer_drops_nothing(self):
        empty = gpd.GeoDataFrame(geometry = [], crs = "EPSG:4326")
        df = _df([(None, "leisure", "pitch", Point(0.5, 0.5))])
        assert residential_drop_mask(
            df, empty, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_missing_geometry_is_kept_not_crashed(self):
        df = _df([(None, "leisure", "pitch", None)])
        assert residential_drop_mask(
            df, BLOCK, SCOPED, FILTER_KEYS).tolist() == [False]

    def test_prebuilt_index_agrees_with_the_geodataframe(self):
        """The streaming filter passes a prebuilt tree; results must match.

        `filter_parquet_by_residential` builds the STRtree once and reuses it
        across batches -- rebuilding it per batch over a national layer would
        dominate the runtime. That is a different code path from passing the
        GeoDataFrame, so pin the two together.
        """
        df = _df([
            (None, "leisure", "pitch", Point(0.5, 0.5)),
            (None, "leisure", "pitch", Point(5, 5)),
            ("A", "leisure", "pitch", Point(0.5, 0.5)),
            (None, "amenity", "fountain", Point(0.25, 0.75)),
        ])
        from openpois.osm.residential import build_index

        direct = residential_drop_mask(df, BLOCK, SCOPED, FILTER_KEYS)
        viatree = residential_drop_mask(
            df, build_index(BLOCK), SCOPED, FILTER_KEYS,
        )
        assert direct.tolist() == viatree.tolist() == [True, False, False, True]


class TestLoadResidentialAreas:
    def test_keeps_only_areal_geometries_of_the_wanted_values(self, tmp_path):
        src = gpd.GeoDataFrame(
            {
                "landuse": ["residential", "residential", "retail"],
                "geometry": [
                    box(0, 0, 1, 1),
                    Point(0, 0),          # open way -> centroid, unusable
                    box(2, 2, 3, 3),      # wrong landuse value
                ],
            },
            crs = "EPSG:4326",
        )
        path = tmp_path / "landuse.parquet"
        src.to_parquet(path)
        out = load_residential_areas(path, ["residential"], verbose = False)
        assert len(out) == 1
        assert list(out.columns) == ["geometry"]
        assert out.crs.to_epsg() == 4326


class TestFilterParquetByResidential:
    @staticmethod
    def _write(path, rows):
        df = _df(rows)
        gdf = gpd.GeoDataFrame(
            df.drop(columns = ["geometry"]),
            geometry = df["geometry"].to_numpy(),
            crs = "EPSG:4326",
        )
        gdf.to_parquet(path)

    def test_streams_out_only_the_dropped_rows(self, tmp_path):
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        self._write(src, [
            (None, "leisure", "swimming_pool", Point(0.5, 0.5)),  # drop
            ("Lido", "leisure", "swimming_pool", Point(0.5, 0.5)),
            (None, "leisure", "swimming_pool", Point(5, 5)),
            (None, "amenity", "restaurant", Point(0.5, 0.5)),
        ])
        kept, report = filter_parquet_by_residential(
            src, dst, BLOCK, SCOPED, FILTER_KEYS, verbose = False,
        )
        assert kept == 3
        out = gpd.read_parquet(dst)
        assert len(out) == 3
        assert report["n_dropped"].sum() == 1
        assert report.iloc[0]["osm_value"] == "swimming_pool"

    def test_empty_scope_is_a_faithful_copy(self, tmp_path):
        src, dst = tmp_path / "in.parquet", tmp_path / "out.parquet"
        self._write(src, [(None, "leisure", "swimming_pool", Point(0.5, 0.5))])
        kept, report = filter_parquet_by_residential(
            src, dst, BLOCK, {}, FILTER_KEYS, verbose = False,
        )
        assert kept == 1
        assert report.empty

    def test_refuses_to_overwrite_its_own_input(self, tmp_path):
        src = tmp_path / "in.parquet"
        self._write(src, [(None, "leisure", "pitch", Point(0.5, 0.5))])
        with pytest.raises(ValueError, match = "must differ"):
            filter_parquet_by_residential(
                src, src, BLOCK, SCOPED, FILTER_KEYS, verbose = False,
            )


class TestConfigConsistency:
    """The two changes must not contradict each other.

    Change B dropped the amenity/office/leisure/tourism wildcards, so a
    scoped_tags value that is not an explicit crosswalk row would be
    unlabelled and dropped at conflation anyway -- the spatial rule would
    silently have nothing to filter.
    """

    @staticmethod
    def _config():
        from config_versioned import Config

        root = Path(__file__).resolve().parents[1]
        return Config(str(root / "config.yaml"))

    def test_every_scoped_tag_is_a_published_crosswalk_row(self):
        from openpois.conflation.taxonomy import (
            EXCLUDE_LABEL,
            load_osm_crosswalk,
        )

        cfg = self._config()
        scoped = cfg.get(
            "download", "osm", "residential_exclusion", fail_if_none = False,
        )["scoped_tags"]
        cw = load_osm_crosswalk()
        rows = cw.set_index(["osm_key", "osm_value"])["shared_label"].to_dict()
        wildcards = set(cw.loc[cw["osm_value"] == "*", "osm_key"])

        missing = []
        for key, values in scoped.items():
            for value in values:
                label = rows.get((key, value))
                if label is None and key not in wildcards:
                    missing.append(f"{key}={value} (no crosswalk row)")
                elif label == EXCLUDE_LABEL:
                    missing.append(f"{key}={value} (EXCLUDE)")
        assert not missing, f"scoped_tags with nothing to filter: {missing}"

    def test_scoped_keys_are_all_filter_keys(self):
        cfg = self._config()
        scoped = cfg.get(
            "download", "osm", "residential_exclusion", fail_if_none = False,
        )["scoped_tags"]
        filter_keys = cfg.get("download", "osm", "filter_keys")
        assert set(scoped) <= set(filter_keys)
