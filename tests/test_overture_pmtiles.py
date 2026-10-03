"""Tests for scripts/overture/prepare_pmtiles.py (Overture web tiles + filter groups)."""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "overture" / "prepare_pmtiles.py"


@pytest.fixture(scope = "module")
def prep():
    spec = importlib.util.spec_from_file_location("overture_prepare_pmtiles", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["overture_prepare_pmtiles"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(tmp_path):
    n = 6
    gdf = gpd.GeoDataFrame(
        {
            "overture_id": [f"id{i}" for i in range(n)],
            "overture_name": ["A", "B", None, "D", "E", "F"],
            "brand_name": [None] * n,
            "confidence": [0.9, 0.5, 0.7, 0.2, 0.99, 0.4],
            "taxonomy_l0": ["food_and_drink"] * 3 + ["shopping"] * 2 + ["food_and_drink"],
            "basic_category": [
                "casual_eatery", "casual_eatery", None, "pharmacy", "casual_eatery", "bakery",
            ],
            "taxonomy_primary": ["x", "y", "z", "w", "v", "u"],
            "taxonomy_hierarchy": [
                ["food_and_drink", "restaurant", "x"], ["food_and_drink", "y"], [], None,
                ["shopping", "v"], ["food_and_drink", "u"],
            ],
            "overture_addr_street": ["1 Main"] + [None] * (n - 1),
            "overture_addr_city": [None] * n,
            "overture_addr_state": [None] * n,
            "overture_websites": [["https://a"], [], None, ["https://d", "https://e"], None, None],
            "overture_phones": [None] * n,
        },
        geometry = [Point(-122.33 + i * 0.001, 47.68) for i in range(n)],
        crs = "EPSG:4326",
    )
    path = tmp_path / "overture_snapshot.parquet"
    gdf.to_parquet(path)
    return path


def test_category_groups_file_basic_category_under_modal_l0(prep, snapshot):
    result = prep.build_category_groups(snapshot)
    assert result["total"] == 6
    groups = {g["l0"]: g for g in result["groups"]}
    assert set(groups) == {"food_and_drink", "shopping"}

    # casual_eatery: 2 rows under food_and_drink, 1 under shopping -> all 3 filed
    # under food_and_drink; the null basic category stays in its own L0 as None.
    food = {c["key"]: c["count"] for c in groups["food_and_drink"]["categories"]}
    assert food == {"bakery": 1, "casual_eatery": 3, None: 1}
    assert groups["food_and_drink"]["count"] == 5
    assert groups["shopping"]["categories"] == [{"key": "pharmacy", "count": 1}]
    # Counts across groups add back up to the snapshot.
    assert sum(g["count"] for g in result["groups"]) == result["total"]


def test_tile_properties_flatten_list_columns(prep, snapshot):
    df = pd.read_parquet(snapshot).drop(columns = "geometry")
    out = prep.to_tile_properties(df)
    assert list(out["id"]) == [f"id{i}" for i in range(6)]
    assert out.loc[0, "taxonomy_hierarchy"] == "food_and_drink > restaurant > x"
    assert out["taxonomy_hierarchy"].iloc[2:4].isna().all()   # empty list, null
    assert out.loc[0, "website"] == "https://a"
    assert out.loc[3, "website"] == "https://d"
    assert out["website"].iloc[[1, 2]].isna().all()
    assert out.loc[0, "addr_street"] == "1 Main"
    # Every configured snapshot column is consumed by the transform.
    assert set(prep.PROPERTIES) <= set(df.columns)
