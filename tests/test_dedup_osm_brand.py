"""Tests for openpois.conflation.dedup_osm_brand (same-brand OSM dedup)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openpois.conflation import stable_ids as si
import geopandas as gpd
from shapely.geometry import Point, Polygon

from openpois.conflation.dedup_osm_brand import (
    ABSORBED_COLUMN,
    brand_key,
    drop_rows,
    element_ref,
    find_brand_duplicates,
    plan_row_updates,
    rewrite_conflated,
)

# ~1 m of latitude in degrees.
M = 1 / 111_320


def _row(source, osm, label = "Pharmacy", brand = "CVS Pharmacy", name = None,
         dlat_m = 0.0, conf = 0.8, overture_id = None, lat0 = 38.0):
    type_of = {"n": "node", "w": "way", "r": "relation"}
    return {
        "source": source,
        "osm_type": type_of[osm[0]] if osm else None,
        "osm_id": int(osm[1:]) if osm else None,
        "overture_id": overture_id,
        "shared_label": label,
        "osm_brand": brand,
        "brand": brand,
        "osm_name": name or brand,
        "name": name or brand,
        "conf_mean": conf,
        "lon": -85.7,
        "lat": lat0 + dlat_m * M,
    }


def _frame(rows):
    return pd.DataFrame(rows)


def test_brand_key_drops_generic_tokens():
    assert brand_key("CVS Pharmacy", None) == "cvs"
    assert brand_key(None, "Walgreens") == "walgreens"
    assert brand_key(None, "Pharmacy") == ""
    assert brand_key("", "Rite Aid Pharmacy") == "rite aid"


def test_cvs_building_and_counter_collapse_into_matched_row():
    frame = _frame([
        _row("matched", "w974965131", overture_id = "ovt-cvs"),
        _row("osm", "n9363968307", dlat_m = 14),
    ])
    out = find_brand_duplicates(frame)
    assert len(out) == 1
    assert out.loc[0, "loser_osm_key"] == "osm:n9363968307"
    assert out.loc[0, "winner_osm_key"] == "osm:w974965131"
    assert out.loc[0, "winner_overture_id"] == "ovt-cvs"
    assert 13 < out.loc[0, "distance_m"] < 15


def test_osm_only_pair_keeps_the_building_way():
    frame = _frame([
        _row("osm", "n5", conf = 0.95),
        _row("osm", "w7", conf = 0.70, dlat_m = 10),
    ])
    out = find_brand_duplicates(frame)
    assert out["loser_osm_key"].tolist() == ["osm:n5"]
    assert out["winner_osm_key"].tolist() == ["osm:w7"]


def test_different_brands_20m_apart_stay():
    frame = _frame([
        _row("matched", "w1", brand = "CVS Pharmacy", overture_id = "a"),
        _row("osm", "n2", brand = "Walgreens", dlat_m = 20),
    ])
    assert find_brand_duplicates(frame).empty


def test_same_brand_80m_apart_stays():
    frame = _frame([
        _row("matched", "w1", overture_id = "a"),
        _row("osm", "n2", dlat_m = 80),
    ])
    assert find_brand_duplicates(frame).empty


def test_bare_generic_name_does_not_match_a_brand():
    frame = _frame([
        _row("matched", "w1", overture_id = "a"),
        _row("osm", "n2", brand = None, name = "Pharmacy", dlat_m = 10),
    ])
    assert find_brand_duplicates(frame).empty


def test_two_matched_rows_and_overture_only_rows_are_never_absorbed():
    frame = _frame([
        _row("matched", "w1", overture_id = "a"),
        _row("matched", "n2", overture_id = "b", dlat_m = 10),
        _row("overture", None, overture_id = "c", dlat_m = 5),
    ])
    assert find_brand_duplicates(frame).empty


def test_other_labels_follow_the_labels_setting():
    rows = [
        _row("matched", "w1", label = "Gas Station", brand = "Shell", overture_id = "a"),
        _row("osm", "n2", label = "Gas Station", brand = "Shell", dlat_m = 15),
        _row("osm", "n3", label = "Convenience Store", brand = "Shell", dlat_m = 5),
    ]
    frame = _frame(rows)
    # All labels: the two Gas Station rows pair; the shop never crosses labels.
    out_all = find_brand_duplicates(frame)
    assert out_all["loser_osm_key"].tolist() == ["osm:n2"]
    # Limited to Pharmacy (the shipped default): nothing changes.
    assert find_brand_duplicates(frame, labels = ["Pharmacy"]).empty


def test_chain_through_an_absorbed_row_does_not_reach_past_radius():
    frame = _frame([
        _row("matched", "w1", overture_id = "a"),
        _row("osm", "n2", dlat_m = 40),
        _row("osm", "n3", dlat_m = 80),
    ])
    out = find_brand_duplicates(frame)
    assert out["loser_osm_key"].tolist() == ["osm:n2"]


def test_drop_rows_keeps_schema_order_and_metadata(tmp_path):
    table = pa.table({"i": np.arange(10), "s": [str(i) for i in range(10)]})
    table = table.replace_schema_metadata({b"geo": json.dumps({"v": 1}).encode()})
    path = tmp_path / "c.parquet"
    pq.write_table(table, path, row_group_size = 3)
    n = drop_rows(path, path, [1, 4, 9], chunk_rows = 4)
    out = pq.read_table(path)
    assert n == 7
    assert out["i"].to_pylist() == [0, 2, 3, 5, 6, 7, 8]
    assert out.schema.metadata[b"geo"] == table.schema.metadata[b"geo"]


def _release(path, rows):
    type_of = {"n": "node", "w": "way"}
    pq.write_table(pa.table({
        "unified_id": [f"old{i}" for i in range(len(rows))],
        "osm_id": pa.array([int(r["osm"][1:]) if r.get("osm") else None
                            for r in rows], pa.int64()),
        "osm_type": [type_of[r["osm"][0]] if r.get("osm") else None for r in rows],
        "overture_id": pa.array([r.get("ovt") for r in rows], pa.string()),
        "osm_name": ["CVS Pharmacy"] * len(rows),
        "overture_name": ["CVS"] * len(rows),
    }), path)
    return path


def _reconcile(tmp_path, release, rows, registry = None, osm_alias = None):
    path = _release(tmp_path / f"c_{release}.parquet", rows)
    con = si.connect("1GB")
    si.load_release(con, [path])
    si.load_registry(con, registry)
    alias_path = None
    if osm_alias:
        alias_path = tmp_path / f"osm_alias_{release}.parquet"
        pd.DataFrame([{
            "loser_osm_key": loser, "winner_osm_key": winner_osm,
            "winner_overture_id": winner_ovt,
        } for loser, winner_osm, winner_ovt in osm_alias]).to_parquet(alias_path)
    si.load_dedup_aliases(con, None, alias_path)
    si.reconcile(con, release)
    reg = tmp_path / f"reg_{release}.parquet"
    changes = tmp_path / f"chg_{release}.parquet"
    si.write_outputs(con, reg, changes)
    ids = dict(con.execute("SELECT rid, unified_id FROM assigned").fetchall())
    return ids, pq.read_table(changes).to_pylist(), reg


def test_absorbed_element_id_merges_into_the_kept_row(tmp_path):
    # Release 1: building way (with Overture) and counter node are two rows.
    ids1, _, reg1 = _reconcile(
        tmp_path, "20260101", [{"osm": "w1", "ovt": "X"}, {"osm": "n2"}],
    )
    # Release 2: the node is absorbed into the way's row.
    ids2, changes, _ = _reconcile(
        tmp_path, "20260201", [{"osm": "w1", "ovt": "X"}], registry = reg1,
        osm_alias = [("osm:n2", "osm:w1", "X")],
    )
    assert ids2[0] == ids1[0]
    merged = [(c["old_id"], c["new_id"]) for c in changes
              if c["change"] == "merged_into"]
    assert merged == [(ids1[1], ids1[0])]


def test_absorbed_id_follows_when_the_kept_row_is_new(tmp_path):
    # Release 1: only the counter node was mapped.
    ids1, _, reg1 = _reconcile(tmp_path, "20260101", [{"osm": "n2"}])
    # Release 2: a new matched building appears and absorbs the node.
    ids2, _, _ = _reconcile(
        tmp_path, "20260201", [{"osm": "w1", "ovt": "X"}], registry = reg1,
        osm_alias = [("osm:n2", "osm:w1", "X")],
    )
    assert ids2[0] == ids1[0]


def test_element_ref():
    assert element_ref("osm:n123") == "node/123"
    assert element_ref("osm:w45") == "way/45"
    assert element_ref("osm:r6") == "relation/6"


def test_plan_takes_the_building_for_a_kept_node():
    # Kept: matched counter node; absorbed: the building way and a stray node.
    frame = _frame([
        _row("matched", "n10", overture_id = "a"),
        _row("osm", "w20", dlat_m = 15),
        _row("osm", "n30", dlat_m = 5),
    ])
    losers = find_brand_duplicates(frame)
    absorbed, geometry_from = plan_row_updates(losers)
    assert absorbed == {0: ["node/30", "way/20"]}
    assert geometry_from == {0: 1}


def test_plan_keeps_a_way_geometry():
    frame = _frame([
        _row("matched", "w10", overture_id = "a"),
        _row("osm", "n20", dlat_m = 15),
    ])
    absorbed, geometry_from = plan_row_updates(find_brand_duplicates(frame))
    assert absorbed == {0: ["node/20"]}
    assert geometry_from == {}


def _conflated(path):
    building = Polygon([(-85.70, 38.0), (-85.70, 38.0002), (-85.6998, 38.0002),
                        (-85.6998, 38.0), (-85.70, 38.0)])
    gdf = gpd.GeoDataFrame(
        {
            "unified_id": ["a", "b", "c", "d"],
            "osm_type": ["node", "way", "node", "node"],
            "osm_id": [10, 20, 30, 40],
            "name": ["CVS Pharmacy", "CVS Pharmacy", "Other", "CVS Pharmacy"],
        },
        geometry = [Point(-85.6999, 38.0001), building,
                    Point(-85.0, 38.5), Point(-85.6999, 38.00015)],
        crs = "EPSG:4326",
    )
    gdf.to_parquet(path)
    return gdf


def test_rewrite_drops_records_and_swaps_geometry(tmp_path):
    path = tmp_path / "conflated.parquet"
    src = _conflated(path)
    meta_before = pq.read_schema(path).metadata[b"geo"]
    n = rewrite_conflated(
        path, path, drop = [1, 3],
        absorbed = {0: ["node/40", "way/20"]}, geometry_from = {0: 1},
        chunk_rows = 3,
    )
    out = gpd.read_parquet(path)
    assert n == 2
    assert out["unified_id"].tolist() == ["a", "c"]
    assert out.geometry.iloc[0].equals(src.geometry.iloc[1])
    assert out.geometry.iloc[1].equals(src.geometry.iloc[2])
    assert list(out[ABSORBED_COLUMN].iloc[0]) == ["node/40", "way/20"]
    assert out[ABSORBED_COLUMN].iloc[1] is None
    schema = pq.read_schema(path)
    assert schema.field(ABSORBED_COLUMN).type == pa.list_(pa.string())
    assert schema.metadata[b"geo"] == meta_before


def test_rewrite_adds_an_all_null_column_when_nothing_is_absorbed(tmp_path):
    path = tmp_path / "conflated.parquet"
    _conflated(path)
    rewrite_conflated(path, path)
    table = pq.read_table(path)
    assert table.num_rows == 4
    assert table.column(ABSORBED_COLUMN).null_count == 4
    # A second pass over a file that already has the column keeps one copy.
    rewrite_conflated(path, path, absorbed = {2: ["node/9"]})
    table = pq.read_table(path)
    assert table.schema.names.count(ABSORBED_COLUMN) == 1
    assert table.column(ABSORBED_COLUMN).to_pylist()[2] == ["node/9"]
