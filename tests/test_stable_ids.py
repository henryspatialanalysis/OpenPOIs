"""Tests for openpois.conflation.stable_ids (release-stable unified_id)."""
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpois.conflation import stable_ids as si


def _release(path, rows, osm_id_type = pa.int64()):
    """Write a conflated-shaped parquet. ``rows`` are dicts with any of
    ``osm`` ("n9"), ``ovt`` ("X"), ``osm_name``, ``ovt_name``."""
    type_of = {"n": "node", "w": "way", "r": "relation"}
    osm_type, osm_id, ovt = [], [], []
    for r in rows:
        if r.get("osm"):
            osm_type.append(type_of[r["osm"][0]])
            osm_id.append(int(r["osm"][1:]))
        else:
            osm_type.append(None)
            osm_id.append(None)
        ovt.append(r.get("ovt"))
    table = pa.table({
        "unified_id": pa.array([f"old{i}" for i in range(len(rows))], pa.large_string()),
        "osm_id": pa.array(osm_id, pa.int64()).cast(osm_id_type),
        "osm_type": pa.array(osm_type, pa.string()),
        "overture_id": pa.array(ovt, pa.string()),
        "osm_name": pa.array([r.get("osm_name") for r in rows], pa.string()),
        "overture_name": pa.array([r.get("ovt_name") for r in rows], pa.string()),
        "conf_mean": pa.array([0.5 + i / 100 for i in range(len(rows))]),
    })
    table = table.replace_schema_metadata({b"geo": json.dumps({"v": 1}).encode()})
    pq.write_table(table, path)
    return path


def _run(tmp_path, release, rows, registry = None, aliases = None, **kwargs):
    """Reconcile one release; return ({row index: id}, changes rows, stats,
    registry path)."""
    path = _release(tmp_path / f"conflated_{release}.parquet", rows)
    con = si.connect("1GB")
    si.load_release(con, [path])
    si.load_registry(con, registry)
    alias_path = None
    if aliases:
        alias_path = tmp_path / f"dedup_{release}.parquet"
        pq.write_table(pa.table({
            "overture_id": [a for a, _ in aliases],
            "winner_overture_id": [w for _, w in aliases],
        }), alias_path)
    si.load_dedup_aliases(con, alias_path)
    stats = si.reconcile(con, release, **kwargs)
    reg_path = tmp_path / f"registry_{release}.parquet"
    changes_path = tmp_path / f"changes_{release}.parquet"
    si.write_outputs(con, reg_path, changes_path)
    ids = dict(con.execute(
        "SELECT rid, unified_id FROM assigned ORDER BY rid"
    ).fetchall())
    changes = pq.read_table(changes_path).to_pylist()
    return ids, changes, stats, reg_path, con, path


def _change(changes, kind):
    return [(c["old_id"], c["new_id"]) for c in changes if c["change"] == kind]


# -----------------------------------------------------------------
# Minting
# -----------------------------------------------------------------


def test_mint_id_shape_and_determinism():
    a = si.mint_id("ovt:X")
    assert a == si.mint_id("ovt:X")
    assert a.startswith("opoi_") and len(a) == len("opoi_") + 16
    assert a[5:].isalnum() and a[5:] == a[5:].lower()
    assert si.mint_id("ovt:X", salt = 1) != a
    assert si.mint_id("osm:n1") != si.mint_id("osm:w1")


def test_bootstrap_mints_from_founding_key(tmp_path):
    ids, changes, stats, *_ = _run(tmp_path, "20260101", [
        {"osm": "n9", "ovt": "X"}, {"ovt": "Y"}, {"osm": "w9"},
    ])
    assert ids[0] == si.mint_id("osm:n9")
    assert ids[1] == si.mint_id("ovt:Y")
    assert ids[2] == si.mint_id("osm:w9")
    assert stats["n_minted"] == 3 and stats["changes"] == {"minted": 3}


def test_node_and_way_with_same_number_get_distinct_ids(tmp_path):
    ids, *_ = _run(tmp_path, "20260101", [{"osm": "n5"}, {"osm": "w5"}])
    assert ids[0] != ids[1]


# -----------------------------------------------------------------
# Carry-forward
# -----------------------------------------------------------------


def test_overture_only_becomes_matched_keeps_id(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X"}])
    ids2, changes, *_ = _run(
        tmp_path, "20260201", [{"osm": "n9", "ovt": "X"}], registry = reg1,
    )
    assert ids2[0] == ids1[0]
    assert changes == []


def test_partner_replaced_twice_keeps_id(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"osm": "n9", "ovt": "X"}])
    ids2, _, _, reg2, *_ = _run(
        tmp_path, "20260201", [{"osm": "n9", "ovt": "Y"}], registry = reg1,
    )
    ids3, *_ = _run(
        tmp_path, "20260301", [{"osm": "n7", "ovt": "Y"}], registry = reg2,
    )
    assert ids1[0] == ids2[0] == ids3[0]


def test_matched_to_osm_only_and_back(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"osm": "n9", "ovt": "X"}])
    ids2, changes, *_ = _run(tmp_path, "20260201", [{"osm": "n9"}], registry = reg1)
    assert ids2[0] == ids1[0]
    assert changes == []


def test_split_keeps_id_on_osm_side(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X", "osm": "n9"}])
    ids2, changes, *_ = _run(
        tmp_path, "20260201", [{"ovt": "X"}, {"osm": "n9"}], registry = reg1,
    )
    assert ids2[1] == ids1[0]
    assert ids2[0] != ids1[0]
    assert _change(changes, "split_from") == [(ids1[0], ids2[0])]


def test_merge_keeps_osm_side_id(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X"}, {"osm": "n9"}])
    ids2, changes, *_ = _run(
        tmp_path, "20260201", [{"osm": "n9", "ovt": "X"}], registry = reg1,
    )
    assert ids2[0] == ids1[1]
    assert _change(changes, "merged_into") == [(ids1[0], ids1[1])]


def test_resplit_after_merge_does_not_revive_merged_id(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X"}, {"osm": "n9"}])
    _, _, _, reg2, *_ = _run(
        tmp_path, "20260201", [{"osm": "n9", "ovt": "X"}], registry = reg1,
    )
    ids3, *_ = _run(
        tmp_path, "20260301", [{"ovt": "X"}, {"osm": "n9"}], registry = reg2,
    )
    assert ids3[1] == ids1[1]
    # hash(ovt:X) is the tombstoned id; minting must re-salt past it.
    assert ids3[0] not in (ids1[0], ids1[1])
    assert ids3[0] == si.mint_id("ovt:X", salt = 1)
    reg = pq.read_table(reg2).to_pylist()
    assert {"source_key": None, "unified_id": ids1[0]}.items() <= next(
        r for r in reg if r["unified_id"] == ids1[0]
    ).items()


def test_key_absent_one_release_then_back_recovers_id(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X"}, {"ovt": "Z"}])
    _, changes2, _, reg2, *_ = _run(tmp_path, "20260201", [{"ovt": "Z"}], registry = reg1)
    assert _change(changes2, "retired") == [(ids1[0], None)]
    ids3, changes3, *_ = _run(
        tmp_path, "20260301", [{"ovt": "X"}, {"ovt": "Z"}], registry = reg2,
    )
    assert ids3[0] == ids1[0]
    assert _change(changes3, "revived") == [(ids1[0], ids1[0])]


def test_stale_key_loses_to_current_key(tmp_path):
    # A = (n9, Y); then (n7, Y): n9 is now a stale key of A. When n9 comes
    # back as its own row while Y continues, Y's row keeps A.
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"osm": "n9", "ovt": "Y"}])
    _, _, _, reg2, *_ = _run(
        tmp_path, "20260201", [{"osm": "n7", "ovt": "Y"}], registry = reg1,
    )
    ids3, *_ = _run(
        tmp_path, "20260301", [{"osm": "n9"}, {"ovt": "Y"}], registry = reg2,
    )
    assert ids3[1] == ids1[0]
    assert ids3[0] != ids1[0]


def test_dedup_alias_recovers_id_after_winner_flip(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X"}])
    # X is now a dedup loser to W, which is new.
    ids2, changes, *_ = _run(
        tmp_path, "20260201", [{"ovt": "W"}], registry = reg1, aliases = [("X", "W")],
    )
    assert ids2[0] == ids1[0]


def test_dedup_alias_ranks_below_direct_key(tmp_path):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X"}, {"ovt": "W"}])
    ids2, changes, *_ = _run(
        tmp_path, "20260201", [{"ovt": "W"}], registry = reg1, aliases = [("X", "W")],
    )
    assert ids2[0] == ids1[1]
    assert _change(changes, "merged_into") == [(ids1[0], ids1[1])]


# -----------------------------------------------------------------
# Replacement guard
# -----------------------------------------------------------------


def test_guard_mints_on_osm_replacement(tmp_path):
    ids1, _, _, reg1, *_ = _run(
        tmp_path, "20260101", [{"osm": "n9", "osm_name": "7-Eleven"}],
    )
    ids2, changes, stats, *_ = _run(
        tmp_path, "20260201", [{"osm": "n9", "osm_name": "Phillips 66"}],
        registry = reg1,
    )
    assert ids2[0] != ids1[0]
    assert stats["n_guard_blocked"] == 1
    assert _change(changes, "renamed_guard") == [(ids1[0], ids2[0])]


def test_guard_blocks_the_id_through_the_overture_edge_too(tmp_path):
    ids1, _, _, reg1, *_ = _run(
        tmp_path, "20260101",
        [{"osm": "n9", "ovt": "X", "osm_name": "Cafe India", "ovt_name": "Cafe India"}],
    )
    ids2, *_ = _run(
        tmp_path, "20260201",
        [{"osm": "n9", "ovt": "X", "osm_name": "The Yolk Grill",
          "ovt_name": "Cafe India"}],
        registry = reg1,
    )
    assert ids2[0] != ids1[0]


@pytest.mark.parametrize("old, new", [
    ("Walgreens", "Walgreens Pharmacy"),
    (None, "Joe's Diner"),
    ("Joe's Diner", None),
    ("Cafe Allegro", "Café Allegro"),
])
def test_guard_passes_rewording_and_naming(tmp_path, old, new):
    ids1, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"osm": "n9", "osm_name": old}])
    ids2, *_ = _run(
        tmp_path, "20260201", [{"osm": "n9", "osm_name": new}], registry = reg1,
    )
    assert ids2[0] == ids1[0]


def test_overture_rename_is_not_guarded(tmp_path):
    ids1, _, _, reg1, *_ = _run(
        tmp_path, "20260101", [{"ovt": "X", "ovt_name": "Exzact Customs"}],
    )
    ids2, *_ = _run(
        tmp_path, "20260201", [{"ovt": "X", "ovt_name": "Window Tinting Co"}],
        registry = reg1,
    )
    assert ids2[0] == ids1[0]


# -----------------------------------------------------------------
# Registry and invariants
# -----------------------------------------------------------------


def test_registry_rejects_same_or_later_release(tmp_path):
    _, _, _, reg1, *_ = _run(tmp_path, "20260101", [{"ovt": "X"}])
    with pytest.raises(ValueError, match = "prior release"):
        _run(tmp_path, "20260101", [{"ovt": "X"}], registry = reg1)


def test_duplicate_source_key_rejected(tmp_path):
    with pytest.raises(ValueError, match = "more than one row"):
        _run(tmp_path, "20260101", [{"ovt": "X"}, {"ovt": "X"}])


def test_mint_resalts_on_collision_with_registry_id(tmp_path):
    # A registry tombstone holding hash(ovt:Q) forces a re-salt.
    reg = tmp_path / "reg.parquet"
    pq.write_table(pa.table({
        "source_key": pa.array([None], pa.string()),
        "unified_id": [si.mint_id("ovt:Q")],
        "id_minted": ["20251201"], "first_release": pa.array([None], pa.string()),
        "last_release": ["20251201"], "last_name": pa.array([None], pa.string()),
    }), reg)
    ids, *_ = _run(tmp_path, "20260101", [{"ovt": "Q"}], registry = reg)
    assert ids[0] == si.mint_id("ovt:Q", salt = 1)


def test_registry_keeps_absent_keys_and_updates_present(tmp_path):
    _, _, _, reg1, *_ = _run(
        tmp_path, "20260101", [{"ovt": "X", "ovt_name": "A"}, {"ovt": "Z"}],
    )
    _, _, _, reg2, *_ = _run(
        tmp_path, "20260201", [{"ovt": "X", "ovt_name": "B"}], registry = reg1,
    )
    rows = {r["source_key"]: r for r in pq.read_table(reg2).to_pylist()}
    assert rows["ovt:X"]["first_release"] == "20260101"
    assert rows["ovt:X"]["last_release"] == "20260201"
    assert rows["ovt:X"]["last_name"] == "B"
    assert rows["ovt:Z"]["last_release"] == "20260101"


# -----------------------------------------------------------------
# Rewriting the parquet
# -----------------------------------------------------------------


def test_rewrite_in_place_preserves_everything_but_the_id(tmp_path):
    ids, _, _, _, con, path = _run(tmp_path, "20260101", [
        {"osm": "n1", "ovt": "X"}, {"ovt": "Y"}, {"osm": "w2"},
    ])
    before = pq.read_table(path)
    n = si.rewrite_unified_id(con, path, path)
    after = pq.read_table(path)
    assert n == 3
    assert after.column("unified_id").to_pylist() == [ids[0], ids[1], ids[2]]
    for col in before.column_names:
        if col != "unified_id":
            assert after.column(col).equals(before.column(col)), col
    assert after.schema.metadata[b"geo"] == before.schema.metadata[b"geo"]
    assert si.verify_rewrite(con, path) == 0


def test_rewrite_casts_double_osm_id(tmp_path):
    path = _release(
        tmp_path / "c.parquet", [{"osm": "n12345678901"}, {"ovt": "Y"}],
        osm_id_type = pa.float64(),
    )
    con = si.connect("1GB")
    si.load_release(con, [path])
    si.load_registry(con, None)
    si.reconcile(con, "20260101")
    out = tmp_path / "v1" / "c.parquet"
    si.rewrite_unified_id(con, path, out, cast_osm_id = True)
    table = pq.read_table(out)
    assert table.schema.field("osm_id").type == pa.int64()
    assert table.column("osm_id").to_pylist() == [12345678901, None]
    assert si.verify_rewrite(con, out, source_path = path) == 0


def test_migration_table_maps_v0_to_v1(tmp_path):
    ids, _, _, _, con, path = _run(tmp_path, "20260101", [{"osm": "n1"}, {"ovt": "Y"}])
    si.load_v0_ids(con, [path])
    mig = si.migration_table(con).to_pylist()
    assert {(m["v0_unified_id"], m["v1_unified_id"]) for m in mig} == {
        ("old0", ids[0]), ("old1", ids[1]),
    }


def test_assign_from_registry_applies_a_release_to_another_copy(tmp_path):
    rows = [{"osm": "n1", "ovt": "X"}, {"ovt": "Y"}, {"osm": "w2"}]
    ids, _, _, reg, *_ = _run(tmp_path, "20260101", rows)
    copy = _release(tmp_path / "canonical.parquet", rows[::-1])
    con = si.connect("1GB")
    si.load_release(con, [copy])
    si.load_registry(con, reg)
    assert si.assign_from_registry(con, "20260101") == 3
    si.rewrite_unified_id(con, copy, copy)
    assert pq.read_table(copy).column("unified_id").to_pylist() == [
        ids[2], ids[1], ids[0],
    ]
    # A row the registry has never seen is an error.
    other = _release(tmp_path / "other.parquet", [{"ovt": "Q"}])
    si.load_release(con, [other])
    with pytest.raises(ValueError, match = "without an id"):
        si.assign_from_registry(con, "20260101")
