"""Tests for openpois.io.osm_history_incremental: rolling the history parquets
forward with Geofabrik-style (extract-derived) daily diffs."""
from __future__ import annotations

import datetime
import gzip
import json
import shutil
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import openpois.io.osm_history_incremental as inc
from openpois.conflation.ghost_osm import build_ghosts
from openpois.io.osm_history_incremental import (
    DiffsUnavailableError,
    HistoryCoverage,
    ReplicationState,
    check_history_vs_snapshot,
    plan_sequences,
    read_coverage,
    require_full_history,
    roll_osm_history,
    write_coverage,
)
from openpois.io.osm_history_pbf import (
    CHANGES_SCHEMA,
    VERSIONS_SCHEMA,
    _diff_tag_sets,
)

UTC = datetime.timezone.utc
FILTER = ["nwr/amenity=cafe,pharmacy", "nwr/shop"]
POI_KEYS = ["shop", "amenity"]


def _ts(day: int, hour: int = 20, minute: int = 24) -> datetime.datetime:
    return datetime.datetime(2026, 8, day, hour, minute, tzinfo = UTC)


# --- fixtures ---------------------------------------------------------------

def _tagset(tags: dict | None, lat: str | None, lon: str | None, visible: bool):
    out = set()
    if visible:
        out.update((tags or {}).items())
        if lat is not None:
            out |= {("lat", lat), ("lon", lon)}
    out.add(("visible", "true" if visible else "false"))
    return out


def _write_base(base_dir: Path, elements: dict) -> None:
    """elements: {id: [(version, iso_ts, tags|None, lat, lon, visible), ...]}"""
    v_rows = {n: [] for n in VERSIONS_SCHEMA.names}
    c_rows = {n: [] for n in CHANGES_SCHEMA.names}
    for oid, versions in elements.items():
        prev = set()
        for version, ts, tags, lat, lon, visible in versions:
            curr = _tagset(tags, lat, lon, visible)
            for col, val in zip(
                VERSIONS_SCHEMA.names,
                (oid, version, 1, ts, "u", 1, "node"),
            ):
                v_rows[col].append(val)
            for row in _diff_tag_sets(prev, curr):
                for col in ("key", "value", "change"):
                    c_rows[col].append(row[col])
                c_rows["id"].append(oid)
                c_rows["version"].append(version)
                c_rows["type"].append("node")
            prev = curr
    base_dir.mkdir(parents = True, exist_ok = True)
    pq.write_table(
        pa.table(v_rows, schema = VERSIONS_SCHEMA), base_dir / "osm_versions.parquet",
    )
    pq.write_table(
        pa.table(c_rows, schema = CHANGES_SCHEMA), base_dir / "osm_changes.parquet",
    )


def _node(oid, version, ts, tags = None, lat = "47.6", lon = "-122.3"):
    tag_xml = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in (tags or {}).items())
    loc = f' lat="{lat}" lon="{lon}"' if lat is not None else ""
    return (
        f'<node id="{oid}" version="{version}" timestamp="{ts}"{loc}>'
        f"{tag_xml}</node>"
    )


def _osc(path: Path, create = (), modify = (), delete = ()) -> None:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<osmChange version="0.6" generator="test">'
        f'<create>{"".join(create)}</create>'
        f'<modify>{"".join(modify)}</modify>'
        f'<delete>{"".join(delete)}</delete>'
        "</osmChange>\n"
    )
    with gzip.open(path, "wt") as fh:
        fh.write(body)


BASE_TS = "2026-07-01T00:00:00+00:00"
LOC = {
    1: ("47.6", "-122.3"), 2: ("47.61", "-122.31"), 3: ("47.62", "-122.32"),
    4: ("47.63", "-122.33"), 7: ("47.64", "-122.34"), 10: ("47.65", "-122.35"),
    11: ("47.66", "-122.36"),
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Base history + two daily diffs served by a fake replication server."""
    base_dir = tmp_path / "osm_data" / "base"
    shop = "shop"
    _write_base(base_dir, {
        # 1: named bakery, deleted in diff 102
        1: [(1, BASE_TS, {shop: "bakery", "name": "Joe's Bakery"}, *LOC[1], True),
            (2, BASE_TS, {shop: "bakery", "name": "Joe's Bakery"}, *LOC[1], True)],
        # 2: cafe renamed in 101, then deleted in 102
        2: [(1, BASE_TS, {"amenity": "cafe", "name": "Alpha Cafe"}, *LOC[2], True)],
        # 3: shop tag removed in 101
        3: [(1, BASE_TS, {shop: "books", "name": "Book Nook"}, *LOC[3], True)],
        # 4: lifecycle prefix added in 101
        4: [(1, BASE_TS, {shop: "shoes", "name": "Shoe Stop"}, *LOC[4], True)],
        # 7: already deleted in the base (boundary file re-sends the delete)
        7: [(1, BASE_TS, {shop: "gifts", "name": "Gift Hut"}, *LOC[7], True),
            (2, "2026-07-31T22:00:00+00:00", None, None, None, False)],
        # 10: every tag removed in 101
        10: [(1, BASE_TS, {shop: "toys", "name": "Toy Barn"}, *LOC[10], True)],
        # 11: straddle: the boundary file repeats base version 3
        11: [(3, "2026-07-31T21:00:00+00:00", {shop: "deli", "name": "Deli"},
              *LOC[11], True)],
    })

    feed = tmp_path / "feed"
    feed.mkdir()
    _osc(
        feed / "101.osc.gz",
        create = [
            _node(8, 1, "2026-08-01T10:00:00Z", {"highway": "traffic_signals"}),
            _node(9, 1, "2026-08-01T10:00:00Z", {"shop": "florist", "name": "Bloom"}),
            _node(12, 1, "2026-08-01T10:00:00Z", {}),  # fresh untagged vertex
        ],
        modify = [
            _node(2, 2, "2026-08-01T11:00:00Z",
                  {"amenity": "cafe", "name": "Zeta Diner"}, "47.61", "-122.31"),
            _node(3, 2, "2026-08-01T11:00:00Z", {"name": "Book Nook"}, *LOC[3]),
            _node(4, 2, "2026-08-01T11:00:00Z",
                  {"disused:shop": "shoes", "name": "Shoe Stop"}, *LOC[4]),
            _node(10, 2, "2026-08-01T11:00:00Z", {}, "47.65", "-122.35"),
            _node(11, 3, "2026-07-31T21:00:00Z",
                  {"shop": "deli", "name": "Deli"}, "47.66", "-122.36"),
        ],
        delete = [
            _node(7, 1, "2026-07-01T00:00:00Z", {"shop": "gifts", "name": "Gift Hut"},
                  "47.64", "-122.34"),
        ],
    )
    _osc(
        feed / "102.osc.gz",
        delete = [
            # carried last-live versions / timestamps, as Geofabrik publishes
            _node(1, 2, "2026-07-01T00:00:00Z",
                  {"shop": "bakery", "name": "Joe's Bakery"}, "47.6", "-122.3"),
            _node(2, 2, "2026-08-01T11:00:00Z",
                  {"amenity": "cafe", "name": "Zeta Diner"}, "47.61", "-122.31"),
            # 5: never in the base; carried POI tags put it in the universe
            _node(5, 4, "2025-01-01T00:00:00Z",
                  {"amenity": "pharmacy", "name": "Corner Rx"}, "47.67", "-122.37"),
            # 6: never in the base and not a POI
            _node(6, 2, "2024-01-01T00:00:00Z", {"highway": "stop"}, "47.68", "-122.38"),
        ],
    )

    # Base coverage ends 2026-07-31 22:00 (newest base timestamp), so the
    # window starts at 101, the file straddling that instant.
    july_31 = datetime.datetime(2026, 7, 31, 20, 24, tzinfo = UTC)
    states = {
        99: ReplicationState(99, july_31 - datetime.timedelta(days = 1)),
        100: ReplicationState(100, july_31),
        101: ReplicationState(101, _ts(1)),
        102: ReplicationState(102, _ts(2)),
    }
    newest = states[102]

    def factory(url):
        def get_state(seq):
            return newest if seq is None else states.get(seq)
        return get_state

    def fake_download(url, output_path, **kwargs):
        seq = int(url.rsplit("/", 1)[1].split(".")[0])
        shutil.copy(feed / f"{seq}.osc.gz", output_path)
        return output_path

    monkeypatch.setattr(inc, "download_resilient", fake_download)
    return {"tmp": tmp_path, "base_dir": base_dir, "factory": factory}


def _roll(world, urls = None, **kwargs):
    out_dir = world["tmp"] / "osm_data" / "rolled"
    cov = roll_osm_history(
        base_dir = world["base_dir"],
        base_version = "base",
        out_dir = out_dir,
        out_versions_path = out_dir / "osm_versions.parquet",
        out_changes_path = out_dir / "osm_changes.parquet",
        replication_urls = urls or {"us": "https://example.test/us-updates/"},
        tag_filter_exprs = FILTER,
        state_getter_factory = world["factory"],
        verbose = False,
        **kwargs,
    )
    return out_dir, cov


def _rows(path: Path, where: str = "") -> list[tuple]:
    return duckdb.sql(
        f"SELECT * FROM read_parquet('{path}') {where} ORDER BY ALL"
    ).fetchall()


# --- roll-forward -----------------------------------------------------------

def test_deletion_encoding_matches_full_parser(world):
    out_dir, _ = _roll(world)
    changes = _rows(
        out_dir / "osm_changes.parquet", "WHERE id = 1 AND version = 3",
    )
    assert sorted((k, v, c) for k, v, c, *_ in changes) == [
        ("lat", "47.6", "Deleted"),
        ("lon", "-122.3", "Deleted"),
        ("name", "Joe's Bakery", "Deleted"),
        ("shop", "bakery", "Deleted"),
        ("visible", "false", "Changed"),
    ]
    (v,) = _rows(out_dir / "osm_versions.parquet", "WHERE id = 1 AND version = 3")
    assert v[3] == _ts(2).isoformat()          # stamped with state_ts
    assert v[2] is None and v[4] is None       # no changeset / user


def test_unknown_element_delete_emits_carried_version(world):
    out_dir, _ = _roll(world)
    versions = _rows(out_dir / "osm_versions.parquet", "WHERE id = 5")
    assert [(r[1], r[3]) for r in versions] == [
        (4, "2025-01-01T00:00:00+00:00"),
        (5, _ts(2).isoformat()),
    ]
    assert not _rows(out_dir / "osm_versions.parquet", "WHERE id = 6")


def test_universe_and_dedupe(world):
    out_dir, _ = _roll(world)
    vpath = out_dir / "osm_versions.parquet"
    assert not _rows(vpath, "WHERE id IN (8, 12)")          # not POIs
    assert [r[1] for r in _rows(vpath, "WHERE id = 9")] == [1]
    assert [r[1] for r in _rows(vpath, "WHERE id = 11")] == [3]   # no dup
    assert [r[1] for r in _rows(vpath, "WHERE id = 7")] == [1, 2]  # no re-delete
    # an unchanged location emits no lat/lon rows
    assert not _rows(
        out_dir / "osm_changes.parquet",
        "WHERE id = 2 AND version = 2 AND key IN ('lat', 'lon')",
    )


def test_rolled_parquets_feed_build_ghosts(world):
    out_dir, _ = _roll(world)
    ghosts = build_ghosts(
        out_dir / "osm_versions.parquet", out_dir / "osm_changes.parquet",
        POI_KEYS, verbose = False,
    )
    got = {
        (int(r.osm_id), r.event_type, r.prior_name)
        for r in ghosts.itertuples()
    }
    assert got == {
        (1, "hard_delete", "Joe's Bakery"),
        (2, "substantial_rename", "Alpha Cafe"),
        (2, "hard_delete", "Zeta Diner"),
        (3, "primary_tag_deleted", "Book Nook"),
        (4, "lifecycle_prefix_added", "Shoe Stop"),
        (5, "hard_delete", "Corner Rx"),
        (7, "hard_delete", "Gift Hut"),        # from the base itself
        (10, "primary_tag_deleted", "Toy Barn"),
    }
    rename = ghosts[ghosts["event_type"] == "substantial_rename"].iloc[0]
    assert rename["new_name"] == "Zeta Diner"


def test_coverage_written_and_chained(world):
    out_dir, cov = _roll(world)
    assert cov.mode == "incremental"
    assert cov.chain_length == 1
    assert cov.extracts["us"]["last_sequence"] == 102
    assert cov.coverage_end == _ts(2)
    again = read_coverage(out_dir)
    assert again.extracts == cov.extracts
    assert again.filter_exprs == FILTER
    assert not (out_dir / "diffs").exists()


def test_cross_extract_duplicates_are_dropped(world):
    urls = {"us": "https://example.test/us/", "pr": "https://example.test/pr/"}
    out_dir, _ = _roll(world, urls = urls)
    dupes = duckdb.sql(f"""
        SELECT type, id, version, count(*) FROM
        read_parquet('{out_dir / "osm_versions.parquet"}')
        GROUP BY ALL HAVING count(*) > 1
    """).fetchall()
    assert dupes == []
    n_changes = duckdb.sql(f"""
        SELECT count(*) FROM read_parquet('{out_dir / "osm_changes.parquet"}')
        WHERE id = 1 AND version = 3
    """).fetchone()[0]
    assert n_changes == 5


def test_refuses_base_as_output(world):
    with pytest.raises(ValueError, match = "base directory"):
        roll_osm_history(
            base_dir = world["base_dir"], base_version = "base",
            out_dir = world["base_dir"],
            out_versions_path = world["base_dir"] / "x.parquet",
            out_changes_path = world["base_dir"] / "y.parquet",
            replication_urls = {"us": "u"}, tag_filter_exprs = FILTER,
            state_getter_factory = world["factory"], verbose = False,
        )


def test_refuses_widened_filter(world):
    write_coverage(world["base_dir"], HistoryCoverage(
        mode = "full", coverage_end = _ts(1, 0, 0),
        filter_exprs = ["nwr/amenity=cafe"],
    ))
    with pytest.raises(ValueError, match = "wider"):
        _roll(world)


def test_refuses_stale_end_date(world):
    with pytest.raises(ValueError, match = "advance"):
        _roll(world, end_date = datetime.datetime(2026, 7, 31, tzinfo = UTC))


def test_refuses_empty_window(world):
    # end_date after the base coverage but before the first diff's cut.
    with pytest.raises(ValueError, match = "No new diffs"):
        _roll(world, end_date = datetime.datetime(2026, 8, 1, 12, tzinfo = UTC))


def test_require_full_history(world):
    out_dir, _ = _roll(world)
    require_full_history(world["base_dir"], allow_incremental = False)
    with pytest.raises(SystemExit):
        require_full_history(out_dir, allow_incremental = False)
    require_full_history(out_dir, allow_incremental = True)


def test_read_coverage_infers_full_build(world):
    cov = read_coverage(world["base_dir"])
    assert cov.mode == "full" and cov.inferred
    assert cov.coverage_end == datetime.datetime(2026, 7, 31, 22, tzinfo = UTC)


def test_check_history_vs_snapshot(world, tmp_path):
    out_dir, cov = _roll(world)
    snap = tmp_path / "snap.parquet"
    pq.write_table(pa.table({
        "osm_id": [9, 11, 99],
        "osm_type": ["node", "node", "node"],
        "name": ["Bloom", "Deli", "Ghost Town"],
        "last_edited": pa.array([
            datetime.datetime(2026, 8, 1, 10, tzinfo = UTC),
            datetime.datetime(2026, 7, 31, 21, tzinfo = UTC),
            datetime.datetime(2026, 7, 1, tzinfo = UTC),
        ], type = pa.timestamp("us", tz = "UTC")),
    }), snap)
    result = check_history_vs_snapshot(out_dir, snap, cov.coverage_end)
    assert result["n_checked"] == 3
    assert result["n_missing"] == 1
    assert result["n_ts_match"] == 2
    assert result["n_name_match"] == 2


# --- sequence planning ------------------------------------------------------

def _getter(first: int, last: int, first_day: int = 1):
    states = {
        seq: ReplicationState(seq, _ts(first_day + seq - first))
        for seq in range(first, last + 1)
    }
    return lambda seq: states[last] if seq is None else states.get(seq)


def test_plan_from_coverage_end():
    get = _getter(1, 10)
    plan = plan_sequences(get, _ts(3, 23, 59))
    assert (plan.start, plan.end) == (4, 10)


def test_plan_from_last_sequence_and_end_date():
    get = _getter(1, 10)
    plan = plan_sequences(
        get, _ts(3), end_date = _ts(8, 0, 0), last_sequence = 5,
    )
    assert (plan.start, plan.end) == (6, 7)
    assert plan.end_timestamp == _ts(7)


def test_plan_nothing_new():
    get = _getter(1, 10)
    assert plan_sequences(get, _ts(3), last_sequence = 10) is None
    assert plan_sequences(get, _ts(11)) is None


def test_plan_pruned_raises():
    get = _getter(5, 10, first_day = 5)
    with pytest.raises(DiffsUnavailableError):
        plan_sequences(get, _ts(2))
    with pytest.raises(DiffsUnavailableError):
        plan_sequences(get, _ts(2), last_sequence = 2)


def test_fetch_gap_raises(world, tmp_path):
    states = {101: ReplicationState(101, _ts(1))}
    plan = inc.SequencePlan(101, 102, _ts(2))
    with pytest.raises(DiffsUnavailableError):
        inc.fetch_diffs(
            "https://example.test/us/", plan,
            lambda seq: states.get(seq), tmp_path / "d", verbose = False,
        )


def test_coverage_round_trip(tmp_path):
    cov = HistoryCoverage(
        mode = "incremental", coverage_end = _ts(5), chain_length = 3,
        base_version = "20260902", filter_exprs = FILTER,
        extracts = {"us": {"last_sequence": 7}},
    )
    write_coverage(tmp_path, cov)
    raw = json.loads((tmp_path / "history_coverage.json").read_text())
    assert "inferred" not in raw
    assert read_coverage(tmp_path) == cov
