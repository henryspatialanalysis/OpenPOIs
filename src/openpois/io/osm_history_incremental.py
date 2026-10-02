#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Roll the OSM history parquets forward with Geofabrik's daily replication
diffs, instead of re-downloading the ~23 GB full-history extract.

``download.osm.history_mode: incremental`` routes
``scripts/osm_data/download_history.py`` here. The output is a new
``osm_data/<v>/`` directory whose ``osm_versions.parquet`` /
``osm_changes.parquet`` have the same schema as a full build: the base run's
rows plus one row set per element version seen in the diff window. Ghost
building (``build_ghosts``) runs over them unchanged.

Geofabrik diff semantics this module relies on (verified 2026-09-26 on the
public ``us-updates`` feed):

- Diffs are derived extract-to-extract, one file per day (cut ~20:24 UTC).
  An object appears at most once per file, so same-day edits are collapsed
  into the day's final version.
- ``<delete>`` elements carry the prior location and tags, but their
  ``version`` / ``timestamp`` are the **last live edit**, not the deletion.
  The deletion is written as version ``last + 1`` stamped with the diff
  file's state timestamp (an upper bound, at most ~24 h late).
- Public files drop ``user`` / ``uid`` / ``changeset``; those columns are
  null for window rows.
- Retention is ~131 days: a base older than that cannot be rolled forward
  and a full run is required.

Rolled parquets are **ghost grade**: same-day collapse and missing edit
metadata make them unsuitable for a turnover-model refit, which
``format_tabular.py`` / ``osm_turnover.py`` refuse unless overridden. Each
``osm_data`` directory records its provenance in ``history_coverage.json``.
"""
from __future__ import annotations

import datetime
import gzip
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from osmium.replication.server import ReplicationServer

from openpois.conflation.taxonomy import (
    matches_osm_tag_filter,
    parse_osm_tag_filter_expressions,
)
from openpois.io._download import download_resilient
from openpois.io.osm_history_pbf import (
    CHANGES_SCHEMA,
    VERSIONS_SCHEMA,
    _diff_tag_sets,
)


COVERAGE_FILENAME = "history_coverage.json"

UTC = datetime.timezone.utc


# -----------------------------------------------------------------------------
# Coverage metadata
# -----------------------------------------------------------------------------


@dataclass
class HistoryCoverage:
    """Provenance of one ``osm_data/<v>/`` directory.

    Attributes:
        mode: ``"full"`` (built from a full-history PBF) or ``"incremental"``.
        coverage_end: Every edit before this instant is in the parquets.
        extracts: Per-extract replication state after an incremental run:
            ``{name: {"server", "last_sequence", "last_timestamp"}}``.
        chain_length: Incremental runs since the last full build (0 = full).
        base_version: ``osm_data`` version this run rolled forward from.
        filter_exprs: The osmium ingest filter expressions in force.
        inferred: True when read from a directory without a coverage file.
    """
    mode: str
    coverage_end: datetime.datetime
    extracts: dict[str, dict] = field(default_factory = dict)
    chain_length: int = 0
    base_version: str | None = None
    filter_exprs: list[str] | None = None
    inferred: bool = False


def _parse_ts(value: str | datetime.datetime) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        ts = value
    else:
        ts = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo = UTC)
    return ts.astimezone(UTC)


def write_coverage(osm_data_dir: Path, coverage: HistoryCoverage) -> Path:
    """Write ``history_coverage.json`` into ``osm_data_dir``."""
    payload = asdict(coverage)
    payload["coverage_end"] = coverage.coverage_end.isoformat()
    payload.pop("inferred")
    path = Path(osm_data_dir) / COVERAGE_FILENAME
    path.write_text(json.dumps(payload, indent = 2) + "\n")
    return path


def read_coverage(
    osm_data_dir: Path,
    versions_filename: str = "osm_versions.parquet",
) -> HistoryCoverage:
    """Read ``history_coverage.json``, or infer coverage for a directory built
    before the file existed.

    Inference treats the directory as a full build whose ``coverage_end`` is
    the newest version timestamp in ``osm_versions.parquet`` (a full build's
    ``end_date`` less the few quiet minutes before it).
    """
    osm_data_dir = Path(osm_data_dir)
    path = osm_data_dir / COVERAGE_FILENAME
    if path.exists():
        raw = json.loads(path.read_text())
        raw["coverage_end"] = _parse_ts(raw["coverage_end"])
        return HistoryCoverage(**raw)

    versions_path = osm_data_dir / versions_filename
    if not versions_path.exists():
        raise FileNotFoundError(
            f"No {COVERAGE_FILENAME} or {versions_filename} in {osm_data_dir}"
        )
    # Epoch seconds, not TIMESTAMPTZ: DuckDB needs pytz to hand tz-aware
    # values to Python, and the env does not carry it.
    con = duckdb.connect()
    try:
        max_epoch = con.execute(
            "SELECT epoch(max(CAST(timestamp AS TIMESTAMPTZ))) "
            f"FROM read_parquet('{versions_path}')"
        ).fetchone()[0]
    finally:
        con.close()
    if max_epoch is None:
        raise ValueError(f"{versions_path} has no timestamps")
    max_ts = datetime.datetime.fromtimestamp(max_epoch, tz = UTC)
    print(
        f"No {COVERAGE_FILENAME} in {osm_data_dir}; inferred a full build "
        f"covering edits before {max_ts.isoformat()}."
    )
    return HistoryCoverage(mode = "full", coverage_end = max_ts, inferred = True)


def require_full_history(osm_data_dir: Path, allow_incremental: bool) -> None:
    """Refuse model-fit inputs built by the incremental roll-forward.

    Rolled parquets collapse same-day edits and carry no user / changeset,
    which biases the per-version turnover likelihood. A refit month must run
    ``history_mode: full``.
    """
    try:
        coverage = read_coverage(osm_data_dir)
    except FileNotFoundError:
        return
    if coverage.mode == "incremental" and not allow_incremental:
        raise SystemExit(
            f"{osm_data_dir} was built by the incremental history roll-forward "
            f"(base {coverage.base_version}, chain length "
            f"{coverage.chain_length}). It is ghost-grade only: re-run "
            "download_history.py with download.osm.history_mode: full before "
            "refitting, or pass --allow-incremental-history to override."
        )


# -----------------------------------------------------------------------------
# Replication sequence planning
# -----------------------------------------------------------------------------


class ReplicationState(NamedTuple):
    sequence: int
    timestamp: datetime.datetime


# ``get_state(None)`` returns the newest state; ``get_state(seq)`` returns that
# sequence's state, or None when it is not on the server.
StateGetter = Callable[[int | None], ReplicationState | None]


class DiffsUnavailableError(RuntimeError):
    """The diff window starts before the server's retention horizon."""


def replication_state_getter(server_url: str) -> StateGetter:
    """``StateGetter`` backed by pyosmium's ``ReplicationServer``, memoised."""
    server = ReplicationServer(server_url.rstrip("/"))
    cache: dict[int | None, ReplicationState | None] = {}

    def get_state(seq: int | None) -> ReplicationState | None:
        if seq not in cache:
            state = server.get_state_info(seq)
            cache[seq] = (
                None if state is None
                else ReplicationState(state.sequence, _parse_ts(state.timestamp))
            )
        return cache[seq]

    return get_state


class SequencePlan(NamedTuple):
    start: int
    end: int
    end_timestamp: datetime.datetime


def plan_sequences(
    get_state: StateGetter,
    coverage_end: datetime.datetime,
    end_date: datetime.datetime | None = None,
    last_sequence: int | None = None,
) -> SequencePlan | None:
    """Pick the diff sequences covering ``(coverage_end, end_date]``.

    Start: ``last_sequence + 1`` when the base recorded one (exact
    continuity); otherwise the first sequence whose state timestamp is after
    ``coverage_end`` — the file whose window contains the boundary. End: the
    last sequence whose state timestamp is ``<= end_date`` (or the newest).

    Returns None when there is nothing new to fetch.

    Raises:
        DiffsUnavailableError: the start sequence (or the state needed to
            confirm it) has been pruned from the server.
        RuntimeError: the server's newest state cannot be read.
    """
    coverage_end = _parse_ts(coverage_end)
    newest = get_state(None)
    if newest is None:
        raise RuntimeError("Could not read the replication server's state.txt")
    end_cap = _parse_ts(end_date) if end_date is not None else None

    prev = get_state(newest.sequence - 1)
    interval = (
        (newest.timestamp - prev.timestamp).total_seconds()
        if prev is not None else 86_400.0
    )
    interval = max(interval, 60.0)

    def _need(seq: int) -> ReplicationState:
        state = get_state(seq)
        if state is None:
            raise DiffsUnavailableError(
                f"Replication sequence {seq} is no longer on the server "
                f"(needed to cover edits after {coverage_end.isoformat()}). "
                "Geofabrik keeps ~131 days of daily diffs. Set "
                "download.osm.history_mode: full for this run."
            )
        return state

    if last_sequence is not None:
        start = last_sequence + 1
        if start > newest.sequence:
            return None
        _need(start)
    else:
        if coverage_end >= newest.timestamp:
            return None
        lag = (newest.timestamp - coverage_end).total_seconds()
        start = max(newest.sequence - math.floor(lag / interval), 0)
        state = _need(start)
        while state.timestamp <= coverage_end:
            start += 1
            if start > newest.sequence:
                return None
            state = _need(start)
        while start > 0 and _need(start - 1).timestamp > coverage_end:
            start -= 1

    end = newest.sequence
    if end_cap is not None and newest.timestamp > end_cap:
        lag = (newest.timestamp - end_cap).total_seconds()
        end = max(newest.sequence - math.floor(lag / interval), start - 1)
        while end >= start and _need(end).timestamp > end_cap:
            end -= 1
        while end + 1 <= newest.sequence and _need(end + 1).timestamp <= end_cap:
            end += 1
    if end < start:
        return None
    return SequencePlan(start, end, _need(end).timestamp)


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------


class DiffFile(NamedTuple):
    sequence: int
    state_timestamp: datetime.datetime
    path: Path


def _diff_url(server_url: str, seq: int) -> str:
    base = server_url.rstrip("/")
    path = f"{seq // 1_000_000:03d}/{(seq // 1000) % 1000:03d}/{seq % 1000:03d}"
    return f"{base}/{path}.osc.gz"


def fetch_diffs(
    server_url: str,
    plan: SequencePlan,
    get_state: StateGetter,
    work_dir: Path,
    verbose: bool = True,
) -> list[DiffFile]:
    """Download every ``.osc.gz`` in ``plan`` and check it is a whole gzip.

    Uses ``download_resilient`` per file (not pyosmium's ``collect_diffs``,
    which returns partial data on a mid-run error). Any failure raises.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents = True, exist_ok = True)
    files: list[DiffFile] = []
    for seq in range(plan.start, plan.end + 1):
        state = get_state(seq)
        if state is None:
            raise DiffsUnavailableError(
                f"State for sequence {seq} vanished mid-run from {server_url}"
            )
        path = work_dir / f"{seq:09d}.osc.gz"
        if path.exists() and not _gzip_ok(path):
            path.unlink()
        if not path.exists():
            download_resilient(
                _diff_url(server_url, seq), path,
                label = f"diff {seq}", n_segments = 1,
            )
        if not _gzip_ok(path):
            raise RuntimeError(f"Corrupt diff after download: {path}")
        files.append(DiffFile(seq, state.timestamp, path))
        if verbose:
            print(f"  diff {seq} ({state.timestamp:%Y-%m-%d %H:%M}Z) ok")
    return files


def _gzip_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except (OSError, EOFError):
        return False


# -----------------------------------------------------------------------------
# Pass 1: parse diff files into a compact window table
# -----------------------------------------------------------------------------


WINDOW_SCHEMA = pa.schema([
    ("seq", pa.int64()),
    ("state_ts", pa.string()),
    ("type", pa.string()),
    ("id", pa.int64()),
    ("version", pa.int64()),
    ("timestamp", pa.string()),
    ("deleted", pa.bool_()),
    ("lat", pa.string()),
    ("lon", pa.string()),
    ("tags", pa.list_(pa.struct([("k", pa.string()), ("v", pa.string())]))),
    ("matches_filter", pa.bool_()),
])


_ACTIONS = frozenset({"create", "modify", "delete"})
_OBJECT_KINDS = frozenset({"node", "way", "relation"})


def parse_diff_file(
    diff: DiffFile,
    out_path: Path,
    parsed_filter: dict[str, frozenset[str] | None],
) -> int:
    """Write the objects of one ``.osc.gz`` that could matter to history.

    Kept: deletes, tagged objects, and untagged objects with ``version > 1``
    (a POI node that lost every tag). Fresh untagged vertices are dropped.

    Parsed as streaming XML rather than with pyosmium: pyosmium's Python tag
    iterator costs ~60 µs per tag (64 s vs 6 s on a 13 MB US daily file).
    Values are normalised to what the full parser writes: ``lat`` / ``lon``
    as ``str(float)`` (identical to ``str(location.lat)``) and timestamps as
    ``+00:00`` ISO strings (verified identical on 200k nodes).

    Returns:
        Number of rows written.
    """
    rows: dict[str, list] = {name: [] for name in WINDOW_SCHEMA.names}
    state_ts = diff.state_timestamp.isoformat()
    action = None
    with gzip.open(diff.path, "rb") as fh:
        for event, el in ET.iterparse(fh, events = ("start", "end")):
            kind = el.tag
            if event == "start":
                if kind in _ACTIONS:
                    action = kind
                continue
            if kind in _ACTIONS:
                el.clear()
                continue
            if kind not in _OBJECT_KINDS:
                continue
            tags = {c.get("k"): c.get("v") for c in el if c.tag == "tag"}
            deleted = action == "delete"
            version = int(el.get("version"))
            if deleted or tags or version > 1:
                lat = el.get("lat")
                lon = el.get("lon")
                if kind == "node" and lat is not None and lon is not None:
                    lat, lon = str(float(lat)), str(float(lon))
                else:
                    lat = lon = None
                ts = el.get("timestamp")
                rows["seq"].append(diff.sequence)
                rows["state_ts"].append(state_ts)
                rows["type"].append(kind)
                rows["id"].append(int(el.get("id")))
                rows["version"].append(version)
                rows["timestamp"].append(
                    ts.replace("Z", "+00:00") if ts else None
                )
                rows["deleted"].append(deleted)
                rows["lat"].append(lat)
                rows["lon"].append(lon)
                rows["tags"].append([{"k": k, "v": v} for k, v in tags.items()])
                rows["matches_filter"].append(
                    matches_osm_tag_filter(tags, parsed_filter)
                )
            el.clear()
    pq.write_table(
        pa.table(rows, schema = WINDOW_SCHEMA), out_path, compression = "zstd",
    )
    return len(rows["seq"])


# -----------------------------------------------------------------------------
# Pass 2: replay the window against the base state
# -----------------------------------------------------------------------------


def _version_tag_set(
    kind: str,
    tags: list[dict] | None,
    lat: str | None,
    lon: str | None,
    visible: bool,
) -> set[tuple[str, str]]:
    """Tag set in the full parser's convention (``_tag_set_for_version``):
    OSM tags plus ``visible`` and, for located nodes, ``lat`` / ``lon``."""
    out: set[tuple[str, str]] = set()
    if visible:
        out.update((t["k"], t["v"]) for t in (tags or []))
    out.add(("visible", "true" if visible else "false"))
    if visible and kind == "node" and lat is not None and lon is not None:
        out.add(("lat", lat))
        out.add(("lon", lon))
    return out


def roll_extract(
    window_files: list[Path],
    base_versions_path: Path,
    base_changes_path: Path,
    out_versions_path: Path,
    out_changes_path: Path,
    duckdb_memory_limit: str = "6GB",
    verbose: bool = True,
) -> tuple[int, int]:
    """Emit history rows for one extract's diff window.

    Universe: window elements already in the base history, or matching the
    ingest filter in any window version (the full path's "ever carried a POI
    tag" rule). Each element's prior state is the base's last tag state
    (folded from ``osm_changes``); window versions are replayed in sequence
    order and diffed with the full parser's ``_diff_tag_sets``.

    Deletes (see module docstring) become version ``carried + 1`` with tag
    set ``{visible: false}`` stamped with the diff's state timestamp. When
    the carried last-live version is newer than anything known, it is emitted
    first as an ordinary version.

    Returns:
        ``(n_version_rows, n_change_rows)`` written for the window.
    """
    files_sql = ", ".join(f"'{p}'" for p in window_files)
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
        con.execute("SET preserve_insertion_order = false")
        con.execute(
            f"CREATE TABLE win AS SELECT * FROM read_parquet([{files_sql}])"
        )
        con.execute(f"""
            CREATE TABLE keep AS
            SELECT type, id FROM win GROUP BY type, id HAVING bool_or(matches_filter)
            UNION
            SELECT DISTINCT w.type, w.id FROM win w
            SEMI JOIN (
                SELECT DISTINCT type, id FROM read_parquet('{base_versions_path}')
            ) b ON w.type = b.type AND w.id = b.id
        """)
        n_keep = con.execute("SELECT count(*) FROM keep").fetchone()[0]
        if verbose:
            n_win = con.execute("SELECT count(*) FROM win").fetchone()[0]
            print(
                f"  window rows kept by pass 1: {n_win:,}; "
                f"elements in history universe: {n_keep:,}"
            )
        base_last = con.execute(f"""
            SELECT v.type, v.id, max(v.version) AS version
            FROM read_parquet('{base_versions_path}') v
            SEMI JOIN keep k ON v.type = k.type AND v.id = k.id
            GROUP BY v.type, v.id
        """).fetchall()
        base_state = con.execute(f"""
            SELECT type, id, key,
                   arg_max(value, version) AS value,
                   arg_max(change, version) AS change
            FROM read_parquet('{base_changes_path}') c
            SEMI JOIN keep k ON c.type = k.type AND c.id = k.id
            GROUP BY type, id, key
        """).fetchall()
        window = con.execute("""
            SELECT w.type, w.id, w.seq, w.state_ts, w.version, w.timestamp,
                   w.deleted, w.lat, w.lon, w.tags
            FROM win w SEMI JOIN keep k ON w.type = k.type AND w.id = k.id
            ORDER BY w.type, w.id, w.seq
        """).fetchall()
    finally:
        con.close()

    last_version: dict[tuple[str, int], int] = {
        (t, i): v for t, i, v in base_last
    }
    state: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for t, i, key, value, change in base_state:
        if change != "Deleted":
            state.setdefault((t, i), set()).add((key, value))

    v_rows: dict[str, list] = {name: [] for name in VERSIONS_SCHEMA.names}
    c_rows: dict[str, list] = {name: [] for name in CHANGES_SCHEMA.names}

    def _emit(kind, oid, version, timestamp, prev, curr) -> None:
        v_rows["id"].append(oid)
        v_rows["version"].append(version)
        v_rows["changeset"].append(None)
        v_rows["timestamp"].append(timestamp)
        v_rows["user"].append(None)
        v_rows["uid"].append(None)
        v_rows["type"].append(kind)
        for row in _diff_tag_sets(prev, curr):
            c_rows["key"].append(row["key"])
            c_rows["value"].append(row["value"])
            c_rows["change"].append(row["change"])
            c_rows["id"].append(oid)
            c_rows["version"].append(version)
            c_rows["type"].append(kind)

    for kind, oid, _seq, state_ts, version, ts, deleted, lat, lon, tags in window:
        key = (kind, oid)
        known = last_version.get(key)
        prev = state.get(key, set())
        if deleted:
            if known is None or known < version:
                # Carried last-live version is news to us: record it first.
                live = _version_tag_set(kind, tags, lat, lon, True)
                _emit(kind, oid, version, ts, prev, live)
                prev = live
                known = version
            if known > version or ("visible", "false") in prev:
                continue  # deletion (or a later version) already recorded
            gone = _version_tag_set(kind, None, None, None, False)
            _emit(kind, oid, version + 1, state_ts, prev, gone)
            state[key] = gone
            last_version[key] = version + 1
        else:
            if known is not None and version <= known:
                continue  # already in the base (boundary file)
            curr = _version_tag_set(kind, tags, lat, lon, True)
            _emit(kind, oid, version, ts, prev, curr)
            state[key] = curr
            last_version[key] = version

    pq.write_table(
        pa.table(v_rows, schema = VERSIONS_SCHEMA), out_versions_path,
        compression = "zstd",
    )
    pq.write_table(
        pa.table(c_rows, schema = CHANGES_SCHEMA), out_changes_path,
        compression = "zstd",
    )
    return len(v_rows["id"]), len(c_rows["id"])


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def _filter_widening(
    base_exprs: list[str], current_exprs: list[str],
) -> list[str]:
    """``key`` / ``key=value`` entries the current filter adds over the base."""
    base = parse_osm_tag_filter_expressions(base_exprs)
    current = parse_osm_tag_filter_expressions(current_exprs)
    added: list[str] = []
    for key, values in current.items():
        if key not in base:
            added.append(key)
        elif base[key] is None:
            continue
        elif values is None:
            added.append(f"{key}=*")
        else:
            added.extend(f"{key}={v}" for v in sorted(values - base[key]))
    return added


def _combine(
    base_path: Path,
    window_paths: list[Path],
    out_path: Path,
    columns: list[str],
    duckdb_memory_limit: str,
) -> None:
    """``base UNION ALL window`` → ``out_path``. Window rows are deduped
    across extracts on ``(type, id, version)``, earliest extract wins."""
    cols = ", ".join(f'"{c}"' for c in columns)
    union_parts = [
        f"SELECT {cols}, {rank} AS _rank FROM read_parquet('{p}')"
        for rank, p in enumerate(window_paths)
    ]
    window_sql = " UNION ALL ".join(union_parts) if union_parts else None
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
        con.execute("SET preserve_insertion_order = false")
        if window_sql is None:
            select = f"SELECT {cols} FROM read_parquet('{base_path}')"
        else:
            con.execute(f"CREATE TABLE w AS {window_sql}")
            con.execute("""
                CREATE TABLE first_rank AS
                SELECT type, id, version, min(_rank) AS _rank
                FROM w GROUP BY type, id, version
            """)
            select = f"""
                SELECT {cols} FROM read_parquet('{base_path}')
                UNION ALL
                SELECT {", ".join(f'w."{c}"' for c in columns)} FROM w
                SEMI JOIN first_rank f
                  ON w.type = f.type AND w.id = f.id
                 AND w.version = f.version AND w._rank = f._rank
            """
        tmp = out_path.with_suffix(".tmp.parquet")
        con.execute(
            f"COPY ({select}) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)"
        )
    finally:
        con.close()
    tmp.replace(out_path)


def roll_osm_history(
    base_dir: Path,
    base_version: str,
    out_dir: Path,
    out_versions_path: Path,
    out_changes_path: Path,
    replication_urls: dict[str, str],
    tag_filter_exprs: list[str],
    end_date: datetime.datetime | None = None,
    max_chain_months: int = 12,
    keep_diffs: bool = False,
    duckdb_memory_limit: str = "6GB",
    state_getter_factory: Callable[[str], StateGetter] = replication_state_getter,
    verbose: bool = True,
) -> HistoryCoverage:
    """Roll ``base_dir``'s history parquets forward to ``end_date``.

    Args:
        base_dir: ``osm_data/<base_version>/`` holding the base parquets.
        base_version: Its version string (recorded in the coverage file).
        out_dir: New ``osm_data/<v>/`` directory; must differ from base_dir.
        out_versions_path / out_changes_path: Final parquet paths.
        replication_urls: ``{extract name: Geofabrik *-updates/ URL}``; the
            first entry should be the US mainland extract.
        tag_filter_exprs: Current ingest filter
            (``build_osm_tag_filter_expressions``).
        end_date: Last instant to cover; None = newest diff on the server.
        max_chain_months: Warn when this many incremental runs are chained.
        keep_diffs: Keep the downloaded ``.osc.gz`` under ``out_dir/diffs``.
        state_getter_factory: Replication-state source (tests inject one).

    Returns:
        The coverage written to ``out_dir/history_coverage.json``.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if base_dir == out_dir:
        raise ValueError(
            f"Incremental output {out_dir} is the base directory. Bump "
            "versions.osm_data (or point incremental_history.base_version at "
            "the previous run)."
        )
    base_versions = base_dir / "osm_versions.parquet"
    base_changes = base_dir / "osm_changes.parquet"
    base_cov = read_coverage(base_dir)
    if end_date is not None and _parse_ts(end_date) <= base_cov.coverage_end:
        raise ValueError(
            f"end_date {_parse_ts(end_date).isoformat()} is not after the base "
            f"run's coverage ({base_cov.coverage_end.isoformat()}); advance "
            "download.osm.end_date to this run's cutoff."
        )

    if base_cov.filter_exprs is not None:
        added = _filter_widening(base_cov.filter_exprs, tag_filter_exprs)
        if added:
            raise ValueError(
                "The ingest filter is wider than the base run's; elements "
                "newly in scope would have no history before the diff window. "
                f"Added: {added[:20]}{' ...' if len(added) > 20 else ''}. "
                "Run download.osm.history_mode: full."
            )
    else:
        print(
            "WARNING: base run recorded no filter expressions; cannot check "
            "for a widened ingest filter."
        )

    chain_length = base_cov.chain_length + 1
    if chain_length >= max_chain_months:
        print(
            f"WARNING: {chain_length} chained incremental history runs since "
            f"the last full build (limit {max_chain_months}). Schedule a "
            "history_mode: full run."
        )

    out_dir.mkdir(parents = True, exist_ok = True)
    diff_root = out_dir / "diffs"
    parsed_filter = parse_osm_tag_filter_expressions(tag_filter_exprs)

    window_versions: list[Path] = []
    window_changes: list[Path] = []
    extract_state: dict[str, dict] = {}
    for name, url in replication_urls.items():
        if verbose:
            print(f"\n=== {name}: {url}")
        get_state = state_getter_factory(url)
        try:
            newest = get_state(None)
        except Exception as exc:  # noqa: BLE001 - network layer varies
            newest = None
            print(f"  could not read state: {exc}")
        if newest is None:
            if name == next(iter(replication_urls)):
                raise RuntimeError(f"Replication feed unavailable: {url}")
            print(
                f"WARNING: {name} replication feed unavailable; its base rows "
                "are carried forward unchanged."
            )
            continue
        last_seq = base_cov.extracts.get(name, {}).get("last_sequence")
        plan = plan_sequences(
            get_state, base_cov.coverage_end, end_date, last_sequence = last_seq,
        )
        if plan is None:
            print(f"  {name}: no new diffs.")
            if name in base_cov.extracts:
                extract_state[name] = base_cov.extracts[name]
            continue
        if verbose:
            print(
                f"  sequences {plan.start}-{plan.end} "
                f"({plan.end - plan.start + 1} files, through "
                f"{plan.end_timestamp.isoformat()})"
            )
        work = diff_root / name
        diffs = fetch_diffs(url, plan, get_state, work, verbose = verbose)
        parsed_dir = work / "parsed"
        parsed_dir.mkdir(parents = True, exist_ok = True)
        window_files = []
        for diff in diffs:
            out = parsed_dir / f"{diff.sequence:09d}.parquet"
            parse_diff_file(diff, out, parsed_filter)
            window_files.append(out)
        wv = work / "window_versions.parquet"
        wc = work / "window_changes.parquet"
        n_v, n_c = roll_extract(
            window_files, base_versions, base_changes, wv, wc,
            duckdb_memory_limit = duckdb_memory_limit, verbose = verbose,
        )
        if verbose:
            print(f"  {name}: {n_v:,} new versions, {n_c:,} change rows")
        window_versions.append(wv)
        window_changes.append(wc)
        extract_state[name] = {
            "server": url,
            "last_sequence": plan.end,
            "last_timestamp": plan.end_timestamp.isoformat(),
        }

    if not window_versions:
        raise ValueError(
            "No new diffs in any feed between the base coverage "
            f"({base_cov.coverage_end.isoformat()}) and end_date; advance "
            "download.osm.end_date past the next daily diff (~20:20 UTC)."
        )
    if verbose:
        print("\nWriting rolled parquets ...")
    _combine(
        base_versions, window_versions, Path(out_versions_path),
        VERSIONS_SCHEMA.names, duckdb_memory_limit,
    )
    _combine(
        base_changes, window_changes, Path(out_changes_path),
        CHANGES_SCHEMA.names, duckdb_memory_limit,
    )

    ends = [_parse_ts(s["last_timestamp"]) for s in extract_state.values()]
    coverage = HistoryCoverage(
        mode = "incremental",
        coverage_end = min(ends) if ends else base_cov.coverage_end,
        extracts = extract_state,
        chain_length = chain_length,
        base_version = base_version,
        filter_exprs = list(tag_filter_exprs),
    )
    write_coverage(out_dir, coverage)
    if not keep_diffs and diff_root.exists():
        shutil.rmtree(diff_root)
    if verbose:
        print(
            f"Rolled history written to {out_dir} "
            f"(coverage_end {coverage.coverage_end.isoformat()}, "
            f"chain length {chain_length})."
        )
    return coverage


# -----------------------------------------------------------------------------
# Monthly QA: history vs. snapshot
# -----------------------------------------------------------------------------


def check_history_vs_snapshot(
    osm_data_dir: Path,
    snapshot_path: Path,
    coverage_end: datetime.datetime | None = None,
    duckdb_memory_limit: str = "6GB",
) -> dict:
    """Compare the history's last state of each snapshot node with the snapshot.

    For every node in ``snapshot_path`` last edited before ``coverage_end``,
    the history must hold a last version with the same timestamp and name. A
    shortfall means missed diffs or a bad fold.

    Returns:
        Dict with ``n_checked``, ``n_missing`` (not in history),
        ``n_ts_match``, ``n_name_match`` (among timestamp matches) and the
        rates ``ts_match_rate`` / ``name_match_rate``.
    """
    osm_data_dir = Path(osm_data_dir)
    if coverage_end is None:
        coverage_end = read_coverage(osm_data_dir).coverage_end
    versions = osm_data_dir / "osm_versions.parquet"
    changes = osm_data_dir / "osm_changes.parquet"
    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit = '{duckdb_memory_limit}'")
        con.execute(f"""
            CREATE TABLE snap AS
            SELECT CAST(osm_id AS BIGINT) AS id,
                   CAST(last_edited AS TIMESTAMPTZ) AS last_edited,
                   name
            FROM read_parquet('{snapshot_path}')
            WHERE osm_type = 'node'
              AND CAST(last_edited AS TIMESTAMPTZ)
                  < TIMESTAMPTZ '{_parse_ts(coverage_end).isoformat()}'
        """)
        con.execute(f"""
            CREATE TABLE hist AS
            SELECT id, max(version) AS version,
                   arg_max(CAST(timestamp AS TIMESTAMPTZ), version) AS ts
            FROM read_parquet('{versions}') v
            WHERE type = 'node' AND id IN (SELECT id FROM snap)
            GROUP BY id
        """)
        con.execute(f"""
            CREATE TABLE hist_name AS
            SELECT id, arg_max(value, version) AS value,
                   arg_max(change, version) AS change
            FROM read_parquet('{changes}')
            WHERE type = 'node' AND key = 'name'
              AND id IN (SELECT id FROM snap)
            GROUP BY id
        """)
        row = con.execute("""
            SELECT count(*),
                   count(*) FILTER (WHERE h.id IS NULL),
                   count(*) FILTER (WHERE h.ts = s.last_edited),
                   count(*) FILTER (
                       WHERE h.ts = s.last_edited
                         AND coalesce(s.name, '') = coalesce(
                             CASE WHEN n.change <> 'Deleted' THEN n.value END, '')
                   )
            FROM snap s
            LEFT JOIN hist h ON s.id = h.id
            LEFT JOIN hist_name n ON s.id = n.id
        """).fetchone()
    finally:
        con.close()
    n_checked, n_missing, n_ts, n_name = (int(x) for x in row)
    return {
        "n_checked": n_checked,
        "n_missing": n_missing,
        "n_ts_match": n_ts,
        "n_name_match": n_name,
        "ts_match_rate": n_ts / n_checked if n_checked else float("nan"),
        "name_match_rate": n_name / n_ts if n_ts else float("nan"),
    }
