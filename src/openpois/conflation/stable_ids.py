#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Release-stable ``unified_id`` values for the conflated dataset.

The matcher re-pairs OSM and Overture from scratch every release, so an id
built from the current pairing (``matched:<osm>_<overture>``) changes for an
unchanged place whenever either partner changes. This module carries ids
forward instead, through an **id registry**: one row per source key ever seen
(``osm:n123`` / ``osm:w123`` / ``osm:r123`` / ``ovt:<gers id>``) mapping it to
the opaque id of the row it last belonged to.

Each release, every conflated row looks up the ids of its source keys:

1. **Edges.** OSM key -> registry id; Overture key -> registry id; and, for
   an Overture dedup winner, each dropped loser's key -> registry id (an
   *alias* edge). An OSM edge whose name changed to a different business
   (``ghost_osm.names_match`` fails on both non-empty names) is a
   *replacement*: the row may not take that id by any edge.
2. **Ranking.** A key the id held on its most recent row outranks a key it
   held only earlier ("stale"); then OSM > Overture > alias; then the older
   id; then the id and key strings, so the result is deterministic.
3. **Greedy one-to-one.** Edges are taken in rank order; each id goes to at
   most one row and each row takes at most one id. So a merge keeps the
   OSM-side id, and a split keeps the id on the OSM side.
4. **Minting.** Rows left without an id get ``<prefix>`` + 16 base32
   characters of ``blake2b(founding key)`` (80 bits), founding key = the OSM
   key if present, else the Overture key. A hash already taken (by any id in
   the registry, including retired ones) is re-salted ``key|1``, ``key|2``...

The registry is updated for every key present and keeps absent keys, so an
id whose key returns after a gap is recovered. An id that no key maps to any
more is kept as a tombstone row (null ``source_key``) so it is never minted
again for a different place. ``id_changes`` records, per release, which ids
were minted, merged into another id, split from one, retired, revived, or
re-minted by the replacement guard.

All heavy work runs in DuckDB over key columns; Python handles only the
conflicted edges, the guard's name comparisons and the hashing. The
conflated parquet is never loaded whole: ``rewrite_unified_id`` streams it
one record batch at a time and joins the new ids back by row position.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import os
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openpois.conflation.ghost_osm import names_match

ID_PREFIX = "opoi_"
GUARD_THRESHOLD = 70.0

REGISTRY_SCHEMA = pa.schema([
    ("source_key", pa.string()),
    ("unified_id", pa.string()),
    ("id_minted", pa.string()),
    ("first_release", pa.string()),
    ("last_release", pa.string()),
    ("last_name", pa.string()),
])
CHANGES_SCHEMA = pa.schema([
    ("old_id", pa.string()),
    ("new_id", pa.string()),
    ("change", pa.string()),
    ("release", pa.string()),
])
CHANGE_TYPES = (
    "minted", "merged_into", "split_from", "retired", "revived",
    "renamed_guard",
)

# Edge kinds, in precedence order.
_KIND_OSM = 0
_KIND_OVT = 1
_KIND_ALIAS = 2


# -----------------------------------------------------------------
# Keys and minting
# -----------------------------------------------------------------


def osm_key_sql(type_col: str = "osm_type", id_col: str = "osm_id") -> str:
    """DuckDB expression for the OSM source key (``osm:n123``).

    Accepts a double ``osm_id`` (published releases up to 2026-09-02).
    """
    return (
        f"CASE WHEN {id_col} IS NOT NULL AND {type_col} IS NOT NULL "
        f"THEN 'osm:' || left(lower({type_col}), 1) || "
        f"CAST(CAST({id_col} AS BIGINT) AS VARCHAR) END"
    )


def overture_key_sql(id_col: str = "overture_id") -> str:
    """DuckDB expression for the Overture source key (``ovt:<gers id>``)."""
    return f"CASE WHEN {id_col} IS NOT NULL THEN 'ovt:' || {id_col} END"


def mint_id(key: str, salt: int = 0, prefix: str = ID_PREFIX) -> str:
    """Opaque id for a founding source key: prefix + 16 base32 characters."""
    text = key if salt == 0 else f"{key}|{salt}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size = 10).digest()
    return prefix + base64.b32encode(digest).decode("ascii").lower()


# -----------------------------------------------------------------
# Loading
# -----------------------------------------------------------------


def connect(memory_limit: str = "8GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = true")
    return con


def _sql_list(paths: list[Path | str]) -> str:
    return "[" + ", ".join(f"'{Path(p)}'" for p in paths) + "]"


def load_release(
    con: duckdb.DuckDBPyConnection,
    paths: list[Path | str],
) -> int:
    """Load one release's key columns into table ``cur``.

    ``paths`` is the single conflated parquet or every file of a
    partitioned release. ``cur`` holds ``rid`` (dense row number),
    ``filename``, ``file_row_number``, both source keys and both
    source-specific names. Raises if a source key repeats: the whole
    scheme relies on each key belonging to one row per release.
    """
    paths = [Path(p) for p in paths]
    con.execute("DROP TABLE IF EXISTS cur")
    con.execute(f"""
        CREATE TABLE cur AS
        SELECT row_number() OVER (ORDER BY filename, file_row_number) - 1
                   AS rid,
               filename, file_row_number,
               {osm_key_sql()} AS osm_key,
               {overture_key_sql()} AS ovt_key,
               osm_name, overture_name AS ovt_name
        FROM read_parquet({_sql_list(paths)}, filename = true,
                          file_row_number = true, hive_partitioning = false)
    """)
    n_rows = con.execute("SELECT count(*) FROM cur").fetchone()[0]
    n_bad = con.execute(
        "SELECT count(*) FROM cur WHERE osm_key IS NULL AND ovt_key IS NULL"
    ).fetchone()[0]
    if n_bad:
        raise ValueError(f"{n_bad:,} rows carry neither an OSM nor an Overture key")
    for col in ("osm_key", "ovt_key"):
        n_dup = con.execute(f"""
            SELECT count(*) FROM (
                SELECT {col} FROM cur WHERE {col} IS NOT NULL
                GROUP BY 1 HAVING count(*) > 1)
        """).fetchone()[0]
        if n_dup:
            raise ValueError(f"{n_dup:,} {col} values occur on more than one row")
    return int(n_rows)


def load_registry(
    con: duckdb.DuckDBPyConnection,
    path: Path | str | None,
) -> int:
    """Load a registry parquet into table ``reg`` (empty when ``path`` is None)."""
    con.execute("DROP TABLE IF EXISTS reg")
    con.execute("""
        CREATE TABLE reg (
            source_key VARCHAR, unified_id VARCHAR, id_minted VARCHAR,
            first_release VARCHAR, last_release VARCHAR, last_name VARCHAR)
    """)
    if path is not None:
        con.execute(f"""
            INSERT INTO reg
            SELECT source_key, unified_id, id_minted, first_release,
                   last_release, last_name
            FROM read_parquet('{Path(path)}')
        """)
    return int(con.execute("SELECT count(*) FROM reg").fetchone()[0])


def load_dedup_aliases(
    con: duckdb.DuckDBPyConnection,
    path: Path | str | None,
) -> int:
    """Load an ``overture_dedup_dropped.parquet`` audit into table ``aliases``
    (``loser_key``, ``winner_key``); empty when the file is absent."""
    con.execute("DROP TABLE IF EXISTS aliases")
    con.execute("CREATE TABLE aliases (loser_key VARCHAR, winner_key VARCHAR)")
    if path is not None and Path(path).exists():
        con.execute(f"""
            INSERT INTO aliases
            SELECT DISTINCT 'ovt:' || overture_id, 'ovt:' || winner_overture_id
            FROM read_parquet('{Path(path)}')
            WHERE overture_id IS NOT NULL AND winner_overture_id IS NOT NULL
        """)
    return int(con.execute("SELECT count(*) FROM aliases").fetchone()[0])


# -----------------------------------------------------------------
# Reconciliation
# -----------------------------------------------------------------


def _build_edges(con: duckdb.DuckDBPyConnection) -> None:
    """Candidate (row, id) edges with their rank columns -> table ``edges``."""
    con.execute("DROP TABLE IF EXISTS edges")
    con.execute(f"""
        CREATE TABLE edges AS
        WITH r AS (
            SELECT source_key, unified_id, id_minted, last_release, last_name,
                   last_release < max(last_release) OVER (
                       PARTITION BY unified_id) AS stale
            FROM reg WHERE source_key IS NOT NULL
        ),
        raw AS (
            SELECT c.rid, r.unified_id, {_KIND_OSM} AS kind, r.stale,
                   r.id_minted, c.osm_key AS via_key,
                   r.last_name AS prior_name, c.osm_name AS cur_name
            FROM cur c JOIN r ON c.osm_key = r.source_key
            UNION ALL
            SELECT c.rid, r.unified_id, {_KIND_OVT}, r.stale, r.id_minted,
                   c.ovt_key, NULL, NULL
            FROM cur c JOIN r ON c.ovt_key = r.source_key
            UNION ALL
            SELECT c.rid, r.unified_id, {_KIND_ALIAS}, r.stale, r.id_minted,
                   a.loser_key, NULL, NULL
            FROM aliases a
            JOIN cur c ON c.ovt_key = a.winner_key
            JOIN r ON r.source_key = a.loser_key
            WHERE a.loser_key NOT IN (
                SELECT ovt_key FROM cur WHERE ovt_key IS NOT NULL)
        )
        SELECT * FROM raw
    """)


def _apply_guard(
    con: duckdb.DuckDBPyConnection,
    threshold: float,
) -> int:
    """Move replacement edges (OSM name changed to a different business) to
    table ``blocked`` and drop every edge between those rows and ids."""
    candidates = con.execute(f"""
        SELECT rid, unified_id, prior_name, cur_name FROM edges
        WHERE kind = {_KIND_OSM}
          AND coalesce(prior_name, '') <> '' AND coalesce(cur_name, '') <> ''
          AND prior_name <> cur_name
    """).fetchall()
    blocked = [
        (rid, uid) for rid, uid, a, b in candidates
        if not names_match(a, b, threshold = threshold)
    ]
    con.execute("DROP TABLE IF EXISTS blocked")
    con.execute("CREATE TABLE blocked (rid BIGINT, unified_id VARCHAR)")
    if blocked:
        con.executemany("INSERT INTO blocked VALUES (?, ?)", blocked)
        con.execute("""
            DELETE FROM edges e USING blocked b
            WHERE e.rid = b.rid AND e.unified_id = b.unified_id
        """)
    return len(blocked)


def _assign(con: duckdb.DuckDBPyConnection) -> dict:
    """Greedy one-to-one assignment of ids to rows -> table ``assigned``.

    Isolated edges (row and id each on exactly one edge) are assigned in SQL;
    the rest go through an ordered greedy pass in Python. Taking isolated
    components out first gives the same result as one global greedy pass,
    since components do not interact.
    """
    # One edge per (row, id): its best rank.
    con.execute("""
        CREATE OR REPLACE TABLE best AS
        SELECT * FROM edges
        QUALIFY row_number() OVER (
            PARTITION BY rid, unified_id
            ORDER BY stale, kind, via_key) = 1
    """)
    con.execute("""
        CREATE OR REPLACE TABLE best_deg AS
        SELECT *, count(*) OVER (PARTITION BY rid) AS row_deg,
                  count(*) OVER (PARTITION BY unified_id) AS id_deg
        FROM best
    """)
    con.execute("""
        CREATE OR REPLACE TABLE assigned AS
        SELECT rid, unified_id FROM best_deg WHERE row_deg = 1 AND id_deg = 1
    """)
    contested = con.execute("""
        SELECT rid, unified_id FROM best_deg
        WHERE NOT (row_deg = 1 AND id_deg = 1)
        ORDER BY stale, kind, id_minted, unified_id, via_key, rid
    """).fetchall()
    used_rows: set[int] = set()
    used_ids: set[str] = set()
    picks = []
    for rid, uid in contested:
        if rid in used_rows or uid in used_ids:
            continue
        used_rows.add(rid)
        used_ids.add(uid)
        picks.append((rid, uid))
    if picks:
        con.executemany("INSERT INTO assigned VALUES (?, ?)", picks)
    return {"n_contested_edges": len(contested)}


def _mint(con: duckdb.DuckDBPyConnection, prefix: str) -> int:
    """Mint ids for rows without one and add them to ``assigned`` (with a
    ``minted`` flag in table ``minted``)."""
    rows = con.execute("""
        SELECT c.rid, coalesce(c.osm_key, c.ovt_key)
        FROM cur c ANTI JOIN assigned a ON c.rid = a.rid
        ORDER BY c.rid
    """).fetchall()
    con.execute("DROP TABLE IF EXISTS minted")
    con.execute("CREATE TABLE minted (rid BIGINT, unified_id VARCHAR, salt INT)")
    if not rows:
        return 0
    rids = np.array([r[0] for r in rows], dtype = np.int64)
    keys = [r[1] for r in rows]
    salts = np.zeros(len(rows), dtype = np.int32)
    ids = np.array([mint_id(k, prefix = prefix) for k in keys], dtype = object)
    pending = np.arange(len(rows))
    while True:
        con.execute("DROP TABLE IF EXISTS cand")
        cand = pa.table({
            "i": pa.array(pending, type = pa.int64()),
            "unified_id": pa.array(ids[pending].tolist(), type = pa.string()),
        })
        con.register("cand_arrow", cand)
        con.execute("CREATE TABLE cand AS SELECT * FROM cand_arrow")
        con.unregister("cand_arrow")
        # Taken: any registry id (tombstones included), any reused id, and
        # any id minted for another row in this release.
        clash = con.execute(f"""
            SELECT i FROM cand
            WHERE unified_id IN (SELECT unified_id FROM reg)
               OR unified_id IN (SELECT unified_id FROM assigned)
            UNION
            SELECT i FROM (
                SELECT i, row_number() OVER (
                    PARTITION BY unified_id ORDER BY i) AS k
                FROM cand) WHERE k > 1
        """).fetchall()
        settled = np.setdiff1d(pending, [c[0] for c in clash])
        # Fix settled ids first so a re-salted id clashes against them.
        if len(settled):
            batch = pa.table({
                "rid": pa.array(rids[settled], type = pa.int64()),
                "unified_id": pa.array(ids[settled].tolist(), type = pa.string()),
                "salt": pa.array(salts[settled], type = pa.int32()),
            })
            con.register("batch_arrow", batch)
            con.execute("INSERT INTO minted SELECT * FROM batch_arrow")
            con.execute(
                "INSERT INTO assigned SELECT rid, unified_id FROM batch_arrow"
            )
            con.unregister("batch_arrow")
        if not clash:
            break
        pending = np.array(sorted(c[0] for c in clash), dtype = np.int64)
        for i in pending:
            salts[i] += 1
            ids[i] = mint_id(keys[i], salt = int(salts[i]), prefix = prefix)
    con.execute("DROP TABLE IF EXISTS cand")
    return len(rows)


def _update_registry(con: duckdb.DuckDBPyConnection, release: str) -> None:
    """Build table ``new_reg``: present keys remapped to their row's id,
    absent keys kept, and tombstones for ids no key maps to any more."""
    con.execute(f"""
        CREATE OR REPLACE TABLE new_reg AS
        WITH id_minted AS (
            SELECT unified_id, min(id_minted) AS id_minted FROM reg
            GROUP BY 1
        ),
        row_ids AS (
            SELECT c.*, a.unified_id,
                   coalesce(m.id_minted, '{release}') AS id_minted
            FROM cur c JOIN assigned a USING (rid)
            LEFT JOIN id_minted m USING (unified_id)
        ),
        present AS (
            SELECT osm_key AS source_key, unified_id, id_minted,
                   osm_name AS last_name
            FROM row_ids WHERE osm_key IS NOT NULL
            UNION ALL
            SELECT ovt_key, unified_id, id_minted, ovt_name
            FROM row_ids WHERE ovt_key IS NOT NULL
        ),
        keyed AS (
            SELECT p.source_key, p.unified_id, p.id_minted,
                   coalesce(r.first_release, '{release}') AS first_release,
                   '{release}' AS last_release, p.last_name
            FROM present p LEFT JOIN reg r ON r.source_key = p.source_key
            UNION ALL
            SELECT r.* FROM reg r
            WHERE r.source_key IS NOT NULL
              AND r.source_key NOT IN (SELECT source_key FROM present)
        ),
        orphans AS (
            SELECT NULL AS source_key, r.unified_id, min(r.id_minted),
                   NULL AS first_release, max(r.last_release) AS last_release,
                   NULL AS last_name
            FROM reg r
            WHERE r.unified_id NOT IN (SELECT unified_id FROM keyed)
            GROUP BY r.unified_id
        )
        SELECT * FROM keyed
        UNION ALL
        SELECT * FROM orphans
        ORDER BY source_key NULLS LAST, unified_id
    """)


def _record_changes(con: duckdb.DuckDBPyConnection, release: str) -> None:
    """Build table ``changes`` (``old_id, new_id, change, release``)."""
    con.execute("""
        CREATE OR REPLACE TABLE changes_raw AS
        WITH prev AS (SELECT max(last_release) AS r FROM reg
                      WHERE source_key IS NOT NULL),
        prev_live AS (
            SELECT DISTINCT unified_id FROM reg, prev
            WHERE source_key IS NOT NULL AND last_release = prev.r
        ),
        now_ids AS (SELECT a.rid, a.unified_id FROM assigned a),
        lost AS (
            -- prior-live ids with no row now, but an unblocked edge to one
            SELECT b.unified_id AS old_id, n.unified_id AS new_id,
                   row_number() OVER (
                       PARTITION BY b.unified_id
                       ORDER BY b.stale, b.kind, b.via_key, b.rid) AS k
            FROM best b JOIN now_ids n ON n.rid = b.rid
            WHERE b.unified_id IN (SELECT unified_id FROM prev_live)
              AND b.unified_id NOT IN (SELECT unified_id FROM now_ids)
        ),
        guarded AS (
            SELECT b.unified_id AS old_id, m.unified_id AS new_id,
                   row_number() OVER (
                       PARTITION BY m.rid ORDER BY b.unified_id) AS k
            FROM blocked b JOIN minted m ON m.rid = b.rid
        ),
        split AS (
            SELECT b.unified_id AS old_id, m.unified_id AS new_id,
                   row_number() OVER (
                       PARTITION BY m.rid
                       ORDER BY b.stale, b.kind, b.via_key) AS k
            FROM best b JOIN minted m ON m.rid = b.rid
        )
        SELECT old_id, new_id, 'merged_into' AS change FROM lost WHERE k = 1
        UNION ALL
        SELECT old_id, new_id, 'renamed_guard' FROM guarded WHERE k = 1
        UNION ALL
        SELECT s.old_id, s.new_id, 'split_from' FROM split s
        WHERE s.k = 1 AND s.new_id NOT IN (SELECT new_id FROM guarded)
        UNION ALL
        SELECT NULL, m.unified_id, 'minted' FROM minted m
        WHERE m.rid NOT IN (SELECT rid FROM best)
          AND m.rid NOT IN (SELECT rid FROM blocked)
        UNION ALL
        SELECT p.unified_id, NULL, 'retired' FROM prev_live p
        WHERE p.unified_id NOT IN (SELECT unified_id FROM now_ids)
          AND p.unified_id NOT IN (SELECT old_id FROM lost)
          AND p.unified_id NOT IN (SELECT old_id FROM guarded)
        UNION ALL
        SELECT n.unified_id, n.unified_id, 'revived' FROM now_ids n
        WHERE n.unified_id IN (SELECT unified_id FROM reg)
          AND n.unified_id NOT IN (SELECT unified_id FROM prev_live)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE changes AS
        SELECT old_id, new_id, change, '{release}' AS release FROM changes_raw
        ORDER BY change, old_id NULLS FIRST, new_id
    """)
    con.execute("DROP TABLE changes_raw")


def reconcile(
    con: duckdb.DuckDBPyConnection,
    release: str,
    guard_threshold: float = GUARD_THRESHOLD,
    prefix: str = ID_PREFIX,
) -> dict:
    """Assign ids to table ``cur`` from table ``reg`` (and ``aliases``).

    Call ``load_release``, ``load_registry`` and ``load_dedup_aliases``
    first. Leaves tables ``assigned`` (``rid, unified_id``), ``new_reg``
    and ``changes``, and returns a stats dict. ``release`` must sort after
    every ``last_release`` in the registry.
    """
    for table in ("cur", "reg"):
        con.execute(f"SELECT 1 FROM {table} LIMIT 0")
    if not con.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_name = 'aliases'"
    ).fetchone()[0]:
        load_dedup_aliases(con, None)
    latest = con.execute("SELECT max(last_release) FROM reg").fetchone()[0]
    if latest is not None and latest >= release:
        raise ValueError(
            f"registry already holds release {latest}; reconcile needs the "
            f"prior release's registry, not {release} or later"
        )
    _build_edges(con)
    n_blocked = _apply_guard(con, guard_threshold)
    stats = _assign(con)
    n_minted = _mint(con, prefix)
    _update_registry(con, release)
    _record_changes(con, release)

    n_rows, n_ids = con.execute(
        "SELECT count(*), count(DISTINCT unified_id) FROM assigned"
    ).fetchone()
    n_cur = con.execute("SELECT count(*) FROM cur").fetchone()[0]
    if n_rows != n_cur or n_ids != n_cur:
        raise AssertionError(
            f"id assignment broken: {n_cur:,} rows, {n_rows:,} assigned, "
            f"{n_ids:,} distinct ids"
        )
    stats.update({
        "release": release,
        "n_rows": int(n_cur),
        "n_minted": int(n_minted),
        "n_guard_blocked": int(n_blocked),
        "n_registry_in": int(con.execute("SELECT count(*) FROM reg").fetchone()[0]),
        "n_registry_out": int(
            con.execute("SELECT count(*) FROM new_reg").fetchone()[0]
        ),
        "changes": dict(con.execute(
            "SELECT change, count(*) FROM changes GROUP BY 1 ORDER BY 1"
        ).fetchall()),
    })
    stats["n_reused"] = stats["n_rows"] - stats["n_minted"]
    return stats


def format_stats(stats: dict) -> str:
    """Multi-line summary of a ``reconcile`` stats dict, for logs and
    ``run_summary.md``."""
    lines = [
        f"release {stats['release']}: {stats['n_rows']:,} rows, "
        f"{stats['n_reused']:,} ids carried forward, "
        f"{stats['n_minted']:,} minted",
        f"  guard-blocked edges: {stats['n_guard_blocked']:,}; contested "
        f"edges: {stats['n_contested_edges']:,}",
        f"  registry: {stats['n_registry_in']:,} -> "
        f"{stats['n_registry_out']:,} rows",
    ]
    for change in CHANGE_TYPES:
        lines.append(f"  {change:>13}: {stats['changes'].get(change, 0):,}")
    return "\n".join(lines)


def write_outputs(
    con: duckdb.DuckDBPyConnection,
    registry_path: Path | str | None,
    changes_path: Path | str | None,
) -> None:
    """Write ``new_reg`` and ``changes`` as parquet (zstd)."""
    for table, path, schema in (
        ("new_reg", registry_path, REGISTRY_SCHEMA),
        ("changes", changes_path, CHANGES_SCHEMA),
    ):
        if path is None:
            continue
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)
        arrow = con.execute(f"SELECT * FROM {table}").fetch_arrow_table()
        pq.write_table(arrow.cast(schema), path, compression = "zstd")


def migration_table(con: duckdb.DuckDBPyConnection) -> pa.Table:
    """``v0_unified_id, osm_type, osm_id, overture_id, v1_unified_id`` for
    the release in ``cur`` (needs ``cur_v0``, built by ``load_v0_ids``)."""
    return con.execute("""
        SELECT v.v0_unified_id, v.osm_type, v.osm_id, v.overture_id,
               a.unified_id AS v1_unified_id
        FROM cur_v0 v JOIN assigned a USING (rid)
        ORDER BY a.unified_id
    """).fetch_arrow_table()


def load_v0_ids(
    con: duckdb.DuckDBPyConnection,
    paths: list[Path | str],
) -> None:
    """Table ``cur_v0``: each ``cur`` row's published v0 id and raw keys."""
    con.execute(f"""
        CREATE OR REPLACE TABLE cur_v0 AS
        SELECT c.rid, s.unified_id AS v0_unified_id, s.osm_type,
               CAST(s.osm_id AS BIGINT) AS osm_id, s.overture_id
        FROM read_parquet({_sql_list(paths)}, filename = true,
                          file_row_number = true, hive_partitioning = false) s
        JOIN cur c ON c.filename = s.filename
                  AND c.file_row_number = s.file_row_number
    """)


# -----------------------------------------------------------------
# Rewriting the conflated parquet
# -----------------------------------------------------------------


def _ids_for_file(con: duckdb.DuckDBPyConnection, path: Path) -> pa.Array:
    table = con.execute("""
        SELECT a.unified_id
        FROM cur c JOIN assigned a USING (rid)
        WHERE c.filename = ?
        ORDER BY c.file_row_number
    """, [str(path)]).fetch_arrow_table()
    return table.column(0).combine_chunks()


def rewrite_unified_id(
    con: duckdb.DuckDBPyConnection,
    in_path: Path | str,
    out_path: Path | str,
    cast_osm_id: bool = False,
    batch_rows: int = 1_000_000,
) -> int:
    """Stream ``in_path`` to ``out_path`` with ``unified_id`` replaced.

    The ids come from tables ``cur`` / ``assigned`` (``in_path`` must be one
    of the files passed to ``load_release``), aligned by parquet row
    position. Every other column, the row order and the schema metadata
    (GeoParquet ``geo``) are preserved; with ``cast_osm_id`` a double
    ``osm_id`` becomes int64 (it must hold whole numbers). ``out_path`` may
    equal ``in_path``: the result is written beside it and swapped in.
    Returns the row count.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    in_place = out_path.resolve() == in_path.resolve()
    write_path = (
        out_path.with_name(out_path.name + ".tmp") if in_place else out_path
    )
    write_path.parent.mkdir(parents = True, exist_ok = True)

    ids = _ids_for_file(con, in_path)
    pf = pq.ParquetFile(str(in_path))
    if len(ids) != pf.metadata.num_rows:
        raise ValueError(
            f"{in_path}: {pf.metadata.num_rows:,} rows but {len(ids):,} ids "
            "(was it loaded with load_release?)"
        )
    schema = pf.schema_arrow
    uid_idx = schema.get_field_index("unified_id")
    schema = schema.set(uid_idx, pa.field("unified_id", pa.string()))
    osm_idx = schema.get_field_index("osm_id")
    recast = cast_osm_id and osm_idx >= 0 and schema.field(osm_idx).type != pa.int64()
    if recast:
        schema = schema.set(osm_idx, pa.field("osm_id", pa.int64()))

    offset = 0
    with pq.ParquetWriter(str(write_path), schema, compression = "zstd") as writer:
        for batch in pf.iter_batches(batch_size = batch_rows):
            table = pa.Table.from_batches([batch])
            n = table.num_rows
            table = table.set_column(
                uid_idx, "unified_id", ids.slice(offset, n),
            )
            if recast:
                col = table.column(osm_idx)
                as_int = pc.cast(col, pa.int64(), safe = False)
                back = pc.cast(as_int, col.type)
                differs = pc.fill_null(pc.not_equal(back, col), False)
                if pc.any(differs).as_py():
                    raise ValueError(f"{in_path}: non-integral osm_id values")
                table = table.set_column(osm_idx, "osm_id", as_int)
            writer.write_table(table.cast(schema))
            offset += n
            del table, batch
        gc.collect()
    if in_place:
        os.replace(write_path, out_path)
    return offset


def verify_rewrite(
    con: duckdb.DuckDBPyConnection,
    path: Path | str,
    source_path: Path | str | None = None,
) -> int:
    """Check a rewritten file against ``cur`` / ``assigned``: each row's
    ``unified_id`` must be the id assigned to the row holding its source
    keys. Returns the number of mismatched rows (0 when it is right)."""
    source_path = Path(source_path or path)
    return int(con.execute(f"""
        WITH w AS (
            SELECT unified_id,
                   coalesce({osm_key_sql()}, {overture_key_sql()}) AS k
            FROM read_parquet('{Path(path)}')
        ),
        expected AS (
            SELECT coalesce(c.osm_key, c.ovt_key) AS k, a.unified_id
            FROM cur c JOIN assigned a USING (rid)
            WHERE c.filename = '{source_path}'
        )
        SELECT count(*) FROM w FULL JOIN expected e USING (k)
        WHERE w.unified_id IS DISTINCT FROM e.unified_id
    """).fetchone()[0])
