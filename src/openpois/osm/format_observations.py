#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module formats OSM changes and versions into observations, which can be more easily
queried and statistically analyzed.
"""

import os
import re
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_:]+$")


def _validate_key(k: str) -> str:
    """Allow only alphanumerics, underscores, and colons in interpolated keys.

    OSM tag keys such as ``addr:street`` are valid; anything else is rejected
    to avoid opening a SQL injection path through the pivot CTE.
    """
    if not isinstance(k, str) or not _SAFE_KEY_RE.match(k):
        raise ValueError(f"Unsafe tag key for SQL interpolation: {k!r}")
    return k


def _init_scan_state(keep_keys: list[str]) -> dict:
    return {
        "add_to_list": False,
        "last_tag_timestamp": None,
        "last_obs_timestamp": None,
        "last_tag_user": None,
        "last_tag_value": None,
        "tag_value": None,
        "keep_current": {k: None for k in keep_keys},
        "keep_last": {k: None for k in keep_keys},
    }


def _advance_scan_state(
    state: dict,
    row: tuple,
    col_idx: dict,
    tag_key: str,
    keep_keys: list[str],
) -> dict | None:
    """Run one row through the per-POI state machine.

    Returns the observation dict to emit, or ``None`` if this version is
    before the tag was first added (so ``add_to_list`` is still False).
    """
    elem_id = row[col_idx["id"]]
    osm_type = row[col_idx["type"]]
    version = row[col_idx["version"]]
    changeset = row[col_idx["changeset"]]
    obs_timestamp = row[col_idx["timestamp"]]
    user = row[col_idx["user"]]

    # `last_tag_value` on the emitted obs must reflect the PRE-update state;
    # the other `last_*` fields are updated below after `obs` is built.
    prev_last_tag_value = state["last_tag_value"]

    # Keep-keys: shift current → last only when this version's changeset
    # touches the key; otherwise current + last both stay sticky.
    for k in keep_keys:
        ch = row[col_idx[f"{k}__change"]]
        if ch is not None:
            state["keep_last"][k] = state["keep_current"][k]
            state["keep_current"][k] = row[col_idx[f"{k}__value"]]

    tag_val = row[col_idx[f"{tag_key}__value"]]
    tag_ch = row[col_idx[f"{tag_key}__change"]]
    vis_val = row[col_idx["visible__value"]]
    vis_ch = row[col_idx["visible__change"]]

    tag_added = tag_ch == "Added"
    tag_changed = tag_ch == "Changed"
    tag_deleted = tag_ch == "Deleted"
    poi_deleted = (vis_ch is not None) and (vis_val == "false")
    poi_re_added = (
        state["add_to_list"]
        and (vis_ch is not None)
        and (vis_val == "true")
    )
    any_change = (
        tag_added or tag_changed or tag_deleted or poi_deleted or poi_re_added
    )

    if tag_added:
        state["add_to_list"] = True
    if tag_added or tag_changed:
        state["last_tag_value"] = tag_val
        state["tag_value"] = tag_val
    if tag_deleted or poi_deleted:
        state["tag_value"] = None
    if poi_re_added:
        state["tag_value"] = state["last_tag_value"]

    if not state["add_to_list"]:
        return None

    obs = {
        "id": elem_id,
        "osm_type": osm_type,
        "version": version,
        "changeset": changeset,
        "obs_timestamp": obs_timestamp,
        "last_obs_timestamp": state["last_obs_timestamp"],
        "last_tag_timestamp": state["last_tag_timestamp"],
        "user": user,
        "last_tag_user": state["last_tag_user"],
        "tag_value": state["tag_value"],
        "last_tag_value": prev_last_tag_value,
        "changed": int(any_change),
        "deleted": None,
        "tag_key": tag_key,
    }
    for k in keep_keys:
        obs[k] = state["keep_current"][k]
        obs[f"{k}_last_value"] = state["keep_last"][k]

    if any_change:
        state["last_tag_timestamp"] = obs_timestamp
        state["last_tag_user"] = user
    state["last_obs_timestamp"] = obs_timestamp
    return obs


def format_observations_duckdb(
    changes_path: Path,
    versions_path: Path,
    output_path: Path,
    tag_key: str,
    keep_keys: list[str],
    duckdb_memory_limit: str = "4GB",
    duckdb_threads: int | None = None,
    duckdb_temp_dir: Path | None = None,
    batch_rows: int = 100_000,
    verbose: bool = True,
) -> int:
    """
    Stream POI observations from Parquet inputs to Parquet via DuckDB.

    DuckDB pivots the long-form ``osm_changes.parquet`` wide by tag key,
    LEFT-joins ``osm_versions.parquet`` on ``(type, id, version)``, and
    returns rows sorted by ``(type, id, version)``; the sort spills to
    ``duckdb_temp_dir`` past ``duckdb_memory_limit``. A Python scan then
    iterates the sorted stream through :func:`_advance_scan_state`,
    buffering emitted observations per DuckDB fetch batch and flushing
    them as ``pyarrow.Table`` record batches to a ``ParquetWriter``.

    Peak RSS is bounded to roughly ``duckdb_memory_limit`` plus one
    fetch batch of observations, regardless of input size.

    Args:
        changes_path: Input ``osm_changes.parquet``.
        versions_path: Input ``osm_versions.parquet``.
        output_path: Destination ``.parquet``. Overwritten.
        tag_key: Tag key to model (e.g. ``"name"``).
        keep_keys: Tag keys to retain on each observation. Must not
            include special characters (validated against
            ``[A-Za-z0-9_:]+``).
        duckdb_memory_limit: DuckDB ``memory_limit`` setting. The sort
            operator spills to disk past this.
        duckdb_threads: DuckDB worker thread count. Defaults to
            ``os.cpu_count()``.
        duckdb_temp_dir: Sort-spill directory. Defaults to
            ``output_path.parent``.
        batch_rows: Rows pulled per ``fetchmany`` call; also the
            ParquetWriter flush size.
        verbose: Print progress.

    Returns:
        Total number of observation rows written.
    """
    changes_path = Path(changes_path)
    versions_path = Path(versions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    tag_key = _validate_key(tag_key)
    keep_keys = [_validate_key(k) for k in keep_keys]
    # Pivot needs tag_key, 'visible', and all keep_keys (deduplicated).
    pivot_keys: list[str] = [tag_key, "visible"]
    for k in keep_keys:
        if k not in pivot_keys:
            pivot_keys.append(k)

    threads = duckdb_threads if duckdb_threads is not None else (os.cpu_count() or 1)
    temp_dir = (
        Path(duckdb_temp_dir) if duckdb_temp_dir is not None else output_path.parent
    )
    temp_dir.mkdir(parents = True, exist_ok = True)

    pivot_exprs: list[str] = []
    for k in pivot_keys:
        pivot_exprs.append(
            f"MAX(CASE WHEN key = '{k}' THEN value  END) AS \"{k}__value\""
        )
        pivot_exprs.append(
            f"MAX(CASE WHEN key = '{k}' THEN change END) AS \"{k}__change\""
        )
    pivot_select = ",\n            ".join(pivot_exprs)
    key_list_sql = ", ".join(f"'{k}'" for k in pivot_keys)
    pivot_cols_sql = ", ".join(
        f'p."{k}__value", p."{k}__change"' for k in pivot_keys
    )

    sql = f"""
    WITH pivoted AS (
        SELECT type, id, version,
            {pivot_select}
        FROM read_parquet('{changes_path.as_posix()}')
        WHERE key IN ({key_list_sql})
        GROUP BY type, id, version
    )
    SELECT v.type, v.id, v.version, v.changeset, v.timestamp, v."user",
           {pivot_cols_sql}
    FROM read_parquet('{versions_path.as_posix()}') v
    LEFT JOIN pivoted p USING (type, id, version)
    ORDER BY v.type, v.id, v.version
    """

    base_cols = ["type", "id", "version", "changeset", "timestamp", "user"]
    col_idx: dict = {c: i for i, c in enumerate(base_cols)}
    for k in pivot_keys:
        col_idx[f"{k}__value"] = len(col_idx)
        col_idx[f"{k}__change"] = len(col_idx)

    schema_fields = [
        ("id", pa.int64()),
        ("osm_type", pa.string()),
        ("version", pa.int64()),
        ("changeset", pa.int64()),
        ("obs_timestamp", pa.string()),
        ("last_obs_timestamp", pa.string()),
        ("last_tag_timestamp", pa.string()),
        ("user", pa.string()),
        ("last_tag_user", pa.string()),
        ("tag_value", pa.string()),
        ("last_tag_value", pa.string()),
        ("changed", pa.int8()),
        ("deleted", pa.int8()),
    ]
    for k in keep_keys:
        schema_fields.append((k, pa.string()))
    for k in keep_keys:
        schema_fields.append((f"{k}_last_value", pa.string()))
    schema_fields.append(("tag_key", pa.string()))
    schema = pa.schema(schema_fields)

    con = duckdb.connect()
    try:
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.execute(f"SET threads TO {int(threads)}")
        con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
        if verbose:
            print(
                f"DuckDB streaming observations "
                f"(threads={threads}, mem={duckdb_memory_limit})..."
            )

        cursor = con.execute(sql)

        total = 0
        # Column-oriented buffers (one list per output field) avoid building and
        # re-iterating a Python dict per row in pa.Table.from_pylist; the
        # emitted observation dict is mapped straight into per-column lists.
        field_names = [f.name for f in schema]
        col_buffers: dict[str, list] = {name: [] for name in field_names}

        def _flush(writer: pq.ParquetWriter) -> None:
            if not col_buffers[field_names[0]]:
                return
            table = pa.table(
                {
                    f.name: pa.array(col_buffers[f.name], type = f.type)
                    for f in schema
                },
                schema = schema,
            )
            writer.write_table(table)
            for values in col_buffers.values():
                values.clear()

        with pq.ParquetWriter(output_path, schema, compression = "zstd") as writer:
            current_poi = None
            state = None
            while True:
                rows = cursor.fetchmany(batch_rows)
                if not rows:
                    break
                for row in rows:
                    poi_key = (row[col_idx["type"]], row[col_idx["id"]])
                    if poi_key != current_poi:
                        current_poi = poi_key
                        state = _init_scan_state(keep_keys)
                    obs = _advance_scan_state(
                        state, row, col_idx, tag_key, keep_keys
                    )
                    if obs is not None:
                        for name in field_names:
                            col_buffers[name].append(obs.get(name))
                        total += 1
                _flush(writer)
            _flush(writer)
    finally:
        con.close()

    if verbose:
        print(f"Wrote {total:,} observations to {output_path}")
    return total


# -----------------------------------------------------------------------------
# DuckDB window-function implementation (Stage 2b)
#
# A pure-SQL reimplementation of the per-POI state machine above, expressed as
# window functions over `PARTITION BY (type, id) ORDER BY version`. It produces
# byte-for-byte the same observations as `format_observations_duckdb` (gated by
# tests/test_format_observations.py) but runs inside DuckDB's streaming,
# out-of-core engine instead of a Python row loop, so it is far faster while
# staying memory-bounded.
#
# Two unlikely sentinels stand in for SQL NULL so that `last_value(... IGNORE
# NULLS)` carry-forward can distinguish "deleted / explicitly None" (a real
# state to propagate) from "no event this row" (skip). They are mapped back to
# NULL in the final projection.
# -----------------------------------------------------------------------------

_DEL_SENTINEL = "__openpois_deleted_sentinel__"
_NULL_SENTINEL = "__openpois_null_sentinel__"


def _window_sql(
    changes_path: Path,
    versions_path: Path,
    tag_key: str,
    keep_keys: list[str],
    num_buckets: int = 1,
    bucket: int = 0,
) -> str:
    """Build the window-function SQL that reproduces the state machine.

    See the state machine in ``_advance_scan_state`` for the semantics each
    clause mirrors. Key correspondences:

    * ``add_to_list`` → cumulative ``MAX(tag_added)`` (never resets within a POI).
    * ``tag_value`` → carry-forward of a per-row "set event" (add/change → new
      value, delete → sentinel, re-add → last set value), via
      ``last_value(... IGNORE NULLS)``.
    * emitted ``last_tag_value`` → last add/change value *strictly before* the row.
    * ``last_obs_timestamp`` / ``last_tag_timestamp`` / ``last_tag_user`` →
      carries computed over **emitted rows only** (the state machine updates
      these only after the ``add_to_list`` gate), so they live in a CTE that
      filters to ``add_to_list = 1`` first.
    * ``keep_current`` / ``keep_last`` → carries over the **full** partition
      (the state machine updates keep-keys even on pre-add rows).
    """
    pivot_keys: list[str] = [tag_key, "visible"]
    for k in keep_keys:
        if k not in pivot_keys:
            pivot_keys.append(k)

    pivot_exprs = []
    for k in pivot_keys:
        pivot_exprs.append(
            f"MAX(CASE WHEN key = '{k}' THEN value  END) AS \"{k}__value\""
        )
        pivot_exprs.append(
            f"MAX(CASE WHEN key = '{k}' THEN change END) AS \"{k}__change\""
        )
    pivot_select = ",\n            ".join(pivot_exprs)
    key_list_sql = ", ".join(f"'{k}'" for k in pivot_keys)
    joined_pivot_cols = ", ".join(
        f'p."{k}__value", p."{k}__change"' for k in pivot_keys
    )

    w_all = (
        "PARTITION BY type, id ORDER BY version "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    )
    w_excl = (
        "PARTITION BY type, id ORDER BY version "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
    )

    # Keep-key carry expressions (full-partition windows).
    kc_set = ",\n        ".join(
        f'CASE WHEN "{k}__change" IS NOT NULL '
        f"THEN COALESCE(\"{k}__value\", '{_NULL_SENTINEL}') END AS \"{k}__kc_set\""
        for k in keep_keys
    )
    kc_carried = ",\n        ".join(
        f'last_value("{k}__kc_set" IGNORE NULLS) OVER w_all AS "{k}__kc_carried", '
        f'last_value("{k}__kc_set" IGNORE NULLS) OVER w_excl AS "{k}__kc_excl"'
        for k in keep_keys
    )
    kl_event = ",\n        ".join(
        f'CASE WHEN "{k}__change" IS NOT NULL '
        f"THEN COALESCE(\"{k}__kc_excl\", '{_NULL_SENTINEL}') END AS \"{k}__kl_event\""
        for k in keep_keys
    )
    kl_carried = ",\n        ".join(
        f'last_value("{k}__kl_event" IGNORE NULLS) OVER w_all AS "{k}__kl_carried"'
        for k in keep_keys
    )
    keep_passthrough = ", ".join(
        f'"{k}__kc_carried", "{k}__kl_carried"' for k in keep_keys
    )
    keep_out_current = ",\n        ".join(
        f'NULLIF("{k}__kc_carried", \'{_NULL_SENTINEL}\') AS "{k}"'
        for k in keep_keys
    )
    keep_out_last = ",\n        ".join(
        f'NULLIF("{k}__kl_carried", \'{_NULL_SENTINEL}\') AS "{k}_last_value"'
        for k in keep_keys
    )

    tag = tag_key
    kc_passthrough_sets = ", ".join(f'"{k}__kc_set"' for k in keep_keys)
    kc_passthrough_excl = ", ".join(f'"{k}__kc_excl"' for k in keep_keys)
    kl_passthrough = ", ".join(f'"{k}__kl_event"' for k in keep_keys)

    # Keep-key carry CTEs are only emitted when there are keep_keys (an empty
    # ``SELECT * EXCLUDE ()`` is a syntax error). When there are none, tv1 reads
    # straight from ``flags``.
    if keep_keys:
        keep_ctes = f"""
    kc1 AS (
        SELECT *,
        {kc_set}
        FROM flags
    ),
    kc2 AS (
        SELECT * EXCLUDE ({kc_passthrough_sets}),
        {kc_carried}
        FROM kc1
        WINDOW w_all AS ({w_all}), w_excl AS ({w_excl})
    ),
    kc3 AS (
        SELECT *,
        {kl_event}
        FROM kc2
    ),
    kc4 AS (
        SELECT * EXCLUDE ({kc_passthrough_excl}, {kl_passthrough}),
        {kl_carried}
        FROM kc3
        WINDOW w_all AS ({w_all})
    ),"""
        tv1_source = "kc4"
    else:
        keep_ctes = ""
        tv1_source = "flags"

    # Per-bucket execution keeps memory bounded: windows partition by (type,id)
    # and hash(id) depends only on id, so a given POI's rows always land in one
    # bucket — bucketing is exact, never splitting a partition.
    changes_bucket = (
        f" AND hash(id) % {num_buckets} = {bucket}" if num_buckets > 1 else ""
    )
    versions_bucket = (
        f" WHERE hash(v.id) % {num_buckets} = {bucket}" if num_buckets > 1 else ""
    )

    return f"""
    WITH pivoted AS (
        SELECT type, id, version,
            {pivot_select}
        FROM read_parquet('{changes_path.as_posix()}')
        WHERE key IN ({key_list_sql}){changes_bucket}
        GROUP BY type, id, version
    ),
    joined AS (
        SELECT v.type, v.id, v.version, v.changeset, v.timestamp, v."user",
               {joined_pivot_cols}
        FROM read_parquet('{versions_path.as_posix()}') v
        LEFT JOIN pivoted p USING (type, id, version){versions_bucket}
    ),
    flags AS (
        SELECT *,
            ("{tag}__change" = 'Added')   AS tag_added,
            ("{tag}__change" = 'Changed') AS tag_changed,
            ("{tag}__change" = 'Deleted') AS tag_deleted,
            ("visible__change" IS NOT NULL AND "visible__value" = 'false')
                AS poi_deleted,
            MAX(CASE WHEN "{tag}__change" = 'Added' THEN 1 ELSE 0 END)
                OVER w_all AS add_to_list
        FROM joined
        WINDOW w_all AS ({w_all})
    ),{keep_ctes}
    tv1 AS (
        SELECT *,
            (add_to_list = 1 AND "visible__change" IS NOT NULL
                AND "visible__value" = 'true') AS poi_re_added
        FROM {tv1_source}
    ),
    tv2 AS (
        SELECT *,
            last_value(
                CASE WHEN tag_added OR tag_changed THEN "{tag}__value" END
                IGNORE NULLS) OVER w_all AS last_set_val,
            last_value(
                CASE WHEN tag_added OR tag_changed THEN "{tag}__value" END
                IGNORE NULLS) OVER w_excl AS last_set_val_excl
        FROM tv1
        WINDOW w_all AS ({w_all}), w_excl AS ({w_excl})
    ),
    tv3 AS (
        SELECT *,
            (tag_added OR tag_changed OR tag_deleted OR poi_deleted
                OR poi_re_added) AS any_change,
            CASE
                WHEN tag_added OR tag_changed THEN "{tag}__value"
                WHEN tag_deleted OR poi_deleted THEN '{_DEL_SENTINEL}'
                WHEN poi_re_added THEN last_set_val
                ELSE NULL
            END AS tv_set
        FROM tv2
    ),
    tv4 AS (
        SELECT *,
            last_value(tv_set IGNORE NULLS) OVER w_all AS tag_value_raw
        FROM tv3
        WINDOW w_all AS ({w_all})
    ),
    emitted AS (
        SELECT * FROM tv4 WHERE add_to_list = 1
    ),
    final AS (
        SELECT *,
            LAG(timestamp) OVER w_part AS last_obs_ts,
            last_value(CASE WHEN any_change THEN timestamp END IGNORE NULLS)
                OVER w_excl AS last_tag_ts,
            last_value(CASE WHEN any_change THEN "user" END IGNORE NULLS)
                OVER w_excl AS last_tag_usr
        FROM emitted
        WINDOW w_part AS (PARTITION BY type, id ORDER BY version),
               w_excl AS ({w_excl})
    )
    SELECT
        id,
        type AS osm_type,
        version,
        changeset,
        timestamp AS obs_timestamp,
        last_obs_ts AS last_obs_timestamp,
        last_tag_ts AS last_tag_timestamp,
        "user",
        last_tag_usr AS last_tag_user,
        CASE WHEN tag_value_raw = '{_DEL_SENTINEL}' THEN NULL ELSE tag_value_raw END
            AS tag_value,
        last_set_val_excl AS last_tag_value,
        CAST(CASE WHEN any_change THEN 1 ELSE 0 END AS TINYINT) AS changed,
        CAST(NULL AS TINYINT) AS deleted,
        {keep_out_current + ',' if keep_keys else ''}
        {keep_out_last + ',' if keep_keys else ''}
        '{tag}' AS tag_key
    FROM final
    ORDER BY type, id, version
    """


def format_observations_window(
    changes_path: Path,
    versions_path: Path,
    output_path: Path,
    tag_key: str,
    keep_keys: list[str],
    duckdb_memory_limit: str = "4GB",
    duckdb_threads: int | None = None,
    duckdb_temp_dir: Path | None = None,
    num_buckets: int = 16,
    verbose: bool = True,
) -> int:
    """
    Stream POI observations using DuckDB window functions (Stage 2b).

    Drop-in replacement for :func:`format_observations_duckdb` that produces
    identical observations (validated row-for-row by the golden test) but
    executes the per-POI state machine as SQL window functions inside DuckDB
    rather than a Python row loop.

    The chained window operators materialise per-partition state, so a
    single-pass national query needs many GB. To stay within
    ``duckdb_memory_limit`` the work is split into ``num_buckets`` passes on
    ``hash(id) % num_buckets``: windows partition by ``(type, id)`` and the hash
    depends only on ``id``, so every POI's rows fall in exactly one bucket
    (bucketing never splits a partition). Each bucket is streamed to a single
    output Parquet via a record-batch reader, bounding peak RAM to roughly
    ``(total rows / num_buckets)`` worth of window state.

    Row order is by ``(type, id, version)`` within each bucket but not across
    buckets; downstream consumers do not depend on global row order, and the
    golden test compares as an unordered multiset.

    Args:
        num_buckets: Number of ``hash(id)`` passes. Higher → lower peak RAM,
            slightly more total scan work. ``1`` runs a single pass.
        (others) see :func:`format_observations_duckdb`.

    Returns:
        Total number of observation rows written.
    """
    changes_path = Path(changes_path)
    versions_path = Path(versions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    tag_key = _validate_key(tag_key)
    keep_keys = [_validate_key(k) for k in keep_keys]

    threads = duckdb_threads if duckdb_threads is not None else (os.cpu_count() or 1)
    temp_dir = (
        Path(duckdb_temp_dir) if duckdb_temp_dir is not None else output_path.parent
    )
    temp_dir.mkdir(parents = True, exist_ok = True)

    con = duckdb.connect()
    total = 0
    writer: pq.ParquetWriter | None = None
    try:
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.execute(f"SET threads TO {int(threads)}")
        con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
        con.execute("SET preserve_insertion_order=false")
        if verbose:
            print(
                f"DuckDB window-function observations "
                f"(threads={threads}, mem={duckdb_memory_limit}, "
                f"buckets={num_buckets})..."
            )
        for bucket in range(num_buckets):
            sql = _window_sql(
                changes_path, versions_path, tag_key, keep_keys,
                num_buckets = num_buckets, bucket = bucket,
            )
            reader = con.execute(sql).fetch_record_batch()
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, batch.schema, compression = "zstd"
                    )
                writer.write_batch(batch)
                total += batch.num_rows
            if verbose:
                print(f"  bucket {bucket + 1}/{num_buckets}: {total:,} rows so far")
    finally:
        if writer is not None:
            writer.close()
        con.close()

    if verbose:
        print(f"Wrote {total:,} observations to {output_path}")
    return int(total)
