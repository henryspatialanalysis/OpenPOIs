#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Manual confidence overrides, applied AFTER calibration.

A small hand-curated CSV (``manual_overrides.csv``) pins individual POIs
that the automated pipeline gets wrong at any threshold: Overture-only
listings a Close report has confirmed closed, wrong POIs that keep
surfacing, or POIs Close must carry despite a low score. Rows are
appended by the Close triage lane and carry the report id that justified
them.

CSV columns (all required, header exact)::

    unified_id, overture_id, osm_type, osm_id, action, reason, date, report_id

``action`` is ``exclude`` or ``include``. Every row names its POI by at
least one of ``unified_id``, ``overture_id`` or the pair ``osm_type``
(``node`` / ``way`` / ``relation``) + ``osm_id``. When several are given,
any one hitting a conflated row applies the override. ``unified_id`` is
stable across releases (see ``openpois.conflation.stable_ids``); the
source ids keep a pin attached even if the replacement guard gives the
place a new ``unified_id``.

- ``exclude`` → ``conf_mean = conf_lower = conf_upper = 0`` and
  ``calibration_flag = 'manual_exclude'``. The row is kept, so the
  published schema and row counts are unchanged.
- ``include`` → ``conf_mean = conf_lower = conf_upper = 1`` and
  ``calibration_flag = 'manual_include'``.

This stage runs **last** in ``make conflate`` — after
``apply_calibration`` — so a forced value is never re-scaled by the
calibration curves. It is idempotent: re-running it over its own output
changes nothing. When the same id appears more than once, the last row in
the file wins; when an id is both excluded (via one column) and included
(via the other), exclude wins.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS = (
    "unified_id", "overture_id", "osm_type", "osm_id", "action", "reason",
    "date", "report_id",
)
OSM_TYPES = ("node", "way", "relation")
ACTIONS = ("exclude", "include")
FLAG_EXCLUDE = "manual_exclude"
FLAG_INCLUDE = "manual_include"
CONF_COLUMNS = ("conf_mean", "conf_lower", "conf_upper")
_FLAG_COLUMN = "calibration_flag"


def empty_overrides() -> pd.DataFrame:
    """An overrides frame with the required columns and no rows."""
    return pd.DataFrame({c: pd.Series(dtype = object) for c in REQUIRED_COLUMNS})


def read_overrides(path: Path | str | None) -> pd.DataFrame:
    """Read and validate ``manual_overrides.csv``.

    A missing path (or ``None``) yields an empty frame so the stage is a
    no-op; the caller logs that. Malformed rows raise ``ValueError`` with
    the offending 1-based data row numbers, since a silent skip would
    hide a triage entry that was meant to take effect.
    """
    if path is None:
        return empty_overrides()
    path = Path(path).expanduser()
    if not path.exists():
        return empty_overrides()
    df = pd.read_csv(path, dtype = str, keep_default_na = False)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}; "
            f"expected {list(REQUIRED_COLUMNS)}"
        )
    df = df[list(REQUIRED_COLUMNS)].copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df["action"] = df["action"].str.lower()
    df["osm_type"] = df["osm_type"].str.lower()
    # Drop fully blank lines (a trailing newline pasted in by hand).
    blank = (df[list(REQUIRED_COLUMNS)] == "").all(axis = 1)
    df = df.loc[~blank].reset_index(drop = True)

    bad_action = ~df["action"].isin(ACTIONS)
    if bad_action.any():
        rows = (df.index[bad_action] + 1).tolist()
        raise ValueError(
            f"{path}: action must be one of {list(ACTIONS)}; bad rows "
            f"(1-based, excluding header): {rows}"
        )
    has_type = df["osm_type"] != ""
    has_osm_id = df["osm_id"] != ""
    bad_osm = (
        (has_type != has_osm_id)
        | (has_type & ~df["osm_type"].isin(OSM_TYPES))
        | (has_osm_id & ~df["osm_id"].str.fullmatch(r"\d+"))
    )
    if bad_osm.any():
        rows = (df.index[bad_osm] + 1).tolist()
        raise ValueError(
            f"{path}: osm_type ({'/'.join(OSM_TYPES)}) and a numeric osm_id "
            f"go together; bad rows (1-based, excluding header): {rows}"
        )
    no_id = (df["unified_id"] == "") & (df["overture_id"] == "") & ~has_type
    if no_id.any():
        rows = (df.index[no_id] + 1).tolist()
        raise ValueError(
            f"{path}: every row needs a unified_id, an overture_id or an "
            f"osm_type + osm_id; bad rows (1-based, excluding header): {rows}"
        )
    return df


def _osm_key(osm_type, osm_id) -> str:
    """``node/123``; "" when either part is missing."""
    if osm_type is None or osm_id is None:
        return ""
    if not isinstance(osm_type, str) or not osm_type:
        return ""
    try:
        if pd.isna(osm_id):
            return ""
        return f"{osm_type.lower()}/{int(float(osm_id))}"
    except (TypeError, ValueError):
        return ""


def build_override_index(
    overrides: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Map ``unified_id``, ``overture_id`` and ``osm_type/osm_id`` to the
    row's action.

    Later rows override earlier ones for the same id.
    """
    by_unified: dict[str, str] = {}
    by_overture: dict[str, str] = {}
    by_osm: dict[str, str] = {}
    for row in overrides.itertuples(index = False):
        if row.unified_id:
            by_unified[row.unified_id] = row.action
        if row.overture_id:
            by_overture[row.overture_id] = row.action
        key = _osm_key(row.osm_type, row.osm_id)
        if key:
            by_osm[key] = row.action
    return by_unified, by_overture, by_osm


def _flags_to_list(values) -> list[str | None]:
    """String-or-None list for a pyarrow string column (pandas 3 may
    infer a ``str`` dtype and turn ``None`` into NaN)."""
    return [v if isinstance(v, str) else None for v in list(values)]


def _mask_for(
    ids: np.ndarray, index: dict[str, str], action: str,
) -> np.ndarray:
    wanted = {k for k, v in index.items() if v == action}
    if not wanted:
        return np.zeros(len(ids), dtype = bool)
    return pd.Series(ids, dtype = object).isin(wanted).to_numpy(dtype = bool)


def _frame_osm_keys(frame: pd.DataFrame) -> np.ndarray:
    if "osm_type" not in frame.columns or "osm_id" not in frame.columns:
        return np.full(len(frame), "", dtype = object)
    return np.array(
        [_osm_key(t, i) for t, i in zip(
            frame["osm_type"].astype(object), frame["osm_id"].astype(object)
        )],
        dtype = object,
    )


def apply_overrides_to_frame(
    frame: pd.DataFrame,
    by_unified: dict[str, str],
    by_overture: dict[str, str],
    by_osm: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Apply the overrides to an in-memory frame.

    ``frame`` needs ``unified_id``, ``overture_id`` (may be null), the
    three ``conf_*`` columns and, optionally, ``osm_type`` / ``osm_id``
    and ``calibration_flag``.
    Returns ``(frame, exclude_mask, include_mask)``; exclude wins when
    both apply to one row.
    """
    uid = frame["unified_id"].astype(object).where(
        frame["unified_id"].notna(), ""
    ).to_numpy()
    if "overture_id" in frame.columns:
        oid = frame["overture_id"].astype(object).where(
            frame["overture_id"].notna(), ""
        ).to_numpy()
    else:
        oid = np.full(len(frame), "", dtype = object)

    by_osm = by_osm or {}
    osm = _frame_osm_keys(frame) if by_osm else np.full(len(frame), "", dtype = object)

    exclude = (
        _mask_for(uid, by_unified, "exclude")
        | _mask_for(oid, by_overture, "exclude")
        | _mask_for(osm, by_osm, "exclude")
    )
    include = (
        _mask_for(uid, by_unified, "include")
        | _mask_for(oid, by_overture, "include")
        | _mask_for(osm, by_osm, "include")
    ) & ~exclude

    out = frame.copy()
    for col in CONF_COLUMNS:
        values = out[col].to_numpy(dtype = float, na_value = np.nan).copy()
        values[exclude] = 0.0
        values[include] = 1.0
        out[col] = values
    if _FLAG_COLUMN in out.columns:
        flags = np.array(_flags_to_list(out[_FLAG_COLUMN]), dtype = object)
    else:
        flags = np.full(len(out), None, dtype = object)
    flags[exclude] = FLAG_EXCLUDE
    flags[include] = FLAG_INCLUDE
    out[_FLAG_COLUMN] = pd.Series(flags, index = out.index, dtype = object)
    return out, exclude, include


def apply_manual_overrides(
    input_path: Path,
    output_path: Path,
    overrides: pd.DataFrame,
    chunk_rows: int = 2_000_000,
    verbose: bool = True,
) -> dict:
    """Stream ``input_path`` to ``output_path`` applying the overrides.

    Row order, row count and every pass-through column (including the
    GeoParquet metadata) are preserved; only the three ``conf_*``
    columns and ``calibration_flag`` change. ``calibration_flag`` is
    appended when the input lacks it (an uncalibrated ablation file).
    ``output_path`` may equal ``input_path``: the result is then written
    beside it and swapped in atomically, which is how the canonical
    ``conflated.parquet`` is updated in place at the end of
    ``make conflate``.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    by_unified, by_overture, by_osm = build_override_index(overrides)
    in_place = output_path.resolve() == input_path.resolve()
    write_path = (
        output_path.with_name(output_path.name + ".tmp")
        if in_place else output_path
    )

    pf = pq.ParquetFile(str(input_path))
    base_schema = pf.schema_arrow
    if "unified_id" not in base_schema.names:
        raise ValueError(f"{input_path}: no unified_id column")
    has_flag = _FLAG_COLUMN in base_schema.names
    out_schema = base_schema if has_flag else pa.schema(
        list(base_schema) + [pa.field(_FLAG_COLUMN, pa.string())],
        metadata = base_schema.metadata,
    )
    needed = ["unified_id"] + list(CONF_COLUMNS)
    for col in ("overture_id", "osm_type", "osm_id"):
        if col in base_schema.names:
            needed.append(col)
    if has_flag:
        needed.append(_FLAG_COLUMN)

    stats = {
        "rows": 0,
        "n_overrides": int(len(overrides)),
        "n_excluded": 0,
        "n_included": 0,
        "hit_unified_ids": set(),
        "hit_overture_ids": set(),
        "hit_osm_keys": set(),
    }
    with pq.ParquetWriter(
        str(write_path), out_schema, compression = "zstd",
    ) as writer:
        for batch in pf.iter_batches(batch_size = chunk_rows):
            table = pa.Table.from_batches([batch], schema = batch.schema)
            if by_unified or by_overture or by_osm:
                frame = table.select(needed).to_pandas()
                updated, exclude, include = apply_overrides_to_frame(
                    frame, by_unified, by_overture, by_osm,
                )
                touched = exclude | include
                if touched.any():
                    for col in CONF_COLUMNS:
                        idx = table.schema.get_field_index(col)
                        table = table.set_column(
                            idx, col,
                            pa.array(
                                updated[col].to_numpy(dtype = float),
                                type = pa.float64(),
                            ),
                        )
                    flag_arr = pa.array(
                        _flags_to_list(updated[_FLAG_COLUMN]),
                        type = pa.string(),
                    )
                    if has_flag:
                        idx = table.schema.get_field_index(_FLAG_COLUMN)
                        table = table.set_column(idx, _FLAG_COLUMN, flag_arr)
                    else:
                        table = table.append_column(_FLAG_COLUMN, flag_arr)
                    stats["n_excluded"] += int(exclude.sum())
                    stats["n_included"] += int(include.sum())
                    stats["hit_unified_ids"].update(
                        frame["unified_id"].astype(object)[touched]
                        .dropna().tolist()
                    )
                    if "overture_id" in frame.columns:
                        stats["hit_overture_ids"].update(
                            frame["overture_id"].astype(object)[touched]
                            .dropna().tolist()
                        )
                    if by_osm:
                        stats["hit_osm_keys"].update(
                            k for k in _frame_osm_keys(frame)[touched] if k
                        )
                elif not has_flag:
                    table = table.append_column(
                        _FLAG_COLUMN,
                        pa.array([None] * table.num_rows, type = pa.string()),
                    )
                del frame, updated
            elif not has_flag:
                table = table.append_column(
                    _FLAG_COLUMN,
                    pa.array([None] * table.num_rows, type = pa.string()),
                )
            table = table.cast(out_schema)
            writer.write_table(table)
            stats["rows"] += table.num_rows
            del table, batch
            gc.collect()

    if in_place:
        os.replace(write_path, output_path)

    # Overrides that matched nothing: worth a warning, never an error (an
    # id can legitimately vanish from a later Overture release).
    override_osm = pd.Series(
        [_osm_key(t, i) for t, i in zip(overrides["osm_type"], overrides["osm_id"])],
        index = overrides.index, dtype = object,
    )
    unmatched = overrides[
        ~(
            overrides["unified_id"].isin(stats["hit_unified_ids"])
            | overrides["overture_id"].isin(stats["hit_overture_ids"])
            | override_osm.isin(stats["hit_osm_keys"])
        )
    ]
    stats["n_unmatched_overrides"] = int(len(unmatched))
    stats["hit_unified_ids"] = sorted(stats["hit_unified_ids"])
    stats["hit_overture_ids"] = sorted(stats["hit_overture_ids"])
    stats["hit_osm_keys"] = sorted(stats["hit_osm_keys"])
    if verbose:
        print(f"  Wrote {stats['rows']:,} rows to {output_path}")
        print(
            f"  Overrides: {stats['n_overrides']:,} rows -> "
            f"{stats['n_excluded']:,} excluded, "
            f"{stats['n_included']:,} included"
        )
        if len(unmatched):
            print(
                f"  WARNING: {len(unmatched):,} override row(s) matched no "
                f"conflated row:"
            )
            for row in unmatched.itertuples(index = False):
                print(
                    f"    unified_id={row.unified_id or '-'} "
                    f"overture_id={row.overture_id or '-'} "
                    f"osm={_osm_key(row.osm_type, row.osm_id) or '-'} "
                    f"action={row.action} report_id={row.report_id or '-'}"
                )
    return stats
