#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Row-filtered streaming copy of a (Geo)Parquet file.

Used by the conflation driver to spill the post-dedup Overture
snapshot to disk with its full merge-phase column set. The driver
loads Overture with a narrow match-only schema to bound memory, so
the spill cannot be written from the in-memory frame — the kept rows
are instead re-read straight from the source parquet, one row group
at a time, keeping peak memory at roughly a single row group of the
projected columns regardless of dataset size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def spill_rows(
    source_path: Path,
    dest_path: Path,
    keep_rows: np.ndarray,
    columns: list[str],
) -> int:
    """Copy the ``keep_rows`` rows of ``source_path`` to ``dest_path``.

    Source row order is preserved: row ``i`` of the output is the
    ``i``-th smallest index in ``keep_rows`` (the order of ``keep_rows``
    itself does not matter, and duplicate indices are collapsed — the
    caller should verify the returned row count). GeoParquet (``geo``)
    schema metadata is carried over so the output stays readable with
    ``geopandas.read_parquet``; ``pandas`` metadata is dropped because
    its serialized index no longer describes the filtered rows.

    Args:
        source_path: Parquet file to copy rows from.
        dest_path: Output parquet path (zstd-compressed).
        keep_rows: Positional indices of source rows to keep.
        columns: Column names to project. Every name must exist in the
            source schema; missing names raise ``ValueError`` rather
            than being silently skipped.

    Returns:
        Number of rows written.
    """
    pf = pq.ParquetFile(source_path)
    available = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        raise ValueError(
            f"Columns missing from {source_path}: {missing}"
        )

    n_source = pf.metadata.num_rows
    keep_rows = np.asarray(keep_rows, dtype = np.int64)
    if len(keep_rows) > 0 and (
        keep_rows.min() < 0 or keep_rows.max() >= n_source
    ):
        raise ValueError(
            f"keep_rows out of range for {n_source}-row source "
            f"{source_path}"
        )
    keep_mask = np.zeros(n_source, dtype = bool)
    keep_mask[keep_rows] = True

    source_meta = pf.schema_arrow.metadata or {}
    dest_meta = {
        k: v for k, v in source_meta.items() if k != b"pandas"
    }

    writer: pq.ParquetWriter | None = None
    offset = 0
    n_written = 0
    try:
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg, columns = columns)
            mask = keep_mask[offset:offset + len(table)]
            offset += len(table)
            filtered = table.filter(
                pa.array(mask)
            ).replace_schema_metadata(dest_meta)
            if writer is None:
                writer = pq.ParquetWriter(
                    dest_path,
                    filtered.schema,
                    compression = "zstd",
                )
            if len(filtered) > 0:
                writer.write_table(filtered)
                n_written += len(filtered)
        if writer is None:
            # Zero-row-group source: still emit a valid empty file
            # so downstream reads fail on content, not existence.
            schema = pa.schema(
                [pf.schema_arrow.field(c) for c in columns],
                metadata = dest_meta,
            )
            writer = pq.ParquetWriter(
                dest_path, schema, compression = "zstd",
            )
    finally:
        if writer is not None:
            writer.close()
    return n_written
