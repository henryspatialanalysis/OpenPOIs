#!/usr/bin/env python
"""
Drop unnamed private / no-access POIs from the rated OSM snapshot.

OSM POIs with no ``name`` tag AND ``access`` in (``private``, ``no``) are
anonymous non-public features — overwhelmingly residential and HOA swimming
pools — that add noise without value. See the "Exclusion" section of
docs/data-sources.md for the rationale and for why ``access=restricted`` is
deliberately kept.

TRANSITIONAL. As of 2026-07-26 the same predicate runs at snapshot build time
(``download.osm.excluded_access`` → ``_drop_unnamed_private_rows`` in
``openpois.io.osm_snapshot``), so from the 2026-08 pull onward the snapshot
arrives already filtered and this script becomes a no-op reporting 0 dropped.
It is still needed for the 2026-07 snapshot, which was built before that
change: ``make rate`` regenerates the rated file from the *unfiltered*
snapshot every time, so the exclusion has to be re-applied after each rating
pass. ``tests/test_access_exclusion.py`` pins the two implementations to the
same row selection.

Always pass ``--expect-kept`` when the target count is known. It is what
caught a pyarrow null-propagation bug that over-dropped 2.44M rows.

The pre-exclusion file is kept alongside the output as
``osm_snapshot_rated.prefilter.parquet`` unless ``--no-archive`` is passed.
Row groups stream through pyarrow, so peak memory stays well under the file
size.

Config keys used (config.yaml):
    snapshot_osm.rated_snapshot — input and output (rewritten in place)

Usage:
    python scripts/osm_snapshot/apply_access_exclusion.py
    python scripts/osm_snapshot/apply_access_exclusion.py --expect-kept 5015126
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from config_versioned import Config

config = Config("~/repos/openpois/config.yaml")
RATED_PATH = Path(config.get_file_path("snapshot_osm", "rated_snapshot"))

EXCLUDED_ACCESS = ("private", "no")


def _keep_mask(batch: pa.RecordBatch) -> pa.Array:
    """Rows to KEEP: named, or access outside the excluded set.

    The mask is guaranteed null-free. That matters more than it looks:
    ``Table.filter`` drops rows whose mask value is null, and
    ``pc.or_`` propagates nulls instead of applying Kleene logic — so
    ``or_(is_null(name), equal(name, ""))`` evaluates to *null* for a
    null name (true OR null = null), which silently dropped every
    unnamed POI regardless of its access value. Use the ``_kleene``
    variants and fill what is left.
    """
    name = batch.column("name")
    access = batch.column("access")
    unnamed = pc.fill_null(
        pc.or_kleene(pc.is_null(name), pc.equal(name, "")), False,
    )
    blocked = pc.fill_null(
        pc.is_in(access, value_set = pa.array(EXCLUDED_ACCESS)), False,
    )
    keep = pc.invert(pc.and_(unnamed, blocked))
    if keep.null_count:
        raise ValueError(
            f"keep mask has {keep.null_count} nulls; filter() would drop "
            "those rows silently."
        )
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--expect-kept", type = int, default = None,
        help = "fail if the kept row count differs (guards a silent "
        "predicate change between runs)",
    )
    parser.add_argument(
        "--no-archive", action = "store_true",
        help = "delete the pre-exclusion file instead of keeping it",
    )
    args = parser.parse_args()

    if not RATED_PATH.exists():
        raise SystemExit(f"No rated snapshot at {RATED_PATH}")

    prefilter_path = RATED_PATH.with_suffix(".prefilter.parquet")
    print(f"Reading  {RATED_PATH}")
    RATED_PATH.rename(prefilter_path)

    source = pq.ParquetFile(prefilter_path)
    schema = source.schema_arrow
    for required in ("name", "access"):
        if required not in schema.names:
            prefilter_path.rename(RATED_PATH)
            raise SystemExit(f"Column '{required}' missing — nothing to do.")

    n_in = source.metadata.num_rows
    n_out = 0
    writer = pq.ParquetWriter(
        RATED_PATH, schema, compression = "snappy",
    )
    try:
        for batch in source.iter_batches(batch_size = 250_000):
            keep = _keep_mask(batch)
            kept = pa.Table.from_batches([batch], schema = schema).filter(keep)
            n_out += kept.num_rows
            writer.write_table(kept)
    except Exception:
        writer.close()
        RATED_PATH.unlink(missing_ok = True)
        prefilter_path.rename(RATED_PATH)
        raise
    finally:
        if not writer.is_open:
            pass
        else:
            writer.close()

    dropped = n_in - n_out
    print(f"Wrote    {RATED_PATH}")
    print(
        f"  {n_in:,} rated -> {n_out:,} kept "
        f"({dropped:,} dropped, {100 * dropped / max(n_in, 1):.1f}%)"
    )

    if args.no_archive:
        prefilter_path.unlink()
        print(f"  removed {prefilter_path.name}")
    else:
        print(f"  pre-exclusion copy kept at {prefilter_path.name}")

    if args.expect_kept is not None and n_out != args.expect_kept:
        print(
            f"\nEXPECTED {args.expect_kept:,} kept rows, got {n_out:,}.",
            file = sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
