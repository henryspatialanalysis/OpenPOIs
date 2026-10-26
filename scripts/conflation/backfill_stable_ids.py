#!/usr/bin/env python
"""
Backfill release-stable ``unified_id`` values over every published release.

Replays the id registry (``openpois.conflation.stable_ids``) forward from the
first published release, one release at a time, and writes a ``-v1`` copy of
each release's conflated parquet with only ``unified_id`` replaced (and a
double ``osm_id`` cast to int64). Reads the published ``<release>-v0``
partitioned parquet anonymously from Source Cooperative, so it needs no local
pipeline outputs. Uploading is a separate step (``publish_backfill_v1.py``).

For each release ``R`` (``YYYY-MM-DD``), under ``--work-dir``/``R``/::

    v0/conflated-parquet/...           download of R-v0 (resumable)
    v1/conflated-parquet/...           rewritten copy, same Hive layout
    v1/id_registry.parquet             registry after R
    v1/id_changes.parquet              changes vs the previous release
    v1/id_migration_v0_v1.parquet      v0 id -> v1 id, with the source keys
    reconcile_stats.json

A release is skipped when its ``reconcile_stats.json`` exists (pass
``--force`` to redo it), so the chain can be resumed after an interruption.
Before moving on, each release is checked: row count per file equal to v0,
every non-id column unchanged (per-column hash), ids unique, and every row's
id equal to the id assigned to its source keys.

An Overture dedup audit (``conflation/<YYYYMMDD>/overture_dedup_dropped.parquet``)
is used for alias edges when it exists on this host.

Usage:
    python -u scripts/conflation/backfill_stable_ids.py
    python -u scripts/conflation/backfill_stable_ids.py --releases 2026-04-23 2026-05-21
    python -u scripts/conflation/backfill_stable_ids.py --no-rewrite   # ids + stats only
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq

from openpois.conflation import stable_ids as si
from openpois.io.source_coop import (
    LEGACY_READ_BUCKET,
    download_prefix,
    make_anonymous_read_client,
)

RELEASES = (
    "2026-04-23", "2026-05-21", "2026-06-27", "2026-07-30",
    "2026-09-02", "2026-10-01", "2026-10-22",
)
REPO_PREFIX = "henryspatialanalysis/openpois"
DATA_ROOT = Path("~/data/openpois").expanduser()


def _yyyymmdd(release: str) -> str:
    return release.replace("-", "")


def _column_hashes(con, paths: list[Path], exclude: tuple[str, ...]) -> dict:
    """Per-column order-insensitive hash of ``paths``, skipping ``exclude``."""
    files = si._sql_list(paths)
    rel = con.sql(
        f"SELECT * FROM read_parquet({files}, hive_partitioning = false) LIMIT 0"
    )
    cols = [c for c in rel.columns if c not in exclude]
    exprs = ", ".join(f'bit_xor(hash("{c}")) AS "{c}"' for c in cols)
    row = con.execute(
        f"SELECT {exprs} FROM read_parquet({files}, hive_partitioning = false)"
    ).fetchone()
    return dict(zip(cols, row))


def check_release(con, v0_files: list[Path], v1_files: list[Path]) -> None:
    """Raise unless v1 differs from v0 only in ``unified_id`` / ``osm_id`` type."""
    for a, b in zip(v0_files, v1_files):
        na = pq.ParquetFile(a).metadata.num_rows
        nb = pq.ParquetFile(b).metadata.num_rows
        if na != nb:
            raise AssertionError(f"{b}: {nb:,} rows, v0 has {na:,}")
    exclude = ("unified_id", "osm_id")
    if _column_hashes(con, v0_files, exclude) != _column_hashes(con, v1_files, exclude):
        raise AssertionError("a non-id column changed between v0 and v1")
    osm_sums = [
        con.execute(
            f"SELECT sum(CAST(osm_id AS HUGEINT)), count(osm_id) "
            f"FROM read_parquet({si._sql_list(files)}, hive_partitioning = false)"
        ).fetchone()
        for files in (v0_files, v1_files)
    ]
    if osm_sums[0] != osm_sums[1]:
        raise AssertionError(f"osm_id values changed: {osm_sums}")
    n, n_ids = con.execute(
        f"SELECT count(*), count(DISTINCT unified_id) "
        f"FROM read_parquet({si._sql_list(v1_files)}, hive_partitioning = false)"
    ).fetchone()
    if n != n_ids:
        raise AssertionError(f"v1 ids not unique: {n:,} rows, {n_ids:,} ids")


def run_release(
    release: str,
    prev_registry: Path | None,
    work_dir: Path,
    rewrite: bool,
    memory_limit: str,
    drop_v0: bool = False,
) -> Path:
    """Reconcile (and rewrite) one release; return its registry path."""
    rel = _yyyymmdd(release)
    root = work_dir / release
    v0_dir = root / "v0" / "conflated-parquet"
    v1_dir = root / "v1" / "conflated-parquet"
    registry_path = root / "v1" / "id_registry.parquet"
    started = time.time()

    print(f"== {release}: downloading {release}-v0/conflated-parquet", flush = True)
    client = make_anonymous_read_client()
    v0_files = sorted(download_prefix(
        client, LEGACY_READ_BUCKET,
        f"{REPO_PREFIX}/{release}-v0/conflated-parquet/", v0_dir,
    ))
    v0_files = [p for p in v0_files if p.suffix == ".parquet"]
    print(f"  {len(v0_files)} files ({time.time() - started:.0f}s)", flush = True)

    con = si.connect(memory_limit)
    n_rows = si.load_release(con, v0_files)
    si.load_registry(con, prev_registry)
    dedup = DATA_ROOT / "conflation" / rel / "overture_dedup_dropped.parquet"
    n_alias = si.load_dedup_aliases(con, dedup if dedup.exists() else None)
    print(f"  {n_rows:,} rows; prior registry: {prev_registry or 'none'}; "
          f"{n_alias:,} dedup alias pairs", flush = True)

    stats = si.reconcile(con, rel)
    print(si.format_stats(stats), flush = True)
    si.write_outputs(con, registry_path, root / "v1" / "id_changes.parquet")
    si.load_v0_ids(con, v0_files)
    pq.write_table(
        si.migration_table(con), root / "v1" / "id_migration_v0_v1.parquet",
        compression = "zstd",
    )

    if rewrite:
        v1_files = []
        for path in v0_files:
            out = v1_dir / path.relative_to(v0_dir)
            si.rewrite_unified_id(con, path, out, cast_osm_id = True)
            bad = si.verify_rewrite(con, out, source_path = path)
            if bad:
                raise AssertionError(f"{out}: {bad:,} rows carry the wrong id")
            v1_files.append(out)
        check_release(con, v0_files, v1_files)
        print(f"  rewrote and checked {len(v1_files)} files", flush = True)

    stats["elapsed_s"] = round(time.time() - started, 1)
    stats["rewritten"] = rewrite
    (root / "reconcile_stats.json").write_text(json.dumps(stats, indent = 2))
    con.close()
    if drop_v0 and rewrite:
        shutil.rmtree(root / "v0")
    print(f"  done in {stats['elapsed_s']:.0f}s", flush = True)
    return registry_path


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--releases", nargs = "+", default = list(RELEASES))
    parser.add_argument(
        "--work-dir", type = Path, default = DATA_ROOT / "conflation_backfill",
    )
    parser.add_argument(
        "--no-rewrite", action = "store_true",
        help = "Reconcile and write the id files only; skip the parquet rewrite.",
    )
    parser.add_argument("--force", action = "store_true")
    parser.add_argument(
        "--drop-v0", action = "store_true",
        help = "Delete each release's v0 download once its v1 passes the checks.",
    )
    parser.add_argument("--memory-limit", default = "24GB")
    args = parser.parse_args()

    releases = sorted(args.releases)
    if releases[0] != RELEASES[0]:
        print(f"Starting at {releases[0]}: the previous release's registry must "
              "already be in the work dir.")
    prev_registry = None
    first = RELEASES.index(releases[0])
    if first > 0:
        prev_registry = args.work_dir / RELEASES[first - 1] / "v1" / "id_registry.parquet"
        if not prev_registry.exists():
            raise SystemExit(f"Missing prior registry {prev_registry}")

    for release in releases:
        stats_path = args.work_dir / release / "reconcile_stats.json"
        registry = args.work_dir / release / "v1" / "id_registry.parquet"
        done = stats_path.exists() and json.loads(stats_path.read_text()).get(
            "rewritten", False) >= (not args.no_rewrite)
        if done and not args.force:
            print(f"== {release}: already done, skipping", flush = True)
        else:
            registry = run_release(
                release, prev_registry, args.work_dir,
                rewrite = not args.no_rewrite, memory_limit = args.memory_limit,
                drop_v0 = args.drop_v0,
            )
        prev_registry = registry

    last = releases[-1]
    if last == RELEASES[-1]:
        target = DATA_ROOT / "conflation" / _yyyymmdd(last) / "id_registry.parquet"
        target.parent.mkdir(parents = True, exist_ok = True)
        shutil.copy2(prev_registry, target)
        print(f"Copied the {last} registry to {target} (the next run's prior).")
    print("BACKFILL DONE", flush = True)


if __name__ == "__main__":
    main()
