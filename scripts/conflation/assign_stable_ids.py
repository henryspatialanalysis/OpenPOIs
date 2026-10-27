#!/usr/bin/env python
"""
Replace the provisional ``unified_id`` values in ``conflated_baseline.parquet``
with release-stable ones.

Runs right after ``conflate.py`` inside ``make conflate_baseline``, so every
later stage (change detection, calibration, manual overrides) and the
published data carry the stable ids. Looks up each row's source keys in the
prior release's id registry (``openpois.conflation.stable_ids``), mints ids
for the rest, rewrites the baseline in place, and writes this release's
registry and id change log next to it.

Config keys used:
  - versions.conflation, versions.id_registry (the PRIOR release)
  - directories.conflation.files.{conflated, id_registry, id_changes}
  - directories.id_registry.files.registry   (prior registry)
  - conflation.stable_ids.{guard_threshold, id_prefix}

A missing prior registry is an error unless ``--bootstrap`` is passed (then
every id is minted). ``--test`` runs against ``conflated_baseline_test.parquet``,
mints every id and writes no registry.

Usage:
    python scripts/conflation/assign_stable_ids.py [--test] [--bootstrap]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config_versioned import Config

from openpois.conflation import stable_ids as si

DEDUP_DROPPED_FILE = "overture_dedup_dropped.parquet"


def _suffixed(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}") if suffix else path


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--input-suffix", default = "baseline",
        help = "Suffix of the conflated parquet to rewrite (default: baseline).",
    )
    parser.add_argument("--test", action = "store_true")
    parser.add_argument(
        "--bootstrap", action = "store_true",
        help = "Allow a missing prior registry: mint every id.",
    )
    args = parser.parse_args()
    started = time.time()

    config = Config("~/repos/openpois/config.yaml")
    release = str(config.get("versions", "conflation"))
    settings = config.get("conflation", "stable_ids", fail_if_none = False) or {}
    guard = float(settings.get("guard_threshold", si.GUARD_THRESHOLD))
    prefix = str(settings.get("id_prefix", si.ID_PREFIX))

    conflated = config.get_file_path("conflation", "conflated")
    target = _suffixed(conflated, args.input_suffix)
    if args.test:
        target = target.with_name(f"{target.stem}_test{target.suffix}")
    if not target.exists():
        raise SystemExit(f"Input parquet not found: {target}")

    prior = None
    if not args.test:
        prior = config.get_file_path("id_registry", "registry")
        if not prior.exists():
            if not args.bootstrap:
                raise SystemExit(
                    f"No prior id registry at {prior}. Set versions.id_registry to "
                    "the previous release, or pass --bootstrap to mint every id."
                )
            prior = None
    dedup = conflated.parent / DEDUP_DROPPED_FILE

    print(f"Release:  {release}")
    print(f"Target:   {target} (rewritten in place)")
    print(f"Registry: {prior or 'none (all ids minted)'}")

    con = si.connect()
    n_rows = si.load_release(con, [target])
    si.load_registry(con, prior)
    n_alias = si.load_dedup_aliases(
        con, dedup if (dedup.exists() and not args.test) else None,
    )
    print(f"  {n_rows:,} rows, {n_alias:,} dedup alias pairs")
    stats = si.reconcile(con, release, guard_threshold = guard, prefix = prefix)
    print(si.format_stats(stats))

    si.rewrite_unified_id(con, target, target)
    bad = si.verify_rewrite(con, target)
    if bad:
        raise SystemExit(f"{bad:,} rows carry the wrong id after the rewrite")

    if not args.test:
        registry_out = config.get_file_path("conflation", "id_registry")
        changes_out = config.get_file_path("conflation", "id_changes")
        si.write_outputs(con, registry_out, changes_out)
        stats_out = registry_out.with_name("id_stats.json")
        stats_out.write_text(json.dumps(stats, indent = 2))
        print(f"  Wrote {registry_out.name}, {changes_out.name}, {stats_out.name}")
    con.close()
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
