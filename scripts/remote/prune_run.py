#!/usr/bin/env python
"""
Delete a finished run's working files on the remote host, keeping only what next
month's run reads (the carry-forward). Dry run unless --apply.

Runs on the remote pipeline host (`openpois-remote.sh prune`) after the release is
published and verified. Kept, for the versions currently set in config.yaml:
    osm_data/<osm_data>/                      incremental history base_version
    snapshots/overture/<snapshot_overture>/   overture_snapshot.parquet and viz/
                                              (prior month for compare_confidence.py)
    conflation/<conflation>/                  conflated.parquet, calibration/, the
                                              Bayes eval dirs and the small CSV/MD
                                              files (type-affinity input, curve reuse)
    ghost_osm/<ghost_osm>/                    (22 MB; regenerated monthly anyway)
    osm_turnover_model/<model_output>/        the pinned rating model
    boundary/, census_areas/, logs            untouched
Everything else under those roots is removed, including every OSM snapshot version:
next month builds its own, and check_history compares it with its own history.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config_versioned import Config

config = Config("~/repos/openpois/config.yaml")

# Files inside the kept conflation version that are rebuilt or published, not reused.
CONFLATION_DROP = [
    "conflated_baseline.parquet",
    "conflated_cd.parquet",
    "conflated_partitioned",
    "conflated.pmtiles",
    "match_diagnostics.parquet",
    "overture_dedup_dropped.parquet",
]
OVERTURE_KEEP = {"overture_snapshot.parquet", "viz"}


def size_gb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e9
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def targets() -> list[Path]:
    out = []
    # Whole version directories other than the kept one.
    for key in ["osm_data", "snapshot_overture", "conflation", "ghost_osm",
                "model_output"]:
        keep = config.get_dir_path(key)
        if keep.parent.is_dir():
            out += [p for p in keep.parent.iterdir() if p.is_dir() and p != keep]
    # Every OSM snapshot version, including the current one.
    osm_root = config.get_dir_path("snapshot_osm").parent
    if osm_root.is_dir():
        out += [p for p in osm_root.iterdir() if p.is_dir()]
    # Inside the kept versions.
    overture = config.get_dir_path("snapshot_overture")
    if overture.is_dir():
        out += [p for p in overture.iterdir() if p.name not in OVERTURE_KEEP]
    conflation = config.get_dir_path("conflation")
    out += [conflation / name for name in CONFLATION_DROP if (conflation / name).exists()]
    testing = config.get_dir_path("testing")
    if testing.exists():
        out.append(testing)
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    parser.add_argument("--apply", action = "store_true", help = "delete (default: list)")
    args = parser.parse_args()
    # Remote-only: on the laptop the same paths are the local archive.
    if args.apply and "microsoft" in Path("/proc/version").read_text().lower():
        raise SystemExit("refusing --apply under WSL: prune is for the remote host only")

    paths = targets()
    total = 0.0
    for path in paths:
        gb = size_gb(path)
        total += gb
        print(f"{'DELETE' if args.apply else 'would delete'}  {gb:7.2f} GB  {path}")
        if args.apply:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    free = shutil.disk_usage(Path.home()).free / 1e9
    verb = "Freed" if args.apply else "Would free"
    print(f"{verb} {total:.2f} GB; {free:.1f} GB free now.")


if __name__ == "__main__":
    main()
