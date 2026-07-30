#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
Render the top-level and per-version README.md files that ship to Source Coop.

The per-version README is regenerated on every upload and summarises the OSM
snapshot date, Overture release, fitted-model version, and row counts pulled
straight from the partitioned parquet outputs.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
from pathlib import Path

import pyarrow.dataset as pads

TEMPLATES_DIR = Path(__file__).parent / "templates"
REPO_ROOT = Path(__file__).resolve().parents[3]


def build_top_readme() -> str:
    """Return the static top-level README text."""
    return (TEMPLATES_DIR / "top_readme.md").read_text()


def load_license() -> str:
    """Return the ODbL license text that ships with the published data."""
    return (TEMPLATES_DIR / "LICENSE").read_text()


def _count_rows(partitioned_dir: Path) -> int:
    """Cheap row count over a partitioned parquet tree; 0 if dir missing."""
    if not partitioned_dir.exists():
        return 0
    dataset = pads.dataset(
        str(partitioned_dir),
        format = "parquet",
        partitioning = "hive",
    )
    return dataset.count_rows()


def _meta(config, key: str, default = None):
    """Read ``publish.version_metadata.<key>``; return ``default`` if absent."""
    return config.get(
        "publish", "version_metadata", key, fail_if_none = False
    ) or default


def _yyyymmdd_to_iso(value: str) -> str:
    """Turn a ``YYYYMMDD`` version string into ``YYYY-MM-DD``; pass anything
    else through unchanged so we don't mangle already-formatted dates."""
    value = str(value)
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _resolve_osm_snapshot_date(config) -> str:
    """YYYY-MM-DD date of the OSM download used for this release."""
    override = _meta(config, "osm_snapshot_date")
    if override:
        return str(override)
    return _yyyymmdd_to_iso(config.get("versions", "snapshot_osm"))


def _resolve_overture_release(config) -> str:
    """Overture Maps release string (e.g. ``2026-04-15.0``).

    Preference order:
      1. ``publish.version_metadata.overture_release`` (manual pin).
      2. ``download.overture.release_date`` if pinned (non-null).
      3. The release date recorded in the Overture snapshot's saved
         ``.parts/<release>/`` marker, if present.
      4. Fall back to ``versions.snapshot_overture`` with a note.
    """
    override = _meta(config, "overture_release")
    if override:
        return str(override)

    pinned = config.get(
        "download", "overture", "release_date", fail_if_none = False
    )
    if pinned:
        return str(pinned)

    overture_dir = config.get_dir_path("snapshot_overture")
    parts_dir = overture_dir / ".parts"
    if parts_dir.is_dir():
        subdirs = [p.name for p in parts_dir.iterdir() if p.is_dir()]
        if subdirs:
            return sorted(subdirs)[-1]

    return (
        f"{config.get('versions', 'snapshot_overture')} (auto-detected at "
        "download time)"
    )


def _resolve_model_commit(config) -> str:
    """Short git SHA identifying the turnover-model code for this release.

    Preference: explicit override → current HEAD of the openpois repo.
    Falls back to the literal string ``unknown`` if ``git`` is not available.
    """
    override = _meta(config, "model_commit")
    if override:
        return str(override)
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output = True, text = True, check = True, timeout = 5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _format_int(n: int) -> str:
    return f"{n:,}" if n else "_(not present)_"


def _config_hash(config_path: Path) -> str:
    """Short SHA-256 of the effective config.yaml."""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]


def build_version_readme(
    config,
    version_folder: str,
    config_path: Path | None = None,
) -> str:
    """Render the per-version README for an upload round.

    Pulls metadata straight from the live ``Config`` object, and row counts
    from the partitioned parquet trees resolved via ``get_file_path``.
    ``config_path`` is used only to compute the short-hash fingerprint; if
    omitted we try the default repo location.
    """
    osm_snapshot_date = _resolve_osm_snapshot_date(config)
    overture_release = _resolve_overture_release(config)
    model_commit = _resolve_model_commit(config)
    calibration_round = config.get("versions", "calibration",
                                   fail_if_none = False) or "not calibrated"

    osm_partitioned = config.get_file_path("snapshot_osm", "partitioned")
    conflated_partitioned = config.get_file_path("conflation", "partitioned")
    osm_rows = _count_rows(osm_partitioned)
    conflated_rows = _count_rows(conflated_partitioned)

    if config_path is None:
        config_path = Path("~/repos/openpois/config.yaml").expanduser()
    config_path = Path(config_path).expanduser()
    config_hash = _config_hash(config_path) if config_path.exists() else "—"

    template = (TEMPLATES_DIR / "version_readme.md.tmpl").read_text()
    return template.format(
        version_folder = version_folder,
        osm_snapshot_date = osm_snapshot_date,
        overture_release = overture_release,
        model_commit = model_commit,
        calibration_round = calibration_round,
        conflated_row_count = _format_int(conflated_rows),
        osm_row_count = _format_int(osm_rows),
        generation_date = dt.date.today().isoformat(),
        config_hash = config_hash,
    )


if __name__ == "__main__":
    # Quick sanity check: render both documents to stdout.
    from config_versioned import Config

    cfg = Config("~/repos/openpois/config.yaml")
    version = cfg.get("versions", "source_coop")
    print("=" * 72)
    print(build_top_readme())
    print("=" * 72)
    print(build_version_readme(cfg, version))
