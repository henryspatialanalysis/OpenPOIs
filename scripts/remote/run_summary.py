#!/usr/bin/env python
"""
Write run_summary.md for a monthly run: per-stage wall time and peak RSS, row
counts against the prior version, and the history / drift-gate verdicts.

Runs on the remote pipeline host after each batch of stages
(`openpois-remote.sh summary`), and is pulled back with the tier-1 results.

Inputs:
    Stage logs written by scripts/remote/stage_runner.sh under
    ~/data/openpois/logs/. A log belongs to this run when its START line carries
    `conflation=<versions.conflation>`. `/usr/bin/time -v` supplies the
    "Elapsed (wall clock) time" and "Maximum resident set size" lines.
    Parquet footers (row counts only; no data is read) for the current version and
    the newest other version on disk, which on the remote is last month's
    carry-forward.

Output:
    conflation/<version>/run_summary.md
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from config_versioned import Config

config = Config("~/repos/openpois/config.yaml")
LOG_DIR = Path("~/data/openpois/logs").expanduser()

# (label, config directory key, file name) for the row-count table.
COUNTED_FILES = [
    ("OSM history versions", "osm_data", "osm_versions.parquet"),
    ("OSM snapshot", "snapshot_osm", "osm_snapshot.parquet"),
    ("OSM snapshot, rated", "snapshot_osm", "osm_snapshot_rated.parquet"),
    ("Overture snapshot", "snapshot_overture", "overture_snapshot.parquet"),
    ("Ghosts", "ghost_osm", "ghosts.parquet"),
    ("Conflated, baseline", "conflation", "conflated_baseline.parquet"),
    ("Conflated, CD", "conflation", "conflated_cd.parquet"),
    ("Conflated, canonical", "conflation", "conflated.parquet"),
]
START_RE = re.compile(r"^=== STAGE (\S+) START (\S+) .*conflation=(\S+) ===")
DONE_RE = re.compile(r"^=== STAGE (\S+) DONE rc=(\d+) (\S+) ===")
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
WALL_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")
VERDICT_RE = re.compile(r"^(PASS|FAIL)\b|timestamp match:|Residential exclusion:")


def markdown_table(df: pd.DataFrame) -> str:
    """Pipe table without the optional tabulate dependency of DataFrame.to_markdown."""
    cells = df.astype(object).where(df.notna(), "").astype(str)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("|" + "---|" * len(df.columns))
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index = False)]
    return "\n".join(lines)


def parse_stage_log(path: Path, version: str) -> dict | None:
    """One row for a stage log of this run, or None if it belongs to another run."""
    lines = path.read_text(errors = "replace").splitlines()
    start = next((START_RE.match(x) for x in lines if START_RE.match(x)), None)
    if start is None or start.group(3) != version:
        return None
    row = {"stage": start.group(1), "started": start.group(2), "log": path.name}
    done = next((DONE_RE.match(x) for x in reversed(lines) if DONE_RE.match(x)), None)
    row["rc"] = int(done.group(2)) if done else None
    rss = [int(m.group(1)) for m in map(RSS_RE.search, lines) if m]
    row["peak_rss_gb"] = round(max(rss) / 1024**2, 2) if rss else None
    wall = [m.group(1) for m in map(WALL_RE.search, lines) if m]
    row["wall"] = wall[-1] if wall else None
    row["verdicts"] = [x.strip() for x in lines if VERDICT_RE.search(x.strip())]
    return row


def prior_version(root: Path, current: str) -> str | None:
    """Newest sibling version directory other than the current one."""
    if not root.is_dir():
        return None
    others = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != current)
    return others[-1] if others else None


def row_count(path: Path) -> int | None:
    return pq.ParquetFile(path).metadata.num_rows if path.exists() else None


def count_table() -> pd.DataFrame:
    rows = []
    for label, key, file_name in COUNTED_FILES:
        current_dir = config.get_dir_path(key)
        prior = prior_version(current_dir.parent, current_dir.name)
        now = row_count(current_dir / file_name)
        before = row_count(current_dir.parent / prior / file_name) if prior else None
        change = (
            f"{100 * (now - before) / before:+.1f}%" if now and before else ""
        )
        rows.append({
            "file": label,
            "version": current_dir.name,
            "rows": f"{now:,}" if now is not None else "missing",
            "prior version": prior or "",
            "prior rows": f"{before:,}" if before is not None else "",
            "change": change,
        })
    return pd.DataFrame(rows)


def drift_gate_table() -> pd.DataFrame | None:
    metrics = (
        config.get_dir_path("snapshot_overture") / "viz"
        / "confidence_comparison_metrics.csv"
    )
    return pd.read_csv(metrics) if metrics.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    parser.add_argument(
        "--version", default = None,
        help = (
            "versions.conflation whose stage logs to summarise (default: the "
            "configured one). Row counts always use the configured versions."
        ),
    )
    args = parser.parse_args()
    version = args.version or str(config.get("versions", "conflation"))

    stages = [
        row for path in sorted(LOG_DIR.glob("*.log"))
        if (row := parse_stage_log(path, version)) is not None
    ]
    stages.sort(key = lambda r: r["started"])

    out = [f"# Run summary: conflation {version}", ""]
    out += ["## Stages", ""]
    if stages:
        table = pd.DataFrame(stages).drop(columns = ["verdicts"])
        out += [markdown_table(table), ""]
        failed = [r["stage"] for r in stages if r["rc"] not in (0, None)]
        running = [r["stage"] for r in stages if r["rc"] is None]
        if failed:
            out += [f"**Failed:** {', '.join(failed)}", ""]
        if running:
            out += [f"**Still running or killed:** {', '.join(running)}", ""]
    else:
        out += ["No stage logs for this version.", ""]

    verdicts = [(r["stage"], v) for r in stages for v in r["verdicts"]]
    if verdicts:
        out += ["## Verdict lines", ""]
        out += [f"- `{stage}`: {line}" for stage, line in verdicts] + [""]

    out += ["## Row counts", "", markdown_table(count_table()), ""]

    drift = drift_gate_table()
    if drift is not None:
        out += ["## Overture confidence drift gate", ""]
        out += [markdown_table(drift), ""]

    disk = shutil.disk_usage(Path.home())
    out += [
        "## Disk",
        "",
        f"Root disk at summary time: {disk.used / 1e9:.1f} GB used, "
        f"{disk.free / 1e9:.1f} GB free.",
        "",
    ]

    target = config.get_dir_path("conflation").parent / version / "run_summary.md"
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_text("\n".join(out))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
