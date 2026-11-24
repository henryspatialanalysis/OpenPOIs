#!/usr/bin/env python
"""
Copy the prior version's calibration curves into this version's ``calibration/``.

Used in a curve-reuse month (``make conflate_reuse PRIOR=<v>``): the Overture
confidence-drift gate passed, the calibration method is unchanged and there is no
new validation round. ``apply_calibration.py --curves-dir`` reads another version's
curves but does not copy them, which would leave this version's ``calibration/``
without the curves that ``plot_calibration.py`` and ``ht_review.py`` need.

Refuses to copy when:
  - the drift gate was not run for this Overture snapshot, or it breaches
    (RMSE > 0.10, |mean bias| > 0.03, or more than 10% of matched ids moving > 0.1)
  - a curve or metadata file is missing in the prior directory
  - a segment's ``accepted`` is false, or its ``validation_round`` differs from
    ``versions.calibration``
  - this version already holds curves (``--overwrite`` replaces them)

It cannot see a calibration *method* change; the method-change override in
.claude/docs/confidence-calibration.md still needs a person to decide.

Config keys used:
  - versions.conflation, versions.snapshot_overture, versions.calibration
  - directories.conflation, directories.snapshot_overture

Prerequisites:
  - scripts/overture/compare_confidence.py has run for versions.snapshot_overture

Output file(s), in conflation/<version>/calibration/:
  - {segment}_curve.parquet, {segment}_metadata.json  (copied verbatim)
  - fit_report_<prior>.md (the original fit's report), PROVENANCE.md

Usage:
    python scripts/conflation/reuse_calibration.py --prior 20261022 [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from config_versioned import Config

SEGMENTS = ("matched", "osm", "overture")
METRICS_FILE = "confidence_comparison_metrics.csv"
MAX_RMSE = 0.10
MAX_ABS_BIAS = 0.03
MAX_SHARE_MOVED = 0.10


def read_drift_metrics(path: Path) -> dict[str, float]:
    """Metric name -> value from the compare_confidence.py metrics CSV."""
    table = pd.read_csv(path)
    return dict(zip(table["metric"], table["value"].astype(float)))


def drift_breaches(metrics: dict[str, float]) -> list[str]:
    """The drift-gate criteria that fail; empty means the gate passes."""
    checks = [
        ("rmse", metrics["rmse"], MAX_RMSE),
        ("|bias_mean_diff|", abs(metrics["bias_mean_diff"]), MAX_ABS_BIAS),
        ("share_absdiff_gt_0.1", metrics["share_absdiff_gt_0.1"], MAX_SHARE_MOVED),
    ]
    # Written as "not <=" so a NaN metric fails.
    return [
        f"{name} = {value:.4f} > {limit}"
        for name, value, limit in checks if not value <= limit
    ]


def check_metadata(prior_dir: Path, calibration_round: str) -> list[str]:
    """Problems with the prior directory's curve files and metadata."""
    problems = []
    for segment in SEGMENTS:
        curve = prior_dir / f"{segment}_curve.parquet"
        meta_path = prior_dir / f"{segment}_metadata.json"
        missing = [p.name for p in (curve, meta_path) if not p.exists()]
        if missing:
            problems.append(f"{segment}: missing {', '.join(missing)}")
            continue
        meta = json.loads(meta_path.read_text(encoding = "utf-8"))
        if meta.get("accepted") is not True:
            problems.append(f"{segment}: accepted = {meta.get('accepted')}")
        if str(meta.get("validation_round")) != calibration_round:
            problems.append(
                f"{segment}: validation_round {meta.get('validation_round')} != "
                f"versions.calibration {calibration_round}"
            )
    return problems


def copy_curves(prior_dir: Path, out_dir: Path, prior_version: str) -> list[str]:
    """Copy the curves, metadata and fit report; return the copied file names."""
    out_dir.mkdir(parents = True, exist_ok = True)
    copied = []
    for segment in SEGMENTS:
        for name in (f"{segment}_curve.parquet", f"{segment}_metadata.json"):
            shutil.copy2(prior_dir / name, out_dir / name)
            copied.append(name)
    # A reuse month carries fit_report_<fit version>.md; a fit month has fit_report.md.
    for report in sorted(prior_dir.glob("fit_report*.md")):
        target = (
            f"fit_report_{prior_version}.md" if report.name == "fit_report.md"
            else report.name
        )
        shutil.copy2(report, out_dir / target)
        copied.append(target)
    return copied


def provenance_text(
    prior_version: str, meta: dict, metrics: dict[str, float],
) -> str:
    """The PROVENANCE.md body."""
    pooled = [r for r in meta["rounds"] if str(r) != str(meta["validation_round"])]
    pooled_note = f", pooled with {', '.join(pooled)}" if pooled else ""
    current = str(int(metrics["current_version"]))
    prior_snapshot = str(int(metrics["prior_version"]))
    return (
        "# Calibration provenance\n\n"
        f"Curves copied verbatim from conflation/{prior_version}/calibration "
        f"(method {meta['method']}, validation round {meta['validation_round']}"
        f"{pooled_note}; fit directory {meta['eval_dir']}).\n"
        f"Reused because the Overture confidence-drift gate passed on snapshot "
        f"{current} against {prior_snapshot} (RMSE {metrics['rmse']:.4f}, bias "
        f"{metrics['bias_mean_diff']:+.4f}, share |diff| > 0.1 = "
        f"{metrics['share_absdiff_gt_0.1']:.2%}) and the calibration method is "
        "unchanged.\n"
        "fit_report_*.md is the original fit's report.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--prior", required = True,
        help = "Conflation version whose calibration/ curves are reused.",
    )
    parser.add_argument(
        "--overwrite", action = "store_true",
        help = "Replace curves already in this version's calibration/.",
    )
    args = parser.parse_args()

    config = Config("~/repos/openpois/config.yaml")
    conflation_dir = config.get_dir_path("conflation")
    version = str(config.get("versions", "conflation"))
    overture_version = str(config.get("versions", "snapshot_overture"))
    prior_dir = conflation_dir.parent / args.prior / "calibration"
    out_dir = conflation_dir / "calibration"
    if args.prior == version:
        raise SystemExit("--prior is this version; pass an earlier one")

    metrics_path = config.get_dir_path("snapshot_overture") / "viz" / METRICS_FILE
    if not metrics_path.exists():
        raise SystemExit(
            f"No drift gate result at {metrics_path}; run "
            "scripts/overture/compare_confidence.py first"
        )
    metrics = read_drift_metrics(metrics_path)
    if str(int(metrics["current_version"])) != overture_version:
        raise SystemExit(
            f"{metrics_path} is for snapshot {int(metrics['current_version'])}, "
            f"not {overture_version}; re-run compare_confidence.py"
        )
    breaches = drift_breaches(metrics)
    if breaches:
        raise SystemExit(
            "Drift gate BREACH: " + "; ".join(breaches) + ". Curves must not be "
            "reused; see .claude/docs/confidence-calibration.md"
        )
    problems = check_metadata(prior_dir, str(config.get("versions", "calibration")))
    if problems:
        raise SystemExit(f"Cannot reuse {prior_dir}:\n  " + "\n  ".join(problems))
    existing = [s for s in SEGMENTS if (out_dir / f"{s}_curve.parquet").exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"{out_dir} already holds curves for {', '.join(existing)}; pass "
            "--overwrite to replace them"
        )

    copied = copy_curves(prior_dir, out_dir, args.prior)
    meta = json.loads(
        (out_dir / "matched_metadata.json").read_text(encoding = "utf-8")
    )
    (out_dir / "PROVENANCE.md").write_text(
        provenance_text(args.prior, meta, metrics), encoding = "utf-8",
    )
    print(f"Drift gate PASS: rmse {metrics['rmse']:.4f}, "
          f"bias {metrics['bias_mean_diff']:+.4f}, "
          f"share > 0.1 {metrics['share_absdiff_gt_0.1']:.2%}")
    print(f"Copied {len(copied)} files from {prior_dir} to {out_dir}")


if __name__ == "__main__":
    main()
