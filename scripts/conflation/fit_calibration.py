#!/usr/bin/env python
"""
Fit per-segment existence-confidence calibration curves (the v4 estimator).

Reads the condensed validation handoff exported by openpois-validator
(``scripts/08_export_handoff.py``) plus the conflated parquet's own score
distribution, and writes one monotone lookup table per detection segment for
``apply_calibration.py`` to deploy.

The estimator is a model-assisted difference estimator on the validation's
two-phase design; see ``openpois.conflation.calibration_fit`` and
``~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md``.

Config keys used:
  - versions.calibration, versions.conflation
  - directories.calibration.files.{validation_rows, metadata}
  - directories.conflation.files.conflated
  - conflation.calibration.* (fit knobs)
  - conflation.overture_confidence_weight (matched-segment score collapse)

Prerequisites:
  - openpois-validator: scripts/08_export_handoff.py has run for the round
  - the conflated parquet exists for versions.conflation

Output file(s):
  - ~/data/openpois/conflation/<version>/calibration/{segment}_curve.parquet
  - ~/data/openpois/conflation/<version>/calibration/{segment}_metadata.json
  - ~/data/openpois/conflation/<version>/calibration/fit_report.md

Usage:
    python scripts/conflation/fit_calibration.py [--input-suffix cd] [--test]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from config_versioned import Config

from openpois.conflation import calibration, calibration_fit


def _suffixed_path(base_path: Path, suffix: str | None) -> Path:
    """Insert ``suffix`` before the parquet extension."""
    if not suffix:
        return base_path
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def population_by_segment(conflated_path: Path,
                          chunk_rows: int = 2_000_000) -> dict:
    """Per-segment production source scores, for lookup bin placement.

    The lookup's equal-mass bins must span the *population* score
    distribution, not the validation sample's (which over-represents thin
    strata by design). The raw source columns are collected rather than a
    single score, because the matched segment's index depends on pool
    coefficients that are not known until the fit runs. Read column-scoped and
    streamed: the conflated parquet is ~2.5 GB.

    Shadow-matched rows are excluded: they keep their change-detection value
    and never ride a curve, so they should not influence the bin edges.
    """
    columns = ["source", "osm_conf_mean", "overture_confidence"]
    pf = pq.ParquetFile(str(conflated_path))
    available = set(pf.schema_arrow.names)
    if "shadow_matched" in available:
        columns.append("shadow_matched")
    collected = {segment: [] for segment in calibration.SEGMENTS}
    for batch in pf.iter_batches(batch_size = chunk_rows, columns = columns):
        frame = batch.to_pandas()
        keep = np.ones(len(frame), dtype = bool)
        if "shadow_matched" in frame.columns:
            keep &= ~frame["shadow_matched"].to_numpy(dtype = bool)
        for segment in calibration.SEGMENTS:
            mask = (frame["source"] == segment).to_numpy() & keep
            if not mask.any():
                continue
            collected[segment].append(
                pd.DataFrame(
                    {
                        "osm_score": frame["osm_conf_mean"].to_numpy(
                            dtype = float
                        )[mask],
                        "overture_score": frame[
                            "overture_confidence"
                        ].to_numpy(dtype = float)[mask],
                    }
                )
            )
    return {
        segment: (
            pd.concat(parts, ignore_index = True) if parts
            else pd.DataFrame(columns = ["osm_score", "overture_score"])
        )
        for segment, parts in collected.items()
    }


def write_fit_report(out_dir: Path, results: dict, handoff_metadata: dict,
                     fit_config: calibration_fit.FitConfig) -> Path:
    """Human-readable fit diagnostics beside the curve artifacts."""
    lines = [
        "# Confidence calibration fit report",
        "",
        f"- Estimator: `{calibration_fit.ESTIMATOR_TAG}`",
        f"- Validation round: {handoff_metadata.get('validation_round')}",
        f"- Conflation version: {handoff_metadata.get('conflation_version')}",
        f"- Overture snapshot: {handoff_metadata.get('snapshot_overture')}",
        f"- Validator provenance: `{handoff_metadata.get('validator_git_sha')}`",
        "",
        "## Per-segment fit",
        "",
        "| segment | phase-1 rows | gold | Kish ESS | median band width |",
        "|---|---|---|---|---|",
    ]
    for segment, result in sorted(results.items()):
        lines.append(
            f"| {segment} | {result['n_rows']:,} | {result['n_gold']:,} | "
            f"{result['kish_ess']:.1f} | {result['band_width_median']:.3f} |"
        )

    pooled = {s: r for s, r in results.items() if r.get("pool")}
    if pooled:
        lines += [
            "", "## Fitted source pool (matched segment)", "",
            "Log-odds pool of the two source scores with fitted weights, "
            "replacing the 0.588/0.412 blend and the flat 0.7 downweight. A "
            "coefficient above 1 sharpens that source's evidence; below 1 "
            "damps it for dependence with the other source.", "",
            "| segment | intercept | coef OSM | coef Overture | gold | method |",
            "|---|---|---|---|---|---|",
        ]
        for segment, result in sorted(pooled.items()):
            pool = result["pool"]
            lines.append(
                f"| {segment} | {pool['intercept']:.4f} | "
                f"{pool['coef_osm']:.4f} | {pool['coef_overture']:.4f} | "
                f"{pool['n_gold']:,} | `{pool['method']}` |"
            )

    lines += ["", "## Composite vs Horvitz-Thompson reference", ""]
    for segment, result in sorted(results.items()):
        reference = result["reference_curve"]
        finite = np.isfinite(reference)
        if not finite.any():
            lines.append(f"- {segment}: no reference curve (too little gold)")
            continue
        gap = np.abs(result["curve"][finite] - reference[finite])
        inside = (
            (reference[finite] >= result["summary"]["lower"][finite])
            & (reference[finite] <= result["summary"]["upper"][finite])
        )
        lines.append(
            f"- {segment}: mean |composite - HT| = {gap.mean():.4f}, "
            f"max {gap.max():.4f}; HT curve inside the composite band at "
            f"{100.0 * inside.mean():.1f}% of grid points"
        )

    lines += ["", "## Refined classes (phase-2 inclusion)", "",
              "| segment | class | population | gold | inclusion | HT weight |",
              "|---|---|---|---|---|---|"]
    for segment, result in sorted(results.items()):
        for name, info in sorted(result["inclusion"].items()):
            lines.append(
                f"| {segment} | {name} | {info['n_pop']:,} | "
                f"{info['n_gold']:,} | {info['inclusion']:.4f} | "
                f"{info['weight']:.2f} |"
            )

    lines += ["", "## Constancy check (flat-rate assumption)", "",
              "Gold existence rate in the low vs high half of the score, per "
              "class. A large gap in a *definitive* class argues for the "
              "isotonic treatment instead of a constant.", "",
              "| segment | class | n low | n high | rate low | rate high | gap |",
              "|---|---|---|---|---|---|---|"]
    for segment, result in sorted(results.items()):
        for name, info in sorted(result["constancy"].items()):
            fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
            lines.append(
                f"| {segment} | {name} | {info['n_low']} | {info['n_high']} | "
                f"{fmt(info['rate_low'])} | {fmt(info['rate_high'])} | "
                f"{fmt(info['gap'])} |"
            )

    lines += ["", "## Cross-fit calibration error", "",
              "Design-respecting K-fold: gold folded within refined class, "
              "the whole pipeline refit per fold, held-out gold scored with "
              "design weights. Calibration error is the binned observed-minus-"
              "predicted gap; the debiased column subtracts each bin's own "
              "sampling variance (Kumar, Liang & Ma 2019). The Brier score is "
              "shown for context and includes irreducible outcome noise.", "",
              "| segment | folds | gold | bins | cross-fit Brier | cal. error "
              "(sq, plug-in) | cal. error (sq, debiased) | RMS cal. error |",
              "|---|---|---|---|---|---|---|---|"]
    for segment, result in sorted(results.items()):
        cross = result["cross_fit"]
        if not cross.get("n_folds"):
            lines.append(f"| {segment} | 0 | - | - | - | - | - | - |")
            continue
        debiased = cross["calibration_error_sq_debiased"]
        lines.append(
            f"| {segment} | {cross['n_folds']} | {cross['n_gold']:,} | "
            f"{cross['n_bins']} | {cross['brier_crossfit']:.4f} | "
            f"{cross['calibration_error_sq_plugin']:.5f} | "
            f"{debiased:.5f} | {np.sqrt(debiased):.4f} |"
        )

    lines += ["", "## Fit configuration", "",
              f"- min_cell_gold: {fit_config.min_cell_gold}",
              f"- refine_by_confidence: {fit_config.refine_by_confidence}",
              f"- grid_points: {fit_config.grid_points}",
              f"- output_bins: {fit_config.output_bins}",
              f"- bootstrap_reps: {fit_config.bootstrap_reps}",
              f"- band_alpha: {fit_config.band_alpha}",
              f"- rng_seed: {fit_config.rng_seed}", ""]

    report_path = out_dir / "fit_report.md"
    report_path.write_text("\n".join(lines), encoding = "utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--input-suffix", default = "cd",
        help = ("Suffix of the conflated parquet whose score distribution "
                "sets the lookup bins (default: 'cd')."),
    )
    parser.add_argument(
        "--test", action = "store_true",
        help = "Read/write the *_test.parquet variants.",
    )
    args = parser.parse_args()
    started = time.time()

    config = Config("~/repos/openpois/config.yaml")
    knobs = config.get("conflation", "calibration")

    rows_path = config.get_file_path("calibration", "validation_rows")
    meta_path = config.get_file_path("calibration", "metadata")
    print(f"Validation handoff: {rows_path}")
    validation_rows = pd.read_parquet(rows_path)
    with open(meta_path, encoding = "utf-8") as handle:
        handoff_metadata = json.load(handle)
    print(f"  {len(validation_rows):,} phase-1 rows, "
          f"{int(validation_rows['gold'].sum()):,} gold")

    conflated_base = config.get_file_path("conflation", "conflated")
    if args.test:
        conflated_base = conflated_base.with_name(
            f"{conflated_base.stem}_test{conflated_base.suffix}"
        )
    conflated_path = _suffixed_path(conflated_base, args.input_suffix)
    if not conflated_path.exists():
        raise SystemExit(f"Conflated parquet not found: {conflated_path}")
    print(f"Population scores from: {conflated_path}")
    population = population_by_segment(conflated_path)
    for segment, frame in sorted(population.items()):
        print(f"  {segment}: {len(frame):,} curve-eligible rows")

    fit_config = calibration_fit.FitConfig(
        min_cell_gold = int(knobs["min_cell_gold"]),
        grid_points = int(knobs["grid_points"]),
        output_bins = int(knobs["curve_output_bins"]),
        bootstrap_reps = int(knobs["bootstrap_reps"]),
        band_alpha = float(knobs["band_alpha"]),
        rng_seed = int(knobs["rng_seed"]),
        refine_by_confidence = bool(knobs["refine_by_confidence"]),
        matched_index_mode = str(knobs.get("matched_index_mode", "pool")),
    )

    print("Fitting segment curves...")
    results = calibration_fit.fit_all_segments(
        validation_rows, fit_config, populations = population
    )

    out_dir = config.get_dir_path("conflation") / "calibration"
    out_dir.mkdir(parents = True, exist_ok = True)
    for segment, result in sorted(results.items()):
        metadata = calibration_fit.curve_metadata(
            segment, result, handoff_metadata, fit_config
        )
        calibration_fit.write_curve(out_dir, segment, result["lookup"],
                                    metadata)
        print(f"  {segment}: ESS {result['kish_ess']:.1f}, median band "
              f"{result['band_width_median']:.3f} -> "
              f"{segment}_curve.parquet")

    report_path = write_fit_report(out_dir, results, handoff_metadata,
                                   fit_config)
    print(f"Fit report: {report_path}")
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
