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

Prerequisites:
  - openpois-validator: scripts/08_export_handoff.py has run for the round
  - the conflated parquet exists for versions.conflation

Output file(s):
  - ~/data/openpois/conflation/<version>/calibration/{segment}_curve.parquet
  - ~/data/openpois/conflation/<version>/calibration/{segment}_metadata.json
  - ~/data/openpois/conflation/<version>/calibration/fit_report.md
  - ~/data/openpois/conflation/<version>/calibration/ht_review_<round>.pdf
    (+ ht_review_<round>_bins.csv): the design-weighted check of the curves
    just written (``ht_review.py``); it never fails the run

Usage:
    python scripts/conflation/fit_calibration.py [--input-suffix cd] [--test]
        [--out-dir DIR [--allow-deployed]] [--matched-index-mode MODE]
        [--skip-monotonicity] [--skip-ht-review]

``--out-dir`` writes the curves and report elsewhere (evaluation runs); it
refuses the deployed ``conflation/<version>/calibration`` directory unless
``--allow-deployed`` is passed. The default (no ``--out-dir``) is the
production path and writes there, as ``make fit_calibration`` expects.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from config_versioned import Config

from openpois.conflation import calibration, calibration_fit

# Gold rows a monotonicity-table bin needs before its reversal z is reported;
# thinner bins are merged into a neighbour rather than skipped.
MONOTONICITY_MIN_GOLD = 5


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


def monotonicity_tables(validation_rows: pd.DataFrame,
                        fit_config: calibration_fit.FitConfig) -> dict:
    """Per-axis atom-aware monotonicity tables for every segment.

    Bins with fewer than ``MONOTONICITY_MIN_GOLD`` gold rows are merged into a
    neighbour first, so every adjacent pair gets a reversal z.
    """
    usable = validation_rows[
        validation_rows["llm_verdict"].isin(calibration_fit.VERDICTS)
        & validation_rows["stratum"].isin(calibration_fit.SEGMENTS)
    ]
    axes = {"matched": (("osm_score", 10), ("overture_score", 5)),
            "osm": (("osm_score", 10),),
            "overture": (("overture_score", 10),)}
    out = {}
    for segment, specs in axes.items():
        rows = usable[usable["segment"] == segment].reset_index(drop = True)
        if len(rows) < 50:
            continue
        for column, n_bins in specs:
            edges = calibration_fit.merge_thin_bins(
                calibration_fit.atom_aware_edges(rows[column], n_bins),
                rows[column], rows["gold"], MONOTONICITY_MIN_GOLD,
            )
            out[(segment, column)] = calibration_fit.axis_monotonicity_table(
                rows, column, edges, fit_config, segment = segment,
                min_gold = MONOTONICITY_MIN_GOLD,
            )
    return out


def ht_review_lines(out_dir: Path, validation_rows: pd.DataFrame,
                    handoff_metadata: dict,
                    fit_config: calibration_fit.FitConfig) -> list:
    """Run the design-weighted check on the curves just written.

    Writes ``ht_review_<round>.pdf`` beside them and returns the fit report's
    section. The check is a review aid and never fails the run: any error is
    printed and recorded in the section instead.
    """
    try:
        # Sibling script, imported here so that importers of this module (the
        # Bayesian scripts use population_by_segment) do not load matplotlib.
        import ht_review

        result = ht_review.run_review(out_dir, validation_rows,
                                      handoff_metadata,
                                      fit_config = fit_config)
    except Exception as error:  # the check must never fail the run
        traceback.print_exc()
        return ["## Design-weighted (Horvitz-Thompson) check", "",
                f"The check raised `{error!r}` and was skipped. It never "
                f"fails the run; rerun it with "
                f"`scripts/conflation/ht_review.py --curves-dir {out_dir}`.",
                ""]
    summary = result["check"]["summary"]
    for row in summary.itertuples():
        print(f"  HT check {row.segment}/{row.view}: {row.beyond_1sd} of "
              f"{row.tested} bins beyond 1 SD, {row.beyond_2sd} beyond 2 SD")
    print(f"HT review: {result['pdf']}")
    return result["lines"]


def write_fit_report(out_dir: Path, results: dict, handoff_metadata: dict,
                     fit_config: calibration_fit.FitConfig,
                     monotonicity: dict = None, ht_lines: list = None) -> Path:
    """Human-readable fit diagnostics beside the curve artifacts.

    ``ht_lines`` is the design-weighted check's section
    (:func:`ht_review_lines`), placed after the HT reference comparison.
    """
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
        "Median band width is the published band's, over the lookup bins.",
        "",
        "| segment | phase-1 rows | gold | Kish ESS | median band width |",
        "|---|---|---|---|---|",
    ]
    for segment, result in sorted(results.items()):
        lines.append(
            f"| {segment} | {result['n_rows']:,} | {result['n_gold']:,} | "
            f"{result['kish_ess']:.1f} | {result['band_width_median']:.3f} |"
        )

    for segment, result in sorted(results.items()):
        index = result.get("index") or {}
        form = index.get("form")
        if form == "additive":
            lines += [
                "", f"## Fitted additive index ({segment} segment)", "",
                f"`a + h_osm(s_osm) + h_ov(s_ov)`, each h nondecreasing; "
                f"intercept {index['intercept']:.4f}; PAV blocks "
                f"{index['n_blocks_osm']} (OSM) / {index['n_blocks_overture']} "
                f"(Overture); {index['n_clipped']} gold rows on the ±logit "
                f"clip; {index['n_outer_iterations']} local-scoring "
                f"iterations, converged = {index['converged']}.",
            ]
        elif form == "interaction":
            lines += [
                "", f"## Fitted interaction index ({segment} segment)", "",
                "`a0 + a1 x + a2 y + a3 x y` on logits rescaled to [0, 1] over "
                "the clip range. a3 < 0 is substitutive (either source "
                "suffices).", "",
                "| a0 | a1 | a2 | a3 | active constraints |",
                "|---|---|---|---|---|",
                f"| {index['a0']:.4f} | {index['a1']:.4f} | {index['a2']:.4f} "
                f"| {index['a3']:.4f} | "
                f"{', '.join(index['constraints_active']) or 'none'} |",
            ]
        surface = result.get("surface")
        if surface is not None:
            shape = surface["shape"]
            lines += [
                "", f"## Doubly-monotone surface ({segment} segment)", "",
                f"{shape[0]} OSM × {shape[1]} Overture cells; published band: "
                f"{surface['band_method']}; Wald mixture over "
                f"{surface['bands']['wald']['n_patterns']} distinct binding "
                f"sets; {surface['n_bootstrap_dropped']} bootstrap replicates "
                f"dropped. Rows OSM low → high, columns Overture low → high.",
                "",
                f"- OSM edges: {np.round(surface['edges']['osm'], 4).tolist()}",
                f"- Overture edges: "
                f"{np.round(surface['edges']['overture'], 6).tolist()}",
                "",
            ]
            for title, matrix, fmt in (
                ("Projected estimate", surface["theta"], "{:.3f}"),
                ("Unconstrained estimate", surface["unconstrained"], "{:.3f}"),
                ("HT reference (projected)", surface["reference_surface"],
                 "{:.3f}"),
                ("Gold per cell", surface["gold_counts"], "{:.0f}"),
                ("Percentile band width",
                 surface["bands"]["percentile"]["upper"]
                 - surface["bands"]["percentile"]["lower"], "{:.3f}"),
                ("Wald-mixture band width",
                 surface["bands"]["wald"]["upper"]
                 - surface["bands"]["wald"]["lower"], "{:.3f}"),
            ):
                lines += [f"**{title}**", "",
                          "| OSM bin | " + " | ".join(
                              f"ov{j + 1}" for j in range(shape[1])) + " |",
                          "|---" * (shape[1] + 1) + "|"]
                for i in range(shape[0]):
                    lines.append(f"| osm{i + 1} | " + " | ".join(
                        fmt.format(v) for v in np.asarray(matrix)[i]) + " |")
                lines.append("")
            lines.append(
                f"The bottom cell (osm1, ov1; {surface['corner_cell_flag']['gold']}"
                f" gold) has no cell below it in the product order, so its "
                f"band cannot borrow strength downward (Liao, Meyer & Xu 2024 "
                f"p. 5)."
            )

    pooled = {s: r for s, r in results.items() if r.get("pool")}
    if pooled:
        lines += [
            "", "## Fitted source pool (matched segment)", "",
            "Log-odds pool of the two source scores with fitted weights "
            "(slopes constrained >= pool_min_coef), replacing the "
            "0.588/0.412 blend and the flat 0.7 downweight. A coefficient "
            "below 1 mixes damping for dependence with the other source and "
            "that raw score's own miscalibration; the two are not "
            "separable here.", "",
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
        if reference is None:
            surface = result["surface"]
            ref = surface["reference_surface"]
            band = surface["bands"][surface["band_method"]]
            if not np.isfinite(ref).all():
                lines.append(f"- {segment}: no HT reference surface (a cell "
                             f"has no gold)")
                continue
            gap = np.abs(surface["theta"] - ref)
            inside = (ref >= band["lower"]) & (ref <= band["upper"])
            lines.append(
                f"- {segment} (surface): mean |composite - HT| = "
                f"{gap.mean():.4f}, max {gap.max():.4f}; HT inside the band "
                f"in {100.0 * inside.mean():.1f}% of cells"
            )
            continue
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

    if ht_lines:
        lines += [""] + list(ht_lines)

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

    if monotonicity:
        lines += [
            "", "## Monotonicity by axis (standing per-round check)", "",
            "Atom-aware bins (each Overture atom its own bin). DE = the "
            "difference estimator with the working model; HT = gold-only "
            "Hajek; SEs from a 200-replicate two-phase bootstrap. drop z = "
            "(DE here - DE in the next bin) / bootstrap SE of that "
            "difference; a positive z is a reversal. Oliva-Aviles, Meyer & "
            "Opsomer (2019)'s CIC is the formal test; it has little power "
            "with this many small bins.", "",
        ]
        for (segment, axis), table in monotonicity.items():
            lines += [f"**{segment}, {axis}**", "",
                      "| bin | range | phase-1 | gold | DE (se) | HT (se) | "
                      "drop z |", "|---|---|---|---|---|---|---|"]
            for row in table.itertuples():
                z = "-" if not np.isfinite(row.drop_z) else f"{row.drop_z:+.2f}"
                lines.append(
                    f"| {row.bin + 1} | {row.lo:.6f}–{row.hi:.6f} | "
                    f"{row.n_phase1} | {row.n_gold} | {row.de:.3f} "
                    f"({row.de_se:.3f}) | {row.ht:.3f} ({row.ht_se:.3f}) | "
                    f"{z} |"
                )
            worst = np.nanmax(table["drop_z"].to_numpy(dtype = float))
            lines += ["", f"Largest reversal z: {worst:+.2f}.", ""]

    lines += ["", "## Fit configuration", "",
              f"- min_cell_gold: {fit_config.min_cell_gold}",
              f"- refine_by_confidence: {fit_config.refine_by_confidence}",
              f"- grid_points: {fit_config.grid_points}",
              f"- output_bins: {fit_config.output_bins}",
              f"- bootstrap_reps: {fit_config.bootstrap_reps}",
              f"- band_alpha: {fit_config.band_alpha}",
              f"- rng_seed: {fit_config.rng_seed}",
              f"- matched_index_mode: {fit_config.matched_index_mode}",
              f"- band_aggregation: {fit_config.band_aggregation}",
              f"- pool_min_coef: {fit_config.pool_min_coef}",
              f"- surface bins requested: {fit_config.surface_osm_bins} × "
              f"{fit_config.surface_ov_bins}",
              f"- score rounding: {calibration_fit.SCORE_DECIMALS} dp", ""]

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
    parser.add_argument(
        "--out-dir", default = None,
        help = ("Write curves and report here instead of the deployed "
                "conflation/<version>/calibration directory."),
    )
    parser.add_argument(
        "--allow-deployed", action = "store_true",
        help = "Permit --out-dir to be the deployed calibration directory.",
    )
    parser.add_argument(
        "--matched-index-mode", default = None,
        choices = list(calibration_fit.INDEX_MODES),
        help = "Override conflation.calibration.matched_index_mode.",
    )
    parser.add_argument(
        "--skip-monotonicity", action = "store_true",
        help = "Skip the per-axis monotonicity tables in the report.",
    )
    parser.add_argument(
        "--skip-ht-review", action = "store_true",
        help = "Skip the design-weighted check and its review PDF.",
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
        band_aggregation = str(knobs.get(
            "band_aggregation", calibration_fit.FitConfig.band_aggregation
        )),
        pool_min_coef = float(knobs.get(
            "pool_min_coef", calibration_fit.FitConfig.pool_min_coef
        )),
        matched_index_mode = str(
            args.matched_index_mode
            or knobs.get("matched_index_mode",
                         calibration_fit.FitConfig.matched_index_mode)
        ),
    )
    deployed_dir = config.get_dir_path("conflation") / "calibration"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else deployed_dir
    if (args.out_dir and out_dir.resolve() == deployed_dir.resolve()
            and not args.allow_deployed):
        raise SystemExit(
            f"--out-dir {out_dir} is the deployed calibration directory; "
            f"pass --allow-deployed to write there"
        )

    print("Fitting segment curves...")
    results = calibration_fit.fit_all_segments(
        validation_rows, fit_config, populations = population
    )

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

    monotonicity = (
        {} if args.skip_monotonicity
        else monotonicity_tables(validation_rows, fit_config)
    )
    ht_lines = (
        [] if args.skip_ht_review
        else ht_review_lines(out_dir, validation_rows, handoff_metadata,
                             fit_config)
    )
    report_path = write_fit_report(out_dir, results, handoff_metadata,
                                   fit_config, monotonicity = monotonicity,
                                   ht_lines = ht_lines)
    print(f"Fit report: {report_path}")
    print(f"Done in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
