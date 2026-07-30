#!/usr/bin/env python
"""
Compare candidate curve indices for the matched segment on predictive validity.

The matched segment carries two source scores and must reduce them to one index
before the calibration curve is fit. Two candidates:

``pool``
    the fitted log-odds pool, ``b0 + b_osm*logit(s_osm) + b_ov*logit(s_ov)``
    (three parameters estimated from matched gold)
``average``
    the unweighted mean of the two source scores (no parameters)

The comparison is a **design-respecting K-fold cross-fit**: within each fold
everything estimated from gold is re-estimated on the training split alone,
including the pool coefficients, so a multi-parameter index gets no in-sample
advantage. Scores are design-weighted on held-out gold.

Reported measures, and why each:

- **Brier** and **log score** -- both proper scoring rules, so either ranks
  forecasts honestly. Reporting both guards against a verdict that depends on
  one rule's particular sensitivity.
- **CORP decomposition** ``mean = MCB - DSC + UNC`` (Dimitriadis, Gneiting &
  Jordan 2021). This is the measure that actually answers the question: ``DSC``
  is the discrimination the index supplies, and ``MCB`` is miscalibration that
  the isotonic step removes regardless. An index that keeps ``DSC`` has not cost
  predictive validity, even if its raw Brier differs slightly. ``UNC`` depends
  only on the outcomes and is identical for both.
- **Paired bootstrap of the difference**, resampling gold rows within fold and
  class. A confidence interval spanning zero means the simpler index is not
  measurably worse, which is the decision rule this script exists to serve.

Config keys used (config.yaml):
    calibration.validation_rows — the validation handoff table
    conflation.calibration.*    — fit knobs (min_cell_gold, grid_points, ...)

Prerequisites:
    openpois-validator's scripts/08_export_handoff.py has run for the round.

Output file(s):
    ~/data/openpois/conflation/<version>/calibration/matched_index_comparison.md

Usage:
    python scripts/conflation/compare_matched_index.py [--folds 5] [--reps 400]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from config_versioned import Config

from openpois.conflation import calibration_fit

SEGMENT = "matched"
MODES = ("pool", "average")
# Direction of improvement per measure: -1 where lower is better (Brier, log
# score, miscalibration), +1 where higher is better (discrimination). Getting
# this wrong silently inverts the DSC verdict.
BETTER_WHEN = {"brier": -1, "log_score": -1, "mcb": -1, "dsc": +1}


def _verdict(measure: str, stat: dict) -> str:
    """Read a paired difference (candidate minus reference) as a verdict."""
    if stat["lower"] <= 0 <= stat["upper"]:
        return "no difference"
    improved = (stat["mean"] < 0) == (BETTER_WHEN[measure] < 0)
    return "average better" if improved else "pool better"


def evaluate(rows: pd.DataFrame, classes: pd.Series,
             fit_config: calibration_fit.FitConfig, index_mode: str,
             n_folds: int) -> dict:
    """Cross-fit out-of-fold predictions and their scores for one index."""
    out_of_fold = calibration_fit.cross_fit_predictions(
        rows, classes, fit_config, n_folds = n_folds, segment = SEGMENT,
        index_mode = index_mode,
    )
    if not out_of_fold.get("n_folds"):
        raise SystemExit("Too little gold for a cross-fit comparison")
    scores = calibration_fit.scoring_rules(
        out_of_fold["predicted"], out_of_fold["actual"], out_of_fold["weight"]
    )
    return {"index_mode": index_mode, "out_of_fold": out_of_fold, **scores}


def paired_bootstrap(reference: dict, candidate: dict, classes: pd.Series,
                     rows: pd.DataFrame, reps: int, seed: int) -> dict:
    """Bootstrap the score differences on the same resampled gold rows.

    Pairing matters: both indices are scored on identical rows in every
    replicate, so the interval reflects the *difference* rather than the sum of
    two independent sampling errors.
    """
    rng = np.random.default_rng(seed)
    ref = reference["out_of_fold"]
    cand = candidate["out_of_fold"]
    # Both cross-fits use the same seed and therefore the same fold assignment
    # and row order, so positions align one-to-one.
    if len(ref["predicted"]) != len(cand["predicted"]):
        raise SystemExit("Cross-fit row sets differ; cannot pair")

    strata = ref["fold"].astype(str)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    deltas = {"brier": [], "log_score": [], "dsc": [], "mcb": []}
    for _ in range(reps):
        idx = np.concatenate([
            g[rng.integers(0, len(g), len(g))] for g in groups if len(g)
        ])
        ref_scores = calibration_fit.scoring_rules(
            ref["predicted"][idx], ref["actual"][idx], ref["weight"][idx]
        )
        cand_scores = calibration_fit.scoring_rules(
            cand["predicted"][idx], cand["actual"][idx], cand["weight"][idx]
        )
        for key in deltas:
            deltas[key].append(cand_scores[key] - ref_scores[key])
    return {
        key: {
            "mean": float(np.mean(values)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for key, values in deltas.items()
    }


def deployed_impact(rows: pd.DataFrame, config, fit_config, input_suffix: str
                    ) -> dict:
    """How far the two indices move the *published* value, per matched POI.

    Statistical significance and product significance are different questions:
    a difference can be real yet too small to change any published number, or
    small in score terms yet large enough to move POIs across a band edge. This
    fits the full curve both ways and compares the deployed probabilities on the
    production matched population.
    """
    # Imported here so the core comparison does not pay for pyarrow/geo imports.
    # pylint: disable-next=import-outside-toplevel
    import pyarrow.parquet as pq

    # pylint: disable-next=import-outside-toplevel
    from openpois.conflation import calibration

    conflated = config.get_file_path("conflation", "conflated")
    if input_suffix:
        conflated = conflated.with_name(
            f"{conflated.stem}_{input_suffix}{conflated.suffix}"
        )
    if not conflated.exists():
        return {}

    osm, overture = [], []
    columns = ["source", "osm_conf_mean", "overture_confidence"]
    for batch in pq.ParquetFile(str(conflated)).iter_batches(
        batch_size = 2_000_000, columns = columns
    ):
        frame = batch.to_pandas()
        mask = (frame["source"] == SEGMENT).to_numpy()
        if mask.any():
            osm.append(frame["osm_conf_mean"].to_numpy(dtype = float)[mask])
            overture.append(
                frame["overture_confidence"].to_numpy(dtype = float)[mask]
            )
    if not osm:
        return {}
    population = pd.DataFrame({"osm_score": np.concatenate(osm),
                               "overture_score": np.concatenate(overture)})

    deployed, bands = {}, {}
    for mode in MODES:
        mode_config = replace(fit_config, matched_index_mode = mode)
        result = calibration_fit.fit_segment(rows, SEGMENT, mode_config,
                                             population = population)
        index = calibration_fit.segment_scores(
            population, SEGMENT, result["pool"], mode
        )
        deployed[mode] = calibration.apply_curve(
            index, result["lookup"]
        )["conf_mean"].to_numpy()
        bands[mode] = result["band_width_median"]

    diff = deployed["average"] - deployed["pool"]
    edges = [0.3, 0.7, 0.9]
    crossed = int(
        (np.digitize(deployed["average"], edges)
         != np.digitize(deployed["pool"], edges)).sum()
    )
    return {
        "n": int(len(diff)),
        "band_width_median": bands,
        "deployed_mean": {m: float(v.mean()) for m, v in deployed.items()},
        "mean_abs_diff": float(np.abs(diff).mean()),
        "p95_abs_diff": float(np.quantile(np.abs(diff), 0.95)),
        "max_abs_diff": float(np.abs(diff).max()),
        "share_over_0p05": float((np.abs(diff) > 0.05).mean()),
        "band_edge_crossings": crossed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--folds", type = int, default = 5)
    parser.add_argument("--reps", type = int, default = 400,
                        help = "Paired bootstrap replicates (default 400).")
    parser.add_argument("--with-deployed-impact", action = "store_true",
                        help = ("Also fit both curves fully and compare the "
                                "published values on the production matched "
                                "population (reads the conflated parquet)."))
    parser.add_argument("--input-suffix", default = "cd",
                        help = "Conflated parquet suffix for --with-deployed-impact.")
    args = parser.parse_args()

    config = Config("~/repos/openpois/config.yaml")
    knobs = config.get("conflation", "calibration")
    fit_config = calibration_fit.FitConfig(
        min_cell_gold = int(knobs["min_cell_gold"]),
        grid_points = int(knobs["grid_points"]),
        output_bins = int(knobs["curve_output_bins"]),
        bootstrap_reps = int(knobs["bootstrap_reps"]),
        band_alpha = float(knobs["band_alpha"]),
        rng_seed = int(knobs["rng_seed"]),
        refine_by_confidence = bool(knobs["refine_by_confidence"]),
    )

    validation_rows = pd.read_parquet(
        config.get_file_path("calibration", "validation_rows")
    )
    rows = validation_rows[
        (validation_rows["segment"] == SEGMENT)
        & validation_rows["stratum"].isin(calibration_fit.SEGMENTS)
        & validation_rows["llm_verdict"].isin(calibration_fit.VERDICTS)
    ].reset_index(drop = True)
    classes = calibration_fit.merge_thin_cells(
        calibration_fit.refined_class(
            rows, refine = fit_config.refine_by_confidence
        ),
        rows["gold"].to_numpy(dtype = bool),
        fit_config.min_cell_gold,
    ).reset_index(drop = True)
    print(f"matched: {len(rows):,} phase-1 rows, "
          f"{int(rows['gold'].sum()):,} gold; {args.folds}-fold cross-fit")

    evaluated = {}
    for mode in MODES:
        # cross_fit_predictions needs a `score` column present; each mode
        # recomputes it per fold, so the initial value only has to be finite.
        seeded = rows.assign(
            score = calibration_fit.segment_scores(
                rows, SEGMENT, None, "average"
            )
        )
        evaluated[mode] = evaluate(seeded, classes, fit_config, mode,
                                   args.folds)
        e = evaluated[mode]
        print(f"  {mode:8s} Brier {e['brier']:.5f}  log {e['log_score']:.5f}  "
              f"MCB {e['mcb']:.5f}  DSC {e['dsc']:.5f}  UNC {e['unc']:.5f}")

    delta = paired_bootstrap(evaluated["pool"], evaluated["average"], classes,
                             rows, args.reps, fit_config.rng_seed + 3100)
    print("\naverage minus pool:")
    for key, stat in delta.items():
        print(f"  {key:10s} {stat['mean']:+.5f} "
              f"[{stat['lower']:+.5f}, {stat['upper']:+.5f}]  "
              f"{_verdict(key, stat)}")

    lines = [
        "# Matched-segment curve index: pool vs average",
        "",
        f"- Validation round: {config.get('versions', 'calibration')}",
        f"- Cross-fit folds: {args.folds}; paired bootstrap reps: {args.reps}",
        f"- Matched phase-1 rows: {len(rows):,}; gold: "
        f"{int(rows['gold'].sum()):,}",
        "",
        "Everything estimated from gold is re-estimated inside each training "
        "fold, pool coefficients included, so neither index gets an in-sample "
        "advantage. Scores are design-weighted on held-out gold. Lower Brier "
        "and log score are better; higher DSC (discrimination) is better; "
        "lower MCB (miscalibration) is better.",
        "",
        "| index | Brier | log score | MCB | DSC | UNC |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        e = evaluated[mode]
        lines.append(
            f"| `{mode}` | {e['brier']:.5f} | {e['log_score']:.5f} | "
            f"{e['mcb']:.5f} | {e['dsc']:.5f} | {e['unc']:.5f} |"
        )
    lines += [
        "",
        "## Paired difference (average minus pool)",
        "",
        "Brier, log score and MCB are better *lower*; DSC is better *higher*. "
        "An interval spanning zero means the simpler index is not measurably "
        "worse on that measure.",
        "",
        "| measure | difference | 95% interval | verdict |",
        "|---|---|---|---|",
    ]
    for key, stat in delta.items():
        lines.append(
            f"| {key} | {stat['mean']:+.5f} | "
            f"[{stat['lower']:+.5f}, {stat['upper']:+.5f}] | "
            f"{_verdict(key, stat)} |"
        )
    if args.with_deployed_impact:
        impact = deployed_impact(rows, config, fit_config, args.input_suffix)
        if impact:
            print(f"\ndeployed impact over {impact['n']:,} matched POIs:")
            print(f"  median band width: pool "
                  f"{impact['band_width_median']['pool']:.3f}, average "
                  f"{impact['band_width_median']['average']:.3f}")
            print(f"  mean |difference| {impact['mean_abs_diff']:.4f}, "
                  f"p95 {impact['p95_abs_diff']:.4f}, "
                  f"max {impact['max_abs_diff']:.4f}")
            print(f"  moved > 0.05: {100 * impact['share_over_0p05']:.1f}%; "
                  f"crossed a band edge: "
                  f"{impact['band_edge_crossings']:,}")
            lines += [
                "", "## Deployed impact on the production matched population",
                "",
                "Statistical and product significance are separate questions. "
                "Both curves fitted in full, then compared on the published "
                "value per POI.",
                "",
                f"- Matched POIs: {impact['n']:,}",
                f"- Median band width: pool "
                f"{impact['band_width_median']['pool']:.3f}, average "
                f"{impact['band_width_median']['average']:.3f} (the average "
                f"has no pool coefficients to refit per bootstrap replicate, "
                f"so it carries less parameter uncertainty)",
                f"- Deployed mean: pool "
                f"{impact['deployed_mean']['pool']:.4f}, average "
                f"{impact['deployed_mean']['average']:.4f}",
                f"- Mean |difference| {impact['mean_abs_diff']:.4f}; p95 "
                f"{impact['p95_abs_diff']:.4f}; max "
                f"{impact['max_abs_diff']:.4f}",
                f"- Moved more than 0.05: "
                f"{100 * impact['share_over_0p05']:.1f}%; crossed a published "
                f"band edge: {impact['band_edge_crossings']:,}",
                "",
            ]

    lines.append("")

    out_dir = config.get_dir_path("conflation") / "calibration"
    out_dir.mkdir(parents = True, exist_ok = True)
    out_path = out_dir / "matched_index_comparison.md"
    out_path.write_text("\n".join(lines), encoding = "utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
