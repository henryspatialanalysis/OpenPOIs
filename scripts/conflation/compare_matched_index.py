#!/usr/bin/env python
"""
Compare how the matched segment combines its two source scores.

v4-era evaluation tool: it compares the v4 matched index modes, which were
retired when the Bayesian grid curves went to production in October 2026.
Its deployed-curve reads assume a v4 calibration directory.

The matched segment carries an OSM and an Overture score. Each mode below maps
the pair to a calibrated P(exists and open), monotone in both scores:

``pool``
    constrained log-odds pool, ``b0 + b_osm*logit(s_osm) + b_ov*logit(s_ov)``
    with both slopes >= ``pool_min_coef`` (3 parameters; production to
    2026-09)
``average``
    unweighted mean of the two raw scores (no parameters)
``additive``
    ``a + h_osm(s_osm) + h_ov(s_ov)``, each ``h`` nondecreasing
``interaction``
    monotone bilinear model in rescaled logits (4 parameters; production
    since the October 2026 run)
``surface`` / ``surface_8x6``
    per-cell difference estimator on a 6x4 (8x6) grid projected onto the
    doubly-monotone cone -- no index

Read the paired differences as a ladder: additive vs pool asks whether the
per-source shapes are wrong in logit; interaction vs pool and surface vs
additive ask whether there is an interaction; surface vs interaction asks
whether the interaction is more than bilinear.

The comparison is a **design-respecting K-fold cross-fit**: within each fold
everything estimated from gold is re-estimated on the training split alone,
index parameters included, so a multi-parameter mode gets no in-sample
advantage. Held-out gold is scored through each mode's **published lookup**
(40 equal-mass bins, or the cells) built per fold on the production
population, so a 40-bin mode and a 24-cell mode are compared on the maps that
would actually ship. The pre-2026-09 grid-interpolated score is reported
alongside for continuity with the July comparison.

Reported measures:

- **Brier** and **log score** -- both proper scoring rules.
- **CORP decomposition** ``mean = MCB - DSC + UNC`` (Dimitriadis, Gneiting &
  Jordan 2021): ``DSC`` is the discrimination the map supplies, ``MCB`` its
  miscalibration.
- **Paired bootstrap of each ladder difference**, resampling held-out gold
  rows within fold (every mode shares the fold assignment and row order).

Config keys used (config.yaml):
    calibration.validation_rows, calibration.metadata
    conflation.calibration.*    -- fit knobs
    directories.conflation      -- population parquets, deployed curves

Output file(s) (in --out-dir, never the deployed calibration directory):
    matched_index_comparison_<tag>.md / .json
    fits/<mode>/matched_{curve.parquet,metadata.json}  (--with-deployed-impact)

Usage:
    python scripts/conflation/compare_matched_index.py [--folds 5] [--reps 400]
        [--seed N] [--modes pool,average,...] [--out-dir DIR]
        [--with-deployed-impact]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from config_versioned import Config

from openpois.conflation import calibration, calibration_fit

SEGMENT = "matched"
# name -> (index_mode, requested surface bins or None)
MODES = {
    "pool": ("pool", None),
    "average": ("average", None),
    "additive": ("additive", None),
    "interaction": ("interaction", None),
    "surface": ("surface", (6, 6)),
    "surface_8x6": ("surface", (8, 8)),
}
# (candidate, reference): the difference reported is candidate minus
# reference.
LADDER = (
    ("additive", "pool"),
    ("interaction", "pool"),
    ("surface", "additive"),
    ("surface", "interaction"),
    ("interaction", "additive"),
    ("surface", "pool"),
    ("average", "pool"),
    ("surface_8x6", "surface"),
)
MEASURES = ("brier", "log_score", "dsc", "mcb")
# Direction of improvement per measure: -1 where lower is better (Brier, log
# score, miscalibration), +1 where higher is better (discrimination). Getting
# this wrong silently inverts the DSC verdict.
BETTER_WHEN = {"brier": -1, "log_score": -1, "mcb": -1, "dsc": +1}
# Published-value edges used to count POIs that change colour band.
BAND_EDGES = (0.3, 0.7, 0.9)
OVERTURE_ATOMS = (0.919912, 0.990219)


def _verdict(measure: str, stat: dict, candidate: str = "candidate",
             reference: str = "reference") -> str:
    """Read a paired difference (candidate minus reference) as a verdict."""
    if stat["lower"] <= 0 <= stat["upper"]:
        return "no difference"
    improved = (stat["mean"] < 0) == (BETTER_WHEN[measure] < 0)
    return f"{candidate} better" if improved else f"{reference} better"


def load_population(conflated_path: Path) -> pd.DataFrame:
    """Non-shadow matched rows' two source scores, streamed column-scoped.

    Mirrors ``fit_calibration.population_by_segment``: shadow-matched rows
    keep their change-detection value and never ride a curve, so they are
    excluded here too. Never loads the conflated parquet whole.
    """
    pf = pq.ParquetFile(str(conflated_path))
    columns = ["source", "osm_conf_mean", "overture_confidence"]
    if "shadow_matched" in pf.schema_arrow.names:
        columns.append("shadow_matched")
    osm, overture = [], []
    for batch in pf.iter_batches(batch_size = 2_000_000, columns = columns):
        frame = batch.to_pandas()
        mask = (frame["source"] == SEGMENT).to_numpy().copy()
        if "shadow_matched" in frame.columns:
            mask &= ~frame["shadow_matched"].to_numpy(dtype = bool)
        osm.append(frame["osm_conf_mean"].to_numpy(dtype = float)[mask])
        overture.append(
            frame["overture_confidence"].to_numpy(dtype = float)[mask]
        )
    return pd.DataFrame({"osm_score": np.concatenate(osm),
                         "overture_score": np.concatenate(overture)})


def evaluate(rows: pd.DataFrame, classes: pd.Series,
             fit_config: calibration_fit.FitConfig, mode: str,
             n_folds: int, population: pd.DataFrame) -> dict:
    """Cross-fit out-of-fold predictions and their scores for one mode."""
    index_mode, bins = MODES[mode]
    out_of_fold = calibration_fit.cross_fit_predictions(
        rows, classes, fit_config, n_folds = n_folds, segment = SEGMENT,
        index_mode = index_mode, population = population,
        surface_bins = bins,
    )
    if not out_of_fold.get("n_folds"):
        raise SystemExit("Too little gold for a cross-fit comparison")
    scores = calibration_fit.scoring_rules(
        out_of_fold["predicted"], out_of_fold["actual"], out_of_fold["weight"]
    )
    grid = calibration_fit.scoring_rules(
        out_of_fold["predicted_grid"], out_of_fold["actual"],
        out_of_fold["weight"],
    )
    return {"mode": mode, "out_of_fold": out_of_fold, **scores,
            "grid": grid}


def paired_bootstrap(reference: dict, candidate: dict, reps: int, seed: int,
                     key: str = "predicted") -> dict:
    """Bootstrap the score differences on the same resampled gold rows.

    Pairing matters: both modes are scored on identical rows in every
    replicate, so the interval reflects the *difference* rather than the sum
    of two independent sampling errors. Rows are resampled within cross-fit
    fold (the strata are folds only; refined class is already balanced across
    folds by the fold assignment).
    """
    rng = np.random.default_rng(seed)
    ref = reference["out_of_fold"]
    cand = candidate["out_of_fold"]
    # Every cross-fit uses the same seed and therefore the same fold
    # assignment and row order, so positions align one-to-one.
    if (len(ref[key]) != len(cand[key])
            or not np.array_equal(ref["fold"], cand["fold"])
            or not np.array_equal(ref["actual"], cand["actual"])):
        raise SystemExit("Cross-fit row sets differ; cannot pair")

    strata = ref["fold"].astype(str)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    deltas = {measure: [] for measure in MEASURES}
    for _ in range(reps):
        idx = np.concatenate([
            g[rng.integers(0, len(g), len(g))] for g in groups if len(g)
        ])
        ref_scores = calibration_fit.scoring_rules(
            ref[key][idx], ref["actual"][idx], ref["weight"][idx]
        )
        cand_scores = calibration_fit.scoring_rules(
            cand[key][idx], cand["actual"][idx], cand["weight"][idx]
        )
        for measure in deltas:
            deltas[measure].append(cand_scores[measure] - ref_scores[measure])
    return {
        measure: {
            "mean": float(np.mean(values)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for measure, values in deltas.items()
    }


def _region_labels(osm: np.ndarray, overture: np.ndarray,
                   osm_edges: np.ndarray) -> np.ndarray:
    """4x4 region per POI: OSM quartile x Overture {below atom, atom,
    between atoms, top atom and above}, as in the 2026-09-25 audit."""
    q = np.clip(np.searchsorted(osm_edges, osm, side = "right") - 1, 0, 3)
    low, high = OVERTURE_ATOMS
    ov = np.select(
        [overture < low, overture == low, overture < high],
        [0, 1, 2], default = 3,
    )
    return q * 4 + ov


def deployed_values(population: pd.DataFrame, lookup: pd.DataFrame,
                    metadata: dict) -> pd.DataFrame:
    """Run the production deploy path (``calibrate_frame``) on matched rows."""
    frame = pd.DataFrame({
        "source": np.full(len(population), SEGMENT, dtype = object),
        "osm_conf_mean": population["osm_score"].to_numpy(dtype = float),
        "overture_confidence": population["overture_score"].to_numpy(
            dtype = float
        ),
        "conf_mean": np.full(len(population), np.nan),
    })
    meta = {SEGMENT: metadata}
    return calibration.calibrate_frame(
        frame, {SEGMENT: lookup},
        pool_params = calibration.pool_params_from_metadata(meta),
        index_modes = calibration.index_modes_from_metadata(meta),
        score_decimals = calibration.score_decimals_from_metadata(meta),
    )


def deployed_impact(rows: pd.DataFrame, fit_config, modes: list,
                    population: pd.DataFrame, deployed_dir: Path,
                    handoff_metadata: dict, fits_dir: Path) -> dict:
    """Each mode, fit in full, against the DEPLOYED matched curve, per POI.

    Statistical and product significance are separate questions: a difference
    can be real yet too small to change a published number, or small in score
    terms yet large enough to move POIs across a band edge. The deployed
    curve and metadata are read as shipped (not refit); each mode is fit in
    full with its bins placed on ``population`` (the next release's frame)
    and both go through the production deploy path on the same rows. Each
    mode's fitted curve and metadata are written under ``fits_dir``.
    """
    deployed_lookup = pd.read_parquet(deployed_dir / f"{SEGMENT}_curve.parquet")
    with open(deployed_dir / f"{SEGMENT}_metadata.json",
              encoding = "utf-8") as handle:
        deployed_meta = json.load(handle)
    base = deployed_values(population, deployed_lookup, deployed_meta)
    base_mean = base["conf_mean"].to_numpy()
    base_width = (base["conf_upper"] - base["conf_lower"]).to_numpy()

    rounded = calibration_fit.round_scores(population)
    osm_edges = np.quantile(rounded["osm_score"], [0, 0.25, 0.5, 0.75, 1.0])
    regions = _region_labels(rounded["osm_score"].to_numpy(),
                             rounded["overture_score"].to_numpy(), osm_edges)

    out = {
        "n": int(len(population)),
        "deployed": {
            "mean": float(np.nanmean(base_mean)),
            "band_width_mean": float(np.nanmean(base_width)),
            "band_width_median": float(np.nanmedian(base_width)),
            "index_mode": deployed_meta.get("index_mode"),
        },
        "osm_quartile_edges": osm_edges.tolist(),
        "modes": {},
    }
    for mode in modes:
        started = time.time()
        index_mode, bins = MODES[mode]
        mode_config = replace(fit_config, matched_index_mode = index_mode)
        result = calibration_fit.fit_segment(
            rows, SEGMENT, mode_config, population = population,
            surface_bins = bins,
        )
        metadata = calibration_fit.curve_metadata(
            SEGMENT, result, handoff_metadata, mode_config
        )
        calibration_fit.write_curve(fits_dir / mode, SEGMENT, result["lookup"],
                                    metadata)
        values = deployed_values(population, result["lookup"], metadata)
        mean = values["conf_mean"].to_numpy()
        width = (values["conf_upper"] - values["conf_lower"]).to_numpy()
        diff = mean - base_mean
        crossed = int(
            (np.digitize(mean, BAND_EDGES)
             != np.digitize(base_mean, BAND_EDGES)).sum()
        )
        by_region = {}
        for region in range(16):
            in_region = regions == region
            if in_region.any():
                by_region[f"osm_q{region // 4 + 1}_ov{region % 4 + 1}"] = {
                    "n": int(in_region.sum()),
                    "mean_abs_shift": float(np.abs(diff[in_region]).mean()),
                    "mean_shift": float(diff[in_region].mean()),
                }
        entry = {
            "deployed_mean": float(np.nanmean(mean)),
            "mean_abs_diff": float(np.nanmean(np.abs(diff))),
            "p95_abs_diff": float(np.nanquantile(np.abs(diff), 0.95)),
            "max_abs_diff": float(np.nanmax(np.abs(diff))),
            "share_over_0p05": float(np.nanmean(np.abs(diff) > 0.05)),
            "band_edge_crossings": crossed,
            "band_width_mean": float(np.nanmean(width)),
            "band_width_median": float(np.nanmedian(width)),
            "band_width_grid_median": result.get("band_width_grid_median"),
            "by_region": by_region,
            "effective_parameters": metadata.get("effective_parameters"),
            "index": result.get("index"),
            "cross_fit_in_fit": result["cross_fit"],
        }
        surface = result.get("surface")
        if surface is not None:
            for method, band in surface["bands"].items():
                cell_width = band["upper"] - band["lower"]
                weights = surface["population_counts"]
                entry[f"band_width_mean_{method}"] = float(
                    np.sum(cell_width * weights) / np.sum(weights)
                )
            entry["surface_theta"] = surface["theta"].tolist()
            entry["surface_edges"] = {
                k: v.tolist() for k, v in surface["edges"].items()
            }
        out["modes"][mode] = entry
        print(f"  {mode:12s} fit + deploy in {time.time() - started:.0f}s: "
              f"moved >0.05 {100 * entry['share_over_0p05']:.1f}%, "
              f"crossed {crossed:,}, band mean "
              f"{entry['band_width_mean']:.3f}")
    return out


def _git_state() -> dict:
    """Commit and dirty flag of the working tree, for reproducibility."""
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output = True, text = True,
                             check = True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output = True, text = True, check = True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


def _fmt_interval(stat: dict) -> str:
    return (f"{stat['mean']:+.5f} "
            f"[{stat['lower']:+.5f}, {stat['upper']:+.5f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--folds", type = int, default = 5)
    parser.add_argument("--reps", type = int, default = 400,
                        help = "Paired bootstrap replicates (default 400).")
    parser.add_argument("--seed", type = int, default = None,
                        help = ("Fold-assignment and bootstrap seed "
                                "(default: conflation.calibration.rng_seed, "
                                "which reproduces the July comparison)."))
    parser.add_argument("--modes", default = ",".join(MODES),
                        help = "Comma-separated modes (default: all).")
    parser.add_argument("--out-dir", default = None,
                        help = ("Output directory (default: "
                                "<conflation dir>/calibration_eval_<today>)."))
    parser.add_argument("--allow-deployed", action = "store_true",
                        help = "Permit --out-dir to be the deployed dir.")
    parser.add_argument("--tag", default = None,
                        help = "Output file tag (default k<folds>_s<seed>).")
    parser.add_argument("--population-version", default = None,
                        help = ("Conflation version whose population places "
                                "the cross-fit bins (default: the validation "
                                "round's own conflation_version)."))
    parser.add_argument("--with-deployed-impact", action = "store_true",
                        help = ("Also fit every mode fully and compare its "
                                "published values with the deployed curve on "
                                "the current production matched population."))
    parser.add_argument("--input-suffix", default = "cd",
                        help = "Conflated parquet suffix for populations.")
    args = parser.parse_args()
    started = time.time()

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
        band_aggregation = str(knobs.get(
            "band_aggregation", calibration_fit.FitConfig.band_aggregation
        )),
        pool_min_coef = float(knobs.get(
            "pool_min_coef", calibration_fit.FitConfig.pool_min_coef
        )),
    )
    if args.seed is not None:
        fit_config = replace(fit_config, rng_seed = args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise SystemExit(f"Unknown modes: {unknown}")

    conflation_dir = config.get_dir_path("conflation")
    deployed_dir = conflation_dir / "calibration"
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        conflation_dir / f"calibration_eval_{time.strftime('%Y%m%d')}"
    )
    if out_dir.resolve() == deployed_dir.resolve() and not args.allow_deployed:
        raise SystemExit(
            f"--out-dir {out_dir} is the deployed calibration directory; "
            f"pass --allow-deployed to write there"
        )
    out_dir.mkdir(parents = True, exist_ok = True)
    tag = args.tag or f"k{args.folds}_s{fit_config.rng_seed}"

    with open(config.get_file_path("calibration", "metadata"),
              encoding = "utf-8") as handle:
        handoff_metadata = json.load(handle)
    validation_rows = pd.read_parquet(
        config.get_file_path("calibration", "validation_rows")
    )
    rows = calibration_fit.round_scores(validation_rows[
        (validation_rows["segment"] == SEGMENT)
        & validation_rows["stratum"].isin(calibration_fit.SEGMENTS)
        & validation_rows["llm_verdict"].isin(calibration_fit.VERDICTS)
    ].reset_index(drop = True))
    classes = calibration_fit.merge_thin_cells(
        calibration_fit.refined_class(
            rows, refine = fit_config.refine_by_confidence
        ),
        rows["gold"].to_numpy(dtype = bool),
        fit_config.min_cell_gold,
    ).reset_index(drop = True)

    population_version = (args.population_version
                          or handoff_metadata.get("conflation_version"))
    population_path = (
        conflation_dir.parent / str(population_version)
        / f"conflated_{args.input_suffix}.parquet"
    )
    population = load_population(population_path)
    print(f"matched: {len(rows):,} phase-1 rows, "
          f"{int(rows['gold'].sum()):,} gold; {args.folds}-fold cross-fit, "
          f"seed {fit_config.rng_seed}; bins on {population_path} "
          f"({len(population):,} POIs)")

    evaluated = {}
    for mode in modes:
        mode_started = time.time()
        # cross_fit_predictions needs a finite `score` column; each mode
        # recomputes it per fold.
        seeded = rows.assign(
            score = calibration_fit.segment_scores(rows, SEGMENT, None,
                                                   "average")
        )
        evaluated[mode] = evaluate(seeded, classes, fit_config, mode,
                                   args.folds, population)
        e = evaluated[mode]
        print(f"  {mode:12s} Brier {e['brier']:.5f}  log {e['log_score']:.5f}"
              f"  MCB {e['mcb']:.5f}  DSC {e['dsc']:.5f}  | grid Brier "
              f"{e['grid']['brier']:.5f} log {e['grid']['log_score']:.5f}"
              f"  ({time.time() - mode_started:.0f}s)")

    ladder = {}
    ladder_grid = {}
    for offset, (cand, ref) in enumerate(LADDER):
        if cand not in evaluated or ref not in evaluated:
            continue
        # average - pool keeps the July comparison's bootstrap seed so its
        # interval reproduces the published one; the rest get their own.
        seed = fit_config.rng_seed + (
            3100 if (cand, ref) == ("average", "pool") else 3200 + offset
        )
        ladder[f"{cand}-{ref}"] = paired_bootstrap(
            evaluated[ref], evaluated[cand], args.reps, seed
        )
        ladder_grid[f"{cand}-{ref}"] = paired_bootstrap(
            evaluated[ref], evaluated[cand], args.reps, seed,
            key = "predicted_grid",
        )
    for pair, delta in ladder.items():
        cand, ref = pair.split("-")
        print(f"\n{cand} minus {ref}:")
        for measure, stat in delta.items():
            print(f"  {measure:10s} {_fmt_interval(stat)}  "
                  f"{_verdict(measure, stat, cand, ref)}")

    lines = [
        "# Matched-segment combination modes: cross-fit comparison",
        "",
        f"- Validation round: {config.get('versions', 'calibration')}",
        f"- Cross-fit folds: {args.folds}; seed: {fit_config.rng_seed}; "
        f"paired bootstrap reps: {args.reps}",
        f"- Matched phase-1 rows: {len(rows):,}; gold: "
        f"{int(rows['gold'].sum()):,}",
        f"- Lookup bins / cells placed on: `{population_path}` "
        f"({len(population):,} non-shadow matched POIs)",
        "",
        "Everything estimated from gold is re-estimated inside each training "
        "fold, index parameters included. Held-out gold is scored through "
        "each mode's published lookup, built per fold. Lower Brier, log "
        "score and MCB are better; higher DSC is better.",
        "",
        "## Scores through the published lookup (primary)",
        "",
        "| mode | Brier | log score | MCB | DSC | UNC |",
        "|---|---|---|---|---|---|",
    ]
    for mode in modes:
        e = evaluated[mode]
        lines.append(
            f"| `{mode}` | {e['brier']:.5f} | {e['log_score']:.5f} | "
            f"{e['mcb']:.5f} | {e['dsc']:.5f} | {e['unc']:.5f} |"
        )
    lines += [
        "",
        "## Scores through the fold curve (pre-2026-09 method, continuity)",
        "",
        "| mode | Brier | log score | MCB | DSC |",
        "|---|---|---|---|---|",
    ]
    for mode in modes:
        g = evaluated[mode]["grid"]
        lines.append(
            f"| `{mode}` | {g['brier']:.5f} | {g['log_score']:.5f} | "
            f"{g['mcb']:.5f} | {g['dsc']:.5f} |"
        )
    for title, table in (("published lookup", ladder),
                         ("fold curve", ladder_grid)):
        lines += [
            "",
            f"## Paired differences along the ladder ({title})",
            "",
            "Candidate minus reference. An interval spanning zero means no "
            "measurable difference.",
            "",
            "| pair | measure | difference [95% interval] | verdict |",
            "|---|---|---|---|",
        ]
        for pair, delta in table.items():
            cand, ref = pair.split("-")
            for measure, stat in delta.items():
                lines.append(
                    f"| {cand} − {ref} | {measure} | {_fmt_interval(stat)} | "
                    f"{_verdict(measure, stat, cand, ref)} |"
                )

    impact = {}
    if args.with_deployed_impact:
        impact_population = load_population(
            conflation_dir / f"conflated_{args.input_suffix}.parquet"
        )
        print(f"\nDeployed impact on {len(impact_population):,} matched POIs "
              f"(curves in {deployed_dir}):")
        impact = deployed_impact(rows, fit_config, modes, impact_population,
                                 deployed_dir, handoff_metadata,
                                 out_dir / "fits")
        lines += [
            "", "## Deployed impact on the production matched population", "",
            f"Each mode fit in full (bins on the current release's "
            f"population, {impact['n']:,} non-shadow matched POIs) and "
            f"compared POI by POI with the **deployed** curve "
            f"(`{deployed_dir}`), both through `calibrate_frame`. Deployed "
            f"mean {impact['deployed']['mean']:.4f}; deployed band width mean "
            f"{impact['deployed']['band_width_mean']:.3f}.", "",
            "| mode | deployed mean | mean abs shift | p95 | max | moved > "
            "0.05 | band-edge crossings | band width (POI mean) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for mode, entry in impact["modes"].items():
            lines.append(
                f"| `{mode}` | {entry['deployed_mean']:.4f} | "
                f"{entry['mean_abs_diff']:.4f} | {entry['p95_abs_diff']:.4f} "
                f"| {entry['max_abs_diff']:.4f} | "
                f"{100 * entry['share_over_0p05']:.1f}% | "
                f"{entry['band_edge_crossings']:,} | "
                f"{entry['band_width_mean']:.3f} |"
            )

    lines.append("")
    stem = out_dir / f"matched_index_comparison_{tag}"
    stem.with_suffix(".md").write_text("\n".join(lines), encoding = "utf-8")

    def _strip(e):
        return {k: v for k, v in e.items() if k != "out_of_fold"}

    record = {
        "tag": tag,
        "folds": args.folds,
        "seed": fit_config.rng_seed,
        "reps": args.reps,
        "population": str(population_path),
        "git": _git_state(),
        "modes": {m: _strip(e) for m, e in evaluated.items()},
        "ladder": ladder,
        "ladder_grid": ladder_grid,
        "deployed_impact": impact,
        "elapsed_s": time.time() - started,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(calibration_fit._jsonable(record), indent = 2),
        encoding = "utf-8",
    )
    # Out-of-fold predictions per mode, for later re-analysis.
    oof = pd.DataFrame({
        f"{m}_{k}": e["out_of_fold"][k]
        for m, e in evaluated.items()
        for k in ("predicted", "predicted_grid")
    })
    first = next(iter(evaluated.values()))["out_of_fold"]
    oof = oof.assign(actual = first["actual"], weight = first["weight"],
                     fold = first["fold"])
    oof.to_parquet(stem.with_name(stem.name + "_oof.parquet"), index = False)
    print(f"\nWrote {stem.with_suffix('.md')} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
