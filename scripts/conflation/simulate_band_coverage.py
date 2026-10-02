#!/usr/bin/env python
"""
Coverage simulation for the calibration bands, under the real two-phase design.

Audits every matched-segment mode's published band -- and the shipped 1-D
``osm`` / ``overture`` bands -- by simulating validation rounds with a known
truth and checking how often each published interval contains the quantity it
publishes.

Generative model (round 20260730's structure held fixed):

1. keep every phase-1 row's source scores;
2. draw ``Y ~ Bernoulli(T(s))``;
3. draw the refined LLM class from ``P(class | Y)``, estimated design-weighted
   on the segment's gold;
4. draw phase-2 gold within verdict class at the round's realized inclusion
   rate, with the unverifiable class censused.

Truths ``T``:

``pool``         the fitted constrained-pool map (additive in logit)
``interaction``  the fitted monotone-bilinear map (substitutive on 20260730)
``overture_only`` a matched truth that depends on the Overture score alone
                 (plateau-heavy)
``osm_1d`` / ``overture_1d``  each 1-D segment's own fitted curve

Target: the published quantity, i.e. the mean of ``T`` over the production
POIs in each published bin (index modes; bins are placed per simulation on
that simulation's fitted index) or cell (surface). Regions, defined on the
target vector: ``plateau`` (target >= 0.95 and within 0.01 of every
neighbour), ``floor`` (within 0.01 of the lowest target), ``edge`` (first /
last bin, or a boundary cell), ``corner`` (2-D corners), else ``interior``.

Each simulation's records are written to ``coverage_sims/sim_<i>.parquet`` as
they finish, so a rerun resumes. ``--aggregate-only`` rebuilds the report.

Config keys used:
    calibration.validation_rows, conflation.calibration.*

Output (in --out-dir):
    coverage_sims/sim_<i>.parquet, band_coverage.md, band_coverage.parquet

Usage:
    python scripts/conflation/simulate_band_coverage.py --sims 200 --reps 200 \\
        --out-dir DIR [--workers 10] [--time-one] [--aggregate-only]
"""
from __future__ import annotations

import os

# One BLAS thread per worker: the parallelism is across simulations.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from config_versioned import Config  # noqa: E402

from openpois.conflation import calibration_fit as cf  # noqa: E402

MATCHED_MODES = ("pool", "average", "additive", "interaction", "surface")
# Populations are subsampled for bin placement and targets: 1M POIs pin the
# equal-mass edges to well under a bin's width and keep workers light.
MAX_POPULATION = 1_000_000
POPULATION_SEED = 20260925

_STATE = {}


# ---------------------------------------------------------------------------
# Setup (runs once per worker)
# ---------------------------------------------------------------------------

def _segment_rows(validation_rows: pd.DataFrame, segment: str) -> pd.DataFrame:
    return cf.round_scores(validation_rows[
        (validation_rows["segment"] == segment)
        & validation_rows["stratum"].isin(cf.SEGMENTS)
        & validation_rows["llm_verdict"].isin(cf.VERDICTS)
    ].reset_index(drop = True))


def load_population(conflated_path: Path) -> dict:
    """Non-shadow production scores per segment, subsampled, rounded."""
    pf = pq.ParquetFile(str(conflated_path))
    columns = ["source", "osm_conf_mean", "overture_confidence"]
    if "shadow_matched" in pf.schema_arrow.names:
        columns.append("shadow_matched")
    parts = {s: [] for s in cf.SEGMENTS}
    for batch in pf.iter_batches(batch_size = 2_000_000, columns = columns):
        frame = batch.to_pandas()
        keep = np.ones(len(frame), dtype = bool)
        if "shadow_matched" in frame.columns:
            keep &= ~frame["shadow_matched"].to_numpy(dtype = bool)
        for segment in cf.SEGMENTS:
            mask = (frame["source"] == segment).to_numpy() & keep
            parts[segment].append(pd.DataFrame({
                "osm_score": frame["osm_conf_mean"].to_numpy(dtype = float)[mask],
                "overture_score": frame["overture_confidence"].to_numpy(
                    dtype = float)[mask],
            }))
    rng = np.random.default_rng(POPULATION_SEED)
    out = {}
    for segment, frames in parts.items():
        frame = pd.concat(frames, ignore_index = True)
        if len(frame) > MAX_POPULATION:
            frame = frame.iloc[
                np.sort(rng.choice(len(frame), MAX_POPULATION, replace = False))
            ].reset_index(drop = True)
        out[segment] = cf.round_scores(frame)
    return out


def _curve_truth(result: dict):
    """Truth = the fitted point-estimate curve evaluated at the index."""
    grid, curve, params = result["grid"], result["curve"], result["index"]
    mode, segment = result["index_mode"], result["segment"]

    def truth(frame: pd.DataFrame) -> np.ndarray:
        index = cf.segment_scores(frame, segment, params,
                                  mode if mode != "native" else "pool")
        return np.interp(index, grid, curve)

    return truth


def build_truths(rows_by_segment: dict, fit_config) -> dict:
    """``{name: (segment, truth_fn)}`` from full-data fits on the round."""
    cfg = replace(fit_config, bootstrap_reps = 5)
    matched = rows_by_segment["matched"]
    truths = {}
    for mode in ("pool", "interaction"):
        result = cf.fit_segment(matched, "matched",
                                replace(cfg, matched_index_mode = mode),
                                cross_fit = False)
        truths[mode] = ("matched", _curve_truth(result))
    # Overture-only matched truth: the matched rows' 1-D curve in the
    # Overture score alone.
    result = cf.fit_segment(matched, "overture", cfg, cross_fit = False)
    truths["overture_only"] = ("matched", _curve_truth(result))
    for segment in ("osm", "overture"):
        result = cf.fit_segment(rows_by_segment[segment], segment, cfg,
                                cross_fit = False)
        truths[f"{segment}_1d"] = (segment, _curve_truth(result))
    return truths


def class_given_y(rows: pd.DataFrame, fit_config) -> dict:
    """Design-weighted ``P(refined class | Y)`` on the segment's gold."""
    classes = cf.merge_thin_cells(
        cf.refined_class(rows, refine = fit_config.refine_by_confidence),
        rows["gold"].to_numpy(dtype = bool), fit_config.min_cell_gold,
    )
    inclusion = cf.inclusion_by_class(classes, rows["gold"].to_numpy(dtype = bool))
    gold = rows["gold"].to_numpy(dtype = bool)
    frame = pd.DataFrame({
        "cls": classes.to_numpy()[gold],
        "y": rows["y"].to_numpy(dtype = float)[gold],
        "w": classes[gold].map(lambda c: inclusion[c]["weight"]).to_numpy(),
    })
    out = {}
    for y in (0.0, 1.0):
        sub = frame[frame["y"] == y].groupby("cls")["w"].sum()
        out[y] = (sub.index.to_numpy(), (sub / sub.sum()).to_numpy())
    verdicts = rows["llm_verdict"].astype(str)
    rates = {
        v: float(gold[(verdicts == v).to_numpy()].mean())
        for v in verdicts.unique()
    }
    return {"p_class": out, "inclusion": rates}


def _init_worker(out_dir: str, reps: int, seed: int, truths: tuple = None,
                 modes: tuple = None, band_aggregation: str = "bin") -> None:
    config = Config("~/repos/openpois/config.yaml")
    knobs = config.get("conflation", "calibration")
    fit_config = cf.FitConfig(
        min_cell_gold = int(knobs["min_cell_gold"]),
        grid_points = int(knobs["grid_points"]),
        output_bins = int(knobs["curve_output_bins"]),
        bootstrap_reps = reps,
        band_alpha = float(knobs["band_alpha"]),
        rng_seed = seed,
        refine_by_confidence = bool(knobs["refine_by_confidence"]),
        band_aggregation = band_aggregation,
    )
    validation_rows = pd.read_parquet(
        config.get_file_path("calibration", "validation_rows")
    )
    rows = {s: _segment_rows(validation_rows, s) for s in cf.SEGMENTS}
    cache = Path(out_dir) / "coverage_population.parquet"
    if cache.exists():
        stacked = pd.read_parquet(cache)
        population = {s: stacked[stacked["segment"] == s][
            ["osm_score", "overture_score"]].reset_index(drop = True)
            for s in cf.SEGMENTS}
    else:
        with open(config.get_file_path("calibration", "metadata"),
                  encoding = "utf-8") as handle:
            version = json.load(handle)["conflation_version"]
        path = (config.get_dir_path("conflation").parent / str(version)
                / "conflated_cd.parquet")
        population = load_population(path)
    all_truths = build_truths(rows, fit_config)
    _STATE.update({
        "fit_config": fit_config,
        "rows": rows,
        "population": population,
        "truths": {k: v for k, v in all_truths.items()
                   if not truths or k in truths},
        "modes": modes,
        "seed": seed,
        "design": {s: class_given_y(rows[s], fit_config) for s in cf.SEGMENTS},
        "out_dir": Path(out_dir),
    })


# ---------------------------------------------------------------------------
# One simulation
# ---------------------------------------------------------------------------

def simulate_round(rows: pd.DataFrame, truth_values: np.ndarray, design: dict,
                   rng: np.random.Generator) -> pd.DataFrame:
    """One synthetic validation round on the real phase-1 scores."""
    n = len(rows)
    y = (rng.random(n) < truth_values).astype(float)
    cls = np.empty(n, dtype = object)
    for value in (0.0, 1.0):
        names, probs = design["p_class"][value]
        at = np.flatnonzero(y == value)
        cls[at] = rng.choice(names, size = len(at), p = probs)
    split = pd.Series(cls).str.split(":", n = 1, expand = True)
    verdict = split[0].to_numpy()
    confidence = (split[1] if split.shape[1] > 1
                  else pd.Series([None] * n)).fillna("merged").to_numpy()
    gold = np.zeros(n, dtype = bool)
    for v, rate in design["inclusion"].items():
        at = np.flatnonzero(verdict == v)
        if v == "unverifiable" or rate >= 1.0:
            gold[at] = True
            continue
        k = int(round(rate * len(at)))
        if k:
            gold[rng.choice(at, size = min(k, len(at)), replace = False)] = True
    return rows.assign(llm_verdict = verdict, llm_confidence = confidence,
                       gold = gold, y = np.where(gold, y, np.nan))


def _label_regions(target: np.ndarray, shape: tuple = None) -> np.ndarray:
    """Region per published bin or cell, from the target vector."""
    flat = np.asarray(target, dtype = float).ravel()
    labels = np.full(len(flat), "interior", dtype = object)
    if shape is None:
        neighbours = [[j for j in (i - 1, i + 1) if 0 <= j < len(flat)]
                      for i in range(len(flat))]
        edge = np.zeros(len(flat), dtype = bool)
        edge[[0, -1]] = True
        corner = np.zeros(len(flat), dtype = bool)
    else:
        rows, cols = shape
        neighbours, edge, corner = [], [], []
        for r in range(rows):
            for c in range(cols):
                neighbours.append([
                    rr * cols + cc
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1),
                                   (r, c + 1))
                    if 0 <= rr < rows and 0 <= cc < cols
                ])
                edge.append(r in (0, rows - 1) or c in (0, cols - 1))
                corner.append(r in (0, rows - 1) and c in (0, cols - 1))
        edge, corner = np.array(edge), np.array(corner)
    labels[edge] = "edge"
    labels[corner] = "corner"
    floor = flat <= np.nanmin(flat) + 0.01
    labels[floor] = "floor"
    for i, nbrs in enumerate(neighbours):
        if flat[i] >= 0.95 and all(abs(flat[i] - flat[j]) < 0.01
                                   for j in nbrs):
            labels[i] = "plateau"
    return labels


def _index_mode_records(result: dict, population: pd.DataFrame,
                        truth_pop: np.ndarray, segment: str) -> pd.DataFrame:
    lookup = result["lookup"]
    mode = result["index_mode"] if result["index_mode"] != "native" else "pool"
    index = cf.segment_scores(population, segment, result["index"], mode)
    edges = lookup["score_lo"].to_numpy()
    bins = np.clip(np.searchsorted(edges, index, side = "right") - 1, 0,
                   len(lookup) - 1)
    mass = np.bincount(bins, minlength = len(lookup)).astype(float)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        target = np.bincount(bins, weights = truth_pop,
                             minlength = len(lookup)) / mass
    return pd.DataFrame({
        "cell": np.arange(len(lookup)),
        "target": target,
        "estimate": lookup["conf_mean"].to_numpy(),
        "lower": lookup["conf_lower"].to_numpy(),
        "upper": lookup["conf_upper"].to_numpy(),
        "mass": mass / mass.sum(),
        "region": _label_regions(target),
        "band_method": "percentile",
    })


def _surface_records(result: dict, population: pd.DataFrame,
                     truth_pop: np.ndarray) -> pd.DataFrame:
    surface = result["surface"]
    shape = surface["shape"]
    i_osm, i_ov = cf.surface_cells(population["osm_score"],
                                   population["overture_score"],
                                   surface["edges"])
    flat = i_osm * shape[1] + i_ov
    mass = np.bincount(flat, minlength = shape[0] * shape[1]).astype(float)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        target = np.bincount(flat, weights = truth_pop,
                             minlength = len(mass)) / mass
    regions = _label_regions(target, shape)
    frames = []
    for method, band in surface["bands"].items():
        frames.append(pd.DataFrame({
            "cell": np.arange(len(mass)),
            "target": target,
            "estimate": surface["theta"].ravel(),
            "lower": band["lower"].ravel(),
            "upper": band["upper"].ravel(),
            "mass": mass / mass.sum(),
            "region": regions,
            "band_method": method,
        }))
    return pd.concat(frames, ignore_index = True)


def run_simulation(sim: int) -> str:
    """Every truth x mode for one simulated round; writes its parquet."""
    path = _STATE["out_dir"] / "coverage_sims" / f"sim_{sim:04d}.parquet"
    if path.exists():
        return f"sim {sim}: exists"
    started = time.time()
    fit_config = replace(_STATE["fit_config"],
                         rng_seed = _STATE["fit_config"].rng_seed + sim)
    # The data draw depends on --seed too, so a new seed is a new study.
    rng = np.random.default_rng([POPULATION_SEED, _STATE["seed"], sim])
    records = []
    for truth_name, (segment, truth) in _STATE["truths"].items():
        rows = _STATE["rows"][segment]
        population = _STATE["population"][segment]
        truth_pop = truth(population)
        synthetic = simulate_round(rows, truth(rows), _STATE["design"][segment],
                                   rng)
        modes = MATCHED_MODES if segment == "matched" else ("native",)
        if _STATE["modes"] and segment == "matched":
            modes = tuple(m for m in modes if m in _STATE["modes"])
        for mode in modes:
            cfg = replace(fit_config, matched_index_mode = (
                mode if mode != "native" else "pool"))
            try:
                result = cf.fit_segment(synthetic, segment, cfg,
                                        population = population,
                                        cross_fit = False)
            except (ValueError, RuntimeError) as err:
                records.append(pd.DataFrame({
                    "truth": [truth_name], "mode": [mode],
                    "error": [str(err)[:200]],
                }))
                continue
            if mode == "surface":
                frame = _surface_records(result, population, truth_pop)
            else:
                frame = _index_mode_records(result, population, truth_pop,
                                            segment)
            records.append(frame.assign(truth = truth_name, mode = mode,
                                        error = None))
    out = pd.concat(records, ignore_index = True).assign(sim = sim)
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index = False)
    tmp.rename(path)
    return f"sim {sim}: {time.time() - started:.0f}s"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(out_dir: Path) -> Path:
    files = sorted((out_dir / "coverage_sims").glob("sim_*.parquet"))
    if not files:
        raise SystemExit("No simulation results yet")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index = True)
    errors = frame[frame["error"].notna()]
    frame = frame[frame["error"].isna() & frame["target"].notna()].copy()
    frame["covered"] = ((frame["lower"] <= frame["target"] + 1e-12)
                        & (frame["target"] <= frame["upper"] + 1e-12))
    frame["width"] = frame["upper"] - frame["lower"]
    frame["error_est"] = frame["estimate"] - frame["target"]
    keys = ["truth", "mode", "band_method"]

    overall = frame.groupby(keys).apply(lambda g: pd.Series({
        "coverage": g["covered"].mean(),
        "coverage_mass": np.average(g["covered"], weights = g["mass"]),
        "median_width": g["width"].median(),
        "mean_width_mass": np.average(g["width"], weights = g["mass"]),
        "bias_mass": np.average(g["error_est"], weights = g["mass"]),
        "rmse_mass": np.sqrt(np.average(g["error_est"] ** 2,
                                        weights = g["mass"])),
        "n_sims": g["sim"].nunique(),
    }), include_groups = False).reset_index()
    simultaneous = frame.groupby(keys + ["sim"])["covered"].all().groupby(
        keys).mean().rename("simultaneous").reset_index()
    overall = overall.merge(simultaneous, on = keys)
    by_region = frame.groupby(keys + ["region"]).agg(
        coverage = ("covered", "mean"), median_width = ("width", "median"),
        n = ("covered", "size"),
    ).reset_index()

    summary_path = out_dir / "band_coverage.parquet"
    overall.to_parquet(summary_path, index = False)
    by_region.to_parquet(out_dir / "band_coverage_by_region.parquet",
                         index = False)

    lines = [
        "# Band coverage simulation", "",
        f"- Simulations: {frame['sim'].nunique()}; fit failures: "
        f"{len(errors)}",
        "- Nominal pointwise coverage 0.95. `coverage` averages over bins or "
        "cells; `coverage_mass` weights by production mass. `simultaneous` "
        "is the share of simulations with every bin/cell covered. Bias and "
        "RMSE are of the point estimate against the target, mass-weighted.",
        "",
        "| truth | mode | band | coverage | mass-wtd | simultaneous | median "
        "width | mass-wtd width | bias | RMSE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in overall.itertuples():
        lines.append(
            f"| {row.truth} | {row.mode} | {row.band_method} | "
            f"{row.coverage:.3f} | {row.coverage_mass:.3f} | "
            f"{row.simultaneous:.3f} | {row.median_width:.3f} | "
            f"{row.mean_width_mass:.3f} | {row.bias_mass:+.4f} | "
            f"{row.rmse_mass:.4f} |"
        )
    lines += ["", "## Coverage by region", "",
              "| truth | mode | band | region | coverage | median width | n |",
              "|---|---|---|---|---|---|---|"]
    for row in by_region.itertuples():
        lines.append(
            f"| {row.truth} | {row.mode} | {row.band_method} | {row.region} | "
            f"{row.coverage:.3f} | {row.median_width:.3f} | {row.n} |"
        )
    if len(errors):
        lines += ["", "## Fit failures", ""]
        for row in errors.groupby(["truth", "mode"])["error"].first() \
                .reset_index().itertuples():
            lines.append(f"- {row.truth} / {row.mode}: {row.error}")
    report = out_dir / "band_coverage.md"
    report.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--sims", type = int, default = 200)
    parser.add_argument("--reps", type = int, default = 200,
                        help = "Bootstrap replicates per fit (default 200).")
    parser.add_argument("--seed", type = int, default = 20260925)
    parser.add_argument("--out-dir", required = True)
    parser.add_argument("--workers", type = int, default = 10)
    parser.add_argument("--time-one", action = "store_true",
                        help = "Run simulation 0 serially and report timing.")
    parser.add_argument("--aggregate-only", action = "store_true")
    parser.add_argument("--truths", default = "",
                        help = "Comma-separated subset of truths (default all).")
    parser.add_argument("--modes", default = "",
                        help = "Comma-separated subset of matched modes.")
    parser.add_argument("--band-aggregation", default = "bin",
                        choices = ["anchored_kernel", "bin"],
                        help = "1-D band aggregation (FitConfig).")
    args = parser.parse_args()
    truths = tuple(t for t in args.truths.split(",") if t)
    modes = tuple(m for m in args.modes.split(",") if m)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents = True, exist_ok = True)

    if args.aggregate_only:
        print(f"Wrote {aggregate(out_dir)}")
        return

    # Cache the subsampled populations once so workers skip the parquet scan.
    # Each run must use its own --out-dir: sim files are keyed by index only.
    cache = out_dir / "coverage_population.parquet"
    if not cache.exists():
        config = Config("~/repos/openpois/config.yaml")
        with open(config.get_file_path("calibration", "metadata"),
                  encoding = "utf-8") as handle:
            version = json.load(handle)["conflation_version"]
        path = (config.get_dir_path("conflation").parent / str(version)
                / "conflated_cd.parquet")
        population = load_population(path)
        pd.concat([f.assign(segment = s) for s, f in population.items()],
                  ignore_index = True).to_parquet(cache, index = False)
        print(f"Cached populations from {path}")

    if args.time_one:
        started = time.time()
        _init_worker(str(out_dir), args.reps, args.seed, truths, modes,
                     args.band_aggregation)
        print(f"setup {time.time() - started:.0f}s")
        print(run_simulation(0))
        return

    started = time.time()
    done = 0
    with ProcessPoolExecutor(
        max_workers = args.workers, initializer = _init_worker,
        initargs = (str(out_dir), args.reps, args.seed, truths, modes,
                    args.band_aggregation),
    ) as pool:
        futures = [pool.submit(run_simulation, sim) for sim in range(args.sims)]
        for future in as_completed(futures):
            done += 1
            print(f"[{time.strftime('%H:%M:%S')}] {future.result()} "
                  f"({done}/{args.sims}, {time.time() - started:.0f}s)",
                  flush = True)
    print(f"Wrote {aggregate(out_dir)}")
    print("COVERAGE_DONE")


if __name__ == "__main__":
    main()
