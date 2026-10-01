"""Export the production Bayesian calibration fits as deployable grid curves.

Reads the three single-segment fixed-rate mixture fits that ``run_bayes_phase1.sh
MODE=mixture`` writes under ``<eval dir>/fits/<tag>/`` (default tags
``mixture_overture``, ``mixture_osm``, ``mixture_matched``) and writes, into the
deployed calibration directory (default ``conflation/<version>/calibration/``):

- ``osm_curve.parquet``, ``overture_curve.parquet``: ``segment, score, conf_mean,
  conf_lower, conf_upper`` on an evenly spaced score grid over [0, 1]
- ``matched_curve.parquet``: ``segment, osm_score, overture_score, conf_mean,
  conf_lower, conf_upper`` on the full rectangular grid, row-major with
  ``osm_score`` the outer axis
- ``<segment>_metadata.json``: method, lookup, grid size, tag, rounds, forward
  rates, acceptance, diagnostics, knots, fit time and git state
- ``fit_report.md``: per-segment acceptance and diagnostics, forward rates, the
  deployed-impact table, and the name of the HT review PDF

``conf_mean`` is the posterior mean of each grid node and the band its 2.5% and
97.5% posterior quantiles. Every draw is monotone, so the pointwise mean and
quantiles are too; the export asserts it before writing.

The acceptance gate: if a tag is missing, is not an arm C fixed-mixture fit, or
fails the §5.1 acceptance rule (``acceptance["all"]`` false), the export prints the
failing items, writes nothing and exits with code 2. ``--allow-unaccepted`` is for
testing only (short local chains): it exports failed fits and records
``"accepted": false`` in their metadata. An out-dir that already holds curves is
refused (exit code 3) unless ``--overwrite`` is given.

Usage::

    python -u scripts/conflation/export_bayes_curves.py
    python -u scripts/conflation/export_bayes_curves.py --eval-dir <scratch> \\
        --out-dir <scratch>/curves --allow-unaccepted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bayes_calibration_common as common  # noqa: E402

from openpois.conflation import calibration_bayes as cb  # noqa: E402

DEFAULT_TAGS = ("mixture_overture", "mixture_osm", "mixture_matched")
DEFAULT_EVAL_NAME = "calibration_bayes"
GRID_1D = 2001
GRID_2D = 201
SCORE_DECIMALS = 6
METHOD = "bayes_fixed_mixture"
# Grid nodes per curve_draws call on the matched surface: bounds the (draws x
# points) block to about 20M values at 4 x 1,000 draws.
CHUNK_POINTS = 5000
MONOTONE_TOL = 1e-9
BAND = (0.025, 0.975)
EXIT_GATE = 2
EXIT_OVERWRITE = 3

# evaluate(segment, osm, overture) -> (draws, points) posterior draws of m_g(s).
Evaluator = Callable[[str, np.ndarray, np.ndarray], np.ndarray]


class ExportRefused(Exception):
    """The export wrote nothing; ``code`` is the exit status."""

    def __init__(self, message: str, code: int = EXIT_GATE):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def load_summaries(eval_dir: Path, tags: tuple) -> dict:
    """{tag: summary.json contents, or None when the fit or its draws are missing}."""
    out = {}
    for tag in tags:
        fit_dir = Path(eval_dir) / "fits" / tag
        path = fit_dir / "summary.json"
        if path.exists() and (fit_dir / "draws.npz").exists():
            out[tag] = json.loads(path.read_text())
        else:
            out[tag] = None
    return out


def fit_segments(summary: dict) -> tuple:
    """The segments a fit modelled; summaries without the key are joint fits."""
    return tuple(summary.get("segments") or cb.SEGMENT_ORDER)


def gate(summaries: dict, allow_unaccepted: bool = False) -> tuple:
    """(problems, unaccepted, {segment: tag}) for the requested fits.

    ``problems`` stop the export. A failed acceptance rule is a problem unless
    ``allow_unaccepted``; either way it is listed in ``unaccepted``.
    """
    problems, unaccepted, segment_tag = [], [], {}
    for tag, summary in summaries.items():
        if summary is None:
            problems.append(f"{tag}: missing (no fits/{tag}/summary.json and "
                            f"draws.npz)")
            continue
        spec = spec_of(summary)
        if spec.arm != "C" or spec.label_noise != "fixed_mixture":
            problems.append(f"{tag}: not a fixed-rate mixture fit (arm {spec.arm}, "
                            f"label_noise {spec.label_noise})")
        for segment in fit_segments(summary):
            if segment in segment_tag:
                problems.append(f"{tag}: {segment} is already exported from "
                                f"{segment_tag[segment]}")
            segment_tag.setdefault(segment, tag)
        acceptance = summary.get("acceptance") or {}
        if not acceptance.get("all", False):
            failing = [k for k, v in acceptance.items() if k != "all" and not v]
            line = (f"{tag}: acceptance FAIL ("
                    + (", ".join(failing) or "no acceptance recorded") + ")")
            unaccepted.append(line)
            if not allow_unaccepted:
                problems.append(line)
    if summaries and all(s is not None for s in summaries.values()):
        for segment in cb.SEGMENT_ORDER:
            if segment not in segment_tag:
                problems.append(f"{segment}: no fit among {list(summaries)}")
    return problems, unaccepted, segment_tag


def spec_of(summary: dict) -> cb.ModelSpec:
    """The fit's exact spec, rebuilt from the repr saved in its summary."""
    return eval(summary["spec"], {"ModelSpec": cb.ModelSpec,
                                  "PriorConfig": cb.PriorConfig})


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def grid_axis(n_points: int) -> np.ndarray:
    """``n_points`` evenly spaced scores on [0, 1], rounded to the lookup's 6 dp."""
    return np.round(np.linspace(0.0, 1.0, int(n_points)), SCORE_DECIMALS)


def summarize_points(evaluate: Evaluator, segment: str, osm: np.ndarray,
                     overture: np.ndarray,
                     chunk_points: int = CHUNK_POINTS) -> tuple:
    """Posterior mean and 95% quantile band at each point, in point chunks."""
    n = len(osm)
    mean, lower, upper = (np.empty(n) for _ in range(3))
    for start in range(0, n, chunk_points):
        stop = min(start + chunk_points, n)
        values = np.asarray(evaluate(segment, osm[start:stop],
                                     overture[start:stop]), dtype = float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{segment}: non-finite curve draws on the grid")
        mean[start:stop] = values.mean(axis = 0)
        lower[start:stop], upper[start:stop] = np.quantile(values, BAND, axis = 0)
    return mean, lower, upper


def assert_monotone(values: np.ndarray, name: str,
                    tol: float = MONOTONE_TOL) -> None:
    """Non-decreasing along every axis of ``values`` (to ``tol``)."""
    for axis in range(values.ndim):
        worst = float(np.min(np.diff(values, axis = axis), initial = 0.0))
        if worst < -tol:
            raise ValueError(f"{name}: decreases by {-worst:.3g} along axis {axis}")


def curve_frame(segment: str, evaluate: Evaluator, grid_1d: int = GRID_1D,
                grid_2d: int = GRID_2D,
                chunk_points: int = CHUNK_POINTS) -> pd.DataFrame:
    """One segment's grid curve in the deploy schema, checked monotone."""
    if segment in cb.ONE_D_SEGMENTS:
        score = grid_axis(grid_1d)
        stats = summarize_points(evaluate, segment, score, score, chunk_points)
        shape = (len(score),)
        frame = pd.DataFrame({"segment": segment, "score": score})
    else:
        axis = grid_axis(grid_2d)
        osm, overture = (g.ravel() for g in np.meshgrid(axis, axis, indexing = "ij"))
        stats = summarize_points(evaluate, segment, osm, overture, chunk_points)
        shape = (len(axis), len(axis))
        frame = pd.DataFrame({"segment": segment, "osm_score": osm,
                              "overture_score": overture})
    for name, values in zip(("conf_mean", "conf_lower", "conf_upper"), stats):
        assert_monotone(values.reshape(shape), f"{segment} {name}")
        frame[name] = np.clip(values, 0.0, 1.0).astype("float64")
    frame["segment"] = frame["segment"].astype(str)
    return frame


# ---------------------------------------------------------------------------
# Metadata and report
# ---------------------------------------------------------------------------

def segment_knots(summary: dict, segment: str) -> dict | list:
    knots = summary.get("knots") or {}
    if segment == "matched":
        return {k: knots.get(k) for k in ("matched_x", "matched_y")}
    return knots.get(segment)


def segment_metadata(segment: str, tag: str, summary: dict, grid_1d: int,
                     grid_2d: int, accepted: bool, eval_dir: Path) -> dict:
    rounds = summary.get("rounds") or []
    meta = {
        "segment": segment,
        "method": METHOD,
        "lookup": "grid",
        "score_decimals": SCORE_DECIMALS,
        "grid_points": [grid_2d, grid_2d] if segment == "matched" else grid_1d,
        "band": "posterior 2.5% and 97.5% quantiles",
        "accepted": bool(accepted),
        "tag": tag,
        "eval_dir": str(eval_dir),
        "spec": summary.get("spec"),
        "rounds": rounds,
        "validation_round": rounds[0] if rounds else None,
        "forward_rates": (summary.get("forward_rates") or {}).get(segment),
        "acceptance": summary.get("acceptance"),
        "diagnostics": {**(summary.get("diagnostics") or {}),
                        "curve_convergence": summary.get("curve_convergence")},
        "knots": segment_knots(summary, segment),
        "fit_minutes": summary.get("fit_minutes"),
        "git": summary.get("git"),
        "export_git": common.git_state(),
    }
    if segment == "matched":
        meta["index_mode"] = "grid"
    return meta


def fit_report(metadata: dict, summaries: dict, unaccepted: list) -> str:
    """``fit_report.md`` for the exported curves."""
    first = next(iter(metadata.values()))
    round_id = first.get("validation_round") or "<round>"
    lines = ["# Existence-confidence calibration: Bayesian fixed-rate mixture", "",
             f"Three monotone-spline fits, one per segment, with the fixed-rate "
             f"mixture label layer. Rounds (current first): "
             f"{', '.join(first.get('rounds') or []) or 'unrecorded'}. Bands are "
             f"the posterior 2.5% and 97.5% quantiles at each grid node.", ""]
    if unaccepted:
        lines += ["**Exported with `--allow-unaccepted` (testing only). Not for "
                  "release:**", ""] + [f"- {line}" for line in unaccepted] + [""]
    lines += ["## Fits and acceptance (§5.1)", "",
              "| segment | tag | grid | fit min | max R̂ | min ESS bulk / tail | "
              "curve max R̂ | curve min ESS | divergences | min E-BFMI | "
              "tree-depth hits | acceptance |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for segment, meta in metadata.items():
        d = meta["diagnostics"]
        cc = d.get("curve_convergence") or {}
        acc = meta["acceptance"] or {}
        failing = [k for k, v in acc.items() if k != "all" and not v]
        grid = meta["grid_points"]
        grid = " × ".join(map(str, grid)) if isinstance(grid, list) else str(grid)
        lines.append(
            f"| {segment} | {meta['tag']} | {grid} | "
            f"{meta['fit_minutes'] or float('nan'):.1f} | "
            f"{d['max_rhat']:.4f} | {d['min_ess_bulk']:.0f} / "
            f"{d['min_ess_tail']:.0f} | {cc.get('max_rhat', float('nan')):.4f} | "
            f"{min(cc.get('min_ess_bulk', 0), cc.get('min_ess_tail', 0)):.0f} | "
            f"{sum(d['divergences_per_chain'])} | {min(d['ebfmi_per_chain']):.2f} | "
            f"{d['treedepth_saturated']} | "
            + ("PASS" if acc.get("all") else "FAIL (" + ", ".join(failing) + ")")
            + " |")
    lines += ["", "## Forward rates (fixed in the mixture)", ""]
    rates = {segment: meta["forward_rates"] for segment, meta in metadata.items()
             if meta["forward_rates"]}
    if rates:
        lines += common.forward_rate_table(rates)
    impact = {}
    for segment, meta in metadata.items():
        entry = ((summaries[meta["tag"]].get("deployed_impact") or {})
                 .get("segments", {}).get(segment))
        if entry:
            impact[segment] = entry
    if impact:
        lines += ["", "## Deployed-impact preview (unflagged rows of the comparison "
                  "release)", "",
                  "| segment | n | mean Bayes | mean published | mean abs Δ vs "
                  "published | share abs Δ > 0.05 | share abs Δ > 0.10 |",
                  "|---|---|---|---|---|---|---|"]
        for segment, v in impact.items():
            lines.append(f"| {segment} | {v['n']:,} | {v['mean_bayes']:.3f} | "
                         f"{v['mean_published']:.3f} | "
                         f"{v['mean_abs_vs_published']:.3f} | "
                         f"{v['share_gt_0.05_vs_published']:.1%} | "
                         f"{v['share_gt_0.10_vs_published']:.1%} |")
    lines += ["", "## Files", "",
              "- `<segment>_curve.parquet`, `<segment>_metadata.json`: the grid "
              "curves the deploy step applies (`lookup: grid`).",
              f"- `ht_review_{round_id}.pdf`: the design-weighted review of the "
              f"deployed map (written later by `ht_review.py`).",
              f"- Fit outputs: `{first['eval_dir']}/fits/<tag>/`.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def existing_curves(out_dir: Path) -> list:
    out_dir = Path(out_dir)
    return sorted(p.name for p in out_dir.glob("*_curve.parquet")) \
        if out_dir.is_dir() else []


def fit_evaluator(eval_dir: Path, tag: str, summary: dict, config = None) -> Evaluator:
    """Rebuild a fit's prepared data and draws, as report_bayes_calibration does."""
    import jax.numpy as jnp

    config = config or common.load_config()
    fit_config = common.fit_config_from(config)
    rows = common.fit_rows(config, eval_dir, tag)
    prepared = cb.prepare_data(rows, spec_of(summary), fit_config = fit_config,
                               silver_rates = summary.get("silver_rates"),
                               forward_rates = summary.get("forward_rates"))
    saved = np.load(Path(eval_dir) / "fits" / tag / "draws.npz")
    draws = {k: jnp.asarray(saved[k].reshape((-1,) + saved[k].shape[2:]))
             for k in saved.files}

    def evaluate(segment, osm, overture):
        return cb.curve_draws(draws, prepared, segment, osm = osm,
                              overture = overture)

    return evaluate


def export(eval_dir: Path, out_dir: Path, tags: tuple = DEFAULT_TAGS,
           grid_1d: int = GRID_1D, grid_2d: int = GRID_2D,
           overwrite: bool = False, allow_unaccepted: bool = False,
           evaluator_factory: Callable = None,
           chunk_points: int = CHUNK_POINTS) -> dict:
    """Gate, evaluate and write; returns {segment: metadata}.

    Raises ``ExportRefused`` (nothing written) on a failed gate or an existing
    export without ``overwrite``. Every curve is computed and checked before the
    first file is written.
    """
    eval_dir, out_dir = Path(eval_dir).expanduser(), Path(out_dir).expanduser()
    summaries = load_summaries(eval_dir, tags)
    problems, unaccepted, segment_tag = gate(summaries, allow_unaccepted)
    if problems:
        raise ExportRefused("export refused, nothing written:\n  "
                            + "\n  ".join(problems))
    present = existing_curves(out_dir)
    if present and not overwrite:
        raise ExportRefused(f"{out_dir} already holds {present}; pass --overwrite "
                            f"to replace them", EXIT_OVERWRITE)
    if unaccepted:
        print("WARNING --allow-unaccepted (testing only): exporting failed fits:\n  "
              + "\n  ".join(unaccepted), flush = True)
    evaluator_factory = evaluator_factory or fit_evaluator
    evaluators = {}
    frames, metadata = {}, {}
    for segment in cb.SEGMENT_ORDER:
        tag = segment_tag[segment]
        if tag not in evaluators:
            evaluators[tag] = evaluator_factory(eval_dir, tag, summaries[tag])
        frames[segment] = curve_frame(segment, evaluators[tag], grid_1d, grid_2d,
                                      chunk_points)
        accepted = bool((summaries[tag].get("acceptance") or {}).get("all", False))
        metadata[segment] = segment_metadata(segment, tag, summaries[tag], grid_1d,
                                             grid_2d, accepted, eval_dir)
        print(f"{segment}: {len(frames[segment]):,} grid nodes from {tag} "
              f"(accepted {accepted})", flush = True)
    report = fit_report(metadata, summaries, unaccepted)

    out_dir.mkdir(parents = True, exist_ok = True)
    for segment, frame in frames.items():
        frame.to_parquet(out_dir / f"{segment}_curve.parquet", index = False)
        (out_dir / f"{segment}_metadata.json").write_text(
            json.dumps(metadata[segment], indent = 2, default = str))
    (out_dir / "fit_report.md").write_text(report)
    print(f"wrote {len(frames)} grid curves, metadata and fit_report.md to "
          f"{out_dir}", flush = True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", default = None,
                        help = ("Directory holding fits/<tag>/ (default "
                                "conflation/<version>/calibration_bayes)."))
    parser.add_argument("--out-dir", default = None,
                        help = "Deploy directory (default conflation/<version>/"
                               "calibration).")
    parser.add_argument("--tags", default = ",".join(DEFAULT_TAGS),
                        help = "Comma list of fit tags (default %(default)s).")
    parser.add_argument("--grid-1d", type = int, default = None,
                        help = f"1-D grid nodes on [0, 1] (default {GRID_1D}).")
    parser.add_argument("--grid-2d", type = int, default = None,
                        help = f"Matched grid nodes per axis (default {GRID_2D}).")
    parser.add_argument("--overwrite", action = "store_true",
                        help = "Replace curves already in --out-dir.")
    parser.add_argument("--allow-unaccepted", action = "store_true",
                        help = ("TESTING ONLY: export fits that fail the "
                                "acceptance rule; their metadata records "
                                "accepted: false."))
    args = parser.parse_args()

    from openpois.models.jax_core import enable_high_precision

    enable_high_precision()
    config = common.load_config()
    bayes = (config.get("conflation", "calibration") or {}).get("bayes") or {}
    conflation_dir = config.get_dir_path("conflation")
    eval_dir = (Path(args.eval_dir) if args.eval_dir else
                conflation_dir / bayes.get("eval_dir_name", DEFAULT_EVAL_NAME))
    out_dir = Path(args.out_dir) if args.out_dir else conflation_dir / "calibration"
    tags = tuple(t.strip() for t in args.tags.split(",") if t.strip())
    try:
        export(eval_dir, out_dir, tags,
               grid_1d = args.grid_1d or int(bayes.get("grid_1d", GRID_1D)),
               grid_2d = args.grid_2d or int(bayes.get("grid_2d", GRID_2D)),
               overwrite = args.overwrite,
               allow_unaccepted = args.allow_unaccepted,
               evaluator_factory = lambda e, t, s: fit_evaluator(e, t, s, config))
    except ExportRefused as refusal:
        print(str(refusal), file = sys.stderr, flush = True)
        sys.exit(refusal.code)


if __name__ == "__main__":
    main()
