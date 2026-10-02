"""Shared helpers for the Bayesian calibration prototype scripts (Phase 1).

Used by ``fit_bayes_calibration.py``, ``cv_bayes_calibration.py`` and
``simulate_bayes_recovery.py``. Nothing here touches the deployed calibration:
every output goes under ``<conflation root>/<round version>/calibration_eval_bayes_*``
and the scripts refuse a deployed ``calibration/`` directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from config_versioned import Config

from openpois.conflation import calibration, calibration_bayes as cb
from openpois.conflation import calibration_fit as cf

CONFIG_PATH = "~/repos/openpois/config.yaml"
DEFAULT_EVAL_NAME = "calibration_eval_bayes_20260927"
# The October 2026 production configuration (interaction index, bin band),
# fit on round 20260730 with bins on the 20260902 population: the curves the
# October run would publish (2026-09-26 dry run).
PRODUCTION_CURVES = Path(
    "~/data/openpois/conflation/20260902/calibration_eval_20260925/"
    "fit_production_config"
).expanduser()
OVERTURE_ATOMS = (0.919912, 0.990219)
# Figure palette: the dataviz skill's validated categorical order (light mode).
COLORS = {"A": "#2a78d6", "production": "#eb6834", "B": "#1baf7a",
          "C": "#eda100", "ink": "#0b0b0b", "muted": "#52514e",
          "surface": "#fcfcfb", "grid": "#e4e3dd"}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
              "#0d366b"]


def load_config() -> Config:
    return Config(CONFIG_PATH)


def fit_config_from(config: Config) -> cf.FitConfig:
    """The production calibration knobs, as the compare harness reads them."""
    knobs = config.get("conflation", "calibration")
    return cf.FitConfig(
        min_cell_gold = int(knobs["min_cell_gold"]),
        grid_points = int(knobs["grid_points"]),
        output_bins = int(knobs["curve_output_bins"]),
        bootstrap_reps = int(knobs["bootstrap_reps"]),
        band_alpha = float(knobs["band_alpha"]),
        rng_seed = int(knobs["rng_seed"]),
        refine_by_confidence = bool(knobs["refine_by_confidence"]),
        matched_index_mode = str(knobs.get("matched_index_mode", "interaction")),
        band_aggregation = str(knobs.get("band_aggregation", "bin")),
        pool_min_coef = float(knobs.get("pool_min_coef", 1e-3)),
    )


def pooled_rounds(config: Config, override: str = None) -> list:
    """Earlier validation rounds pooled into the fit (execution log, decision 23).

    Their phase-1 rows join the curve fit and their gold joins arm C's fixed
    silver-label rates, each round weighted by its own phase-2 design.
    ``override`` (a comma list from ``--pooled-rounds``) wins over the config
    key ``conflation.calibration.pooled_rounds``; "" pools nothing.
    """
    if override is not None:
        return [r.strip() for r in override.split(",") if r.strip()]
    knobs = config.get("conflation", "calibration")
    return [str(r) for r in (knobs.get("pooled_rounds") or [])]


def read_round(config: Config, round_id: str) -> pd.DataFrame:
    """One validation round's usable rows, tagged with its round id."""
    path = (config.get_dir_path("calibration").parent / str(round_id)
            / "validation_rows.parquet")
    if not path.exists():
        raise SystemExit(f"Validation round {round_id}: {path} not found")
    raw = pd.read_parquet(path)
    if "validation_round" in raw and set(raw["validation_round"].astype(str)) \
            != {str(round_id)}:
        raise SystemExit(f"{path}: validation_round does not match {round_id}")
    return raw.assign(validation_round = str(round_id))


def load_handoff(config: Config, override: str = None,
                 rounds: list = None) -> tuple:
    """(usable validation rows, handoff metadata of the current round).

    The rows are the current round (``versions.calibration``) plus every
    pooled round (``pooled_rounds``), concatenated in segment order; the
    ``validation_round`` column says which round a row came from. Pass
    ``rounds`` (a fit summary's ``rounds``) to rebuild an earlier fit's exact
    table; the first entry is then taken as the current round.
    """
    with open(config.get_file_path("calibration", "metadata"),
              encoding = "utf-8") as handle:
        metadata = json.load(handle)
    current = str(metadata["validation_round"])
    if rounds is None:
        rounds = [current] + [r for r in pooled_rounds(config, override)
                              if r != current]
    frames = [read_round(config, r) for r in rounds]
    return cb.usable_rows(pd.concat(frames, ignore_index = True)), metadata


def fit_rows(config: Config, out_dir: Path, tag: str) -> pd.DataFrame:
    """The exact validation table an earlier fit in ``out_dir`` was made on.

    Fits record their rounds (current first) in ``summary.json``. Fits from
    before execution-log decision 23 do not; they were single-round fits of the
    round then in ``versions.calibration``, so the current round alone is
    loaded and checked against the recorded row count.
    """
    summary = json.loads((Path(out_dir) / "fits" / tag / "summary.json").read_text())
    if summary.get("rounds"):
        return load_handoff(config, rounds = summary["rounds"])[0]
    rows = load_handoff(config, override = "")[0]
    if len(rows) != summary["n_rows"]:
        raise SystemExit(
            f"{tag}: fit had {summary['n_rows']} rows, the current round has "
            f"{len(rows)}; set versions.calibration to the fit's round")
    return rows


def current_round_mask(rows: pd.DataFrame, metadata: dict) -> np.ndarray:
    """Rows of the current round: the only ones CV holds out and scores."""
    return (rows["validation_round"].astype(str)
            == str(metadata["validation_round"])).to_numpy()


def conflation_root(config: Config) -> Path:
    """Parent of the versioned conflation directories."""
    return config.get_dir_path("conflation").parent


def eval_dir(config: Config, metadata: dict, out_dir: str = None,
             allow_deployed: bool = False) -> Path:
    """Output directory, refusing any deployed ``calibration/`` directory."""
    root = conflation_root(config)
    path = (Path(out_dir).expanduser() if out_dir else
            root / str(metadata["conflation_version"]) / DEFAULT_EVAL_NAME)
    if path.name == "calibration" and not allow_deployed:
        raise SystemExit(f"{path} looks like a deployed calibration directory")
    for sub in ("", "fits", "figures", "cv", "coverage", "logs"):
        (path / sub).mkdir(parents = True, exist_ok = True)
    return path


def git_state() -> dict:
    """Commit and dirty flag of the repo, recorded with every output."""
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output = True, text = True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output = True, text = True).stdout.strip())
    except OSError:
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


def spec_from_args(args) -> cb.ModelSpec:
    """ModelSpec from the shared CLI flags."""
    priors = cb.PriorConfig()
    if getattr(args, "tau_mult", None):
        priors = replace(priors, tau_scale = priors.tau_scale * args.tau_mult)
    if getattr(args, "beta_prior", None):
        a, b = (float(x) for x in args.beta_prior.split(","))
        priors = replace(priors, beta_label_a = a, beta_label_b = b)
    if getattr(args, "tau_matched_mult", None):
        priors = replace(priors, tau_scale_matched = (
            cb.PriorConfig().tau_scale * args.tau_matched_mult))
    return cb.ModelSpec(
        arm = args.arm,
        differential = not getattr(args, "no_differential", False),
        refined_classes = not getattr(args, "verdict_classes", False),
        class_scheme = getattr(args, "class_scheme", None),
        pool_segments = not getattr(args, "no_pooling", False),
        label_noise = getattr(args, "label_noise", "fixed"),
        degree = int(getattr(args, "degree", 2)),
        max_gap = float(getattr(args, "max_gap", 0.2)),
        equal_knots = getattr(args, "equal_knots", None),
        spacing_scaled_rw = bool(getattr(args, "spacing_scaled_rw", False)),
        smooth_max_t = float(getattr(args, "smooth_max_t", 0.0)),
        segments = parse_segments(getattr(args, "segments", None)),
        priors = priors,
    )


def parse_segments(value: str = None) -> tuple:
    """``--segments`` comma list as a tuple (None or "" means all three)."""
    if not value:
        return cb.SEGMENT_ORDER
    segments = tuple(s.strip() for s in value.split(",") if s.strip())
    unknown = sorted(set(segments) - set(cb.SEGMENT_ORDER))
    if unknown:
        raise SystemExit(f"--segments: unknown {unknown}; choose from "
                         f"{','.join(cb.SEGMENT_ORDER)}")
    return segments


def add_spec_arguments(parser) -> None:
    parser.add_argument("--arm", default = "A", choices = cb.ARMS)
    parser.add_argument("--no-differential", action = "store_true",
                        help = "Drop the measurement layer's score slope (S2).")
    parser.add_argument("--verdict-classes", action = "store_true",
                        help = "3 verdict classes instead of 9 refined (S3).")
    parser.add_argument("--class-scheme", default = None,
                        choices = ["refined9", "merged6", "verdict3"],
                        help = "LLM class scheme (overrides --verdict-classes).")
    parser.add_argument("--label-noise", default = "fixed",
                        choices = list(cb.LABEL_NOISE),
                        help = ("Arm C silver-label noise: fixed rates from gold "
                                "(default); fixed_mixture, the mixture with "
                                "forward rates Se, Sp from gold (October test "
                                "model, design doc §3.5c); symmetric / "
                                "asymmetric estimated (first run, S2C); none "
                                "(S3C-a)."))
    parser.add_argument("--segments", default = None,
                        help = ("Comma list of segments to fit, from "
                                "overture,osm,matched (default all three). Arm C "
                                "with fixed rates factorizes by segment, so "
                                "production fits each one separately."))
    parser.add_argument("--pooled-rounds", default = None,
                        help = ("Comma list of earlier validation rounds pooled "
                                "into the fit and the silver-label rates; '' for "
                                "none (default: config "
                                "conflation.calibration.pooled_rounds)."))
    parser.add_argument("--beta-prior", default = None,
                        help = "Arm C beta_label Beta prior as 'a,b' (S3C-b 9,1).")
    parser.add_argument("--no-pooling", action = "store_true",
                        help = "Fit each segment's measurement layer separately.")
    parser.add_argument("--degree", type = int, default = 2,
                        help = "B-spline degree (3 is S4).")
    parser.add_argument("--max-gap", type = float, default = 0.2,
                        help = "Maximum knot gap (0.1 is S5).")
    parser.add_argument("--equal-knots", type = int, default = None,
                        help = "Equally spaced knot intervals instead (S5).")
    parser.add_argument("--tau-mult", type = float, default = None,
                        help = "Multiply every tau prior scale (S6).")
    parser.add_argument("--tau-matched-mult", type = float, default = None,
                        help = "Multiply only the matched tau prior scale (S6).")
    parser.add_argument("--spacing-scaled-rw", action = "store_true",
                        help = "Spacing-scaled random walk (S7).")
    parser.add_argument("--smooth-max-t", type = float, default = 0.05,
                        help = "Smooth-max temperature for (M9c); "
                               "0 = exact max (decision 9).")
    parser.add_argument("--target-accept", type = float, default = None)
    parser.add_argument("--dense-mass", action = "store_true")
    parser.add_argument("--warmup", type = int, default = 1000)
    parser.add_argument("--samples", type = int, default = 1000)
    parser.add_argument("--chains", type = int, default = 4)
    parser.add_argument("--seed", type = int, default = 20260927)


def adaptation_kwargs_from(args) -> dict | None:
    kwargs = {}
    if getattr(args, "target_accept", None):
        kwargs["target_acceptance_rate"] = float(args.target_accept)
    if getattr(args, "dense_mass", False):
        kwargs["is_mass_matrix_diagonal"] = False
    return kwargs or None


# ---------------------------------------------------------------------------
# Production curves (the comparator)
# ---------------------------------------------------------------------------

def load_production(curves_dir: Path = PRODUCTION_CURVES) -> dict | None:
    """The v4 comparator curves, or None where they are not on disk.

    They live only in the laptop's 20260902 evaluation directory. A production fit
    (openpois-01, from October 2026) has no v4 comparator, so its comparison
    columns come out NaN rather than failing the fit.
    """
    if not Path(curves_dir).is_dir():
        print(f"No v4 comparator curves at {curves_dir}; comparisons are NaN.")
        return None
    return {"curves": calibration.read_curves(curves_dir),
            "metadata": calibration.read_curve_metadata(curves_dir)}


def production_values(production: dict | None, segment: str, osm = None,
                      overture = None) -> np.ndarray:
    """The production lookup's conf_mean at the given raw scores.

    NaN everywhere when there is no comparator (``production`` is None).
    """
    if production is None:
        scores = osm if osm is not None else overture
        return np.full(len(np.asarray(scores)), np.nan)
    curves, metadata = production["curves"], production["metadata"]
    meta = metadata[segment]
    decimals = meta.get("score_decimals")
    osm = None if osm is None else np.asarray(osm, dtype = float)
    overture = None if overture is None else np.asarray(overture, dtype = float)
    if decimals is not None:
        osm = None if osm is None else np.round(osm, decimals)
        overture = None if overture is None else np.round(overture, decimals)
    if segment == "matched":
        index = cf.index_score(osm, overture, meta["index"])
        return calibration.apply_curve(index, curves["matched"])[
            "conf_mean"].to_numpy()
    scores = osm if segment == "osm" else overture
    return calibration.apply_curve(scores, curves[segment])["conf_mean"].to_numpy()


# ---------------------------------------------------------------------------
# Design-weighted reference rates
# ---------------------------------------------------------------------------

def design_weights(rows: pd.DataFrame, fit_config: cf.FitConfig) -> np.ndarray:
    """1 / pi_class for gold rows (production classes), 0 otherwise."""
    classes = cb.production_classes(rows, fit_config)
    gold = rows["gold"].to_numpy(dtype = bool)
    weights = np.zeros(len(rows))
    for segment in cb.SEGMENT_ORDER:
        mask = (rows["segment"] == segment).to_numpy()
        inclusion = cf.inclusion_by_class(
            pd.Series(classes[mask]).reset_index(drop = True), gold[mask]
        )
        w = np.array([inclusion.get(c, {}).get("weight", 0.0)
                      for c in classes[mask]])
        weights[mask] = w * gold[mask]
    return weights


def binned_ht_rates(scores, y, weights, edges) -> pd.DataFrame:
    """Design-weighted (Hajek) gold existence rate per score bin."""
    scores = np.asarray(scores, dtype = float)
    idx = np.clip(np.searchsorted(edges, scores, side = "right") - 1, 0,
                  len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        sel = (idx == b) & (weights > 0)
        if sel.sum() == 0:
            continue
        w, yy = weights[sel], y[sel]
        rate = float(np.sum(w * yy) / np.sum(w))
        ess = float(w.sum() ** 2 / np.sum(w ** 2))
        se = float(np.sqrt(max(rate * (1 - rate), 1e-4) / max(ess, 1.0)))
        out.append({"lo": edges[b], "hi": edges[b + 1],
                    "x": float(np.average(scores[sel], weights = w)),
                    "rate": rate, "se": se, "n_gold": int(sel.sum()),
                    "ess": ess})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

def population_scores(conflated_path: Path) -> dict:
    """Per-segment production source scores (non-shadow), streamed.

    Delegates to ``fit_calibration.population_by_segment`` so bin placement
    matches the production fit exactly.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fit_calibration import population_by_segment  # noqa: E402

    return population_by_segment(conflated_path)


def style_axes(ax) -> None:
    """Recessive grid and axes per the dataviz mark specs."""
    ax.set_facecolor(COLORS["surface"])
    ax.grid(True, color = COLORS["grid"], linewidth = 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors = COLORS["muted"], labelsize = 8)


def forward_rate_table(forward_rates: dict) -> list:
    """Markdown rows of the fixed-rate mixture's forward rates, per segment.

    Rates over all verdicts (``e1`` ... ``g0``, from 2026-10-01) when the
    summary carries them, else the older Se / Sp among definitive verdicts.
    """
    if all("e1" in r for r in forward_rates.values()):
        lines = ["P(verdict | y) over all verdicts (unverifiable included), "
                 "design-weighted and Jeffreys-smoothed on the Kish ESS; "
                 "exists + gone = P(definitive | y).", "",
                 "| segment | P(exists \\| 1) | P(gone \\| 1) | ESS y = 1 | "
                 "P(exists \\| 0) | P(gone \\| 0) | ESS y = 0 |",
                 "|---|---|---|---|---|---|---|"]
        for segment, r in forward_rates.items():
            lines.append(f"| {segment} | {r['e1']:.4f} | {r['g1']:.4f} | "
                         f"{r['ess_1']:.1f} | {r['e0']:.4f} | {r['g0']:.4f} | "
                         f"{r['ess_0']:.1f} |")
        return lines
    lines = ["Se = P(verdict exists | exists), Sp = P(verdict gone | gone), "
             "among definitive verdicts; design-weighted and Jeffreys-smoothed "
             "on the Kish ESS.", "",
             "| segment | Se | raw | ESS | Sp | raw | ESS |",
             "|---|---|---|---|---|---|---|"]
    for segment, r in forward_rates.items():
        lines.append(f"| {segment} | {r['se']:.4f} | {r['raw_se']:.4f} | "
                     f"{r['ess_se']:.1f} | {r['sp']:.4f} | {r['raw_sp']:.4f} | "
                     f"{r['ess_sp']:.1f} |")
    return lines
