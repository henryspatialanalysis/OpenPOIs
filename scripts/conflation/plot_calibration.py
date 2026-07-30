#!/usr/bin/env python
"""
Diagnostic figures for the existence-confidence calibration.

Produces the "unadjusted vs adjusted" panel per detection segment (the fitted
curve with its 95% band, the validator's Horvitz-Thompson reference curve, the
identity line, and the population score distribution), a design-weighted
reliability diagram from the validation rows, and a before/after distribution
of ``conf_mean`` per segment.

Config keys used (config.yaml):
    conflation.conflated       — calibrated parquet (before/after panel)
    calibration.validation_rows — the validation handoff table
    conflation                 — directory; output PNGs land in its viz/ subdir

Prerequisites:
    scripts/conflation/fit_calibration.py, then apply_calibration.py.

Output files (in conflation/<version>/viz/):
    calibration_curves.png
    calibration_reliability.png
    calibration_shift.png

Usage:
    python scripts/conflation/plot_calibration.py [--input-suffix ""]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from config_versioned import Config

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from openpois.conflation import calibration, calibration_fit  # noqa: E402

# ----------------------------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")
VIZ_DIR = config.get_dir_path("conflation") / "viz"
CURVES_DIR = config.get_dir_path("conflation") / "calibration"

FIGTREE_CANDIDATES = [
    Path("/mnt/c/Users/nathe/AppData/Local/Microsoft/Windows/Fonts/"
         "Figtree-VariableFont_wght.ttf"),
    Path("/mnt/d/Users/Lenovo/AppData/Local/Microsoft/Windows/Fonts/"
         "Figtree-VariableFont_wght.ttf"),
]
for _font in FIGTREE_CANDIDATES:
    if _font.exists():
        fm.fontManager.addfont(str(_font))
        plt.rcParams["font.family"] = "Figtree"
        break
plt.rcParams["font.size"] = 12

# Segment palette shared with plot_source_contributions.py.
SEGMENTS = [
    ("matched", "Both sources", "#3d00a5"),
    ("osm", "OSM only", "#a0d787"),
    ("overture", "Overture only", "#2e86c9"),
]
GRID_COLOR = "#DDDDDD"

# The matched curve is indexed on the fitted pool of both source scores, not on
# a single provider score, so its x-axis needs its own name.
X_LABELS = {
    "matched": "Pooled source index",
    "osm": "OSM turnover posterior",
    "overture": "Overture confidence",
}


def _chrome(ax, xlabel = None, ylabel = None, title = None) -> None:
    """House style: no spines, no tick marks, light gridlines behind."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize = plt.rcParams["font.size"] * 1.15)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(color = GRID_COLOR, linewidth = 1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length = 0)


def curve_index_for_rows(rows: pd.DataFrame, segment: str,
                         metadata: dict) -> np.ndarray:
    """Validation rows' curve-index score, using the fitted pool if any."""
    pool = (metadata.get(segment) or {}).get("pool")
    return calibration_fit.segment_scores(rows, segment, pool)


def plot_curves(curves: dict, metadata: dict, population: dict,
                out_path: Path) -> None:
    """Per-segment raw score -> calibrated probability, with band + reference."""
    fig, axes = plt.subplots(1, len(SEGMENTS), figsize = (13.33, 5.2),
                             sharey = True)
    for ax, (segment, legend, color) in zip(np.atleast_1d(axes), SEGMENTS):
        lookup = curves.get(segment)
        if lookup is None:
            ax.set_visible(False)
            continue
        centers = (lookup["score_lo"].to_numpy()
                   + lookup["score_hi"].to_numpy()) / 2.0

        # Population score distribution behind the curve, on a twin axis so
        # the probability axis keeps its 0-1 scale.
        scores = population.get(segment)
        if scores is not None and len(scores):
            hist_ax = ax.twinx()
            hist_ax.hist(scores, bins = 40, color = color, alpha = 0.15)
            hist_ax.set_yticks([])
            for spine in hist_ax.spines.values():
                spine.set_visible(False)
            hist_ax.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)

        ax.plot([0, 1], [0, 1], color = "#999999", linewidth = 1.0,
                linestyle = (0, (4, 3)), label = "No adjustment")
        ax.fill_between(centers, lookup["conf_lower"], lookup["conf_upper"],
                        color = color, alpha = 0.25, linewidth = 0,
                        label = "95% band")
        ax.plot(centers, lookup["conf_mean"], color = color, linewidth = 2.4,
                label = "Calibrated")

        ref_path = (
            Path(config.get_file_path("calibration", "reference_curves"))
            / f"{segment}_curve.parquet"
        )
        if ref_path.exists():
            ref = pd.read_parquet(ref_path)
            ref_centers = (ref["score_lo"].to_numpy()
                           + ref["score_hi"].to_numpy()) / 2.0
            ax.plot(ref_centers, ref["conf_mean"], color = "#444444",
                    linewidth = 1.3, linestyle = (0, (2, 2)),
                    label = "Gold-only reference")

        # Report BOTH sample sizes: the LLM-verified phase-1 rows supply the
        # class mix that shapes the curve, and the gold rows pin its level. A
        # subtitle showing only gold understates what the fit is built on.
        meta = metadata.get(segment, {})
        ess = meta.get("effective_sample_size")
        subtitle = (f"{legend}\nLLM-verified {meta.get('n_phase1_rows', 0):,}"
                    f" · gold {meta.get('n_gold', 0):,}")
        if ess:
            subtitle += f" · ESS {ess:.0f}"
        _chrome(ax, xlabel = X_LABELS[segment], title = subtitle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    np.atleast_1d(axes)[0].set_ylabel("P(exists and open)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = len(labels),
               frameon = False)
    fig.suptitle("Confidence calibration by detection segment",
                 fontsize = plt.rcParams["font.size"] * 1.44)
    fig.subplots_adjust(left = 0.06, right = 0.98, top = 0.80, bottom = 0.20,
                        wspace = 0.12)
    VIZ_DIR.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)
    print(f"  {out_path}")


def plot_reliability(validation_rows: pd.DataFrame, metadata: dict,
                     out_path: Path, n_bins: int = 8) -> None:
    """Design-weighted observed existence rate vs uncalibrated score."""
    fig, axes = plt.subplots(1, len(SEGMENTS), figsize = (13.33, 5.0),
                             sharey = True)
    for ax, (segment, legend, color) in zip(np.atleast_1d(axes), SEGMENTS):
        rows = validation_rows[
            (validation_rows["segment"] == segment)
            & validation_rows["stratum"].isin(calibration.SEGMENTS)
        ].copy()
        if rows.empty:
            ax.set_visible(False)
            continue
        rows["score"] = curve_index_for_rows(rows, segment, metadata)
        classes = calibration_fit.merge_thin_cells(
            calibration_fit.refined_class(rows),
            rows["gold"].to_numpy(dtype = bool),
            int(config.get("conflation", "calibration", "min_cell_gold")),
        )
        inclusion = calibration_fit.inclusion_by_class(
            classes, rows["gold"].to_numpy(dtype = bool)
        )
        weights = classes.astype(str).map(
            lambda c: inclusion.get(c, {}).get("weight", 0.0)
        ).to_numpy(dtype = float)

        gold = rows["gold"].to_numpy(dtype = bool)
        edges = np.quantile(rows["score"].to_numpy(dtype = float),
                            np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        centers, observed = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = gold & (rows["score"] >= lo).to_numpy() & (
                rows["score"] <= hi
            ).to_numpy() & (weights > 0)
            if in_bin.sum() < 5:
                continue
            w = weights[in_bin]
            y = rows["y"].to_numpy(dtype = float)[in_bin]
            centers.append(float(rows["score"].to_numpy()[in_bin].mean()))
            observed.append(float(np.average(y, weights = w)))

        ax.plot([0, 1], [0, 1], color = "#999999", linewidth = 1.0,
                linestyle = (0, (4, 3)))
        ax.plot(centers, observed, marker = "o", color = color,
                linewidth = 2.0, markersize = 6)
        _chrome(ax, xlabel = X_LABELS[segment], title = legend)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    np.atleast_1d(axes)[0].set_ylabel("Design-weighted existence rate")
    fig.suptitle("Reliability of the uncalibrated score (validation sample)",
                 fontsize = plt.rcParams["font.size"] * 1.44)
    fig.subplots_adjust(left = 0.07, right = 0.98, top = 0.84, bottom = 0.13,
                        wspace = 0.12)
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)
    print(f"  {out_path}")


def plot_shift(conflated_path: Path, out_path: Path) -> None:
    """Before/after distribution of the published confidence, per segment."""
    frame = pd.read_parquet(
        conflated_path,
        columns = ["source", "conf_mean", "conf_mean_uncalibrated"],
    )
    fig, axes = plt.subplots(1, len(SEGMENTS), figsize = (13.33, 5.0),
                             sharey = True)
    bins = np.linspace(0, 1, 41)
    for ax, (segment, legend, color) in zip(np.atleast_1d(axes), SEGMENTS):
        rows = frame[frame["source"] == segment]
        if rows.empty:
            ax.set_visible(False)
            continue
        ax.hist(rows["conf_mean_uncalibrated"].dropna(), bins = bins,
                color = "#999999", alpha = 0.55, label = "Before")
        ax.hist(rows["conf_mean"].dropna(), bins = bins, color = color,
                alpha = 0.75, label = "After")
        before = rows["conf_mean_uncalibrated"].mean()
        after = rows["conf_mean"].mean()
        _chrome(ax, xlabel = "conf_mean",
                title = f"{legend}\nmean {before:.3f} -> {after:.3f}")
        ax.set_xlim(0, 1)

    np.atleast_1d(axes)[0].set_ylabel("POIs")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = 2,
               frameon = False)
    fig.suptitle("Published confidence before and after calibration",
                 fontsize = plt.rcParams["font.size"] * 1.44)
    fig.subplots_adjust(left = 0.07, right = 0.98, top = 0.82, bottom = 0.18,
                        wspace = 0.12)
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)
    print(f"  {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--input-suffix", default = "",
                        help = "Suffix of the calibrated conflated parquet.")
    parser.add_argument("--skip-shift", action = "store_true",
                        help = "Skip the before/after panel (needs the "
                               "calibrated parquet).")
    args = parser.parse_args()

    curves = calibration.read_curves(CURVES_DIR)
    metadata = calibration.read_curve_metadata(CURVES_DIR)
    validation_rows = pd.read_parquet(
        config.get_file_path("calibration", "validation_rows")
    )

    population = {}
    for segment in calibration.SEGMENTS:
        rows = validation_rows[validation_rows["segment"] == segment]
        if len(rows):
            population[segment] = curve_index_for_rows(rows, segment, metadata)

    VIZ_DIR.mkdir(parents = True, exist_ok = True)
    print("Writing figures:")
    plot_curves(curves, metadata, population,
                VIZ_DIR / "calibration_curves.png")
    plot_reliability(validation_rows, metadata,
                     VIZ_DIR / "calibration_reliability.png")

    if not args.skip_shift:
        conflated = config.get_file_path("conflation", "conflated")
        if args.input_suffix:
            conflated = conflated.with_name(
                f"{conflated.stem}_{args.input_suffix}{conflated.suffix}"
            )
        if conflated.exists():
            plot_shift(conflated, VIZ_DIR / "calibration_shift.png")
        else:
            print(f"  (skipped shift panel: {conflated} not found)")


if __name__ == "__main__":
    main()
