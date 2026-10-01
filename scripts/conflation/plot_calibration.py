#!/usr/bin/env python
"""
Diagnostic figures for the existence-confidence calibration.

Produces the "unadjusted vs adjusted" panel per detection segment (the fitted
curve with its 95% band, the validator's Horvitz-Thompson reference curve, the
identity line, and the population score distribution), a design-weighted
reliability diagram from the validation rows, and a before/after distribution
of ``conf_mean`` per segment.

Production curves (October 2026 on) are the Bayesian grid lookups from
``export_bayes_curves.py``: the 1-D panels draw the posterior-mean line with its
95% posterior band, and the matched panel draws OSM slices of the surface at
the two Overture atoms, with a heatmap of the whole surface in its own figure.
v4 step curves still plot as before.

Config keys used (config.yaml):
    conflation.conflated       — calibrated parquet (before/after panel)
    calibration.validation_rows — the validation handoff table
    conflation                 — directory; output PNGs land in its viz/ subdir

Prerequisites:
    scripts/conflation/export_bayes_curves.py, then apply_calibration.py.

Output files (in conflation/<version>/viz/):
    calibration_curves.png
    calibration_matched_surface.png   (grid curves only)
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

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
# The two Overture score atoms (rounded) that hold most matched POIs; the
# matched surface is drawn as OSM slices at each.
OVERTURE_ATOMS = (0.919912, 0.990219)
SLICE_STYLES = ("-", (0, (5, 2)))

# The matched curve is indexed on a fitted combination of both source scores
# (the interaction index in production), not on a single provider score.
X_LABELS = {
    "matched": "Combined source index",
    "osm": "OSM turnover posterior",
    "overture": "Overture confidence",
}
GRID_X_LABELS = {"matched": "OSM turnover posterior (matched POIs)"}
RELIABILITY_X_LABELS = {"matched": "Mean of the two source scores"}


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
    """Validation rows' curve-index score under the segment's fitted index.

    The index form and parameters come from the curve metadata (``index``,
    falling back to ``pool`` for pre-2026-09 curves), so every index mode plots
    on its own index. A ``surface`` curve has no 1-D index; its rows get the
    working-model pool index for the reliability panel. A ``grid`` curve has
    neither, so its rows get the average of the two scores.
    """
    meta = metadata.get(segment) or {}
    rows = calibration_fit.round_scores(rows)
    mode = meta.get("index_mode") or "pool"
    params = meta.get("index") or meta.get("pool")
    if mode == "surface":
        mode, params = "pool", meta.get("working_index")
    elif mode == "grid":
        mode, params = "average", None
    if segment not in calibration_fit.POOLED_SEGMENTS or mode == "native":
        return calibration_fit.segment_scores(rows, segment)
    return calibration_fit.segment_scores(rows, segment, params, mode)


def _rounds_label(meta: dict) -> str:
    rounds = meta.get("rounds") or meta.get("validation_round") or ""
    if isinstance(rounds, (list, tuple)):
        rounds = ", ".join(str(r) for r in rounds)
    return str(rounds)


def _grid_slices(ax, lookup: pd.DataFrame, color: str) -> None:
    """OSM slices of the matched node grid at the two Overture atoms."""
    osm = np.linspace(0.0, 1.0, 401)
    for atom, style in zip(OVERTURE_ATOMS, SLICE_STYLES):
        triple = calibration.apply_grid_surface(
            osm, np.full(len(osm), atom), lookup
        )
        ax.fill_between(osm, triple["conf_lower"], triple["conf_upper"],
                        color = color, alpha = 0.15, linewidth = 0)
        ax.plot(osm, triple["conf_mean"], color = color, linewidth = 2.4,
                linestyle = style, label = f"Calibrated, Overture {atom:g}")


def _grid_subtitle(legend: str, meta: dict, method: bool = False) -> str:
    """Panel title for a grid curve: rounds, and a failed acceptance if any."""
    accepted = (meta.get("acceptance") or {}).get("all")
    rounds = _rounds_label(meta)
    parts = [p for p in ("Bayesian mixture" if method else "",
                         f"rounds {rounds}" if rounds else "",
                         "NOT accepted" if accepted is False else "") if p]
    return f"{legend}\n" + " · ".join(parts)


def plot_curves(curves: dict, metadata: dict, population: dict,
                out_path: Path) -> None:
    """Per-segment raw score -> calibrated probability, with band + reference.

    Grid curves: 1-D segments draw the posterior-mean line and band, the
    matched segment its OSM slices at the Overture atoms. v4 step curves draw
    at their bin midpoints; a v4 surface curve has no 1-D panel here.
    """
    fig, axes = plt.subplots(1, len(SEGMENTS), figsize = (13.33, 5.2),
                             sharey = True)
    for ax, (segment, legend, color) in zip(np.atleast_1d(axes), SEGMENTS):
        lookup = curves.get(segment)
        if lookup is None:
            ax.set_visible(False)
            continue
        meta = metadata.get(segment, {})
        grid = calibration.is_grid_lookup(lookup)
        surface = grid and calibration.is_grid_surface(lookup)
        if grid and not surface:
            centers = lookup["score"].to_numpy(dtype = float)
        elif not grid and not calibration.is_surface_lookup(lookup):
            centers = (lookup["score_lo"].to_numpy()
                       + lookup["score_hi"].to_numpy()) / 2.0
        else:
            centers = None

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
        band_label = "95% posterior band" if grid else "95% band"
        if surface:
            _grid_slices(ax, lookup, color)
        elif centers is not None:
            ax.fill_between(centers, lookup["conf_lower"], lookup["conf_upper"],
                            color = color, alpha = 0.25, linewidth = 0,
                            label = band_label)
            ax.plot(centers, lookup["conf_mean"], color = color,
                    linewidth = 2.4, label = "Calibrated")
        else:
            ax.text(0.5, 0.5, "2-D surface lookup\n(see the HT review)",
                    ha = "center", va = "center", transform = ax.transAxes)

        ref_path = (
            Path(config.get_file_path("calibration", "reference_curves"))
            / f"{segment}_curve.parquet"
        )
        # The matched reference is on the validator's own index, which a 2-D
        # panel's OSM axis does not share.
        if ref_path.exists() and not surface:
            ref = pd.read_parquet(ref_path)
            ref_centers = (ref["score_lo"].to_numpy()
                           + ref["score_hi"].to_numpy()) / 2.0
            ax.plot(ref_centers, ref["conf_mean"], color = "#444444",
                    linewidth = 1.3, linestyle = (0, (2, 2)),
                    label = "Gold-only reference")

        # Report BOTH sample sizes: the LLM-verified phase-1 rows supply the
        # class mix that shapes the curve, and the gold rows pin its level. A
        # subtitle showing only gold understates what the fit is built on.
        if grid:
            subtitle = _grid_subtitle(legend, meta)
        else:
            ess = meta.get("effective_sample_size")
            subtitle = (f"{legend}\nLLM-verified "
                        f"{meta.get('n_phase1_rows', 0):,}"
                        f" · gold {meta.get('n_gold', 0):,}")
            if ess:
                subtitle += f" · ESS {ess:.0f}"
        xlabel = (GRID_X_LABELS.get(segment, X_LABELS[segment]) if surface
                  else X_LABELS[segment])
        _chrome(ax, xlabel = xlabel, title = subtitle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    np.atleast_1d(axes)[0].set_ylabel("P(exists and open)")
    # Panels carry different series (slices, reference), so merge the labels.
    merged = {}
    for ax in np.atleast_1d(axes):
        if not ax.get_visible():
            continue
        for handle, label in zip(*ax.get_legend_handles_labels()):
            merged.setdefault(label, handle)
    fig.legend(list(merged.values()), list(merged.keys()), loc = "lower center",
               ncol = min(len(merged), 4), frameon = False)
    title = "Confidence calibration by detection segment"
    if any(calibration.is_grid_lookup(c) for c in curves.values()):
        title += " (Bayesian mixture)"
    fig.suptitle(title, fontsize = plt.rcParams["font.size"] * 1.44)
    fig.subplots_adjust(left = 0.06, right = 0.98, top = 0.80, bottom = 0.20,
                        wspace = 0.12)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, dpi = 300)
    plt.close(fig)
    print(f"  {out_path}")


def plot_matched_surface(lookup: pd.DataFrame, metadata: dict,
                         out_path: Path) -> None:
    """Heatmap of the matched grid's posterior mean, with the atom columns."""
    pivot = lookup.pivot(index = "osm_score", columns = "overture_score",
                         values = "conf_mean").sort_index().sort_index(axis = 1)
    osm = pivot.index.to_numpy(dtype = float)
    overture = pivot.columns.to_numpy(dtype = float)
    cmap = LinearSegmentedColormap.from_list(
        "matched_seq", ["#f4f1fa", "#9d7fd6", SEGMENTS[0][2]]
    )
    fig, ax = plt.subplots(figsize = (7.5, 6.2))
    mesh = ax.pcolormesh(overture, osm, pivot.to_numpy(dtype = float),
                         cmap = cmap, vmin = 0.0, vmax = 1.0,
                         shading = "nearest", rasterized = True)
    levels = [0.3, 0.5, 0.7, 0.9]
    contours = ax.contour(overture, osm, pivot.to_numpy(dtype = float),
                          levels = levels, colors = "white", linewidths = 0.8)
    ax.clabel(contours, fmt = "%.1f", fontsize = 8)
    for atom, style in zip(OVERTURE_ATOMS, SLICE_STYLES):
        ax.axvline(atom, color = "#222222", linewidth = 1.0, linestyle = style)
    fig.colorbar(mesh, ax = ax, label = "P(exists and open), posterior mean")
    _chrome(ax, xlabel = "Overture confidence",
            ylabel = "OSM turnover posterior",
            title = _grid_subtitle("Both sources: calibrated surface",
                                   metadata.get("matched", {}), method = True))
    ax.grid(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
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
        xlabel = X_LABELS[segment]
        if (metadata.get(segment) or {}).get("index_mode") == "grid":
            xlabel = RELIABILITY_X_LABELS.get(segment, xlabel)
        _chrome(ax, xlabel = xlabel, title = legend)
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
    parser.add_argument("--curves-dir", default = None,
                        help = f"Curves to plot (default {CURVES_DIR}).")
    parser.add_argument("--viz-dir", default = None,
                        help = f"Where the PNGs go (default {VIZ_DIR}).")
    args = parser.parse_args()
    curves_dir = Path(args.curves_dir) if args.curves_dir else CURVES_DIR
    viz_dir = Path(args.viz_dir) if args.viz_dir else VIZ_DIR

    curves = calibration.read_curves(curves_dir)
    metadata = calibration.read_curve_metadata(curves_dir)
    validation_rows = pd.read_parquet(
        config.get_file_path("calibration", "validation_rows")
    )

    population = {}
    for segment in calibration.SEGMENTS:
        rows = validation_rows[validation_rows["segment"] == segment]
        if not len(rows):
            continue
        lookup = curves.get(segment)
        if lookup is not None and calibration.is_grid_lookup(lookup) and (
            calibration.is_grid_surface(lookup)
        ):
            # The matched panel's x axis is the OSM score of its slices.
            population[segment] = calibration_fit.round_scores(rows)[
                "osm_score"
            ].to_numpy(dtype = float)
        else:
            population[segment] = curve_index_for_rows(rows, segment, metadata)

    viz_dir.mkdir(parents = True, exist_ok = True)
    print("Writing figures:")
    plot_curves(curves, metadata, population,
                viz_dir / "calibration_curves.png")
    matched = curves.get("matched")
    if matched is not None and calibration.is_grid_lookup(matched) and (
        calibration.is_grid_surface(matched)
    ):
        plot_matched_surface(matched, metadata,
                             viz_dir / "calibration_matched_surface.png")
    plot_reliability(validation_rows, metadata,
                     viz_dir / "calibration_reliability.png")

    if not args.skip_shift:
        conflated = config.get_file_path("conflation", "conflated")
        if args.input_suffix:
            conflated = conflated.with_name(
                f"{conflated.stem}_{args.input_suffix}{conflated.suffix}"
            )
        if conflated.exists():
            plot_shift(conflated, viz_dir / "calibration_shift.png")
        else:
            print(f"  (skipped shift panel: {conflated} not found)")


if __name__ == "__main__":
    main()
