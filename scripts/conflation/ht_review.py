#!/usr/bin/env python
"""
Design-weighted (Horvitz-Thompson) review of a deployed calibration map.

Runs ``openpois.conflation.calibration_ht.run_ht_check`` against a directory of
fitted curves and writes the review document. ``make calibrate`` runs it on the
exported Bayesian grid curves after they are applied; the retired v4
``fit_calibration.py`` called :func:`run_review` itself. A reuse month (curves
copied, not fit) gets its review the same way. The check never fails a run: it
is a review aid.

Config keys used:
  - versions.calibration, versions.conflation
  - directories.calibration.files.{validation_rows, metadata}
  - conflation.calibration.{min_cell_gold, refine_by_confidence} (fallback
    when the curve metadata does not record them)

Prerequisites:
  - the validation handoff for the round
  - fitted or copied curves (``{segment}_curve.parquet`` + metadata)

Output file(s), in --out-dir (default: the curves directory):
  - ht_review_<round>.pdf       the review document
  - ht_review_<round>_bins.csv  the bin table
  - ht_review_<round>.md        the report section (standalone runs only; a fit
                                run puts it in fit_report.md)

Usage:
    python scripts/conflation/ht_review.py [--curves-dir DIR] [--out-dir DIR]
        [--rows PATH --metadata PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from openpois.conflation import calibration, calibration_fit  # noqa: E402
from openpois.conflation import calibration_ht as cht  # noqa: E402

# Palette: the dataviz skill's reference instance (light mode). HT points are
# series blue, the deployed map is neutral ink, flags use the status colours.
COLORS = {"ht": "#2a78d6", "model": "#52514e", "ink": "#0b0b0b",
          "muted": "#52514e", "surface": "#fcfcfb", "grid": "#e4e3dd",
          "flag1": "#ec835a", "flag2": "#d03b3b", "rows": "#b7d3f6",
          "gold": "#9a9994"}
DIVERGING = LinearSegmentedColormap.from_list(
    "ht_diverging", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b52a2a"]
)
PAGE = (8.5, 11.0)
TABLE_ROWS_PER_PAGE = 42
FLAGGED_LIST_MAX = 24
Z_CLIP = 3.0
# Tolerance for matching a rounded score to an edge (a tenth of a rounding
# step; np.isclose's default relative tolerance spans several steps).
ATOM_TOL = 0.1 * 10.0 ** -calibration_fit.SCORE_DECIMALS
SOURCE_LABELS = {"silver_label_rates": "arm C silver rate",
                 "design-weighted gold": "gold share"}
VIEW_TITLES = {
    "osm_score": "OSM-only segment: native OSM score",
    "overture_score": "Overture-only segment: native Overture score",
    "raw_score_deciles": "Matched segment: deciles of the handoff raw_score",
    "cells": "Matched segment: 2-D cells",
}


def _style(ax) -> None:
    ax.set_facecolor(COLORS["surface"])
    ax.grid(True, color = COLORS["grid"], linewidth = 0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors = COLORS["muted"], labelsize = 8)


def _midpoints(table: pd.DataFrame) -> np.ndarray:
    return ((table["lo"] + table["hi"]) / 2.0).to_numpy(dtype = float)


def _ht_points(ax, x, table: pd.DataFrame) -> None:
    """Corrected rate with +-1 SD (thick) and +-2 SD (thin) bars, flags
    marked, and the gold-only Hajek rate as a faint reference marker."""
    r = table["rate"].to_numpy(dtype = float)
    sd = table["sd"].to_numpy(dtype = float)
    ax.scatter(x, table["gold_rate"].to_numpy(dtype = float), marker = "x",
               s = 22, color = COLORS["gold"], linewidth = 1.0, zorder = 3)
    ax.errorbar(x, r, yerr = 2.0 * sd, fmt = "none", ecolor = COLORS["ht"],
                elinewidth = 0.8, capsize = 0, zorder = 3)
    ax.errorbar(x, r, yerr = sd, fmt = "none", ecolor = COLORS["ht"],
                elinewidth = 2.6, capsize = 0, zorder = 3)
    flag = table["flag"].to_numpy(dtype = int)
    tested = table["tested"].to_numpy(dtype = bool)
    styles = (
        ((flag == 0) & tested, "o", COLORS["ht"]),
        (~tested, "o", "white"),
        (flag == 1, "s", COLORS["flag1"]),
        (flag == 2, "D", COLORS["flag2"]),
    )
    for mask, marker, color in styles:
        if mask.any():
            ax.scatter(x[mask], r[mask], marker = marker, s = 42,
                       facecolor = color, edgecolor = COLORS["ink"],
                       linewidth = 0.6, zorder = 5)
    for xi, ri, zi, fi in zip(x, r, table["z"], flag):
        if fi >= 1:
            ax.annotate(f"z {zi:+.1f}", (xi, ri), xytext = (5, -12),
                        textcoords = "offset points", fontsize = 7,
                        color = COLORS["ink"])
    model = table["model_mean"].to_numpy(dtype = float)
    ax.scatter(x, model, marker = "_", s = 160, color = COLORS["model"],
               linewidth = 2.0, zorder = 4)


def _auto_ylim(ax, table: pd.DataFrame) -> None:
    """Fit the y range to the +-2 SD bars and the model, within [0, 1]."""
    r = table["rate"].to_numpy(dtype = float)
    sd = table["sd"].to_numpy(dtype = float)
    values = np.concatenate([r - 2.0 * sd, r + 2.0 * sd,
                             table["model_mean"].to_numpy(dtype = float)])
    values = values[np.isfinite(values)]
    if not len(values):
        ax.set_ylim(-0.02, 1.02)
        return
    ax.set_ylim(max(-0.02, values.min() - 0.05),
                min(1.02, values.max() + 0.03))


def _legend(ax, deployed_label: str, per_row: bool = False) -> None:
    deployed = (
        Line2D([], [], color = COLORS["rows"], marker = "o", markersize = 4,
               linestyle = "none", label = deployed_label) if per_row
        else Line2D([], [], color = COLORS["model"], linewidth = 1.6,
                    label = deployed_label)
    )
    handles = [
        deployed,
        Line2D([], [], color = COLORS["model"], marker = "_", markersize = 12,
               linestyle = "none", markeredgewidth = 2.0,
               label = "deployed mean over the bin's phase-1 rows"),
        Line2D([], [], color = COLORS["ht"], marker = "o", linestyle = "none",
               markeredgecolor = COLORS["ink"],
               label = "corrected rate, bars +-1 SD (thick), +-2 SD (thin)"),
        Line2D([], [], color = COLORS["flag1"], marker = "s",
               linestyle = "none", markeredgecolor = COLORS["ink"],
               label = "|z| > 1 SD"),
        Line2D([], [], color = COLORS["flag2"], marker = "D",
               linestyle = "none", markeredgecolor = COLORS["ink"],
               label = "|z| > 2 SD"),
        Line2D([], [], color = "white", marker = "o", linestyle = "none",
               markeredgecolor = COLORS["ink"], label = "not tested (thin)"),
        Line2D([], [], color = COLORS["gold"], marker = "x",
               linestyle = "none",
               label = "gold-only HT rate (reference, not flagged)"),
    ]
    ax.legend(handles = handles, fontsize = 7, loc = "lower right",
              frameon = False)


def _gold_strip(ax, x, table: pd.DataFrame, xlabel: str) -> None:
    """Gold rows per bin, labelled gold / phase-1 rows."""
    n = table["n_gold"].to_numpy(dtype = float)
    rows = table["n_phase1"].to_numpy(dtype = float)
    ax.vlines(x, 0, n, color = COLORS["ht"], linewidth = 2.0)
    for xi, ni, ri in zip(x, n, rows):
        ax.annotate(f"{int(ni)}/{int(ri)}", (xi, ni), xytext = (0, 2),
                    textcoords = "offset points", ha = "center", fontsize = 5.5,
                    color = COLORS["muted"])
    ax.set_ylabel("gold rows\n(label: gold/rows)", fontsize = 8)
    ax.set_xlabel(xlabel, fontsize = 9)
    ax.set_ylim(0, max(n.max() * 1.3, 1.0) if len(n) else 1.0)
    _style(ax)


def _page_table(fig, rect, title: str, cells: list, columns: list) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_title(title, fontsize = 10, loc = "left")
    table = ax.table(cellText = cells or [["-"] * len(columns)],
                     colLabels = columns, loc = "upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.2)
    table.auto_set_column_width(list(range(len(columns))))


def _summary_page(pdf, check: dict, info: dict) -> None:
    fig = plt.figure(figsize = PAGE)
    fig.text(0.07, 0.95, "Design-weighted (Horvitz-Thompson) review",
             fontsize = 15, weight = "bold", color = COLORS["ink"])
    meta = (f"Validation round {info.get('round')}  |  curves: "
            f"{info.get('curves_dir')}")
    fig.text(0.07, 0.928, meta, fontsize = 7.5, color = COLORS["muted"])
    fig.text(0.07, 0.912, textwrap.fill(cht.method_text(check["min_rows"]),
                                        120),
             fontsize = 7.2, va = "top", color = COLORS["ink"])
    fmt = cht._fmt

    _page_table(
        fig, [0.07, 0.70, 0.86, 0.12], "Share of tested bins beyond 1 and 2 SD",
        [[s.segment, s.view, str(s.tested),
          f"{s.beyond_1sd} ({fmt(s.share_1sd, '.0%')})",
          f"{s.beyond_2sd} ({fmt(s.share_2sd, '.0%')})"]
         for s in check["summary"].itertuples()],
        ["segment", "view", "bins tested", "> 1 SD (chance 32%)",
         "> 2 SD (chance 5%)"],
    )
    _page_table(
        fig, [0.07, 0.585, 0.86, 0.09], "Calibration in the large",
        [[r.segment, f"{r.n_phase1:,}", f"{r.n_gold:,}", fmt(r.rate, ".3f"),
          fmt(r.model_mean, ".3f"), fmt(r.diff, "+.3f"), fmt(r.z, "+.2f"),
          fmt(r.gold_rate, ".3f")]
         for r in check["large"].itertuples()],
        ["segment", "rows", "gold", "corrected rate", "model mean",
         "model - rate", "z", "gold-only HT"],
    )
    _page_table(
        fig, [0.07, 0.36, 0.86, 0.2],
        "Correction rates q = P(exists | segment, LLM verdict)",
        [[r.segment, r.verdict, fmt(r.q, ".4f"), fmt(r.raw, ".4f"),
          str(r.n_gold), fmt(r.ess, ".1f"), SOURCE_LABELS.get(r.source, r.source)]
         for r in check["rates"].itertuples()],
        ["segment", "verdict", "q (smoothed)", "raw", "gold", "Kish ESS",
         "source"],
    )

    flagged = check["bins"][check["bins"]["flag"] >= 1]
    lines = [f"Flagged bins: {len(flagged)} (** beyond 2 SD)"]
    for _, row in flagged.head(FLAGGED_LIST_MAX).iterrows():
        mark = "**" if row["flag"] >= 2 else "  "
        lines.append(
            f"{mark} {row['segment']:<9}{row['view']:<18}"
            f"{cht.bin_label(row):<44} rate {row['rate']:.3f}  "
            f"model {row['model_mean']:.3f}  z {row['z']:+.2f}"
        )
    if len(flagged) > FLAGGED_LIST_MAX:
        lines.append(f"... and {len(flagged) - FLAGGED_LIST_MAX} more "
                     f"(see the bin table)")
    for note in check["notes"]:
        lines.append(f"Note: {note}")
    fig.text(0.07, 0.32, "\n".join(lines), fontsize = 6.5, va = "top",
             family = "monospace", color = COLORS["ink"])
    pdf.savefig(fig)
    plt.close(fig)


def _step(ax, lookup: pd.DataFrame) -> None:
    """The deployed 1-D lookup, drawn as served: a step curve as steps, a
    grid curve as the line through its nodes (deploy interpolates linearly)."""
    if calibration.is_grid_lookup(lookup):
        order = np.argsort(lookup["score"].to_numpy(dtype = float))
        ax.plot(lookup["score"].to_numpy(dtype = float)[order],
                lookup["conf_mean"].to_numpy(dtype = float)[order],
                color = COLORS["model"], linewidth = 1.6, zorder = 2)
        return
    lo = lookup["score_lo"].to_numpy(dtype = float)
    hi = lookup["score_hi"].to_numpy(dtype = float)
    mean = lookup["conf_mean"].to_numpy(dtype = float)
    xs = np.column_stack([lo, hi]).ravel()
    ys = np.repeat(mean, 2)
    ax.plot(xs, ys, color = COLORS["model"], linewidth = 1.6, zorder = 2)


def _one_d_page(pdf, check: dict, segment: str, view: str,
                curves: dict) -> None:
    table = check["bins"][(check["bins"]["segment"] == segment)
                          & (check["bins"]["view"] == view)]
    rows = check["rows"][segment]
    fig, (ax, strip) = plt.subplots(
        2, 1, figsize = PAGE, gridspec_kw = {"height_ratios": [4, 1]},
        sharex = True,
    )
    x = _midpoints(table)
    lookup = curves.get(segment)
    if view == "raw_score_deciles":
        column = "raw_score"
        ax.scatter(rows[column], rows["model"], s = 4, color = COLORS["rows"],
                   linewidth = 0, zorder = 1)
        deployed_label = ("deployed conf_mean per phase-1 row (the lookup is "
                          "on the source scores, not raw_score)")
    else:
        column = view
        one_d = lookup is not None and not (
            calibration.is_surface_lookup(lookup)
            or calibration.is_grid_surface(lookup)
        )
        if one_d:
            _step(ax, lookup)
        deployed_label = (
            "deployed grid curve (posterior mean conf_mean)"
            if one_d and calibration.is_grid_lookup(lookup)
            else "deployed step lookup (conf_mean)"
        )
    _ht_points(ax, x, table)
    _legend(ax, deployed_label, per_row = view == "raw_score_deciles")
    values = rows[column].to_numpy(dtype = float)
    values = values[np.isfinite(values)]
    if len(values):
        pad = 0.02 * max(values.max() - values.min(), 1e-3)
        ax.set_xlim(values.min() - pad, values.max() + pad)
    _auto_ylim(ax, table)
    ax.set_ylabel("P(exists)", fontsize = 9)
    summary = check["summary"]
    row = summary[(summary["segment"] == segment) & (summary["view"] == view)]
    subtitle = ""
    if len(row) and row["tested"].iloc[0]:
        s = row.iloc[0]
        subtitle = (f"{s['tested']} bins tested; beyond 1 SD "
                    f"{s['beyond_1sd']} ({s['share_1sd']:.0%}, chance 32%), "
                    f"beyond 2 SD {s['beyond_2sd']} ({s['share_2sd']:.0%}, "
                    f"chance 5%)")
    ax.set_title(f"{VIEW_TITLES.get(view, view)}\n{subtitle}", fontsize = 10,
                 loc = "left")
    _style(ax)
    _gold_strip(strip, x, table, column)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _slice_curve(segment: str, atom: float, osm_grid: np.ndarray,
                 curves: dict, metadata: dict) -> np.ndarray:
    frame = pd.DataFrame({"osm_score": osm_grid,
                          "overture_score": np.full(len(osm_grid), atom)})
    try:
        return cht.deployed_means(frame, segment, curves, metadata)
    except (ValueError, KeyError):
        return np.full(len(osm_grid), np.nan)


def _matched_page(pdf, check: dict, curves: dict, metadata: dict) -> None:
    segment = "matched"
    table = check["bins"][(check["bins"]["segment"] == segment)
                          & (check["bins"]["view"] == "cells")]
    grid = check["grids"].get(segment)
    fig = plt.figure(figsize = PAGE)
    heat = fig.add_axes([0.09, 0.52, 0.72, 0.38])
    cbar_ax = fig.add_axes([0.84, 0.52, 0.025, 0.38])
    ov_edges = grid["ov_edges"] if grid else np.array([])
    n_cols = max(len(ov_edges) - 1, 0)
    atoms = cht.atom_columns(ov_edges) if n_cols else []
    for _, cell in table.iterrows():
        j = int(np.searchsorted(ov_edges, cell["ov_lo"], side = "right") - 1)
        z = cell["z"] if cell["tested"] else np.nan
        color = (DIVERGING((np.clip(z, -Z_CLIP, Z_CLIP) + Z_CLIP)
                           / (2.0 * Z_CLIP))
                 if np.isfinite(z) else "#ffffff")
        edge = {2: (COLORS["ink"], 2.4, "-"),
                1: (COLORS["ink"], 1.2, "--")}.get(int(cell["flag"]),
                                                   (COLORS["grid"], 0.8, "-"))
        heat.add_patch(Rectangle(
            (j, cell["lo"]), 1.0, cell["hi"] - cell["lo"], facecolor = color,
            edgecolor = edge[0], linewidth = edge[1], linestyle = edge[2],
        ))
        counts = f"{int(cell['n_phase1'])} rows, {int(cell['n_gold'])} gold"
        label = f"z {z:+.1f}\n{counts}" if np.isfinite(z) else counts
        heat.text(j + 0.5, (cell["lo"] + cell["hi"]) / 2.0, label,
                  ha = "center", va = "center", fontsize = 6.5,
                  color = COLORS["ink"])
    labels = []
    for j in range(n_cols):
        if j in atoms:
            labels.append(f"atom\n{ov_edges[j]:.6f}")
        else:
            labels.append(f"{ov_edges[j]:.3f}-\n{ov_edges[j + 1]:.3f}")
    heat.set_xlim(0, max(n_cols, 1))
    osm = check["rows"][segment]["osm_score"].to_numpy(dtype = float)
    osm = osm[np.isfinite(osm)]
    if len(osm):
        heat.set_ylim(osm.min(), osm.max())
    heat.set_xticks(np.arange(n_cols) + 0.5)
    heat.set_xticklabels(labels, fontsize = 7)
    heat.set_xlabel("Overture score column", fontsize = 9)
    heat.set_ylabel("OSM score", fontsize = 9)
    heat.set_title(
        "Matched segment: z = (model - rate) / SD per cell (red: map above the "
        "corrected rate)\nsolid outline beyond 2 SD, dashed beyond 1 SD; white: not "
        "tested", fontsize = 9, loc = "left",
    )
    scalar = plt.cm.ScalarMappable(cmap = DIVERGING,
                                   norm = plt.Normalize(-Z_CLIP, Z_CLIP))
    fig.colorbar(scalar, cax = cbar_ax, label = "z (clipped at +-3)")

    rows = check["rows"][segment]
    for k, j in enumerate(atoms[:2]):
        ax = fig.add_axes([0.09 + k * 0.46, 0.08, 0.40, 0.32])
        in_col = table[np.isclose(table["ov_lo"], ov_edges[j], rtol = 0.0,
                                  atol = ATOM_TOL)]
        # The atom as the handoff stores it: curves fit before score rounding
        # have unrounded edges, and deploy applies them to unrounded scores.
        at_atom = rows.loc[np.isclose(rows["overture_score"], ov_edges[j],
                                      rtol = 0.0, atol = ATOM_TOL),
                           "overture_score_unrounded"]
        atom = float(at_atom.mode().iloc[0]) if len(at_atom) else ov_edges[j]
        grid_osm = np.linspace(osm.min(), osm.max(), 400) if len(osm) else []
        if len(grid_osm):
            ax.plot(grid_osm, _slice_curve(segment, atom, grid_osm, curves,
                                           metadata),
                    color = COLORS["model"], linewidth = 1.6, zorder = 2)
        _ht_points(ax, _midpoints(in_col), in_col)
        _auto_ylim(ax, in_col)
        ax.set_xlabel("OSM score", fontsize = 9)
        if k == 0:
            ax.set_ylabel("P(exists)", fontsize = 9)
        ax.set_title(f"OSM slice at Overture = {ov_edges[j]:.6f}\n(line: "
                     f"deployed map at that Overture score)", fontsize = 9,
                     loc = "left")
        _style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _table_pages(pdf, bins: pd.DataFrame) -> int:
    columns = ["segment", "view", "range", "rows", "gold", "rate", "SD",
               "model", "d/SD", "flag", "gold HT", "gold ESS", "gold raw"]
    cells = []
    fmt = cht._fmt
    for _, row in bins.iterrows():
        cells.append([
            row["segment"], row["view"].replace("_score_deciles", " deciles"),
            cht.bin_label(row), f"{int(row['n_phase1'])}",
            f"{int(row['n_gold'])}", fmt(row["rate"], ".3f"),
            fmt(row["sd"], ".3f"), fmt(row["model_mean"], ".3f"),
            fmt(row["z"], "+.2f"),
            {0: "" if row["tested"] else "thin", 1: "*", 2: "**"}[
                int(row["flag"])],
            fmt(row["gold_rate"], ".3f"), fmt(row["gold_ess"], ".1f"),
            fmt(row["gold_raw_rate"], ".3f"),
        ])
    pages = max(1, math.ceil(len(cells) / TABLE_ROWS_PER_PAGE))
    for page in range(pages):
        chunk = cells[page * TABLE_ROWS_PER_PAGE:
                      (page + 1) * TABLE_ROWS_PER_PAGE] or [["-"] * len(columns)]
        fig = plt.figure(figsize = PAGE)
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.9])
        ax.axis("off")
        ax.set_title(f"Bin table ({page + 1} of {pages}); flag * beyond 1 SD, "
                     f"** beyond 2 SD", fontsize = 10, loc = "left")
        table = ax.table(cellText = chunk, colLabels = columns,
                         loc = "upper left")
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.auto_set_column_width(list(range(len(columns))))
        pdf.savefig(fig)
        plt.close(fig)
    return pages


def expected_pages(check: dict) -> int:
    """Pages :func:`write_review_pdf` writes for ``check``."""
    views = check["summary"]
    n = 1 + int(len(views))
    n_bins = len(check["bins"])
    return n + max(1, math.ceil(n_bins / TABLE_ROWS_PER_PAGE))


def write_review_pdf(check: dict, path: Path, curves: dict, metadata: dict,
                     info: dict = None) -> int:
    """Write the review PDF; returns the page count.

    Page 1 is the summary; then one page per 1-D view (osm, overture, the
    matched raw-score deciles), the matched 2-D page (heatmap of z and the
    OSM slices at the two Overture atoms), and last the bin table.
    """
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    pages = 0
    with PdfPages(path) as pdf:
        _summary_page(pdf, check, info or {})
        pages += 1
        for view in check["summary"].itertuples():
            if view.view == "cells":
                _matched_page(pdf, check, curves, metadata)
            else:
                _one_d_page(pdf, check, view.segment, view.view, curves)
            pages += 1
        pages += _table_pages(pdf, check["bins"])
        info_dict = pdf.infodict()
        info_dict["Title"] = f"HT review, round {(info or {}).get('round')}"
    return pages


def fit_config_for(metadata: dict, knobs: dict = None
                   ) -> calibration_fit.FitConfig:
    """The class settings the curves were fit with, else the config's."""
    recorded = {}
    for entry in metadata.values():
        recorded = entry.get("fit_config") or {}
        if recorded:
            break
    knobs = knobs or {}
    defaults = calibration_fit.FitConfig()

    def setting(key: str):
        return recorded.get(key, knobs.get(key, getattr(defaults, key)))

    return calibration_fit.FitConfig(
        min_cell_gold = int(setting("min_cell_gold")),
        refine_by_confidence = bool(setting("refine_by_confidence")),
    )


def run_review(curves_dir: Path, validation_rows: pd.DataFrame,
               handoff_metadata: dict, out_dir: Path = None,
               fit_config: calibration_fit.FitConfig = None,
               knobs: dict = None) -> dict:
    """Run the check on the curves in ``curves_dir`` and write the outputs.

    Returns the check, the report lines and the paths written. The PDF lands
    at ``<out_dir>/ht_review_<round>.pdf`` (``out_dir`` defaults to
    ``curves_dir``).
    """
    curves_dir = Path(curves_dir)
    out_dir = Path(out_dir) if out_dir else curves_dir
    curves = calibration.read_curves(curves_dir)
    metadata = calibration.read_curve_metadata(curves_dir)
    fit_config = fit_config or fit_config_for(metadata, knobs)
    round_id = handoff_metadata.get("validation_round") or "unknown"
    check = cht.run_ht_check(validation_rows, curves, metadata, fit_config)
    out_dir.mkdir(parents = True, exist_ok = True)
    pdf_path = out_dir / f"ht_review_{round_id}.pdf"
    csv_path = out_dir / f"ht_review_{round_id}_bins.csv"
    check["bins"].to_csv(csv_path, index = False)
    pages = write_review_pdf(check, pdf_path, curves, metadata,
                             info = {"round": round_id,
                                     "curves_dir": str(curves_dir)})
    return {"check": check, "pdf": pdf_path, "csv": csv_path,
            "pages": pages, "lines": cht.report_lines(check, pdf_path.name)}


def main() -> None:
    from config_versioned import Config

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--curves-dir", default = None,
                        help = ("Fitted or copied curves (default: "
                                "conflation/<version>/calibration)."))
    parser.add_argument("--out-dir", default = None,
                        help = "Where the review lands (default: curves dir).")
    parser.add_argument("--rows", default = None,
                        help = "Validation rows parquet (default: config).")
    parser.add_argument("--metadata", default = None,
                        help = "Handoff metadata.json (default: config).")
    args = parser.parse_args()

    config = Config("~/repos/openpois/config.yaml")
    curves_dir = (Path(args.curves_dir).expanduser() if args.curves_dir
                  else config.get_dir_path("conflation") / "calibration")
    rows_path = (Path(args.rows).expanduser() if args.rows
                 else config.get_file_path("calibration", "validation_rows"))
    meta_path = (Path(args.metadata).expanduser() if args.metadata
                 else config.get_file_path("calibration", "metadata"))
    print(f"Curves: {curves_dir}")
    print(f"Validation handoff: {rows_path}")
    validation_rows = pd.read_parquet(rows_path)
    with open(meta_path, encoding = "utf-8") as handle:
        handoff_metadata = json.load(handle)
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else curves_dir
    result = run_review(curves_dir, validation_rows, handoff_metadata,
                        out_dir = out_dir,
                        knobs = config.get("conflation", "calibration"))
    round_id = handoff_metadata.get("validation_round") or "unknown"
    md_path = out_dir / f"ht_review_{round_id}.md"
    md_path.write_text("\n".join(result["lines"]), encoding = "utf-8")
    print(result["check"]["summary"].to_string(index = False))
    print(f"Review: {result['pdf']} ({result['pages']} pages)")
    print(f"Bin table: {result['csv']}")
    print(f"Report section: {md_path}")


if __name__ == "__main__":
    main()
