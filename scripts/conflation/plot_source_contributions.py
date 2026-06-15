#!/usr/bin/env python
"""
Plot confidence-weighted source contributions to the conflated dataset.

Reads conflated.parquet and draws a horizontal stacked bar chart. Each row
is a shared_label; the 15 most common shared labels (by POI count) in the
most recent database are shown, ordered descending so the largest sits at
the top. Each bar is split left-to-right into the three provenance classes
of the conflated dataset:

    Overture only (left)  — source == "overture"
    Both          (mid)   — source == "matched"
    OSM only      (right) — source == "osm"

Bar lengths are **confidence-weighted** observation counts: each POI is
weighted by its combined confidence score (``conf_mean``, the final
post-change-detection blended confidence carried in the published dataset)
rather than counted as 1. Each colored sub-bar is annotated, in white, with
its confidence-weighted contribution rounded roughly to the nearest
thousand.

Blank/null shared labels are excluded — they are not a meaningful label
type — so the 15 rows are the 15 most common *named* labels.

Config keys used (config.yaml):
    conflation.conflated — input GeoParquet path (conflated.parquet)
    conflation            — directory; output PNG lands in its viz/ subdir

Prerequisites:
    Run scripts/conflation/conflate.py (and apply_change_detection.py) first.

Output file (in conflation/<version>/viz/):
    source_contributions.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from config_versioned import Config

from pathlib import Path  # noqa: E402

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

# ----------------------------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")
INPUT_PATH = config.get_file_path("conflation", "conflated")
VIZ_DIR = config.get_dir_path("conflation") / "viz"
OUTPUT_PATH = VIZ_DIR / "source_contributions.png"

TOP_N = 14

# Shared labels to drop before ranking: every "Other ..." catch-all plus a
# few named labels we don't want surfaced here.
EXCLUDE_EXACT = {
    "Home Service", "Swimming Pool", "Real Estate", "Specialty Store",
    "Hotel", "Recreation",
}

# Register the Figtree variable font from the first candidate path that
# exists, then make it the default family. Falls back to matplotlib's
# default if none are found.
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

# Base font size, bumped 20% over the matplotlib default of 10.
plt.rcParams["font.size"] = 12

# source value -> (legend label, color). Listed left-to-right.
SOURCES = [
    ("overture", "Overture only", "#2e86c9"),
    ("matched", "Both", "#3d00a5"),
    ("osm", "OSM only", "#a0d787"),
]

# Skip the in-bar number when a sub-bar is narrower than this fraction of the
# widest total bar — the text would overflow its segment and collide.
LABEL_MIN_FRAC = 0.018

# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def fmt_weighted(value: float) -> str:
    """Format a confidence-weighted count roughly to the nearest thousand.

    >= 1k  -> integer thousands with a "k" suffix ("837k", "17k", "9k")
    < 1k   -> raw integer ("751")
    """
    if value >= 1_000:
        return f"{round(value / 1000):d}k"
    return f"{round(value):d}"


# ----------------------------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Reading {INPUT_PATH} ...")
    df = pd.read_parquet(
        INPUT_PATH,
        columns = ["shared_label", "source", "conf_mean"],
    )
    print(f"  {len(df):,} rows")

    # Drop blank/null shared labels — not a meaningful label type — plus the
    # excluded "Other ..." catch-alls and named labels.
    label = df["shared_label"]
    keep = (
        label.notna()
        & (label != "")
        & ~label.str.startswith("Other ")
        & ~label.isin(EXCLUDE_EXACT)
    )
    df = df[keep]

    # Confidence-weighted sum and raw count per (shared_label, source).
    agg = df.groupby(["shared_label", "source"], observed = True).agg(
        wsum = ("conf_mean", "sum"),
        n = ("conf_mean", "size"),
    )
    wsum = agg["wsum"].unstack(fill_value = 0.0)
    counts = agg["n"].unstack(fill_value = 0)
    for source, _, _ in SOURCES:
        if source not in wsum.columns:
            wsum[source] = 0.0
            counts[source] = 0

    # Top N shared labels by total confidence-weighted POI count; largest at
    # the top of the chart.
    top_labels = (
        wsum.sum(axis = 1).sort_values(ascending = False).head(TOP_N).index
    )
    wsum = wsum.loc[top_labels]

    # ----------------------------------------------------------------------------
    # Draw the horizontal stacked bar chart
    # ----------------------------------------------------------------------------
    y = np.arange(len(top_labels))
    max_total = wsum.sum(axis = 1).max()
    label_threshold = max_total * LABEL_MIN_FRAC

    fig, ax = plt.subplots(figsize = (13.33, 7.5))

    left = np.zeros(len(top_labels))
    for source, _, color in SOURCES:
        widths = wsum[source].to_numpy()
        ax.barh(y, widths, left = left, color = color, height = 0.90)
        # White in-bar annotation at each segment's center.
        for yi, w, x0 in zip(y, widths, left):
            if w >= label_threshold:
                ax.text(
                    x0 + w / 2,
                    yi,
                    fmt_weighted(w),
                    ha = "center",
                    va = "center",
                    color = "white",
                    fontsize = 10,
                    fontweight = "bold",
                )
        left += widths

    ax.set_yticks(y)
    ax.set_yticklabels(top_labels)
    ax.invert_yaxis()  # largest label at the top
    ax.set_xlabel("Confidence weighted POI count")
    # Title 20% larger than matplotlib's default (1.2x base -> 1.44x base).
    ax.set_title(
        "Confidence-weighted records in the conflated dataset, by source",
        fontsize = plt.rcParams["font.size"] * 1.44,
    )

    # Light-grey panel with white gridlines at each 100k tick, drawn behind
    # the bars. Ticks labelled "100k", "200k", ... rather than in millions.
    ax.set_xlim(0, left.max() * 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(100_000))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k")
    )
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis = "x", color = "#DDDDDD", linewidth = 1.0)
    ax.grid(axis = "y", visible = False)

    # Drop the border (all four spines) and tick marks.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length = 0)

    # Single-row legend at the bottom.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color = color)
        for _, _, color in SOURCES
    ]
    labels = [legend for _, legend, _ in SOURCES]
    ax.legend(
        handles,
        labels,
        loc = "upper center",
        bbox_to_anchor = (0.5, -0.065),
        ncol = len(SOURCES),
        frameon = False,
    )

    # Low padding so the plot fills the fixed 13.33 x 7.5" canvas.
    fig.subplots_adjust(
        left = 0.14, right = 0.99, top = 0.94, bottom = 0.11,
    )
    VIZ_DIR.mkdir(parents = True, exist_ok = True)
    # No bbox_inches="tight" — keep the canvas at the exact 13.33 x 7.5".
    fig.savefig(OUTPUT_PATH, dpi = 300)
    print(f"\nSaved to {OUTPUT_PATH}")
