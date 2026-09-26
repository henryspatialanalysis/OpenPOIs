#!/usr/bin/env python
"""
Shape diagnostics for the matched-segment combination modes.

Reads the per-mode full fits written by
``compare_matched_index.py --with-deployed-impact`` (``<eval dir>/fits``) and
draws, on the common 6x4 surface cells:

- per-mode heatmaps of the published value, averaged over production POIs in
  each cell, and each mode's difference from the pool;
- 1-D slices along the OSM score at each Overture bin, with bands;
- the **independence surface** ``logit p_osm + logit p_ov - logit(base rate)``
  (Genest & Schervish, in Genest & Zidek 1986 p. 9): where the fitted matched
  maps depart from it, the two sources are conditionally dependent. ``p_osm``
  and ``p_ov`` are the matched segment's OWN one-score marginal curves. The
  deployed osm-only / overture-only segment curves are the wrong marginals --
  a POI found by one source alone is far less likely to exist than a matched
  POI with the same score -- so that version (kept in the JSON as
  ``independence_segment_curves``) sits 0.2-0.5 below every fitted map for
  reasons that have nothing to do with dependence;
- the additive index's components ``h_osm``, ``h_ov`` with bootstrap bands;
- the interaction index's ``a3`` with its bootstrap interval.

Config keys used (config.yaml):
    calibration.validation_rows, conflation.calibration.*,
    directories.conflation (population; deployed 1-D curves)

Output (in --eval-dir):
    viz/matched_modes_heatmaps.png, viz/matched_modes_differences.png,
    viz/matched_modes_slices.png, viz/matched_independence.png,
    viz/matched_additive_components.png, viz/matched_interaction_a3.png,
    shape_diagnostics.json

Usage:
    python scripts/conflation/plot_matched_modes.py --eval-dir DIR [--reps 300]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from config_versioned import Config

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from openpois.conflation import calibration, calibration_fit as cf  # noqa: E402

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
plt.rcParams["font.size"] = 11

MODES = ("pool", "average", "additive", "interaction", "surface",
         "surface_8x6")
# Fixed categorical order (reference palette slots 1-5); never cycled.
MODE_COLORS = {"pool": "#2a78d6", "additive": "#eb6834",
               "interaction": "#1baf7a", "surface": "#4a3aa7",
               "average": "#eda100", "surface_8x6": "#e87ba4"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#e34948", "#f0efec", "#2a78d6"])
GRID_COLOR = "#DDDDDD"
TEXT = "#333333"


def _compare_module():
    path = Path(__file__).with_name("compare_matched_index.py")
    spec = importlib.util.spec_from_file_location("compare_matched_index",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chrome(ax, xlabel = None, ylabel = None, title = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize = plt.rcParams["font.size"] * 1.1)
    ax.set_axisbelow(True)
    ax.grid(color = GRID_COLOR, linewidth = 0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length = 0)


def cell_means(values: np.ndarray, cells: np.ndarray, shape: tuple
               ) -> np.ndarray:
    """Mean of per-POI ``values`` in each cell (NaN where a cell is empty)."""
    flat_n = shape[0] * shape[1]
    mass = np.bincount(cells, minlength = flat_n).astype(float)
    total = np.bincount(cells, weights = np.nan_to_num(values),
                        minlength = flat_n)
    with np.errstate(invalid = "ignore", divide = "ignore"):
        return (total / mass).reshape(shape)


def _heatmap(ax, matrix, edges, cmap, norm, fmt = "{:.3f}", title = None):
    image = ax.imshow(matrix, origin = "lower", cmap = cmap, norm = norm,
                      aspect = "auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                shade = norm(value) if norm is not None else 0.5
                ink = "white" if (shade > 0.7 or shade < 0.12) and \
                    cmap is SEQUENTIAL else TEXT
                if cmap is SEQUENTIAL and shade < 0.12:
                    ink = TEXT
                ax.text(j, i, fmt.format(value), ha = "center",
                        va = "center", fontsize = 8.5, color = ink)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"{lo:.3f}–\n{hi:.3f}" for lo, hi in
                        zip(edges["overture"][:-1], edges["overture"][1:])],
                       fontsize = 7.5)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([f"{lo:.2f}–{hi:.2f}" for lo, hi in
                        zip(edges["osm"][:-1], edges["osm"][1:])],
                       fontsize = 7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length = 0)
    if title:
        ax.set_title(title)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--eval-dir", required = True)
    parser.add_argument("--reps", type = int, default = 300,
                        help = "Bootstrap replicates for a3 / components.")
    args = parser.parse_args()
    eval_dir = Path(args.eval_dir).expanduser()
    viz = eval_dir / "viz"
    viz.mkdir(parents = True, exist_ok = True)
    compare = _compare_module()

    config = Config("~/repos/openpois/config.yaml")
    knobs = config.get("conflation", "calibration")
    fit_config = cf.FitConfig(
        min_cell_gold = int(knobs["min_cell_gold"]),
        grid_points = int(knobs["grid_points"]),
        output_bins = int(knobs["curve_output_bins"]),
        bootstrap_reps = int(knobs["bootstrap_reps"]),
        band_alpha = float(knobs["band_alpha"]),
        rng_seed = int(knobs["rng_seed"]),
        refine_by_confidence = bool(knobs["refine_by_confidence"]),
    )
    conflation_dir = config.get_dir_path("conflation")
    population = compare.load_population(conflation_dir / "conflated_cd.parquet")
    rounded = cf.round_scores(population)

    fits = {}
    for mode in MODES:
        mode_dir = eval_dir / "fits" / mode
        if (mode_dir / "matched_curve.parquet").exists():
            with open(mode_dir / "matched_metadata.json",
                      encoding = "utf-8") as handle:
                fits[mode] = (pd.read_parquet(mode_dir / "matched_curve.parquet"),
                              json.load(handle))
    surface_meta = fits["surface"][1]["surface"]
    edges = {"osm": np.array(surface_meta["osm_edges"]),
             "overture": np.array(surface_meta["overture_edges"])}
    shape = tuple(surface_meta["shape"])
    i_osm, i_ov = cf.surface_cells(rounded["osm_score"],
                                   rounded["overture_score"], edges)
    cells = i_osm * shape[1] + i_ov

    published = {}
    for mode, (lookup, meta) in fits.items():
        values = compare.deployed_values(population, lookup, meta)
        published[mode] = {
            "mean": values["conf_mean"].to_numpy(),
            "width": (values["conf_upper"] - values["conf_lower"]).to_numpy(),
        }
    tables = {m: cell_means(v["mean"], cells, shape)
              for m, v in published.items()}
    widths = {m: cell_means(v["width"], cells, shape)
              for m, v in published.items()}

    # --- Independence surface from the deployed 1-D curves ----------------
    deployed = calibration.read_curves(conflation_dir / "calibration")
    p_osm = calibration.apply_curve(population["osm_score"],
                                    deployed["osm"])["conf_mean"].to_numpy()
    p_ov = calibration.apply_curve(population["overture_score"],
                                   deployed["overture"])["conf_mean"].to_numpy()
    validation_rows = pd.read_parquet(
        config.get_file_path("calibration", "validation_rows")
    )
    rows = cf.round_scores(validation_rows[
        (validation_rows["segment"] == "matched")
        & validation_rows["stratum"].isin(cf.SEGMENTS)
        & validation_rows["llm_verdict"].isin(cf.VERDICTS)
    ].reset_index(drop = True))
    classes = cf.merge_thin_cells(cf.refined_class(rows),
                                  rows["gold"].to_numpy(dtype = bool), 25)
    inclusion = cf.inclusion_by_class(classes, rows["gold"].to_numpy(bool))
    gold = rows["gold"].to_numpy(dtype = bool)
    w = classes[gold].map(lambda c: inclusion[c]["weight"]).to_numpy()
    base = float(np.average(rows["y"].to_numpy()[gold], weights = w))
    segment_version = cf.expit(cf.logit(p_osm) + cf.logit(p_ov)
                               - cf.logit(base))
    # The valid benchmark: each score's marginal curve WITHIN matched.
    marginal = {}
    for axis in ("osm", "overture"):
        result = cf.fit_segment(rows, axis, fit_config, cross_fit = False)
        marginal[axis] = np.interp(rounded[f"{axis}_score"], result["grid"],
                                   result["curve"])
    independence = cf.expit(cf.logit(marginal["osm"])
                            + cf.logit(marginal["overture"]) - cf.logit(base))
    tables["independence"] = cell_means(independence, cells, shape)
    independence_segment = cell_means(segment_version, cells, shape)

    # --- Heatmaps -----------------------------------------------------------
    shown = [m for m in ("pool", "additive", "interaction", "surface",
                         "surface_8x6", "average") if m in tables]
    norm = plt.Normalize(0.6, 1.0)
    fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    fig.subplots_adjust(hspace = 0.5, wspace = 0.42, right = 0.84)
    for ax, mode in zip(axes.ravel(), shown):
        image = _heatmap(ax, tables[mode], edges, SEQUENTIAL, norm,
                         title = mode)
        ax.set_xlabel("Overture confidence bin")
        ax.set_ylabel("OSM score bin")
    fig.colorbar(image, cax = fig.add_axes([0.87, 0.3, 0.015, 0.4]),
                 label = "Published P(exists and open), POI mean per cell")
    fig.suptitle("Matched segment: published value on the common 6×4 cells "
                 "(20260902 production POIs)", fontsize = 14)
    fig.savefig(viz / "matched_modes_heatmaps.png", dpi = 200,
                bbox_inches = "tight")
    plt.close(fig)

    diff_modes = [m for m in ("additive", "interaction", "surface",
                              "surface_8x6", "average", "independence")
                  if m in tables]
    div = TwoSlopeNorm(vcenter = 0.0, vmin = -0.12, vmax = 0.12)
    fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    fig.subplots_adjust(hspace = 0.5, wspace = 0.42, right = 0.84)
    for ax, mode in zip(axes.ravel(), diff_modes):
        image = _heatmap(ax, tables[mode] - tables["pool"], edges, DIVERGING,
                         div, fmt = "{:+.3f}", title = f"{mode} − pool")
        ax.set_xlabel("Overture confidence bin")
        ax.set_ylabel("OSM score bin")
    fig.colorbar(image, cax = fig.add_axes([0.87, 0.3, 0.015, 0.4]),
                 label = "Difference from the pool (POI mean per cell)")
    fig.suptitle("Where each mode departs from the constrained pool",
                 fontsize = 14)
    fig.savefig(viz / "matched_modes_differences.png", dpi = 200,
                bbox_inches = "tight")
    plt.close(fig)

    # --- Slices along OSM at each Overture bin ------------------------------
    osm_grid = np.round(np.linspace(0.3, 1.0, 141), 6)
    fig, axes = plt.subplots(1, shape[1], figsize = (15, 4.6), sharey = True)
    slice_modes = [m for m in ("pool", "additive", "interaction", "surface")
                   if m in fits]
    for j, ax in enumerate(np.atleast_1d(axes)):
        in_bin = i_ov == j
        ov_value = float(np.median(rounded["overture_score"][in_bin]))
        frame = pd.DataFrame({"osm_score": osm_grid,
                              "overture_score": np.full(len(osm_grid),
                                                        ov_value)})
        for mode in slice_modes:
            lookup, meta = fits[mode]
            values = compare.deployed_values(frame, lookup, meta)
            color = MODE_COLORS[mode]
            if mode in ("pool", "surface"):
                ax.fill_between(osm_grid, values["conf_lower"],
                                values["conf_upper"], color = color,
                                alpha = 0.15, linewidth = 0, step = "post")
            ax.step(osm_grid, values["conf_mean"], where = "post",
                    color = color, linewidth = 2.0, label = mode)
        _chrome(ax, xlabel = "OSM score",
                title = f"Overture bin {j + 1} (median {ov_value:.3f})")
        ax.set_ylim(0.5, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("Published P(exists and open)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = len(labels),
               frameon = False)
    fig.suptitle("Slices along the OSM score, Overture held at each bin's "
                 "median (bands: pool and surface)", fontsize = 13)
    fig.subplots_adjust(bottom = 0.22, top = 0.84, wspace = 0.08)
    fig.savefig(viz / "matched_modes_slices.png", dpi = 200)
    plt.close(fig)

    # --- Independence surface vs fitted maps --------------------------------
    fig, axes = plt.subplots(1, 3, figsize = (15, 4.8))
    fig.subplots_adjust(wspace = 0.42)
    _heatmap(axes[0], tables["independence"], edges, SEQUENTIAL, norm,
             title = "Independence surface (matched marginals)")
    _heatmap(axes[1], tables["surface"] - tables["independence"], edges,
             DIVERGING, div, fmt = "{:+.3f}",
             title = "surface − independence")
    _heatmap(axes[2], tables["pool"] - tables["independence"], edges,
             DIVERGING, div, fmt = "{:+.3f}", title = "pool − independence")
    for ax in axes:
        ax.set_xlabel("Overture confidence bin")
    axes[0].set_ylabel("OSM score bin")
    fig.suptitle(f"Conditional-independence benchmark (base rate "
                 f"{base:.3f}); negative = the sources overlap", fontsize = 13)
    fig.savefig(viz / "matched_independence.png", dpi = 200,
                bbox_inches = "tight")
    plt.close(fig)

    # --- Additive components with bootstrap bands ----------------------------
    boot_additive = cf.bootstrap_index_params(rows, "additive", fit_config,
                                              reps = args.reps)
    boot_additive = [p for p in boot_additive if p.get("form") == "additive"]
    fitted = fits["additive"][1]["index"]
    component_summary = {}
    fig, axes = plt.subplots(1, 2, figsize = (12, 4.6))
    for ax, axis, label in ((axes[0], "osm", "OSM score"),
                            (axes[1], "overture", "Overture confidence")):
        grid = np.round(np.linspace(
            float(np.quantile(rounded[f"{axis}_score"], 0.01)), 1.0, 200), 6)
        draws = np.array([
            np.interp(grid, p[f"knots_{axis}"], p[f"levels_{axis}"])
            for p in boot_additive
        ])
        point = np.interp(grid, fitted[f"knots_{axis}"],
                          fitted[f"levels_{axis}"])
        lo, hi = np.quantile(draws, [0.025, 0.975], axis = 0)
        color = MODE_COLORS["additive"]
        ax.fill_between(grid, lo, hi, color = color, alpha = 0.18,
                        linewidth = 0)
        ax.plot(grid, point, color = color, linewidth = 2.0)
        _chrome(ax, xlabel = label, ylabel = f"h_{axis} (log-odds)",
                title = f"h_{axis}: {fitted[f'n_blocks_{axis}']} PAV blocks")
        ax.set_ylim(-6, 6)
        component_summary[axis] = {
            "n_blocks": fitted[f"n_blocks_{axis}"],
            "boot_n_blocks_median": float(np.median(
                [p[f"n_blocks_{axis}"] for p in boot_additive])),
        }
    fig.suptitle("Additive index components with 95% bootstrap bands "
                 "(clipped at ±6 for display)", fontsize = 13)
    fig.savefig(viz / "matched_additive_components.png", dpi = 200,
                bbox_inches = "tight")
    plt.close(fig)

    # --- Interaction a3 -------------------------------------------------------
    boot_inter = cf.bootstrap_index_params(rows, "interaction", fit_config,
                                           reps = args.reps)
    a3 = np.array([p["a3"] for p in boot_inter
                   if p.get("form") == "interaction"])
    active = pd.Series([",".join(p.get("constraints_active", []))
                        for p in boot_inter]).value_counts(normalize = True)
    a3_point = fits["interaction"][1]["index"]["a3"]
    fig, ax = plt.subplots(figsize = (7, 4.2))
    ax.hist(a3, bins = 40, color = MODE_COLORS["interaction"], alpha = 0.85)
    ax.axvline(a3_point, color = TEXT, linewidth = 1.5)
    ax.axvline(0.0, color = "#999999", linewidth = 1.0,
               linestyle = (0, (4, 3)))
    lo, hi = np.quantile(a3, [0.025, 0.975])
    _chrome(ax, xlabel = "a3 (rescaled-logit interaction)",
            ylabel = "Bootstrap replicates",
            title = f"a3 = {a3_point:.2f}, 95% [{lo:.2f}, {hi:.2f}]; "
                    f"share < 0: {np.mean(a3 < 0):.2f}")
    fig.savefig(viz / "matched_interaction_a3.png", dpi = 200,
                bbox_inches = "tight")
    plt.close(fig)

    out = {
        "surface_edges": {k: v.tolist() for k, v in edges.items()},
        "base_rate": base,
        "cell_tables": {k: v.tolist() for k, v in tables.items()},
        "independence_segment_curves": independence_segment.tolist(),
        "cell_band_widths": {k: v.tolist() for k, v in widths.items()},
        "cell_population_share": (np.bincount(
            cells, minlength = shape[0] * shape[1]).reshape(shape)
            / len(cells)).tolist(),
        "a3": {"point": a3_point, "lower": float(lo), "upper": float(hi),
               "share_negative": float(np.mean(a3 < 0)), "n": int(len(a3)),
               "constraints_active_share": active.to_dict()},
        "additive_components": component_summary,
        "independence_minus": {
            m: float(np.nanmean(np.abs(tables[m] - tables["independence"])))
            for m in ("pool", "additive", "interaction", "surface")
            if m in tables
        },
    }
    (eval_dir / "shape_diagnostics.json").write_text(
        json.dumps(cf._jsonable(out), indent = 2), encoding = "utf-8"
    )
    print(f"Wrote figures to {viz} and {eval_dir / 'shape_diagnostics.json'}")


if __name__ == "__main__":
    main()
