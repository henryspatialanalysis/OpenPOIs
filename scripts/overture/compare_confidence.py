"""
Compare Overture Places confidence scores between two local snapshot releases
(prior month vs current month), restricted to POIs inside the shared-label
schema.

Read-only. Informs the monthly decision on whether the post-conflation
confidence calibration needs a new validation round: the calibration curves
map Overture's raw confidence to P(exists and is open), so a large shift in
what the same POI scores month-over-month means the curves' x-axis no longer
carries the meaning it was fitted against (see
.claude/docs/confidence-calibration.md).

Decision rule (adopted 2026-09-02, see docs/confidence-calibration.md): PASS —
reuse the prior fitted curves verbatim — unless overall matched RMSE > 0.10, or
|mean bias| > 0.03, or more than 10% of matched POIs move by |diff| > 0.1;
any breach requires a new openpois-validator round before publishing.

Method: POIs are matched across releases on the GERS id (``overture_id``).
Shared labels are assigned by labelling the *distinct taxonomy tuples* once
with the standard 6-tier crosswalk cascade and joining back in DuckDB, so the
full snapshots are never loaded into memory. Rows whose taxonomy falls outside
the crosswalk (label == "") are excluded on both sides. Faceting uses the
CURRENT release's label; the share of matched POIs whose label changed between
releases is reported separately.

Reports (printed as markdown, saved as CSV):
  1. Row counts and join coverage in both directions — GERS id churn is
     itself a drift signal, not an error.
  2. Difference metrics on matched POIs: quantiles of (current - prior) from
     0 to 100 by 5, RMSE, MAE, bias, share exactly equal, share |diff| above
     0.05 / 0.10 / 0.20, Pearson and Spearman correlation.
  3. Per-shared-label n / RMSE / MAE / bias, sorted by RMSE.
  4. Marginal confidence deciles per release (catches regime shifts among
     unmatched/new POIs that the matched join cannot see).

Plots (house style, written next to the current snapshot):
  - confidence_comparison_overall.png — prior-vs-current 2D histogram with
    identity line, plus a histogram of differences.
  - confidence_comparison_by_label.png — faceted 2D histograms for the
    largest shared labels plus any label whose RMSE stands out.

Usage:
    # After downloading the new snapshot (config already bumped):
    python scripts/overture/compare_confidence.py

    # Explicit versions (directory names under directories.snapshot_overture):
    python scripts/overture/compare_confidence.py \
        --prior-version 20260722_tax3 --current-version 20260819

Config keys used (config.yaml):
    versions.snapshot_overture              — current-version default
    directories.snapshot_overture           — snapshot root + file name
    download.overture.duckdb.memory_limit / threads

Output file(s), under <snapshot root>/<current version>/viz/:
    confidence_comparison_metrics.csv
    confidence_comparison_by_label.csv
    confidence_comparison_overall.png
    confidence_comparison_by_label.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from config_versioned import Config

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from openpois.conflation.taxonomy import (  # noqa: E402
    assign_overture_shared_label,
    load_match_radii,
    load_overture_crosswalk,
)

CONFIG_PATH = "~/repos/openpois/config.yaml"
LEVELS = ["taxonomy_l0", "taxonomy_l1", "taxonomy_l2", "taxonomy_l3"]
DIFF_QUANTILES = list(range(0, 101, 5))
ABS_DIFF_THRESHOLDS = [0.05, 0.10, 0.20]
HEATMAP_BINS = 80

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

GRID_COLOR = "#DDDDDD"


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


def _parse_args() -> argparse.Namespace:
    config = Config(CONFIG_PATH)
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--current-version",
        default = config.get("versions", "snapshot_overture"),
        help = "Current snapshot version directory (default: config "
        "versions.snapshot_overture).",
    )
    parser.add_argument(
        "--prior-version",
        default = None,
        help = "Prior snapshot version directory. Defaults to the "
        "lexicographically largest other version present on disk.",
    )
    parser.add_argument(
        "--min-label-n",
        type = int,
        default = 5_000,
        help = "Per-label metrics rows below this matched count are kept in "
        "the CSV but excluded from outlier flagging (default 5000).",
    )
    parser.add_argument(
        "--facet-labels",
        type = int,
        default = 12,
        help = "Number of largest shared labels to facet (default 12).",
    )
    return parser.parse_args()


def _snapshot_path(root: Path, version: str, file_name: str) -> Path:
    path = root / version / file_name
    if not path.exists():
        raise FileNotFoundError(f"No snapshot at {path}")
    return path


def _default_prior_version(root: Path, current: str, file_name: str) -> str:
    candidates = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and d.name != current and (d / file_name).exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prior snapshot directory found under {root}"
        )
    return candidates[-1]


def _label_distinct_tuples(
    con: duckdb.DuckDBPyConnection, paths: list[Path],
) -> pd.DataFrame:
    """Label every distinct taxonomy tuple across the given snapshots.

    Returns a frame with the four (empty-string-filled) taxonomy levels and
    the assigned ``shared_label``, restricted to in-schema tuples.
    """
    selects = [
        "SELECT DISTINCT "
        + ", ".join(
            f"coalesce({col}, '') AS {col}" for col in LEVELS
        )
        + f" FROM read_parquet('{p}')"
        for p in paths
    ]
    tuples = con.execute(" UNION ".join(selects)).df().astype(object)
    labels, _ = assign_overture_shared_label(
        tuples, load_overture_crosswalk(), load_match_radii(),
    )
    tuples["shared_label"] = labels
    tuples = tuples[tuples["shared_label"] != ""].reset_index(drop = True)
    print(
        f"Distinct taxonomy tuples: {len(labels):,} "
        f"({len(tuples):,} in-schema)"
    )
    # Final .astype(object): DuckDB 1.4.1 cannot register pandas-3
    # ``str``-dtype frames (see CLAUDE.md gotchas), and pandas 3 re-infers
    # ``str`` on column assignment, so the cast must come last.
    return tuples.astype(object)


def _register_release(
    con: duckdb.DuckDBPyConnection, name: str, path: Path,
) -> int:
    """Create a slim in-schema temp table (overture_id, shared_label,
    confidence) for one release; returns its row count."""
    join_cond = " AND ".join(
        f"coalesce(p.{col}, '') = t.{col}" for col in LEVELS
    )
    con.execute(
        f"CREATE TEMP TABLE {name} AS "
        f"SELECT p.overture_id, t.shared_label, p.confidence "
        f"FROM read_parquet('{path}') p JOIN tuple_labels t ON {join_cond}"
    )
    n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    n_dup = n - con.execute(
        f"SELECT count(DISTINCT overture_id) FROM {name}"
    ).fetchone()[0]
    if n_dup:
        print(f"WARNING: {name} has {n_dup:,} duplicate overture_id rows")
    return n


def _marginal_deciles(
    con: duckdb.DuckDBPyConnection, name: str,
) -> np.ndarray:
    q = ", ".join(str(x / 10.0) for x in range(11))
    return np.array(
        con.execute(
            f"SELECT quantile_cont(confidence, [{q}]) FROM {name}"
        ).fetchone()[0]
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import rankdata
        rx, ry = rankdata(x), rankdata(y)
    except ImportError:
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _heatmap(ax, prior: np.ndarray, current: np.ndarray, title: str) -> None:
    counts, _, _ = np.histogram2d(
        prior, current, bins = HEATMAP_BINS, range = [[0, 1], [0, 1]],
    )
    ax.imshow(
        counts.T, origin = "lower", extent = [0, 1, 0, 1],
        norm = LogNorm(vmin = 1), cmap = "viridis", aspect = "equal",
        interpolation = "nearest",
    )
    ax.plot([0, 1], [0, 1], color = "#DDDDDD", linewidth = 0.8)
    _chrome(ax, title = title)
    ax.grid(False)


def main() -> None:
    args = _parse_args()
    config = Config(CONFIG_PATH)
    root = Path(
        config.get("directories", "snapshot_overture", "path")
    ).expanduser()
    file_name = config.get(
        "directories", "snapshot_overture", "files", "snapshot"
    )
    current_version = args.current_version
    prior_version = args.prior_version or _default_prior_version(
        root, current_version, file_name,
    )
    current_path = _snapshot_path(root, current_version, file_name)
    prior_path = _snapshot_path(root, prior_version, file_name)
    viz_dir = root / current_version / "viz"
    viz_dir.mkdir(parents = True, exist_ok = True)
    print(
        f"Comparing confidence: prior={prior_version} -> "
        f"current={current_version}\n"
    )

    con = duckdb.connect()
    con.execute(
        "SET memory_limit = "
        f"'{config.get('download', 'overture', 'duckdb', 'memory_limit')}'"
    )
    con.execute(
        f"SET threads = {config.get('download', 'overture', 'duckdb', 'threads')}"
    )

    tuples = _label_distinct_tuples(con, [prior_path, current_path])
    con.register("tuple_labels", tuples)
    n_prior = _register_release(con, "rel_prior", prior_path)
    n_current = _register_release(con, "rel_current", current_path)

    matched = con.execute(
        "SELECT b.shared_label, a.shared_label AS prior_label, "
        "a.confidence AS conf_prior, b.confidence AS conf_current "
        "FROM rel_prior a JOIN rel_current b USING (overture_id)"
    ).df()
    n_matched = len(matched)
    conf_prior = matched["conf_prior"].to_numpy()
    conf_current = matched["conf_current"].to_numpy()
    diff = conf_current - conf_prior
    label_changed = float(
        (matched["shared_label"] != matched["prior_label"]).mean()
    )
    matched = matched.drop(columns = ["prior_label"])

    # -- Overall metrics -------------------------------------------
    metrics = {
        "prior_version": prior_version,
        "current_version": current_version,
        "n_in_schema_prior": n_prior,
        "n_in_schema_current": n_current,
        "n_matched": n_matched,
        "match_share_of_prior": n_matched / n_prior,
        "match_share_of_current": n_matched / n_current,
        "share_label_changed": label_changed,
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias_mean_diff": float(np.mean(diff)),
        "sd_diff": float(np.std(diff)),
        "share_exactly_equal": float(np.mean(diff == 0)),
        "pearson_r": float(np.corrcoef(conf_prior, conf_current)[0, 1]),
        "spearman_rho": _spearman(conf_prior, conf_current),
    }
    for thr in ABS_DIFF_THRESHOLDS:
        metrics[f"share_absdiff_gt_{thr:g}"] = float(
            np.mean(np.abs(diff) > thr)
        )
    for pct, val in zip(
        DIFF_QUANTILES, np.percentile(diff, DIFF_QUANTILES)
    ):
        metrics[f"diff_p{pct:03d}"] = float(val)
    for name, tbl in [("prior", "rel_prior"), ("current", "rel_current")]:
        for pct, val in zip(range(0, 101, 10), _marginal_deciles(con, tbl)):
            metrics[f"marginal_{name}_p{pct:03d}"] = float(val)

    metrics_df = pd.DataFrame(
        {"metric": list(metrics), "value": list(metrics.values())}
    )
    metrics_path = viz_dir / "confidence_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index = False)

    # -- Per-label metrics -----------------------------------------
    matched["diff"] = diff
    by_label = (
        matched.groupby("shared_label")["diff"]
        .agg(
            n = "size",
            rmse = lambda s: float(np.sqrt(np.mean(s.to_numpy() ** 2))),
            mae = lambda s: float(np.mean(np.abs(s.to_numpy()))),
            bias = "mean",
        )
        .sort_values("rmse", ascending = False)
        .reset_index()
    )
    by_label_path = viz_dir / "confidence_comparison_by_label.csv"
    by_label.to_csv(by_label_path, index = False)

    # -- Printed summary -------------------------------------------
    print("\n## Overall\n")
    for key in [
        "n_in_schema_prior", "n_in_schema_current", "n_matched",
        "match_share_of_prior", "match_share_of_current",
        "share_label_changed", "rmse", "mae", "bias_mean_diff",
        "share_exactly_equal", "pearson_r", "spearman_rho",
    ] + [f"share_absdiff_gt_{t:g}" for t in ABS_DIFF_THRESHOLDS]:
        val = metrics[key]
        fmt = f"{val:,}" if isinstance(val, int) else f"{val:.4f}"
        print(f"| {key} | {fmt} |")
    print("\n## Difference quantiles (current - prior)\n")
    print("| pct | " + " | ".join(f"p{p}" for p in DIFF_QUANTILES) + " |")
    print(
        "| diff | "
        + " | ".join(
            f"{metrics[f'diff_p{p:03d}']:.3f}" for p in DIFF_QUANTILES
        )
        + " |"
    )
    print("\n## Worst shared labels by RMSE (n >= "
          f"{args.min_label_n:,})\n")
    flagged = by_label[by_label["n"] >= args.min_label_n]
    print(flagged.head(15).to_string(index = False))

    # -- Plots ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize = (13.33, 7.5))
    _heatmap(
        axes[0], conf_prior, conf_current,
        f"Overture confidence, {prior_version} vs {current_version}",
    )
    axes[0].set_xlabel(f"Confidence, {prior_version}")
    axes[0].set_ylabel(f"Confidence, {current_version}")
    axes[1].hist(
        diff, bins = 120, range = (-1, 1), color = "#2e86c9", log = True,
    )
    _chrome(
        axes[1], xlabel = "Confidence difference (current - prior)",
        ylabel = "POIs (log)", title = "Distribution of differences",
    )
    fig.suptitle(
        "Overture confidence drift, matched GERS ids",
        fontsize = plt.rcParams["font.size"] * 1.44,
    )
    overall_path = viz_dir / "confidence_comparison_overall.png"
    fig.savefig(overall_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

    # Facets: largest labels, plus RMSE outliers (> 2x overall).
    top_labels = list(
        by_label.sort_values("n", ascending = False)
        .head(args.facet_labels)["shared_label"]
    )
    outliers = list(
        flagged[flagged["rmse"] > 2 * metrics["rmse"]]["shared_label"]
    )
    facet_labels = top_labels + [
        lab for lab in outliers if lab not in top_labels
    ]
    n_cols = 4
    n_rows = int(np.ceil(len(facet_labels) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize = (13.33, 3.4 * n_rows),
    )
    grouped = dict(list(matched.groupby("shared_label")))
    for ax, lab in zip(np.ravel(axes), facet_labels):
        sub = grouped[lab]
        flag = " *" if lab in outliers else ""
        _heatmap(
            ax, sub["conf_prior"].to_numpy(),
            sub["conf_current"].to_numpy(),
            f"{lab}{flag} (n={len(sub):,})",
        )
    for ax in np.ravel(axes)[len(facet_labels):]:
        ax.set_visible(False)
    fig.suptitle(
        "Confidence drift by shared label (* = RMSE > 2x overall)",
        fontsize = plt.rcParams["font.size"] * 1.44,
    )
    fig.tight_layout()
    by_label_png = viz_dir / "confidence_comparison_by_label.png"
    fig.savefig(by_label_png, dpi = 300, bbox_inches = "tight")
    plt.close(fig)

    print("\nWrote:")
    for path in [metrics_path, by_label_path, overall_path, by_label_png]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
