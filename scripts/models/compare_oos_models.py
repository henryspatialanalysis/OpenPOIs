"""
Compile out-of-sample cross-validation results across model specifications.

Reads the per-spec OOS metric files written by ``osm_turnover_cv.py`` and
produces comparison + diagnostic tables that rank the specifications by
out-of-sample RMSE and out-of-sample log predictive density (LPD), plus a
plain-language README so a first-time viewer can read them.

Outputs (in ``<model_output base>/<out-dir>/``):
    summary.csv          — one row per spec: OOS RMSE / LPD, ranks, Δ-vs-best
    per_fold_rmse.csv    — fold × spec RMSE (fold-to-fold stability)
    per_fold_lpd.csv     — fold × spec held-out LPD
    subgroup_winners.csv — per shared_label, the best spec by LPD and by RMSE
    README.md            — how to read the tables + the headline verdict
"""
import argparse
from pathlib import Path

import pandas as pd
from config_versioned import Config


config = Config("~/repos/openpois/config.yaml")

# version → human-readable label (λ random-effects structure).
DEFAULT_SPECS = {
    "2026-06-04-oos-full": "Full (amenity + MSA + amenity×MSA + urbanicity)",
    "2026-06-04-oos-msa": "MSA only",
    "2026-06-04-oos-amenity": "Amenity only",
    "2026-06-04-oos-noamenity": "MSA + urbanicity",
    "2026-06-04-oos-nourbanicity": "MSA + amenity + amenity×MSA",
}


def load_runs(model_base: Path, specs: dict[str, str]):
    agg_rows, per_fold, subgroup = [], {}, {}
    for version, label in specs.items():
        run_dir = model_base / version
        agg_fp = run_dir / "oos_metrics_aggregate.csv"
        if not agg_fp.exists():
            print(f"  WARNING: {agg_fp} missing — skipping {version}")
            continue
        agg = pd.read_csv(agg_fp).iloc[0].to_dict()
        agg_rows.append({"version": version, "label": label, **agg})
        per_fold[label] = pd.read_csv(run_dir / "oos_metrics_per_fold.csv")
        sg_fp = run_dir / "oos_metrics_subgroup.csv"
        if sg_fp.exists():
            subgroup[label] = pd.read_csv(sg_fp)
    if not agg_rows:
        raise SystemExit("No OOS runs found — nothing to compare.")
    return pd.DataFrame(agg_rows), per_fold, subgroup


def build_summary(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()
    df["rank_rmse"] = df["rmse_oos_mean"].rank(method = "min").astype(int)
    df["rank_lpd"] = (
        df["lpd_oos_per_obs"].rank(method = "min", ascending = False).astype(int)
    )
    df["rmse_vs_best"] = df["rmse_oos_mean"] - df["rmse_oos_mean"].min()
    df["lpd_per_obs_vs_best"] = (
        df["lpd_oos_per_obs"].max() - df["lpd_oos_per_obs"]
    )
    cols = [
        "label", "version", "n_folds", "rmse_oos_mean", "rank_rmse",
        "rmse_vs_best", "lpd_oos_per_obs", "rank_lpd", "lpd_per_obs_vs_best",
        "lpd_oos_sum",
    ]
    return df.loc[:, cols].sort_values("rmse_oos_mean").reset_index(drop = True)


def fold_table(per_fold: dict[str, pd.DataFrame], value: str) -> pd.DataFrame:
    series = {
        label: df.set_index("fold")[value] for label, df in per_fold.items()
    }
    return pd.DataFrame(series).sort_index().rename_axis("fold").reset_index()


def subgroup_winners(subgroup: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for label, df in subgroup.items():
        sl = df[df["grouping"] == "shared_label"].copy()
        sl["label"] = label
        frames.append(sl)
    if not frames:
        return pd.DataFrame()
    allsg = pd.concat(frames, ignore_index = True)
    rows = []
    for shared_label, g in allsg.groupby("level", observed = True):
        best_lpd = g.loc[g["lppd_per_obs"].idxmax()]
        best_rmse = g.loc[g["rmse"].idxmin()]
        rows.append({
            "shared_label": shared_label,
            "n": int(best_lpd["n"]),
            "best_by_lpd": best_lpd["label"],
            "best_lpd_per_obs": float(best_lpd["lppd_per_obs"]),
            "best_by_rmse": best_rmse["label"],
            "best_rmse": float(best_rmse["rmse"]),
        })
    return pd.DataFrame(rows).sort_values("n", ascending = False).reset_index(drop = True)


def write_readme(out_dir: Path, summary: pd.DataFrame):
    best_rmse = summary.iloc[0]
    best_lpd = summary.sort_values("rank_lpd").iloc[0]
    n_folds = int(summary["n_folds"].iloc[0])
    lines = [
        "# Out-of-sample model comparison",
        "",
        f"Every specification below was refit on a shared set of {n_folds} "
        "holdout folds and scored on the held-out POIs it never saw during "
        "fitting. Because the folds are identical across specs, the numbers are "
        "directly comparable.",
        "",
        "## How to read the metrics",
        "",
        "- **OOS RMSE** (`rmse_oos_mean`): the root-mean-square error between "
        "the model's predicted probability that a POI changed and what actually "
        "happened (0 or 1), averaged over folds. It lives on a 0–1 scale and "
        "**lower is better** — it measures how accurate the point predictions "
        "are.",
        "- **OOS log predictive density** (`lpd_oos_per_obs`): the average "
        "log-probability the model assigned to the outcomes that actually "
        "occurred on held-out data. It is negative, and **higher (closer to "
        "zero) is better**. Unlike RMSE it rewards *calibrated uncertainty*: a "
        "model that is confidently wrong is punished hard. `lpd_oos_sum` is the "
        "same quantity summed over all held-out observations.",
        "- **rank_rmse / rank_lpd**: 1 = best on that metric. **_vs_best** "
        "columns show the gap to the leader (0 = the leader).",
        "",
        "## Tables",
        "",
        "- `summary.csv` — one row per spec, sorted best-RMSE first.",
        "- `per_fold_rmse.csv` / `per_fold_lpd.csv` — the score on each of the "
        f"{n_folds} folds, so you can see whether a spec wins consistently or "
        "just on average.",
        "- `subgroup_winners.csv` — for each taxonomy category (`shared_label`),"
        " which spec predicts its held-out POIs best. Useful for spotting that "
        "(say) the geography terms only help certain categories.",
        "",
        "## Verdict",
        "",
        f"- **Best out-of-sample RMSE:** {best_rmse['label']} "
        f"(`{best_rmse['version']}`), RMSE = {best_rmse['rmse_oos_mean']:.5f}.",
        f"- **Best out-of-sample LPD:** {best_lpd['label']} "
        f"(`{best_lpd['version']}`), LPD/obs = {best_lpd['lpd_oos_per_obs']:.5f}.",
    ]
    if best_rmse["label"] == best_lpd["label"]:
        lines.append(
            f"- Both metrics agree: **{best_rmse['label']}** is the strongest "
            "specification overall."
        )
    else:
        lines.append(
            "- The two metrics disagree: one spec gives sharper point "
            "predictions while another is better calibrated. Prefer LPD when "
            "the downstream use cares about the confidence values, RMSE when it "
            "only cares about the ranking of POIs."
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Compile and rank OOS cross-validation results.",
    )
    parser.add_argument(
        "--versions", nargs = "*", default = None,
        help = "OOS model_output versions to compare (default: the 5 OOS specs).",
    )
    parser.add_argument("--out-dir", default = "oos_comparison_2026-06-04")
    args = parser.parse_args()

    model_base = config.get_dir_path("model_output").parent
    if args.versions:
        specs = {v: DEFAULT_SPECS.get(v, v) for v in args.versions}
    else:
        specs = DEFAULT_SPECS

    agg, per_fold, subgroup = load_runs(model_base, specs)
    summary = build_summary(agg)

    out_dir = model_base / args.out_dir
    out_dir.mkdir(parents = True, exist_ok = True)
    summary.to_csv(out_dir / "summary.csv", index = False)
    fold_table(per_fold, "rmse").to_csv(out_dir / "per_fold_rmse.csv", index = False)
    fold_table(per_fold, "lppd").to_csv(out_dir / "per_fold_lpd.csv", index = False)
    winners = subgroup_winners(subgroup)
    if not winners.empty:
        winners.to_csv(out_dir / "subgroup_winners.csv", index = False)
    write_readme(out_dir, summary)

    print(f"\nWrote comparison tables → {out_dir}")
    print("\nSummary (best RMSE first):")
    print(summary.to_string(index = False))
