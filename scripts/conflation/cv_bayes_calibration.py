"""Ten-fold cross-validation of the Bayesian calibration arms (Phase 1).

Design doc ``.claude/plans/bayesian-monotone-calibration.md`` §5.4-§5.4a.

- **Folds.** Gold rows are folded within production refined class, per segment,
  exactly as ``calibration_fit.cross_fit_predictions`` folds them (asserted).
  Fold f holds out fold f of every segment in one joint Bayesian fit. A
  held-out row keeps its LLM verdict and loses y and gold status.
- **Pooled rounds** (execution log, decision 23). Only the current round is
  folded, held out and scored. Rows of earlier pooled rounds stay in every
  training fit, and the comparators see the current round only, as production
  does.
- **Comparators.** The production cross-fit, with the October 2026
  configuration (interaction index for matched, native 1-D curves), bins on
  the round's own conflation population. Each comparator is scored as
  published (40-bin lookup) and as its curve.
- **Scores.** Held-out rows are weighted by 1/pi_class from the training
  split (the comparator's own weights). Brier, RMSE = sqrt(Brier), log score,
  LPD and CORP MCB/DSC per segment. The pooled figure gives each segment equal
  weight (decision 9).
- **Uncertainty.** A paired bootstrap within fold x refined class.

The orchestrator (default mode) launches one worker subprocess per (arm, fold)
up to ``--parallel`` at a time, then assembles. Workers write
``cv/<arm tag>/fold<k>.parquet`` and are skipped when it already exists, so a
killed run resumes.

Usage::

    python -u scripts/conflation/cv_bayes_calibration.py --arms A,B,C \\
        --parallel 4 2>&1 | tee <eval dir>/logs/cv.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bayes_calibration_common as common  # noqa: E402

from openpois.conflation import calibration_bayes as cb  # noqa: E402
from openpois.conflation import calibration_fit as cf  # noqa: E402

EPS = 1e-12


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush = True)


def arm_tag(arm: str, suffix: str = "") -> str:
    return f"arm{arm}{suffix}"


# ---------------------------------------------------------------------------
# Worker: one (arm, fold) fit
# ---------------------------------------------------------------------------

def run_worker(args) -> None:
    from openpois.models.jax_core import enable_high_precision

    enable_high_precision()
    config = common.load_config()
    fit_config = common.fit_config_from(config)
    rows, metadata = common.load_handoff(config, args.pooled_rounds)
    out_dir = common.eval_dir(config, metadata, args.out_dir)
    spec = common.spec_from_args(args)
    tag = arm_tag(spec.arm, args.tag_suffix)
    target = out_dir / "cv" / tag / f"fold{args.fold}.parquet"
    if target.exists():
        log(f"{tag} fold {args.fold}: exists, skipping")
        return
    target.parent.mkdir(parents = True, exist_ok = True)
    folds = current_round_folds(rows, metadata, fit_config, args.folds)[0]
    held = folds == args.fold
    knots = cb.segment_knots(rows, spec)
    # Arm C fixed noise: prepare_data computes the rates from the training gold
    # only (held-out labels never inform them), every pooled round included.
    prepared = cb.prepare_data(rows, spec, knots = knots, held_out = held,
                               fit_config = fit_config)
    started = time.time()
    result = cb.fit(prepared, num_warmup = args.warmup, num_samples = args.samples,
                    num_chains = args.chains, seed = args.seed + args.fold,
                    adaptation_kwargs = common.adaptation_kwargs_from(args))
    diag = result.diagnostics
    parts = []
    for segment in cb.SEGMENT_ORDER:
        mask = held & (rows["segment"] == segment).to_numpy()
        idx = np.flatnonzero(mask)
        seg = rows.iloc[idx]
        draws = cb.curve_draws(result.draws, prepared, segment,
                               osm = seg["osm_score"].to_numpy(),
                               overture = seg["overture_score"].to_numpy())
        summary = cb.summarize_draws(draws)
        parts.append(pd.DataFrame({"row_id": idx, "segment": segment,
                                   "fold": args.fold,
                                   "p_mean": summary["mean"].to_numpy(),
                                   "p_lower": summary["lower"].to_numpy(),
                                   "p_upper": summary["upper"].to_numpy()}))
    frame = pd.concat(parts, ignore_index = True)
    info = {
        "tag": tag, "fold": args.fold, "minutes": (time.time() - started) / 60,
        "max_rhat": diag["max_rhat"], "min_ess_bulk": diag["min_ess_bulk"],
        "min_ess_tail": diag["min_ess_tail"],
        "divergences": int(sum(diag["divergences_per_chain"])),
        "min_ebfmi": float(min(diag["ebfmi_per_chain"])),
        "treedepth_saturated": diag["treedepth_saturated"],
        "mean_steps": diag["mean_steps"], "mean_accept": diag["mean_accept"],
        "n_held_out": int(held.sum()),
    }
    if "logit_beta_label" in result.draws:
        beta = 1 / (1 + np.exp(-np.asarray(result.draws["logit_beta_label"])))
        info["beta_label_mean"] = float(beta.mean())
    (target.parent / f"fold{args.fold}.json").write_text(json.dumps(info, indent = 2))
    frame.to_parquet(target, index = False)
    log(f"{tag} fold {args.fold} done: {json.dumps(info)}")


def current_round_folds(rows, metadata, fit_config, n_folds) -> tuple:
    """(folds over all rows, current-round positions, current-round classes).

    Folds and classes are built on the current round alone, exactly as the
    production cross-fit sees it; rows of pooled rounds get fold -1, so they
    are never held out. With one round this is the plain fold assignment.
    """
    current_idx = np.flatnonzero(common.current_round_mask(rows, metadata))
    current = rows.iloc[current_idx].reset_index(drop = True)
    classes = cb.production_classes(current, fit_config)
    folds = np.full(len(rows), -1)
    folds[current_idx] = cb.assign_segment_folds(current, classes, n_folds,
                                                 fit_config.rng_seed)
    return folds, current_idx, current, classes


# ---------------------------------------------------------------------------
# Comparators: the production cross-fit
# ---------------------------------------------------------------------------

def comparator_predictions(rows, classes, folds, fit_config, n_folds,
                           population_path: Path) -> pd.DataFrame:
    populations = common.population_scores(population_path)
    parts = []
    for segment in cb.SEGMENT_ORDER:
        mask = (rows["segment"] == segment).to_numpy()
        idx = np.flatnonzero(mask)
        seg = rows.iloc[idx].reset_index(drop = True)
        seg_classes = classes[mask].reset_index(drop = True)
        if segment == "matched":
            index_mode = fit_config.matched_index_mode
            seeded = seg.assign(score = cf.segment_scores(seg, segment, None,
                                                          "average"))
        else:
            index_mode = "pool"
            seeded = seg.assign(score = cf.segment_scores(seg, segment))
        started = time.time()
        out = cf.cross_fit_predictions(
            seeded, seg_classes, fit_config, n_folds = n_folds, segment = segment,
            index_mode = index_mode, population = populations[segment],
        )
        gold = seg["gold"].to_numpy(dtype = bool)
        if out["n_gold"] != gold.sum():
            raise SystemExit(f"{segment}: comparator dropped gold rows")
        if not np.array_equal(out["fold"], folds[idx][gold]):
            raise SystemExit(f"{segment}: comparator folds differ from ours")
        parts.append(pd.DataFrame({
            "row_id": idx[gold], "segment": segment, "fold": out["fold"],
            "y": out["actual"], "weight": out["weight"],
            "cls": seg_classes[gold].to_numpy(),
            "production_published": out["predicted"],
            "production_curve": out["predicted_grid"],
        }))
        log(f"comparator {segment} ({index_mode}): {int(gold.sum())} gold, "
            f"{time.time() - started:.0f}s")
    return pd.concat(parts, ignore_index = True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def row_scores(p, y) -> dict:
    p = np.clip(np.asarray(p, dtype = float), EPS, 1 - EPS)
    y = np.asarray(y, dtype = float)
    return {"sq": (y - p) ** 2,
            "lpd": y * np.log(p) + (1 - y) * np.log(1 - p)}


def segment_metrics(frame: pd.DataFrame, column: str) -> dict:
    """Design-weighted scores for one segment's held-out rows."""
    w = frame["weight"].to_numpy(dtype = float)
    y = frame["y"].to_numpy(dtype = float)
    p = frame[column].to_numpy(dtype = float)
    s = row_scores(p, y)
    brier = float(np.sum(w * s["sq"]) / w.sum())
    per_fold_rmse = []
    for _, sub in frame.groupby("fold"):
        ws = sub["weight"].to_numpy(dtype = float)
        ss = row_scores(sub[column], sub["y"])
        per_fold_rmse.append(float(np.sqrt(np.sum(ws * ss["sq"]) / ws.sum())))
    w_norm = w * len(w) / w.sum()
    corp = cf.scoring_rules(p, y, w)
    return {
        "n": int(len(frame)), "brier": brier, "rmse": float(np.sqrt(brier)),
        "rmse_mbg_mean_of_folds": float(np.mean(per_fold_rmse)),
        "log_score": float(-np.sum(w * s["lpd"]) / w.sum()),
        "lpd_per_row": float(np.sum(w * s["lpd"]) / w.sum()),
        "lpd_mbg_sum": float(np.sum(w_norm * s["lpd"])),
        "mcb": corp["mcb"], "dsc": corp["dsc"], "unc": corp["unc"],
    }


def pooled(metrics: dict) -> dict:
    """Equal-weight mean over segments (decision 9)."""
    keys = ["brier", "rmse", "rmse_mbg_mean_of_folds", "log_score",
            "lpd_per_row"]
    out = {k: float(np.mean([metrics[s][k] for s in cb.SEGMENT_ORDER]))
           for k in keys}
    out["rmse_of_pooled_brier"] = float(np.sqrt(out["brier"]))
    out["lpd_mbg_sum"] = float(np.sum([metrics[s]["lpd_mbg_sum"]
                                       for s in cb.SEGMENT_ORDER]))
    return out


def paired_bootstrap(frame: pd.DataFrame, candidate: str, reference: str,
                     reps: int, seed: int) -> dict:
    """Candidate minus reference, resampling rows within fold x class.

    Returns per segment and pooled (equal segment weights) intervals for
    delta Brier, Brier ratio and delta LPD per row.
    """
    rng = np.random.default_rng(seed)
    out = {}
    stats = {s: {"d_brier": [], "ratio": [], "d_lpd": []} for s in
             list(cb.SEGMENT_ORDER) + ["pooled"]}
    groups = {}
    for segment in cb.SEGMENT_ORDER:
        sub = frame[frame["segment"] == segment].reset_index(drop = True)
        strata = sub["fold"].astype(str) + "|" + sub["cls"].astype(str)
        groups[segment] = (sub, [np.flatnonzero(strata == k)
                                 for k in strata.unique()])
    for _ in range(reps):
        pooled_vals = {"cb": [], "rb": [], "cl": [], "rl": []}
        for segment, (sub, strata) in groups.items():
            idx = np.concatenate([g[rng.integers(0, len(g), len(g))]
                                  for g in strata])
            w = sub["weight"].to_numpy()[idx]
            y = sub["y"].to_numpy()[idx]
            c = row_scores(sub[candidate].to_numpy()[idx], y)
            r = row_scores(sub[reference].to_numpy()[idx], y)
            cb_ = np.sum(w * c["sq"]) / w.sum()
            rb_ = np.sum(w * r["sq"]) / w.sum()
            cl_ = np.sum(w * c["lpd"]) / w.sum()
            rl_ = np.sum(w * r["lpd"]) / w.sum()
            stats[segment]["d_brier"].append(cb_ - rb_)
            stats[segment]["ratio"].append(cb_ / rb_)
            stats[segment]["d_lpd"].append(cl_ - rl_)
            for k, v in zip(("cb", "rb", "cl", "rl"), (cb_, rb_, cl_, rl_)):
                pooled_vals[k].append(v)
        pc, pr = np.mean(pooled_vals["cb"]), np.mean(pooled_vals["rb"])
        stats["pooled"]["d_brier"].append(pc - pr)
        stats["pooled"]["ratio"].append(pc / pr)
        stats["pooled"]["d_lpd"].append(np.mean(pooled_vals["cl"])
                                        - np.mean(pooled_vals["rl"]))
    for key, values in stats.items():
        out[key] = {
            name: {"mean": float(np.mean(v)),
                   "lower": float(np.quantile(v, 0.025)),
                   "upper": float(np.quantile(v, 0.975))}
            for name, v in values.items()
        }
    return out


def assemble(out_dir: Path, arm_tags: list, reps: int, seed: int) -> dict:
    comparators = pd.read_parquet(out_dir / "cv" / "comparators.parquet")
    frame = comparators.copy()
    columns = ["production_published", "production_curve"]
    fold_info = {}
    for tag in arm_tags:
        parts = sorted((out_dir / "cv" / tag).glob("fold*.parquet"))
        preds = pd.concat([pd.read_parquet(p) for p in parts], ignore_index = True)
        frame = frame.merge(
            preds[["row_id", "p_mean", "p_lower", "p_upper"]].rename(columns = {
                "p_mean": tag, "p_lower": f"{tag}_lower",
                "p_upper": f"{tag}_upper"}),
            on = "row_id", how = "left")
        columns.append(tag)
        fold_info[tag] = [json.loads(p.with_suffix(".json").read_text())
                          for p in parts]
    missing = frame[columns].isna().any(axis = 1)
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} held-out rows lack a prediction")
    frame.to_parquet(out_dir / "cv" / "oof_all.parquet", index = False)

    results = {"n_rows": int(len(frame)), "models": {}, "pairs": {},
               "fold_info": fold_info}
    for column in columns:
        per_segment = {s: segment_metrics(frame[frame["segment"] == s], column)
                       for s in cb.SEGMENT_ORDER}
        results["models"][column] = {**per_segment, "pooled": pooled(per_segment)}
    pairs = []
    for tag in arm_tags:
        pairs += [(tag, "production_published"), (tag, "production_curve")]
    # Arm C is the main model (execution log, decision 12): compare it with
    # every other arm; "armC-vs-armA" answers whether the full measurement
    # layer adds out-of-sample performance.
    main_tag = next((t for t in arm_tags if t.startswith("armC")), arm_tags[0])
    pairs += [(main_tag, t) for t in arm_tags if t != main_tag]
    for offset, (cand, ref) in enumerate(pairs):
        results["pairs"][f"{cand}-vs-{ref}"] = paired_bootstrap(
            frame, cand, ref, reps, seed + offset)
    # Coverage of held-out y is not meaningful for binary outcomes; report the
    # mean band width of the Bayesian arms instead.
    for tag in arm_tags:
        results["models"][tag]["mean_band_width"] = {
            s: float((frame.loc[frame["segment"] == s, f"{tag}_upper"]
                      - frame.loc[frame["segment"] == s, f"{tag}_lower"]).mean())
            for s in cb.SEGMENT_ORDER}
    return results


def results_markdown(results: dict, arm_tags: list) -> str:
    lines = ["# Ten-fold cross-validation: Bayesian arms vs production", ""]
    lines.append("Design-weighted held-out scores (weights 1/pi from the training "
                 "split). Pooled = equal weight per segment. Lower Brier/RMSE/log "
                 "score is better; higher (closer to 0) LPD is better; higher DSC "
                 "and lower MCB are better.")
    lines.append("")
    header = ("| model | segment | n | Brier | RMSE | RMSE (mbg: mean of folds) | "
              "log score | LPD/row | LPD (mbg sum) | MCB | DSC |")
    lines += [header, "|" + "---|" * 11]
    for model, by_seg in results["models"].items():
        for segment in list(cb.SEGMENT_ORDER) + ["pooled"]:
            m = by_seg[segment]
            lines.append(
                f"| {model} | {segment} | {m.get('n', '')} | {m['brier']:.5f} | "
                f"{m['rmse']:.4f} | {m['rmse_mbg_mean_of_folds']:.4f} | "
                f"{m['log_score']:.4f} | {m['lpd_per_row']:.4f} | "
                f"{m['lpd_mbg_sum']:.1f} | {m.get('mcb', float('nan')):.5f} | "
                f"{m.get('dsc', float('nan')):.5f} |")
    lines += ["", "## Paired comparisons (candidate minus reference; 95% bootstrap)",
              "",
              "| pair | segment | Δ Brier | Brier ratio | Δ LPD/row |",
              "|---|---|---|---|---|"]
    for pair, by_seg in results["pairs"].items():
        for segment in list(cb.SEGMENT_ORDER) + ["pooled"]:
            s = by_seg[segment]

            def fmt(d, digits = 5):
                return (f"{d['mean']:+.{digits}f} [{d['lower']:+.{digits}f}, "
                        f"{d['upper']:+.{digits}f}]")

            ratio = s["ratio"]
            lines.append(
                f"| {pair} | {segment} | {fmt(s['d_brier'])} | "
                f"{ratio['mean']:.3f} [{ratio['lower']:.3f}, {ratio['upper']:.3f}]"
                f" | {fmt(s['d_lpd'], 4)} |")
    lines += ["", "## Fold fits", ""]
    for tag, infos in results["fold_info"].items():
        mins = [i["minutes"] for i in infos]
        lines.append(
            f"- {tag}: {len(infos)} folds, {np.mean(mins):.1f} min/fold; max R-hat "
            f"{max(i['max_rhat'] for i in infos):.3f}; min ESS bulk "
            f"{min(i['min_ess_bulk'] for i in infos):.0f}; divergences "
            f"{sum(i['divergences'] for i in infos)}; min E-BFMI "
            f"{min(i['min_ebfmi'] for i in infos):.2f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    common.add_spec_arguments(parser)
    parser.add_argument("--arms", default = "A,B,C")
    parser.add_argument("--folds", type = int, default = 10)
    parser.add_argument("--parallel", type = int, default = 4)
    parser.add_argument("--threads-per-worker", type = int, default = 3)
    parser.add_argument("--out-dir", default = None)
    parser.add_argument("--tag-suffix", default = "")
    parser.add_argument("--bootstrap-reps", type = int, default = 2000)
    parser.add_argument("--worker", action = "store_true")
    parser.add_argument("--fold", type = int, default = None)
    parser.add_argument("--assemble-only", action = "store_true")
    parser.add_argument("--arm-flags", action = "append", default = [],
                        help = ("Extra worker flags for one arm, as "
                                "'A=--no-pooling --class-scheme merged6'."))
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return

    config = common.load_config()
    fit_config = common.fit_config_from(config)
    rows, metadata = common.load_handoff(config, args.pooled_rounds)
    out_dir = common.eval_dir(config, metadata, args.out_dir)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    tags = [arm_tag(a, args.tag_suffix) for a in arms]
    folds, current_idx, current, classes = current_round_folds(
        rows, metadata, fit_config, args.folds)
    arm_flags = {}
    for item in args.arm_flags:
        arm, _, flags = item.partition("=")
        arm_flags[arm.strip()] = flags.split()
    if arm_flags:
        log(f"per-arm flags: {arm_flags}")

    comparator_path = out_dir / "cv" / "comparators.parquet"
    if not comparator_path.exists():
        population_path = (common.conflation_root(config)
                           / str(metadata["conflation_version"])
                           / "conflated_cd.parquet")
        log(f"comparators: production cross-fit, {args.folds} folds, bins on "
            f"{population_path}")
        comparators = comparator_predictions(current, classes, folds[current_idx],
                                             fit_config, args.folds, population_path)
        # Comparator row ids index the current round; map them to the pooled table.
        comparators["row_id"] = current_idx[comparators["row_id"].to_numpy()]
        comparators.to_parquet(comparator_path, index = False)

    if not args.assemble_only:
        # Forward every model flag to the workers, minus orchestrator options.
        cleaned, skip = [], False
        for token in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if token in ("--arms", "--parallel", "--threads-per-worker",
                         "--bootstrap-reps", "--arm-flags"):
                skip = True
                continue
            if token.startswith(("--arms=", "--parallel=", "--threads-per-worker=",
                                 "--bootstrap-reps=", "--arm-flags=")) \
                    or token == "--assemble-only":
                continue
            if token.startswith("--arm=") or token == "--arm":
                raise SystemExit("Use --arms, not --arm, with the orchestrator")
            cleaned.append(token)
        jobs = [(arm, fold) for arm in arms for fold in range(args.folds)
                if not (out_dir / "cv" / arm_tag(arm, args.tag_suffix)
                        / f"fold{fold}.parquet").exists()]
        log(f"{len(jobs)} fold fits to run, {args.parallel} at a time")
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(args.threads_per_worker)
        running = []
        failures = []
        while jobs or running:
            while jobs and len(running) < args.parallel:
                arm, fold = jobs.pop(0)
                log_name = f"cv_{arm_tag(arm, args.tag_suffix)}_fold{fold}.log"
                log_path = out_dir / "logs" / log_name
                cmd = [sys.executable, "-u", __file__, "--worker", "--arm", arm,
                       "--fold", str(fold), *cleaned,
                       *arm_flags.get(arm, [])]
                handle = open(log_path, "w")
                proc = subprocess.Popen(cmd, stdout = handle,
                                        stderr = subprocess.STDOUT, env = env)
                running.append((proc, handle, arm, fold))
                log(f"started {arm_tag(arm, args.tag_suffix)} fold {fold} "
                    f"(pid {proc.pid})")
            time.sleep(10)
            still = []
            for proc, handle, arm, fold in running:
                if proc.poll() is None:
                    still.append((proc, handle, arm, fold))
                    continue
                handle.close()
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode != 0:
                    failures.append((arm, fold))
                log(f"finished {arm_tag(arm, args.tag_suffix)} fold {fold}: {status}")
            running = still
        if failures:
            raise SystemExit(f"Fold fits failed: {failures}")

    results = assemble(out_dir, tags, args.bootstrap_reps, fit_config.rng_seed + 4100)
    results["git"] = common.git_state()
    suffix = args.tag_suffix
    (out_dir / "cv" / f"cv_results{suffix}.json").write_text(
        json.dumps(results, indent = 2, default = str))
    (out_dir / "cv" / f"cv_results{suffix}.md").write_text(
        results_markdown(results, tags))
    pooled_rows = {m: r["pooled"]["brier"] for m, r in results["models"].items()}
    log(f"CV assembled: pooled Brier {json.dumps(pooled_rows)}")


if __name__ == "__main__":
    main()
