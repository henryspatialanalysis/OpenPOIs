"""Fake-data coverage study of the Bayesian calibration bands (Phase 1).

Design doc ``.claude/plans/bayesian-monotone-calibration.md`` §5.2, with arm C
as the fitted main model (execution log, decision 12).

Each round simulates a validation table on the real design: the same phase-1
rows and scores, a known truth, LLM verdicts drawn given the truth, and gold
drawn within verdict class at the round's realized per-segment inclusion
rates (unverifiable censused). The model is refit and the 95% band of m_g(s)
is checked against the truth at fixed evaluation points.

The truth curves are arm C's full-data posterior-mean curves throughout.

``in_family`` (arm C's world)
    y ~ Bernoulli(m(s)). A row is LLM-unverifiable with the design-weighted
    rate u_{g,y} estimated from gold; otherwise the LLM says "exists" with
    probability Se_g if y = 1 and 1 - Sp_g if y = 0. With the fixed-rate arm C
    (the default since decision 21), Se_g and Sp_g are the design-weighted
    forward rates P(verdict | truth, not unverifiable) from gold
    (``forward_silver_rates``); for fits made with an estimated symmetric
    beta_label, Se = Sp = its posterior mean.
``realistic``
    Verdicts drawn from arm A's fitted measurement layer (9 refined classes,
    asymmetric, score-dependent): arm C's symmetric noise is misspecified.
``category``
    A latent category c ~ Bernoulli(0.3) shifts the existence logit by
    +1.0 (c - 0.3) and the unverifiable logit by +1.5 c. The target is the
    c-marginal curve, computed exactly.
``step``
    As ``in_family``, but truly-existing rows at or above the 0.919912
    Overture atom (OSM segment: osm_score >= 0.9) are unverifiable with logit
    +1.2: verdict error that steps with score.

Arm C is fit on every round; arm B (design-weighted, gold only) on the
misspecified rounds as the design-based benchmark. Coverage is reported by
segment and by region (interior slope, flat, edge), because over- and
under-coverage can cancel in a pooled figure (§6.11).

Usage::

    python -u scripts/conflation/simulate_bayes_recovery.py --rounds 20 \\
        --misspecified-rounds 5 --parallel 3 2>&1 | tee <eval dir>/logs/coverage.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bayes_calibration_common as common  # noqa: E402

from openpois.conflation import calibration_bayes as cb  # noqa: E402

EDGE_Q = (0.05, 0.95)
FLAT_SLOPE = 0.1  # |dm/ds| per unit score below which a point counts as flat


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush = True)


def load_truth(out_dir: Path, prepared, tag: str) -> dict:
    """Posterior-mean parameters of a full-data fit (``fits/<tag>``)."""
    draws = np.load(out_dir / "fits" / tag / "draws.npz")
    template = cb.parameter_template(prepared)
    truth = {}
    for name in template:
        arr = np.asarray(draws[name], dtype = float)
        truth[name] = jax.numpy.asarray(arr.reshape((-1,) + arr.shape[2:]).mean(axis = 0))
    return truth


def unverifiable_rates(rows: pd.DataFrame, weights: np.ndarray) -> dict:
    """Design-weighted P(LLM unverifiable | y, segment) from gold."""
    out = {}
    for segment in cb.SEGMENT_ORDER:
        mask = (rows["segment"] == segment).to_numpy() & (weights > 0)
        y = rows.loc[mask, "y"].to_numpy(dtype = float)
        w = weights[mask]
        unv = (rows.loc[mask, "llm_verdict"] == "unverifiable").to_numpy()
        out[segment] = {yv: float(np.sum(w * unv * (y == yv)) / np.sum(w * (y == yv)))
                        for yv in (0, 1)}
    return out


def forward_silver_rates(rows: pd.DataFrame, weights: np.ndarray) -> dict:
    """Design-weighted (Se, Sp) of the definitive LLM verdicts, per segment.

    Se = P(verdict exists | y = 1, verdict definitive) and Sp = P(verdict gone |
    y = 0, verdict definitive), from gold rows weighted by 1 / pi.
    """
    out = {}
    for segment in cb.SEGMENT_ORDER:
        mask = ((rows["segment"] == segment)
                & rows["llm_verdict"].isin(["exists", "gone"])).to_numpy().copy()
        mask &= weights > 0
        y = rows.loc[mask, "y"].to_numpy(dtype = float)
        w = weights[mask]
        exists = (rows.loc[mask, "llm_verdict"] == "exists").to_numpy()
        se = float(np.sum(w * exists * (y == 1)) / np.sum(w * (y == 1)))
        sp = float(np.sum(w * ~exists * (y == 0)) / np.sum(w * (y == 0)))
        out[segment] = (se, sp)
    return out


def confidence_mix(rows: pd.DataFrame) -> dict:
    """Empirical LLM-confidence frequencies by segment x verdict."""
    out = {}
    for (segment, verdict), g in rows.groupby(["segment", "llm_verdict"]):
        freq = g["llm_confidence"].value_counts(normalize = True)
        out[(segment, verdict)] = (freq.index.to_numpy(), freq.to_numpy())
    return out


def evaluation_points(rows: pd.DataFrame, seed: int = 7) -> dict:
    """Fixed points per segment: phase-1 score quantiles (1-D), sampled pairs (2-D)."""
    rng = np.random.default_rng(seed)
    points = {}
    for segment in cb.ONE_D_SEGMENTS:
        s = rows.loc[rows["segment"] == segment, cb.SCORE_COLUMN[segment]]
        points[segment] = {"score": np.unique(np.round(
            np.quantile(s, np.linspace(0.01, 0.99, 40)), 6))}
    matched = rows[rows["segment"] == "matched"]
    pick = rng.choice(len(matched), 150, replace = False)
    points["matched"] = {"osm": matched["osm_score"].to_numpy()[pick],
                         "overture": matched["overture_score"].to_numpy()[pick]}
    return points


def truth_curves(truth: dict, prepared, points: dict, scenario: str) -> dict:
    """True m_g at the evaluation points (c-marginal for ``category``)."""
    one = jax.tree_util.tree_map(lambda x: x[None], truth)
    out = {}
    for segment in cb.SEGMENT_ORDER:
        if segment == "matched":
            osm, ov = points[segment]["osm"], points[segment]["overture"]
        else:
            osm = ov = points[segment]["score"]
        m = cb.curve_draws(one, prepared, segment, osm = osm, overture = ov)[0]
        if scenario == "category":
            f = np.log(m / (1 - m))
            m = (0.3 / (1 + np.exp(-(f + 0.7)))
                 + 0.7 / (1 + np.exp(-(f - 0.3))))
        out[segment] = m
    return out


def region_labels(rows, prepared, truth, points, scenario) -> dict:
    """edge / flat / interior per evaluation point."""
    labels = {}
    h = 1e-3
    for segment in cb.SEGMENT_ORDER:
        if segment in cb.ONE_D_SEGMENTS:
            s = points[segment]["score"]
            ref = rows.loc[rows["segment"] == segment, cb.SCORE_COLUMN[segment]]
            lo, hi = np.quantile(ref, EDGE_Q)
            up = truth_curves(truth, prepared, {**points, segment: {
                "score": np.clip(s + h, 0, 1)}}, scenario)[segment]
            dn = truth_curves(truth, prepared, {**points, segment: {
                "score": np.clip(s - h, 0, 1)}}, scenario)[segment]
            slope = np.abs(up - dn) / (np.clip(s + h, 0, 1) - np.clip(s - h, 0, 1))
            edge = (s < lo) | (s > hi)
            flat = slope < FLAT_SLOPE
        else:
            x, y = points[segment]["osm"], points[segment]["overture"]
            ref = rows[rows["segment"] == "matched"]
            xlo, xhi = np.quantile(ref["osm_score"], EDGE_Q)
            ylo, yhi = np.quantile(ref["overture_score"], EDGE_Q)
            edge = (x < xlo) | (x > xhi) | (y < ylo) | (y > yhi)

            def at(xx, yy):
                return truth_curves(truth, prepared, {**points, "matched": {
                    "osm": xx, "overture": yy}}, scenario)["matched"]

            dx = np.abs(at(np.clip(x + h, 0, 1), y) - at(np.clip(x - h, 0, 1), y))
            dx /= (np.clip(x + h, 0, 1) - np.clip(x - h, 0, 1))
            dy = np.abs(at(x, np.clip(y + h, 0, 1)) - at(x, np.clip(y - h, 0, 1)))
            dy /= (np.clip(y + h, 0, 1) - np.clip(y - h, 0, 1))
            flat = np.minimum(dx, dy) < FLAT_SLOPE
        labels[segment] = np.where(edge, "edge", np.where(flat, "flat",
                                                          "interior"))
    return labels


def simulate(rows: pd.DataFrame, prep_c, truth_c: dict, prep_a, truth_a: dict,
             scenario: str, rng: np.random.Generator, inclusion: dict,
             unv_rate: dict, se_sp: dict, conf_mix: dict) -> pd.DataFrame:
    """One synthetic validation table on the real design."""
    data_c = prep_c.to_jax()
    logits = cb.segment_logits(
        cb.curve_coefficients(truth_c, prep_c.geometry, prep_c.spec), data_c)
    data_a = prep_a.to_jax()
    sim = rows.copy()
    verdicts, confidences, ys, golds = [], [], [], []
    for g, segment in enumerate(cb.SEGMENT_ORDER):
        seg_rows = rows[rows["segment"] == segment]
        n = len(seg_rows)
        f = np.asarray(logits[segment], dtype = float)
        c = np.zeros(n)
        if scenario == "category":
            c = (rng.uniform(size = n) < 0.3).astype(float)
            f = f + 1.0 * (c - 0.3)
        y = (rng.uniform(size = n) < 1 / (1 + np.exp(-f))).astype(int)
        if scenario == "realistic":
            log_theta = np.asarray(cb.class_log_probs(truth_a, prep_a.spec, g,
                                                      data_a[segment]["r"]))
            prob = np.exp(log_theta[np.arange(n), y, :])
            prob /= prob.sum(axis = 1, keepdims = True)
            cls = np.minimum((rng.uniform(size = n)[:, None]
                              > np.cumsum(prob, axis = 1)).sum(axis = 1),
                             prob.shape[1] - 1)
            names = np.asarray(prep_a.spec.class_levels)[cls]
            verdict = np.array([s.split(":")[0] for s in names])
            confidence = np.array([s.split(":")[1] if ":" in s else "high"
                                   for s in names])
        else:
            u = np.where(y == 1, unv_rate[segment][1], unv_rate[segment][0])
            logit_u = np.log(u / (1 - u))
            if scenario == "category":
                logit_u = logit_u + 1.5 * c
            if scenario == "step":
                column = "osm_score" if segment == "osm" else "overture_score"
                cut = 0.9 if segment == "osm" else 0.919912
                logit_u = logit_u + 1.2 * ((seg_rows[column].to_numpy() >= cut)
                                           & (y == 1))
            unverifiable = rng.uniform(size = n) < 1 / (1 + np.exp(-logit_u))
            se, sp = se_sp[segment]
            says_exists = rng.uniform(size = n) < np.where(y == 1, se, 1 - sp)
            verdict = np.where(unverifiable, "unverifiable",
                               np.where(says_exists, "exists", "gone"))
            confidence = np.empty(n, dtype = object)
            for v in cb.VERDICTS:
                idx = np.flatnonzero(verdict == v)
                levels, freq = conf_mix.get((segment, v),
                                            (np.array(["high"]), np.array([1.0])))
                confidence[idx] = rng.choice(levels, len(idx), p = freq)
        # Phase 2 per round, at each round's realized inclusion (execution log,
        # decision 23).
        gold = np.zeros(n, dtype = bool)
        round_of = seg_rows["validation_round"].astype(str).to_numpy()
        for round_id, rates in inclusion[segment].items():
            for v in cb.VERDICTS:
                idx = np.flatnonzero((verdict == v) & (round_of == round_id))
                rate = rates.get(v, 1.0)
                k = len(idx) if v == "unverifiable" else int(round(rate * len(idx)))
                if k:
                    gold[rng.choice(idx, k, replace = False)] = True
        verdicts.append(verdict)
        confidences.append(confidence)
        ys.append(y)
        golds.append(gold)
    sim["llm_verdict"] = np.concatenate(verdicts)
    sim["llm_confidence"] = np.concatenate(confidences).astype(str)
    sim["gold"] = np.concatenate(golds)
    sim["y"] = np.where(sim["gold"], np.concatenate(ys).astype(float), np.nan)
    return sim


def realized_inclusion(rows: pd.DataFrame) -> dict:
    """{segment: {round: {verdict: share of phase-1 rows sent to gold}}}."""
    out = {}
    for segment in cb.SEGMENT_ORDER:
        seg = rows[rows["segment"] == segment]
        out[segment] = {
            str(round_id): {
                v: float(g.loc[g["llm_verdict"] == v, "gold"].mean())
                for v in cb.VERDICTS if (g["llm_verdict"] == v).any()
            }
            for round_id, g in seg.groupby("validation_round", sort = False)
        }
    return out


def run_worker(args) -> None:
    from openpois.models.jax_core import enable_high_precision

    enable_high_precision()
    config = common.load_config()
    fit_config = common.fit_config_from(config)
    _, metadata = common.load_handoff(config)
    out_dir = common.eval_dir(config, metadata, args.out_dir)
    # Simulate on the table the truth fit was made on (all its pooled rounds).
    rows = common.fit_rows(config, out_dir, "armC")
    target = (out_dir / "coverage"
              / f"{args.scenario}_r{args.round:02d}_arm{args.arm}.parquet")
    if target.exists():
        log(f"{target.name} exists, skipping")
        return
    prep_c = cb.prepare_data(rows, cb.ModelSpec(arm = "C"), fit_config = fit_config)
    truth_c = load_truth(out_dir, prep_c, "armC")
    prep_a = cb.prepare_data(rows, cb.ModelSpec(arm = "A"), fit_config = fit_config)
    truth_a = load_truth(out_dir, prep_a, "armA")
    if "logit_beta_label" in truth_c:
        beta = float(1 / (1 + np.exp(-float(truth_c["logit_beta_label"]))))
        se_sp = {s: (beta, beta) for s in cb.SEGMENT_ORDER}
    else:
        se_sp = forward_silver_rates(rows, common.design_weights(rows, fit_config))
    weights = common.design_weights(rows, fit_config)
    points = evaluation_points(rows)
    true_m = truth_curves(truth_c, prep_c, points, args.scenario)
    regions = region_labels(rows, prep_c, truth_c, points, args.scenario)
    rng = np.random.default_rng(args.seed * 1000 + args.round * 7 + {
        "in_family": 0, "step": 1, "category": 2, "realistic": 3}[args.scenario])
    sim = simulate(rows, prep_c, truth_c, prep_a, truth_a, args.scenario, rng,
                   realized_inclusion(rows), unverifiable_rates(rows, weights),
                   se_sp, confidence_mix(rows))
    spec = cb.ModelSpec(arm = args.arm)
    prepared = cb.prepare_data(sim, spec, knots = prep_c.knots,
                               fit_config = fit_config)
    started = time.time()
    result = cb.fit(prepared, num_warmup = args.warmup, num_samples = args.samples,
                    num_chains = args.chains, seed = args.seed + args.round,
                    adaptation_kwargs = common.adaptation_kwargs_from(args))
    parts = []
    for segment in cb.SEGMENT_ORDER:
        if segment == "matched":
            osm, ov = points[segment]["osm"], points[segment]["overture"]
        else:
            osm = ov = points[segment]["score"]
        summary = cb.summarize_draws(cb.curve_draws(result.draws, prepared,
                                                    segment, osm = osm,
                                                    overture = ov))
        parts.append(pd.DataFrame({
            "scenario": args.scenario, "round": args.round, "arm": args.arm,
            "segment": segment, "point": np.arange(len(summary)),
            "truth": true_m[segment], "mean": summary["mean"],
            "lower": summary["lower"], "upper": summary["upper"],
            "region": regions[segment],
        }))
    frame = pd.concat(parts, ignore_index = True)
    frame["covered"] = (frame["lower"] <= frame["truth"]) & (
        frame["truth"] <= frame["upper"])
    diag = result.diagnostics
    info = {"minutes": (time.time() - started) / 60, "max_rhat": diag["max_rhat"],
            "min_ess_bulk": diag["min_ess_bulk"],
            "divergences": int(sum(diag["divergences_per_chain"])),
            "n_gold": int(sim["gold"].sum()),
            "se_sp_true": {s: list(v) for s, v in se_sp.items()}}
    target.with_suffix(".json").write_text(json.dumps(info))
    frame.to_parquet(target, index = False)
    log(f"{target.name}: coverage {frame['covered'].mean():.3f} "
        f"({json.dumps(info)})")


def summarize(out_dir: Path) -> dict:
    files = sorted((out_dir / "coverage").glob("*_arm*.parquet"))
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index = True)
    frame["width"] = frame["upper"] - frame["lower"]
    frame["error"] = frame["mean"] - frame["truth"]
    group = ["scenario", "arm", "segment", "region"]
    by_region = frame.groupby(group).agg(
        coverage = ("covered", "mean"), width = ("width", "mean"),
        bias = ("error", "mean"),
        rmse = ("error", lambda e: float(np.sqrt(np.mean(e ** 2)))),
        n_points = ("covered", "size"), rounds = ("round", "nunique"),
    ).reset_index()
    by_segment = frame.groupby(["scenario", "arm", "segment"]).agg(
        coverage = ("covered", "mean"), width = ("width", "mean"),
        bias = ("error", "mean"), rounds = ("round", "nunique"),
    ).reset_index()
    by_region.to_csv(out_dir / "coverage" / "coverage_by_region.csv", index = False)
    by_segment.to_csv(out_dir / "coverage" / "coverage_by_segment.csv",
                      index = False)
    infos = [json.loads(p.with_suffix(".json").read_text()) for p in files]
    return {"by_segment": by_segment.to_dict(orient = "records"),
            "by_region": by_region.to_dict(orient = "records"),
            "fits": len(files),
            "max_rhat": max(i["max_rhat"] for i in infos),
            "divergences": sum(i["divergences"] for i in infos),
            "mean_minutes": float(np.mean([i["minutes"] for i in infos]))}


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    common.add_spec_arguments(parser)
    parser.add_argument("--rounds", type = int, default = 20)
    parser.add_argument("--misspecified-rounds", type = int, default = 5)
    parser.add_argument("--parallel", type = int, default = 4)
    parser.add_argument("--threads-per-worker", type = int, default = 3)
    parser.add_argument("--out-dir", default = None)
    parser.add_argument("--worker", action = "store_true")
    parser.add_argument("--scenario", default = "in_family")
    parser.add_argument("--round", type = int, default = 0)
    parser.add_argument("--summarize-only", action = "store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
        return
    config = common.load_config()
    _, metadata = common.load_handoff(config)
    out_dir = common.eval_dir(config, metadata, args.out_dir)
    if not args.summarize_only:
        jobs = [("in_family", r, "C") for r in range(args.rounds)]
        for scenario in ("realistic", "category", "step"):
            for r in range(args.misspecified_rounds):
                jobs += [(scenario, r, "C"), (scenario, r, "B")]
        jobs = [j for j in jobs if not (
            out_dir / "coverage" / f"{j[0]}_r{j[1]:02d}_arm{j[2]}.parquet").exists()]
        log(f"{len(jobs)} coverage fits to run, {args.parallel} at a time")
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(args.threads_per_worker)
        passthrough = ["--warmup", str(args.warmup), "--samples", str(args.samples),
                       "--chains", str(args.chains), "--seed", str(args.seed)]
        if args.out_dir:
            passthrough += ["--out-dir", args.out_dir]
        if args.target_accept:
            passthrough += ["--target-accept", str(args.target_accept)]
        if args.dense_mass:
            passthrough += ["--dense-mass"]
        running, failures = [], []
        while jobs or running:
            while jobs and len(running) < args.parallel:
                scenario, rnd, arm = jobs.pop(0)
                name = f"coverage_{scenario}_r{rnd:02d}_arm{arm}"
                handle = open(out_dir / "logs" / f"{name}.log", "w")
                proc = subprocess.Popen(
                    [sys.executable, "-u", __file__, "--worker", "--scenario",
                     scenario, "--round", str(rnd), "--arm", arm, *passthrough],
                    stdout = handle, stderr = subprocess.STDOUT, env = env)
                running.append((proc, handle, name))
                log(f"started {name} (pid {proc.pid})")
            time.sleep(10)
            still = []
            for proc, handle, name in running:
                if proc.poll() is None:
                    still.append((proc, handle, name))
                    continue
                handle.close()
                if proc.returncode != 0:
                    failures.append(name)
                log(f"finished {name}: "
                    f"{'ok' if proc.returncode == 0 else 'FAILED'}")
            running = still
        if failures:
            log(f"FAILED coverage fits: {failures}")
    summary = summarize(out_dir)
    (out_dir / "coverage" / "coverage_summary.json").write_text(
        json.dumps(summary, indent = 2, default = str))
    log("coverage summary written")
    for row in summary["by_segment"]:
        log(f"  {row['scenario']:9s} arm {row['arm']} {row['segment']:8s} "
            f"coverage {row['coverage']:.3f} width {row['width']:.3f} "
            f"bias {row['bias']:+.4f} ({row['rounds']} rounds)")


if __name__ == "__main__":
    main()
