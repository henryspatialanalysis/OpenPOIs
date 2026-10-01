"""Assemble ``fit_report.md`` for the Bayesian calibration prototype (Phase 1).

Reads what the other three scripts wrote under the evaluation directory:

- ``fits/<tag>/summary.json`` (full-data fits: main arms and sensitivity runs)
- ``cv/cv_results.json`` (10-fold CV)
- ``coverage/coverage_summary.json`` (fake-data coverage study)

and writes ``fit_report.md`` beside them (design doc §5.6). It does no model
fitting.

When the main tag is a single-segment fit (``summary.json`` lists fewer than three
``segments``; the production ``MODE=mixture`` run), the report reads the three
segment fits that share its prefix (``mixture_overture``, ``mixture_osm``,
``mixture_matched``) and shows each segment's fit, acceptance, forward rates and
deployed impact as one model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bayes_calibration_common as common  # noqa: E402

# Arm C is the main model (execution log, decision 12); A and B are comparators.
# The fixed-rate mixture (armC_mixture) is the model the monthly update runs from
# October 2026; `--main armC_mixture` makes it the report's main model.
MAIN = "armC"
MAIN_TAGS = ("armC", "armC_mixture", "armA", "armB")
# Production segment fits (run_bayes_phase1.sh MODE=mixture), one per segment.
SEGMENT_TAGS = {"overture": "mixture_overture", "osm": "mixture_osm",
                "matched": "mixture_matched"}
SEGMENTS = ("overture", "osm", "matched")
SENSITIVITY = {
    "S2C_asym": "asymmetric label noise (Se, Sp; priors at 0.998 / 0.913)",
    "S3C_exact": "silver labels treated as exact (β = 1)",
    "S3C_weak": "weak β prior, Beta(9, 1)",
    "S4_cubic": "cubic basis",
    "S5_gap01": "max knot gap 0.1",
    "S5_equal20": "20 equally spaced knot intervals",
    "S6_tau2": "τ prior scales × 2",
    "S6_tauhalf": "τ prior scales × ½",
    "S6_taumatched4": "τ_M prior scale × 4 (matched only)",
    "S7_spacing": "spacing-scaled random walk",
}


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(x, digits = 4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{digits}f}"


def diag_line(summary: dict) -> str:
    d, acc = summary["diagnostics"], summary["acceptance"]
    cc = summary["curve_convergence"]
    return (f"{summary['fit_minutes']:.1f} min; max R̂ {d['max_rhat']:.4f}; "
            f"min ESS bulk/tail {d['min_ess_bulk']:.0f}/{d['min_ess_tail']:.0f}; "
            f"curve max R̂ {cc['max_rhat']:.4f}, min ESS "
            f"{min(cc['min_ess_bulk'], cc['min_ess_tail']):.0f}; divergences "
            f"{sum(d['divergences_per_chain'])}; min E-BFMI "
            f"{min(d['ebfmi_per_chain']):.2f}; tree-depth hits "
            f"{d['treedepth_saturated']}; mean steps {d['mean_steps']:.0f}; "
            f"**acceptance {'PASS' if acc['all'] else 'FAIL'}**")


def segment_fit_tags(main_tag: str, main: dict) -> dict | None:
    """{segment: tag} when ``main_tag`` is a single-segment fit, else None.

    The sibling tags share the main tag's prefix (``mixture_matched`` ->
    ``mixture_overture``, ``mixture_osm``); a tag without a segment suffix
    falls back to ``SEGMENT_TAGS``.
    """
    segments = (main or {}).get("segments")
    if not segments or len(segments) == len(SEGMENTS):
        return None
    for segment in segments:
        suffix = f"_{segment}"
        if main_tag.endswith(suffix):
            prefix = main_tag[:-len(suffix)]
            return {s: f"{prefix}_{s}" for s in SEGMENTS}
    return dict(SEGMENT_TAGS)


def merged_segment_summary(out_dir: Path, tags: dict) -> dict:
    """One summary-shaped dict from the segment fits, each segment from its own.

    Per-segment entries (parameters, prior predictive, rates, deployed impact)
    are taken from the fit of that segment; ``rounds`` from the first fit found.
    """
    merged = {"parameters": {}, "prior_predictive": {}, "silver_rates": {},
              "forward_rates": {}, "ppc_tags": []}
    impact = {}
    for segment, tag in tags.items():
        s = load_json(out_dir / "fits" / tag / "summary.json")
        if not s:
            continue
        merged.setdefault("rounds", s.get("rounds"))
        merged["parameters"].update(s.get("parameters") or {})
        merged["prior_predictive"].update(s.get("prior_predictive") or {})
        for key in ("silver_rates", "forward_rates"):
            if (s.get(key) or {}).get(segment):
                merged[key][segment] = s[key][segment]
        if "ppc" in s:
            merged["ppc"] = True
            merged["ppc_tags"].append(tag)
        if (s.get("deployed_impact") or {}).get("segments", {}).get(segment):
            impact[segment] = s["deployed_impact"]["segments"][segment]
    if impact:
        merged["deployed_impact"] = {"segments": impact}
    return {k: v for k, v in merged.items() if v not in ({}, None)}


def curve_shift(out_dir: Path, tag: str, reference: str = MAIN) -> dict:
    """Max and mean |Δ posterior mean| of a variant's curves vs the reference."""
    a = out_dir / "fits" / tag / "curves.parquet"
    b = out_dir / "fits" / reference / "curves.parquet"
    if not (a.exists() and b.exists()):
        return {}
    fa, fb = pd.read_parquet(a), pd.read_parquet(b)
    out = {}
    for segment in ("overture", "osm", "matched"):
        da = fa[fa["segment"] == segment]["mean"].to_numpy()
        db = fb[fb["segment"] == segment]["mean"].to_numpy()
        diff = np.abs(da - db)
        out[segment] = (float(diff.max()), float(diff.mean()))
    return out


def gold_rate_table(out_dir: Path, tags: tuple, segment_tags: dict = None) -> list:
    """Posterior-mean P(exists) vs the design-weighted gold rate (quintiles).

    Rows are phase-1 validation rows grouped by segment x raw-score quintile;
    the model value is the mean of each arm's posterior-mean curve over the
    group's rows, the reference the Hajek-weighted gold rate (1 / pi_class).
    ``segment_tags`` ({segment: tag}) adds one column built from the
    single-segment fits, each segment from its own fit.
    """
    import jax

    from openpois.conflation import calibration_bayes as cb
    from openpois.models.jax_core import enable_high_precision

    enable_high_precision()
    config = common.load_config()
    fit_config = common.fit_config_from(config)

    def has_draws(tag):
        return (out_dir / "fits" / tag / "draws.npz").exists()

    # Column label -> {segment: tag}: a joint fit serves every segment.
    columns = {t: {s: t for s in cb.SEGMENT_ORDER} for t in tags if has_draws(t)}
    if segment_tags:
        found = {s: t for s, t in segment_tags.items() if has_draws(t)}
        if found:
            columns = {"segment fits": found, **columns}
    available = list(columns)
    # Every fit in one directory shares its rounds; the main fit's table is the
    # reference (all its pooled rounds, each under its own design weights).
    if not available:
        return ["(no fits with saved draws)"]
    rows = common.fit_rows(config, out_dir, next(iter(columns[available[0]].values())))
    weights = common.design_weights(rows, fit_config)
    production = common.load_production()
    preds = {}
    for label, by_segment in columns.items():
        for tag in dict.fromkeys(by_segment.values()):
            # Rebuild the fit's exact spec from the repr saved in its summary.
            summary = load_json(out_dir / "fits" / tag / "summary.json")
            spec = eval(summary["spec"], {"ModelSpec": cb.ModelSpec,
                                          "PriorConfig": cb.PriorConfig})
            prepared = cb.prepare_data(rows, spec, fit_config = fit_config,
                                       silver_rates = summary.get("silver_rates"),
                                       forward_rates = summary.get("forward_rates"))
            saved = np.load(out_dir / "fits" / tag / "draws.npz")
            draws = {k: jax.numpy.asarray(
                saved[k].reshape((-1,) + saved[k].shape[2:])[::4])
                for k in saved.files}
            for segment in spec.segments:
                if by_segment.get(segment) != tag:
                    continue
                seg = rows[rows["segment"] == segment]
                preds[(label, segment)] = cb.curve_draws(
                    draws, prepared, segment, osm = seg["osm_score"].to_numpy(),
                    overture = seg["overture_score"].to_numpy()).mean(axis = 0)
    lines = ["| segment | raw-score quintile | gold | design-weighted gold rate | "
             "production | " + " | ".join(available) + " |",
             "|---|---|---|---|---|" + "---|" * len(available)]
    for segment in cb.SEGMENT_ORDER:
        mask = (rows["segment"] == segment).to_numpy()
        seg = rows[mask].reset_index(drop = True)
        w, y = weights[mask], seg["y"].fillna(0).to_numpy()
        prod = common.production_values(production, segment,
                                        osm = seg["osm_score"].to_numpy(),
                                        overture = seg["overture_score"].to_numpy())
        groups = list(pd.qcut(seg["raw_score"], 5, duplicates = "drop")
                      .groupby(pd.qcut(seg["raw_score"], 5, duplicates = "drop"),
                               observed = True).groups.items())
        groups.append(("all", seg.index))
        for label, idx in groups:
            idx = np.asarray(idx)
            sel = w[idx] > 0
            ht = float(np.sum(w[idx][sel] * y[idx][sel]) / np.sum(w[idx][sel]))
            values = " | ".join(f"{preds[(t, segment)][idx].mean():.3f}"
                                if (t, segment) in preds else "—"
                                for t in available)
            lines.append(f"| {segment} | {label} | {int(sel.sum())} | {ht:.3f} | "
                         f"{prod[idx].mean():.3f} | {values} |")
    return lines


def sensitivity_at_rows(out_dir: Path) -> list:
    """Mean |shift| of each sensitivity fit vs the main fit at the validation rows."""
    import jax

    from openpois.conflation import calibration_bayes as cb
    from openpois.models.jax_core import enable_high_precision

    enable_high_precision()
    config = common.load_config()
    fit_config = common.fit_config_from(config)
    if not (out_dir / "fits" / MAIN / "draws.npz").exists():
        return ["(main fit missing; no at-row shifts)"]
    rows = common.fit_rows(config, out_dir, MAIN)

    def curves(tag):
        summary = load_json(out_dir / "fits" / tag / "summary.json")
        spec = eval(summary["spec"], {"ModelSpec": cb.ModelSpec,
                                      "PriorConfig": cb.PriorConfig})
        prepared = cb.prepare_data(rows, spec, fit_config = fit_config,
                                   silver_rates = summary.get("silver_rates"),
                                   forward_rates = summary.get("forward_rates"))
        saved = np.load(out_dir / "fits" / tag / "draws.npz")
        draws = {k: jax.numpy.asarray(saved[k].reshape((-1,) + saved[k].shape[2:])[::4])
                 for k in saved.files}
        out = {}
        for segment in cb.SEGMENT_ORDER:
            seg = rows[rows["segment"] == segment]
            out[segment] = cb.curve_draws(
                draws, prepared, segment, osm = seg["osm_score"].to_numpy(),
                overture = seg["overture_score"].to_numpy()).mean(axis = 0)
        return out

    base = curves(MAIN)
    lines = [f"Mean abs shift of posterior-mean P(exists) vs {MAIN} at the phase-1 "
             "validation rows:", "",
             "| run | Overture | OSM | matched | matched p95 |", "|---|---|---|---|---|"]
    for tag in SENSITIVITY:
        if not (out_dir / "fits" / tag / "draws.npz").exists():
            continue
        c = curves(tag)
        d = {s: np.abs(c[s] - base[s]) for s in cb.SEGMENT_ORDER}
        lines.append(f"| {tag} | {d['overture'].mean():.3f} | {d['osm'].mean():.3f} | "
                     f"{d['matched'].mean():.3f} | "
                     f"{np.quantile(d['matched'], 0.95):.3f} |")
    return lines


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--out-dir", default = None)
    parser.add_argument("--main", default = MAIN,
                        help = "tag of the main model (default %(default)s)")
    args = parser.parse_args()
    main_tag = args.main
    config = common.load_config()
    _, metadata = common.load_handoff(config)
    out_dir = common.eval_dir(config, metadata, args.out_dir)
    main = load_json(out_dir / "fits" / main_tag / "summary.json")
    segment_tags = segment_fit_tags(main_tag, main)
    lines = ["# Bayesian monotone-spline calibration: Phase 1 report", ""]
    lines.append(f"Validation round {metadata['validation_round']} "
                 f"(conflation {metadata['conflation_version']}). Design doc: "
                 f"`.claude/plans/bayesian-monotone-calibration.md`; execution log: "
                 f"`bayesian-monotone-calibration-notes.md`. "
                 + ("Production calibration: one fit per segment "
                    f"({', '.join(segment_tags.values())}); the release needs all "
                    "three to pass the acceptance rule." if segment_tags else
                    "Nothing here is on the production path."))
    lines.append("")

    lines += ["## 1. Full-data fits", ""]
    if segment_tags:
        listed = [(f"{segment}: {tag}", tag) for segment, tag in segment_tags.items()]
        main_label = "segment fits"
    else:
        listed = [(tag, tag) for tag in MAIN_TAGS + ("armA_variant",)]
        main_label = main_tag
    for label, tag in listed:
        s = load_json(out_dir / "fits" / tag / "summary.json")
        if not s:
            lines.append(f"- {label}: not run")
            continue
        lines.append(f"- **{label}** ({s['num_parameters']} parameters): "
                     f"{diag_line(s)}")
    lines.append("")
    if segment_tags:
        main = merged_segment_summary(out_dir, segment_tags) or None
    ppc_tags = main.get("ppc_tags", []) if segment_tags and main else [main_tag]
    if main:
        lines += [f"### Key parameters ({main_label}, posterior mean [95% interval])",
                  "",
                  "| parameter | mean | 95% interval |", "|---|---|---|"]
        for name, v in main["parameters"].items():
            lines.append(f"| {name} | {fmt(v['mean'])} | [{fmt(v['lower'])}, "
                         f"{fmt(v['upper'])}] |")
        lines.append("")
        lines += ["### Prior predictive (§5.2)", "",
                  "| curve | median span | share span > 0.5 | share within 0.01 of "
                  "a bound | P(0) 5/50/95% | P(1) 5/50/95% |",
                  "|---|---|---|---|---|---|"]
        for seg, v in main["prior_predictive"].items():
            lines.append(
                f"| {seg} | {v['median_span']:.3f} | {v['share_span_gt_0.5']:.2f} | "
                f"{v['share_near_bounds']:.2f} | "
                f"{'/'.join(f'{x:.2f}' for x in v['p_at_0_quantiles'])} | "
                f"{'/'.join(f'{x:.2f}' for x in v['p_at_1_quantiles'])} |")
        lines.append("")
        for tag in (tuple(segment_tags.values()) if segment_tags else MAIN_TAGS):
            s = load_json(out_dir / "fits" / tag / "summary.json")
            if not s or "ppc" not in s:
                continue
            lines += [f"### Posterior predictive checks ({tag}, §5.3)", ""]
            for name, v in s["ppc"].items():
                share = v["share_inside"]
                lines.append(f"- {name}: {v['n']} cells"
                             + (f", {share:.0%} inside the 95% interval"
                                if share is not None else ""))
            lines.append("")
        if "ppc" in main:
            knot_rows = []
            for tag in ppc_tags:
                ppc = load_json(out_dir / "fits" / tag / "ppc.json")
                knot_rows += (ppc or {}).get("rate_by_knot") or []
            if knot_rows:
                lines += ["", "| segment | interval | n gold | HT rate | model mean "
                          "[95%] |", "|---|---|---|---|---|"]
                for r in knot_rows:
                    lines.append(
                        f"| {r['segment']} | [{r['lo']:.3f}, {r['hi']:.3f}) | "
                        f"{r['n_gold']} | {r['ht_rate']:.3f} | {r['model_mean']:.3f} "
                        f"[{r['model_lower']:.3f}, {r['model_upper']:.3f}] |")
            lines.append("")
        if "deployed_impact" in main:
            lines += ["### Deployed-impact preview (20260902 population, "
                      "unflagged rows)", "",
                      "| segment | n | mean Bayes | mean published | mean Oct "
                      "production | mean abs Δ vs Oct | share abs Δ > 0.05 vs Oct | "
                      "share abs Δ > 0.10 vs Oct | crosses 0.3/0.7/0.9 vs Oct* |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for seg, v in main["deployed_impact"]["segments"].items():
                lines.append(
                    f"| {seg} | {v['n']:,} | {v['mean_bayes']:.3f} | "
                    f"{v['mean_published']:.3f} | {v['mean_october_production']:.3f} "
                    f"| {v['mean_abs_vs_october']:.3f} | "
                    f"{v['share_gt_0.05_vs_october']:.1%} | "
                    f"{v['share_gt_0.10_vs_october']:.1%} | "
                    f"{v['share_band_change_vs_october']:.1%} |")
            lines += ["", "\\* Crossings of the 0.3 / 0.7 / 0.9 edges inherited from "
                      "`compare_matched_index.BAND_EDGES`. The site has no discrete "
                      "bands: it colours confidence on a continuous gradient in 0.05 "
                      "steps, so this column overstates visible change. On Overture it "
                      "is driven by two atoms: 0.919912 (39% of POIs), whose value "
                      "straddles 0.7, and 0.990219 / 1.0 (~10%). On the site these are "
                      "one- and two-step colour changes (execution log, decision 20).",
                      ""]
    if main and main.get("silver_rates"):
        lines += ["### Silver-label rates used (fixed-rate arm C)", "",
                  f"Validation rounds in the fit (current first): "
                  f"{', '.join(main.get('rounds') or []) or 'current round only'}. "
                  "Each round's gold enters under its own design weights.", "",
                  "| segment | q(exists) | raw | gold n | q(gone) | raw | gold n |",
                  "|---|---|---|---|---|---|---|"]
        for seg, r in main["silver_rates"].items():
            lines.append(f"| {seg} | {r['exists']:.4f} | {r['raw_exists']:.4f} | "
                         f"{r['n_exists']} | {r['gone']:.4f} | {r['raw_gone']:.4f} | "
                         f"{r['n_gone']} |")
        lines.append("")
    if main and main.get("forward_rates"):
        lines += ["### Forward rates used (fixed-rate mixture)", "",
                  "Se = P(verdict exists | exists), Sp = P(verdict gone | gone), "
                  "among definitive verdicts; Jeffreys-smoothed on the Kish ESS.", "",
                  "| segment | Se | raw | ESS | Sp | raw | ESS |",
                  "|---|---|---|---|---|---|---|"]
        for seg, r in main["forward_rates"].items():
            lines.append(f"| {seg} | {r['se']:.4f} | {r['raw_se']:.4f} | "
                         f"{r['ess_se']:.1f} | {r['sp']:.4f} | {r['raw_sp']:.4f} | "
                         f"{r['ess_sp']:.1f} |")
        lines.append("")
    lines += ["### Calibration against the design-weighted gold rate", "",
              "Mean posterior-mean P(exists) over the validation rows in each group, "
              "against the Hajek-weighted gold rate. The gold rate is itself noisy "
              "(few gold rows per quintile), so read systematic offsets across "
              "quintiles, not single cells.", ""]
    lines += gold_rate_table(out_dir, MAIN_TAGS + ("armA_variant",), segment_tags)
    lines.append("")
    lines += ["Figures: `figures/<tag>_curves_1d.png`, `figures/<tag>_matched.png`, "
              "`figures/<tag>_prior_predictive.png`, `figures/armC_beta_label.png`.",
              ""]

    lines += ["## 2. Ten-fold cross-validation (§5.4)", ""]
    cv_md = out_dir / "cv" / "cv_results.md"
    if cv_md.exists():
        body = cv_md.read_text().splitlines()
        lines += [l for l in body if not l.startswith("# ")]
    else:
        lines.append("Not run.")
    lines.append("")
    cv_mixture_md = out_dir / "cv" / "cv_results_mixture.md"
    if cv_mixture_md.exists():
        lines += ["### Fixed-rate mixture against arms C and B", ""]
        body = cv_mixture_md.read_text().splitlines()
        lines += [line for line in body if not line.startswith("# ")]
        lines.append("")

    lines += ["## 3. Coverage study (§5.2)", ""]
    cov = load_json(out_dir / "coverage" / "coverage_summary.json")
    if cov:
        lines.append(f"{cov['fits']} fits; max R̂ {cov['max_rhat']:.3f}; "
                     f"divergences {cov['divergences']}; "
                     f"{cov['mean_minutes']:.1f} min per fit.")
        lines += ["", "| scenario | arm | segment | coverage of 95% band | mean width "
                  "| bias | rounds |", "|---|---|---|---|---|---|---|"]
        for r in cov["by_segment"]:
            lines.append(f"| {r['scenario']} | {r['arm']} | {r['segment']} | "
                         f"{r['coverage']:.3f} | {r['width']:.3f} | "
                         f"{r['bias']:+.4f} | {r['rounds']} |")
        lines += ["", "By region:", "",
                  "| scenario | arm | segment | region | coverage | width | bias | "
                  "points |", "|---|---|---|---|---|---|---|---|"]
        for r in cov["by_region"]:
            lines.append(f"| {r['scenario']} | {r['arm']} | {r['segment']} | "
                         f"{r['region']} | {r['coverage']:.3f} | {r['width']:.3f} | "
                         f"{r['bias']:+.4f} | {r['n_points']} |")
    else:
        lines.append("Not run.")
    lines.append("")

    ran_sensitivity = any((out_dir / "fits" / tag / "summary.json").exists()
                          for tag in SENSITIVITY)
    if segment_tags and not ran_sensitivity:
        # The production segment run has no sensitivity fits.
        lines += ["## 4. Sensitivity runs (§5.5)", "", "Not run.", ""]
    else:
        lines += [f"## 4. Sensitivity runs (§5.5; full data, vs {MAIN})", "",
                  "| run | change | acceptance | max abs Δ curve (overture / osm / "
                  "matched) | mean abs Δ curve (overture / osm / matched) |",
                  "|---|---|---|---|---|"]
        for tag, label in SENSITIVITY.items():
            s = load_json(out_dir / "fits" / tag / "summary.json")
            if not s:
                lines.append(f"| {tag} | {label} | not run | | |")
                continue
            shift = curve_shift(out_dir, tag)
            mx = " / ".join(f"{shift[k][0]:.3f}" for k in SEGMENTS)
            mn = " / ".join(f"{shift[k][1]:.3f}" for k in SEGMENTS)
            verdict = "PASS" if s["acceptance"]["all"] else "FAIL"
            lines.append(f"| {tag} | {label} | {verdict} | {mx} | {mn} |")
        lines += ["", "The grid shifts include data-empty corners of the matched "
                  "surface. The at-row shifts below are the ones that matter. "
                  "Sensitivity fits use the light setting (4 × 600 + 400), so a "
                  "FAIL is usually the strict "
                  "§5.1 rule (R̂ ≤ 1.01, ESS ≥ 400, zero divergences), not a gross "
                  "failure. A large S2C shift is the collapse of Sp described in the "
                  "execution log (decision 15).", ""]
        lines += sensitivity_at_rows(out_dir)
        lines.append("")
    (out_dir / "fit_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'fit_report.md'}")


if __name__ == "__main__":
    main()
