"""Full-data fit of the Bayesian monotone-spline calibration (Phase 1).

Off the production path: writes only under
``<conflation root>/<round>/calibration_eval_bayes_20260927/`` (design doc
``.claude/plans/bayesian-monotone-calibration.md`` §4-§5).

One invocation fits one model variant (``--arm`` plus the sensitivity flags) and
writes, under ``fits/<tag>/``:

- ``draws.npz``: posterior draws with (chain, draw) axes
- ``convergence.csv``, ``curve_convergence.csv``, ``summary.json``
  (diagnostics, the §5.1 acceptance checks, key parameters, timing)
- ``curves.parquet``: posterior mean and 95% band on score grids, with the
  production (October 2026 configuration) values alongside
- ``ppc.json``: posterior predictive checks (arm A)
- ``deployed_impact.json`` (``--deployed-impact``): the posterior-mean curve
  applied to the 20260902 population against the published and the October
  production values

``--prior-predictive-only`` draws curves from the priors and stops.

Usage::

    python -u scripts/conflation/fit_bayes_calibration.py --arm A --tag armA \\
        --deployed-impact 2>&1 | tee <eval dir>/logs/fit_armA.log
"""

from __future__ import annotations

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bayes_calibration_common as common  # noqa: E402

from openpois.conflation import calibration_bayes as cb  # noqa: E402
from openpois.models.jax_core import enable_high_precision  # noqa: E402

GRID_1D = np.linspace(0.0, 1.0, 401)
GRID_2D = np.linspace(0.0, 1.0, 81)
# Points whose curve draws must also meet the convergence rule (§5.1).
CONV_1D = np.linspace(0.05, 1.0, 20)
CONV_2D = np.linspace(0.1, 1.0, 10)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush = True)


def flat_draws(chain_draws: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]),
                                  chain_draws)


def curve_grid_draws(draws: dict, prepared) -> dict:
    """Posterior draws of every curve on its grid."""
    out = {}
    for segment in cb.ONE_D_SEGMENTS:
        out[segment] = cb.curve_draws(draws, prepared, segment, osm = GRID_1D,
                                      overture = GRID_1D)
    xx, yy = np.meshgrid(GRID_2D, GRID_2D, indexing = "ij")
    out["matched"] = cb.curve_draws(draws, prepared, "matched", osm = xx.ravel(),
                                    overture = yy.ravel())
    return out


def curve_convergence(chain_draws: dict, prepared) -> pd.DataFrame:
    """R-hat and ESS of m_g(s) at fixed published points, per chain."""
    n_chains, n_draws = jax.tree_util.tree_leaves(chain_draws)[0].shape[:2]
    draws = flat_draws(chain_draws)
    series = {}
    for segment in cb.ONE_D_SEGMENTS:
        m = cb.curve_draws(draws, prepared, segment, osm = CONV_1D,
                           overture = CONV_1D).reshape(n_chains, n_draws, -1)
        for i, s in enumerate(CONV_1D):
            series[f"{segment}@{s:.3f}"] = m[:, :, i]
    xx, yy = np.meshgrid(CONV_2D, CONV_2D, indexing = "ij")
    m = cb.curve_draws(draws, prepared, "matched", osm = xx.ravel(),
                       overture = yy.ravel()).reshape(n_chains, n_draws, -1)
    for i, (a, b) in enumerate(zip(xx.ravel(), yy.ravel())):
        series[f"matched@{a:.2f},{b:.2f}"] = m[:, :, i]
    return cb.convergence_table(series)


def curves_frame(grid_draws: dict, production: dict) -> pd.DataFrame:
    frames = []
    for segment in cb.ONE_D_SEGMENTS:
        summary = cb.summarize_draws(grid_draws[segment])
        prod = common.production_values(production, segment, osm = GRID_1D,
                                         overture = GRID_1D)
        frames.append(pd.DataFrame({
            "segment": segment, "osm_score": GRID_1D if segment == "osm"
            else np.nan, "overture_score": GRID_1D if segment == "overture"
            else np.nan, **summary, "production": prod,
        }))
    xx, yy = np.meshgrid(GRID_2D, GRID_2D, indexing = "ij")
    summary = cb.summarize_draws(grid_draws["matched"])
    prod = common.production_values(production, "matched", osm = xx.ravel(),
                                    overture = yy.ravel())
    frames.append(pd.DataFrame({
        "segment": "matched", "osm_score": xx.ravel(),
        "overture_score": yy.ravel(), **summary, "production": prod,
    }))
    return pd.concat(frames, ignore_index = True)


# ---------------------------------------------------------------------------
# Posterior predictive checks (arm A)
# ---------------------------------------------------------------------------

def posterior_predictive_checks(draws: dict, prepared, rows: pd.DataFrame,
                                weights: np.ndarray, n_ppc: int = 200,
                                seed: int = 0) -> dict:
    """Posterior predictive checks (§5.3), each observed vs 95% interval.

    - ``rate_by_knot`` (every arm): the design-weighted gold rate per knot
      interval against the model's mean P(exists) over the same phase-1 rows,
      allowing for both uncertainties.
    - ``verdict_mix`` and ``gold_rate_by_class`` (arm A): the LLM verdict share
      by score decile, and the gold existence rate within verdict x score
      tercile, replicated from the measurement layer.
    - ``label_mix`` (arm C): the share of rows labelled "exists" (the gold label
      where there is one, else the silver label) by score decile, replicated
      from the curve and the label-noise model.
    """
    spec, geometry = prepared.spec, prepared.geometry
    data = prepared.to_jax()
    n_total = jax.tree_util.tree_leaves(draws)[0].shape[0]
    pick = np.linspace(0, n_total - 1, n_ppc).astype(int)
    sub = jax.tree_util.tree_map(lambda x: x[pick], draws)
    rng = np.random.default_rng(seed)

    @jax.jit
    def probs(params):
        coefficients = cb.curve_coefficients(params, geometry, spec)
        logits = cb.segment_logits(coefficients, data)
        out = {}
        for g, segment in enumerate(cb.SEGMENT_ORDER):
            p = jax.nn.sigmoid(logits[segment])
            if spec.arm == "A":
                extra = jnp.exp(cb.class_log_probs(params, spec, g,
                                                   data[segment]["r"]))
            elif spec.arm == "C" and spec.label_noise != "fixed":
                log_se, _, log_sp, _ = cb.silver_log_rates(params, spec)
                extra = jnp.stack([jnp.exp(log_se), jnp.exp(log_sp)])
            else:
                extra = jnp.zeros(2)
            out[segment] = (p, extra)
        return out

    verdict_of = np.array([name.split(":")[0] for name in spec.class_levels])
    checks = {"rate_by_knot": []}
    if spec.arm == "A":
        checks.update({"verdict_mix": [], "gold_rate_by_class": []})
    if spec.arm == "C":
        checks["label_mix"] = []
    evaluated = [probs(jax.tree_util.tree_map(lambda x: x[d], sub))
                 for d in range(n_ppc)]
    for segment in cb.SEGMENT_ORDER:
        seg_rows = rows[rows["segment"] == segment].reset_index(drop = True)
        seg = prepared.segments[segment]
        r = seg_rows[cb.R_COLUMN[segment]].to_numpy(dtype = float)
        deciles = np.clip(np.searchsorted(np.quantile(r, np.linspace(0, 1, 11)[1:-1]),
                                          r, side = "right"), 0, 9)
        terciles = np.clip(np.searchsorted(np.quantile(r, [1 / 3, 2 / 3]), r,
                                           side = "right"), 0, 2)
        observed_verdict = seg_rows["llm_verdict"].to_numpy()
        gold = seg["gold"] > 0.5
        y = seg["y"]
        rep_m, rep_mix, rep_gold, rep_label = [], [], [], []
        for d in range(n_ppc):
            p, extra = evaluated[d][segment]
            p, extra = np.asarray(p), np.asarray(extra)
            rep_m.append(p)
            if spec.arm == "A":
                theta = extra
                pv = p[:, None] * theta[:, 1, :] + (1 - p[:, None]) * theta[:, 0, :]
                cum = np.cumsum(pv, axis = 1)
                draw = (rng.uniform(size = len(p))[:, None] > cum).sum(axis = 1)
                rep_verdict = verdict_of[np.minimum(draw, pv.shape[1] - 1)]
                rep_mix.append(np.array([[np.mean(rep_verdict[deciles == k] == v)
                                          for v in cb.VERDICTS] for k in range(10)]))
                obs_cls = seg["cls"]
                t1 = theta[np.arange(len(p)), 1, obs_cls]
                t0 = theta[np.arange(len(p)), 0, obs_cls]
                post = p * t1 / (p * t1 + (1 - p) * t0)
                y_rep = rng.uniform(size = len(p)) < post
                rates = np.full((3, 3), np.nan)
                for vi, v in enumerate(cb.VERDICTS):
                    for tt in range(3):
                        sel = gold & (observed_verdict == v) & (terciles == tt)
                        if sel.sum() >= 5:
                            rates[vi, tt] = y_rep[sel].mean()
                rep_gold.append(rates)
            if spec.arm == "C":
                silver = seg["silver_label"] >= 0
                if spec.label_noise == "fixed":
                    # Fixed rates: the observed side is the corrected label
                    # (gold y, or q for silver rows), so replicate true y ~ p.
                    p_label = p
                else:
                    se, sp = float(extra[0]), float(extra[1])
                    p_label = np.where(silver, p * se + (1 - p) * (1 - sp), p)
                labelled = (rng.uniform(size = len(p)) < p_label) & (gold | silver)
                rep_label.append([labelled[(deciles == k) & (gold | silver)].mean()
                                  for k in range(10)])
        rep_m = np.array(rep_m)
        if spec.arm == "A":
            rep_mix, rep_gold = np.array(rep_mix), np.array(rep_gold)
            for k in range(10):
                for vi, v in enumerate(cb.VERDICTS):
                    obs = float(np.mean(observed_verdict[deciles == k] == v))
                    lo, hi = np.quantile(rep_mix[:, k, vi], [0.025, 0.975])
                    checks["verdict_mix"].append({
                        "segment": segment, "decile": k, "verdict": v,
                        "observed": obs, "lower": float(lo), "upper": float(hi),
                        "inside": bool(lo <= obs <= hi)})
            for vi, v in enumerate(cb.VERDICTS):
                for tt in range(3):
                    sel = gold & (observed_verdict == v) & (terciles == tt)
                    if sel.sum() < 5:
                        continue
                    obs = float(y[sel].mean())
                    lo, hi = np.nanquantile(rep_gold[:, vi, tt], [0.025, 0.975])
                    checks["gold_rate_by_class"].append({
                        "segment": segment, "verdict": v, "tercile": tt,
                        "n_gold": int(sel.sum()), "observed": obs,
                        "lower": float(lo), "upper": float(hi),
                        "inside": bool(lo <= obs <= hi)})
        if spec.arm == "C":
            rep_label = np.array(rep_label)
            silver = seg["silver_label"] >= 0
            silver_value = (seg["silver_q"] if spec.label_noise == "fixed"
                            else seg["silver_label"])
            label = np.where(gold, y, np.where(silver, silver_value, np.nan))
            for k in range(10):
                sel = (deciles == k) & (gold | silver)
                obs = float(np.mean(label[sel]))
                lo, hi = np.quantile(rep_label[:, k], [0.025, 0.975])
                checks["label_mix"].append({
                    "segment": segment, "decile": k, "n": int(sel.sum()),
                    "observed": obs, "lower": float(lo), "upper": float(hi),
                    "inside": bool(lo <= obs <= hi)})
        # Design-weighted gold rate per knot interval vs the model mean.
        seg_weights = weights[(rows["segment"] == segment).to_numpy()]
        if segment in cb.ONE_D_SEGMENTS:
            score = seg_rows[cb.SCORE_COLUMN[segment]].to_numpy(dtype = float)
            edges = prepared.knots[segment]
        else:
            score = seg_rows["raw_score"].to_numpy(dtype = float)
            edges = np.unique(np.quantile(score, np.linspace(0, 1, 11)))
        bins = np.clip(np.searchsorted(edges, score, side = "right") - 1, 0,
                       len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = bins == b
            wsel = sel & (seg_weights > 0)
            if wsel.sum() < 5:
                continue
            w_b = seg_weights[wsel]
            ht = float(np.sum(w_b * y[wsel]) / w_b.sum())
            ess = float(w_b.sum() ** 2 / np.sum(w_b ** 2))
            se_ht = float(np.sqrt(max(ht * (1 - ht), 1e-4) / max(ess, 1.0)))
            model = rep_m[:, sel].mean(axis = 1)
            lo, hi = np.quantile(model, [0.025, 0.975])
            # Both sides are uncertain: the HT rate by its design-based SE, the
            # model mean by its posterior SD.
            z = (ht - model.mean()) / np.sqrt(se_ht ** 2 + model.std() ** 2)
            checks["rate_by_knot"].append({
                "segment": segment, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                "n_rows": int(sel.sum()), "n_gold": int(wsel.sum()),
                "ht_rate": ht, "ht_se": se_ht, "model_mean": float(model.mean()),
                "model_lower": float(lo), "model_upper": float(hi), "z": float(z),
                "inside": bool(abs(z) < 1.96)})
    summary = {
        name: {"n": len(items),
               "share_inside": float(np.mean([c["inside"] for c in items]))
               if items else None}
        for name, items in checks.items()
    }
    return {"summary": summary, **checks}


# ---------------------------------------------------------------------------
# Deployed impact
# ---------------------------------------------------------------------------

def deployed_impact(grid_draws: dict, production: dict, conflated_path: Path,
                    chunk_rows: int = 2_000_000) -> dict:
    """Posterior-mean curve on the population vs published and October values.

    Streams column-scoped (never a whole-file load). Rows whose
    ``calibration_flag`` is set (shadow CD, manual pins, missing conf,
    unnamed extrapolation) are skipped: they do not ride the plain curve.
    """
    mean_1d = {s: grid_draws[s].mean(axis = 0) for s in cb.ONE_D_SEGMENTS}
    surface = grid_draws["matched"].mean(axis = 0).reshape(len(GRID_2D),
                                                          len(GRID_2D))
    interp_2d = RegularGridInterpolator((GRID_2D, GRID_2D), surface)
    columns = ["source", "osm_conf_mean", "overture_confidence", "conf_mean",
               "calibration_flag"]
    pf = pq.ParquetFile(str(conflated_path))
    if "shadow_matched" in pf.schema_arrow.names:
        columns.append("shadow_matched")
    acc = {s: {"n": 0, "abs_pub": 0.0, "gt05_pub": 0, "gt10_pub": 0,
               "abs_prod": 0.0, "gt05_prod": 0, "gt10_prod": 0,
               "sum_bayes": 0.0, "sum_pub": 0.0, "sum_prod": 0.0,
               "band_changes_prod": 0}
           for s in cb.SEGMENT_ORDER}
    flags_seen = {}
    edges = np.array([0.3, 0.7, 0.9])
    for batch in pf.iter_batches(batch_size = chunk_rows, columns = columns):
        frame = batch.to_pandas()
        flag = frame["calibration_flag"]
        for value, count in flag.fillna("<none>").value_counts().items():
            flags_seen[value] = flags_seen.get(value, 0) + int(count)
        keep = flag.isna() | (flag.astype(str) == "")
        if "shadow_matched" in frame.columns:
            keep &= ~frame["shadow_matched"].fillna(False).astype(bool)
        for segment in cb.SEGMENT_ORDER:
            sel = (keep & (frame["source"] == segment)).to_numpy()
            if not sel.any():
                continue
            osm = frame["osm_conf_mean"].to_numpy(dtype = float)[sel]
            ov = frame["overture_confidence"].to_numpy(dtype = float)[sel]
            pub = frame["conf_mean"].to_numpy(dtype = float)[sel]
            if segment == "matched":
                pts = np.column_stack([np.clip(np.round(osm, 6), 0, 1),
                                       np.clip(np.round(ov, 6), 0, 1)])
                bayes = interp_2d(pts)
            else:
                s = osm if segment == "osm" else ov
                bayes = np.interp(np.clip(np.round(s, 6), 0, 1), GRID_1D,
                                  mean_1d[segment])
            prod = common.production_values(production, segment, osm = osm,
                                            overture = ov)
            ok = np.isfinite(bayes) & np.isfinite(pub) & np.isfinite(prod)
            bayes, pub, prod = bayes[ok], pub[ok], prod[ok]
            a = acc[segment]
            a["n"] += int(ok.sum())
            d_pub, d_prod = np.abs(bayes - pub), np.abs(bayes - prod)
            a["abs_pub"] += float(d_pub.sum())
            a["gt05_pub"] += int((d_pub > 0.05).sum())
            a["gt10_pub"] += int((d_pub > 0.10).sum())
            a["abs_prod"] += float(d_prod.sum())
            a["gt05_prod"] += int((d_prod > 0.05).sum())
            a["gt10_prod"] += int((d_prod > 0.10).sum())
            a["sum_bayes"] += float(bayes.sum())
            a["sum_pub"] += float(pub.sum())
            a["sum_prod"] += float(prod.sum())
            a["band_changes_prod"] += int(
                (np.searchsorted(edges, bayes) != np.searchsorted(edges, prod)).sum()
            )
    out = {"flags_seen": flags_seen, "segments": {}}
    for segment, a in acc.items():
        n = max(a["n"], 1)
        out["segments"][segment] = {
            "n": a["n"],
            "mean_bayes": a["sum_bayes"] / n,
            "mean_published": a["sum_pub"] / n,
            "mean_october_production": a["sum_prod"] / n,
            "mean_abs_vs_published": a["abs_pub"] / n,
            "share_gt_0.05_vs_published": a["gt05_pub"] / n,
            "share_gt_0.10_vs_published": a["gt10_pub"] / n,
            "mean_abs_vs_october": a["abs_prod"] / n,
            "share_gt_0.05_vs_october": a["gt05_prod"] / n,
            "share_gt_0.10_vs_october": a["gt10_prod"] / n,
            "share_band_change_vs_october": a["band_changes_prod"] / n,
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_curves(frame: pd.DataFrame, ht: dict, path: Path, label: str,
                  color: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize = (10, 4), dpi = 150)
    fig.patch.set_facecolor(common.COLORS["surface"])
    for ax, segment in zip(axes, cb.ONE_D_SEGMENTS):
        sub = frame[frame["segment"] == segment]
        x = sub[cb.SCORE_COLUMN[segment]].to_numpy()
        ax.fill_between(x, sub["lower"], sub["upper"], color = color,
                        alpha = 0.18, linewidth = 0)
        ax.plot(x, sub["mean"], color = color, linewidth = 2, label = label)
        ax.step(x, sub["production"], where = "post", linewidth = 1.5,
                color = common.COLORS["production"],
                label = "production (Oct 2026 config)")
        pts = ht[segment]
        ax.errorbar(pts["x"], pts["rate"], yerr = 1.96 * pts["se"], fmt = "o",
                    markersize = 4, color = common.COLORS["muted"],
                    elinewidth = 0.8, label = "design-weighted gold rate")
        common.style_axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{segment} segment", fontsize = 10,
                     color = common.COLORS["ink"])
        ax.set_xlabel(f"{cb.SCORE_COLUMN[segment]}", fontsize = 9)
        ax.set_ylabel("P(exists)", fontsize = 9)
    axes[0].legend(fontsize = 7, frameon = False, loc = "lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_matched(frame: pd.DataFrame, path: Path, label: str,
                   color: str) -> None:
    sub = frame[frame["segment"] == "matched"]
    n = len(GRID_2D)
    mean = sub["mean"].to_numpy().reshape(n, n)
    lower = sub["lower"].to_numpy().reshape(n, n)
    upper = sub["upper"].to_numpy().reshape(n, n)
    prod = sub["production"].to_numpy().reshape(n, n)
    fig = plt.figure(figsize = (13, 7.5), dpi = 150)
    fig.patch.set_facecolor(common.COLORS["surface"])
    grid = fig.add_gridspec(2, 3)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", common.SEQUENTIAL)
    diverging = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", ["#184f95", "#6da7ec", "#e4e3dd", "#f3a37f", "#b0451a"])
    ax = fig.add_subplot(grid[0, 0])
    im = ax.imshow(mean.T, origin = "lower", extent = (0, 1, 0, 1), vmin = 0,
                   vmax = 1, cmap = cmap, aspect = "auto")
    fig.colorbar(im, ax = ax, fraction = 0.046).set_label("P(exists)", fontsize = 8)
    ax.set_title(f"{label}: posterior mean", fontsize = 9)
    ax2 = fig.add_subplot(grid[0, 1])
    im2 = ax2.imshow((mean - prod).T, origin = "lower", extent = (0, 1, 0, 1),
                     vmin = -0.3, vmax = 0.3, cmap = diverging, aspect = "auto")
    fig.colorbar(im2, ax = ax2, fraction = 0.046).set_label(
        "Bayes - production", fontsize = 8)
    ax2.set_title("difference from production", fontsize = 9)
    ax3 = fig.add_subplot(grid[0, 2])
    im3 = ax3.imshow((upper - lower).T, origin = "lower", extent = (0, 1, 0, 1),
                     vmin = 0, vmax = 0.5, cmap = cmap, aspect = "auto")
    fig.colorbar(im3, ax = ax3, fraction = 0.046).set_label("95% band width",
                                                            fontsize = 8)
    ax3.set_title("band width", fontsize = 9)
    for a in (ax, ax2, ax3):
        a.set_xlabel("osm_score", fontsize = 8)
        a.set_ylabel("overture_score", fontsize = 8)
        a.tick_params(labelsize = 7)
    # Slices along OSM at fixed Overture values.
    for i, ov in enumerate((0.75, common.OVERTURE_ATOMS[0],
                            common.OVERTURE_ATOMS[1])):
        axs = fig.add_subplot(grid[1, i])
        k = int(np.argmin(np.abs(GRID_2D - ov)))
        axs.fill_between(GRID_2D, lower[:, k], upper[:, k], color = color,
                         alpha = 0.18, linewidth = 0)
        axs.plot(GRID_2D, mean[:, k], color = color, linewidth = 2, label = label)
        xs = GRID_2D
        prod_slice = common.production_values(
            _PRODUCTION[0], "matched", osm = xs, overture = np.full(len(xs), ov))
        axs.step(xs, prod_slice, where = "post", linewidth = 1.5,
                 color = common.COLORS["production"], label = "production")
        common.style_axes(axs)
        axs.set_ylim(0, 1)
        axs.set_title(f"overture_score = {ov:g} (grid {GRID_2D[k]:.3f} "
                      f"for the band)", fontsize = 8)
        axs.set_xlabel("osm_score", fontsize = 8)
        if i == 0:
            axs.set_ylabel("P(exists)", fontsize = 8)
            axs.legend(fontsize = 7, frameon = False, loc = "lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


_PRODUCTION = [None]


def figure_prior(prior_draws: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize = (12, 3.6), dpi = 150)
    fig.patch.set_facecolor(common.COLORS["surface"])
    for ax, segment in zip(axes, cb.SEGMENT_ORDER):
        m = prior_draws[segment]
        x = GRID_1D if segment != "matched" else GRID_2D
        for row in m[:60]:
            ax.plot(x, row, color = common.COLORS["A"], linewidth = 0.6,
                    alpha = 0.5)
        common.style_axes(ax)
        ax.set_ylim(0, 1)
        title = (f"{segment}: 60 prior draws" if segment != "matched" else
                 "matched: prior draws along osm_score at overture 0.92")
        ax.set_title(title, fontsize = 9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_beta_label(draws: np.ndarray, spec, path: Path) -> None:
    from scipy.stats import beta as beta_dist

    fig, ax = plt.subplots(figsize = (5, 3.4), dpi = 150)
    fig.patch.set_facecolor(common.COLORS["surface"])
    x = np.linspace(0.85, 1.0, 400)
    ax.plot(x, beta_dist.pdf(x, spec.priors.beta_label_a, spec.priors.beta_label_b),
            color = common.COLORS["production"], linewidth = 2, label = "prior")
    ax.hist(draws, bins = 60, density = True, color = common.COLORS["C"],
            alpha = 0.6, label = "posterior")
    common.style_axes(ax)
    ax.set_xlabel("beta_label = P(silver label correct)", fontsize = 8)
    ax.legend(fontsize = 7, frameon = False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def key_parameters(draws: dict, spec) -> dict:
    out = {}

    def summ(values):
        values = np.asarray(values, dtype = float)
        return {"mean": float(values.mean()),
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975))}

    for g, segment in enumerate(cb.SEGMENT_ORDER):
        out[f"tau_{segment}"] = summ(np.exp(draws["log_tau"][:, g]))
        out[f"mu_{segment}"] = summ(draws["mu"][:, g])
        out[f"alpha_{segment}"] = summ(draws["alpha"][:, g])
    if "log_omega_psi" in draws:
        out["omega_psi"] = summ(np.exp(draws["log_omega_psi"]))
    if "log_omega_beta" in draws:
        out["omega_beta"] = summ(np.exp(draws["log_omega_beta"]))
    if "logit_beta_label" in draws:
        out["beta_label"] = summ(1 / (1 + np.exp(-draws["logit_beta_label"])))
    if "logit_se" in draws:
        out["se_label"] = summ(1 / (1 + np.exp(-draws["logit_se"])))
        out["sp_label"] = summ(1 / (1 + np.exp(-draws["logit_sp"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    common.add_spec_arguments(parser)
    parser.add_argument("--tag", default = None)
    parser.add_argument("--out-dir", default = None)
    parser.add_argument("--prior-predictive-only", action = "store_true")
    parser.add_argument("--deployed-impact", action = "store_true")
    parser.add_argument("--impact-version", default = "20260902",
                        help = "Conflation version for the deployed impact.")
    parser.add_argument("--no-ppc", action = "store_true")
    parser.add_argument("--reuse-draws", action = "store_true",
                        help = ("Skip sampling: reload fits/<tag>/draws.npz and "
                                "the saved diagnostics, recompute everything else."))
    args = parser.parse_args()
    enable_high_precision()
    started = time.time()

    config = common.load_config()
    fit_config = common.fit_config_from(config)
    rows, metadata = common.load_handoff(config, args.pooled_rounds)
    out_dir = common.eval_dir(config, metadata, args.out_dir)
    spec = common.spec_from_args(args)
    current = str(metadata["validation_round"])
    rounds = [current] + [r for r in dict.fromkeys(rows["validation_round"].astype(str))
                          if r != current]
    tag = args.tag or f"arm{spec.arm}"
    fit_dir = out_dir / "fits" / tag
    fit_dir.mkdir(parents = True, exist_ok = True)
    production = common.load_production()
    _PRODUCTION[0] = production
    per_round = rows.groupby("validation_round", sort = False)["gold"].agg(
        ["size", "sum"])
    log(f"{tag}: {len(rows):,} phase-1 rows, {int(rows['gold'].sum()):,} gold "
        f"(rounds: " + ", ".join(f"{r} {int(v['size']):,}/{int(v['sum']):,}"
                                 for r, v in per_round.iterrows())
        + f"); spec {spec}")

    # Arm C fixed noise: prepare_data computes the rates from every round's
    # gold, each round under its own design weights (execution log, decisions 21 and 23).
    prepared = cb.prepare_data(rows, spec, fit_config = fit_config)
    silver_rates = prepared.silver_rates
    if silver_rates is not None:
        log("silver-label rates P(exists | segment, verdict): " + "; ".join(
            f"{s} exists {r['exists']:.4f} gone {r['gone']:.4f}"
            for s, r in silver_rates.items()))
    log("knots: " + "; ".join(f"{k} {len(v) - 1} intervals"
                              for k, v in prepared.knots.items()))

    # Prior predictive (§5.2).
    prior = cb.sample_prior_curve_params(prepared, 400,
                                         np.random.default_rng(args.seed))
    prior_curves = {s: cb.curve_draws(prior, prepared, s, osm = GRID_1D,
                                      overture = GRID_1D)
                    for s in cb.ONE_D_SEGMENTS}
    prior_curves["matched"] = cb.curve_draws(
        prior, prepared, "matched", osm = GRID_2D,
        overture = np.full(len(GRID_2D), common.OVERTURE_ATOMS[0]))
    prior_summary = {}
    for segment, m in prior_curves.items():
        span = m.max(axis = 1) - m.min(axis = 1)
        prior_summary[segment] = {
            "median_span": float(np.median(span)),
            "share_span_gt_0.5": float(np.mean(span > 0.5)),
            "share_near_bounds": float(np.mean((m < 0.01) | (m > 0.99))),
            "median_max_step": float(np.median(np.max(np.diff(m, axis = 1),
                                                      axis = 1))),
            "p_at_0_quantiles": np.quantile(m[:, 0], [0.05, 0.5, 0.95]).tolist(),
            "p_at_1_quantiles": np.quantile(m[:, -1], [0.05, 0.5, 0.95]).tolist(),
        }
    figure_prior(prior_curves, out_dir / "figures" / f"{tag}_prior_predictive.png")
    (fit_dir / "prior_predictive.json").write_text(json.dumps(prior_summary,
                                                              indent = 2))
    log(f"prior predictive: {json.dumps(prior_summary)}")
    if args.prior_predictive_only:
        return

    if args.reuse_draws:
        saved = np.load(fit_dir / "draws.npz")
        chain_draws = {k: jnp.asarray(saved[k]) for k in saved.files}
        previous = json.loads((fit_dir / "summary.json").read_text())
        diag_summary = previous["diagnostics"]
        acceptance = previous["acceptance"]
        curve_summary = previous["curve_convergence"]
        fit_minutes = previous["fit_minutes"]
        map_success = previous.get("map_success")
        log(f"reusing draws and diagnostics from {fit_dir}")
    else:
        fit_started = time.time()
        result = cb.fit(prepared, num_warmup = args.warmup,
                        num_samples = args.samples, num_chains = args.chains,
                        seed = args.seed,
                        adaptation_kwargs = common.adaptation_kwargs_from(args))
        fit_minutes = (time.time() - fit_started) / 60
        diag = result.diagnostics
        log(f"NUTS done in {fit_minutes:.1f} min: max R-hat {diag['max_rhat']:.4f}, "
            f"min ESS bulk {diag['min_ess_bulk']:.0f} tail "
            f"{diag['min_ess_tail']:.0f}, divergences "
            f"{diag['divergences_per_chain']}, E-BFMI "
            f"{np.round(diag['ebfmi_per_chain'], 2).tolist()}, tree-depth hits "
            f"{diag['treedepth_saturated']}, accept {diag['mean_accept']:.3f}, "
            f"steps {diag['mean_steps']:.0f}")
        chain_draws = result.chain_draws
        np.savez_compressed(fit_dir / "draws.npz", **{
            k: np.asarray(v) for k, v in chain_draws.items()})
        diag["table"].to_csv(fit_dir / "convergence.csv", index = False)
        curve_table = curve_convergence(chain_draws, prepared)
        curve_table.to_csv(fit_dir / "curve_convergence.csv", index = False)
        acceptance = cb.passes_acceptance(diag, curve_table)
        log(f"acceptance: {acceptance}")
        diag_summary = {k: v for k, v in diag.items() if k != "table"}
        curve_summary = {
            "max_rhat": float(curve_table["rhat"].max()),
            "min_ess_bulk": float(curve_table["ess_bulk"].min()),
            "min_ess_tail": float(curve_table["ess_tail"].min())}
        map_success = (bool(result.map_result.success) if result.map_result
                       else None)

    draws = flat_draws(chain_draws)
    grid_draws = curve_grid_draws(draws, prepared)
    frame = curves_frame(grid_draws, production)
    frame.to_parquet(fit_dir / "curves.parquet", index = False)

    weights = common.design_weights(rows, fit_config)
    y = np.nan_to_num(rows["y"].to_numpy(dtype = float))
    ht = {}
    for segment in cb.ONE_D_SEGMENTS:
        mask = (rows["segment"] == segment).to_numpy()
        ht[segment] = common.binned_ht_rates(
            rows.loc[mask, cb.SCORE_COLUMN[segment]].to_numpy(dtype = float),
            y[mask], weights[mask], prepared.knots[segment])
    color = common.COLORS.get(spec.arm, common.COLORS["A"])
    figure_curves(frame, ht, out_dir / "figures" / f"{tag}_curves_1d.png",
                  f"Bayes arm {spec.arm}", color)
    figure_matched(frame, out_dir / "figures" / f"{tag}_matched.png",
                   f"Bayes arm {spec.arm}", color)
    if "logit_beta_label" in draws:
        figure_beta_label(
            1 / (1 + np.exp(-np.asarray(draws["logit_beta_label"]))), spec,
            out_dir / "figures" / f"{tag}_beta_label.png")

    summary = {
        "tag": tag, "spec": repr(spec), "git": common.git_state(),
        "n_rows": int(len(rows)), "n_gold": int(rows["gold"].sum()),
        "knots": {k: np.round(v, 6).tolist() for k, v in prepared.knots.items()},
        "num_parameters": int(sum(np.prod(np.shape(v)) for v in
                                  cb.parameter_template(prepared).values())),
        "map_success": map_success,
        "fit_minutes": fit_minutes,
        "diagnostics": diag_summary,
        "curve_convergence": curve_summary,
        "acceptance": acceptance,
        "parameters": key_parameters({k: np.asarray(v) for k, v in draws.items()},
                                     spec),
        "prior_predictive": prior_summary,
        "silver_rates": silver_rates,
        # Current round first; report_bayes_calibration rebuilds the table
        # from this list (common.load_handoff(rounds = ...)).
        "rounds": rounds,
        "n_rows_by_round": {r: int(v["size"]) for r, v in per_round.iterrows()},
        "n_gold_by_round": {r: int(v["sum"]) for r, v in per_round.iterrows()},
    }
    if not args.no_ppc:
        ppc = posterior_predictive_checks(draws, prepared, rows, weights,
                                          seed = args.seed)
        (fit_dir / "ppc.json").write_text(json.dumps(ppc, indent = 2))
        summary["ppc"] = ppc["summary"]
        log(f"PPC: {ppc['summary']}")
    if args.deployed_impact:
        path = (common.conflation_root(config) / args.impact_version
                / "conflated.parquet")
        impact = deployed_impact(grid_draws, production, path)
        (fit_dir / "deployed_impact.json").write_text(json.dumps(impact, indent = 2))
        summary["deployed_impact"] = impact
        log(f"deployed impact: {json.dumps(impact['segments'])}")
    summary["total_minutes"] = (time.time() - started) / 60
    (fit_dir / "summary.json").write_text(json.dumps(summary, indent = 2,
                                                     default = str))
    log(f"{tag} finished in {summary['total_minutes']:.1f} min -> {fit_dir}")


if __name__ == "__main__":
    main()
