"""Design-weighted (Horvitz-Thompson) check of a deployed calibration map.

A model-free guard against bias in whatever calibration map is deployed (the
v4 curves, or the Bayesian arm C if it is adopted). Per segment and score bin,
the existence rate of the bin's phase-1 rows is estimated from the validation
labels and set against the deployed map's mean over the same rows. The check
is a review aid: nothing in the pipeline gates on it, and it reports thin or
missing data instead of raising.

**Corrected labels.** Every usable phase-1 row gets a label ``y~``: the gold
truth where the row is gold, otherwise the misclassification-corrected rate
``q(segment, verdict) = P(exists | segment, LLM verdict)`` estimated from
gold. For the exists and gone verdicts ``q`` is
``calibration_bayes.silver_label_rates`` (arm C's silver-label rates, so the
check and the Bayesian prototype apply the same correction); for the
unverifiable verdict it is the same quantity computed the same way. Each is
the design-weighted (Hajek, w = 1/pi over the production refined classes, per
validation round) share of gold rows that exist, Jeffreys-smoothed on its
Kish ESS, ``(r n_eff + 0.5) / (n_eff + 1)``.

**Estimator.** In each bin ``B`` with ``n_B`` phase-1 rows,
``r_B = mean(y~)`` over those rows, with
``SD_B = sqrt(r_B (1 - r_B) / n_B)``, the binomial SD of the bin's
exists/checked ratio. When ``r_B`` is 0 or 1 the SD alone uses the
Jeffreys-smoothed rate ``(r n + 0.5) / (n + 1)``, so that it is not zero.
The SD ignores the uncertainty in the correction rates, so it is somewhat
optimistic. The gold-only Hajek rate per bin is kept as a reference column,
not flagged on.

**Comparison.** ``d_B = m_B - r_B``, where ``m_B`` is the deployed map's mean
over the same phase-1 rows, computed through ``calibration.calibrate_frame``
exactly as deploy computes it. A bin is flagged when ``|d_B| > 1 SD`` and
flagged more strongly when ``|d_B| > 2 SD``. Under an exact map about 32% and
5% of bins cross those lines by chance (two-sided normal tails), and the
review reports the share flagged against that baseline.

**Bins.** Atom-aware bins on each single-source segment's native score
(``calibration_fit.atom_aware_edges``), merged until each holds at least
``min_rows`` phase-1 rows (``calibration_fit.merge_thin_bins``). The matched
segment gets a coarse 2-D grid (Overture columns that isolate the atoms,
crossed with OSM quartiles merged within each column) and deciles of the
handoff's ``raw_score``, which does not depend on the deployed index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpois.conflation import calibration, calibration_fit
from openpois.conflation import calibration_bayes

# Phase-1 rows each bin needs after merging.
HT_MIN_ROWS = 20
# Base atom-aware bins per single-source axis, before merging.
HT_BASE_BINS = 10
# OSM quantile bins per Overture column of the matched grid, before merging.
MATCHED_OSM_BINS = 4
MATCHED_DECILES = 10
# Chance shares of |z| > 1 and |z| > 2 under a standard normal.
CHANCE_1SD = 0.3173
CHANCE_2SD = 0.0455
VERDICTS = calibration_fit.VERDICTS
BIN_COLUMNS = (
    "segment", "view", "bin", "lo", "hi", "ov_lo", "ov_hi", "n_phase1",
    "n_gold", "rate", "sd", "ci_lower", "ci_upper", "model_mean", "diff", "z",
    "flag", "tested", "gold_rate", "gold_ess", "gold_raw_rate",
)
RATE_COLUMNS = ("segment", "verdict", "q", "raw", "n_gold", "ess", "source")


def usable_rows(validation_rows: pd.DataFrame) -> pd.DataFrame:
    """Rows the fit uses: a known LLM verdict and a current segment stratum.

    Rows of a retired stratum (round 20260730's ``overture_missing_conf``)
    were drawn under their own design and are excluded, as
    ``calibration_fit.fit_all_segments`` excludes them.
    """
    keep = (
        validation_rows["llm_verdict"].isin(VERDICTS)
        & validation_rows["stratum"].isin(calibration_fit.SEGMENTS)
    )
    return validation_rows[keep].reset_index(drop = True)


def design_weights(rows: pd.DataFrame,
                   fit_config: calibration_fit.FitConfig) -> np.ndarray:
    """``1 / pi_class`` for gold rows of one segment, 0 for the rest.

    The classes are the fit's own (verdict x LLM confidence, thin cells
    merged back to the verdict), built within each validation round when the
    table holds more than one, since each round has its own phase-2 design.
    """
    rows = rows.reset_index(drop = True)
    gold = rows["gold"].to_numpy(dtype = bool)
    weights = np.zeros(len(rows))
    rounds = (rows["validation_round"].astype(str)
              if "validation_round" in rows.columns
              else pd.Series("", index = rows.index))
    for round_id in rounds.unique():
        mask = (rounds == round_id).to_numpy()
        part = rows[mask].reset_index(drop = True)
        classes = calibration_fit.merge_thin_cells(
            calibration_fit.refined_class(
                part, refine = fit_config.refine_by_confidence
            ),
            gold[mask], fit_config.min_cell_gold,
        ).reset_index(drop = True)
        inclusion = calibration_fit.inclusion_by_class(classes, gold[mask])
        weights[mask] = classes.astype(str).map(
            lambda c: inclusion.get(c, {}).get("weight", 0.0)
        ).to_numpy(dtype = float)
    return weights * gold


def correction_rates(rows: pd.DataFrame,
                     fit_config: calibration_fit.FitConfig) -> pd.DataFrame:
    """``q = P(exists | segment, LLM verdict)`` from gold, one row per cell.

    Exists and gone come from ``calibration_bayes.silver_label_rates`` (arm
    C's q). Unverifiable, which that function does not cover, is the same
    Jeffreys-smoothed design-weighted gold share, computed with its helpers.
    If ``silver_label_rates`` refuses the table (a segment without gold of a
    definitive verdict), every cell is computed with the helpers, and a cell
    with no gold gets a NaN ``q``. Columns: ``RATE_COLUMNS``; ``source``
    names where the rate came from.
    """
    empty = pd.DataFrame(columns = list(RATE_COLUMNS))
    if rows.empty:
        return empty
    try:
        silver = calibration_bayes.silver_label_rates(rows, fit_config)
    except (ValueError, KeyError):
        silver = {}
    pooled = calibration_bayes._weighted_gold(rows, fit_config)
    out = []
    for segment in calibration_fit.SEGMENTS:
        if not (pooled["segment"] == segment).any():
            continue
        for verdict in VERDICTS:
            entry = silver.get(segment, {})
            if verdict in entry:
                out.append({
                    "segment": segment, "verdict": verdict,
                    "q": float(entry[verdict]),
                    "raw": float(entry[f"raw_{verdict}"]),
                    "n_gold": int(entry[f"n_{verdict}"]),
                    "ess": float(entry[f"ess_{verdict}"]),
                    "source": "silver_label_rates",
                })
                continue
            sel = ((pooled["segment"] == segment)
                   & (pooled["verdict"] == verdict)
                   & (pooled["w"] > 0)).to_numpy()
            w = pooled.loc[sel, "w"].to_numpy(dtype = float)
            y = pooled.loc[sel, "y"].to_numpy(dtype = float)
            if len(w) == 0:
                out.append({"segment": segment, "verdict": verdict,
                            "q": np.nan, "raw": np.nan, "n_gold": 0,
                            "ess": 0.0, "source": "no gold"})
                continue
            q, raw, ess = calibration_bayes._jeffreys_rate(w, y)
            out.append({"segment": segment, "verdict": verdict, "q": float(q),
                        "raw": float(raw), "n_gold": int(len(w)),
                        "ess": float(ess), "source": "design-weighted gold"})
    return pd.DataFrame(out, columns = list(RATE_COLUMNS)) if out else empty


def corrected_labels(rows: pd.DataFrame, rates: pd.DataFrame) -> np.ndarray:
    """``y~`` per row: gold truth for gold rows, else ``q`` of its cell.

    ``rates`` is :func:`correction_rates`' table. A silver row whose
    (segment, verdict) has no rate gets NaN and drops out of the bin rates.
    """
    lookup = {(r.segment, r.verdict): r.q for r in rates.itertuples()}
    q = np.array([
        lookup.get((s, v), np.nan)
        for s, v in zip(rows["segment"].astype(str),
                        rows["llm_verdict"].astype(str))
    ], dtype = float)
    gold = rows["gold"].to_numpy(dtype = bool)
    return np.where(gold, rows["y"].to_numpy(dtype = float), q)


def binomial_sd(rate: float, n: float) -> float:
    """``sqrt(r (1 - r) / n)``, Jeffreys-smoothed when ``r`` is 0 or 1.

    The smoothed rate ``(r n + 0.5) / (n + 1)`` is used for the SD only, so a
    bin whose labels are all one outcome still gets a nonzero SD. NaN when
    ``n`` is not positive.
    """
    if not np.isfinite(rate) or not np.isfinite(n) or n <= 0:
        return float("nan")
    r = float(rate)
    if r <= 0.0 or r >= 1.0:
        r = (r * n + 0.5) / (n + 1.0)
    return float(np.sqrt(r * (1.0 - r) / n))


def corrected_rate(labels) -> dict:
    """Mean corrected label over a bin's rows, with its binomial SD and CI."""
    labels = np.asarray(labels, dtype = float)
    labels = labels[np.isfinite(labels)]
    n = len(labels)
    if n == 0:
        return {"n_rows": 0, "rate": float("nan"), "sd": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan")}
    rate = float(labels.mean())
    sd = binomial_sd(rate, float(n))
    return {
        "n_rows": int(n),
        "rate": rate,
        "sd": sd,
        "ci_lower": float(max(0.0, rate - 1.96 * sd)),
        "ci_upper": float(min(1.0, rate + 1.96 * sd)),
    }


def hajek(y, weights) -> dict:
    """Gold-only Hajek rate and Kish ESS over rows with positive weight.

    The reference beside the corrected rate. ``raw_rate`` is the unweighted
    exists/checked ratio over the same gold rows.
    """
    y = np.asarray(y, dtype = float)
    w = np.asarray(weights, dtype = float)
    keep = (w > 0) & np.isfinite(y)
    y, w = y[keep], w[keep]
    if len(y) == 0:
        return {"n_gold": 0, "ess": 0.0, "raw_rate": float("nan"),
                "ht_rate": float("nan"), "sd": float("nan")}
    rate = float(np.sum(w * y) / np.sum(w))
    ess = float(np.sum(w) ** 2 / np.sum(w ** 2))
    return {
        "n_gold": int(len(y)),
        "ess": ess,
        "raw_rate": float(np.mean(y)),
        "ht_rate": rate,
        "sd": binomial_sd(rate, ess),
    }


def flag_level(z) -> np.ndarray:
    """0 within 1 SD, 1 beyond 1 SD, 2 beyond 2 SD (NaN z gives 0)."""
    a = np.abs(np.asarray(z, dtype = float))
    a = np.where(np.isfinite(a), a, 0.0)
    return np.where(a > 2.0, 2, np.where(a > 1.0, 1, 0))


def bin_table(bin_ids, n_bins: int, labels, model, y = None, weights = None,
              min_rows: int = HT_MIN_ROWS) -> pd.DataFrame:
    """Per-bin corrected rate against the deployed map's mean.

    ``bin_ids`` assigns each phase-1 row to a bin (``-1`` leaves it out);
    ``labels`` is each row's corrected label ``y~``; ``model`` its deployed
    ``conf_mean``. ``y`` and ``weights`` (gold truth and 1/pi, 0 off gold)
    give the gold-only Hajek reference columns. A bin with fewer than
    ``min_rows`` labelled rows or no finite model value is reported but not
    tested (``tested`` False, ``flag`` 0).
    """
    bin_ids = np.asarray(bin_ids, dtype = int)
    labels = np.asarray(labels, dtype = float)
    model = np.asarray(model, dtype = float)
    y = np.full(len(labels), np.nan) if y is None else np.asarray(y, float)
    weights = (np.zeros(len(labels)) if weights is None
               else np.asarray(weights, dtype = float))
    out = []
    for b in range(n_bins):
        sel = bin_ids == b
        stats = corrected_rate(labels[sel])
        gold = hajek(y[sel], weights[sel])
        use = sel & np.isfinite(labels) & np.isfinite(model)
        model_mean = float(model[use].mean()) if use.any() else np.nan
        diff = model_mean - stats["rate"]
        z = diff / stats["sd"] if stats["sd"] > 0 else np.nan
        tested = bool(stats["n_rows"] >= min_rows and np.isfinite(z))
        out.append({
            "bin": b, "n_phase1": int(sel.sum()), "n_gold": gold["n_gold"],
            "rate": stats["rate"], "sd": stats["sd"],
            "ci_lower": stats["ci_lower"], "ci_upper": stats["ci_upper"],
            "model_mean": model_mean, "diff": float(diff), "z": float(z),
            "flag": int(flag_level(z)) if tested else 0, "tested": tested,
            "gold_rate": gold["ht_rate"], "gold_ess": gold["ess"],
            "gold_raw_rate": gold["raw_rate"],
        })
    return pd.DataFrame(out)


def axis_edges(values, n_bins: int = HT_BASE_BINS,
               min_rows: int = HT_MIN_ROWS) -> np.ndarray:
    """Atom-aware edges on one score axis, merged to ``min_rows`` per bin."""
    values = np.asarray(values, dtype = float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([])
    edges = calibration_fit.atom_aware_edges(values, n_bins)
    if len(edges) < 2:
        edges = np.array([values.min(), values.max() + 1e-9])
    return calibration_fit.merge_thin_bins(
        edges, values, np.ones(len(values), dtype = bool), min_rows
    )


def quantile_edges(values, n_bins: int,
                   min_rows: int = HT_MIN_ROWS) -> np.ndarray:
    """Equal-count edges over the phase-1 rows, merged to ``min_rows``."""
    values = np.round(np.asarray(values, dtype = float),
                      calibration_fit.SCORE_DECIMALS)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([values.min(), values.max() + 1e-9])
    return calibration_fit.merge_thin_bins(
        edges, values, np.ones(len(values), dtype = bool), min_rows
    )


def _assign(values, edges: np.ndarray) -> np.ndarray:
    """Deploy-convention bin per value; -1 for NaN or when there are no bins."""
    values = np.round(np.asarray(values, dtype = float),
                      calibration_fit.SCORE_DECIMALS)
    if len(edges) < 2:
        return np.full(len(values), -1)
    out = calibration_fit._bin_index(values, edges)
    return np.where(np.isfinite(values), out, -1)


def matched_grid(rows: pd.DataFrame, min_rows: int = HT_MIN_ROWS,
                 n_osm: int = MATCHED_OSM_BINS) -> dict:
    """The matched segment's coarse 2-D cells.

    Overture columns come from ``atom_aware_edges`` with a single non-atom
    split, so each atom (0.919912 and 0.990219 on round 20260730) is its own
    column between the stretches below, between and above them; thin columns
    merge. Within each column, OSM quartiles of the matched phase-1 rows are
    merged until each cell holds ``min_rows`` rows, so the rows of the grid
    need not line up across columns. Returns the Overture edges, one OSM edge
    array per column, and each row's flat cell id.
    """
    osm = np.round(rows["osm_score"].to_numpy(dtype = float),
                   calibration_fit.SCORE_DECIMALS)
    overture = rows["overture_score"].to_numpy(dtype = float)
    ov_edges = axis_edges(overture, n_bins = 1, min_rows = min_rows)
    column = _assign(overture, ov_edges)
    finite = np.isfinite(osm)
    base = (np.unique(np.quantile(osm[finite],
                                  np.linspace(0.0, 1.0, n_osm + 1)))
            if finite.any() else np.array([]))
    osm_edges, cell = [], np.full(len(rows), -1)
    offset = 0
    for j in range(max(len(ov_edges) - 1, 0)):
        in_column = (column == j) & finite
        edges = base
        if len(edges) >= 2:
            edges = calibration_fit.merge_thin_bins(
                base, osm[in_column], np.ones(int(in_column.sum()), bool),
                min_rows,
            )
        osm_edges.append(edges)
        if len(edges) >= 2:
            ids = _assign(osm, edges)
            cell[in_column] = offset + ids[in_column]
            offset += len(edges) - 1
    return {"ov_edges": ov_edges, "osm_edges": osm_edges, "cell": cell,
            "n_cells": offset}


def atom_columns(ov_edges: np.ndarray) -> list:
    """Indices of the Overture columns that hold exactly one atom."""
    step = 10.0 ** -calibration_fit.SCORE_DECIMALS
    return [j for j in range(len(ov_edges) - 1)
            if np.isclose(ov_edges[j + 1] - ov_edges[j], step, rtol = 0.0,
                          atol = 0.1 * step)]


def deployed_means(rows: pd.DataFrame, segment: str, curves: dict,
                   metadata: dict) -> np.ndarray:
    """Each row's deployed ``conf_mean``, through the deploy code itself.

    The rows are passed to ``calibration.calibrate_frame`` as conflated POIs
    of ``segment`` with the index parameters, index mode and score rounding
    read from the curve metadata, so pool-mode and interaction-mode curves,
    rounded and unrounded, are applied exactly as deploy applies them. Rows
    the map cannot score are NaN.
    """
    if segment not in curves:
        return np.full(len(rows), np.nan)
    frame = pd.DataFrame({
        "source": segment,
        "osm_conf_mean": rows["osm_score"].to_numpy(dtype = float),
        "overture_confidence": rows["overture_score"].to_numpy(dtype = float),
        "conf_mean": np.nan,
    })
    out = calibration.calibrate_frame(
        frame, {segment: curves[segment]},
        pool_params = calibration.pool_params_from_metadata(metadata),
        index_modes = calibration.index_modes_from_metadata(metadata),
        score_decimals = calibration.score_decimals_from_metadata(metadata),
    )
    return out["conf_mean"].to_numpy(dtype = float)


def _with_edges(table: pd.DataFrame, segment: str, view: str,
                lo, hi, ov_lo = None, ov_hi = None) -> pd.DataFrame:
    table = table.copy()
    table["segment"] = segment
    table["view"] = view
    table["lo"] = np.asarray(lo, dtype = float)
    table["hi"] = np.asarray(hi, dtype = float)
    nan = np.full(len(table), np.nan)
    table["ov_lo"] = nan if ov_lo is None else np.asarray(ov_lo, dtype = float)
    table["ov_hi"] = nan if ov_hi is None else np.asarray(ov_hi, dtype = float)
    return table[list(BIN_COLUMNS)]


def segment_views(rows: pd.DataFrame, segment: str, labels, model, y,
                  weights, min_rows: int = HT_MIN_ROWS,
                  n_bins: int = HT_BASE_BINS) -> tuple:
    """Bin tables for one segment's views, plus the matched grid (or None)."""
    tables, grid = [], None
    common = {"y": y, "weights": weights, "min_rows": min_rows}
    if segment in calibration_fit.POOLED_SEGMENTS:
        grid = matched_grid(rows, min_rows = min_rows)
        if grid["n_cells"]:
            table = bin_table(grid["cell"], grid["n_cells"], labels, model,
                              **common)
            lo, hi, ov_lo, ov_hi = [], [], [], []
            ov_edges = grid["ov_edges"]
            for j, edges in enumerate(grid["osm_edges"]):
                for k in range(max(len(edges) - 1, 0)):
                    lo.append(edges[k])
                    hi.append(edges[k + 1])
                    ov_lo.append(ov_edges[j])
                    ov_hi.append(ov_edges[j + 1])
            tables.append(_with_edges(table, segment, "cells", lo, hi,
                                      ov_lo, ov_hi))
        if "raw_score" in rows.columns:
            raw = rows["raw_score"].to_numpy(dtype = float)
            edges = quantile_edges(raw, MATCHED_DECILES, min_rows = min_rows)
            if len(edges) >= 2:
                table = bin_table(_assign(raw, edges), len(edges) - 1, labels,
                                  model, **common)
                tables.append(_with_edges(table, segment, "raw_score_deciles",
                                          edges[:-1], edges[1:]))
    else:
        column = "osm_score" if segment == "osm" else "overture_score"
        values = rows[column].to_numpy(dtype = float)
        edges = axis_edges(values, n_bins = n_bins, min_rows = min_rows)
        if len(edges) >= 2:
            table = bin_table(_assign(values, edges), len(edges) - 1, labels,
                              model, **common)
            tables.append(_with_edges(table, segment, column, edges[:-1],
                                      edges[1:]))
    return tables, grid


def run_ht_check(validation_rows: pd.DataFrame, curves: dict,
                 metadata: dict, fit_config: calibration_fit.FitConfig,
                 min_rows: int = HT_MIN_ROWS,
                 n_bins: int = HT_BASE_BINS) -> dict:
    """The design-weighted check of a deployed map, for every segment.

    ``curves`` and ``metadata`` are what ``calibration.read_curves`` and
    ``read_curve_metadata`` return. Never raises on thin or empty data: a
    segment with no rows, no gold or no curve is recorded in ``notes``.

    Returns ``bins`` (one row per bin and view, ``BIN_COLUMNS``), ``large``
    (calibration in the large per segment), ``summary`` (share of tested bins
    beyond 1 and 2 SD per view), ``rates`` (the correction rates used),
    ``notes``, ``grids`` (the matched cells) and ``rows`` (per-segment row
    frames with the corrected label, weight and model, for the plots).
    """
    usable = usable_rows(validation_rows)
    rates = correction_rates(usable, fit_config)
    tables, large, notes, grids, seg_rows = [], [], [], {}, {}
    for segment in calibration_fit.SEGMENTS:
        raw_rows = usable[usable["segment"] == segment].reset_index(
            drop = True
        )
        # Bins use rounded scores; the deployed map gets the scores as they
        # are and rounds them only if its metadata says the fit did.
        rows = calibration_fit.round_scores(raw_rows)
        if rows.empty:
            notes.append(f"{segment}: no phase-1 rows in the handoff")
            continue
        if not rows["gold"].any():
            notes.append(f"{segment}: no gold rows, so no correction rates; "
                         f"nothing to check")
            continue
        missing = rates[(rates["segment"] == segment) & rates["q"].isna()]
        for verdict in missing["verdict"]:
            notes.append(f"{segment}: no gold {verdict} verdicts, so its "
                         f"silver rows have no corrected label and drop out")
        labels = corrected_labels(rows, rates)
        weights = design_weights(rows, fit_config)
        try:
            model = deployed_means(raw_rows, segment, curves, metadata)
        except (ValueError, KeyError) as error:
            notes.append(f"{segment}: the deployed map could not be applied "
                         f"({error})")
            model = np.full(len(rows), np.nan)
        if segment not in curves:
            notes.append(f"{segment}: no deployed curve")
        y = rows["y"].to_numpy(dtype = float)
        seg_rows[segment] = rows.assign(
            y_corrected = labels, ht_weight = weights, model = model,
            overture_score_unrounded = raw_rows["overture_score"].to_numpy(),
        )
        views, grid = segment_views(rows, segment, labels, model, y, weights,
                                    min_rows = min_rows, n_bins = n_bins)
        tables += views
        if grid is not None:
            grids[segment] = grid
        stats = corrected_rate(labels)
        gold = hajek(y, weights)
        use = np.isfinite(labels) & np.isfinite(model)
        model_mean = float(model[use].mean()) if use.any() else np.nan
        diff = model_mean - stats["rate"]
        z = diff / stats["sd"] if stats["sd"] > 0 else np.nan
        large.append({
            "segment": segment, "n_phase1": int(len(rows)),
            "n_gold": gold["n_gold"], "rate": stats["rate"],
            "sd": stats["sd"], "model_mean": model_mean, "diff": float(diff),
            "z": float(z), "gold_rate": gold["ht_rate"],
            "gold_ess": gold["ess"],
        })
        thin = sum(int((~t["tested"].astype(bool)).sum()) for t in views)
        if thin:
            notes.append(f"{segment}: {thin} bin(s) below {min_rows} rows or "
                         f"without a model value; shown, not tested")
    bins = (pd.concat(tables, ignore_index = True) if tables
            else pd.DataFrame(columns = list(BIN_COLUMNS)))
    return {
        "bins": bins,
        "large": pd.DataFrame(large),
        "summary": flag_summary(bins),
        "rates": rates,
        "notes": notes,
        "grids": grids,
        "rows": seg_rows,
        "min_rows": min_rows,
    }


def flag_summary(bins: pd.DataFrame) -> pd.DataFrame:
    """Per view: bins tested, flagged beyond 1 and 2 SD, and their shares."""
    out = []
    if bins.empty:
        return pd.DataFrame(columns = ["segment", "view", "bins", "tested",
                                       "beyond_1sd", "beyond_2sd",
                                       "share_1sd", "share_2sd"])
    for (segment, view), group in bins.groupby(["segment", "view"],
                                               sort = False):
        tested = group[group["tested"].astype(bool)]
        n = len(tested)
        beyond1 = int((tested["flag"] >= 1).sum())
        beyond2 = int((tested["flag"] >= 2).sum())
        out.append({
            "segment": segment, "view": view, "bins": int(len(group)),
            "tested": n, "beyond_1sd": beyond1, "beyond_2sd": beyond2,
            "share_1sd": beyond1 / n if n else np.nan,
            "share_2sd": beyond2 / n if n else np.nan,
        })
    return pd.DataFrame(out)


def bin_label(row) -> str:
    """Human-readable range of one bin-table row."""
    if np.isfinite(row["ov_lo"]):
        return (f"ov {row['ov_lo']:.6f}-{row['ov_hi']:.6f}, "
                f"osm {row['lo']:.4f}-{row['hi']:.4f}")
    return f"{row['lo']:.6f}-{row['hi']:.6f}"


def _fmt(value, spec: str) -> str:
    return "-" if not np.isfinite(value) else format(value, spec)


def method_text(min_rows: int) -> str:
    """The method statement shared by the report section and the PDF."""
    return (
        "Every phase-1 row carries a corrected label: its gold truth if it is "
        "gold, else q = P(exists | segment, LLM verdict), the design-weighted "
        "(w = 1/pi_class) and Jeffreys-smoothed gold share for its verdict. "
        "Per bin, r = mean corrected label over the bin's phase-1 rows, set "
        "against the deployed map's mean over the same rows; z = (model - r) "
        "/ SD with SD = sqrt(r(1 - r)/n), n the bin's rows. Bins are merged "
        f"to at least {min_rows} rows. An exact map crosses 1 SD in about "
        f"{CHANCE_1SD:.0%} of bins and 2 SD in about {CHANCE_2SD:.0%} by "
        "chance. The SD ignores the uncertainty in the correction rates, so "
        "it is somewhat optimistic. The gold-only Hajek rate is shown for "
        "reference and not flagged on. This check never fails the run."
    )


def report_lines(check: dict, pdf_name: str = None) -> list:
    """The fit report's markdown section for a finished check."""
    lines = ["## Design-weighted (Horvitz-Thompson) check", "",
             method_text(check["min_rows"]), ""]
    if pdf_name:
        lines += [f"Review document: [{pdf_name}]({pdf_name})", ""]
    lines += ["| segment | view | bins tested | beyond 1 SD (32%) | "
              "beyond 2 SD (5%) |", "|---|---|---|---|---|"]
    for row in check["summary"].itertuples():
        lines.append(
            f"| {row.segment} | {row.view} | {row.tested} | "
            f"{row.beyond_1sd} ({_fmt(row.share_1sd, '.0%')}) | "
            f"{row.beyond_2sd} ({_fmt(row.share_2sd, '.0%')}) |"
        )
    lines += ["", "Calibration in the large:", "",
              "| segment | phase-1 rows | corrected rate | model mean | "
              "model - rate | z | gold-only HT |",
              "|---|---|---|---|---|---|---|"]
    for row in check["large"].itertuples():
        lines.append(
            f"| {row.segment} | {row.n_phase1:,} | {_fmt(row.rate, '.3f')} | "
            f"{_fmt(row.model_mean, '.3f')} | {_fmt(row.diff, '+.3f')} | "
            f"{_fmt(row.z, '+.2f')} | {_fmt(row.gold_rate, '.3f')} |"
        )
    lines += ["", "Correction rates q = P(exists | segment, verdict):", "",
              "| segment | verdict | q | raw | gold | ESS |",
              "|---|---|---|---|---|---|"]
    for row in check["rates"].itertuples():
        lines.append(
            f"| {row.segment} | {row.verdict} | {_fmt(row.q, '.4f')} | "
            f"{_fmt(row.raw, '.4f')} | {row.n_gold} | {_fmt(row.ess, '.1f')} |"
        )
    flagged = check["bins"][check["bins"]["flag"] >= 1]
    if len(flagged):
        lines += ["", "Flagged bins (** beyond 2 SD):", ""]
        for _, row in flagged.iterrows():
            mark = "**" if row["flag"] >= 2 else ""
            lines.append(
                f"- {mark}{row['segment']} {row['view']} {bin_label(row)}: "
                f"rate {row['rate']:.3f}, model {row['model_mean']:.3f}, "
                f"z {row['z']:+.2f}{mark}"
            )
    if check["notes"]:
        lines += [""] + [f"- Note: {note}" for note in check["notes"]]
    return lines + [""]
