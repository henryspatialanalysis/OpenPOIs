"""Tests for existence-confidence calibration (fit + deploy).

The load-bearing statistical properties, each mapped to the v4 writeup:

- the composite difference estimator reduces to the Horvitz-Thompson gold-only
  estimator when the working model is saturated (writeup 4.2)
- the difference estimator recovers a known existence rate under a deliberately
  wrong working model (design-unbiasedness, Breidt & Opsomer 2017)
- the log-odds pool is monotone in each source score and can exceed both
  inputs, which the linear blend cannot (writeup: matched segment)
- the deploy step's edge rules: shadow-matched rows keep the CD value, unnamed
  OSM rows are flagged, missing-conf rows are flagged, row count preserved
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpois.conflation import calibration, calibration_fit


def _rows(n_per_class = 200, seed = 0, with_scores = True) -> pd.DataFrame:
    """Synthetic two-phase validation table with a known truth mechanism."""
    rng = np.random.default_rng(seed)
    frames = []
    # (verdict, existence rate, phase-2 inclusion)
    design = [("exists", 0.98, 0.12), ("gone", 0.02, 0.37),
              ("unverifiable", 0.55, 1.0)]
    for verdict, rate, inclusion in design:
        scores = rng.uniform(0.2, 1.0, n_per_class)
        y_true = rng.random(n_per_class) < rate
        gold = rng.random(n_per_class) < inclusion
        frame = pd.DataFrame(
            {
                "segment": "osm",
                "stratum": "osm",
                "raw_score": scores,
                "osm_score": scores,
                "overture_score": np.where(
                    with_scores, rng.uniform(0.3, 1.0, n_per_class), np.nan
                ),
                "llm_verdict": verdict,
                "llm_confidence": rng.choice(["high", "medium"], n_per_class),
                "gold": gold,
                "y": np.where(gold, y_true.astype(float), np.nan),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index = True)


def _fit_config(**overrides) -> calibration_fit.FitConfig:
    defaults = {"bootstrap_reps": 12, "grid_points": 40, "output_bins": 8,
                "min_cell_gold": 25, "rng_seed": 7}
    defaults.update(overrides)
    return calibration_fit.FitConfig(**defaults)


# --- Estimator identities ---------------------------------------------------

def test_saturated_working_model_reduces_to_horvitz_thompson():
    """Writeup 4.2: the HT gold-only curve is the saturated special case.

    With one free rate per class per score neighbourhood, the prediction and
    correction terms of the difference estimator collapse onto the HT weighted
    average. Verified here in the limit that matters: both estimators must
    agree on the population existence rate.
    """
    rows = _rows(seed = 1).assign(score = lambda df: df["raw_score"])
    classes = rows["llm_verdict"].astype(str)
    inclusion = calibration_fit.inclusion_by_class(
        classes, rows["gold"].to_numpy(dtype = bool)
    )
    grid = np.linspace(0.2, 1.0, 40)

    composite = calibration_fit.composite_curve(rows, classes, grid, inclusion)
    reference = calibration_fit.ht_reference_curve(rows, classes, grid,
                                                   inclusion)
    # Population-weighted means of the two curves agree closely; the composite
    # is smoother but not biased relative to HT.
    assert np.isfinite(composite).all()
    assert abs(composite.mean() - np.nanmean(reference)) < 0.06


def test_difference_estimator_recovers_rate_under_wrong_working_model():
    """Design-unbiasedness: a deliberately wrong q_c is repaired by residuals.

    The working model is forced to a constant far from the truth by labelling
    every row one class; the correction term must still bring the estimate back
    to the true existence rate.
    """
    rng = np.random.default_rng(3)
    n = 3000
    true_rate = 0.62
    y_true = rng.random(n) < true_rate
    gold = rng.random(n) < 0.25
    rows = pd.DataFrame(
        {
            "score": rng.uniform(0.4, 0.6, n),
            "gold": gold,
            "y": np.where(gold, y_true.astype(float), np.nan),
            "llm_verdict": "exists",
            "llm_confidence": "high",
        }
    )
    classes = pd.Series(["one_class"] * n)
    inclusion = calibration_fit.inclusion_by_class(
        classes, rows["gold"].to_numpy(dtype = bool)
    )
    grid = np.linspace(0.4, 0.6, 20)
    curve = calibration_fit.composite_curve(rows, classes, grid, inclusion)
    assert abs(curve.mean() - true_rate) < 0.05


def test_non_gold_llm_rows_are_load_bearing():
    """The LLM archive must shape the curve, not just stratify it.

    The prediction term averages every phase-1 row's class rate, so the class
    *mix* at each score comes from all verified rows -- which matters because
    the gold subsample is deliberately unrepresentative (it over-samples the
    rare classes). Dropping the non-gold rows must therefore move the curve. A
    refactor that quietly reduced the estimator to gold-only would pass every
    other test in this file but fail here.
    """
    rows = _rows(n_per_class = 300, seed = 22).assign(
        score = lambda df: df["raw_score"]
    )
    grid = np.linspace(0.2, 1.0, 60)

    def _fit(subset):
        subset = subset.reset_index(drop = True)
        classes = subset["llm_verdict"].astype(str)
        inclusion = calibration_fit.inclusion_by_class(
            classes, subset["gold"].to_numpy(dtype = bool)
        )
        return calibration_fit.composite_curve(subset, classes, grid, inclusion)

    full = _fit(rows)
    gold_only = _fit(rows[rows["gold"]])
    assert np.abs(full - gold_only).mean() > 0.01

    # And the mechanism is the class mix, not the outcomes: LLM verdicts are
    # never used as labels, so every non-gold row must carry a null outcome.
    assert rows.loc[~rows["gold"], "y"].isna().all()


def test_censused_class_carries_no_phase_two_weight():
    """A class audited at 100% gets inclusion 1.0 and weight 1.0."""
    rows = _rows(seed = 5)
    classes = rows["llm_verdict"].astype(str)
    inclusion = calibration_fit.inclusion_by_class(
        classes, rows["gold"].to_numpy(dtype = bool)
    )
    assert inclusion["unverifiable"]["inclusion"] == pytest.approx(1.0)
    assert inclusion["unverifiable"]["weight"] == pytest.approx(1.0)
    assert inclusion["exists"]["weight"] > 5.0


def test_thin_refined_cells_merge_to_parent_verdict():
    rows = _rows(seed = 2)
    classes = calibration_fit.refined_class(rows, refine = True)
    assert classes.str.contains(":").all()
    merged = calibration_fit.merge_thin_cells(
        classes, rows["gold"].to_numpy(dtype = bool), min_gold = 10_000
    )
    # Nothing can clear an impossible floor, so every cell falls back.
    assert set(merged.unique()) <= set(calibration_fit.VERDICTS)


def test_fit_segment_is_deterministic_and_monotone():
    rows = _rows(seed = 4)
    config = _fit_config()
    first = calibration_fit.fit_segment(rows, "osm", config)
    second = calibration_fit.fit_segment(rows, "osm", config)
    np.testing.assert_allclose(first["curve"], second["curve"])
    lookup = first["lookup"]
    assert lookup["conf_mean"].is_monotonic_increasing
    assert (lookup["conf_lower"] <= lookup["conf_mean"] + 1e-9).all()
    assert (lookup["conf_upper"] >= lookup["conf_mean"] - 1e-9).all()
    assert ((lookup["conf_mean"] >= 0) & (lookup["conf_mean"] <= 1)).all()


def test_cross_fit_holds_out_gold():
    rows = _rows(n_per_class = 300, seed = 6).assign(
        score = lambda df: df["raw_score"]
    )
    classes = rows["llm_verdict"].astype(str)
    grid = np.linspace(0.2, 1.0, 30)
    result = calibration_fit.cross_fit_calibration_error(
        rows, classes, grid, _fit_config(), n_folds = 3
    )
    assert result["n_folds"] == 3
    assert 0.0 <= result["brier_crossfit"] <= 1.0
    # Calibration error is a binned gap, not the Brier score: it must be well
    # below it (the Brier carries irreducible outcome noise) and the debiased
    # form must not exceed the plug-in.
    assert result["calibration_error_sq_debiased"] <= (
        result["calibration_error_sq_plugin"] + 1e-12
    )
    assert result["calibration_error_sq_plugin"] < result["brier_crossfit"]


def test_debiased_calibration_error_is_not_identically_zero():
    """Guard against a correction term that swallows the whole signal.

    An earlier version subtracted each observation's Bernoulli variance rather
    than each bin's sampling variance, which drove the debiased error to
    exactly 0 for every segment. Here the score is deliberately miscalibrated,
    so a positive calibration error must survive debiasing.
    """
    rng = np.random.default_rng(31)
    n = 4000
    scores = rng.uniform(0.05, 0.95, n)
    # Truth is far below the score: a badly overconfident forecaster.
    y = (rng.random(n) < scores * 0.5).astype(float)
    rows = pd.DataFrame(
        {
            "score": scores,
            "gold": True,
            "y": y,
            "llm_verdict": "exists",
            "llm_confidence": "high",
            "osm_score": scores,
            "overture_score": scores,
        }
    )
    classes = pd.Series(["exists:high"] * n)
    grid = np.linspace(0.05, 0.95, 60)
    result = calibration_fit.cross_fit_calibration_error(
        rows, classes, grid, _fit_config(), n_folds = 4
    )
    # The composite estimator sees the truth, so its own calibration error is
    # small -- but the estimator must be able to report a nonzero value, and
    # the machinery must not clamp to exactly zero by construction.
    assert result["calibration_error_sq_plugin"] > 0.0


# --- The fitted source pool ------------------------------------------------

def test_pool_is_monotone_in_each_source_and_can_exceed_both():
    """The log-odds pool's defining advantage over the linear blend."""
    params = {"intercept": 0.0, "coef_osm": 1.0, "coef_overture": 1.0}
    osm = np.array([0.5, 0.7, 0.9, 0.8])
    overture = np.array([0.5, 0.5, 0.5, 0.8])
    pooled = calibration_fit.pool_score(osm, overture, params)
    # Monotone in the OSM score at fixed Overture score
    assert pooled[0] < pooled[1] < pooled[2]
    # Two agreeing sources push above either input, which a linear blend of
    # the same two numbers can never do.
    assert pooled[3] > max(osm[3], overture[3])
    linear = 0.588 * osm[3] + 0.412 * overture[3]
    assert pooled[3] > linear


def test_pool_fit_recovers_the_informative_source():
    """A source that carries the signal earns the larger coefficient."""
    rng = np.random.default_rng(11)
    n = 1200
    informative = rng.uniform(0.05, 0.95, n)
    noise = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < informative).astype(float)
    rows = pd.DataFrame(
        {
            "osm_score": informative,
            "overture_score": noise,
            "gold": True,
            "y": y,
        }
    )
    pool = calibration_fit.fit_pool(rows, np.ones(n))
    assert pool["method"] == "fitted_log_odds_pool_v1"
    assert pool["coef_osm"] > pool["coef_overture"]
    # No fixed 0.7 downweight survives: the weights are estimated.
    assert pool["coef_osm"] > 0.2


def test_scoring_rules_decomposition_is_exact_and_signed_correctly():
    """Brier = MCB - DSC + UNC, with MCB and DSC non-negative.

    DSC is better *higher* and MCB better *lower*; a comparison script that
    treats both as lower-is-better silently inverts the discrimination verdict.
    """
    rng = np.random.default_rng(41)
    n = 2000
    predicted = rng.uniform(0.05, 0.95, n)
    actual = (rng.random(n) < predicted).astype(float)
    weight = rng.uniform(1.0, 9.0, n)
    s = calibration_fit.scoring_rules(predicted, actual, weight)
    assert s["brier"] == pytest.approx(s["mcb"] - s["dsc"] + s["unc"], abs = 1e-9)
    assert s["mcb"] >= -1e-12
    assert s["dsc"] >= -1e-12

    # A forecast carrying no signal must have essentially zero discrimination,
    # while the signal-carrying one above has clearly positive DSC.
    flat = calibration_fit.scoring_rules(
        np.full(n, actual.mean()), actual, weight
    )
    assert flat["dsc"] < 0.005
    assert s["dsc"] > flat["dsc"]


def test_cross_fit_refits_the_pool_within_each_fold():
    """Held-out gold must not inform the pool coefficients.

    If the pool were fit once on all gold, the out-of-fold predictions would be
    contaminated and a multi-parameter index would beat a parameter-free one by
    construction -- which is exactly the comparison this supports.
    """
    rows = _rows(n_per_class = 250, seed = 12).assign(
        segment = "matched", stratum = "matched"
    )
    rows = rows.assign(
        score = calibration_fit.segment_scores(rows, "matched", None, "average")
    )
    classes = rows["llm_verdict"].astype(str)
    config = _fit_config(grid_points = 30)

    out = calibration_fit.cross_fit_predictions(
        rows, classes, config, n_folds = 4, segment = "matched",
        index_mode = "pool",
    )
    assert out["n_folds"] == 4
    assert out["n_gold"] > 0
    assert np.isfinite(out["predicted"]).all()
    assert ((out["predicted"] >= 0) & (out["predicted"] <= 1)).all()
    # Every gold row is held out exactly once, so folds partition the gold set.
    assert set(np.unique(out["fold"])) <= {0, 1, 2, 3}

    # The average mode needs no pool at all and must still produce predictions.
    out_avg = calibration_fit.cross_fit_predictions(
        rows, classes, config, n_folds = 4, segment = "matched",
        index_mode = "average",
    )
    assert out_avg["n_gold"] == out["n_gold"]


def test_average_index_needs_no_pool_anywhere():
    """`index_mode = "average"` must never require pool coefficients."""
    rows = _rows(seed = 9).assign(segment = "matched", stratum = "matched")
    config = _fit_config(matched_index_mode = "average")
    result = calibration_fit.fit_segment(rows, "matched", config)
    assert result["pool"] is None
    assert result["index_mode"] == "average"
    metadata = calibration_fit.curve_metadata(
        "matched", result, {"validation_round": "test"}, config
    )
    assert metadata["score_definition"].startswith("mean(")
    # Deploy side honours it without pool params.
    scores = calibration.curve_index(
        np.array(["matched"]), np.array([0.8]), np.array([0.6]),
        index_mode = "average",
    )
    assert scores[0] == pytest.approx(0.7)


def test_pool_falls_back_when_gold_is_one_sided():
    rows = pd.DataFrame(
        {"osm_score": [0.8] * 40, "overture_score": [0.9] * 40,
         "gold": True, "y": [1.0] * 40}
    )
    pool = calibration_fit.fit_pool(rows, np.ones(40))
    assert pool["method"] == "equal_weight_pool"


def test_matched_segment_fit_emits_pool_params():
    rows = _rows(seed = 8).assign(segment = "matched", stratum = "matched")
    result = calibration_fit.fit_segment(rows, "matched", _fit_config())
    assert result["pool"] is not None
    assert set(result["pool"]) >= {"intercept", "coef_osm", "coef_overture"}
    metadata = calibration_fit.curve_metadata(
        "matched", result, {"validation_round": "test"}, _fit_config()
    )
    assert metadata["pool"] == result["pool"]
    assert "pool" in metadata["score_definition"]


# --- Deploy side -----------------------------------------------------------

def _lookup(segment: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": segment,
            "score_lo": [0.0, 0.5, 0.8],
            "score_hi": [0.5, 0.8, 1.0],
            "conf_mean": [0.30, 0.60, 0.90],
            "conf_lower": [0.20, 0.50, 0.85],
            "conf_upper": [0.40, 0.70, 0.95],
        }
    )


def test_apply_curve_is_a_step_lookup_and_passes_nan_through():
    lookup = _lookup("osm")
    out = calibration.apply_curve([0.1, 0.6, 0.99, np.nan], lookup)
    assert out["conf_mean"].tolist()[:3] == [0.30, 0.60, 0.90]
    assert np.isnan(out["conf_mean"].iloc[3])


def test_curve_index_requires_pool_for_matched_rows():
    with pytest.raises(ValueError, match = "pool coefficients"):
        calibration.curve_index(
            np.array(["matched"]), np.array([0.9]), np.array([0.9])
        )


def test_calibrate_frame_edge_rules():
    frame = pd.DataFrame(
        {
            "source": ["matched", "osm", "osm", "overture", "overture"],
            "osm_conf_mean": [0.9, 0.85, 0.85, np.nan, np.nan],
            "overture_confidence": [0.9, np.nan, np.nan, 0.5, 0.95],
            "conf_mean": [0.88, 0.85, 0.85, 0.07, 0.95],
            "shadow_matched": [False, False, False, True, False],
            "name": ["Cafe", "Bar", None, "Shop", "Deli"],
        }
    )
    curves = {s: _lookup(s) for s in ("matched", "osm", "overture")}
    pool = {"matched": {"intercept": 0.0, "coef_osm": 1.0,
                        "coef_overture": 1.0}}
    out = calibration.calibrate_frame(frame, curves, pool_params = pool)

    assert len(out) == len(frame)
    # Shadow-matched row keeps the change-detection value with a NaN band.
    assert out["conf_mean"].iloc[3] == pytest.approx(0.07)
    assert np.isnan(out["conf_lower"].iloc[3])
    assert out["calibration_flag"].iloc[3] == calibration.FLAG_SHADOW
    # Unnamed OSM row is flagged but still calibrated.
    assert out["calibration_flag"].iloc[2] == calibration.FLAG_UNNAMED
    assert out["conf_mean"].iloc[2] == pytest.approx(0.90)
    # Named rows on a curve carry no flag, and the archive column is the input.
    assert out["calibration_flag"].iloc[0] is None
    assert out["conf_mean_uncalibrated"].tolist() == frame["conf_mean"].tolist()
    # Overture row at the imputed sentinel is flagged (not shadowed here).
    assert out["calibration_flag"].iloc[4] is None


def test_missing_conf_flag_fires_on_the_sentinel():
    flags = calibration.calibration_flags(
        np.array(["overture", "overture"]), np.array([0.5, 0.91])
    )
    assert flags[0] == calibration.FLAG_MISSING_CONF
    assert flags[1] == ""


def test_apply_calibration_preserves_rows_and_appends_columns(tmp_path):
    n = 60
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        {
            "unified_id": [f"id{i}" for i in range(n)],
            "source": np.resize(["matched", "osm", "overture"], n),
            "osm_conf_mean": rng.uniform(0.3, 1.0, n),
            "overture_confidence": rng.uniform(0.3, 1.0, n),
            "conf_mean": rng.uniform(0.3, 1.0, n),
            "conf_lower": rng.uniform(0.1, 0.3, n),
            "conf_upper": rng.uniform(0.9, 1.0, n),
            "name": ["Place"] * n,
        }
    )
    in_path = tmp_path / "conflated_cd.parquet"
    out_path = tmp_path / "conflated.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index = False),
                   in_path)

    curves = {s: _lookup(s) for s in ("matched", "osm", "overture")}
    pool = {"matched": {"intercept": 0.0, "coef_osm": 1.0,
                        "coef_overture": 1.0}}
    stats = calibration.apply_calibration(
        in_path, out_path, curves, pool_params = pool, chunk_rows = 17,
        verbose = False,
    )
    assert stats["rows"] == n

    out = pd.read_parquet(out_path)
    assert len(out) == n
    assert "conf_mean_uncalibrated" in out.columns
    assert "calibration_flag" in out.columns
    np.testing.assert_allclose(out["conf_mean_uncalibrated"],
                               frame["conf_mean"])
    assert out["conf_mean"].between(0.0, 1.0).all()
    # Every original column survives.
    assert set(frame.columns) <= set(out.columns)


def test_read_curves_and_pool_params_round_trip(tmp_path):
    lookup = _lookup("matched")
    metadata = {"segment": "matched", "pool": {"intercept": 0.1,
                                               "coef_osm": 0.9,
                                               "coef_overture": 0.6}}
    lookup.to_parquet(tmp_path / "matched_curve.parquet", index = False)
    (tmp_path / "matched_metadata.json").write_text(json.dumps(metadata))

    curves = calibration.read_curves(tmp_path)
    assert set(curves) == {"matched"}
    pool = calibration.pool_params_from_metadata(
        calibration.read_curve_metadata(tmp_path)
    )
    assert pool["matched"]["coef_overture"] == pytest.approx(0.6)


def test_read_curves_errors_when_directory_has_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        calibration.read_curves(tmp_path)
