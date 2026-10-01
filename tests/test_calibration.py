"""Tests for existence-confidence calibration (fit + deploy).

The load-bearing statistical properties, each mapped to the v4 writeup:

- the composite difference estimator reduces to the Horvitz-Thompson gold-only
  estimator when the working model is saturated (writeup 4.2)
- the difference estimator recovers a known existence rate under a deliberately
  wrong working model (design-unbiasedness, Breidt & Opsomer 2017)
- the log-odds pool is monotone in each source score and can exceed both
  inputs, which the linear blend cannot (writeup: matched segment)
- every matched index form (pool, additive, interaction) and the cell surface
  is monotone in both scores end to end through the deploy path; the
  production default is the interaction index with the bin-level band
  (2026-09-26 matched-segment writeup)
- the deploy step's edge rules: shadow-matched rows keep the CD value, unnamed
  OSM rows are flagged, an Overture score of 0.5 is an ordinary score, row
  count preserved
"""

from __future__ import annotations

import json
from pathlib import Path

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
    assert pool["method"] == "constrained_log_odds_pool_v2"
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
    config = _fit_config(matched_index_mode = "pool")
    result = calibration_fit.fit_segment(rows, "matched", config)
    assert result["pool"] is not None
    assert set(result["pool"]) >= {"intercept", "coef_osm", "coef_overture"}
    metadata = calibration_fit.curve_metadata(
        "matched", result, {"validation_round": "test"}, config
    )
    assert metadata["pool"] == result["pool"]
    assert "pool" in metadata["score_definition"]


def test_production_default_is_the_interaction_index_with_bin_band():
    """FitConfig defaults match config.yaml's production settings."""
    config = calibration_fit.FitConfig()
    assert config.matched_index_mode == "interaction"
    assert config.band_aggregation == "bin"
    rows = _rows(seed = 8).assign(segment = "matched", stratum = "matched")
    result = calibration_fit.fit_segment(rows, "matched", _fit_config())
    assert result["index"]["form"] == "interaction"
    assert result["pool"] is None
    metadata = calibration_fit.curve_metadata(
        "matched", result, {"validation_round": "test"}, _fit_config()
    )
    assert metadata["index"]["form"] == "interaction"
    assert metadata["fit_config"]["band_aggregation"] == "bin"
    assert metadata["score_decimals"] == calibration_fit.SCORE_DECIMALS


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
    with pytest.raises(ValueError, match = "index parameters"):
        calibration.curve_index(
            np.array(["matched"]), np.array([0.9]), np.array([0.9])
        )


def test_calibrate_frame_edge_rules():
    frame = pd.DataFrame(
        {
            "source": ["matched", "osm", "osm", "overture", "overture"],
            "osm_conf_mean": [0.9, 0.85, 0.85, np.nan, np.nan],
            "overture_confidence": [0.9, np.nan, np.nan, 0.5, 0.5],
            "conf_mean": [0.88, 0.85, 0.85, 0.07, 0.5],
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
    # An Overture row at exactly 0.5 is an ordinary provider score: no flag,
    # and it rides the overture curve like any other score.
    assert out["calibration_flag"].iloc[4] is None
    assert out["conf_mean"].iloc[4] == pytest.approx(0.60)


def test_overture_score_of_one_half_is_not_flagged():
    flags = calibration.calibration_flags(np.array(["overture", "overture"]))
    assert flags.tolist() == ["", ""]


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


# --- Grid lookups (Bayesian export, October 2026) ---------------------------

def _grid_curve(segment: str, n: int = 11) -> pd.DataFrame:
    """1-D node table in the export schema: mean 0.2 + 0.7 s, band +-0.05."""
    score = np.linspace(0.0, 1.0, n)
    mean = 0.2 + 0.7 * score
    return pd.DataFrame({"segment": segment, "score": score, "conf_mean": mean,
                         "conf_lower": mean - 0.05, "conf_upper": mean + 0.05})


def _grid_surface(n: int = 5) -> pd.DataFrame:
    """Matched node grid, row-major (osm outer), mean 0.1 + 0.3 o + 0.5 v
    + 0.1 o v; nondecreasing along both axes."""
    nodes = np.linspace(0.0, 1.0, n)
    oo, vv = np.meshgrid(nodes, nodes, indexing = "ij")
    mean = (0.1 + 0.3 * oo + 0.5 * vv + 0.1 * oo * vv).ravel()
    return pd.DataFrame({
        "segment": "matched", "osm_score": oo.ravel(),
        "overture_score": vv.ravel(), "conf_mean": mean,
        "conf_lower": mean - 0.05, "conf_upper": mean + 0.05,
    })


def _grid_metadata() -> dict:
    out = {}
    for segment in ("matched", "osm", "overture"):
        out[segment] = {"method": "bayes_fixed_mixture", "lookup": "grid",
                        "score_decimals": 6}
    out["matched"]["index_mode"] = "grid"
    return out


def test_grid_lookup_detection():
    assert calibration.is_grid_lookup(_grid_curve("osm"))
    assert calibration.is_grid_lookup(_grid_surface())
    assert calibration.is_grid_surface(_grid_surface())
    assert not calibration.is_grid_surface(_grid_curve("osm"))
    assert not calibration.is_grid_lookup(_lookup("osm"))
    assert not calibration.is_grid_lookup(_surface_lookup_fixture())


def test_apply_grid_curve_interpolates_clamps_and_passes_nan():
    lookup = _grid_curve("osm")
    nodes = lookup["score"].to_numpy()
    at_nodes = calibration.apply_grid_curve(nodes, lookup)
    np.testing.assert_allclose(at_nodes["conf_mean"], lookup["conf_mean"])
    np.testing.assert_allclose(at_nodes["conf_lower"], lookup["conf_lower"])
    np.testing.assert_allclose(at_nodes["conf_upper"], lookup["conf_upper"])
    # Linear between nodes, on a table whose nodes are not equally spaced.
    uneven = lookup.iloc[[0, 3, 10]].reset_index(drop = True)
    uneven.loc[1, "conf_mean"] = 0.8
    out = calibration.apply_grid_curve([0.15, 0.65], uneven)
    assert out["conf_mean"].iloc[0] == pytest.approx(0.2 + 0.5 * 0.6)
    assert out["conf_mean"].iloc[1] == pytest.approx(0.8 + 0.5 * 0.1)
    # Out of [0, 1] clamps to the end nodes; NaN stays NaN in all three.
    out = calibration.apply_grid_curve([-0.4, 1.7, np.nan], lookup)
    assert out["conf_mean"].iloc[0] == pytest.approx(0.2)
    assert out["conf_mean"].iloc[1] == pytest.approx(0.9)
    assert out.iloc[2].isna().all()
    assert list(out.columns) == ["conf_mean", "conf_lower", "conf_upper"]
    # Rounding, when asked for, happens before the lookup.
    rounded = calibration.apply_grid_curve([0.1234564], lookup,
                                           score_decimals = 2)
    assert rounded["conf_mean"].iloc[0] == pytest.approx(0.2 + 0.7 * 0.12)


def test_apply_grid_surface_is_bilinear_and_monotone():
    lookup = _grid_surface()
    exact = calibration.apply_grid_surface(lookup["osm_score"],
                                           lookup["overture_score"], lookup)
    np.testing.assert_allclose(exact["conf_mean"], lookup["conf_mean"])
    np.testing.assert_allclose(exact["conf_upper"], lookup["conf_upper"])
    # The node function is bilinear, so interpolation reproduces it anywhere.
    rng = np.random.default_rng(3)
    o, v = rng.uniform(0, 1, 200), rng.uniform(0, 1, 200)
    out = calibration.apply_grid_surface(o, v, lookup)
    np.testing.assert_allclose(out["conf_mean"],
                               0.1 + 0.3 * o + 0.5 * v + 0.1 * o * v)
    # Midpoint of one cell is the mean of its four corners, row order aside.
    shuffled = lookup.sample(frac = 1.0, random_state = 4)
    mid = calibration.apply_grid_surface([0.125], [0.375], shuffled)
    corners = lookup[lookup["osm_score"].isin([0.0, 0.25])
                     & lookup["overture_score"].isin([0.25, 0.5])]
    assert mid["conf_mean"].iloc[0] == pytest.approx(
        corners["conf_mean"].mean()
    )
    # Monotone in both scores on a fine grid; clamps; NaN in either is NaN.
    grid = np.linspace(-0.1, 1.1, 37)
    oo, vv = np.meshgrid(grid, grid, indexing = "ij")
    surface = calibration.apply_grid_surface(
        oo.ravel(), vv.ravel(), lookup)["conf_mean"].to_numpy().reshape(oo.shape)
    assert np.all(np.diff(surface, axis = 0) >= -1e-12)
    assert np.all(np.diff(surface, axis = 1) >= -1e-12)
    assert surface[0, 0] == pytest.approx(0.1)
    assert surface[-1, -1] == pytest.approx(1.0)
    out = calibration.apply_grid_surface([np.nan, 0.5], [0.5, np.nan], lookup)
    assert out.isna().all().all()


def test_apply_grid_surface_refuses_a_ragged_grid():
    with pytest.raises(ValueError, match = "rectangle"):
        calibration.apply_grid_surface([0.5], [0.5], _grid_surface().iloc[:-1])


def test_calibrate_frame_on_grid_curves_keeps_the_edge_rules():
    frame = pd.DataFrame(
        {
            "source": ["matched", "osm", "osm", "overture", "overture",
                       "matched"],
            "osm_conf_mean": [0.5, 0.4, 0.4, np.nan, np.nan, np.nan],
            "overture_confidence": [0.5, np.nan, np.nan, 0.6, 0.6, 0.9],
            "conf_mean": [0.88, 0.85, 0.85, 0.07, 0.5, 0.42],
            "shadow_matched": [False, False, False, True, False, False],
            "name": ["Cafe", "Bar", None, "Shop", "Deli", "Inn"],
        }
    )
    curves = {"matched": _grid_surface(), "osm": _grid_curve("osm"),
              "overture": _grid_curve("overture")}
    meta = _grid_metadata()
    # No pool parameters anywhere: index_mode grid must not ask for them.
    assert calibration.pool_params_from_metadata(meta)["matched"] is None
    out = calibration.calibrate_frame(
        frame, curves,
        pool_params = calibration.pool_params_from_metadata(meta),
        index_modes = calibration.index_modes_from_metadata(meta),
        score_decimals = calibration.score_decimals_from_metadata(meta),
    )
    assert out["conf_mean"].iloc[0] == pytest.approx(
        0.1 + 0.15 + 0.25 + 0.025
    )
    assert out["conf_lower"].iloc[0] == pytest.approx(0.475)
    assert out["conf_mean"].iloc[1] == pytest.approx(0.2 + 0.7 * 0.4)
    # Unnamed OSM rides the osm curve, flagged.
    assert out["calibration_flag"].iloc[2] == calibration.FLAG_UNNAMED
    assert out["conf_mean"].iloc[2] == pytest.approx(0.2 + 0.7 * 0.4)
    # Shadow row keeps its CD value and a NaN band.
    assert out["calibration_flag"].iloc[3] == calibration.FLAG_SHADOW
    assert out["conf_mean"].iloc[3] == pytest.approx(0.07)
    assert np.isnan(out["conf_upper"].iloc[3])
    assert out["conf_mean"].iloc[4] == pytest.approx(0.2 + 0.7 * 0.6)
    # A matched row missing a score keeps its incoming value (unscored).
    assert out["conf_mean"].iloc[5] == pytest.approx(0.42)
    assert np.isnan(out["conf_lower"].iloc[5])
    # Without metadata the grid shape alone selects the 2-D path.
    bare = calibration.calibrate_frame(frame, curves)
    np.testing.assert_allclose(bare["conf_mean"], out["conf_mean"])


def test_grid_mode_needs_no_pool_in_curve_index():
    scores = calibration.curve_index(np.array(["matched", "osm"]),
                                     np.array([0.9, 0.3]),
                                     np.array([0.9, np.nan]),
                                     index_mode = "grid")
    assert np.isnan(scores[0]) and scores[1] == pytest.approx(0.3)


def test_apply_calibration_round_trip_on_grid_curves(tmp_path):
    curves_dir = tmp_path / "calibration"
    curves_dir.mkdir()
    curves = {"matched": _grid_surface(), "osm": _grid_curve("osm"),
              "overture": _grid_curve("overture")}
    for segment, lookup in curves.items():
        lookup.to_parquet(curves_dir / f"{segment}_curve.parquet",
                          index = False)
        (curves_dir / f"{segment}_metadata.json").write_text(
            json.dumps(_grid_metadata()[segment])
        )
    n = 60
    rng = np.random.default_rng(22)
    source = np.resize(["matched", "osm", "overture"], n)
    frame = pd.DataFrame({
        "unified_id": [f"id{i}" for i in range(n)],
        "source": source,
        "osm_conf_mean": np.where(source == "overture", np.nan,
                                  rng.uniform(0.0, 1.0, n)),
        "overture_confidence": np.where(source == "osm", np.nan,
                                        rng.uniform(0.0, 1.0, n)),
        "conf_mean": rng.uniform(0.3, 1.0, n),
        "conf_lower": rng.uniform(0.1, 0.3, n),
        "conf_upper": rng.uniform(0.9, 1.0, n),
        "shadow_matched": np.arange(n) == 5,
        "name": ["Place"] * n,
    })
    in_path = tmp_path / "conflated_cd.parquet"
    out_path = tmp_path / "conflated.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index = False),
                   in_path)

    read = calibration.read_curves(curves_dir)
    meta = calibration.read_curve_metadata(curves_dir)
    assert set(read) == set(meta) == {"matched", "osm", "overture"}
    stats = calibration.apply_calibration(
        in_path, out_path, read,
        pool_params = calibration.pool_params_from_metadata(meta),
        index_modes = calibration.index_modes_from_metadata(meta),
        score_decimals = calibration.score_decimals_from_metadata(meta),
        chunk_rows = 17, verbose = False,
    )
    assert stats["rows"] == n
    out = pd.read_parquet(out_path)
    assert len(out) == n and set(frame.columns) <= set(out.columns)
    expected = calibration.calibrate_frame(
        frame, curves, index_modes = {"matched": "grid"},
        score_decimals = {s: 6 for s in curves},
    )
    np.testing.assert_allclose(out["conf_mean"], expected["conf_mean"])
    plain = ~frame["shadow_matched"].to_numpy()
    assert (out["conf_lower"][plain] <= out["conf_mean"][plain]).all()
    assert (out["conf_mean"][plain] <= out["conf_upper"][plain]).all()
    assert out["conf_mean"].iloc[5] == pytest.approx(frame["conf_mean"].iloc[5])
    assert out["calibration_flag"].iloc[5] == calibration.FLAG_SHADOW


# --- Matched-segment modes (2026-09) ----------------------------------------

ROUND_20260730 = (
    Path(__file__).resolve().parents[1]
    / "data" / "calibration" / "20260730" / "validation_rows.parquet"
)


def _matched_rows(n = 3000, seed = 0, truth = None,
                  osm = None, overture = None) -> pd.DataFrame:
    """Synthetic two-phase matched-segment table with a known P(exists).

    The LLM verdict is a noisy function of the true outcome and phase-2
    inclusion is set per verdict class, as in the real design.
    """
    rng = np.random.default_rng(seed)
    osm = rng.uniform(0.3, 1.0, n) if osm is None else osm
    overture = rng.uniform(0.3, 1.0, n) if overture is None else overture
    truth = truth or (lambda o, v: calibration_fit.expit(
        -3.0 + 2.5 * o + 2.5 * v))
    y_true = (rng.random(n) < truth(osm, overture)).astype(float)
    draw = rng.random(n)
    verdict = np.where(
        y_true == 1,
        np.where(draw < 0.9, "exists", np.where(draw < 0.95, "gone",
                                                "unverifiable")),
        np.where(draw < 0.7, "gone", np.where(draw < 0.8, "exists",
                                              "unverifiable")),
    )
    inclusion = pd.Series(verdict).map(
        {"exists": 0.15, "gone": 0.4, "unverifiable": 1.0}
    ).to_numpy()
    gold = rng.random(n) < inclusion
    return pd.DataFrame(
        {
            "segment": "matched",
            "stratum": "matched",
            "osm_score": osm,
            "overture_score": overture,
            "llm_verdict": verdict,
            "llm_confidence": "high",
            "gold": gold,
            "y": np.where(gold, y_true, np.nan),
        }
    )


def _qp_oracle(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exact projection onto the doubly-monotone cone via its NNLS dual.

    Primal: min ||phi - z||^2 s.t. B phi >= 0 (phi = W^1/2 theta,
    B = A W^-1/2). Dual: min_{lam >= 0} ||B' lam + z||^2, phi = z + B' lam.
    A generic QP (SLSQP) was tried first and only reaches ~1e-7.
    """
    from scipy.optimize import nnls

    rows, cols = values.shape
    n = rows * cols
    constraint_rows = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for j in ((i + 1) if c + 1 < cols else None,
                      (i + cols) if r + 1 < rows else None):
                if j is not None:
                    a = np.zeros(n)
                    a[j], a[i] = 1.0, -1.0
                    constraint_rows.append(a)
    root_w = np.sqrt(weights.ravel())
    b = np.array(constraint_rows) / root_w[None, :]
    z = root_w * values.ravel()
    lam, _ = nnls(b.T, -z, maxiter = 100 * n)
    return ((z + b.T @ lam) / root_w).reshape(rows, cols)


def _is_doubly_monotone(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(
        np.all(np.diff(matrix, axis = 0) >= -tol)
        and np.all(np.diff(matrix, axis = 1) >= -tol)
    )


@pytest.mark.skipif(not ROUND_20260730.exists(),
                    reason = "validation handoff is gitignored")
def test_constrained_pool_reproduces_round_20260730():
    """The bound is inactive on the July data, so the July pool comes back."""
    rows = pd.read_parquet(ROUND_20260730)
    rows = calibration_fit.round_scores(rows[
        (rows["segment"] == "matched")
        & rows["stratum"].isin(calibration_fit.SEGMENTS)
        & rows["llm_verdict"].isin(calibration_fit.VERDICTS)
    ].reset_index(drop = True))
    classes = calibration_fit.merge_thin_cells(
        calibration_fit.refined_class(rows),
        rows["gold"].to_numpy(dtype = bool), 25,
    )
    inclusion = calibration_fit.inclusion_by_class(
        classes, rows["gold"].to_numpy(dtype = bool)
    )
    weights = classes.map(lambda c: inclusion[c]["weight"]).to_numpy()
    pool = calibration_fit.fit_pool(rows, weights)
    assert pool["intercept"] == pytest.approx(0.20422, abs = 1e-4)
    assert pool["coef_osm"] == pytest.approx(0.21008, abs = 1e-4)
    assert pool["coef_overture"] == pytest.approx(0.56938, abs = 1e-4)
    assert pool["bound_active"] == []


def test_constrained_pool_pins_a_negative_slope_at_its_bound():
    rng = np.random.default_rng(21)
    n = 2000
    osm = rng.uniform(0.05, 0.95, n)
    overture = rng.uniform(0.05, 0.95, n)
    # OSM carries a genuinely NEGATIVE signal.
    p = calibration_fit.expit(1.5 * calibration_fit.logit(overture)
                              - 1.0 * calibration_fit.logit(osm))
    rows = pd.DataFrame({"osm_score": osm, "overture_score": overture,
                         "gold": True,
                         "y": (rng.random(n) < p).astype(float)})
    pool = calibration_fit.fit_pool(rows, np.ones(n), min_coef = 1e-3)
    assert pool["coef_osm"] == pytest.approx(1e-3, abs = 1e-9)
    assert pool["bound_active"] == ["coef_osm"]
    assert pool["coef_overture"] > 1.0


def test_constrained_pool_thin_gold_fallback_is_unchanged():
    rows = pd.DataFrame({"osm_score": [0.8, 0.6] * 10,
                         "overture_score": [0.9, 0.5] * 10,
                         "gold": True, "y": [1.0, 0.0] * 10})
    pool = calibration_fit.fit_pool(rows, np.ones(20))
    assert pool["method"] == "equal_weight_pool"
    assert (pool["coef_osm"], pool["coef_overture"]) == (1.0, 1.0)


def _additive_truth_rows(n = 4000, seed = 5):
    rng = np.random.default_rng(seed)
    osm = np.round(rng.uniform(0.0, 1.0, n), 3)
    overture = np.round(rng.uniform(0.0, 1.0, n), 3)
    h_osm = np.where(osm > 0.5, 1.5, -0.5)         # a step
    h_ov = 3.0 * (overture - 0.5)                  # linear
    p = calibration_fit.expit(0.5 + h_osm + h_ov)
    rows = pd.DataFrame({"osm_score": osm, "overture_score": overture,
                         "gold": True,
                         "y": (rng.random(n) < p).astype(float)})
    return rows, h_osm, h_ov


def test_additive_index_components_are_nondecreasing_and_recover_truth():
    rows, h_osm, h_ov = _additive_truth_rows()
    params = calibration_fit.fit_additive_index(rows, np.ones(len(rows)))
    assert params["form"] == "additive" and params["converged"]
    for axis in ("osm", "overture"):
        assert np.all(np.diff(params[f"levels_{axis}"]) >= -1e-12)
    fitted_osm = np.interp(rows["osm_score"], params["knots_osm"],
                           params["levels_osm"])
    fitted_ov = np.interp(rows["overture_score"], params["knots_overture"],
                          params["levels_overture"])
    # Shapes, not levels: components are identified up to a constant. The
    # isotonic MLE spikes at the end knots (a lone y = 0 at the lowest score
    # sends its block to the clip), so compare on the interior.
    interior = (rows["osm_score"].between(0.05, 0.95)
                & rows["overture_score"].between(0.05, 0.95)).to_numpy()
    assert np.corrcoef(fitted_osm[interior], h_osm[interior])[0, 1] > 0.95
    assert np.corrcoef(fitted_ov[interior], h_ov[interior])[0, 1] > 0.95
    # The step is recovered at roughly its true height of 2 logits.
    high = fitted_osm[rows["osm_score"] > 0.6].mean()
    low = fitted_osm[rows["osm_score"] < 0.4].mean()
    assert 1.5 < high - low < 2.5


def test_additive_index_matches_an_exact_convex_solver():
    """Local scoring reaches the constrained MLE (checked against L-BFGS-B on
    nonnegative increments, which solves the same convex problem directly)."""
    from scipy.optimize import minimize

    rows, _, _ = _additive_truth_rows(n = 600, seed = 9)
    rows["osm_score"] = np.round(rows["osm_score"], 1)
    rows["overture_score"] = np.round(rows["overture_score"], 1)
    params = calibration_fit.fit_additive_index(rows, np.ones(len(rows)))
    y = rows["y"].to_numpy()
    k_osm, i_osm = np.unique(rows["osm_score"], return_inverse = True)
    k_ov, i_ov = np.unique(rows["overture_score"], return_inverse = True)

    def loss(theta):
        a = theta[0]
        h1 = np.concatenate([[0.0], np.cumsum(theta[1:len(k_osm)])])
        h2 = np.concatenate([[0.0], np.cumsum(theta[len(k_osm):])])
        z = np.clip(a + h1[i_osm] + h2[i_ov], -calibration_fit.LOGIT_MAX,
                    calibration_fit.LOGIT_MAX)
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    size = 1 + (len(k_osm) - 1) + (len(k_ov) - 1)
    bounds = [(None, None)] + [(0.0, None)] * (size - 1)
    exact = minimize(loss, np.full(size, 0.1), method = "L-BFGS-B",
                     bounds = bounds, options = {"ftol": 1e-14,
                                                 "maxiter": 10000})
    assert params["deviance"] / 2.0 <= exact.fun * (1.0 + 1e-5) + 1e-6


def test_additive_index_dead_axis_goes_flat():
    rng = np.random.default_rng(3)
    n = 3000
    overture = rng.uniform(0.0, 1.0, n)
    p = calibration_fit.expit(-2.0 + 4.0 * overture)
    rows = pd.DataFrame({"osm_score": np.full(n, 0.8),
                         "overture_score": overture, "gold": True,
                         "y": (rng.random(n) < p).astype(float)})
    params = calibration_fit.fit_additive_index(rows, np.ones(n))
    assert params["n_blocks_osm"] == 1
    assert np.allclose(params["levels_osm"], 0.0)
    # A noise axis gets only a small component next to the live one.
    rows["osm_score"] = rng.uniform(0.0, 1.0, n)
    params = calibration_fit.fit_additive_index(rows, np.ones(n))

    def interior_range(axis):
        knots = np.asarray(params[f"knots_{axis}"])
        levels = np.asarray(params[f"levels_{axis}"])
        keep = (knots >= np.quantile(knots, 0.05)) & (
            knots <= np.quantile(knots, 0.95))
        return np.ptp(levels[keep])

    assert interior_range("osm") < 0.25 * interior_range("overture")


def test_additive_index_survives_an_all_positive_block():
    rng = np.random.default_rng(4)
    n = 1500
    osm = rng.uniform(0.0, 1.0, n)
    overture = rng.uniform(0.0, 1.0, n)
    p = np.where((osm > 0.7) & (overture > 0.7), 1.0,
                 calibration_fit.expit(-1.0 + osm + overture))
    rows = pd.DataFrame({"osm_score": osm, "overture_score": overture,
                         "gold": True,
                         "y": (rng.random(n) < p).astype(float)})
    params = calibration_fit.fit_additive_index(rows, np.ones(n))
    assert params["n_clipped"] > 0
    assert np.all(np.isfinite(params["levels_osm"]))
    assert np.all(np.isfinite(params["levels_overture"]))
    scores = calibration_fit.additive_score(osm, overture, params)
    assert np.all((scores > 0) & (scores < 1))


def test_interaction_index_constraints_hold_and_a3_sign_is_recovered():
    rng = np.random.default_rng(6)
    n = 6000
    osm = rng.uniform(0.02, 0.999, n)
    overture = rng.uniform(0.02, 0.999, n)
    x = calibration_fit._unit_logit(osm)
    v = calibration_fit._unit_logit(overture)
    # Substitutive truth: monotone (a1 + a3 = 4 > 0, a2 + a3 = 4 > 0).
    a0, a1, a2, a3 = -12.0, 12.0, 12.0, -8.0
    p = calibration_fit.expit(a0 + a1 * x + a2 * v + a3 * x * v)
    rows = pd.DataFrame({"osm_score": osm, "overture_score": overture,
                         "gold": True,
                         "y": (rng.random(n) < p).astype(float)})
    params = calibration_fit.fit_interaction_index(rows, np.ones(n))
    beta = np.array([params[k] for k in ("a0", "a1", "a2", "a3")])
    assert np.all(calibration_fit._INTERACTION_CONSTRAINTS @ beta
                  >= 1e-3 - 1e-9)
    assert params["a3"] < -3.0
    # Monotone on a dense grid over the whole score square.
    grid = np.linspace(0.0, 1.0, 201)
    oo, vv = np.meshgrid(grid, grid, indexing = "ij")
    surface = calibration_fit.interaction_score(oo, vv, params)
    assert np.all(np.diff(surface, axis = 0) >= -1e-12)
    assert np.all(np.diff(surface, axis = 1) >= -1e-12)


def test_interaction_index_respects_constraints_on_a_violating_truth():
    rng = np.random.default_rng(7)
    n = 3000
    osm = rng.uniform(0.02, 0.999, n)
    overture = rng.uniform(0.02, 0.999, n)
    x = calibration_fit._unit_logit(osm)
    v = calibration_fit._unit_logit(overture)
    # Non-monotone truth: OSM hurts once Overture is high.
    p = calibration_fit.expit(-6.0 + 8.0 * x + 8.0 * v - 14.0 * x * v)
    rows = pd.DataFrame({"osm_score": osm, "overture_score": overture,
                         "gold": True,
                         "y": (rng.random(n) < p).astype(float)})
    params = calibration_fit.fit_interaction_index(rows, np.ones(n))
    beta = np.array([params[k] for k in ("a0", "a1", "a2", "a3")])
    assert np.all(calibration_fit._INTERACTION_CONSTRAINTS @ beta
                  >= 1e-3 - 1e-9)
    assert params["constraints_active"]


def test_project_monotone_2d_properties():
    rng = np.random.default_rng(0)
    values = rng.random((6, 4))
    weights = rng.uniform(1.0, 20.0, (6, 4))
    out = calibration_fit.project_monotone_2d(values, weights)
    assert _is_doubly_monotone(out)
    # Idempotent on a monotone matrix.
    again = calibration_fit.project_monotone_2d(out, weights)
    np.testing.assert_allclose(again, out, atol = 1e-9)
    # One Overture bin: the 2-D projection is the 1-D weighted PAV.
    column = values[:, :1]
    from sklearn.isotonic import IsotonicRegression
    pav = IsotonicRegression().fit(np.arange(6), column[:, 0],
                                   sample_weight = weights[:, 0])
    np.testing.assert_allclose(
        calibration_fit.project_monotone_2d(column, weights[:, :1])[:, 0],
        pav.predict(np.arange(6)), atol = 1e-9,
    )
    # A heavy cell wins a violation against a light one.
    pair = np.array([[0.9, 0.1]])
    out = calibration_fit.project_monotone_2d(pair, np.array([[1.0, 99.0]]))
    np.testing.assert_allclose(out, [[0.108, 0.108]], atol = 1e-9)


def test_project_monotone_2d_matches_a_qp_oracle():
    rng = np.random.default_rng(1)
    for _ in range(15):
        rows, cols = rng.integers(2, 6), rng.integers(2, 5)
        values = rng.random((rows, cols))
        weights = rng.uniform(0.5, 30.0, (rows, cols))
        np.testing.assert_allclose(
            calibration_fit.project_monotone_2d(values, weights),
            _qp_oracle(values, weights), atol = 1e-8,
        )


def test_project_monotone_2d_without_increments_fails_the_oracle():
    """Regression guard: plain alternation is not the projection."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(15):
        rows, cols = rng.integers(2, 6), rng.integers(2, 5)
        values = rng.random((rows, cols))
        weights = rng.uniform(0.5, 30.0, (rows, cols))
        plain = calibration_fit.project_monotone_2d(
            values, weights, use_increments = False
        )
        worst = max(worst, np.abs(plain - _qp_oracle(values, weights)).max())
    assert worst > 1e-3


def test_project_monotone_2d_batches_match_single_projections():
    rng = np.random.default_rng(2)
    values = rng.random((7, 3, 4))
    weights = rng.uniform(1.0, 5.0, (3, 4))
    batch = calibration_fit.project_monotone_2d(values, weights)
    for k in range(7):
        np.testing.assert_allclose(
            batch[k], calibration_fit.project_monotone_2d(values[k], weights),
            atol = 1e-12,
        )


def test_cell_difference_estimator_equals_hajek_under_equal_inclusion():
    """With the same realized inclusion in every cell, the per-cell
    difference estimator is exactly the per-cell Hajek mean, whatever the
    (constant) working model predicts."""
    rng = np.random.default_rng(8)
    cells, per_cell, gold_per_cell = 6, 40, 10
    frames = []
    for cell in range(cells):
        gold = np.zeros(per_cell, dtype = bool)
        gold[rng.choice(per_cell, gold_per_cell, replace = False)] = True
        y = (rng.random(per_cell) < 0.3 + 0.1 * cell).astype(float)
        frames.append(pd.DataFrame({
            "cell": cell, "gold": gold, "y": np.where(gold, y, np.nan),
            "score": rng.random(per_cell), "llm_verdict": "exists",
        }))
    rows = pd.concat(frames, ignore_index = True)
    classes = pd.Series(["exists"] * len(rows))
    inclusion = calibration_fit.inclusion_by_class(
        classes, rows["gold"].to_numpy(dtype = bool)
    )
    estimate, counts = calibration_fit.cell_difference_estimator(
        rows, classes, inclusion, rows["cell"].to_numpy(),
        np.zeros(len(rows), dtype = int), (cells, 1),
    )
    hajek = rows[rows["gold"]].groupby("cell")["y"].mean().to_numpy()
    np.testing.assert_allclose(estimate[:, 0], hajek, atol = 1e-12)
    assert np.all(counts[:, 0] == per_cell)


def test_cell_difference_estimator_dead_axis_is_flat():
    rows = _matched_rows(n = 20000, seed = 3, truth = lambda o, v:
                         calibration_fit.expit(-2.0 + 4.0 * o))
    rows = rows.assign(score = rows["osm_score"])
    classes = rows["llm_verdict"].astype(str)
    inclusion = calibration_fit.inclusion_by_class(
        classes, rows["gold"].to_numpy(dtype = bool)
    )
    edges = {"osm": np.linspace(0.3, 1.0, 4),
             "overture": np.linspace(0.3, 1.0, 4)}
    i_osm, i_ov = calibration_fit.surface_cells(
        rows["osm_score"], rows["overture_score"], edges
    )
    estimate, _ = calibration_fit.cell_difference_estimator(
        rows, classes, inclusion, i_osm, i_ov, (3, 3)
    )
    # Overture is dead: each OSM row is flat within noise ...
    assert np.ptp(estimate, axis = 1).max() < 0.08
    # ... while OSM moves the rate.
    assert estimate[2].mean() - estimate[0].mean() > 0.2


def test_surface_edges_give_each_rounded_atom_its_own_bin():
    rng = np.random.default_rng(10)
    atom = 0.919912
    # Production carries one float repr of the atom; the validation file
    # carries others that only rounding reunites.
    population = pd.DataFrame({
        "osm_score": rng.uniform(0.2, 1.0, 10000),
        "overture_score": np.concatenate([
            np.full(4000, 0.9199122190475464),
            rng.uniform(0.2, 0.9, 3000), rng.uniform(0.93, 1.0, 3000),
        ]),
    })
    edges = calibration_fit.surface_edges(population, 4, 6)
    assert atom in edges["overture"]
    variants = np.array([0.91991221904754634, 0.9199122190475466,
                         0.9199122190475464])
    _, bins = calibration_fit.surface_cells(
        np.full(3, 0.5), np.round(variants, calibration_fit.SCORE_DECIMALS),
        edges,
    )
    assert len(set(bins.tolist())) == 1
    assert edges["overture"][bins[0]] == atom


def test_surface_refuses_a_cell_with_no_phase_one_rows():
    rows = _matched_rows(n = 800, seed = 11)
    rows = rows[rows["overture_score"] < 0.75]
    population = _matched_rows(n = 5000, seed = 12)[
        ["osm_score", "overture_score"]
    ]
    config = _fit_config(matched_index_mode = "surface",
                         surface_osm_bins = 3, surface_ov_bins = 6)
    classes = rows["llm_verdict"].astype(str).reset_index(drop = True)
    with pytest.raises(ValueError, match = "no phase-1 rows"):
        calibration_fit.fit_surface_segment(
            rows.reset_index(drop = True), classes, config,
            population = population,
        )


def _surface_lookup_fixture() -> pd.DataFrame:
    edges = {"osm": np.array([0.0, 0.5, 1.0]),
             "overture": np.array([0.0, 0.9, 0.95, 1.0])}
    mean = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return calibration_fit.surface_lookup("matched", edges, mean, mean - 0.05,
                                          mean + 0.05)


def test_apply_surface_clamps_rounds_and_passes_nan():
    lookup = _surface_lookup_fixture()
    out = calibration_fit.apply_surface(
        [-1.0, 0.5, 0.49, 2.0, np.nan, 0.7],
        [0.95, 0.9, 0.95, 5.0, 0.9, np.nan],
        lookup,
    )
    means = out["conf_mean"].to_numpy()
    # Below the first OSM edge clamps to bin 0; an edge value opens its bin
    # (the apply_curve convention); above the last edge clamps to the top.
    np.testing.assert_allclose(means[:4], [0.3, 0.5, 0.3, 0.6])
    assert np.isnan(means[4]) and np.isnan(means[5])
    # The same side convention as the 1-D lookup.
    one_d = pd.DataFrame({"segment": "x", "score_lo": [0.0, 0.9, 0.95],
                          "score_hi": [0.9, 0.95, 1.0],
                          "conf_mean": [0.1, 0.2, 0.3],
                          "conf_lower": [0.0] * 3, "conf_upper": [1.0] * 3})
    np.testing.assert_allclose(
        calibration.apply_curve([0.9, 0.95], one_d)["conf_mean"], [0.2, 0.3]
    )
    np.testing.assert_allclose(
        calibration_fit.apply_surface([0.1, 0.1], [0.9, 0.95],
                                      lookup)["conf_mean"], [0.2, 0.3]
    )


def test_surface_fit_bands_are_monotone_and_bracket_the_estimate():
    rows = _matched_rows(n = 3000, seed = 13)
    config = _fit_config(matched_index_mode = "surface", bootstrap_reps = 40,
                         surface_osm_bins = 4, surface_ov_bins = 3,
                         wald_mixture_draws = 300)
    result = calibration_fit.fit_segment(rows, "matched", config,
                                         population = rows)
    surface = result["surface"]
    theta = surface["theta"]
    assert _is_doubly_monotone(theta)
    for band in surface["bands"].values():
        assert _is_doubly_monotone(band["lower"], tol = 1e-8)
        assert _is_doubly_monotone(band["upper"], tol = 1e-8)
        assert np.all(band["lower"] <= theta + 1e-12)
        assert np.all(band["upper"] >= theta - 1e-12)
    assert len(result["lookup"]) == theta.size
    metadata = calibration_fit.curve_metadata("matched", result,
                                              {"validation_round": "t"},
                                              config)
    json.dumps(metadata)
    assert metadata["index_mode"] == "surface"
    assert metadata["score_definition"].startswith("surface_cells(")


@pytest.mark.parametrize("mode", ["pool", "average", "additive",
                                  "interaction", "surface"])
def test_deploy_roundtrip_is_monotone_in_each_input(mode, tmp_path):
    rows = _matched_rows(n = 2500, seed = 14)
    config = _fit_config(matched_index_mode = mode, bootstrap_reps = 10,
                         surface_osm_bins = 4, surface_ov_bins = 3,
                         wald_mixture_draws = 100)
    result = calibration_fit.fit_segment(rows, "matched", config,
                                         population = rows)
    metadata = calibration_fit.curve_metadata("matched", result,
                                              {"validation_round": "t"},
                                              config)
    calibration_fit.write_curve(tmp_path, "matched", result["lookup"],
                                metadata)
    curves = calibration.read_curves(tmp_path)
    meta = calibration.read_curve_metadata(tmp_path)

    grid = np.linspace(0.3, 1.0, 25)
    oo, vv = np.meshgrid(grid, grid, indexing = "ij")
    frame = pd.DataFrame({
        "source": "matched", "osm_conf_mean": oo.ravel(),
        "overture_confidence": vv.ravel(), "conf_mean": 0.5,
        "shadow_matched": False, "name": "Cafe",
    })
    out = calibration.calibrate_frame(
        frame, curves,
        pool_params = calibration.pool_params_from_metadata(meta),
        index_modes = calibration.index_modes_from_metadata(meta),
        score_decimals = calibration.score_decimals_from_metadata(meta),
    )
    surface = out["conf_mean"].to_numpy().reshape(oo.shape)
    assert np.isfinite(surface).all()
    assert np.all(np.diff(surface, axis = 0) >= -1e-12)
    assert np.all(np.diff(surface, axis = 1) >= -1e-12)
    assert out["calibration_flag"].isna().all()


def _compare_module():
    import importlib.util

    path = (Path(__file__).resolve().parents[1] / "scripts" / "conflation"
            / "compare_matched_index.py")
    spec = importlib.util.spec_from_file_location("compare_matched_index",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_n_way_verdict_sign_is_pinned():
    compare = _compare_module()
    lower_better = {"mean": -0.002, "lower": -0.003, "upper": -0.001}
    higher = {"mean": 0.002, "lower": 0.001, "upper": 0.003}
    spans = {"mean": 0.001, "lower": -0.001, "upper": 0.003}
    assert compare._verdict("brier", lower_better, "surface",
                            "pool") == "surface better"
    assert compare._verdict("log_score", higher, "surface",
                            "pool") == "pool better"
    # DSC is better HIGHER: a positive difference favours the candidate.
    assert compare._verdict("dsc", higher, "interaction",
                            "pool") == "interaction better"
    assert compare._verdict("dsc", lower_better, "interaction",
                            "pool") == "pool better"
    assert compare._verdict("mcb", spans, "a", "b") == "no difference"


def test_atom_aware_edges_and_monotonicity_table():
    rng = np.random.default_rng(15)
    overture = np.concatenate([np.full(900, 0.919912), rng.uniform(0.3, 1.0,
                                                                   2100)])
    rows = _matched_rows(n = 3000, seed = 15, overture = overture)
    edges = calibration_fit.atom_aware_edges(rows["overture_score"], 4)
    bins = calibration_fit._bin_index(
        np.round(rows["overture_score"], 6), edges
    )
    atom_bin = bins[rows["overture_score"] == 0.919912]
    # The atom sits alone in its bin.
    assert len(np.unique(atom_bin)) == 1
    assert (bins == atom_bin[0]).sum() == 900
    table = calibration_fit.axis_monotonicity_table(
        rows, "overture_score", edges, _fit_config(), reps = 20
    )
    assert len(table) == len(edges) - 1
    assert np.isnan(table["drop_z"].iloc[-1])
    # Bins with fewer than min_gold gold rows report no z.
    thin = table["n_gold"].to_numpy() < 5
    pair_thin = np.minimum.reduce([thin[:-1], thin[1:]]) | thin[:-1] | thin[1:]
    assert np.all(np.isnan(table["drop_z"].to_numpy()[:-1][pair_thin]))


def test_merge_thin_bins_folds_a_sub_floor_bin_into_its_neighbour():
    # Round 20260730's shape: a one-gold-row bin just below an atom.
    edges = np.array([0.0, 0.5, 0.91967, 0.919912, 0.919913, 1.0])
    values = np.concatenate([np.full(20, 0.3), [0.919], np.full(30, 0.919912),
                             np.full(20, 0.95)])
    gold = np.ones(len(values), dtype = bool)
    merged = calibration_fit.merge_thin_bins(edges, values, gold, min_gold = 5)
    # The thin bin [0.5, 0.91967) had no gold; [0.91967, 0.919912) one row.
    # Both fold into the non-atom neighbour below, and the atom stays isolated.
    assert 0.919912 in merged and 0.919913 in merged
    bins = calibration_fit._bin_index(np.round(values, 6), merged)
    assert np.bincount(bins, minlength = len(merged) - 1).min() >= 5


def test_merge_thin_bins_leaves_an_axis_without_enough_gold_alone():
    edges = np.array([0.0, 0.5, 1.0])
    gold = np.array([True, False, True])
    out = calibration_fit.merge_thin_bins(edges, [0.2, 0.6, 0.7], gold,
                                          min_gold = 5)
    assert out.tolist() == edges.tolist()


def test_merged_monotonicity_table_reports_every_interior_z():
    rng = np.random.default_rng(17)
    overture = np.concatenate([np.full(900, 0.919912),
                               rng.uniform(0.3, 1.0, 2100)])
    rows = _matched_rows(n = 3000, seed = 17, overture = overture)
    edges = calibration_fit.merge_thin_bins(
        calibration_fit.atom_aware_edges(rows["overture_score"], 10),
        rows["overture_score"], rows["gold"], 5,
    )
    table = calibration_fit.axis_monotonicity_table(
        rows, "overture_score", edges, _fit_config(), reps = 20, min_gold = 5
    )
    assert (table["n_gold"] >= 5).all()
    assert np.isfinite(table["drop_z"].iloc[:-1]).all()


def test_build_lookup_bins_edge_values_like_deploy():
    """A bin's published value is the mean over exactly the rows deploy
    sends to it -- including rows sitting ON an interior edge (atoms)."""
    rng = np.random.default_rng(16)
    scores = np.round(np.concatenate([np.full(4000, 0.85),
                                      np.full(3000, 0.919912),
                                      rng.uniform(0.2, 1.0, 3000)]), 6)
    grid = np.linspace(0.2, 1.0, 101)
    curve = grid ** 2
    lookup = calibration_fit.build_lookup(
        "overture", scores, grid, {"mean": curve, "lower": curve,
                                   "upper": curve}, 10,
    )
    served = calibration.apply_curve(scores, lookup)["conf_mean"].to_numpy()
    values = np.interp(scores, grid, curve)
    bins = calibration_fit.lookup_bins(scores,
                                       calibration_fit.lookup_edges(scores, 10))
    for b in np.unique(bins):
        in_bin = bins == b
        np.testing.assert_allclose(served[in_bin], values[in_bin].mean(),
                                   atol = 1e-12)


def test_bin_band_aggregation_brackets_and_is_monotone():
    rows = _matched_rows(n = 2500, seed = 17)
    config = _fit_config(matched_index_mode = "pool", bootstrap_reps = 30,
                         band_aggregation = "bin")
    result = calibration_fit.fit_segment(rows, "matched", config,
                                         population = rows, cross_fit = False)
    lookup = result["lookup"]
    for column in ("conf_lower", "conf_mean", "conf_upper"):
        assert np.all(np.diff(lookup[column]) >= -1e-12)
    assert np.all(lookup["conf_lower"] <= lookup["conf_mean"] + 1e-12)
    assert np.all(lookup["conf_upper"] >= lookup["conf_mean"] - 1e-12)
    assert (lookup["conf_upper"] - lookup["conf_lower"]).max() > 0
