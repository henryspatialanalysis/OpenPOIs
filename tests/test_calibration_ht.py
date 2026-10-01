"""Tests for the design-weighted (Horvitz-Thompson) check of a deployed map.

- the corrected labels: gold rows keep their truth, silver rows take
  q(segment, verdict), unverifiable included, with q hand-checkable
- the corrected rate and its SD on a hand-checkable example, and the
  gold-only Hajek reference
- the Jeffreys smoothing of the SD at r = 0 and r = 1
- the flag rule at 1 and 2 SD
- bins merged until each holds at least 20 phase-1 rows
- an exact map flags about the chance share of bins
- thin or empty segments are reported, never raised on
- the review PDF has the expected page count
- grid curves (the Bayesian export) review and draw like step curves
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from openpois.conflation import calibration_bayes, calibration_fit
from openpois.conflation import calibration_ht as cht

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "conflation"


def _script_module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _truth(score):
    return 0.35 + 0.6 * np.asarray(score, dtype = float)


def _rows(segment = "osm", n = 2500, seed = 0, atoms = False) -> pd.DataFrame:
    """Synthetic two-phase rows with P(exists | score) = ``_truth(score)``.

    As in the real design, the definitive LLM verdicts are nearly always
    right, phase 2 samples the verdict classes at different rates, and the
    unverifiable class is censused.
    """
    rng = np.random.default_rng(seed)
    osm = rng.uniform(0.05, 1.0, n)
    overture = rng.uniform(0.05, 1.0, n)
    if atoms:
        pick = rng.random(n)
        overture = np.where(pick < 0.3, 0.919912,
                            np.where(pick < 0.5, 0.990219, overture))
    score = osm if segment != "overture" else overture
    y = (rng.random(n) < _truth(score)).astype(float)
    draw = rng.random(n)
    verdict = np.where(
        y == 1,
        np.where(draw < 0.85, "exists",
                 np.where(draw < 0.86, "gone", "unverifiable")),
        np.where(draw < 0.7, "gone",
                 np.where(draw < 0.72, "exists", "unverifiable")),
    )
    inclusion = pd.Series(verdict).map(
        {"exists": 0.12, "gone": 0.37, "unverifiable": 1.0}).to_numpy()
    gold = rng.random(n) < inclusion
    return pd.DataFrame({
        "validation_round": "test",
        "segment": segment, "stratum": segment,
        "raw_score": 0.5 * (osm + overture),
        "osm_score": osm, "overture_score": overture,
        "llm_verdict": verdict, "llm_confidence": "high",
        "gold": gold, "y": np.where(gold, y, np.nan),
    })


def _exact_lookup(segment: str) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, 501)
    return pd.DataFrame({
        "segment": segment, "score_lo": edges[:-1], "score_hi": edges[1:],
        "conf_mean": _truth((edges[:-1] + edges[1:]) / 2.0),
        "conf_lower": 0.0, "conf_upper": 1.0,
    })


def _matched_setup(seed = 0):
    """Matched rows plus an average-index curve that is exact on the mean."""
    rows = _rows("matched", n = 3000, seed = seed, atoms = True)
    curves = {"matched": _exact_lookup("matched")}
    metadata = {"matched": {"index_mode": "average", "score_decimals": 6}}
    return rows, curves, metadata


def _hand_rows(segments = calibration_fit.SEGMENTS) -> pd.DataFrame:
    """Per segment: 3 gold + 2 silver exists (gold y 1, 1, 0); 2 gold + 1
    silver gone (gold y 0, 0); 4 gold + 1 silver unverifiable (gold y 1, 1,
    1, 0). Inclusion is uniform within each verdict, so each Hajek share is
    the raw share and its Kish ESS the gold count."""
    spec = [("exists", [1.0, 1.0, 0.0], 2), ("gone", [0.0, 0.0], 1),
            ("unverifiable", [1.0, 1.0, 1.0, 0.0], 1)]
    out = []
    for segment in segments:
        for verdict, gold_y, n_silver in spec:
            for value in gold_y:
                out.append((segment, verdict, True, value))
            out += [(segment, verdict, False, np.nan)] * n_silver
    frame = pd.DataFrame(out, columns = ["segment", "llm_verdict", "gold",
                                         "y"])
    return frame.assign(stratum = frame["segment"], llm_confidence = "high",
                        osm_score = 0.5, overture_score = 0.5,
                        raw_score = 0.5)


# --- Corrected labels -------------------------------------------------------

def test_correction_rates_and_corrected_labels_by_hand():
    rows = _hand_rows()
    rates = cht.correction_rates(rows, calibration_fit.FitConfig())
    q = {(r.segment, r.verdict): r for r in rates.itertuples()}
    expected = {"exists": (2.0 + 0.5) / 4.0, "gone": 0.5 / 3.0,
                "unverifiable": (3.0 + 0.5) / 5.0}
    for segment in calibration_fit.SEGMENTS:
        for verdict, value in expected.items():
            assert np.isclose(q[(segment, verdict)].q, value)
        # Exists and gone are arm C's silver-label rates.
        assert q[(segment, "exists")].source == "silver_label_rates"
        assert q[(segment, "unverifiable")].source == "design-weighted gold"
    silver = calibration_bayes.silver_label_rates(rows,
                                                  calibration_fit.FitConfig())
    assert np.isclose(q[("osm", "gone")].q, silver["osm"]["gone"])
    assert q[("osm", "unverifiable")].n_gold == 4
    assert np.isclose(q[("osm", "unverifiable")].ess, 4.0)

    labels = cht.corrected_labels(rows, rates)
    gold = rows["gold"].to_numpy()
    assert np.array_equal(labels[gold], rows["y"].to_numpy()[gold])
    for verdict, value in expected.items():
        silver_rows = ~gold & (rows["llm_verdict"] == verdict).to_numpy()
        assert np.allclose(labels[silver_rows], value)


def test_correction_rates_fall_back_when_a_segment_lacks_gold():
    # One segment only: silver_label_rates refuses (the other segments have
    # no gold), so every rate comes from the helpers with the same values.
    rows = _hand_rows(segments = ("osm",))
    rows = rows[~((rows["llm_verdict"] == "gone") & rows["gold"])]
    rates = cht.correction_rates(rows, calibration_fit.FitConfig())
    by_verdict = rates.set_index("verdict")
    assert np.isclose(by_verdict.loc["exists", "q"], 2.5 / 4.0)
    assert by_verdict.loc["exists", "source"] == "design-weighted gold"
    assert np.isnan(by_verdict.loc["gone", "q"])
    labels = cht.corrected_labels(rows, rates)
    silver_gone = (rows["llm_verdict"] == "gone").to_numpy()
    assert np.isnan(labels[silver_gone]).all()


# --- Estimator --------------------------------------------------------------

def test_corrected_rate_sd_and_gold_reference_by_hand():
    stats = cht.corrected_rate([1.0, 0.0, 0.7, 0.9, np.nan])
    assert stats["n_rows"] == 4
    assert np.isclose(stats["rate"], 0.65)
    sd = np.sqrt(0.65 * 0.35 / 4.0)
    assert np.isclose(stats["sd"], sd)
    assert np.isclose(stats["ci_lower"], 0.65 - 1.96 * sd)
    # The gold-only reference: weights 1, 1, 2, 4 on outcomes 1, 0, 1, 1.
    gold = cht.hajek([1.0, 0.0, 1.0, 1.0, np.nan], [1.0, 1.0, 2.0, 4.0, 0.0])
    assert gold["n_gold"] == 4
    assert np.isclose(gold["ht_rate"], 7.0 / 8.0)
    assert np.isclose(gold["ess"], 64.0 / 22.0)
    assert np.isclose(gold["raw_rate"], 0.75)


def test_jeffreys_smoothing_at_zero_and_one():
    n = 9.0
    assert np.isclose(cht.binomial_sd(0.0, n),
                      np.sqrt((0.5 / 10.0) * (9.5 / 10.0) / n))
    assert np.isclose(cht.binomial_sd(1.0, n),
                      np.sqrt((9.5 / 10.0) * (0.5 / 10.0) / n))
    # Interior rates are not smoothed.
    assert np.isclose(cht.binomial_sd(0.3, n), np.sqrt(0.3 * 0.7 / n))
    assert np.isnan(cht.binomial_sd(0.5, 0.0))
    # A bin whose labels all exist keeps its rate and gets a positive SD.
    stats = cht.corrected_rate(np.ones(5))
    assert stats["rate"] == 1.0 and stats["sd"] > 0.0


def test_flag_rule_at_one_and_two_sd():
    z = np.array([0.0, 0.99, -1.01, 1.5, 2.0, -2.01, 3.0, np.nan])
    assert cht.flag_level(z).tolist() == [0, 0, 1, 1, 1, 2, 2, 0]
    # Through the bin table: rate 0.5 on 100 rows per bin (SD 0.05) against
    # model means 0.54, 0.56 and 0.62.
    labels = np.tile([0.0, 1.0], 150)
    bins = np.repeat([0, 1, 2], 100)
    model = np.repeat([0.54, 0.56, 0.62], 100)
    table = cht.bin_table(bins, 3, labels, model)
    assert np.allclose(table["z"], [0.8, 1.2, 2.4])
    assert table["flag"].tolist() == [0, 1, 2]


def test_bins_are_merged_to_twenty_rows():
    rows = _rows("overture", n = 400, seed = 3, atoms = True)
    edges = cht.axis_edges(rows["overture_score"])
    counts = np.bincount(cht._assign(rows["overture_score"], edges),
                         minlength = len(edges) - 1)
    assert counts.min() >= cht.HT_MIN_ROWS
    assert 0.919912 in [edges[j] for j in cht.atom_columns(edges)]
    small = _rows("osm", n = 150, seed = 3)
    check = cht.run_ht_check(small, {"osm": _exact_lookup("osm")}, {},
                             calibration_fit.FitConfig())
    assert len(check["bins"]) < cht.HT_BASE_BINS
    assert (check["bins"]["n_phase1"] >= cht.HT_MIN_ROWS).all()
    assert check["bins"]["tested"].all()
    rows_m, curves, metadata = _matched_setup()
    check = cht.run_ht_check(rows_m.iloc[:600], curves, metadata,
                             calibration_fit.FitConfig())
    for view in ("cells", "raw_score_deciles"):
        table = check["bins"][check["bins"]["view"] == view]
        assert len(table) and (table["n_phase1"] >= cht.HT_MIN_ROWS).all()
    ov_edges = check["grids"]["matched"]["ov_edges"]
    assert 0.919912 in [ov_edges[j] for j in cht.atom_columns(ov_edges)]


def test_design_weights_are_inverse_class_inclusion():
    rows = _rows(seed = 1)
    weights = cht.design_weights(rows, calibration_fit.FitConfig())
    gold = rows["gold"].to_numpy()
    assert np.all(weights[~gold] == 0.0)
    for verdict in ("exists", "gone", "unverifiable"):
        in_class = (rows["llm_verdict"] == verdict).to_numpy()
        expected = in_class.sum() / (in_class & gold).sum()
        assert np.allclose(weights[in_class & gold], expected)


def test_exact_map_flags_about_the_chance_share():
    flagged_1 = flagged_2 = tested = 0
    for seed in range(30):
        check = cht.run_ht_check(_rows(seed = seed),
                                 {"osm": _exact_lookup("osm")}, {},
                                 calibration_fit.FitConfig())
        bins = check["bins"][check["bins"]["tested"]]
        tested += len(bins)
        flagged_1 += int((bins["flag"] >= 1).sum())
        flagged_2 += int((bins["flag"] >= 2).sum())
    assert tested >= 200
    assert 0.2 < flagged_1 / tested < 0.45
    assert flagged_2 / tested < 0.12
    assert abs(check["large"].iloc[0]["z"]) < 3.0


def test_biased_map_is_flagged():
    rows = _rows(seed = 2)
    lookup = _exact_lookup("osm")
    lookup["conf_mean"] = np.clip(lookup["conf_mean"] + 0.15, 0.0, 1.0)
    check = cht.run_ht_check(rows, {"osm": lookup}, {},
                             calibration_fit.FitConfig())
    assert (check["bins"]["flag"] == 2).mean() > 0.5
    assert check["large"].iloc[0]["z"] > 2.0


def test_check_reports_thin_and_empty_segments_without_raising():
    fit_config = calibration_fit.FitConfig()
    # No rows at all, and a legacy stratum that must be excluded.
    empty = _rows(n = 10).iloc[0:0]
    check = cht.run_ht_check(empty, {}, {}, fit_config)
    assert check["bins"].empty and check["summary"].empty
    assert len(check["notes"]) == 3
    legacy = _rows(n = 50).assign(stratum = "overture_missing_conf")
    assert cht.run_ht_check(legacy, {}, {}, fit_config)["bins"].empty
    # A handful of rows: one bin, reported but not tested.
    thin = _rows(n = 12, seed = 4)
    check = cht.run_ht_check(thin, {"osm": _exact_lookup("osm")}, {},
                             fit_config)
    assert len(check["bins"]) >= 1
    assert not check["bins"]["tested"].any()
    assert any("not tested" in note for note in check["notes"])
    # No gold at all, and a segment whose curve is missing.
    no_gold = _rows(n = 100).assign(gold = False, y = np.nan)
    check = cht.run_ht_check(no_gold, {}, {}, fit_config)
    assert any("no gold" in note for note in check["notes"])
    check = cht.run_ht_check(_rows(seed = 5), {}, {}, fit_config)
    assert any("no deployed curve" in note for note in check["notes"])
    assert not check["bins"]["tested"].any()
    # Matched rows whose curve metadata lacks index parameters.
    rows_m, curves, _ = _matched_setup()
    check = cht.run_ht_check(rows_m, curves,
                             {"matched": {"index_mode": "interaction"}},
                             fit_config)
    assert any("could not be applied" in note for note in check["notes"])
    lines = cht.report_lines(check, "ht_review_test.pdf")
    assert any("ht_review_test.pdf" in line for line in lines)
    assert any("unverifiable" in line for line in lines)


def test_deployed_means_match_the_deploy_path():
    rows_m, curves, metadata = _matched_setup()
    model = cht.deployed_means(rows_m, "matched", curves, metadata)
    index = calibration_fit.average_score(rows_m["osm_score"],
                                          rows_m["overture_score"])
    expected = calibration_fit.apply_step_lookup(
        np.round(index, 6), curves["matched"])["conf_mean"].to_numpy()
    assert np.allclose(model, expected)


# --- Review document --------------------------------------------------------

def _pdf_pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def test_review_pdf_has_the_expected_pages(tmp_path):
    review = _script_module("ht_review")
    rows = pd.concat([_matched_setup()[0], _rows("osm", seed = 6),
                      _rows("overture", seed = 7, atoms = True)],
                     ignore_index = True)
    curves = {s: _exact_lookup(s) for s in ("matched", "osm", "overture")}
    metadata = {"matched": {"index_mode": "average", "score_decimals": 6}}
    check = cht.run_ht_check(rows, curves, metadata,
                             calibration_fit.FitConfig())
    path = tmp_path / "calibration" / "ht_review_test.pdf"
    pages = review.write_review_pdf(check, path, curves, metadata,
                                    info = {"round": "test"})
    assert path.exists()
    # Summary, matched cells, matched deciles, osm, overture, bin table(s).
    n_table = int(np.ceil(len(check["bins"]) / review.TABLE_ROWS_PER_PAGE))
    assert pages == 5 + n_table == review.expected_pages(check)
    assert _pdf_pages(path) == pages


def test_run_review_writes_the_pdf_beside_the_curves(tmp_path):
    review = _script_module("ht_review")
    curves_dir = tmp_path / "calibration"
    curves_dir.mkdir()
    _exact_lookup("osm").to_parquet(curves_dir / "osm_curve.parquet")
    (curves_dir / "osm_metadata.json").write_text(
        '{"fit_config": {"min_cell_gold": 25}}', encoding = "utf-8"
    )
    result = review.run_review(curves_dir, _rows(seed = 8),
                               {"validation_round": "r1"})
    assert result["pdf"] == curves_dir / "ht_review_r1.pdf"
    assert result["pdf"].exists() and result["csv"].exists()
    assert _pdf_pages(result["pdf"]) == result["pages"] == 3


def test_fit_run_check_never_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    fit = _script_module("fit_calibration")
    config = calibration_fit.FitConfig()
    # No curves: the error is recorded in the section, not raised.
    lines = fit.ht_review_lines(tmp_path / "missing", _rows(seed = 9),
                                {"validation_round": "r1"}, config)
    assert any("raised" in line for line in lines)
    curves_dir = tmp_path / "calibration"
    curves_dir.mkdir()
    _exact_lookup("osm").to_parquet(curves_dir / "osm_curve.parquet")
    lines = fit.ht_review_lines(curves_dir, _rows(seed = 9),
                                {"validation_round": "r1"}, config)
    assert (curves_dir / "ht_review_r1.pdf").exists()
    assert any("ht_review_r1.pdf" in line for line in lines)


# --- Grid curves (Bayesian export, October 2026) ----------------------------

def _grid_curves() -> tuple:
    """Grid curves in the export schema, exact on ``_truth``: 1-D on the
    native score, matched on the OSM score whatever the Overture score."""
    nodes = np.linspace(0.0, 1.0, 201)
    curves = {}
    for segment in ("osm", "overture"):
        mean = _truth(nodes)
        curves[segment] = pd.DataFrame({
            "segment": segment, "score": nodes, "conf_mean": mean,
            "conf_lower": mean - 0.03, "conf_upper": mean + 0.03,
        })
    axis = np.linspace(0.0, 1.0, 41)
    oo, vv = np.meshgrid(axis, axis, indexing = "ij")
    mean = _truth(oo.ravel())
    curves["matched"] = pd.DataFrame({
        "segment": "matched", "osm_score": oo.ravel(),
        "overture_score": vv.ravel(), "conf_mean": mean,
        "conf_lower": mean - 0.03, "conf_upper": mean + 0.03,
    })
    metadata = {s: {"method": "bayes_fixed_mixture", "lookup": "grid",
                    "score_decimals": 6} for s in curves}
    metadata["matched"]["index_mode"] = "grid"
    return curves, metadata


def test_deployed_means_on_grid_curves():
    curves, metadata = _grid_curves()
    for segment, seed in (("matched", 10), ("osm", 11), ("overture", 12)):
        rows = _rows(segment, n = 400, seed = seed, atoms = True)
        model = cht.deployed_means(rows, segment, curves, metadata)
        score = rows["overture_score" if segment == "overture"
                     else "osm_score"]
        assert np.allclose(model, _truth(np.round(score, 6)))


def test_review_pdf_on_grid_curves(tmp_path):
    review = _script_module("ht_review")
    curves, metadata = _grid_curves()
    curves_dir = tmp_path / "calibration"
    curves_dir.mkdir()
    for segment, lookup in curves.items():
        lookup.to_parquet(curves_dir / f"{segment}_curve.parquet",
                          index = False)
        (curves_dir / f"{segment}_metadata.json").write_text(
            json.dumps(metadata[segment]), encoding = "utf-8"
        )
    rows = pd.concat([_rows("matched", n = 3000, seed = 13, atoms = True),
                      _rows("osm", seed = 14),
                      _rows("overture", seed = 15, atoms = True)],
                     ignore_index = True)
    result = review.run_review(curves_dir, rows, {"validation_round": "g1"})
    check = result["check"]
    assert not any("could not be applied" in n for n in check["notes"])
    assert np.isfinite(check["large"]["model_mean"]).all()
    # Summary, matched cells, matched deciles, osm, overture, bin table(s).
    n_table = int(np.ceil(len(check["bins"]) / review.TABLE_ROWS_PER_PAGE))
    assert result["pages"] == 5 + n_table == review.expected_pages(check)
    assert _pdf_pages(result["pdf"]) == result["pages"]
