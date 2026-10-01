"""Tests for ``scripts/conflation/export_bayes_curves.py`` on synthetic draws.

The export's gate, grids, monotonicity check and writes run against fake fit
directories: summaries written here and an evaluator that returns monotone
logistic draws, so no sampler runs. Pinned:

- the artifact schema the deploy side reads (columns, dtypes, grid order) and
  the per-segment metadata
- posterior mean and band are monotone on the grids and inside [0, 1]
- the gate refuses a failed acceptance, a missing tag and a non-mixture fit,
  writing nothing; ``--allow-unaccepted`` exports with ``accepted: false``
- an existing export is replaced only with ``overwrite``
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from openpois.conflation import calibration_bayes as cb

SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "conflation"
          / "export_bayes_curves.py")
TAGS = ("mixture_overture", "mixture_osm", "mixture_matched")


def _module():
    spec = importlib.util.spec_from_file_location("export_bayes_curves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _module()


def _summary(segment: str, accepted: bool = True, arm: str = "C",
             noise: str = "fixed_mixture") -> dict:
    spec = cb.ModelSpec(arm = arm, label_noise = noise,
                        segments = cb.SEGMENT_ORDER if arm == "A" else (segment,))
    acceptance = {"rhat": True, "ess_bulk": accepted, "ess_tail": True,
                  "divergences": True, "ebfmi": True, "treedepth": True,
                  "curve_rhat": True, "curve_ess": accepted}
    acceptance["all"] = all(acceptance.values())
    knots = ({segment: [0.0, 0.5, 1.0]} if segment != "matched" else
             {"matched_x": [0.0, 0.5, 1.0], "matched_y": [0.0, 0.9, 1.0]})
    return {
        "tag": f"mixture_{segment}", "spec": repr(spec),
        "segments": list(spec.segments), "rounds": ["20260730", "20261029"],
        "git": {"sha": "abc", "dirty": False}, "fit_minutes": 1.5,
        "diagnostics": {"max_rhat": 1.002, "min_ess_bulk": 350.0,
                        "min_ess_tail": 500.0, "divergences_per_chain": [0, 0],
                        "ebfmi_per_chain": [0.9, 1.0], "treedepth_saturated": 0,
                        "mean_accept": 0.85, "mean_steps": 15.0},
        "curve_convergence": {"max_rhat": 1.001, "min_ess_bulk": 420.0,
                              "min_ess_tail": 450.0},
        "acceptance": acceptance, "knots": knots,
        "forward_rates": {segment: {"se": 0.99, "sp": 0.9, "raw_se": 1.0,
                                    "raw_sp": 0.9, "ess_se": 300.0,
                                    "ess_sp": 120.0, "n_se": 400, "n_sp": 150}},
        "deployed_impact": {"segments": {segment: {
            "n": 1000, "mean_bayes": 0.8, "mean_published": 0.78,
            "mean_abs_vs_published": 0.03, "share_gt_0.05_vs_published": 0.2,
            "share_gt_0.10_vs_published": 0.05}}},
    }


def _write_fits(eval_dir: Path, summaries: dict) -> None:
    for tag, summary in summaries.items():
        fit_dir = eval_dir / "fits" / tag
        fit_dir.mkdir(parents = True)
        (fit_dir / "summary.json").write_text(json.dumps(summary))
        (fit_dir / "draws.npz").write_bytes(b"")


@pytest.fixture
def eval_dir(tmp_path):
    path = tmp_path / "calibration_bayes"
    _write_fits(path, {f"mixture_{s}": _summary(s) for s in cb.SEGMENT_ORDER})
    return path


def _factory(n_draws = 60, seed = 0):
    """Evaluator over monotone logistic draws: positive slopes in every score."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(-1, 0.5, n_draws), rng.gamma(4, 0.8, n_draws), \
        rng.gamma(4, 0.6, n_draws)

    def factory(eval_dir, tag, summary):
        def evaluate(segment, osm, overture):
            osm, overture = np.asarray(osm), np.asarray(overture)
            if segment == "matched":
                f = a[:, None] + b[:, None] * osm + c[:, None] * overture
            else:
                s = osm if segment == "osm" else overture
                f = a[:, None] + b[:, None] * s
            return expit(f)
        return evaluate

    return factory


def _export(eval_dir, out_dir, **kwargs):
    kwargs.setdefault("evaluator_factory", _factory())
    return ex.export(eval_dir, out_dir, TAGS, **kwargs)


def test_schema_grids_and_metadata(eval_dir, tmp_path):
    out = tmp_path / "calibration"
    _export(eval_dir, out)
    for segment in ("osm", "overture"):
        frame = pd.read_parquet(out / f"{segment}_curve.parquet")
        assert list(frame.columns) == ["segment", "score", "conf_mean", "conf_lower",
                                       "conf_upper"]
        assert len(frame) == 2001
        assert (frame["segment"] == segment).all()
        assert pd.api.types.is_string_dtype(frame["segment"])
        for column in ("score", "conf_mean", "conf_lower", "conf_upper"):
            assert frame[column].dtype == np.float64
        assert frame["score"].iloc[0] == 0.0 and frame["score"].iloc[-1] == 1.0
        assert np.all(np.diff(frame["score"]) > 0)
        np.testing.assert_allclose(np.diff(frame["score"]), 0.0005, atol = 1e-12)
        meta = json.loads((out / f"{segment}_metadata.json").read_text())
        assert meta["grid_points"] == 2001 and "index_mode" not in meta
    frame = pd.read_parquet(out / "matched_curve.parquet")
    assert list(frame.columns) == ["segment", "osm_score", "overture_score",
                                   "conf_mean", "conf_lower", "conf_upper"]
    assert len(frame) == 201 * 201
    axis = np.round(np.linspace(0, 1, 201), 6)
    # Row-major: osm_score the outer (slow) axis, overture_score the inner.
    np.testing.assert_array_equal(frame["osm_score"].to_numpy()[:201],
                                  np.zeros(201))
    np.testing.assert_array_equal(frame["overture_score"].to_numpy()[:201], axis)
    np.testing.assert_array_equal(frame["osm_score"].to_numpy()[::201], axis)
    meta = json.loads((out / "matched_metadata.json").read_text())
    for segment in cb.SEGMENT_ORDER:
        meta = json.loads((out / f"{segment}_metadata.json").read_text())
        assert meta["segment"] == segment
        assert meta["method"] == "bayes_fixed_mixture"
        assert meta["lookup"] == "grid" and meta["score_decimals"] == 6
        assert meta["tag"] == f"mixture_{segment}" and meta["accepted"] is True
        assert meta["validation_round"] == "20260730"
        assert meta["rounds"] == ["20260730", "20261029"]
        assert meta["forward_rates"]["se"] == 0.99
        assert meta["acceptance"]["all"] is True
        assert meta["diagnostics"]["max_rhat"] == 1.002
        assert meta["fit_minutes"] == 1.5 and meta["git"]["sha"] == "abc"
        assert meta["knots"]
    assert meta["index_mode"] == "grid" and meta["grid_points"] == [201, 201]
    report = (out / "fit_report.md").read_text()
    assert "ht_review_20260730.pdf" in report
    assert "Deployed-impact" in report and "PASS" in report


def test_mean_and_band_are_monotone_and_bounded(eval_dir, tmp_path):
    out = tmp_path / "curves"
    _export(eval_dir, out, grid_1d = 401, grid_2d = 41)
    for segment in ("osm", "overture"):
        frame = pd.read_parquet(out / f"{segment}_curve.parquet")
        for column in ("conf_mean", "conf_lower", "conf_upper"):
            assert np.all(np.diff(frame[column]) >= 0)
        assert np.all(frame["conf_lower"] <= frame["conf_mean"])
        assert np.all(frame["conf_mean"] <= frame["conf_upper"])
        assert frame[["conf_lower", "conf_upper"]].stack().between(0, 1).all()
    frame = pd.read_parquet(out / "matched_curve.parquet")
    for column in ("conf_mean", "conf_lower", "conf_upper"):
        surface = frame[column].to_numpy().reshape(41, 41)
        assert np.all(np.diff(surface, axis = 0) >= 0)
        assert np.all(np.diff(surface, axis = 1) >= 0)


def test_non_monotone_draws_are_refused(eval_dir, tmp_path):
    def factory(eval_dir, tag, summary):
        return lambda segment, osm, overture: np.tile(
            np.cos(3 * np.asarray(osm if segment == "osm" else overture)), (5, 1))

    with pytest.raises(ValueError, match = "decreases"):
        _export(eval_dir, tmp_path / "out", evaluator_factory = factory)
    assert not (tmp_path / "out").exists()


def test_failed_acceptance_is_refused_unless_allowed(tmp_path):
    eval_dir = tmp_path / "eval"
    summaries = {f"mixture_{s}": _summary(s) for s in cb.SEGMENT_ORDER}
    summaries["mixture_matched"] = _summary("matched", accepted = False)
    _write_fits(eval_dir, summaries)
    out = tmp_path / "calibration"
    with pytest.raises(ex.ExportRefused) as refused:
        _export(eval_dir, out)
    assert refused.value.code == 2
    assert "mixture_matched: acceptance FAIL (ess_bulk, curve_ess)" in str(
        refused.value)
    assert not out.exists()
    # Testing only: exported, and the metadata says it was not accepted.
    _export(eval_dir, out, allow_unaccepted = True, grid_1d = 11, grid_2d = 5)
    assert json.loads((out / "matched_metadata.json").read_text())[
        "accepted"] is False
    assert json.loads((out / "osm_metadata.json").read_text())["accepted"] is True
    assert "allow-unaccepted" in (out / "fit_report.md").read_text()


def test_missing_tag_is_refused(tmp_path):
    eval_dir = tmp_path / "eval"
    _write_fits(eval_dir, {f"mixture_{s}": _summary(s) for s in ("overture", "osm")})
    out = tmp_path / "calibration"
    with pytest.raises(ex.ExportRefused) as refused:
        _export(eval_dir, out, allow_unaccepted = True)
    assert refused.value.code == 2 and "mixture_matched: missing" in str(
        refused.value)
    assert not out.exists()


def test_non_mixture_fit_is_refused(tmp_path):
    eval_dir = tmp_path / "eval"
    summaries = {f"mixture_{s}": _summary(s) for s in cb.SEGMENT_ORDER}
    summaries["mixture_osm"] = _summary("osm", noise = "fixed")
    _write_fits(eval_dir, summaries)
    with pytest.raises(ex.ExportRefused, match = "not a fixed-rate mixture"):
        _export(eval_dir, tmp_path / "calibration")


def test_existing_export_needs_overwrite(eval_dir, tmp_path):
    out = tmp_path / "calibration"
    _export(eval_dir, out, grid_1d = 11, grid_2d = 5)
    before = (out / "osm_curve.parquet").read_bytes()
    with pytest.raises(ex.ExportRefused) as refused:
        _export(eval_dir, out, grid_1d = 21, grid_2d = 5)
    assert refused.value.code == 3
    assert (out / "osm_curve.parquet").read_bytes() == before
    _export(eval_dir, out, grid_1d = 21, grid_2d = 5, overwrite = True)
    assert len(pd.read_parquet(out / "osm_curve.parquet")) == 21


def test_cli_exits_2_on_a_missing_fit(tmp_path):
    out = tmp_path / "calibration"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--eval-dir", str(tmp_path / "empty"),
         "--out-dir", str(out)], capture_output = True, text = True)
    assert result.returncode == 2, result.stderr
    assert "mixture_overture: missing" in result.stderr
    assert not out.exists()
