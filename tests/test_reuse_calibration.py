"""Tests for the curve-reuse month's copy step (scripts/conflation/reuse_calibration.py).

- the drift gate passes and breaches on each criterion, and a NaN breaches
- missing files, an unaccepted fit and a different validation round are refused
- the curves are copied byte for byte and the fit report keeps the fit's version
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "conflation" / "reuse_calibration.py"
)
_spec = importlib.util.spec_from_file_location("reuse_calibration", PATH)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

PASSING = {
    "prior_version": 20261021.0, "current_version": 20261118.0,
    "rmse": 0.018, "bias_mean_diff": -0.001, "share_absdiff_gt_0.1": 0.005,
}


def _prior(tmp_path: Path, report: str = "fit_report.md", **meta) -> Path:
    directory = tmp_path / "prior"
    directory.mkdir()
    base = {
        "method": "bayes_fixed_mixture", "accepted": True,
        "validation_round": "20261001", "rounds": ["20261001", "20260730"],
        "eval_dir": "/x/calibration_bayes",
    }
    for segment in rc.SEGMENTS:
        (directory / f"{segment}_curve.parquet").write_bytes(segment.encode())
        (directory / f"{segment}_metadata.json").write_text(
            json.dumps({**base, **meta})
        )
    (directory / report).write_text("report")
    return directory


def test_drift_gate_passes():
    assert rc.drift_breaches(PASSING) == []


@pytest.mark.parametrize("key, value", [
    ("rmse", 0.11), ("bias_mean_diff", -0.031), ("share_absdiff_gt_0.1", 0.101),
    ("rmse", float("nan")),
])
def test_drift_gate_breaches(key, value):
    breaches = rc.drift_breaches({**PASSING, key: value})
    assert len(breaches) == 1 and key in breaches[0]


def test_read_drift_metrics(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("metric,value\nrmse,0.018\ncurrent_version,20261118.0\n")
    assert rc.read_drift_metrics(path) == {
        "rmse": 0.018, "current_version": 20261118.0,
    }


def test_check_metadata_accepts_good_dir(tmp_path):
    assert rc.check_metadata(_prior(tmp_path), "20261001") == []


@pytest.mark.parametrize("meta, text", [
    ({"accepted": False}, "accepted"), ({"validation_round": "20260730"}, "round"),
])
def test_check_metadata_refuses(tmp_path, meta, text):
    problems = rc.check_metadata(_prior(tmp_path, **meta), "20261001")
    assert len(problems) == len(rc.SEGMENTS) and all(text in p for p in problems)


def test_check_metadata_missing_file(tmp_path):
    directory = _prior(tmp_path)
    (directory / "osm_curve.parquet").unlink()
    assert rc.check_metadata(directory, "20261001") == [
        "osm: missing osm_curve.parquet"
    ]


@pytest.mark.parametrize("report, expected", [
    ("fit_report.md", "fit_report_20261022.md"),
    ("fit_report_20261001.md", "fit_report_20261001.md"),
])
def test_copy_curves(tmp_path, report, expected):
    prior = _prior(tmp_path, report)
    out = tmp_path / "out"
    copied = rc.copy_curves(prior, out, "20261022")
    assert expected in copied and (out / expected).exists()
    for segment in rc.SEGMENTS:
        name = f"{segment}_curve.parquet"
        assert (out / name).read_bytes() == (prior / name).read_bytes()


def test_provenance_text(tmp_path):
    meta = json.loads((_prior(tmp_path) / "matched_metadata.json").read_text())
    text = rc.provenance_text("20261022", meta, PASSING)
    assert "conflation/20261022/calibration" in text
    assert "validation round 20261001, pooled with 20260730" in text
    assert "20261118 against 20261021" in text
