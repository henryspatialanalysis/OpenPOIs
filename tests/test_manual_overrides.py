"""Tests for openpois.conflation.manual_overrides (post-calibration pins)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpois.conflation import manual_overrides as mo


def _calibrated(tmp_path, n = 12, with_flag = True):
    """A calibrated-shaped conflated parquet (what apply_calibration writes)."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "unified_id": [f"u{i}" for i in range(n)],
        "overture_id": [f"ov{i}" if i % 3 else None for i in range(n)],
        "source": np.resize(["matched", "osm", "overture"], n),
        "name": ["Place"] * n,
        "conf_mean": rng.uniform(0.3, 0.9, n),
        "conf_lower": rng.uniform(0.1, 0.3, n),
        "conf_upper": rng.uniform(0.9, 1.0, n),
    })
    if with_flag:
        frame["calibration_flag"] = [None if i % 2 else "shadow_cd"
                                     for i in range(n)]
    path = tmp_path / "conflated.parquet"
    table = pa.Table.from_pandas(frame, preserve_index = False)
    table = table.replace_schema_metadata({b"geo": b"{}"})
    pq.write_table(table, path)
    return path, frame


def _csv(tmp_path, rows: list[tuple]):
    path = tmp_path / "manual_overrides.csv"
    lines = ["unified_id,overture_id,action,reason,date,report_id"]
    lines += [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_exclude_and_include_after_calibration(tmp_path):
    in_path, frame = _calibrated(tmp_path)
    csv = _csv(tmp_path, [
        ("u1", "", "exclude", "closed per report", "2026-09-24", "rep-1"),
        ("", "ov4", "exclude", "wrong POI", "2026-09-24", "rep-2"),
        ("u7", "", "include", "must carry", "2026-09-24", "rep-3"),
    ])
    out_path = tmp_path / "conflated_overridden.parquet"
    stats = mo.apply_manual_overrides(
        in_path, out_path, mo.read_overrides(csv), chunk_rows = 5,
        verbose = False,
    )
    out = pd.read_parquet(out_path).set_index("unified_id")

    assert stats["rows"] == len(frame)
    assert stats["n_excluded"] == 2
    assert stats["n_included"] == 1
    assert stats["n_unmatched_overrides"] == 0
    for uid in ("u1", "u4"):
        assert out.loc[uid, ["conf_mean", "conf_lower", "conf_upper"]].tolist() \
            == [0.0, 0.0, 0.0]
        assert out.loc[uid, "calibration_flag"] == "manual_exclude"
    assert out.loc["u7", "conf_mean"] == 1.0
    assert out.loc["u7", "calibration_flag"] == "manual_include"
    # Everything else is untouched, including existing flags and metadata.
    untouched = [u for u in frame["unified_id"] if u not in {"u1", "u4", "u7"}]
    src = frame.set_index("unified_id")
    for col in ("conf_mean", "conf_lower", "conf_upper"):
        np.testing.assert_allclose(out.loc[untouched, col], src.loc[untouched, col])
    assert out.loc["u0", "calibration_flag"] == "shadow_cd"
    assert pq.read_schema(out_path).metadata.get(b"geo") == b"{}"
    assert list(pd.read_parquet(out_path)["unified_id"]) == list(frame["unified_id"])


def test_idempotent_and_in_place(tmp_path):
    in_path, _ = _calibrated(tmp_path)
    overrides = mo.read_overrides(_csv(tmp_path, [
        ("u2", "", "exclude", "r", "2026-09-24", "rep"),
        ("u3", "", "include", "r", "2026-09-24", "rep"),
    ]))
    mo.apply_manual_overrides(in_path, in_path, overrides, verbose = False)
    first = pd.read_parquet(in_path)
    assert not (tmp_path / "conflated.parquet.tmp").exists()
    mo.apply_manual_overrides(in_path, in_path, overrides, verbose = False)
    second = pd.read_parquet(in_path)
    pd.testing.assert_frame_equal(first, second)
    assert first.set_index("unified_id").loc["u2", "conf_mean"] == 0.0
    assert first.set_index("unified_id").loc["u3", "conf_mean"] == 1.0


def test_exclude_wins_over_include_and_last_row_wins(tmp_path):
    in_path, _ = _calibrated(tmp_path)
    overrides = mo.read_overrides(_csv(tmp_path, [
        ("u4", "", "include", "r", "2026-09-24", "rep"),
        ("", "ov4", "exclude", "r", "2026-09-24", "rep"),   # same row, other id
        ("u5", "", "exclude", "r", "2026-09-24", "rep"),
        ("u5", "", "include", "r", "2026-09-25", "rep"),    # later row wins
    ]))
    out = tmp_path / "out.parquet"
    mo.apply_manual_overrides(in_path, out, overrides, verbose = False)
    res = pd.read_parquet(out).set_index("unified_id")
    assert res.loc["u4", "calibration_flag"] == "manual_exclude"
    assert res.loc["u5", "calibration_flag"] == "manual_include"


def test_missing_file_is_a_no_op(tmp_path):
    in_path, frame = _calibrated(tmp_path)
    overrides = mo.read_overrides(tmp_path / "does_not_exist.csv")
    assert overrides.empty
    assert list(overrides.columns) == list(mo.REQUIRED_COLUMNS)
    out = tmp_path / "out.parquet"
    stats = mo.apply_manual_overrides(in_path, out, overrides, verbose = False)
    assert stats["n_excluded"] == stats["n_included"] == 0
    pd.testing.assert_frame_equal(pd.read_parquet(out), frame)


def test_flag_column_is_appended_when_absent(tmp_path):
    in_path, _ = _calibrated(tmp_path, with_flag = False)
    overrides = mo.read_overrides(_csv(tmp_path, [
        ("u1", "", "exclude", "r", "2026-09-24", "rep"),
    ]))
    out = tmp_path / "out.parquet"
    mo.apply_manual_overrides(in_path, out, overrides, chunk_rows = 4,
                              verbose = False)
    res = pd.read_parquet(out).set_index("unified_id")
    assert res.loc["u1", "calibration_flag"] == "manual_exclude"
    assert res["calibration_flag"].isna().sum() == len(res) - 1


def test_unmatched_override_is_counted_not_fatal(tmp_path):
    in_path, _ = _calibrated(tmp_path)
    overrides = mo.read_overrides(_csv(tmp_path, [
        ("nope", "", "exclude", "r", "2026-09-24", "rep"),
    ]))
    stats = mo.apply_manual_overrides(
        in_path, tmp_path / "out.parquet", overrides, verbose = False,
    )
    assert stats["n_unmatched_overrides"] == 1
    assert stats["n_excluded"] == 0


@pytest.mark.parametrize(
    "rows, message",
    [
        ([("u1", "", "drop", "r", "d", "rep")], "action must be one of"),
        ([("", "", "exclude", "r", "d", "rep")], "needs a unified_id"),
    ],
)
def test_read_overrides_rejects_malformed_rows(tmp_path, rows, message):
    with pytest.raises(ValueError, match = message):
        mo.read_overrides(_csv(tmp_path, rows))


def test_read_overrides_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("unified_id,action\nu1,exclude\n")
    with pytest.raises(ValueError, match = "missing required column"):
        mo.read_overrides(path)
