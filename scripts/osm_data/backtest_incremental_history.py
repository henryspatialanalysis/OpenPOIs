#!/usr/bin/env python
"""
Backtest the incremental history roll-forward against a full-history build.

Rolls a full ``osm_data`` run (``--base``) forward with Geofabrik's public
daily diffs to ``--end-date``, builds ghosts from the rolled parquets and from
a full-history run covering the same date (``--reference``) with the *same*
``build_ghosts`` code, and compares them on ``(osm_id, event_type)``.

Gate (set 2026-09-26): recall and precision >= 97% on named + labeled ghosts.
Results go to ``<out>/backtest_report.md``.

Comparison rules:
- Only elements in both histories' universes (a base built before an ingest
  filter change has a different universe).
- Recall: reference ghosts in [window_start, window_end) found among rolled
  ghosts in [window_start - 1 d, window_end + 1 d). Precision: the reverse.
  The one-day pad absorbs the rolled path's delete timestamps, which are the
  diff file's state time (up to ~24 h after the real deletion).

Usage:
    python scripts/osm_data/backtest_incremental_history.py --step roll
    python scripts/osm_data/backtest_incremental_history.py --step compare
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from config_versioned import Config

from openpois.conflation.ghost_osm import build_ghosts
from openpois.conflation.taxonomy import (
    build_osm_tag_filter_expressions,
    load_osm_crosswalk,
)
from openpois.io.osm_history_incremental import (
    check_history_vs_snapshot,
    read_coverage,
    roll_osm_history,
)

UTC = datetime.timezone.utc
GATE = 0.97
PAD = pd.Timedelta(days = 1)
MATCH_FIELDS = ["prior_name", "prior_brand", "new_name", "shared_label"]


def _roll(args, config: Config, out_dir: Path) -> None:
    incremental = config.get("download", "osm", "incremental_history")
    base_dir = config.get_dir_path("osm_data", custom_version = args.base)
    t0 = time.time()
    roll_osm_history(
        base_dir = base_dir,
        base_version = args.base,
        out_dir = out_dir,
        out_versions_path = out_dir / "osm_versions.parquet",
        out_changes_path = out_dir / "osm_changes.parquet",
        replication_urls = incremental["replication_urls"],
        tag_filter_exprs = build_osm_tag_filter_expressions(load_osm_crosswalk()),
        end_date = args.end_date,
        max_chain_months = incremental["max_chain_months"],
        keep_diffs = True,
    )
    print(f"ROLL DONE in {time.time() - t0:.0f}s")


def _universe_nodes(osm_data_dir: Path) -> set[int]:
    return set(
        duckdb.sql(
            "SELECT DISTINCT id FROM "
            f"read_parquet('{osm_data_dir / 'osm_versions.parquet'}') "
            "WHERE type = 'node'"
        ).fetchnumpy()["id"].tolist()
    )


def _ghosts(osm_data_dir: Path, poi_keys: list[str], threshold: float):
    g = build_ghosts(
        osm_data_dir / "osm_versions.parquet",
        osm_data_dir / "osm_changes.parquet",
        poi_keys,
        name_change_similarity_threshold = threshold,
        verbose = False,
    )
    g = g.copy()
    g["named_labeled"] = g["prior_name"].notna() & (g["shared_label"] != "")
    return g


def _match(
    left: pd.DataFrame, right: pd.DataFrame,
    start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """Rows of ``left`` in [start, end) and whether each has a same-key row
    in ``right`` within the padded window."""
    lw = left[(left["event_timestamp"] >= start) & (left["event_timestamp"] < end)]
    rw = right[
        (right["event_timestamp"] >= start - PAD)
        & (right["event_timestamp"] < end + PAD)
    ]
    keys = set(zip(rw["osm_id"], rw["event_type"]))
    hit = pd.Series(
        [k in keys for k in zip(lw["osm_id"], lw["event_type"])],
        index = lw.index,
    )
    return lw, hit


def _compare(args, config: Config, out_dir: Path) -> None:
    ref_dir = config.get_dir_path("osm_data", custom_version = args.reference)
    poi_keys = config.get("download", "osm", "filter_keys")
    threshold = float(config.get(
        "conflation", "change_detection", "name_change_similarity_threshold",
    ))
    cov = read_coverage(out_dir)
    base_cov = read_coverage(
        config.get_dir_path("osm_data", custom_version = args.base)
    )
    start = pd.Timestamp(base_cov.coverage_end).ceil("D")
    end = pd.Timestamp(cov.coverage_end).floor("D")
    print(f"Comparison window [{start}, {end})")

    print("Building ghosts (rolled) ...")
    rolled = _ghosts(out_dir, poi_keys, threshold)
    print("Building ghosts (reference full build) ...")
    ref = _ghosts(ref_dir, poi_keys, threshold)

    shared = _universe_nodes(out_dir) & _universe_nodes(ref_dir)
    rolled = rolled[rolled["osm_id"].isin(shared)]
    ref = ref[ref["osm_id"].isin(shared)]

    lines = [
        "# Incremental history backtest",
        "",
        f"- Base (rolled forward): `osm_data/{args.base}`, coverage end "
        f"{base_cov.coverage_end.isoformat()}",
        f"- Rolled output: `{out_dir}`, coverage end "
        f"{cov.coverage_end.isoformat()}, sequences "
        + ", ".join(
            f"{k} ≤ {v['last_sequence']}" for k, v in cov.extracts.items()
        ),
        f"- Reference (full build): `osm_data/{args.reference}`",
        f"- Window: [{start.date()}, {end.date()}), ±1 day pad on the "
        "matched side; nodes in both universes only",
        "",
        "| Subset | Event type | Reference | Recall | Rolled | Precision |",
        "|---|---|--:|--:|--:|--:|",
    ]
    gate_values = {}
    for subset in ("all", "named + labeled"):
        r = ref if subset == "all" else ref[ref["named_labeled"]]
        o = rolled if subset == "all" else rolled[rolled["named_labeled"]]
        types = ["(all)"] + sorted(set(r["event_type"]) | set(o["event_type"]))
        for et in types:
            rs = r if et == "(all)" else r[r["event_type"] == et]
            os_ = o if et == "(all)" else o[o["event_type"] == et]
            ref_w, ref_hit = _match(rs, o, start, end)
            rol_w, rol_hit = _match(os_, r, start, end)
            recall = ref_hit.mean() if len(ref_hit) else np.nan
            precision = rol_hit.mean() if len(rol_hit) else np.nan
            lines.append(
                f"| {subset} | {et} | {len(ref_w):,} | {recall:.2%} | "
                f"{len(rol_w):,} | {precision:.2%} |"
            )
            if et == "(all)":
                gate_values[subset] = (recall, precision)

    # Field agreement and delete-timestamp lag on matched named+labeled pairs.
    r = ref[ref["named_labeled"]]
    r = r[(r["event_timestamp"] >= start) & (r["event_timestamp"] < end)]
    o = rolled[
        (rolled["event_timestamp"] >= start - PAD)
        & (rolled["event_timestamp"] < end + PAD)
    ]
    pairs = r.merge(
        o, on = ["osm_id", "event_type"], suffixes = ("_ref", "_inc"),
    )
    lines += ["", "## Matched named + labeled pairs", ""]
    lines.append(f"- Pairs: {len(pairs):,}")
    for f in MATCH_FIELDS:
        a = pairs[f"{f}_ref"].fillna("").astype(str)
        b = pairs[f"{f}_inc"].fillna("").astype(str)
        lines.append(f"- `{f}` equal: {(a == b).mean():.2%}")
    same_geom = pairs["geometry_ref"].geom_equals(pairs["geometry_inc"])
    lines.append(f"- geometry identical: {same_geom.mean():.2%}")
    hd = pairs[pairs["event_type"] == "hard_delete"]
    if len(hd):
        lag = (
            hd["event_timestamp_inc"] - hd["event_timestamp_ref"]
        ).dt.total_seconds() / 3600
        q = lag.quantile([0, 0.5, 0.9, 1.0]).round(1).tolist()
        lines.append(
            f"- hard_delete timestamp lag (h), min/median/p90/max: {q}"
        )

    # Missed / extra named+labeled examples for inspection.
    ref_w, ref_hit = _match(ref[ref["named_labeled"]], rolled, start, end)
    rol_w, rol_hit = _match(rolled[rolled["named_labeled"]], ref, start, end)
    cols = ["osm_id", "event_type", "event_timestamp", "prior_name", "shared_label"]
    ref_w.loc[~ref_hit, cols].to_csv(out_dir / "backtest_missed.csv", index = False)
    rol_w.loc[~rol_hit, cols].to_csv(out_dir / "backtest_extra.csv", index = False)

    # History-vs-snapshot QA on both histories.
    snap = config.get_file_path(
        "snapshot_osm", "snapshot", custom_version = args.snapshot,
    )
    lines += ["", "## History vs. snapshot QA", "", f"Snapshot: `{snap}`", ""]
    for label, d, c in [
        ("reference (full)", ref_dir, None),
        ("rolled", out_dir, cov.coverage_end),
    ]:
        res = check_history_vs_snapshot(d, snap, c)
        lines.append(
            f"- {label}: {res['n_checked']:,} nodes checked, "
            f"{res['n_missing']:,} missing from history, timestamp match "
            f"{res['ts_match_rate']:.3%}, name match {res['name_match_rate']:.3%}"
        )

    recall, precision = gate_values["named + labeled"]
    passed = recall >= GATE and precision >= GATE
    lines += [
        "",
        f"## Gate (≥ {GATE:.0%} recall and precision, named + labeled): "
        f"**{'PASS' if passed else 'FAIL'}** "
        f"(recall {recall:.2%}, precision {precision:.2%})",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "backtest_report.md").write_text(report)
    print(report)
    print("BACKTEST " + ("PASS" if passed else "FAIL"))


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    parser.add_argument("--step", choices = ["roll", "compare", "all"], default = "all")
    parser.add_argument("--base", default = "20260724")
    parser.add_argument("--reference", default = "20260902")
    parser.add_argument("--snapshot", default = "20260902")
    parser.add_argument("--end-date", default = "2026-08-19")
    parser.add_argument("--out-version", default = "20260818_incremental_backtest")
    args = parser.parse_args()
    args.end_date = datetime.datetime.fromisoformat(args.end_date).replace(
        tzinfo = UTC,
    )

    config = Config("~/repos/openpois/config.yaml")
    out_dir = config.get_dir_path("osm_data", custom_version = args.out_version)
    if args.step in ("roll", "all"):
        _roll(args, config, out_dir)
    if args.step in ("compare", "all"):
        _compare(args, config, out_dir)


if __name__ == "__main__":
    main()
