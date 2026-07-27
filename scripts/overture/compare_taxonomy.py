"""
Compare an Overture Places release's taxonomy to a prior local snapshot and the
conflation crosswalk, to flag schema/category drift before a monthly refresh.

Read-only. Scans only Overture's ``taxonomy`` column over S3 with a coarse
US-bbox predicate pushdown — it never downloads a full snapshot. Reuses the
DuckDB session config and S3-path helpers from ``openpois.io.overture``, the
coarse bboxes from ``openpois.io.boundary``, and the 6-tier crosswalk cascade
from ``openpois.conflation.taxonomy``.

Reports:
  1. Schema — the ``taxonomy`` struct shape, and presence of ``basic_category``
     and the deprecated ``categories`` field (removal slated ~Sept 2026).
  2. Category census — distinct (L0, L1, L2, L3) taxonomy tuples + counts in the
     new release's US footprint, split by whether they fall inside the
     download allowlist.
  3. Diff vs the prior local snapshot's distinct tuples (added / removed),
     within the ingested (allowlisted) scope.
  4. Crosswalk coverage — which ingested-scope release tuples the cascade leaves
     unmapped (dropped at ingest), highlighting tuples that are new this release.
  5. Allowlist coverage — new L0s / new L1s outside the current allowlist that
     we would miss at download time.
  6. Crosswalk liveness / tier usage — which crosswalk rows match no live
     tuple (stale paths), which are shadowed by a duplicate key, and how much
     of the data resolves only at the L0-only fallback tier. Report 4 cannot
     see this: the fallback rows label everything, so a crosswalk whose finer
     rows have all gone stale still reports full coverage.

``in_allowlist`` is recomputed from the current config on every run, so
``--from-census-csv`` works as a pre-flight gate for a widened allowlist —
edit ``taxonomy_allowlist``, re-run against the cached census, and see the
effect in seconds without re-scanning S3.

Usage:
    # Run BEFORE bumping config so the prior-snapshot default resolves to the
    # current (previous month's) snapshot.
    python scripts/overture/compare_taxonomy.py --release-date 2026-07-22.0

    # Pre-flight gate after editing the crosswalk and/or allowlist:
    python scripts/overture/compare_taxonomy.py --strict \
        --from-census-csv ~/data/openpois/logs/overture_taxonomy_census_*.csv

Config keys used (config.yaml):
    download.overture.s3_bucket / s3_region / release_date
    download.overture.taxonomy_allowlist
    download.overture.duckdb.memory_limit / threads
    download.general.boundary.*        — coarse US bboxes for pushdown
    directories.snapshot_overture      — prior snapshot default location
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
from config_versioned import Config

from openpois.conflation.taxonomy import (
    assign_overture_shared_label,
    load_match_radii,
    load_overture_crosswalk,
)
from openpois.io.boundary import get_us_pr_boundary
from openpois.io.overture import (
    _apply_duckdb_session_config,
    _build_bbox_predicate,
    _build_taxonomy_predicate,
    build_overture_s3_path,
    _list_overture_part_keys,
)

CONFIG_PATH = "~/repos/openpois/config.yaml"
LEVELS = ["l0", "l1", "l2", "l3"]


def _parse_args() -> argparse.Namespace:
    config = Config(CONFIG_PATH)
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--release-date",
        default = config.get(
            "download", "overture", "release_date", fail_if_none = False
        ),
        help = "Overture release to inspect (e.g. 2026-07-22.0). "
        "Defaults to config download.overture.release_date.",
    )
    parser.add_argument(
        "--prior-snapshot",
        default = None,
        help = "Path to the prior overture_snapshot.parquet to diff against. "
        "Defaults to the current config versions.snapshot_overture snapshot.",
    )
    parser.add_argument(
        "--min-count",
        type = int,
        default = 1,
        help = "Only report tuples with at least this many POIs (default 1).",
    )
    parser.add_argument(
        "--out-csv",
        default = None,
        help = "Optional path to write the full release census as CSV.",
    )
    parser.add_argument(
        "--from-census-csv",
        default = None,
        help = "Skip the S3 scan and load a previously written census CSV "
        "(schema check is skipped — it was done on the original scan). "
        "Use to resume the report without re-hitting S3.",
    )
    parser.add_argument(
        "--max-l0-fallback-share",
        type = float,
        default = 0.05,
        help = "Flag when more than this share of ingested POIs resolve "
        "only at the L0-only fallback tier (default 0.05).",
    )
    parser.add_argument(
        "--strict",
        action = "store_true",
        help = "Exit 1 on dead crosswalk rows, unmapped ingested tuples, "
        "shadowed duplicate keys, or an L0-fallback share over the "
        "threshold. Use as a pre-flight gate before a download.",
    )
    return parser.parse_args()


def _connect(config: Config) -> tuple[duckdb.DuckDBPyConnection, Path]:
    """Open a DuckDB connection configured like the Overture downloader."""
    region = config.get("download", "overture", "s3_region")
    memory_limit = (
        config.get(
            "download", "overture", "duckdb", "memory_limit",
            fail_if_none = False,
        )
        or "8GB"
    )
    temp_dir = Path(tempfile.mkdtemp(prefix = "compare_taxonomy_"))
    conn = duckdb.connect()
    _apply_duckdb_session_config(
        conn,
        s3_region = region,
        memory_limit = memory_limit,
        # Single global aggregation — give it a few threads.
        threads = 4,
        temp_directory = temp_dir,
    )
    return conn, temp_dir


def _report_schema(
    conn: duckdb.DuckDBPyConnection, one_part_uri: str,
) -> None:
    """DESCRIBE a single part and report the taxonomy struct + key fields."""
    described = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{one_part_uri}', "
        "hive_partitioning = 1)"
    ).fetchdf()
    cols = dict(zip(described["column_name"], described["column_type"]))

    print("=" * 78)
    print("1. SCHEMA CHECK")
    print("=" * 78)
    tax_type = cols.get("taxonomy")
    print(f"taxonomy column type:\n  {tax_type}")
    if tax_type and "hierarchy" in tax_type and "primary" in tax_type:
        print("  -> OK: taxonomy is still a struct with primary + hierarchy.")
    else:
        print("  -> WARNING: taxonomy struct shape changed — inspect before "
              "relying on hierarchy[1..4].")
    for field in ["basic_category", "categories"]:
        present = field in cols
        note = ""
        if field == "categories" and present:
            note = "  (deprecated; removal slated ~Sept 2026)"
        if field == "basic_category":
            note = "  (watch: predicted flat-field migration)"
        print(f"present? {field:<16} {present}{note}")
    print()


def _census(
    conn: duckdb.DuckDBPyConnection,
    parts_glob: str,
    bbox_predicate: str,
    allowlist_predicate: str,
) -> pd.DataFrame:
    """One S3 scan: distinct (L0..L3) tuples + counts + in_allowlist flag."""
    query = f"""
        SELECT
            taxonomy.hierarchy[1] AS l0,
            taxonomy.hierarchy[2] AS l1,
            taxonomy.hierarchy[3] AS l2,
            taxonomy.hierarchy[4] AS l3,
            ({allowlist_predicate}) AS in_allowlist,
            count(*) AS n
        FROM read_parquet('{parts_glob}', hive_partitioning = 1)
        WHERE {bbox_predicate}
        GROUP BY 1, 2, 3, 4, 5
    """
    df = conn.execute(query).fetchdf()
    return _normalize_census(df)


def _normalize_census(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce level columns to strings and in_allowlist to a plain bool.

    A NULL ``hierarchy[1]`` makes the allowlist predicate evaluate to NULL
    (SQL three-valued logic); such a POI is simply not in the allowlist, so
    NA -> False. Without this, pandas ``pd.NA`` propagates into boolean masks
    and ``groupby(...).max()`` and raises "boolean value of NA is ambiguous".
    """
    for lvl in LEVELS:
        df[lvl] = df[lvl].fillna("").astype(str)
    df["in_allowlist"] = (
        df["in_allowlist"]
        .map({True: True, "True": True, False: False, "False": False})
        .fillna(False)
        .astype(bool)
    )
    return df


def _prior_tuples(prior_path: Path) -> pd.DataFrame:
    """Distinct (L0..L3) tuples + counts from a prior local snapshot."""
    con = duckdb.connect()
    cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{prior_path.as_posix()}')"
    ).fetchdf()["column_name"].tolist()
    # Older snapshots may lack taxonomy_l3.
    l3_sel = "taxonomy_l3 AS l3" if "taxonomy_l3" in cols else "'' AS l3"
    df = con.execute(
        f"""
        SELECT taxonomy_l0 AS l0, taxonomy_l1 AS l1, taxonomy_l2 AS l2,
               {l3_sel}, count(*) AS n
        FROM read_parquet('{prior_path.as_posix()}')
        GROUP BY 1, 2, 3, 4
        """
    ).fetchdf()
    con.close()
    for lvl in LEVELS:
        df[lvl] = df[lvl].fillna("")
    return df[LEVELS + ["n"]]


def _key(df: pd.DataFrame) -> pd.Series:
    return df[LEVELS].agg("|".join, axis = 1)


def _recompute_allowlist(
    census: pd.DataFrame, allowlist: list,
) -> pd.DataFrame:
    """Recompute ``in_allowlist`` from the config allowlist.

    A cached census CSV carries the flag as of the scan, so editing
    ``taxonomy_allowlist`` would silently invalidate it. Recomputing
    here — the predicate is pure (L0, L1) membership, with a null L1
    meaning "all of this L0" — lets ``--from-census-csv`` act as a
    pre-flight gate for a widened allowlist without re-hitting S3.
    """
    whole_l0 = {e[0] for e in allowlist if e[1] is None}
    pairs = {(e[0], e[1]) for e in allowlist if e[1] is not None}
    out = census.copy()
    out["in_allowlist"] = [
        l0 in whole_l0 or (l0, l1) in pairs
        for l0, l1 in zip(out["l0"], out["l1"])
    ]
    return out


# Each crosswalk tier is identified by which level columns it populates,
# mirroring the cascade order in ``assign_overture_shared_label``.
TIERS: list[tuple[int, str, tuple[str, ...]]] = [
    (1, "L0+L1+L2+L3", ("l0", "l1", "l2", "l3")),
    (2, "L0+L3 (leaf)", ("l0", "l3")),
    (3, "L0+L1+L2", ("l0", "l1", "l2")),
    (4, "L0+L2 (leaf)", ("l0", "l2")),
    (5, "L0+L1", ("l0", "l1")),
    (6, "L0 only (fallback)", ("l0",)),
]
TIERS_BY_NUM = [(tier, cols) for tier, _name, cols in TIERS]


def _tier_of_row(row: pd.Series) -> int:
    """Classify a crosswalk row by which levels it populates."""
    has1, has2, has3 = bool(row["overture_l1"]), bool(
        row["overture_l2"]
    ), bool(row["overture_l3"])
    if has3:
        return 1 if (has1 and has2) else 2
    if has2:
        return 3 if has1 else 4
    return 5 if has1 else 6


def _crosswalk_labels(tuples: pd.DataFrame) -> pd.DataFrame:
    """Run the production 6-tier cascade over distinct tuples."""
    crosswalk = load_overture_crosswalk()
    radii = load_match_radii()
    frame = tuples.rename(
        columns = {lvl: f"taxonomy_{lvl}" for lvl in LEVELS}
    ).copy()
    labels, _ = assign_overture_shared_label(frame, crosswalk, radii)
    out = tuples.copy()
    out["shared_label"] = labels
    return out


def _resolve_by_tier(tuples: pd.DataFrame) -> pd.DataFrame:
    """Find, per tuple, the cascade tier that actually labels it.

    Runs the production assigner once per tier with only that tier's
    crosswalk rows: the lowest-numbered tier that produces a label is
    the one the full cascade uses, since each tier's lookup is built
    independently and applied in order.

    Returns ``tuples`` plus ``shared_label``, ``tier``, and
    ``tier_key`` (the ``|``-joined lookup key of the resolving tier,
    used to mark crosswalk rows live).
    """
    crosswalk = load_overture_crosswalk()
    crosswalk = crosswalk.assign(_tier = crosswalk.apply(_tier_of_row, axis = 1))
    radii = load_match_radii()
    frame = tuples.rename(
        columns = {lvl: f"taxonomy_{lvl}" for lvl in LEVELS}
    ).copy()

    out = tuples.copy()
    out["shared_label"] = ""
    out["tier"] = 0
    out["tier_key"] = ""
    for tier, _name, cols in TIERS:
        rows = crosswalk[crosswalk["_tier"] == tier]
        if rows.empty:
            continue
        labels, _ = assign_overture_shared_label(
            frame, rows.drop(columns = "_tier"), radii,
        )
        fresh = (out["tier"] == 0) & (labels != "")
        if not fresh.any():
            continue
        out.loc[fresh, "shared_label"] = labels[fresh.to_numpy()]
        out.loc[fresh, "tier"] = tier
        out.loc[fresh, "tier_key"] = (
            out.loc[fresh, list(cols)].agg("|".join, axis = 1)
        )
    return out


def main() -> None:
    args = _parse_args()
    config = Config(CONFIG_PATH)

    if not args.release_date:
        raise SystemExit(
            "No release date: pass --release-date or set "
            "download.overture.release_date in config."
        )

    prior_path = (
        Path(args.prior_snapshot).expanduser()
        if args.prior_snapshot
        else config.get_file_path("snapshot_overture", "snapshot")
    )

    bucket = config.get("download", "overture", "s3_bucket")
    allowlist = config.get("download", "overture", "taxonomy_allowlist")
    boundary_url = config.get(
        "download", "general", "boundary", "source_url"
    )
    coastline_buffer_m = config.get(
        "download", "general", "boundary", "coastline_buffer_m"
    )
    boundary_dir = config.get_dir_path("boundary")

    print(f"\nRelease under inspection : {args.release_date}")
    print(f"Prior snapshot baseline  : {prior_path}")
    print(f"Prior snapshot exists    : {prior_path.exists()}\n")

    _, coarse_bboxes = get_us_pr_boundary(
        source_url = boundary_url,
        cache_dir = boundary_dir,
        coastline_buffer_m = coastline_buffer_m,
    )
    bbox_predicate = _build_bbox_predicate(coarse_bboxes)
    allowlist_predicate = _build_taxonomy_predicate(
        [tuple(entry) for entry in allowlist]
    )

    if args.from_census_csv:
        src = Path(args.from_census_csv).expanduser()
        print(f"Loading census from {src} (skipping S3 scan + schema check).\n")
        census = _normalize_census(pd.read_csv(src, keep_default_na = False))
    else:
        part_keys = _list_overture_part_keys(
            release_date = args.release_date, bucket = bucket
        )
        parts_glob = build_overture_s3_path(args.release_date, bucket)
        one_part_uri = f"s3://{bucket}/{part_keys[0]}"

        conn, _ = _connect(config)
        try:
            _report_schema(conn, one_part_uri)
            print("Scanning release taxonomy over S3 (one column, bbox "
                  "pushdown; this takes a few minutes)...")
            census = _census(
                conn, parts_glob, bbox_predicate, allowlist_predicate
            )
        finally:
            conn.close()

    # Always recompute from the live config: a cached census carries the
    # flag as of its scan, so an allowlist edit would otherwise be invisible.
    census = _recompute_allowlist(census, allowlist)

    census = census[census["n"] >= args.min_count].copy()
    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser()
        out_csv.parent.mkdir(parents = True, exist_ok = True)
        census.sort_values("n", ascending = False).to_csv(
            out_csv, index = False
        )
        print(f"Wrote full census to {out_csv}\n")

    ingested = census[census["in_allowlist"]].copy()
    print("=" * 78)
    print("2. CATEGORY CENSUS (US footprint)")
    print("=" * 78)
    print(f"distinct (L0,L1,L2,L3) tuples total       : {len(census):,}")
    print(f"  ...within download allowlist (ingested) : {len(ingested):,}")
    print(f"POIs total (bbox scope)                   : "
          f"{int(census['n'].sum()):,}")
    print(f"  ...within allowlist                     : "
          f"{int(ingested['n'].sum()):,}")
    print("\ndistinct L0 categories in release:")
    l0_summary = (
        census.groupby("l0")
        .agg(n = ("n", "sum"), in_allowlist = ("in_allowlist", "max"))
        .sort_values("n", ascending = False)
    )
    for l0, row in l0_summary.iterrows():
        flag = "" if row["in_allowlist"] else "   <-- NOT in allowlist"
        print(f"  {l0:<28} {int(row['n']):>10,}{flag}")
    print()

    # --- Diff ingested scope vs prior snapshot -----------------------------
    print("=" * 78)
    print("3. DIFF vs PRIOR SNAPSHOT (ingested / allowlisted scope)")
    print("=" * 78)
    if not prior_path.exists():
        print("Prior snapshot not found — skipping diff.\n")
        prior = pd.DataFrame(columns = LEVELS + ["n"])
    else:
        prior = _prior_tuples(prior_path)
        prior_keys = set(_key(prior))
        ingested_keys = set(_key(ingested))
        added = ingested[~_key(ingested).isin(prior_keys)]
        removed = prior[~_key(prior).isin(ingested_keys)]
        print(f"tuples in release-ingested not in prior : {len(added):,}")
        print(f"tuples in prior not in release-ingested : {len(removed):,}\n")
        added_labeled = _crosswalk_labels(added).sort_values(
            "n", ascending = False
        )
        if len(added_labeled):
            print("ADDED tuples (new this release), with cascade label:")
            for _, r in added_labeled.iterrows():
                tup = " / ".join(x for x in
                                 [r.l0, r.l1, r.l2, r.l3] if x)
                lab = r["shared_label"] or "*** UNMAPPED (dropped) ***"
                print(f"  [{int(r['n']):>7,}] {tup}")
                print(f"            -> {lab}")
        if len(removed):
            print("\nREMOVED tuples (gone from ingested scope):")
            for _, r in removed.sort_values(
                "n", ascending = False
            ).iterrows():
                tup = " / ".join(x for x in
                                 [r.l0, r.l1, r.l2, r.l3] if x)
                print(f"  [{int(r['n']):>7,}] {tup}")
        print()

    # --- Crosswalk coverage of the ingested scope --------------------------
    print("=" * 78)
    print("4. CROSSWALK COVERAGE (ingested scope)")
    print("=" * 78)
    labeled = _crosswalk_labels(ingested)
    unmapped = labeled[labeled["shared_label"] == ""].copy()
    prior_keys = set(_key(prior)) if len(prior) else set()
    print(f"ingested tuples that map to a shared_label : "
          f"{int((labeled['shared_label'] != '').sum()):,}")
    print(f"ingested tuples UNMAPPED (dropped)         : {len(unmapped):,}")
    if len(unmapped):
        unmapped["is_new"] = ~_key(unmapped).isin(prior_keys)
        print("\nUNMAPPED tuples (POIs silently dropped at conflation):")
        for _, r in unmapped.sort_values("n", ascending = False).iterrows():
            tup = " / ".join(x for x in [r.l0, r.l1, r.l2, r.l3] if x)
            newflag = "  [NEW]" if r["is_new"] else ""
            print(f"  [{int(r['n']):>7,}] {tup}{newflag}")
    print()

    # --- New branches outside the allowlist --------------------------------
    print("=" * 78)
    print("5. ALLOWLIST COVERAGE — branches we do NOT ingest")
    print("=" * 78)
    outside = census[~census["in_allowlist"]].copy()
    by_l0l1 = (
        outside.groupby(["l0", "l1"])
        .agg(n = ("n", "sum"))
        .sort_values("n", ascending = False)
        .reset_index()
    )
    allow_l0_all = {e[0] for e in allowlist if e[1] is None}
    print("(L0, L1) branches present in the release but excluded from ingest,")
    print("sorted by POI count. A branch under an L0 we allowlist wholesale")
    print("(L1=null) but that still shows here would signal a predicate gap.\n")
    for _, r in by_l0l1.head(40).iterrows():
        warn = "  <-- under a wholesale-allowlisted L0!" \
            if r["l0"] in allow_l0_all else ""
        print(f"  [{int(r['n']):>8,}] {r['l0']} / {r['l1'] or '(none)'}{warn}")
    print()

    # --- Crosswalk liveness + tier usage -----------------------------------
    # §4 asks "does every tuple get a label?", which the L0-only fallback
    # rows answer "yes" to even when the finer rows have all gone stale.
    # This section asks the sharper questions: which crosswalk rows match
    # nothing in the release, and how much of the data is resolving only at
    # the catch-all tier.
    print("=" * 78)
    print("6. CROSSWALK LIVENESS / TIER USAGE (ingested scope)")
    print("=" * 78)
    resolved = _resolve_by_tier(ingested)
    total_pois = int(resolved["n"].sum())

    # Self-check: the per-tier resolution must agree with the full cascade.
    full = _crosswalk_labels(ingested)
    disagree = (
        full["shared_label"].to_numpy() != resolved["shared_label"].to_numpy()
    )
    if disagree.any():
        print(
            f"WARNING: tier resolution disagrees with the full cascade on "
            f"{int(disagree.sum())} tuple(s) — the tier patterns in this "
            f"script have drifted from assign_overture_shared_label."
        )

    print("POIs and tuples by resolving tier:\n")
    fallback_pois = 0
    for tier, name, _cols in TIERS:
        sel = resolved["tier"] == tier
        pois = int(resolved.loc[sel, "n"].sum())
        if tier == 6:
            fallback_pois = pois
        print(
            f"  tier {tier} {name:<20} {int(sel.sum()):>5,} tuples  "
            f"{pois:>12,} POIs  ({100 * pois / max(total_pois, 1):>5.2f}%)"
        )
    unresolved = resolved["tier"] == 0
    if unresolved.any():
        print(
            f"  unmatched              {int(unresolved.sum()):>5,} tuples  "
            f"{int(resolved.loc[unresolved, 'n'].sum()):>12,} POIs"
        )

    fallback_share = fallback_pois / max(total_pois, 1)
    print(
        f"\nL0-only fallback share: {100 * fallback_share:.2f}% "
        f"(threshold {100 * args.max_l0_fallback_share:.2f}%)"
    )
    over_threshold = fallback_share > args.max_l0_fallback_share
    if over_threshold:
        print(
            "  ^ ABOVE THRESHOLD — finer crosswalk rows are likely stale. "
            "The biggest offenders:"
        )
        worst = resolved[resolved["tier"] == 6].nlargest(15, "n")
        for _, r in worst.iterrows():
            tup = " / ".join(x for x in [r.l0, r.l1, r.l2, r.l3] if x)
            print(f"    [{int(r['n']):>8,}] {tup} -> {r['shared_label']}")

    # Row-level health. Two distinct conditions:
    #   stale  — the row names a value that does not exist at that level
    #            anywhere in the release. This is taxonomy drift (a rename
    #            or reparent) and the row can never match again.
    #   unused — the row is well-formed but never the resolving tier,
    #            because finer rows cover its subtree. Catch-all rows are
    #            deliberately in this state; it is informational only.
    crosswalk = load_overture_crosswalk()
    crosswalk = crosswalk.assign(
        _tier = crosswalk.apply(_tier_of_row, axis = 1)
    )
    live_keys = set(zip(resolved["tier"], resolved["tier_key"]))
    # Vocabulary of the whole release, per level — not just the ingested
    # scope, so widening the allowlist does not read as drift.
    vocab = {lvl: set(census[lvl]) - {""} for lvl in LEVELS}

    stale_rows = []
    unused_rows = []
    shadowed = []
    seen_keys: set[tuple[int, str]] = set()
    for _, row in crosswalk.iterrows():
        tier = int(row["_tier"])
        cols = dict(TIERS_BY_NUM)[tier]
        key = "|".join(row[f"overture_{c}"] for c in cols)
        entry = (tier, key, row["shared_label"])

        bad_levels = [
            f"{lvl}={row[f'overture_{lvl}']}"
            for lvl in LEVELS
            if row[f"overture_{lvl}"]
            and row[f"overture_{lvl}"] not in vocab[lvl]
        ]
        if bad_levels:
            stale_rows.append((*entry, ", ".join(bad_levels)))
            continue
        if (tier, key) in seen_keys:
            shadowed.append(entry)
            continue
        seen_keys.add((tier, key))
        if (tier, key) not in live_keys:
            unused_rows.append(entry)

    print(f"\ncrosswalk rows                  : {len(crosswalk):,}")
    print(f"STALE rows (value gone from release): {len(stale_rows):,}")
    for tier, key, label, why in stale_rows[:40]:
        print(
            f"    tier {tier}  {key.replace('|', ' / '):<58} -> "
            f"{label:<20} [{why}]"
        )
    if len(stale_rows) > 40:
        print(f"    ... and {len(stale_rows) - 40:,} more")
    print(f"shadowed rows (duplicate key)   : {len(shadowed):,}")
    for tier, key, label in shadowed[:20]:
        print(f"    tier {tier}  {key.replace('|', ' / '):<58} -> {label}")
    print(
        f"unused rows (covered by finer rows; catch-alls expected here): "
        f"{len(unused_rows):,}"
    )
    for tier, key, label in unused_rows[:20]:
        print(f"    tier {tier}  {key.replace('|', ' / '):<58} -> {label}")
    print()

    print("Done. Review §3 (added/removed), §4 (unmapped) and §6 (stale rows "
          "/ fallback share) before bumping the crosswalk.")

    if args.strict:
        problems = []
        if len(unmapped):
            problems.append(f"{len(unmapped):,} unmapped ingested tuples")
        if stale_rows:
            problems.append(f"{len(stale_rows):,} stale crosswalk rows")
        if shadowed:
            problems.append(f"{len(shadowed):,} shadowed crosswalk rows")
        if over_threshold:
            problems.append(
                f"L0-fallback share {100 * fallback_share:.2f}% over threshold"
            )
        if disagree.any():
            problems.append("tier/cascade self-check mismatch")
        if problems:
            raise SystemExit(
                "\nSTRICT CHECK FAILED: " + "; ".join(problems)
            )
        print("\nSTRICT CHECK PASSED.")


if __name__ == "__main__":
    main()
