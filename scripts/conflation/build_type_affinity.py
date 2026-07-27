#!/usr/bin/env python
"""
Build the type-affinity table used by the conflation matcher's type score.

Replaces a hand-assigned "exact match = 1.0, same L0 = 0.5, else 0" rule with a
derived similarity between every (OSM label, Overture label) pair. See
docs/type-affinity-metric.md for the derivation, the alternatives considered,
and references.

The score blends two independent estimates:

1. **Hierarchy prior** — Lin (1998) similarity over the Overture category tree,
   with information content estimated from this snapshot's own category
   frequencies. Dense: defined for every pair of labels that appear on the
   Overture side.
2. **Empirical term** — a row-normalised confusion matrix built from
   identifier-confirmed matched pairs (exact website or phone agreement).
   Sparse, but measures the quantity we actually care about: when two records
   are genuinely the same place, what does each source call it?

    S(a, b) = ( n(a,b) * S_emp(a,b) + k * S_lin(a,b) ) / ( n(a,b) + k )

Neither term suffices alone. The hierarchy scores Gas Station <-> Convenience
Store at 0 (different L0s) though hundreds of confirmed matches show they are
co-located; the empirical term gives Hardware <-> Furniture Store a meaningless
value off 7 observations. `k` is the number of observations at which they carry
equal weight.

The empirical term is calibrated from a *previous* conflation output, so run N
produces the table for run N+1. Not circular: confirmed pairs are selected by
website/phone agreement, which the type score plays no part in.

Config keys used (config.yaml):
    snapshot_overture.snapshot  — category frequencies for the prior
    conflation.conflated        — source of identifier-confirmed pairs
    conflation.type_affinity_k  — shrinkage constant

Output:
    src/openpois/conflation/data/type_affinity.csv
        osm_label, overture_label, affinity, s_lin, s_emp, n_confirmed

Usage:
    python scripts/conflation/build_type_affinity.py
    python scripts/conflation/build_type_affinity.py --k 500 --dry-run
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd
from config_versioned import Config

from openpois.conflation.taxonomy import (
    EXCLUDE_LABEL,
    assign_overture_shared_label,
    load_match_radii,
    load_overture_crosswalk,
)

config = Config("~/repos/openpois/config.yaml")
OVERTURE_PATH = config.get_file_path("snapshot_overture", "snapshot")
CONFLATED_PATH = config.get_file_path("conflation", "conflated")
OUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "openpois" / "conflation" / "data" / "type_affinity.csv"
)
LEVELS = ["l0", "l1", "l2", "l3"]


def _tuple_census(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Distinct (L0..L3) tuples with POI counts and their shared_label."""
    tup = con.execute(f"""
        SELECT coalesce(taxonomy_l0,'') l0, coalesce(taxonomy_l1,'') l1,
               coalesce(taxonomy_l2,'') l2, coalesce(taxonomy_l3,'') l3,
               count(*) n
        FROM '{OVERTURE_PATH}' GROUP BY 1,2,3,4
    """).fetch_df()
    frame = pd.DataFrame(
        {f"taxonomy_{lvl}": tup[lvl].tolist() for lvl in LEVELS}
    )
    labels, _ = assign_overture_shared_label(
        frame, load_overture_crosswalk(), load_match_radii(),
    )
    tup["label"] = [str(x) for x in labels]
    return tup


def _build_prior(tup: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Lin similarity between every pair of Overture-side labels."""
    total = int(tup.n.sum())

    # Every path prefix is a tree node; its count is its subtree's POI total.
    node_count: dict[tuple, int] = defaultdict(int)
    for row in tup.itertuples():
        path = tuple(x for x in (row.l0, row.l1, row.l2, row.l3) if x)
        for depth in range(len(path) + 1):
            node_count[path[:depth]] += int(row.n)

    def ic(node: tuple) -> float:
        count = node_count.get(node, 0)
        return -math.log(count / total) if count > 0 else 0.0

    ic_cache = {node: ic(node) for node in node_count}

    label_paths: dict[str, list[tuple]] = defaultdict(list)
    for row in tup.itertuples():
        if not row.label or row.label == EXCLUDE_LABEL:
            continue
        path = tuple(x for x in (row.l0, row.l1, row.l2, row.l3) if x)
        label_paths[row.label].append(path)

    def lin(path_a: tuple, path_b: tuple) -> float:
        denom = ic_cache.get(path_a, 0.0) + ic_cache.get(path_b, 0.0)
        if denom <= 0:
            return 0.0
        shared = []
        for x, y in zip(path_a, path_b):
            if x != y:
                break
            shared.append(x)
        return 2.0 * ic_cache.get(tuple(shared), 0.0) / denom

    out: dict[tuple[str, str], float] = {}
    names = sorted(label_paths)
    for a in names:
        for b in names:
            # max over path pairs: a label spans several Overture paths, and
            # averaging drags identity below 1.0.
            best = 0.0
            for path_a in label_paths[a]:
                for path_b in label_paths[b]:
                    best = max(best, lin(path_a, path_b))
                    if best >= 1.0:
                        break
                if best >= 1.0:
                    break
            if best > 0:
                out[(a, b)] = best
    return out


def _build_empirical(
    con: duckdb.DuckDBPyConnection, tup: pd.DataFrame, conflated_path,
) -> pd.DataFrame:
    """Confusion counts over identifier-confirmed matched pairs."""
    # DuckDB 1.4.1 cannot register pandas' StringDtype columns (which
    # ``fetch_df`` now returns under pandas 3) — it raises "Data type 'str'
    # not recognized". Cast back to object first.
    con.register(
        "tuplab",
        tup[["l0", "l1", "l2", "l3", "label"]].astype(object),
    )
    return con.execute(f"""
        WITH pairs AS (
          SELECT cf.shared_label AS osm_label, t.label AS overture_label,
            lower(regexp_replace(coalesce(cf.osm_website,''),
                  '^https?://(www\\.)?|/$', '', 'g')) AS w_osm,
            lower(regexp_replace(
                  coalesce(list_extract(cf.overture_websites,1),''),
                  '^https?://(www\\.)?|/$', '', 'g')) AS w_ovt,
            regexp_replace(coalesce(cf.osm_phone,''), '[^0-9]', '', 'g') AS p_osm,
            regexp_replace(coalesce(list_extract(cf.overture_phones,1),''),
                  '[^0-9]', '', 'g') AS p_ovt
          FROM '{conflated_path}' cf
          JOIN '{OVERTURE_PATH}' ov USING (overture_id)
          JOIN tuplab t
            ON coalesce(ov.taxonomy_l0,'') = t.l0
           AND coalesce(ov.taxonomy_l1,'') = t.l1
           AND coalesce(ov.taxonomy_l2,'') = t.l2
           AND coalesce(ov.taxonomy_l3,'') = t.l3
          WHERE cf.source = 'matched'
        )
        SELECT osm_label, overture_label, count(*) AS n
        FROM pairs
        WHERE (w_osm <> '' AND w_osm = w_ovt)
           OR (length(p_osm) >= 10 AND right(p_osm,10) = right(p_ovt,10))
        GROUP BY 1, 2
    """).fetch_df()


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--k", type = float, default = None,
        help = "shrinkage constant; defaults to conflation.type_affinity_k",
    )
    parser.add_argument(
        "--conflated", default = None,
        help = "conflated.parquet to calibrate the empirical term from. "
        "Defaults to conflation.conflated, but that points at the run being "
        "*produced*; pass the previous run's output explicitly when bumping "
        "versions (run N calibrates the table for run N+1).",
    )
    parser.add_argument(
        "--dry-run", action = "store_true",
        help = "report without writing the CSV",
    )
    args = parser.parse_args()
    conflated_path = Path(args.conflated).expanduser() if args.conflated \
        else CONFLATED_PATH
    if not Path(conflated_path).exists():
        raise SystemExit(
            f"No conflated parquet at {conflated_path}. Pass --conflated "
            "with a previous run's output."
        )
    k = args.k if args.k is not None else float(
        config.get("conflation", "type_affinity_k")
    )

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'; SET threads=4;")

    print(f"Reading Overture categories from {OVERTURE_PATH} ...")
    tup = _tuple_census(con)
    print(f"  {len(tup):,} distinct tuples, {int(tup.n.sum()):,} POIs")

    print("Building hierarchy prior (Lin similarity over information content) ...")
    prior = _build_prior(tup)
    print(f"  {len(prior):,} label pairs with non-zero similarity")

    print(f"Reading identifier-confirmed pairs from {conflated_path} ...")
    emp = _build_empirical(con, tup, conflated_path)
    n_conf = int(emp.n.sum())
    print(f"  {n_conf:,} confirmed pairs over {len(emp):,} label combinations")

    row_max = emp.groupby("osm_label").n.max().to_dict()
    counts = {
        (r.osm_label, r.overture_label): int(r.n) for r in emp.itertuples()
    }

    rows = []
    for pair in sorted(set(prior) | set(counts)):
        a, b = pair
        s_lin = prior.get(pair, 0.0)
        n = counts.get(pair, 0)
        s_emp = n / row_max[a] if n and row_max.get(a) else 0.0
        blended = (n * s_emp + k * s_lin) / (n + k)
        # The empirical term may only ADD evidence, never subtract it.
        # S_emp measures frequency ("what does Overture usually call this?"),
        # not compatibility, so row-max normalisation punishes a rare but
        # exact agreement: OSM Tire Store <-> Overture Tire Store blends to
        # 0.493 while Tire Store <-> Car Repair reaches 0.979, and two sources
        # agreeing would score below two sources disagreeing. Flooring at the
        # hierarchy prior keeps semantic compatibility as a lower bound while
        # still letting confirmed co-occurrence lift pairs the tree misses
        # (Gas Station <-> Convenience Store, s_lin = 0).
        affinity = max(blended, s_lin)
        if affinity <= 0:
            continue
        rows.append(
            {
                "osm_label": a, "overture_label": b,
                "affinity": round(affinity, 4),
                "s_lin": round(s_lin, 4), "s_emp": round(s_emp, 4),
                "n_confirmed": n,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["osm_label", "affinity"], ascending = [True, False],
    )
    print(f"\nk = {k:g}; {len(out):,} non-zero pairs")
    identity = out[out.osm_label == out.overture_label]
    print(f"  identity pairs: {len(identity):,}, "
          f"mean affinity {identity.affinity.mean():.3f}")

    if args.dry_run:
        print("\n[dry run] not writing")
        return
    out.to_csv(OUT_PATH, index = False, lineterminator = "\n", encoding = "utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
