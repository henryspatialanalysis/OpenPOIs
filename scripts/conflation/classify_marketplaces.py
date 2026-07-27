#!/usr/bin/env python
"""
Audit the ``amenity=marketplace`` name rules and record their exceptions.

OSM has no tag distinguishing a farmers market from a flea market or a
general/public market: ``marketplace=*`` is used on 7 US features, and the
wiki documents no companion tag. The name carries the signal instead, so
``openpois.conflation.taxonomy.classify_marketplace_name`` applies a set of
grower/produce regexes, defaulting to ``Market``.

This script checks those regexes against a second opinion and writes only
the names they get wrong to ``marketplace_name_labels.csv``, which the
taxonomy consults as an override. Keeping the file to genuine exceptions
means the rules stay the documentation of the policy, not a lookup table.

Passes:

1. Classify every distinct marketplace name in the snapshot with the
   production regexes.
2. Ask Claude to label the names the regexes left at the ``Market``
   default plus (with ``--audit-all``) the ones they called Farmers
   Market, so both misses and false positives surface.
3. Write the disagreements to the exceptions CSV. Rows marked
   ``source=manual`` are preserved untouched.

The LLM pass shells out to the ``claude`` CLI, which refuses to run inside
an existing Claude Code session; use ``--rules-only`` there, or run this
from a plain shell.

Config keys used (config.yaml):
    snapshot_osm.snapshot — input POI snapshot

Usage:
    python scripts/conflation/classify_marketplaces.py --rules-only  # report
    python scripts/conflation/classify_marketplaces.py               # + LLM
    python scripts/conflation/classify_marketplaces.py --audit-all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd
from config_versioned import Config

from openpois.conflation.taxonomy import (
    MARKETPLACE_CACHE_FILENAME,
    MARKETPLACE_DEFAULT_LABEL,
    MARKETPLACE_LABELS,
    classify_marketplace_name,
    load_marketplace_names,
    normalize_marketplace_name,
)

config = Config("~/repos/openpois/config.yaml")
SNAPSHOT_PATH = config.get_file_path("snapshot_osm", "snapshot")
CACHE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "openpois" / "conflation" / "data"
    / MARKETPLACE_CACHE_FILENAME
)

BATCH_SIZE = 60
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

PROMPT_HEADER = """\
You are labelling OpenStreetMap POIs tagged `amenity=marketplace` in the \
United States. For each name below, decide which category the place belongs \
to:

- "Farmers Market": a recurring market where growers sell farm goods \
directly — farmers markets, growers markets, green markets, produce \
markets, farm stands, CSA pickups, tailgate/curb markets.
- "Market": anything else — flea markets, swap meets, antique or craft \
vendor markets, public/city/night market halls, food halls, grocery or \
corner stores named "market", meat/fish markets, bazaars, malls.

Rules:
- Answer "Market" whenever you are unsure. It is the safe default.
- A name containing "market" alone is NOT enough to mean farmers market.

Return ONLY a JSON array of objects, one per input name, in the same order, \
each of the form {"name": "<name exactly as given>", "label": "<Farmers \
Market|Market>"}. No prose, no code fences.

Names:
"""


def load_names(snapshot_path: str) -> pd.DataFrame:
    """Read distinct normalized marketplace names + POI counts."""
    query = f"""
        SELECT name, count(*) AS n
        FROM '{snapshot_path}'
        WHERE amenity = 'marketplace'
          AND name IS NOT NULL AND name <> ''
        GROUP BY 1
    """
    raw = duckdb.connect().execute(query).fetch_df()
    raw["name_normalized"] = raw["name"].map(normalize_marketplace_name)
    raw = raw[raw["name_normalized"] != ""]
    return (
        raw.groupby("name_normalized", as_index = False)["n"]
        .sum()
        .sort_values("n", ascending = False)
        .reset_index(drop = True)
    )


def classify_with_claude(names: list[str]) -> dict[str, str]:
    """Label a batch of names via the ``claude`` CLI. Failures -> {}."""
    prompt = PROMPT_HEADER + "\n".join(f"- {n}" for n in names)
    try:
        result = subprocess.run(
            [
                "claude", "-p", prompt,
                "--model", CLAUDE_MODEL,
                "--output-format", "text",
            ],
            capture_output = True, text = True, timeout = 300, check = True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"    claude call failed: {exc}", file = sys.stderr)
        return {}
    except FileNotFoundError:
        print("    claude CLI not on PATH", file = sys.stderr)
        return {}

    text = result.stdout.strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        print("    no JSON array in response", file = sys.stderr)
        return {}
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        print(f"    bad JSON: {exc}", file = sys.stderr)
        return {}

    valid = set(names)
    out: dict[str, str] = {}
    for item in parsed:
        name = str(item.get("name", "")).strip()
        label = str(item.get("label", "")).strip()
        if name in valid and label in MARKETPLACE_LABELS:
            out[name] = label
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--rules-only", action = "store_true",
        help = "report rule coverage without calling the LLM",
    )
    parser.add_argument(
        "--audit-all", action = "store_true",
        help = "also re-check names the rules called Farmers Market "
        "(catches false positives, not just misses)",
    )
    parser.add_argument(
        "--dry-run", action = "store_true",
        help = "show the disagreements without writing the CSV",
    )
    args = parser.parse_args()

    print(f"Reading marketplace names from {SNAPSHOT_PATH} ...")
    names = load_names(str(SNAPSHOT_PATH))
    total_pois = int(names["n"].sum())
    print(f"  {len(names):,} distinct names over {total_pois:,} named POIs")

    names["rule_label"] = names["name_normalized"].map(
        classify_marketplace_name
    )
    by_rule = names.groupby("rule_label")["n"].sum()
    print("\nRules alone:")
    for label, n in by_rule.items():
        print(f"  {label:<16} {n:,} POIs")

    exceptions = load_marketplace_names()
    manual = exceptions[exceptions["source"] == "manual"]
    print(f"\nException rows on file: {len(exceptions):,} "
          f"({len(manual):,} manual)")

    to_audit = list(
        names.loc[
            names["rule_label"] == MARKETPLACE_DEFAULT_LABEL,
            "name_normalized",
        ]
    )
    if args.audit_all:
        to_audit += list(
            names.loc[
                names["rule_label"] != MARKETPLACE_DEFAULT_LABEL,
                "name_normalized",
            ]
        )
    already = set(exceptions["name_normalized"])
    to_audit = [n for n in to_audit if n not in already]
    print(f"Names to audit against the LLM: {len(to_audit):,}")

    if args.rules_only or not to_audit:
        print("\nNo LLM pass (--rules-only or nothing to audit).")
        return

    rule_of = dict(
        zip(names["name_normalized"], names["rule_label"])
    )
    disagreements: list[dict[str, str]] = []
    n_batches = (len(to_audit) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Sending {n_batches} batches to {CLAUDE_MODEL} ...")
    for i in range(0, len(to_audit), BATCH_SIZE):
        batch = to_audit[i:i + BATCH_SIZE]
        labels = classify_with_claude(batch)
        hits = [
            (name, label) for name, label in labels.items()
            if label != rule_of.get(name)
        ]
        print(
            f"  batch {i // BATCH_SIZE + 1}/{n_batches}: "
            f"{len(labels)}/{len(batch)} labelled, {len(hits)} disagree"
        )
        for name, label in hits:
            disagreements.append(
                {
                    "name_normalized": name,
                    "shared_label": label,
                    "source": "llm",
                }
            )

    print(f"\n{len(disagreements):,} names where the LLM overrides the rules:")
    for row in disagreements:
        print(
            f"  {row['name_normalized']:<50} "
            f"{rule_of.get(row['name_normalized'])} -> {row['shared_label']}"
        )

    if args.dry_run:
        print(f"\n[dry run] would write {len(disagreements):,} rows")
        return

    if disagreements:
        merged = pd.concat(
            [exceptions, pd.DataFrame(disagreements)], ignore_index = True,
        )
        # Existing rows (including every manual correction) win.
        merged = merged.drop_duplicates(subset = ["name_normalized"])
        merged = merged.sort_values("name_normalized").reset_index(drop = True)
        merged.to_csv(
            CACHE_PATH, index = False, lineterminator = "\n",
            encoding = "utf-8",
        )
        print(f"\nWrote {len(merged):,} exception rows to {CACHE_PATH}")
    else:
        print("\nRules agree with the LLM everywhere — nothing to write.")


if __name__ == "__main__":
    main()
