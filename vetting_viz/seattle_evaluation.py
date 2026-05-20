#!/usr/bin/env python
"""
Evaluate a change-detection run against a hand-vetted Seattle CSV.

Joins the new CD parquet's ``shadow_*`` columns to the vetted-truth
CSV by ``unified_id`` and prints:

- A confusion matrix (vetted truth vs. CD's "demoted-or-not" decision).
- Precision / recall / FPR vs the baseline CD output.
- Which previously-demoted POIs are now suppressed, and which previously-
  preserved ones became demoted (rare but worth flagging).

Standalone evaluation tool — not part of the production pipeline.

Usage:
    python vetting_viz/seattle_evaluation.py \
        --vetted /path/to/vetted_pois.csv \
        --baseline ~/data/openpois/conflation/20260423/conflated_baseline.parquet \
        --cd       ~/data/openpois/conflation/20260423/conflated_cd_v2.parquet \
        [--prior-cd ~/data/openpois/conflation/20260423/conflated_cd.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_demoted_set(conflated_path: Path) -> pd.DataFrame:
    """Return the set of Overture-source rows demoted in this run.

    A row is considered demoted iff ``shadow_matched`` is True (i.e.,
    the change-detection pass applied a penalty to its conf_mean).
    """
    df = pd.read_parquet(
        conflated_path,
        columns = [
            "unified_id", "source", "conf_mean", "original_conf_mean",
            "shadow_matched", "shadow_event_type", "shadow_ghost_id",
        ],
    )
    return df[df["shadow_matched"].fillna(False)][
        [
            "unified_id", "conf_mean", "original_conf_mean",
            "shadow_event_type", "shadow_ghost_id",
        ]
    ].reset_index(drop = True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description = (
            "Score a change-detection run against the hand-vetted "
            "Seattle CSV."
        )
    )
    parser.add_argument(
        "--vetted", required = True,
        help = (
            "Hand-vetted CSV (vetted column with values "
            "Unvetted/True drop/False drop)."
        ),
    )
    parser.add_argument(
        "--cd", required = True,
        help = "Change-detection conflated parquet (the new run).",
    )
    parser.add_argument(
        "--prior-cd", default = None,
        help = (
            "Optional: previous CD run. When set, prints which rows "
            "the new suppression rules removed from the demoted set "
            "and which rows became newly demoted."
        ),
    )
    args = parser.parse_args()

    vetted_path = Path(args.vetted).expanduser()
    cd_path = Path(args.cd).expanduser()

    print(f"Vetted CSV: {vetted_path}")
    print(f"New CD:     {cd_path}")
    vetted = pd.read_csv(vetted_path)
    cd_demoted = _load_demoted_set(cd_path)

    print(
        f"  Vetted rows: {len(vetted):,}  "
        f"(True drop={sum(vetted['vetted']=='True drop')}, "
        f"False drop={sum(vetted['vetted']=='False drop')}, "
        f"Unvetted={sum(vetted['vetted']=='Unvetted')})"
    )
    print(f"  New-run demoted rows: {len(cd_demoted):,}")

    merged = vetted.merge(
        cd_demoted[["unified_id"]].assign(new_demoted = True),
        on = "unified_id", how = "left",
    )
    merged["new_demoted"] = merged["new_demoted"].fillna(False)

    # Confusion matrix vs vetted truth (only on reviewed rows).
    reviewed = merged[merged["vetted"].isin(["True drop","False drop"])]
    n_td = (reviewed["vetted"] == "True drop").sum()
    n_fd = (reviewed["vetted"] == "False drop").sum()
    tp = ((reviewed["vetted"] == "True drop") & reviewed["new_demoted"]).sum()
    fp = ((reviewed["vetted"] == "False drop") & reviewed["new_demoted"]).sum()
    fn = ((reviewed["vetted"] == "True drop") & ~reviewed["new_demoted"]).sum()
    tn = ((reviewed["vetted"] == "False drop") & ~reviewed["new_demoted"]).sum()

    print()
    print("=== Confusion matrix on vetted-reviewed rows ===")
    print(f"  True positive (TD kept):    {tp:>4}")
    print(f"  False positive (FD kept):   {fp:>4}")
    print(f"  False negative (TD lost):   {fn:>4}")
    print(f"  True negative (FD removed): {tn:>4}")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / n_fd if n_fd else float("nan")
    print(
        f"  Precision: {precision:.3f}  "
        f"Recall: {recall:.3f}  "
        f"FPR over vetted FDs: {fpr:.3f}"
    )

    # Delta vs prior CD run, if provided.
    if args.prior_cd:
        prior_path = Path(args.prior_cd).expanduser()
        print()
        print(f"Prior CD: {prior_path}")
        prior_demoted = _load_demoted_set(prior_path)
        prior_ids = set(prior_demoted["unified_id"])
        new_ids = set(cd_demoted["unified_id"])

        removed = prior_ids - new_ids
        added = new_ids - prior_ids

        print(f"  Demoted in prior but NOT new (suppressed): {len(removed)}")
        print(f"  Demoted in new but NOT prior (newly added): {len(added)}")

        # Cross-reference suppressed rows with the vetting verdict.
        suppressed = vetted[vetted["unified_id"].isin(removed)]
        if len(suppressed):
            br = suppressed["vetted"].value_counts().to_dict()
            print(f"  Suppressed vetted breakdown: {br}")

        added_v = vetted[vetted["unified_id"].isin(added)]
        if len(added_v):
            br = added_v["vetted"].value_counts().to_dict()
            print(f"  Newly-added vetted breakdown: {br}")


if __name__ == "__main__":
    main()
