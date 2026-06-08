"""
Build a small national-scale modelling fixture for fast model iteration.

Subsets the enriched ``osm_observations.parquet`` to a handful of Metropolitan
Statistical Areas and amenity types, sampling whole POIs (all of a POI's
interval rows stay together, so the holdout CV in
:mod:`openpois.models.metrics` is leak-free) down to roughly ``TARGET_ROWS``.
The result is written to the ``testing`` directory and reused by the model +
metrics test suite.

Usage:
    python scripts/exploratory/build_test_dataset.py

Config keys used:
    directories.osm_data.osm_observations   — enriched source observations
    directories.testing.test_observations   — output fixture
"""
from __future__ import annotations

import duckdb
import pandas as pd
from config_versioned import Config


config = Config("~/repos/openpois/config.yaml")

OBSERVATIONS_PATH = config.get_file_path("osm_data", "osm_observations")
OUT_PATH = config.get_file_path("testing", "test_observations")

# Five large Metropolitan Statistical Areas (CBSA GEOIDs): New York, Chicago,
# Los Angeles, Seattle, Washington DC. Chosen for high POI counts across all
# amenity types so the fixture exercises the MSA + interaction terms.
TEST_MSAS = ["35620", "16980", "31080", "42660", "47900"]

# Ten common shared taxonomy labels.
TEST_LABELS = [
    "Park", "Place of Worship", "Fast Food", "Other Amenity", "Restaurant",
    "School", "Other Shop", "Cemetery", "Gas Station", "Hotel",
]

TARGET_ROWS = 10_000
SEED = 20260605


def build() -> pd.DataFrame:
    msa_in = ", ".join(f"'{m}'" for m in TEST_MSAS)
    label_in = ", ".join("'" + lbl.replace("'", "''") + "'" for lbl in TEST_LABELS)
    con = duckdb.connect()
    # Candidate rows in the chosen MSAs x labels.
    subset = con.execute(
        f"""
        SELECT *
        FROM read_parquet('{OBSERVATIONS_PATH.as_posix()}')
        WHERE msa_code IN ({msa_in})
          AND shared_label IN ({label_in})
        """
    ).fetch_df()
    con.close()
    print(f"Candidate rows in {len(TEST_MSAS)} MSAs x {len(TEST_LABELS)} labels: "
          f"{len(subset):,}")

    # Sample whole POIs (unique id) until ~TARGET_ROWS rows are collected.
    rng = pd.Series(subset["id"].unique()).sample(frac = 1.0, random_state = SEED)
    rows_per_id = subset.groupby("id").size()
    cum = rows_per_id.reindex(rng.to_numpy()).cumsum()
    keep_ids = cum[cum <= TARGET_ROWS].index
    if len(keep_ids) == 0:  # first POI already exceeds target
        keep_ids = rng.to_numpy()[:1]
    out = subset[subset["id"].isin(keep_ids)].reset_index(drop = True)
    return out


if __name__ == "__main__":
    out = build()
    OUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    out.to_parquet(OUT_PATH, index = False)
    print(f"Wrote {len(out):,} rows ({out['id'].nunique():,} POIs) to {OUT_PATH}")
    print("\nrows per MSA:")
    print(out["msa_code"].value_counts().to_string())
    print("\nrows per shared_label:")
    print(out["shared_label"].value_counts().to_string())
    print("\nrows per urban_rural:")
    print(out["urban_rural"].value_counts().to_string())
    print(f"\noverall change rate: {out['changed'].mean():.3f}")
