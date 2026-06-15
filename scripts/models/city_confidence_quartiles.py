#!/usr/bin/env python
"""
Confidence quartiles by city x shared_label for cities over 100k population.

Uses the per-POI ``conf_mean`` written onto the rated OSM snapshot by the best
random-effects turnover model (``2026-06-05-nationwide-full``; ``conf_mean`` =
1 - p_change at the POI's age, for that POI's shared_label x MSA x urbanicity
cell). Each POI's ``model_group`` encodes ``"shared_label | msa_code |
urban_rural"``; the shared_label is the first field.

Steps
-----
1. Census places (cb_2023_us_place_500k) joined to 2020 decennial population,
   filtered to places with population > 100,000.
2. Each rated POI is assigned to a place by its centroid (point-in-polygon).
3. For each city, compute confidence quartiles (min, Q1, median, Q3, max) plus
   count and mean, both overall (all POIs in the city, ``shared_label =
   __ALL__``) and per shared_label.

Output: one row per city x shared_label (plus one __ALL__ row per city).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PROJ_DATA", str(Path(sys.prefix) / "share" / "proj"))
os.environ.setdefault("PROJ_LIB", str(Path(sys.prefix) / "share" / "proj"))

import numpy as np
import pandas as pd
import geopandas as gpd

CENSUS_DIR = Path("~/data/openpois/census_areas").expanduser()
PLACE_SHP = CENSUS_DIR / "cb_2023_us_place_500k.shp"
POP_CSV = CENSUS_DIR / "place_population_2020.csv"
RATED_SNAPSHOT = Path(
    "~/data/openpois/snapshots/osm/20260521/osm_snapshot_rated.parquet"
).expanduser()
POP_THRESHOLD = 100_000
OUT_CSV = Path(
    "~/data/openpois/osm_turnover_model/2026-06-05-nationwide-full/"
    "city_shared_label_confidence_quartiles.csv"
).expanduser()

ALL_LABEL = "__ALL__"


def load_big_cities() -> gpd.GeoDataFrame:
    """Census places with 2020 population > threshold, in EPSG:4326."""
    places = gpd.read_file(PLACE_SHP)[
        ["GEOID", "NAME", "NAMELSAD", "STUSPS", "STATE_NAME", "geometry"]
    ]
    pop = pd.read_csv(POP_CSV, dtype = {"place_geoid": str})
    pop["population"] = pd.to_numeric(pop["population"], errors = "coerce")
    places = places.merge(
        pop, left_on = "GEOID", right_on = "place_geoid", how = "inner"
    )
    big = places[places["population"] > POP_THRESHOLD].copy()
    big = big.to_crs("EPSG:4326")
    print(f"{len(big)} places with population > {POP_THRESHOLD:,}")
    return big


def quantile_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """count / min / Q1 / median / Q3 / max / mean of conf_mean per group."""
    g = df.groupby(group_cols, observed = True)["conf_mean"]
    out = g.agg(
        n_pois = "size",
        conf_min = "min",
        conf_q1 = lambda s: s.quantile(0.25),
        conf_median = "median",
        conf_q3 = lambda s: s.quantile(0.75),
        conf_max = "max",
        conf_mean = "mean",
    ).reset_index()
    return out


def main() -> None:
    big = load_big_cities()

    print(f"Reading rated snapshot from {RATED_SNAPSHOT} ...")
    pois = gpd.read_parquet(
        RATED_SNAPSHOT, columns = ["geometry", "conf_mean", "model_group"]
    )
    print(f"  {len(pois):,} POIs")

    # Assign every POI a representative point (building polygons -> centroid).
    pois["geometry"] = pois.geometry.centroid
    pois["shared_label"] = (
        pois["model_group"].str.split(" | ", regex = False).str[0]
    )
    pois = pois.drop(columns = "model_group")

    print("Spatial join POIs -> big cities ...")
    joined = gpd.sjoin(
        pois, big[["GEOID", "NAME", "STUSPS", "population", "geometry"]],
        how = "inner", predicate = "within",
    )
    print(f"  {len(joined):,} POIs fall within a >100k city")
    joined = pd.DataFrame(joined.drop(columns = "geometry"))

    city_cols = ["GEOID", "NAME", "STUSPS", "population"]

    # Per city x shared_label, plus an __ALL__ row per city (all POIs in city).
    by_label = quantile_table(joined, city_cols + ["shared_label"])
    overall = quantile_table(joined, city_cols)
    overall["shared_label"] = ALL_LABEL

    result = pd.concat([overall, by_label], ignore_index = True)
    result = result.rename(
        columns = {"GEOID": "city_geoid", "NAME": "city_name", "STUSPS": "state"}
    )
    # Order: __ALL__ first within each city, then labels by descending median.
    result["_is_all"] = (result["shared_label"] == ALL_LABEL).astype(int)
    result = result.sort_values(
        ["city_name", "state", "_is_all", "conf_median"],
        ascending = [True, True, False, False],
    ).drop(columns = "_is_all")

    cols = [
        "city_geoid", "city_name", "state", "population", "shared_label",
        "n_pois", "conf_min", "conf_q1", "conf_median", "conf_q3", "conf_max",
        "conf_mean",
    ]
    result = result[cols]
    OUT_CSV.parent.mkdir(parents = True, exist_ok = True)
    result.to_csv(OUT_CSV, index = False)
    print(f"\nWrote {len(result):,} rows to {OUT_CSV}")

    # Brief summary: highest / lowest median-confidence shared_label per city,
    # aggregated nationally for a quick read.
    lab = by_label.rename(columns = {"NAME": "city_name"})
    lab = lab[lab["n_pois"] >= 20]
    nat = (
        lab.groupby("shared_label")
        .apply(
            lambda d: np.average(d["conf_median"], weights = d["n_pois"]),
            include_groups = False,
        )
        .rename("wtd_median_conf")
        .reset_index()
        .sort_values("wtd_median_conf", ascending = False)
    )
    print("\nMost stable shared_labels (pop-weighted median conf, n>=20/city):")
    print(nat.head(8).to_string(index = False))
    print("\nLeast stable shared_labels:")
    print(nat.tail(8).to_string(index = False))


if __name__ == "__main__":
    main()
