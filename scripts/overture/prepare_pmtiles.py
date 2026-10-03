"""
Build overture_snapshot.pmtiles and overture_categories.json from the Overture
snapshot.

The site used to read Overture's own hosted archive (tiles.overturemaps.org).
Since 2026-10 that CloudFront distribution answers any request with an
openpois.org Referer with 403, so we tile the snapshot we already pull each
month: the US + territories footprint, filtered to the conflation taxonomy
allowlist. That is the same Overture data the conflated layer draws on.

Output is a multi-zoom PMTiles archive keyed by the config's
``publish.pmtiles`` block, like the OSM and conflated archives: z10 up to z14,
extended past z14 where the densest tiles still drop points. The snapshot
columns are renamed to short tile properties, and the LIST columns are
flattened, since FlatGeobuf cannot store lists.

The categories JSON drives the site's Overture filter panel: one group per
taxonomy L0, each listing the ``basic_category`` values filed under it, with
POI counts. Overture recommends ``basic_category`` for map filtering. A basic
category that occurs under more than one L0 is filed under its most common L0;
rows with no basic category are listed as a null key in their L0 group.
"""
import json

import pandas as pd
import pyarrow.parquet as pq
from config_versioned import Config

from openpois.io.pmtiles import build_pmtiles

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("snapshot_overture", "snapshot")
OUTPUT_PATH = config.get_file_path("snapshot_overture", "pmtiles")
CATEGORIES_PATH = config.get_file_path("snapshot_overture", "categories")

LAYER_NAME = config.get("publish", "pmtiles", "overture_layer_name")
PROPERTIES = config.get("publish", "pmtiles", "overture_properties")
MIN_ZOOM = config.get("publish", "pmtiles", "min_zoom")
MAX_ZOOM = config.get("publish", "pmtiles", "max_zoom")
DROP_STRATEGY = config.get("publish", "pmtiles", "drop_strategy")
EXTEND_ZOOMS = config.get("publish", "pmtiles", "extend_zooms_if_still_dropping")
OVERTURE_RELEASE = config.get("publish", "version_metadata", "overture_release")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _first(values):
    """First entry of a LIST cell, or None for a null or empty list."""
    if values is None or len(values) == 0:
        return None
    return values[0]


def _join_path(values):
    """Join a taxonomy hierarchy LIST cell into an 'a > b > c' path."""
    if values is None or len(values) == 0:
        return None
    return " > ".join(values)


def to_tile_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Rename snapshot columns to tile properties and flatten LIST columns."""
    return pd.DataFrame({
        "id": df["overture_id"],
        "name": df["overture_name"],
        "brand": df["brand_name"],
        "confidence": df["confidence"],
        "taxonomy_l0": df["taxonomy_l0"],
        "basic_category": df["basic_category"],
        "taxonomy_primary": df["taxonomy_primary"],
        "taxonomy_hierarchy": df["taxonomy_hierarchy"].map(_join_path),
        "addr_street": df["overture_addr_street"],
        "addr_city": df["overture_addr_city"],
        "addr_state": df["overture_addr_state"],
        "website": df["overture_websites"].map(_first),
        "phone": df["overture_phones"].map(_first),
    })


def build_category_groups(input_parquet) -> dict:
    """Count POIs by (taxonomy L0, basic_category) for the site filter panel."""
    table = pq.read_table(input_parquet, columns = ["taxonomy_l0", "basic_category"])
    counts = (
        table.group_by(["taxonomy_l0", "basic_category"], use_threads = False)
        .aggregate([([], "count_all")])
        .to_pandas()
        .rename(columns = {"count_all": "n"})
    )
    if counts["taxonomy_l0"].isna().any():
        raise ValueError("Overture snapshot has rows with no taxonomy L0.")

    # File each basic category under its most common L0.
    named = counts[counts["basic_category"].notna()]
    home_l0 = (
        named.sort_values("n", ascending = False)
        .drop_duplicates("basic_category")
        .set_index("basic_category")["taxonomy_l0"]
    )
    named = (
        named.groupby("basic_category", as_index = False)["n"].sum()
        .assign(taxonomy_l0 = lambda d: d["basic_category"].map(home_l0))
    )
    unnamed = counts[counts["basic_category"].isna()]

    groups = []
    for l0 in sorted(set(named["taxonomy_l0"]) | set(unnamed["taxonomy_l0"])):
        members = named[named["taxonomy_l0"] == l0].sort_values("basic_category")
        categories = [
            {"key": row.basic_category, "count": int(row.n)}
            for row in members.itertuples()
        ]
        n_unnamed = int(unnamed.loc[unnamed["taxonomy_l0"] == l0, "n"].sum())
        if n_unnamed:
            categories.append({"key": None, "count": n_unnamed})
        groups.append({
            "l0": l0,
            "count": sum(c["count"] for c in categories),
            "categories": categories,
        })

    return {
        "overture_release": OVERTURE_RELEASE,
        "total": int(table.num_rows),
        "groups": groups,
    }


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Building Overture PMTiles from {INPUT_PATH}")
    print(f"  layer: {LAYER_NAME}")
    print(f"  zooms: Z{MIN_ZOOM}-z{MAX_ZOOM}")
    print(f"  drop:  --{DROP_STRATEGY}")
    print(f"  props: {', '.join(PROPERTIES)}")
    print(f"  -> {OUTPUT_PATH}")

    missing = set(PROPERTIES) - set(pq.read_schema(INPUT_PATH).names)
    if missing:
        raise ValueError(
            f"Overture snapshot lacks {sorted(missing)}; re-run "
            "scripts/overture/download.py with the current ingest query."
        )

    category_groups = build_category_groups(INPUT_PATH)
    CATEGORIES_PATH.write_text(json.dumps(category_groups, indent = 1))
    n_named = sum(
        1 for g in category_groups["groups"] for c in g["categories"] if c["key"]
    )
    print(
        f"Wrote {len(category_groups['groups'])} L0 groups, {n_named} basic "
        f"categories to {CATEGORIES_PATH}"
    )

    stats = build_pmtiles(
        input_parquet = INPUT_PATH,
        output_pmtiles = OUTPUT_PATH,
        layer_name = LAYER_NAME,
        properties = PROPERTIES,
        min_zoom = MIN_ZOOM,
        max_zoom = MAX_ZOOM,
        drop_strategy = DROP_STRATEGY,
        extend_zooms_if_still_dropping = EXTEND_ZOOMS,
        batch_transform = to_tile_properties,
    )

    print(
        f"Done. Wrote {stats['rows_written']:,} features, "
        f"{stats['pmtiles_bytes'] / 1e9:.2f} GB PMTiles."
    )
