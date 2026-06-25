#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
Taxonomy crosswalk between OSM tags and Overture Maps taxonomy.

Loads four CSV files that map OSM tag key/value pairs and Overture
(L0, L1) categories to a unified ``shared_label``, plus per-label
match radii and top-level OSM-key-to-Overture-L0 mappings.
"""
from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd


WILDCARD = "*"

# Bit flags for the Overture L0 categories.  Used by
# ``compute_osm_l0_bits`` / ``compute_overture_l0_bits`` for
# fast vectorised broad-match checks in type scoring. Backing
# dtype is uint16, leaving headroom past the 12 bits used here.
L0_BIT: dict[str, int] = {
    "arts_and_entertainment": 1,
    "food_and_drink": 2,
    "health_care": 4,
    "shopping": 8,
    "sports_and_recreation": 16,
    "services_and_business": 32,
    "lifestyle_services": 64,
    "community_and_government": 128,
    "cultural_and_historic": 256,
    "education": 512,
    "travel_and_transportation": 1024,
    "lodging": 2048,
}


# -----------------------------------------------------------------
# CSV loaders
# -----------------------------------------------------------------


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from the package data directory."""
    csv_path = (
        resources.files("openpois.conflation.data")
        .joinpath(filename)
    )
    with resources.as_file(csv_path) as p:
        return pd.read_csv(
            p, dtype = str, keep_default_na = False,
        )


def load_osm_crosswalk() -> pd.DataFrame:
    """Load the OSM taxonomy crosswalk CSV.

    Columns: ``osm_key, osm_value, shared_label``.
    """
    return _load_csv("taxonomy_crosswalk_openstreetmap.csv")


def build_osm_tag_filter_expressions(
    osm_crosswalk: pd.DataFrame,
) -> list[str]:
    """Build ``osmium tags-filter`` expressions from the OSM crosswalk.

    The crosswalk is the single source of truth for which OSM tags we
    care about. For each ``osm_key``:

    - If the crosswalk has a wildcard row (``osm_value == "*"``), the
      whole key is matched: ``nwr/<key>``.
    - Otherwise only the specific listed values are matched:
      ``nwr/<key>=<v1>,<v2>,...``.

    This keeps PBF ingest aligned with the taxonomy — we only pull
    elements whose (key, value) is actually mapped to a shared label
    (or a key-level wildcard), instead of every element carrying the
    key (which would drag in e.g. all ``landuse=*`` polygons).

    Returns:
        A list of ``nwr/...`` expression strings, one per key, ordered
        by key name. Pass these to ``filter_pbf`` /
        ``filter_history_pbf`` via ``tag_filter_exprs``.
    """
    exprs: list[str] = []
    for key, grp in osm_crosswalk.groupby("osm_key", sort = True):
        values = grp["osm_value"].tolist()
        if WILDCARD in values:
            exprs.append(f"nwr/{key}")
        else:
            specific = sorted(
                str(v) for v in values if v and v != WILDCARD
            )
            exprs.append(f"nwr/{key}={','.join(specific)}")
    return exprs


def load_overture_crosswalk() -> pd.DataFrame:
    """Load the Overture Maps taxonomy crosswalk CSV.

    Columns: ``overture_l0, overture_l1, overture_l2,
    shared_label``.
    """
    return _load_csv("taxonomy_crosswalk_overture_maps.csv")


def load_match_radii() -> pd.DataFrame:
    """Load the match-radii CSV.

    Columns: ``shared_label, match_radius_m``.
    """
    return _load_csv("match_radii.csv")


def load_top_level_matches() -> pd.DataFrame:
    """Load the top-level OSM-key ↔ Overture-L0 CSV.

    Columns: ``overture_l0, osm_key``.
    """
    return _load_csv("top_level_matches.csv")


# -----------------------------------------------------------------
# Shared-label assignment — OSM
# -----------------------------------------------------------------


def _build_osm_label_lookups(
    osm_crosswalk: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Build per-key label lookups and wildcard fallbacks."""
    specific = osm_crosswalk[
        osm_crosswalk["osm_value"] != WILDCARD
    ].copy()
    wildcards_df = osm_crosswalk[
        osm_crosswalk["osm_value"] == WILDCARD
    ]

    lookups: dict[str, pd.Series] = {}
    for key, grp in specific.groupby("osm_key"):
        lkp = grp.set_index("osm_value")["shared_label"]
        lookups[key] = lkp

    wildcards: dict[str, str] = {}
    for _, row in wildcards_df.iterrows():
        wildcards[row["osm_key"]] = row["shared_label"]

    return lookups, wildcards


def assign_osm_shared_label(
    gdf: pd.DataFrame,
    osm_crosswalk: pd.DataFrame,
    match_radii: pd.DataFrame,
    filter_keys: list[str],
    default_radius_m: float = 100.0,
    return_all: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[list[list[str]], list[list[float]]]
):
    """
    Assign shared taxonomy labels to each OSM POI.

    Two modes, selected by ``return_all``:

    * ``return_all=False`` (default) — produces a single label per row.
      Uses ``filter_keys`` in priority order (first non-null match wins),
      falling back to the per-key wildcard row if the specific value
      is not in the crosswalk. Returns ``(label, radius)`` as object /
      float64 ndarrays of length ``len(gdf)``. Unmatched rows have
      ``label == ""`` and ``radius == default_radius_m``. This is the
      path used by the conflation pipeline and snapshot model
      application.

    * ``return_all=True`` — produces zero or more labels per row,
      used by the model-training pipeline which duplicates
      observations across every applicable taxonomy category.

      Pass 1 (specific matches): for every ``filter_key``, every row
      whose value for that key is in the crosswalk receives that
      label. A row can collect multiple specific labels.

      Pass 2 (wildcard fallback): applied *only* to rows that had
      zero specific matches in pass 1. Within such a row, wildcard
      keys are walked in the order they appear in the crosswalk CSV
      (``_build_osm_label_lookups`` populates the ``wildcards`` dict
      via ``iterrows``, preserving CSV order via dict insertion
      order); the first wildcard key with a non-null/non-empty value
      wins and is the only wildcard label assigned.

      Returns ``(labels_per_row, radii_per_row)`` as lists of lists;
      each inner list has ``>=0`` entries and is de-duplicated (if
      two keys map to the same label, it appears once).
    """
    n = len(gdf)
    lookups, wildcards = _build_osm_label_lookups(osm_crosswalk)

    radii_dict: dict[str, float] = {}
    for _, row in match_radii.iterrows():
        radii_dict[row["shared_label"]] = float(
            row["match_radius_m"]
        )

    if not return_all:
        label = np.full(n, "", dtype = object)
        radius = np.full(n, default_radius_m, dtype = np.float64)
        matched = np.zeros(n, dtype = bool)

        for key in filter_keys:
            if key not in gdf.columns:
                continue

            col = gdf[key]
            has_value = col.notna() & (col != "") & ~matched
            if not has_value.any():
                continue

            eligible_idx = np.where(has_value)[0]
            eligible_vals = col.to_numpy()[eligible_idx]

            lkp = lookups.get(key)
            if lkp is not None:
                mapped_label = (
                    pd.Series(eligible_vals, dtype = str).map(lkp)
                )
                found = mapped_label.notna().to_numpy()
                pos = eligible_idx[found]
                labels_found = mapped_label.to_numpy()[found]
                label[pos] = labels_found
                radius[pos] = np.array(
                    [
                        radii_dict.get(lb, default_radius_m)
                        for lb in labels_found
                    ]
                )
                matched[pos] = True

                not_found = eligible_idx[~found]
            else:
                not_found = eligible_idx

            wildcard_label = wildcards.get(key)
            if wildcard_label is not None and len(not_found) > 0:
                label[not_found] = wildcard_label
                radius[not_found] = radii_dict.get(
                    wildcard_label, default_radius_m,
                )
                matched[not_found] = True

        return label, radius

    # --- return_all=True path ------------------------------------

    specific_frames: list[pd.DataFrame] = []
    for key in filter_keys:
        if key not in gdf.columns:
            continue
        lkp = lookups.get(key)
        if lkp is None:
            continue
        col = gdf[key]
        mask = col.notna() & (col != "")
        if not mask.any():
            continue
        eligible_idx = np.where(mask)[0]
        eligible_vals = col.to_numpy()[eligible_idx]
        mapped = pd.Series(eligible_vals, dtype = str).map(lkp)
        hit = mapped.notna().to_numpy()
        if not hit.any():
            continue
        specific_frames.append(
            pd.DataFrame(
                {
                    "row_idx": eligible_idx[hit],
                    "label": mapped.to_numpy()[hit],
                }
            )
        )

    rows_with_specific = np.zeros(n, dtype = bool)
    if specific_frames:
        specific_df = pd.concat(specific_frames, ignore_index = True)
        rows_with_specific[specific_df["row_idx"].to_numpy()] = True
    else:
        specific_df = pd.DataFrame(
            {"row_idx": pd.Series(dtype = np.int64),
             "label": pd.Series(dtype = object)},
        )

    # Pass 2: one wildcard per row at most, in CSV order.
    wildcard_frames: list[pd.DataFrame] = []
    wildcard_assigned = np.zeros(n, dtype = bool)
    for key, wildcard_label in wildcards.items():
        if key not in gdf.columns:
            continue
        col = gdf[key]
        mask = (
            col.notna()
            & (col != "")
            & ~rows_with_specific
            & ~wildcard_assigned
        )
        if not mask.any():
            continue
        eligible_idx = np.where(mask)[0]
        wildcard_frames.append(
            pd.DataFrame(
                {
                    "row_idx": eligible_idx,
                    "label": np.full(
                        len(eligible_idx), wildcard_label, dtype = object,
                    ),
                }
            )
        )
        wildcard_assigned[eligible_idx] = True

    if wildcard_frames:
        wildcard_df = pd.concat(wildcard_frames, ignore_index = True)
    else:
        wildcard_df = pd.DataFrame(
            {"row_idx": pd.Series(dtype = np.int64),
             "label": pd.Series(dtype = object)},
        )

    long_df = pd.concat([specific_df, wildcard_df], ignore_index = True)
    if len(long_df) == 0:
        empty_labels: list[list[str]] = [[] for _ in range(n)]
        empty_radii: list[list[float]] = [[] for _ in range(n)]
        return empty_labels, empty_radii

    long_df = long_df.drop_duplicates(subset = ["row_idx", "label"])
    long_df["radius"] = (
        long_df["label"]
        .map(radii_dict)
        .fillna(default_radius_m)
        .astype(np.float64)
    )

    grouped = long_df.groupby("row_idx").agg(
        labels = ("label", list),
        radii = ("radius", list),
    )
    labels_by_row = grouped["labels"].to_dict()
    radii_by_row = grouped["radii"].to_dict()

    labels_per_row = [labels_by_row.get(i, []) for i in range(n)]
    radii_per_row = [radii_by_row.get(i, []) for i in range(n)]
    return labels_per_row, radii_per_row


# -----------------------------------------------------------------
# Shared-label assignment — Overture
# -----------------------------------------------------------------


def assign_overture_shared_label(
    gdf: pd.DataFrame,
    overture_crosswalk: pd.DataFrame,
    match_radii: pd.DataFrame,
    default_radius_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign a ``shared_label`` and ``match_radius_m`` to each
    Overture POI using a 4-tier cascade from most to least specific.

    Tiers (applied in order, each only to unmatched rows):

    1. **(L0, L1, L2)** — crosswalk rows with all three populated.
    2. **(L0, L2)** — L1 empty in crosswalk; matches any L1.
    3. **(L0, L1)** — L2 empty in crosswalk; catch-all for an L1.
    4. **L0-only** — both L1 and L2 empty in crosswalk.

    Backward-compatible: if the GeoDataFrame has no
    ``taxonomy_l2`` column, tiers 1-2 produce no matches and
    behaviour falls back to the old (L0, L1) + L0 logic.

    Returns:
        (shared_label ndarray of object, match_radius_m ndarray of
        float)
    """
    n = len(gdf)
    label = np.full(n, "", dtype = object)
    radius = np.full(n, default_radius_m, dtype = np.float64)
    matched = np.zeros(n, dtype = bool)

    cw = overture_crosswalk.copy()
    has_l1 = cw["overture_l1"] != ""
    has_l2 = cw["overture_l2"] != ""

    # Build radius dict
    radii_dict: dict[str, float] = {}
    for _, row in match_radii.iterrows():
        radii_dict[row["shared_label"]] = float(
            row["match_radius_m"]
        )

    # -- Build lookup tables for each tier --------------------------

    # Tier 1: (L0, L1, L2) — all three populated
    t1 = cw[has_l1 & has_l2].copy()
    t1["_key"] = (
        t1["overture_l0"] + "|"
        + t1["overture_l1"] + "|"
        + t1["overture_l2"]
    )
    t1_lkp = (
        t1.drop_duplicates("_key")
        .set_index("_key")["shared_label"]
    )

    # Tier 2: (L0, L2) — L1 empty, L2 populated
    t2 = cw[~has_l1 & has_l2].copy()
    t2["_key"] = t2["overture_l0"] + "|" + t2["overture_l2"]
    t2_lkp = (
        t2.drop_duplicates("_key")
        .set_index("_key")["shared_label"]
    )

    # Tier 3: (L0, L1) — L1 populated, L2 empty
    t3 = cw[has_l1 & ~has_l2].copy()
    t3["_key"] = t3["overture_l0"] + "|" + t3["overture_l1"]
    t3_lkp = (
        t3.drop_duplicates("_key")
        .set_index("_key")["shared_label"]
    )

    # Tier 4: L0-only — both L1 and L2 empty
    t4 = cw[~has_l1 & ~has_l2].copy()
    t4_lkp = (
        t4.drop_duplicates("overture_l0")
        .set_index("overture_l0")["shared_label"]
    )

    # -- Extract columns from the data ------------------------------

    def _col(name: str) -> pd.Series:
        if name in gdf.columns:
            return gdf[name].fillna("").astype(str)
        return pd.Series("", index = gdf.index)

    l0 = _col("taxonomy_l0")
    l1 = _col("taxonomy_l1")
    l2 = _col("taxonomy_l2")

    # -- Helper to apply a tier ------------------------------------

    def _apply_tier(
        keys: pd.Series,
        lkp: pd.Series,
        mask: np.ndarray,
    ) -> None:
        if not mask.any() or lkp.empty:
            return
        mapped = keys[mask].map(lkp)
        hit = mapped.notna().to_numpy()
        idx = np.where(mask)[0][hit]
        labels = mapped.to_numpy()[hit]
        label[idx] = labels
        radius[idx] = np.array(
            [radii_dict.get(lb, default_radius_m) for lb in labels]
        )
        matched[idx] = True

    # -- Apply tiers in order --------------------------------------

    # Tier 1: (L0, L1, L2)
    _apply_tier(
        l0 + "|" + l1 + "|" + l2,
        t1_lkp,
        ~matched & (l0 != ""),
    )

    # Tier 2: (L0, L2) — ignores L1 in the data
    _apply_tier(
        l0 + "|" + l2,
        t2_lkp,
        ~matched & (l2 != ""),
    )

    # Tier 3: (L0, L1) — catch-all for an L1 group
    _apply_tier(
        l0 + "|" + l1,
        t3_lkp,
        ~matched & (l1 != ""),
    )

    # Tier 4: L0-only
    _apply_tier(
        l0,
        t4_lkp,
        ~matched & (l0 != ""),
    )

    return label, radius


# -----------------------------------------------------------------
# L0 bitmask helpers (for type scoring)
# -----------------------------------------------------------------


def compute_osm_l0_bits(
    gdf: pd.DataFrame,
    top_level_matches: pd.DataFrame,
) -> np.ndarray:
    """
    For each OSM POI, compute a uint16 bitmask encoding which
    Overture L0 categories it broadly matches.

    A non-null value in an OSM tag key (e.g. ``amenity``) sets the
    bit(s) for every L0 linked to that key via *top_level_matches*.
    For example, ``amenity`` maps to both ``arts_and_entertainment``
    (bit 1) and ``food_and_drink`` (bit 2), so any POI with a
    non-null ``amenity`` value gets ``1 | 2 = 3``.
    """
    # Build osm_key -> combined bit value
    key_bits: dict[str, int] = {}
    for _, row in top_level_matches.iterrows():
        osm_key = row["osm_key"]
        l0 = row["overture_l0"]
        bit = L0_BIT.get(l0, 0)
        key_bits[osm_key] = key_bits.get(osm_key, 0) | bit

    bits = np.zeros(len(gdf), dtype = np.uint16)
    for osm_key, bval in key_bits.items():
        if osm_key in gdf.columns:
            has_val = gdf[osm_key].notna() & (
                gdf[osm_key] != ""
            )
            bits[has_val] |= bval

    return bits


def compute_overture_l0_bits(
    l0_array: np.ndarray,
) -> np.ndarray:
    """
    For each Overture POI, compute a uint16 bitmask from its
    ``taxonomy_l0`` value.  Each POI has at most one L0 category,
    so a single bit is set.
    """
    bits = np.zeros(len(l0_array), dtype = np.uint16)
    for l0, bval in L0_BIT.items():
        mask = l0_array == l0
        bits[mask] = bval
    return bits
