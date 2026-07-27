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

import re
from importlib import resources

import numpy as np
import pandas as pd


WILDCARD = "*"

# Reserved ``shared_label`` sentinel marking an OSM (key, value) pair as a
# non-POI to be dropped: it never becomes a POI, never inherits the key's
# ``*`` wildcard label, and is hidden from the public taxonomy category list
# (shown instead in a dedicated "excluded tags" section). See
# ``get_osm_exclusions`` / ``drop_osm_exclusions``.
EXCLUDE_LABEL = "EXCLUDE"

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
    "geographic_entities": 4096,
}

# --- Marketplace name refinement ---------------------------------
# OSM has no tag separating farmers markets from flea/public/general
# markets (``marketplace=*`` is used on 7 US features and the wiki
# documents no companion tag), so ``amenity=marketplace`` is split on
# the name. The regexes below do the work; the CSV holds only the
# handful of names they get wrong (see
# ``scripts/conflation/classify_marketplaces.py``).
MARKETPLACE_KEY = "amenity"
MARKETPLACE_VALUE = "marketplace"
MARKETPLACE_CACHE_FILENAME = "marketplace_name_labels.csv"
# Derived (OSM label, Overture label) similarity used by the matcher's type
# score; see docs/type-affinity-metric.md.
TYPE_AFFINITY_FILENAME = "type_affinity.csv"
MARKETPLACE_DEFAULT_LABEL = "Market"
MARKETPLACE_LABELS = ("Farmers Market", "Market")

# Explicitly some other kind of market. Checked first: "antique farm
# equipment flea market" is a flea market, not a farmers market.
MARKETPLACE_RE_MARKET = re.compile(
    r"\bflea\b|\bswap\b|\bantique|\bbazaar\b|\btrade day|\bnight market\b"
    r"|\bpublic market\b|\bfish market\b|\bstreet market\b|\bmercado\b"
    r"|\bmeat market\b|\bstock ?yard|\blivestock\b|\bauction\b"
)
# Grower / produce vocabulary. ``farm`` is deliberately unanchored so it
# also catches farmstand, freshfarm and farmers'. ``grove`` is
# deliberately absent — Oak Grove, Elk Grove and friends are place
# names, not orchards.
MARKETPLACE_RE_FARMERS = re.compile(
    r"farm|\bgrowers?\b|\bgrown\b|greensgrow|\borchards?\b"
    r"|\bproduce\b|\bfruits?\b|\bvegetables?\b|\bveggies?\b|\bmelons?\b"
    r"|\bberr(y|ies)\b|sweet corn|\bcrops?\b|\bharvest\b"
    r"|\btailgate\b|\bcurb market\b|\bagricultur(e|al)\b|\bcsa\b"
    r"|\bgreen ?markets?\b|\bfresh for all\b|\bfamers?\b"
)


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

    ``EXCLUDE``-labelled rows are skipped when building value-scoped
    expressions for keys without a wildcard. (Keys *with* a wildcard
    still ingest the whole key, including excluded values, which are
    dropped post-parse by :func:`drop_osm_exclusions`.)

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
            not_excluded = grp[grp["shared_label"] != EXCLUDE_LABEL]
            specific = sorted(
                str(v) for v in not_excluded["osm_value"].tolist()
                if v and v != WILDCARD
            )
            exprs.append(f"nwr/{key}={','.join(specific)}")
    return exprs


def get_osm_exclusions(
    osm_crosswalk: pd.DataFrame,
) -> dict[str, set[str]]:
    """Return excluded OSM values grouped by key.

    Maps each ``osm_key`` to the set of ``osm_value`` strings whose
    ``shared_label`` is :data:`EXCLUDE_LABEL` — i.e. non-POI tags
    (parking, benches, picnic tables, …) that should never produce a
    POI or an ``Other <X>`` wildcard label.
    """
    excl = osm_crosswalk[
        osm_crosswalk["shared_label"] == EXCLUDE_LABEL
    ]
    out: dict[str, set[str]] = {}
    for key, grp in excl.groupby("osm_key"):
        out[key] = set(grp["osm_value"].tolist())
    return out


def drop_osm_exclusions(
    gdf: pd.DataFrame,
    osm_crosswalk: pd.DataFrame,
    filter_keys: list[str],
    match_radii: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Drop non-POI rows whose only taxonomy signal is an excluded tag.

    A row is dropped iff it carries at least one ``EXCLUDE``-labelled
    ``(key, value)`` tag *and* :func:`assign_osm_shared_label` resolves
    it to no label (``""``) — so a feature that also carries a genuine
    POI tag (e.g. a restaurant that happens to be tagged with a picnic
    table) is preserved, and rows that are unlabelled for unrelated
    reasons are left untouched.

    Returns a new GeoDataFrame with the offending rows removed and the
    index reset. ``match_radii`` is only needed to satisfy
    :func:`assign_osm_shared_label`; radius values are not used here.
    """
    exclusions = get_osm_exclusions(osm_crosswalk)
    if not exclusions:
        return gdf

    has_excluded = np.zeros(len(gdf), dtype = bool)
    for key, vals in exclusions.items():
        if key in gdf.columns:
            has_excluded |= gdf[key].isin(vals).to_numpy()
    if not has_excluded.any():
        return gdf

    if match_radii is None:
        match_radii = load_match_radii()
    labels, _ = assign_osm_shared_label(
        gdf, osm_crosswalk, match_radii, filter_keys,
    )
    drop = has_excluded & (labels == "")
    if not drop.any():
        return gdf
    return gdf.loc[~drop].reset_index(drop = True)


def load_overture_crosswalk() -> pd.DataFrame:
    """Load the Overture Maps taxonomy crosswalk CSV.

    Columns: ``overture_l0, overture_l1, overture_l2, overture_l3,
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


def load_marketplace_names() -> pd.DataFrame:
    """Load the marketplace name → label exceptions.

    Columns: ``name_normalized, shared_label, source``. Only names the
    regexes in :func:`classify_marketplace_name` get wrong live here,
    so the file stays short; an absent file just means no exceptions.
    """
    cols = ["name_normalized", "shared_label", "source"]
    try:
        return _load_csv(MARKETPLACE_CACHE_FILENAME)
    except FileNotFoundError:
        return pd.DataFrame({c: pd.Series(dtype = str) for c in cols})


def classify_marketplace_name(name_normalized: str) -> str:
    """Label a normalized marketplace name from the regexes alone.

    Returns :data:`MARKETPLACE_DEFAULT_LABEL` for anything that is not
    positively a farmers market — "market" on its own means nothing.
    """
    if not name_normalized:
        return MARKETPLACE_DEFAULT_LABEL
    if MARKETPLACE_RE_MARKET.search(name_normalized):
        return "Market"
    if MARKETPLACE_RE_FARMERS.search(name_normalized):
        return "Farmers Market"
    return MARKETPLACE_DEFAULT_LABEL


def load_type_affinity() -> pd.DataFrame:
    """Load the (OSM label, Overture label) type-affinity table.

    Columns: ``osm_label, overture_label, affinity, s_lin, s_emp,
    n_confirmed``. Built by ``scripts/conflation/build_type_affinity.py``;
    see docs/type-affinity-metric.md. Only non-zero pairs are stored, so an
    absent pair means affinity 0. An absent file yields an empty frame, and
    the matcher falls back to exact-match-only scoring.
    """
    cols = [
        "osm_label", "overture_label", "affinity",
        "s_lin", "s_emp", "n_confirmed",
    ]
    try:
        return _load_csv(TYPE_AFFINITY_FILENAME)
    except FileNotFoundError:
        return pd.DataFrame({c: pd.Series(dtype = str) for c in cols})


def build_affinity_matrix(
    affinity: pd.DataFrame,
) -> tuple[dict[str, int], np.ndarray]:
    """Compact the affinity table into a label index + dense matrix.

    Returns ``(label_to_idx, matrix)`` where ``matrix[i, j]`` is the affinity
    of OSM label *i* against Overture label *j*. Dense lookup keeps type
    scoring a single fancy-index into a small float array rather than a
    per-pair dict hit, which matters at ~10^8 candidate pairs.

    Identity is forced to 1.0 for any label present, so a label pair the
    table omits still scores full marks against itself.
    """
    labels = sorted(
        set(affinity["osm_label"]) | set(affinity["overture_label"])
    )
    idx = {lb: i for i, lb in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype = np.float32)
    for row in affinity.itertuples():
        i = idx.get(row.osm_label)
        j = idx.get(row.overture_label)
        if i is not None and j is not None:
            matrix[i, j] = float(row.affinity)
    if len(labels):
        np.fill_diagonal(matrix, 1.0)
    return idx, matrix


def normalize_marketplace_name(name: object) -> str:
    """Normalize a POI name for marketplace cache lookup.

    Lowercases, strips everything but ASCII alphanumerics, and collapses
    whitespace, so ``"Pike Place Farmers' Market"`` and ``"pike place
    farmers market"`` hit the same cache row.
    """
    if not isinstance(name, str):
        return ""
    lowered = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def refine_marketplace_labels(
    gdf: pd.DataFrame,
    label: np.ndarray,
    radius: np.ndarray,
    radii_dict: dict[str, float],
    default_radius_m: float,
    marketplace_names: pd.DataFrame | None = None,
) -> None:
    """Split ``amenity=marketplace`` rows into Market / Farmers Market.

    Mutates ``label`` and ``radius`` in place for rows the crosswalk
    resolved to :data:`MARKETPLACE_DEFAULT_LABEL` via
    ``amenity=marketplace`` — a marketplace that also carries a
    higher-priority tag (say ``shop=supermarket``) keeps that label.
    :func:`classify_marketplace_name` decides, overridden by the
    exceptions CSV where it is known to be wrong.
    """
    if MARKETPLACE_KEY not in gdf.columns:
        return
    is_market = (
        gdf[MARKETPLACE_KEY].to_numpy() == MARKETPLACE_VALUE
    ) & (label == MARKETPLACE_DEFAULT_LABEL)
    if not is_market.any():
        return

    if marketplace_names is None:
        marketplace_names = load_marketplace_names()
    lookup = dict(
        zip(
            marketplace_names["name_normalized"],
            marketplace_names["shared_label"],
        )
    )

    names = (
        gdf["name"].to_numpy()
        if "name" in gdf.columns
        else np.full(len(gdf), "", dtype = object)
    )
    for i in np.where(is_market)[0]:
        norm = normalize_marketplace_name(names[i])
        resolved = lookup.get(norm) or classify_marketplace_name(norm)
        label[i] = resolved
        radius[i] = radii_dict.get(resolved, default_radius_m)


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
                # EXCLUDE values match the crosswalk but are non-POI:
                # assign no label, skip the wildcard, and leave the row
                # unmatched so a lower-priority key can still label it.
                is_excl = (
                    mapped_label == EXCLUDE_LABEL
                ).to_numpy()
                found = mapped_label.notna().to_numpy() & ~is_excl
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

                # Only genuinely-unmapped values (NaN) fall to the
                # wildcard; excluded values are withheld from it.
                not_found = eligible_idx[
                    mapped_label.isna().to_numpy()
                ]
            else:
                not_found = eligible_idx

            wildcard_label = wildcards.get(key)
            if wildcard_label is not None and len(not_found) > 0:
                label[not_found] = wildcard_label
                radius[not_found] = radii_dict.get(
                    wildcard_label, default_radius_m,
                )
                matched[not_found] = True

        refine_marketplace_labels(
            gdf, label, radius, radii_dict, default_radius_m,
        )
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
        # Drop EXCLUDE matches: excluded tags are non-POI and must not
        # count as a specific match (nor fall to the pass-2 wildcard).
        hit = (
            mapped.notna().to_numpy()
            & (mapped.to_numpy() != EXCLUDE_LABEL)
        )
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
        # Withhold the wildcard from excluded values (non-POI tags).
        lkp = lookups.get(key)
        excl_mask = (
            (col.map(lkp) == EXCLUDE_LABEL)
            if lkp is not None
            else pd.Series(False, index = col.index)
        )
        mask = (
            col.notna()
            & (col != "")
            & ~rows_with_specific
            & ~wildcard_assigned
            & ~excl_mask
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

    # Same marketplace split as the single-label path, applied to the
    # exploded rows so model observations carry the refined label too.
    market_rows = long_df["label"] == MARKETPLACE_DEFAULT_LABEL
    if market_rows.any():
        lookup = dict(
            zip(
                (mk := load_marketplace_names())["name_normalized"],
                mk["shared_label"],
            )
        )
        names = (
            gdf["name"].to_numpy()
            if "name" in gdf.columns
            else np.full(n, "", dtype = object)
        )

        def _market_label(row_idx: int) -> str:
            norm = normalize_marketplace_name(names[row_idx])
            return lookup.get(norm) or classify_marketplace_name(norm)

        long_df.loc[market_rows, "label"] = (
            long_df.loc[market_rows, "row_idx"].map(_market_label)
        )

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
    Overture POI using a 6-tier cascade from most to least specific.

    The Overture ``taxonomy.hierarchy`` is a path from general (L0) to
    specific; we read up to four levels (``taxonomy_l0/l1/l2/l3`` =
    ``hierarchy[1..4]``). Crosswalk rows leave deeper columns blank to
    match a whole subtree, and an ``(L0, L3)`` row targets a single
    deep leaf regardless of its intermediate parents.

    Tiers (applied in order, each only to unmatched rows):

    1. **(L0, L1, L2, L3)** — all four populated; exact path.
    2. **(L0, L3)** — only L0 and L3 populated; a deep leaf, ignoring
       L1/L2 (e.g. ``health_care`` + ``speech_therapy``). Runs before
       the L2 tiers so a leaf wins over its container's catch-all.
    3. **(L0, L1, L2)** — L1 and L2 populated, L3 blank.
    4. **(L0, L2)** — only L0 and L2 populated; matches any L1.
    5. **(L0, L1)** — only L0 and L1 populated; catch-all for an L1.
    6. **L0-only** — L1, L2, L3 all blank.

    Backward-compatible: if the GeoDataFrame lacks ``taxonomy_l3`` /
    ``taxonomy_l2`` columns, the tiers that need them produce no
    matches and behaviour degrades gracefully to the shallower logic.

    Returns:
        (shared_label ndarray of object, match_radius_m ndarray of
        float)
    """
    n = len(gdf)
    label = np.full(n, "", dtype = object)
    radius = np.full(n, default_radius_m, dtype = np.float64)
    matched = np.zeros(n, dtype = bool)

    cw = overture_crosswalk.copy()
    if "overture_l3" not in cw.columns:
        cw["overture_l3"] = ""
    has_l1 = cw["overture_l1"] != ""
    has_l2 = cw["overture_l2"] != ""
    has_l3 = cw["overture_l3"] != ""

    # Build radius dict
    radii_dict: dict[str, float] = {}
    for _, row in match_radii.iterrows():
        radii_dict[row["shared_label"]] = float(
            row["match_radius_m"]
        )

    def _lookup(mask: pd.Series, key_cols: list[str]) -> pd.Series:
        """Build a label lookup keyed on ``key_cols`` joined by ``|``."""
        sub = cw[mask].copy()
        if sub.empty:
            return pd.Series(dtype = object)
        key = sub[key_cols[0]].astype(str)
        for col in key_cols[1:]:
            key = key + "|" + sub[col].astype(str)
        sub["_key"] = key
        return (
            sub.drop_duplicates("_key")
            .set_index("_key")["shared_label"]
        )

    # -- Build lookup tables for each tier --------------------------

    t_full_lkp = _lookup(
        has_l1 & has_l2 & has_l3,
        ["overture_l0", "overture_l1", "overture_l2", "overture_l3"],
    )
    t_l0l3_lkp = _lookup(
        ~has_l1 & ~has_l2 & has_l3,
        ["overture_l0", "overture_l3"],
    )
    t1_lkp = _lookup(
        has_l1 & has_l2 & ~has_l3,
        ["overture_l0", "overture_l1", "overture_l2"],
    )
    t2_lkp = _lookup(
        ~has_l1 & has_l2 & ~has_l3,
        ["overture_l0", "overture_l2"],
    )
    t3_lkp = _lookup(
        has_l1 & ~has_l2 & ~has_l3,
        ["overture_l0", "overture_l1"],
    )
    t4_lkp = _lookup(
        ~has_l1 & ~has_l2 & ~has_l3,
        ["overture_l0"],
    )

    # -- Extract columns from the data ------------------------------

    def _col(name: str) -> pd.Series:
        if name in gdf.columns:
            return gdf[name].fillna("").astype(str)
        return pd.Series("", index = gdf.index)

    l0 = _col("taxonomy_l0")
    l1 = _col("taxonomy_l1")
    l2 = _col("taxonomy_l2")
    l3 = _col("taxonomy_l3")

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

    # -- Apply tiers in order (most to least specific) -------------

    # Tier 1: (L0, L1, L2, L3)
    _apply_tier(
        l0 + "|" + l1 + "|" + l2 + "|" + l3,
        t_full_lkp,
        ~matched & (l3 != ""),
    )

    # Tier 2: (L0, L3) — a deep leaf, ignoring L1/L2
    _apply_tier(
        l0 + "|" + l3,
        t_l0l3_lkp,
        ~matched & (l3 != ""),
    )

    # Tier 3: (L0, L1, L2)
    _apply_tier(
        l0 + "|" + l1 + "|" + l2,
        t1_lkp,
        ~matched & (l2 != ""),
    )

    # Tier 4: (L0, L2) — ignores L1 in the data
    _apply_tier(
        l0 + "|" + l2,
        t2_lkp,
        ~matched & (l2 != ""),
    )

    # Tier 5: (L0, L1) — catch-all for an L1 group
    _apply_tier(
        l0 + "|" + l1,
        t3_lkp,
        ~matched & (l1 != ""),
    )

    # Tier 6: L0-only
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
