# Taxonomy setup

The unified taxonomy bridges OSM tags, Overture L0/L1/L2 categories, and a `shared_label` used throughout conflation and the frontend.

## Source CSVs

All four live at [src/openpois/conflation/data/](../../src/openpois/conflation/data/):

| File | Columns | Purpose |
|---|---|---|
| `taxonomy_crosswalk_openstreetmap.csv` | `osm_key`, `osm_value`, `shared_label` | Map OSM tag key/value pairs to a shared label. Wildcard `*` on `osm_value` is a fallback per key. |
| `taxonomy_crosswalk_overture_maps.csv` | `overture_l0`, `overture_l1`, `overture_l2`, `shared_label` | Map Overture hierarchy to a shared label. 4-tier cascade: (L0, L1, L2) → (L0, L2) → (L0, L1) → L0. |
| `match_radii.csv` | `shared_label`, `match_radius_m` | Per-label spatial match radius (meters). Private businesses ~50m, mid-size facilities ~75-100m, areal features ~150-200m. |
| `top_level_matches.csv` | `overture_l0`, `osm_key` | L0/key bitmask for the type-score "same broad group" check. |

## Code

[src/openpois/conflation/taxonomy.py](../../src/openpois/conflation/taxonomy.py) exposes:

- Loaders: `load_osm_crosswalk`, `load_overture_crosswalk`, `load_match_radii`, `load_top_level_matches`
- Assigners: `assign_osm_shared_label`, `assign_overture_shared_label`
- Ingest filter: `build_osm_tag_filter_expressions`

OSM key priority order for label assignment: **shop > healthcare > leisure > amenity** (specific tags win over generic).

### The crosswalk also drives PBF ingest

`build_osm_tag_filter_expressions(load_osm_crosswalk())` turns the OSM crosswalk into `osmium tags-filter` expressions, so we only ingest tag *values* the taxonomy actually maps. Per `osm_key`:

- has a wildcard `*` row → matched at key level (`nwr/amenity`)
- no wildcard row → **value-scoped** to the listed values (`nwr/landuse=cemetery,religious`, `nwr/craft=<28 values>`)

Both ingest pipelines consume these via the `tag_filter_exprs` argument — snapshot (`scripts/osm_snapshot/download.py` → `download_osm_snapshot` → `filter_pbf`) and history (`scripts/osm_data/download_history.py` → `download_osm_history` → `filter_history_pbf`). This keeps both POI sets aligned with the taxonomy and avoids dragging in every value of a broad key (e.g. all `landuse=*` polygons).

Consequence when editing the CSV: adding a row under a **wildcard-less** key (currently `landuse`, `craft`) widens what gets ingested on the next data pull; removing one narrows it. Adding a value under a key that has a `*` row changes only the label, not ingest. `config.yaml` `download.osm.filter_keys` must still list every `osm_key` (it sets the label-assignment priority order and the parse-time element gate).

## Regenerating the site's taxonomy artifacts

After editing any of the four CSVs, run:

```bash
python scripts/build_taxonomy.py
```

This regenerates **both** generated outputs (both are gitignored):

1. [site/public/taxonomy.html](../../site/public/taxonomy.html) — user-facing HTML table showing the full crosswalk + radii.
2. [site/src/taxonomy.generated.js](../../site/src/taxonomy.generated.js) — `SHARED_LABELS`, `OSM_KEYS`, `OVERTURE_L0S` arrays imported by [site/src/constants.js](../../site/src/constants.js).

Then verify there's no drift:

```bash
python scripts/check_taxonomy_sync.py   # exits 0 if clean
pytest tests/test_taxonomy_sync.py      # same check, run in CI
```

The [sync-taxonomy](../skills/sync-taxonomy/SKILL.md) skill wraps this workflow.

## Hand-maintained pieces

Everything in [site/src/constants.js](../../site/src/constants.js) now derives from the generated arrays **except** the display-label maps:

- `OSM_KEY_LABELS` — e.g. `amenity: 'Amenity'`, `tourism: 'Tourism'`.
- `OVERTURE_L0_LABELS` — e.g. `food_and_drink: 'Food & Drink'`.

When a new `osm_key` or `overture_l0` is added to the CSVs, `check_taxonomy_sync.py` prints a `WARN:` line pointing at any missing display-label entries. Missing entries fall back to the raw key — ugly but not broken.

## Upcoming Overture migration (~June 2026)

Overture is deprecating the L0/L1/L2 `categories` hierarchy in favor of a flat `basic_category` field. When that happens:

- `taxonomy_crosswalk_overture_maps.csv` schema will need to change from `(overture_l0, overture_l1, overture_l2)` to `(basic_category)` or equivalent.
- `assign_overture_shared_label` in `taxonomy.py` will need updating to use the new field.
- `scripts/overture/download.py` → SQL queries against `taxonomy.hierarchy[1]` will need updating.

Track the migration status in the Overture Maps changelog.
