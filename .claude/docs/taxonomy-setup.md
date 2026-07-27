# Taxonomy setup

The unified taxonomy bridges OSM tags, Overture L0/L1/L2 categories, and a `shared_label` used throughout conflation and the frontend.

## Source CSVs

All four live at [src/openpois/conflation/data/](../../src/openpois/conflation/data/):

| File | Columns | Purpose |
|---|---|---|
| `taxonomy_crosswalk_openstreetmap.csv` | `osm_key`, `osm_value`, `shared_label` | Map OSM tag key/value pairs to a shared label. Wildcard `*` on `osm_value` is a fallback per key. |
| `taxonomy_crosswalk_overture_maps.csv` | `overture_l0`, `overture_l1`, `overture_l2`, `overture_l3`, `shared_label` | Map Overture hierarchy to a shared label. **6-tier cascade** (see below). |
| `match_radii.csv` | `shared_label`, `match_radius_m` | Per-label spatial match radius (meters). Private businesses ~50m, mid-size facilities ~75-100m, areal features ~150-200m. **Master label list** — `build_taxonomy.py` derives the site's `SHARED_LABELS` from this file, so a label missing here never reaches the frontend. |
| `top_level_matches.csv` | `overture_l0`, `osm_key` | L0/key bitmask for the type-score "same broad group" check. Only affects *near-miss* partial credit; equal labels already score full marks. |
| `marketplace_name_labels.csv` | `name_normalized`, `shared_label`, `source` | Exceptions to the `amenity=marketplace` name rules (below). Only names the regexes get wrong belong here. |

All five are UTF-8 without BOM, LF line endings.

## Code

[src/openpois/conflation/taxonomy.py](../../src/openpois/conflation/taxonomy.py) exposes:

- Loaders: `load_osm_crosswalk`, `load_overture_crosswalk`, `load_match_radii`, `load_top_level_matches`, `load_marketplace_names`
- Assigners: `assign_osm_shared_label`, `assign_overture_shared_label`
- Ingest filter: `build_osm_tag_filter_expressions`
- Marketplace split: `classify_marketplace_name`, `normalize_marketplace_name`, `refine_marketplace_labels`

OSM key priority order for label assignment comes from `download.osm.filter_keys` in config.yaml: **shop > healthcare > leisure > amenity > tourism > office > craft > historic > landuse** (specific tags win over generic).

> **Every consumer must load all nine tag columns.** `assign_osm_shared_label` silently skips keys absent from the frame, so a caller that reads only some of them strips the label from every POI whose primary tag lives under a missing key — this is exactly how 817k POIs (Hotel, Museum, Cemetery, Campground, Real Estate, ...) went missing from the July 2026 conflation. Derive the column list from `FILTER_KEYS`, never hardcode it.

### The Overture cascade is 6 tiers

`assign_overture_shared_label` applies these in order, each only to still-unmatched rows. A row's tier is set by which columns it populates:

1. `(L0, L1, L2, L3)` — full path
2. `(L0, L3)` — deep leaf, ignoring intermediates
3. `(L0, L1, L2)`
4. `(L0, L2)` — leaf, ignoring L1
5. `(L0, L1)`
6. `(L0)` — catch-all fallback

**Prefer the leaf-anchored tiers (2 and 4) where the leaf name is unambiguous.** Overture reparents categories frequently; a leaf-anchored row survives a reparent, a full-path row does not. The `health_care` rows were the only part of the crosswalk to survive Overture's Feb/Mar-2026 restructure intact, precisely because they were leaf-anchored.

### The `amenity=marketplace` name split

OSM has no tag distinguishing a farmers market from a flea or public market (`marketplace=*` is used on 7 US features; the wiki documents no companion tag), so the name decides. `classify_marketplace_name` checks "definitely another kind of market" patterns first (flea, swap, bazaar, meat/fish market, auction), then grower/produce vocabulary (`farm` unanchored so it catches farmstand/freshfarm, plus orchard, produce, fruit, tailgate/curb market, CSA, ...), defaulting to `Market`. `grove` is deliberately excluded — Oak Grove and Elk Grove are place names.

The rules land within a handful of names of an LLM second opinion over all 2,637 US marketplace names, so `marketplace_name_labels.csv` holds only the ~8 they get wrong. Re-audit after a snapshot refresh with `scripts/conflation/classify_marketplaces.py` (`--rules-only` inside a Claude Code session; the LLM pass shells out to the `claude` CLI, which refuses to nest). A test fails if an exceptions row agrees with the rules — such rows are dead weight and should be deleted.

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

## Checking taxonomy drift between Overture releases

Run `scripts/overture/compare_taxonomy.py` whenever switching to a new Overture
release. It scans the release's `taxonomy` over S3 and reports: the schema shape
(confirms `taxonomy` is still the `{primary, hierarchy[], alternates[]}` struct),
the distinct `(L0..L3)` tuples added/removed versus the prior snapshot and versus
`taxonomy_crosswalk_overture_maps.csv`, any allowlisted-but-unmapped tuples, and
(§6) crosswalk liveness. The expensive S3 census is cached to CSV; re-run the
report offline with `--from-census-csv`.

### §6 is the check that matters

The coverage check in §4 asks "does every ingested tuple get a label?" — and the
L0-only fallback rows answer "yes" even when every finer row has gone stale. That
is how the crosswalk silently rotted to 45% dead rows between the Feb/Mar-2026
Overture restructure and July 2026, with 10.5% of POIs collapsing onto catch-alls
(all cafes and bars into Other Amenity; gyms, golf courses, pools and bowling
alleys into Recreation) while the monthly check reported clean.

§6 asks the sharper questions:

- **Stale rows** — rows naming a value that no longer exists *at that level*
  anywhere in the release. This is the drift signal; it would have caught the
  July rot. (`bar` still exists in the release, but as an L2 — a crosswalk row
  using it as an L1 is stale.)
- **Shadowed rows** — duplicate lookup keys within a tier; only the first is ever
  reachable.
- **Unused rows** — well-formed but never the resolving tier because finer rows
  cover the subtree. Catch-alls live here by design; informational only.
- **Tier usage** — the POI share resolving at each tier, flagged when the
  L0-fallback share exceeds `--max-l0-fallback-share` (default 5%).

`in_allowlist` is recomputed from the live config on every run, so
`--from-census-csv` doubles as a **pre-flight gate**: edit the crosswalk and/or
`taxonomy_allowlist`, run with `--strict`, and see the effect in seconds without
re-hitting S3. Do this before starting a download — a crosswalk defect found here
costs seconds instead of an hour.

```bash
python scripts/overture/compare_taxonomy.py --strict \
    --from-census-csv ~/data/openpois/logs/overture_taxonomy_census_20260722.csv
```

`--strict` exits 1 on stale rows, shadowed rows, unmapped tuples, a fallback share
over the threshold, or a mismatch between the per-tier resolution and the full
cascade (a self-check guarding against this script's tier patterns drifting from
`assign_overture_shared_label`).

## Possible future Overture migration to a flat `basic_category`

Overture has long signaled it may deprecate the `categories` hierarchy in favor
of a flat `basic_category` field. This has **not** happened as of the
`2026-07-22.0` release — the data is still the nested `taxonomy.hierarchy[]`
(deepened to four levels in June 2026), and `basic_category` exists alongside it.
If the flat migration ever lands:

- `taxonomy_crosswalk_overture_maps.csv` schema will need to change from
  `(overture_l0, overture_l1, overture_l2, overture_l3)` to `(basic_category)` or
  equivalent.
- `assign_overture_shared_label` in `taxonomy.py` will need updating to use the new field.
- `scripts/overture/download.py` → SQL queries against `taxonomy.hierarchy[1]` will need updating.

Track the migration status in the Overture Maps changelog and via the
`compare_taxonomy.py` schema check above.
