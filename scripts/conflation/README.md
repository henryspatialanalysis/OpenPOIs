# POI Conflation: OSM + Overture Maps

This pipeline conflates rated OpenStreetMap POIs with Overture Maps POIs into a
unified superset. The output includes matched pairs, unmatched OSM POIs, and
unmatched Overture POIs.

`conflate.py` alone writes *uncalibrated* confidence. The full national
pipeline is `make conflate`, which runs matching, change detection, and then
confidence calibration — the last stage replaces the blended/downweighted
confidence with a calibrated P(exists and open) per detection segment. See
`../../.claude/docs/confidence-calibration.md`.

## Usage

```bash
# Full run (~22M POIs; peak RSS ~10 GB projected, phase-by-phase logged to stdout)
python scripts/conflation/conflate.py

# Test mode (Seattle bbox, ~30k + ~19k POIs)
python scripts/conflation/conflate.py --test
```

Output: `~/data/openpois/conflation/{version}/conflated_baseline.parquet`
(matching only). The canonical `conflated.parquet` is produced at the end of
`make conflate`, after change detection and calibration.

## Algorithm Overview

### 1. Taxonomy Crosswalk

A CSV file (`src/openpois/conflation/data/taxonomy_crosswalk.csv`) maps between
OSM tag key/value pairs and Overture taxonomy categories. Each row defines:

| Column | Description |
|--------|-------------|
| `osm_key` | OSM tag key (amenity, shop, healthcare, leisure) |
| `osm_value` | OSM tag value (e.g. restaurant, supermarket) |
| `overture_l0` | Overture L0 category (e.g. food_and_drink) |
| `overture_l1` | Overture L1 category (e.g. restaurant) |
| `poi_category` | Unified category label |
| `match_radius_m` | Maximum spatial match distance (meters) |

OSM POIs are assigned categories using the filter key priority order (shop >
healthcare > leisure > amenity). If the specific tag value is not in the
crosswalk, a wildcard (`*`) fallback for that key is used (default 50m radius).

Overture POIs are matched using a 4-tier cascade from most to least specific:
(L0, L1, L2), then (L0, L2), then (L0, L1), then L0-only.

### 2. Spatial Candidate Search

A scikit-learn `BallTree` with haversine metric is built on all Overture POI
centroids. OSM POIs are queried in chunks of 500k to control memory. For each
OSM POI, all Overture POIs within that POI's category-specific match radius are
returned as candidates.

Match radii vary by POI type:
- Private businesses (restaurants, shops): ~50m
- Mid-size facilities (clinics, sports centres): ~75-100m
- Areal features (parks, hospitals, stadiums): ~150-200m

### 3. Match Scoring

Each candidate pair receives a composite score from four weighted components:

**Distance score (weight: 0.25)**
Linear decay from 1.0 (0 meters) to 0.0 (at the match radius threshold).

```
distance_score = 1.0 - (distance_m / match_radius_m)
```

**Name score (weight: 0.30)**
The maximum `rapidfuzz.fuzz.token_set_ratio` across up to four comparisons:
- OSM `name` vs Overture `overture_name`
- OSM `brand` vs Overture `brand_name`
- OSM `name` vs Overture `brand_name` (cross-compare)
- OSM `brand` vs Overture `overture_name` (cross-compare)

Token set ratio handles brand-as-subset patterns well (e.g. "Starbucks" vs
"Starbucks Coffee" = 100%). When all names and brands are null on both sides,
the score is set to 0.5 (neutral) rather than 0 to avoid penalizing unnamed
POIs.

**Type taxonomy score (weight: 0.25)**
Compares the unified `poi_category` assigned to each POI:
- Exact category match: 1.0
- Same broad group (both food_and_drink, both shopping, etc.): 0.5
- Different broad groups: 0.0
- One or both unmapped: 0.5 (neutral)

**Identifier score (weight: 0.20)**
Reserved for exact matches on `website`, `phone`, and `brand:wikidata`. Since
Overture's current schema does not expose these fields, this component returns
a neutral 0.5 for all pairs. It will become active when Overture adds these
identifiers.

**Composite score:**
```
composite = 0.25 * distance + 0.30 * name + 0.25 * type + 0.20 * identifier
```

### 4. One-to-One Match Selection

Candidate pairs with composite score >= 0.67 (configurable) are eligible. A
greedy algorithm assigns matches:

1. Sort all eligible pairs by composite score (descending).
2. Iterate: assign the pair if neither the OSM POI nor the Overture POI has
   been assigned yet.
3. Skip pairs where either side is already taken.

This produces a strict one-to-one mapping. The greedy approach is O(n log n)
and produces near-optimal results since most POIs have a clearly dominant match.

### 5. Confidence Merging

Let `w` = `overture_confidence_weight` (default 0.7, configurable). This
represents our trust in Overture relative to OSM.

**Matched pairs:**
```
osm_weight     = 1 / (1 + w)      # ~0.77 with w=0.3
overture_weight = w / (1 + w)      # ~0.23 with w=0.3

conf_mean  = osm_conf_mean  * osm_weight + overture_confidence * overture_weight
conf_lower = osm_conf_lower * osm_weight + overture_confidence * overture_weight
conf_upper = osm_conf_upper * osm_weight + overture_confidence * overture_weight
```

**Unmatched OSM:** Confidence scores are carried through as-is.

**Unmatched Overture:** `conf_mean = overture_confidence * w`. No confidence
interval bounds (set to null).

### 6. Geometry Selection

When a matched pair has different geometry types (e.g. OSM Polygon vs Overture
Point), the higher-level geometry is preferred:

```
MultiPolygon > Polygon > LineString > Point
```

On ties (both Points, both Polygons), the OSM geometry is used.

### 7. Tag Conflict Resolution

For name and brand, the source with higher confidence for that specific POI
determines the value. Both original values are preserved with source-prefixed
columns (`osm_name`, `overture_name`, `osm_brand`, `overture_brand`).

### 8. OSM Same-Brand Dedup

OSM often maps one branded business as two valid objects: a drugstore's building
(`shop=chemist`) and the pharmacy counter inside it (`amenity=pharmacy`), or two canopies
of one fuel station. Both get the same `shared_label`, and the matcher pairs at most one
with the Overture listing, so the merge emits two POIs for one place.
`dedup_osm_brand.py` runs on the merged rows, after the save and before
`assign_stable_ids.py`, and absorbs an OSM-only row into a kept row when they share a
`shared_label`, sit within `radius_m` (50 m), and their brand keys agree (brand, else
name, normalised, with generic tokens such as "pharmacy" dropped; `token_set_ratio >=
90`; at least one side carries a brand). A matched row is always kept and an
Overture-only row is never absorbed.

The kept row keeps its id, source and attributes. It lists every absorbed element in
`absorbed_osm_keys` (`node/123`, `way/456`), and when it is a node and a building way or
relation was absorbed into it, it takes that building's geometry. `match_distance_m` still
describes the OSM x Overture link as matched. `bbox` and `geohash` are computed from
`geometry` at publish, so they follow. Each absorbed element is also written to
`osm_brand_dedup_dropped.parquet`, which `assign_stable_ids.py` reads as alias edges, so
its prior id is recorded as `merged_into` the kept row's id.

It runs only for the labels in `conflation.osm_brand_dedup.labels`. Across all labels the
rule would absorb 11,336 rows on 2026-10-22. Each label's pairs were spot-checked against
live OSM tags (2026-11-23) and judged by one question: is this one place a person travels
to? A label is merged when at least 14 of 15 sampled pairs (8 of 8 for labels with fewer
than 50 pairs) were the same place. With the accepted labels the stage absorbs 8,940 rows
into 8,727 kept rows, 2,334 of which take a building's geometry.

| Label | Pairs | Sampled | Same place | Verdict | What the pairs were |
|---|---:|---:|---:|---|---|
| Pharmacy | 4,332 | 35 | 35 | **merged** | store building + in-store pharmacy node |
| Gas Station | 2,111 | 15 | 15 | **merged** | canopies / station node of one station |
| Fast Food | 560 | 15 | 15 | **merged** | building + node of one outlet |
| Hotel | 380 | 15 | 14 | **merged** | building + node; 1 pair was two towers of one hotel complex |
| Charging Station | 350 | 15 | 0 | left alone | separate charge points in one lot |
| ATM | 332 | 15 | 0 | left alone | separate machines |
| Convenience Store | 298 | 15 | 13 | left alone | two airport newsstands; a live store paired with a vacant one |
| Restaurant | 277 | 15 | 14 | **merged** | 1 pair was two restaurants of one brand next door |
| Bank | 274 | 15 | 14 | **merged** | branch + drive-through |
| Clothing Store | 250 | 15 | 7 | left alone | Gap / babyGap, men's / women's: separate stores |
| Place of Worship | 242 | 15 | 15 | **merged** | church building + grounds |
| Self-Storage | 158 | 15 | 15 | **merged** | buildings of one facility |
| Car Dealer | 154 | 15 | 14 | **merged** | lot / showroom of one dealer; 1 auto mall |
| Supermarket | 141 | 15 | 15 | **merged** | building + node |
| Other Shop | 132 | 15 | 13 | left alone | airport newsstands |
| Car Repair | 117 | 15 | 12 | left alone | parts store vs another shop |
| Cafe | 101 | 15 | 13 | left alone | several Starbucks in one airport or campus |
| Shoe Store | 84 | 15 | 5 | left alone | Foot Locker / Kids Foot Locker |
| Other Financial | 65 | 15 | 12 | left alone | separate agents' offices |
| Cell Phone Store | 63 | 15 | 9 | left alone | T-Mobile / Metro by T-Mobile |
| Discount Store | 58 | 15 | 15 | **merged** | building + node |
| Recreation | 51 | 15 | 15 | **merged** | facility + feature inside it (rink, ride) |
| School | 48 | 8 | 7 | left alone | two schools on one site |
| Clinic | 48 | 8 | 2 | left alone | separate departments |
| Public Safety | 48 | 8 | 3 | left alone | police vs fire |
| Fitness Center | 46 | 8 | 7 | left alone | one club, two addresses (unclear) |
| Specialty Store | 44 | 8 | 7 | left alone | two Coach stores (unclear) |
| Car Wash | 41 | 8 | 8 | **merged** | tunnels / vacuums of one wash |
| Real Estate | 35 | 8 | 8 | **merged** | agents in one office |
| Furniture Store | 31 | 8 | 6 | left alone | kids / sub-brand stores |
| Dessert Shop | 29 | 8 | 7 | left alone | two mall kiosks |
| Tire Store | 28 | 8 | 8 | **merged** | building + node |
| Car Rental | 27 | 8 | 8 | **merged** | counter + lot |
| Pet Store | 27 | 8 | 8 | **merged** | store + grooming inside it |
| Dog Park | 26 | 8 | 0 | left alone | unnamed enclosures; no basis to judge |
| Hair and Beauty | 24 | 8 | 5 | left alone | different salons in one suite |
| Bar | 23 | 8 | 8 | **merged** | brewery building + taproom |
| Kindergarten | 21 | 8 | 8 | **merged** | building + grounds |
| 47 labels under 20 pairs | 260 | 20 | 16 | left alone | mixed |

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `unified_id` | str | Release-stable opaque id (`opoi_` + 16 base32 chars), set by `assign_stable_ids.py`; `conflate.py` writes a provisional `matched:{n123}_{overture_id}` / `osm:{w123}` / `overture:{overture_id}` that the stage replaces |
| `source` | str | `"matched"`, `"osm"`, or `"overture"` |
| `osm_type` | str, nullable | `node`, `way` or `relation` |
| `osm_id` | int64, nullable | OpenStreetMap element ID |
| `overture_id` | str, nullable | Overture place ID |
| `name` | str, nullable | Best name (from higher-confidence source) |
| `brand` | str, nullable | Best brand (from higher-confidence source) |
| `poi_category` | str | Unified category from crosswalk |
| `conf_mean` | float64 | Blended confidence score |
| `conf_lower` | float64, nullable | Lower confidence bound |
| `conf_upper` | float64, nullable | Upper confidence bound |
| `match_score` | float64, nullable | Composite match score (matched pairs only) |
| `match_distance_m` | float64, nullable | Distance between matched geometries |
| `osm_name` | str, nullable | Original OSM name |
| `overture_name` | str, nullable | Original Overture name |
| `osm_brand` | str, nullable | Original OSM brand |
| `overture_brand` | str, nullable | Original Overture brand name |
| `osm_conf_mean` | float64, nullable | Original OSM confidence |
| `access` | str, nullable | OSM `access` tag (NULL for Overture-only rows) |
| `airside` | str, nullable | OSM `airside` tag: `yes` for POIs past airport security (NULL for Overture-only rows and snapshots before 2026-11-23) |
| `absorbed_osm_keys` | list<str>, nullable | OSM elements folded into this row by the same-brand dedup (`node/123`, `way/456`); NULL when none |
| `overture_confidence` | float64, nullable | Original Overture confidence |
| `geometry` | Point/Polygon/MultiPolygon | EPSG:4326 |

## Configuration

All parameters are configurable in `config.yaml` under the `conflation` section:

```yaml
conflation:
  overture_confidence_weight: 0.7   # Trust in Overture relative to OSM
  min_match_score: 0.67             # Minimum composite score for a match
  max_radius_m: 200                 # Global upper bound on search radius
  default_radius_m: 50              # Default radius for unmapped categories
  distance_weight: 0.25             # Weight for distance score component
  name_weight: 0.30                 # Weight for name score component
  type_weight: 0.25                 # Weight for type taxonomy score component
  identifier_weight: 0.20           # Weight for identifier score component
  chunk_size: 500_000               # OSM rows per BallTree query batch
  osm_brand_dedup:                  # Section 8; on the merged rows
    enabled: true
    radius_m: 50
    min_name_score: 90
    labels: ["Pharmacy", "Gas Station", ...]  # 19 labels; null = every label
  test_bbox:                        # Geographic filter for --test mode
    xmin: -122.45
    ymin: 47.50
    xmax: -122.25
    ymax: 47.70
```

## Taxonomy Migration Notice

Overture Maps is transitioning from the current hierarchical `categories` taxonomy
(L0/L1) to a flat `basic_category` system. The old taxonomy is scheduled for
deprecation around **June 2026**. When that migration completes, the crosswalk CSV
and the `assign_overture_poi_category()` function will need to be updated to use the
new field names and category values. Track the migration status in the Overture Maps
changelog.

## Memory and Performance

The pipeline targets ~10 GB peak RSS on ~22M total POIs (8.7M OSM + 13M
Overture). Phase-by-phase RSS / VmHWM is logged to stdout via `log_rss()`
reading `/proc/self/status`. Key memory tactics:

- Only columns needed for matching are loaded from parquet files
- BallTree is built once on Overture centroids (~0.4 GB for 13M points)
- OSM queries are chunked (500k rows per batch) with chunk match outputs
  narrowed to int32/float32 dtypes
- Chunk matches stream to a checkpoint parquet per chunk — no in-memory
  `part_dfs` accumulation
- Cross-chunk dedup runs in DuckDB with a bounded memory limit (default 4 GB)
  via `ROW_NUMBER() OVER (PARTITION BY overture_idx ...)` over the checkpoint
  parquet glob, instead of pandas concat + sort + drop_duplicates
- `osm_gdf` / `overture_gdf` are dropped during chunked matching (only
  centroids are kept for the BallTree) and reloaded with narrow merge-only
  columns before the merge step. The post-dedup Overture spill backing that
  reload is streamed row-group-by-row-group from the source snapshot with the
  full match+merge column union (`spill.py`) — never from the narrow in-memory
  match frame, which would (and once did) silently null every Overture
  contact/address column in the output. The merge reload hard-fails on any
  missing column or row-count misalignment instead of null-filling
- Name scoring uses rapidfuzz (C++ backend, ~100x faster than difflib)
- Output is Hilbert-sorted for efficient cloud-native range reads

## Code Structure

```
src/openpois/conflation/
    __init__.py
    data/taxonomy_crosswalk.csv     # Category mapping between OSM and Overture
    taxonomy.py                     # Crosswalk loading and category assignment
    match.py                        # Spatial search, scoring, match selection
    merge.py                        # Confidence blending and output assembly
    spill.py                        # Row-filtered streaming spill (post-dedup Overture)
    dedup_osm_brand.py              # Same-brand OSM dedup on the merged rows

scripts/conflation/
    conflate.py                     # Driver script (loads config, calls library)
    README.md                       # This file

tests/
    test_taxonomy.py                # 12 tests
    test_match.py                   # 19 tests
    test_merge.py                   # 7 tests
```
