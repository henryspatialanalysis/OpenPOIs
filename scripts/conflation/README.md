# POI Conflation: OSM + Overture Maps

This pipeline conflates rated OpenStreetMap POIs with Overture Maps POIs into a
unified superset with blended confidence scores. The output includes matched
pairs, unmatched OSM POIs, and unmatched Overture POIs.

## Usage

```bash
# Full run (~22M POIs; peak RSS ~10 GB projected, phase-by-phase logged to stdout)
python scripts/conflation/conflate.py

# Test mode (Seattle bbox, ~30k + ~19k POIs)
python scripts/conflation/conflate.py --test
```

Output: `~/data/openpois/conflation/{version}/conflated.parquet`

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

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `unified_id` | str | `"matched:{osm_id}_{overture_id}"`, `"osm:{osm_id}"`, or `"overture:{overture_id}"` |
| `source` | str | `"matched"`, `"osm"`, or `"overture"` |
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

scripts/conflation/
    conflate.py                     # Driver script (loads config, calls library)
    README.md                       # This file

tests/
    test_taxonomy.py                # 12 tests
    test_match.py                   # 19 tests
    test_merge.py                   # 7 tests
```
