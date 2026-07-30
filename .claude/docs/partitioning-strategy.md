# Partitioning strategy

How the rated OSM snapshot and the conflated dataset are laid out on disk, and why.

## Why this layout

Historically both datasets were Hive-partitioned by a 4-character geohash (~1,000–3,000 cells over CONUS) and uploaded to S3 so the web frontend could fetch just the cells covering a map viewport. The current use case is different: **local, nationwide queries filtered primarily by destination type**, with spatial filters as a frequent secondary slice.

Geohash partitioning is actively bad for that pattern — a nationwide "all pharmacies" query has to open every geohash directory. Partitioning by destination type gives near-zero scan for type-filtered queries (one file instead of ~1,500), and we retain spatial efficiency by sorting each partition by geohash so bbox / state / region filters prune via Parquet row-group min/max stats.

Confirmed on the real data: `WHERE shared_label = 'Pharmacy'` on the 17.8 M-row conflated set scans `1/93` files in ~5 ms.

## Layouts

### Conflated (`conflated_partitioned/`)

| | |
|---|---|
| Path | `~/data/openpois/conflation/<versions.conflation>/conflated_partitioned/` |
| Partition column | `shared_label` (URL-encoded in dir name; DuckDB `hive_partitioning=1` decodes transparently) |
| Partitions | 93 (incl. one `shared_label=` bucket for ~720 k unlabeled POIs that don't map to any crosswalk entry) |
| Rows | 17,788,585 total for `20260423` |
| Within-partition sort | ascending `geohash` (precision 6, retained as a column) |
| Dropped at write | `shared_label` (lives in the Hive dir name) |
| On-disk size | ~2.7 GB for `20260423` |

### Rated OSM snapshot (`osm_snapshot_partitioned/`)

| | |
|---|---|
| Path | `~/data/openpois/snapshots/osm/<versions.osm_data>/osm_snapshot_partitioned/` |
| Partition column | derived `primary_tag` ∈ {shop, healthcare, leisure, amenity, tourism, office, craft, historic, landuse} (`landuse` added by the crosswalk-derived value-scoped ingest filter; it appears from the next data pull onward — `20260417` predates it) |
| Partitions | 8 for `20260417` (9 once `landuse` lands) |
| Rows | 8,708,504 total for `20260417`. Distribution: amenity 4.90 M, leisure 2.22 M, shop 0.79 M, tourism 0.38 M, office 0.16 M, historic 0.12 M, healthcare 0.11 M, craft 0.03 M |
| Within-partition sort | ascending `geohash` (precision 6, retained as a column) |
| Dropped at write | `primary_tag` (lives in the Hive dir name) |
| On-disk size | ~1.2 GB for `20260417` (down from 1.9 GB under the old geohash layout) |

## `primary_tag` derivation (OSM)

~1.9% of rated OSM POIs carry more than one top-level tag (e.g., OSM id `25603734` has both `shop=convenience` and `amenity=fuel`). To pick one partition per POI we apply the same **first-non-null priority** already used by [assign_osm_shared_label()](../../src/openpois/conflation/taxonomy.py), sourced from [`config.yaml` `download.osm.filter_keys`](../../config.yaml):

```
shop > healthcare > leisure > amenity > tourism > office > craft > historic > landuse
```

`landuse` sits last (lowest priority), so a cemetery also tagged with a more specific POI key is labeled by that key; only `landuse`-only features (value-scoped to `cemetery` / `religious` at ingest) land under `primary_tag=landuse/`.

This keeps OSM-only queries and conflation-side labeling consistent: a shop+amenity POI sits under `primary_tag=shop/` and the conflation side labels it via the `shop` crosswalk. All filter-key tag columns (`shop`, `amenity`, etc.) are retained inside the files, so a secondary filter like `primary_tag = 'shop' AND shop = 'bakery'` still works within the one partition that was opened.

Every POI in the rated snapshot has at least one filter-key tag populated (guaranteed by the PBF filtering step in [scripts/osm_snapshot/download.py](../../scripts/osm_snapshot/download.py)), so no null / `__unlabeled__` bucket is needed.

## How to query

All examples use DuckDB with `hive_partitioning=1`, which URL-decodes partition values back to their original form.

```python
import duckdb

CONFLATED = "~/data/openpois/conflation/20260423/conflated_partitioned/**/*.parquet"
OSM       = "~/data/openpois/snapshots/osm/20260417/osm_snapshot_partitioned/**/*.parquet"
```

**Type-only, nationwide — reads one file.**

```sql
SELECT COUNT(*) FROM read_parquet(CONFLATED, hive_partitioning=1)
WHERE shared_label = 'Pharmacy';
```

**Type + spatial bbox via `geohash` prefix — row-group pruning inside one partition.**

```sql
SELECT name, geohash
FROM read_parquet(CONFLATED, hive_partitioning=1)
WHERE shared_label = 'Pharmacy'
  AND geohash LIKE '9q5%';   -- western US geohash-3 cell
```

For lat/lon bboxes, convert to geohash prefixes with `pygeohash.bbox`/`expand`. A ZXY or state-level filter can usually be expressed as a small disjunction of `geohash LIKE` prefixes.

**Secondary filter inside an OSM partition.**

```sql
SELECT COUNT(*) FROM read_parquet(OSM, hive_partitioning=1)
WHERE primary_tag = 'shop' AND shop = 'bakery';   -- one file scanned
```

**Joining conflated and OSM (e.g., type breakdown by OSM tag).**

```sql
SELECT c.shared_label, o.primary_tag, COUNT(*)
FROM read_parquet(CONFLATED, hive_partitioning=1) c
JOIN read_parquet(OSM, hive_partitioning=1) o USING (osm_id)
WHERE c.shared_label = 'Pharmacy'
GROUP BY 1, 2;
```

## When NOT to use this layout

The geohash-partitioned layout is a better fit for **small-bbox, many-types-at-once** queries — which is exactly the web-map viewport case we moved away from. If the map-viewport path comes back, the helpers are still in place: see `add_geohash_columns` and `write_partitioned_dataset` in [src/openpois/io/geohash_partition.py](../../src/openpois/io/geohash_partition.py), and the Source Cooperative publish step in [scripts/publish/upload_to_source_coop.py](../../scripts/publish/upload_to_source_coop.py). Swap the function calls in the two `format_for_upload.py` scripts back to the geohash variants.

## Maintenance

**Regenerate after a new conflation or snapshot run:**

```bash
python -u scripts/osm_snapshot/format_for_upload.py   2>&1 | tee ~/data/openpois/logs/osm_repartition_<version>.log
python -u scripts/conflation/format_for_upload.py     2>&1 | tee ~/data/openpois/logs/conflated_repartition_<version>.log
```

Each script deletes the existing partitioned directory at its versioned path and rewrites it. Geohash precision is controlled by `publish.geohash_precision_sort` in [config.yaml](../../config.yaml) (currently 6 ≈ 0.6 × 1.2 km).

**Where the code lives:**

- [src/openpois/io/geohash_partition.py](../../src/openpois/io/geohash_partition.py) — `add_geohash_column`, `compute_primary_osm_tag`, `write_label_partitioned_dataset` (plus the older geohash-partition helpers).
- [scripts/conflation/format_for_upload.py](../../scripts/conflation/format_for_upload.py) — conflated partitioning entry point.
- [scripts/osm_snapshot/format_for_upload.py](../../scripts/osm_snapshot/format_for_upload.py) — OSM partitioning entry point.
- [tests/test_geohash_partition.py](../../tests/test_geohash_partition.py) — unit tests + a DuckDB Hive-decode round-trip.

The Source Cooperative publish flow ([scripts/publish/upload_to_source_coop.py](../../scripts/publish/upload_to_source_coop.py)) uploads these same partitioned trees to `<version>/osm-parquet/` and `<version>/conflated-parquet/`. PMTiles generation remains downstream of partitioning.


## Writing the partitioned datasets (memory)

Both trees are written by `openpois.io.geohash_partition`, which offers two
entry points. **Use the streaming one for anything national.**

| Function | Reads | When |
|---|---|---|
| `write_label_partitioned_dataset(gdf, ...)` | a whole in-memory GeoDataFrame | small / test-scale frames, and the reference implementation the streaming path is tested against |
| `write_label_partitioned_from_parquet(path, ...)` | one partition at a time, off disk | production; what both `format_for_upload.py` scripts call |

The whole-frame path does not fit at CONUS scale. Reading the 20260730
conflated parquet (14.6M rows × 58 columns) as a single GeoDataFrame peaked at
**21.5 GB RSS against this machine's 24 GB WSL cap**, spilled 2.2 GB into swap,
and started consuming the Windows C: pagefile — the documented VM-crash mode.
The streaming path holds a few hundred MB and ran with 15–16 GB still free.

Output is identical between the two, and that equivalence is pinned by tests
(`tests/test_geohash_partition.py`): same columns, row counts, Hive directory
names, geohash sort, GeoParquet 1.1 `bbox` covering column, and the same
`__index_level_0__` values — the streaming path carries each row's original
*file position* rather than restarting the index at zero per partition.

Two wrinkles worth knowing:

* **Derived partition columns.** `primary_tag` is computed, not stored, so a
  dataset predicate cannot select its rows. The OSM path therefore derives the
  labels in a narrow pass over just the filter-key columns and passes them via
  `labels=`, after which rows are gathered by a row-group scan. Pass `labels=`
  only when the column is absent from the file.
* **Nullable label dtypes.** `compute_primary_osm_tag` returns pandas `string`
  dtype, so `labels == value` yields `pd.NA` for unlabeled rows and the mask is
  unusable until `.fillna(False)`. Same family of trap as the pyarrow Kleene
  issue in CLAUDE.md, different library.

Cost of streaming: one filtered scan of the source file per partition (102 for
the conflated tree). It is I/O-bound rather than memory-bound, which is the
trade being made deliberately.
