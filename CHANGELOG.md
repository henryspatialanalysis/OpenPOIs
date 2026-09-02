# Changelog

## 2026-09-02-v0

### Snapshot inputs

| Source                 | Value                                       |
| ---------------------- | ------------------------------------------- |
| OSM snapshot date      | 2026-09-01                                  |
| Overture release       | `2026-08-19.0` (pinned)                     |
| OSM snapshot rows      | 4,508,280                                   |
| Overture snapshot rows | 13,785,024                                  |
| Boundary footprint     | US + all territories (PR, USVI, GU, MP, AS) |

### Conflated output

| Metric                       | This run    | Prior        | Δ                       |
| ---------------------------- | ----------- | ------------ | ----------------------- |
| Total rows                   | 15,586,580  | 14,613,331   | +973,249 (+6.66%)       |
| Matched OSM × Overture       | 1,817,729   | 1,787,072    | +30,657 (+1.72%)        |
| OSM-only                     | 2,685,038   | 2,678,337    | +6,701 (+0.25%)         |
| Overture-only                | 11,083,813  | 10,147,922   | +935,891 (+9.22%)       |
| Shadow-matched (CD penalty)  | 31,411      | 30,340       | +1,071                  |
| Shared labels                | 102         | 102          | —                       |

Growth is driven by the upstream Overture release (+9.3% rows month-over-month,
of which 864,165 were removed again by internal dedup). The OSM snapshot is
−5.4% vs July: this is the first pull with the value-scoped ingest filter
applied at PBF-filter time, and the removed rows are entirely non-destination
object values (e.g. `tourism=information`, `tourism=picnic_site`) that the July
taxonomy overhaul had already unlabelled.

### Methods changes vs. prior release

- **Confidence calibration reused, not refit.** A new monthly drift gate
  (`scripts/overture/compare_confidence.py`) compares Overture confidence on
  matched GERS ids across releases; 2026-08-19.0 passed with wide margin
  (RMSE 0.036, mean bias +0.005, 2.0% of POIs moving |Δ| > 0.1), so the July
  validation round's fitted curves were applied verbatim to this release.
  **`conf_mean` is therefore directly comparable to 2026-07-30-v0.** The
  decision rule is documented in the repo
  (`.claude/docs/confidence-calibration.md`).
- **Shadow-match threshold now 0.70.** `min_shadow_match_score` was raised
  0.50 → 0.70 after the July run and takes effect here (first release built
  with it).
- **Crosswalk updates for Overture's August taxonomy.** The
  `beauty_service` → `personal_or_beauty_service` rename and the
  `food_delivery_service` move under `lifestyle_services` are remapped to their
  prior shared labels; the `place_of_worship` restructure is absorbed by the
  existing cascade. No shared labels added or removed.
- **Type-affinity table rebuilt** against the 2026-08-19.0 category hierarchy
  (k = 100, identifier-confirmed pairs from the July conflation).

## 2026-07-30-v0

### Snapshot inputs

| Source                 | Value                                       |
| ---------------------- | ------------------------------------------- |
| OSM snapshot date      | 2026-07-24                                  |
| Overture release       | `2026-07-22.0` (pinned)                     |
| OSM snapshot rows      | 4,764,221                                   |
| Overture snapshot rows | 12,606,804                                  |
| Boundary footprint     | US + all territories (PR, USVI, GU, MP, AS) |

### Conflated output

| Metric                       | This run    | Prior        | Δ                       |
| ---------------------------- | ----------- | ------------ | ----------------------- |
| Total rows                   | 14,613,331  | 14,922,534   | −309,203 (−2.07%)       |
| Matched OSM × Overture       | 1,787,072   | —            | see note                |
| OSM-only                     | 2,678,337   | —            | see note                |
| Overture-only                | 10,147,922  | —            | see note                |
| Shadow-matched (CD penalty)  | 30,340      | —            | —                       |
| Shared labels                | 102         | 93           | +9                      |

Row counts are not directly comparable to the 2026-06-27 release: the July run
also introduced the taxonomy overhaul and two non-POI exclusions (the
residential-landuse spatial rule and the wildcard → inclusion-set change).

### Methods changes vs. prior release

- **Confidence calibration (new, headline change).** `conf_mean` on the
  conflated dataset is now a **calibrated probability that the POI exists and
  is currently open to the public**, estimated from an independent validation
  sample (7,504 researched POIs, 2,362 with human-confirmed status) via a
  design-based two-phase estimator, and fitted separately per detection
  segment. This replaces three uncalibrated constants: the `0.588/0.412`
  matched blend, the flat `×0.7` downweight on Overture-only confidence, and
  the OSM-only passthrough. **`conf_mean` is therefore not comparable to any
  earlier release**; `conf_mean_uncalibrated` retains the old-style value.
  Matched POIs now combine both source scores through a *fitted* log-odds pool
  rather than a fixed linear blend.
  - New columns: `conf_mean_uncalibrated`, `calibration_flag`.
  - `conf_lower` / `conf_upper` are now populated for **all** segments; they
    were null for every Overture-only row in prior releases.
  - Mean `conf_mean` by segment: matched 0.878 → 0.907, OSM-only 0.780 →
    0.802, Overture-only 0.570 → 0.684.
  - `conf_mean` in `osm-parquet/` is unchanged and remains the *uncalibrated*
    OSM turnover posterior. The two datasets' confidence columns are no longer
    numerically comparable — see the README.
- **Taxonomy overhaul.** Crosswalk wildcards for `amenity`/`office`/`leisure`/
  `tourism` replaced with explicit inclusion sets; only `shop` and `healthcare`
  keep a wildcard. Shared labels 93 → 102.
- **Non-POI exclusions.** Unnamed POIs inside `landuse=residential` polygons,
  and unnamed POIs with `access=private|no`, are no longer published.
- **Match scoring rework.** Type affinity replaces the exact/0.5/0 type tiers,
  real identifier scoring, per-pair conditional component weights, and
  `min_match_score` raised 0.50 → 0.70.

## 2026-05-21-v0

### Snapshot inputs

| Source                 | Value                                       |
| ---------------------- | ------------------------------------------- |
| OSM snapshot date      | 2026-05-21                                  |
| Overture release       | `2026-05-20.0` (pinned)                     |
| OSM snapshot rows      | 8,799,633                                   |
| Overture snapshot rows | 13,458,763                                  |
| Boundary footprint     | US + all territories (PR, USVI, GU, MP, AS) |

### Conflated output

| Metric                       | This run    | Prior        | Δ                       |
| ---------------------------- | ----------- | ------------ | ----------------------- |
| Total rows                   | 17,989,377  | 17,788,585   | +200,792 (+1.13%)       |
| Matched OSM × Overture       | 2,696,484   | 2,677,091    | +19,393 (+0.72%)        |
| OSM-only                     | 6,103,149   | 6,031,413    | +71,736 (+1.19%)        |
| Overture-only                | 9,189,744   | 9,080,081    | +109,663 (+1.21%)       |
| Shadow-matched (CD penalty)  | 47,925      | n/a          | new — first run with change detection |
| Shared labels                | 93          | 93           | unchanged set           |

### Methods changes vs. prior release

- **Change detection (new).** Post-conflation pass that reconstructs "ghost" POIs from OSM history (deleted or renamed nodes) and uses them to penalize unmatched Overture POIs that shadow-match a ghost. Penalty multiplies the Overture row's `conf_mean` by the per-`shared_label` δ from the fitted turnover model. 47,925 rows penalized this run. Adds audit columns to every conflated row: `shadow_matched`, `shadow_ghost_id`, `shadow_event_type`, `shadow_event_timestamp`, `shadow_score`, `shadow_distance_m`, `original_conf_mean`. **PR #29**; design in `docs/change-detection.md`.
- **US territory expansion.** Spatial footprint widened from CONUS + PR to include all US territories (PR, USVI, GU, MP, AS). Affects both snapshots and the conflation domain. **PR #31**.
- **Wider metadata propagation.** Additional OSM and Overture metadata fields now flow through to the conflated parquet (website, wikidata, wikipedia, etc.). **PR #30**.
- **PMTiles re-tuned.** Zoom range narrowed to Z10–Z14 with `--drop-densest-as-needed`, so feature drops cascade through low zooms instead of failing tile builds. Site updated with zoom-aware point styling. **PR #33**.
- **Covering bbox in partitioned parquet.** GeoParquet 1.1 `bbox` struct column emitted via `write_covering_bbox=True`, enabling DuckDB row-group pruning on viewport queries. **PR #32**.
- **Overture release pinned.** `download.overture.release_date` set to `2026-05-20.0` (was `null` = auto-detect latest). Future runs against the same pin are deterministic.
- **Pipeline memory hardening (uncommitted on `lifecycle/may-2026-release`).** Both `apply_change_detection.py` and the partitioned-write helper hit the 24 GB WSL cap on nationwide inputs. The CD writer now mutates in place and streams the output parquet in row-group chunks via `pyarrow.parquet.ParquetWriter`; the geohash partition writer drops one full-partition copy (numpy `argsort` + `iloc` instead of pandas `sort_values`) and streams large partitions in chunks. See `src/openpois/conflation/change_detection.py` and `src/openpois/io/geohash_partition.py`.

### Taxonomy changes

**Overture crosswalk** (`src/openpois/conflation/data/taxonomy_crosswalk_overture_maps.csv`, uncommitted on `lifecycle/may-2026-release`): 7 new entries under `services_and_business.family_service`, previously unmapped and dropped from the partitioned output.

| Overture sub-category         | Maps to            |
| ----------------------------- | ------------------ |
| `funeral_service`             | Other Professional |
| `adoption_service`            | Other Professional |
| `family_service_center`       | Other Professional |
| `nanny_service`               | Other Professional |
| `genealogist`                 | Other Professional |
| `elder_care_planning`         | Other Professional |
| `mobility_equipment_service`  | Other Shop         |

This is the proximate cause of the +22,715 row jump (+8.45%) in **Other Professional**.

No OSM-side taxonomy changes since 2026-04-23.

### Top label-level row-count changes

| Shared label        | This run    | Prior       | Δ rows    | Δ %     | Δ matched |
| ------------------- | ----------- | ----------- | --------- | ------- | --------- |
| Specialty Store     | 1,026,395   | 917,422     | +108,973  | +11.88% | +753      |
| Other Amenity       | 3,858,315   | 3,819,068   | +39,247   | +1.03%  | +3,124    |
| Clothing Store      | 317,177     | 288,506     | +28,671   | +9.94%  | +779      |
| Other Professional  | 291,500     | 268,785     | +22,715   | +8.45%  | 0         |
| Other Healthcare    | 995,881     | 1,016,112   | −20,231   | −1.99%  | +54       |
| (unlabeled)         | 701,209     | 719,862     | −18,653   | −2.59%  | +1,506    |
| Car Dealer          | 182,314     | 164,517     | +17,797   | +10.82% | +521      |
| Restaurant          | 718,472     | 702,020     | +16,452   | +2.34%  | +1,092    |
| Supermarket         | 193,777     | 179,783     | +13,994   | +7.78%  | +361      |
| Recreation          | 1,302,776   | 1,293,338   | +9,438    | +0.73%  | −510      |

Drivers:
- Most positive movers (Specialty Store, Clothing Store, Car Dealer, Supermarket, Bakery, Charging Station) track Overture's snapshot growth (+2.5% overall) landing in shared labels with moderate base counts.
- **Other Professional** also reflects the new `family_service` crosswalk entries above.
- **Other Healthcare** dropping by ~20k against a larger Overture snapshot is worth a closer look — likely an Overture taxonomy reshuffle inside `health_and_medical` upstream. Flagged for QA, not blocking.

### Version pins

| Key                       | This run                 | Prior                    |
| ------------------------- | ------------------------ | ------------------------ |
| `versions.conflation`     | 20260521                 | 20260423                 |
| `versions.snapshot_osm`   | 20260521                 | 20260417                 |
| `versions.snapshot_overture` | 20260521              | 20260423                 |
| `versions.osm_data`       | 20260521                 | 20260515                 |
| `versions.ghost_osm`      | 20260521                 | 20260515                 |
| `versions.model_output`   | 20260422_by_shared_label | 20260422_by_shared_label (unchanged — model not refit this cycle) |
