# Changelog

## 2026-10-01-v0

### Snapshot inputs

| Source                 | Value                                       |
| ---------------------- | ------------------------------------------- |
| OSM snapshot date      | 2026-09-29                                  |
| Overture release       | `2026-09-23.1` (pinned)                     |
| OSM snapshot rows      | 4,535,124                                   |
| Overture snapshot rows | 15,003,463                                  |
| Boundary footprint     | US + all territories (PR, USVI, GU, MP, AS) |

### Conflated output

| Metric                       | This run    | Prior        | Δ                       |
| ---------------------------- | ----------- | ------------ | ----------------------- |
| Total rows                   | 16,591,754  | 15,586,580   | +1,005,174 (+6.45%)     |
| Matched OSM × Overture       | 1,855,640   | 1,817,729    | +37,911 (+2.09%)        |
| OSM-only                     | 2,673,948   | 2,685,038    | −11,090 (−0.41%)        |
| Overture-only                | 12,062,166  | 11,083,813   | +978,353 (+8.83%)       |
| Shadow-matched (CD penalty)  | 25,412      | 31,411       | −5,999 (−19.10%)        |
| Shared labels                | 102         | 102          | —                       |

Growth again follows the upstream Overture release (+8.8% rows month-over-month,
of which 1,066,183 were removed by internal dedup, against 864,165 last month).
The shadow-matched count falls with the stricter same-entity rule below.

Confidence moves against 2026-09-02-v0, from the new calibration (unflagged
rows): Overture-only mean `conf_mean` 0.683 → 0.644 (60.4% of POIs move by more
than 0.05, 3.9% by more than 0.10); OSM-only 0.817 → 0.817; matched 0.908 →
0.903 (9.8% and 2.1%).

### Methods changes vs. prior release

- **New `overture-pmtiles/` folder (added 2026-10-02).** `overture.pmtiles`
  tiles this release's Overture input snapshot, and
  `overture_categories.json` counts its POIs by taxonomy L0 and
  `basic_category`. The tile properties follow Overture's current Places
  schema: `basic_category` for filtering, plus `taxonomy_primary` and the full
  `taxonomy_hierarchy` path. The files were added to the published version.
- **All three PMTiles archives rebuilt (2026-10-03) so no POI is dropped.**
  The original archives stopped at z14, and tippecanoe's density-based
  dropping kept only 29% of conflated POIs in a Midtown Manhattan z14 tile
  (44% in downtown LA, 75% in north Seattle). Deeper views only enlarge z14,
  so the dropped POIs were missing at every zoom. The archives now add zoom
  levels past z14 until the top zoom drops nothing. The parquet datasets are
  unchanged.
- **`overture_categories_alternate` follows Overture's new taxonomy.** Overture
  removed its deprecated `categories` field in release 2026-09-23.1, so the
  column now holds `taxonomy.alternates`: the same idea (secondary categories),
  in the new taxonomy's vocabulary. Matching never read it.
- **An Overture confidence of exactly 0.5 is an ordinary score.** Earlier
  releases flagged these rows `calibration_flag = 'missing_conf'`, on the
  belief that conflation had imputed 0.5 for a missing provider confidence.
  No Overture release we have ingested has had a missing confidence (0, 1,178
  and 2,767 rows at exactly 0.5 in the June, July and August 2026 releases,
  all Overture's own values), so the flag is gone and those rows are
  calibrated like any other Overture row. The ingest now stops on a missing or
  out-of-range confidence instead of filling a placeholder.
- **Monthly ghost history rolls forward from daily diffs.** When the turnover
  model is not refit, `download.osm.history_mode: incremental` (now the
  default) builds `osm_data` by rolling last month's history parquets forward
  with Geofabrik's public daily diffs, about 0.4 GB a month instead of the
  23.4 GB full-history extract. Ghost building is unchanged. Two differences
  from a full build: same-day edits collapse into one version, and deletions
  are stamped with the diff's cut time (median 13 h, at most 24 h after the
  real deletion). Backtest (roll `osm_data/20260724` forward 2026-07-13 →
  2026-08-18, then compare with the full `20260902` build on shared nodes):
  named, labeled ghosts 99.06% recall and 99.25% precision (gate ≥ 97%).
  Renames are the weakest type (98.0% / 95.5%) because bursts of same-day
  renames collapse. The rolled history matched the September snapshot's node
  state better than the full build did (99.99% vs 99.91% timestamp match).
  Rolled history is not used for refits.
- **Confidence is calibrated by three Bayesian monotone-spline models.** The
  published `conf_mean` is the posterior mean of P(the POI exists and is open)
  from one model per detection segment: a monotone curve over the Overture
  confidence for Overture-only POIs, a monotone curve over the OSM turnover
  score for OSM-only POIs, and a surface over both scores, monotone in each,
  for matched POIs. The models are fitted in JAX
  (`openpois.conflation.calibration_bayes`) to the validation sample. Gold
  rows count as labelled; rows with only an LLM verdict enter through a
  mixture likelihood whose verdict rates, P(verdict | exists) and P(verdict |
  gone) over all three verdicts (unverifiable included), are fixed from the
  gold, design-weighted. Rates taken among definitive verdicts only would
  bias the curves low, since closed POIs are far more often unverifiable
  (Overture: 56% of closed POIs get a definitive verdict, against 90% of open
  ones); the corrected rates raise the Overture curve by about 0.02. The July
  and October 2026 validation rounds are pooled. All three fits pass the
  acceptance rule (no divergences, R̂ ≤ 1.005, bulk and tail ESS ≥ 1,300);
  the matched fit samples at target acceptance 0.998. Against the pooled
  gold, design-weighted, the deployed map is calibrated in the large in every
  segment (Overture 0.642 against a gold rate of 0.640). It cannot follow one
  feature of the Overture gold: POIs scored below about 0.29 exist more often
  (0.67) than those scored 0.29 to 0.85 (0.40 to 0.49), which a monotone curve
  must average out. `conf_lower` / `conf_upper` are the 95% posterior interval; for
  matched POIs it is too narrow (about 0.75 coverage in simulation), a known
  limitation to be fixed. This replaces the v4 binned curves of the July and
  September releases, which are retired; the matched interaction index and
  bin-level bands that had been planned for this release were superseded
  before shipping. In cross-validation on round 20260730 the same model with
  fixed fractional labels instead of the mixture tied the v4 curves (pooled
  relative Brier 0.996, 95% interval 0.990 to 1.002); the mixture form has not
  been cross-validated. Each release now also carries a design-weighted review of the
  deployed map (`calibration/ht_review_<round>.pdf`), which compares it with
  the validation sample's rates, silver labels corrected for
  misclassification.
- **Change detection demotes only same-entity ghosts.** The shadow matcher
  now requires the ghost's prior name / brand to match the Overture name /
  brand (`min_prior_name_match_score` 0 → 70 on normalised names: accents,
  case, punctuation, legal suffixes and a shared trailing category token
  removed; token subset / superset pairs pass). This reverses the May-2026
  "decision rule A" choice and deletes the rule that *dropped*
  subset/superset pairs as obvious same-entity matches. Motivation: a 57-row
  web-verified sample of 2026-09-02-v0 `shadow_cd` rows measured the loose
  matcher at 31% precision (different-name demotions 76% spurious, same-name
  62% spurious). Guards shipped with it: unnamed ghosts never demote; a
  `substantial_rename` ghost is skipped when Overture already carries the
  new name; the current-OSM-survivor filter searches the full filtered
  snapshot (nodes, ways, relations) within 150 m (was rated POIs within
  50 m); ghosts older than 3 years are dropped at run time; named
  `lifecycle_prefix_added` / `primary_tag_deleted` ghosts are now emitted.
  This release has 25,412 shadow-matched rows, against 31,411 in
  2026-09-02-v0.
  **Release gate deferred.** The planned gate, ≥ 70% precision (listing
  actually closed / moved) on a hand- or LLM-vetted sample of ≥ 100 demoted
  POIs, was not measured before this release, so the precision of the new
  matcher is unmeasured. The vetted sample is carried to the next release.
- **Manual confidence overrides, applied after calibration.** New last
  stage of `make conflate` (`apply_manual_overrides.py`) reads a
  hand-curated CSV (`unified_id, overture_id, action, reason, date,
  report_id`) appended by the Close triage lane and forces `exclude` rows
  to `conf_mean = conf_lower = conf_upper = 0` (`calibration_flag =
  'manual_exclude'`) and `include` rows to 1 (`'manual_include'`). Rows are
  kept, so schema and counts are unchanged; the two new flag values join
  `shadow_cd` and `unnamed_extrapolated`. No overrides were applied in this
  release.
- **Type-affinity table rebuilt** against the 2026-09-23.1 category hierarchy
  from the 2026-09-02-v0 matches.

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
