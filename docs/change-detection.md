# Change detection

OSM edit history is used to downweight Overture POIs whose location has seen a
recent closure / rename / lifecycle event in OSM. This is a post-processing
pass on the conflated dataset; the no-CD baseline is preserved as
``conflated_baseline.parquet`` and the CD-applied result is written to
``conflated_cd.parquet``.

.. note::

   Since 2026-07-30, change detection is **no longer the last stage**. The
   confidence-calibration step consumes ``conflated_cd.parquet`` and writes the
   canonical ``conflated.parquet`` that partition / PMTiles / publish steps
   read. Calibration must run *after* change detection, because the δ penalty
   is multiplicative on ``conf_mean``: calibrating first would leave a
   calibrated probability scaled by ~0.14. See
   ``.claude/docs/confidence-calibration.md``.

## Pipeline

Five stages, each separately runnable and individually inspectable:

```text
                  OSM history parquets
                  (osm_versions, osm_changes)
                              │
                              ▼
            1. build_ghosts.py
               for each (element, version) emit
               at most one of:
                 • hard_delete
                 • lifecycle_prefix_added
                 • primary_tag_deleted
                 • substantial_rename
                              │
                              ▼  ghosts.parquet
─── 2. conflate.py (--output-suffix=baseline) ──────────────────────────
   rated_osm  ──►  match  ──►  conflated_baseline.parquet
   overture   ─────┘            (unchanged from the no-CD pipeline)
                              │
                              ▼
─── 3. apply_change_detection.py ──────────────────────────────────────
                  conflated_baseline.parquet
                              │
                  ghosts older than 3 y dropped
                              │
                       shadow-matcher
                  (same-entity name gate, then
                   match.py composite scoring)
                              │
                              │  matches
                              ▼
                  R1: current-OSM-survivor filter
                  (drop if a live OSM element — node,
                   way or relation — with the same
                   name lives within 150 m)
                              │
                              ▼
              new_conf = old_conf × δ_group
              audit columns appended
                              │
                              ▼  conflated_cd.parquet
─── 4. calibration (fit_calibration.py + apply_calibration.py) ────────
   per-segment calibrated confidence
                              │
                              ▼  conflated.parquet (canonical)
─── 5. apply_manual_overrides.py ──────────────────────────────────────
   hand-curated exclude / include pins from the Close triage CSV,
   rewritten in place (runs LAST so a forced value is never re-scaled)
                              │
                              ▼  conflated.parquet (canonical)
─── 6. downstream ─────────────────────────────────────────────────────
   summarize.py · format_for_upload.py · prepare_pmtiles.py · publish
```

## How to run it

The Makefile target wires the sub-steps together and is the canonical
entry point for national runs:

```bash
conda activate openpois

make conflate            # full CONUS
make conflate TEST=1     # Seattle bbox dry run
```

Each sub-step writes its own log under `~/data/openpois/logs/`. Sub-targets
exist for partial re-runs:

| Target | When to use |
|---|---|
| `make build_ghosts` | Re-derive ghosts after bumping `versions.osm_data`. |
| `make conflate_baseline` | Re-run matching only; reuses the existing ghost build. |
| `make apply_cd` | Re-apply the CD penalty (e.g., after tuning the δ source or `min_prior_name_match_score`). |
| `make calibrate` | Fit + apply the confidence curves (writes the canonical `conflated.parquet`). |
| `make apply_manual_overrides` | Re-apply the manual exclude / include pins in place. Always after `calibrate`. |
| `make conflate` | End-to-end. Runs the steps above in order. |

For one-off A/B experiments outside the make flow, see
[scripts/conflation/apply_change_detection.py](../scripts/conflation/apply_change_detection.py)
— it accepts `--baseline-suffix`, `--output-suffix`, `--no-survivor-filter`,
`--min-prior-name-score` and `--max-ghost-age-years` for ablation.

## Stage details

### 1. Ghost extraction

[src/openpois/conflation/ghost_osm.py](../src/openpois/conflation/ghost_osm.py)
walks `osm_changes.parquet` in one flat pass, maintaining a per-element rolling
tag dictionary. For each `(element, version)` it emits at most one *ghost* of
the priority-ordered event types:

1. `hard_delete` — `visible` flipped `true → false`. Fires regardless of name.
2. `lifecycle_prefix_added` — a `disused:` / `was:` / `demolished:` /
   `abandoned:` / `removed:` / `razed:` key appeared. Fires regardless of
   name since 2026-09-24 (the old no-prior-name gate is gone: the shadow
   matcher's mandatory name gate now decides whether a named ghost is used).
3. `primary_tag_deleted` — a POI tag key was Deleted. Fires regardless of
   name, as above.
4. `substantial_rename` — `name` changed with `rapidfuzz.token_set_ratio < 50`
   **and** neither name is a token-level subset/superset of the other
   (guards "Walgreens" ↔ "Walgreens Pharmacy").

Every ghost also carries `new_name` — the element's name *after* the event
(`None` when it was deleted) — so the matcher can tell a rename Overture
already carries from one that left Overture stale.

Nodes only in the current implementation: ways and relations would require
geometry reconstruction beyond what the per-version parquets capture
(tracked in [.claude/TODO.md](../.claude/TODO.md)). Ghosts are **not**
age-filtered here; `apply_change_detection.py` drops those older than
`max_ghost_age_years` at load time so `ghosts.parquet` stays a pure
history product.

Critical upstream fix: the OSM history ingestion in
[src/openpois/io/osm_history_pbf.py](../src/openpois/io/osm_history_pbf.py)
uses a two-pass filter (`osmium tags-filter` → ID list → `osmium getid
--with-history`). A single `tags-filter` pass silently drops every deletion
version (those rows carry no tags), which makes `hard_delete` impossible to
observe. The two-pass approach recovers ~600 k node deletions nationwide.

### 2. Baseline conflation

[scripts/conflation/conflate.py](../scripts/conflation/conflate.py) runs the
existing matcher unchanged. The only addition is `--output-suffix=baseline`,
which writes `conflated_baseline.parquet` so the no-CD result is preserved
side-by-side with the CD-applied canonical output.

### 3. Shadow matching + penalty

[src/openpois/conflation/change_detection.py](../src/openpois/conflation/change_detection.py)
does three things:

1. **Shadow matching** — for each unmatched-Overture row from the baseline,
   find ghosts within the per-`shared_label` radius (BallTree on ghost
   centroids, haversine metric), then apply the **same-entity name gate**
   before any composite scoring. A candidate pair survives only when the
   ghost and the Overture row are the same business: the max
   `token_set_ratio` over name×name, brand×brand, name×brand and brand×name
   on **normalised** names is ≥ `min_prior_name_match_score` (70), or one
   token set contains the other. `normalise_name()` (in `ghost_osm.py`,
   shared with R1) strips accents (NFKD), lowercases, turns punctuation into
   spaces, expands `co.` → `company`, drops trailing legal-form tokens
   (`inc`, `llc`, `ltd`, `co`, `corp`, `corporation`, `company`) and, when
   both names end in a generic category token (`pharmacy`, `bank`,
   `library`, `cafe`, `coffee`, `restaurant`, `market`, `store`, `shop`,
   `salon`), compares what remains — a shared "Pharmacy" is not evidence.
   When only one side ends in such a token it is dropped only if the
   remainder still matches the other name as a whole. Unnamed ghosts and
   unnamed Overture rows never match (both names are required before the
   brand pairs are consulted). A `substantial_rename` ghost is skipped when
   its `new_name` already matches the Overture name — Overture carries the
   rename (Plaza Pharmacy → CVS Pharmacy) — so a rename demotes only when
   the *prior* name matches and the new one does not. Survivors are scored
   with the composite from
   [src/openpois/conflation/match.py](../src/openpois/conflation/match.py)
   using the production weights (`distance_weight=0.20`, `name_weight=0.40`,
   `type_weight=0.40`, `identifier_weight=0`). Type score is binary on exact
   `shared_label` equality. Greedy one-to-one above
   `min_shadow_match_score = 0.70` (raised from 0.50 after the 20260730 run;
   first exercised on 20260902).

   This deliberately reverses the May-2026 "decision rule A" choice, which
   kept the name gate at 0 so a differently named current business at a
   churned address would still be demoted, and it removes the old
   second-stage rule that *dropped* subset/superset pairs as "obvious
   same-entity matches" — under the new rule those are exactly the pairs to
   demote. The 57-POI September-2026 web-verified sample measured the loose
   matcher at 31% precision (different-name demotions 76% spurious; same-name
   62% spurious, mostly node→way merges, duplicate cleanups and renames
   Overture already carried — which the guards below target).

2. **R1 current-OSM-survivor filter** — for each surviving shadow match,
   spatial-query the live snapshot for OSM elements within 150 m of the
   Overture centroid. If any has `token_set_ratio ≥ 70` against the Overture
   name (both through `normalise_name()`), the match is dropped: the POI is
   still in OSM under different geometry / spelling and the primary matcher
   just missed it. Since 2026-09-24 the candidate set is the **full**
   filtered `osm_snapshot.parquet` (nodes, ways and relations by centroid;
   `suppress_if_current_survivor.use_full_snapshot`), not only the rated
   POIs, and the radius is 150 m (was 50 m): the sample's misses were named
   building ways 50–150 m from the Overture point (Blue Harbor Bank, Clark
   University Campus Store, Girl Scouts of Nassau County, Mendocino Masonic
   Hall). Implemented via DuckDB centroid extraction + sklearn `BallTree`
   haversine query; the rated-snapshot version cost ~90 s nationwide, ~3-5 GB
   peak memory — expect the tree to roughly double on the full snapshot.

   **Ghost age.** Before matching, ghosts whose `event_timestamp` is older
   than `max_ghost_age_years` (3) are dropped. Pre-2021 deletions were 90%
   spurious as closure evidence in the September sample.

3. **Penalty** — multiply Overture's `conf_mean` by the fitted δ for the
   ghost's `shared_label`. δ is the per-shared_label delta from the fit (read
   from `fitted_params.csv` — `delta` rows for a `random_by_type` fit, or
   `sigmoid(logit_delta_0 + eta_amenity[label])` for the production
   `random_effects` fit); falls back to `default_delta` (0.141) for groups
   absent from the fit. Audit columns are
   appended on penalized rows: `shadow_matched`, `shadow_ghost_id`,
   `shadow_event_type`, `shadow_event_timestamp`, `shadow_score`,
   `shadow_distance_m`, `original_conf_mean`.

**Memory-bounded I/O.** `apply_shadow_match` never materializes the full
conflated baseline. It reads only the columns the shadow matcher needs
(`geometry`, `source`, `shared_label`, `name`, `brand`, and the three `conf_*`
columns), computes the per-row confidence updates and audit arrays, then streams
the baseline back out row-group by row-group via `_write_cd_output` — copying the
~40 pass-through Overture attribute columns straight from disk and overwriting
only the confidence + audit columns per batch (unlabeled rows are dropped on the
way out). A single-shot `gpd.read_parquet` of the wide baseline (49 columns once
every Overture contact/address field is retained) materializes ~22 GB of shapely
geometry plus object strings and OOM-crashes the 24 GB WSL guest; the streamed
path holds one batch at a time and peaks near 10 GB. The pass-through columns are
copied as raw Arrow, so geometry is byte-preserved with no shapely round-trip.

### 4. Downstream consumption

`summarize.py`, `format_for_upload.py`, `prepare_pmtiles.py`, and
`publish/upload_to_source_coop.py` all read `conflated.parquet` by config and
require no changes. Since 2026-07-30 that file is the **calibrated** output,
not the CD output directly: change detection writes `conflated_cd.parquet` and
the calibration stage produces the canonical file from it. Two archives are
left on disk for spot-checks and ablation — `conflated_cd.parquet` (CD applied,
uncalibrated) and `conflated_baseline.parquet` (neither).

Shadow-penalized rows are deliberately **passed through the calibration
uncalibrated**, keeping the value this stage wrote, because the segment curve is
indexed on the un-penalized Overture score and applying it would silently undo
the demotion. Those rows carry `calibration_flag = 'shadow_cd'`.

## Tunables

Under `conflation.change_detection` in [config.yaml](../config.yaml):

| Knob | Default | Effect |
|---|---|---|
| `enabled` | `false` | Reserved; the production gate is the matcher itself, not this flag. |
| `min_shadow_match_score` | `0.70` | Composite score threshold for the shadow matcher (0.50 through the 20260730 run). |
| `name_change_similarity_threshold` | `50` | Below this `token_set_ratio`, a name change becomes a `substantial_rename` ghost. |
| `default_delta` | `0.141` | Fallback δ for `shared_label` values absent from the fitted model. Equals `sigmoid(logit_delta_0)` for the current fit (`logit_delta_0 = -1.807`, 20260727 random_effects; unchanged at this precision from the 20260724 fit). |
| `min_prior_name_match_score` | `70` | Same-entity name gate: the `token_set_ratio` a ghost's prior name/brand must reach against the Overture name/brand (normalised) before a penalty can fire; subset/superset pairs pass. Both names are always required. Was `0` (loose matcher) until 2026-09-24; do not lower without re-running the release gate below. |
| `max_ghost_age_years` | `3` | Ghosts older than this are dropped when `apply_change_detection.py` loads them. `0` keeps all. |
| `suppress_if_current_survivor.enabled` | `true` | R1 filter on/off. |
| `suppress_if_current_survivor.radius_m` | `150` | R1 search radius (meters); `50` until 2026-09-24. |
| `suppress_if_current_survivor.name_similarity_threshold` | `70` | R1 token_set_ratio gate (normalised names). |
| `suppress_if_current_survivor.use_full_snapshot` | `true` | R1 candidates come from the full filtered `osm_snapshot.parquet` (nodes + ways + relations); `false` uses the rated snapshot only. |
| `conflation.manual_overrides.enabled` / `.path` | `true` / `null` | Stage 5. `null` resolves the versioned `directories.manual_overrides` CSV; a missing file is a no-op. |

## Validation

### Release gate (since 2026-09-24)

Every change to the matcher rules or thresholds must clear this before the
next published run: `make conflate TEST=1` (Seattle clip) →
`scripts/conflation/diff_change_detection.py` → sample ≥ 100 demoted POIs
(all of them if fewer) → vet each one (listing actually closed or moved?)
with [vetting_viz/](../vetting_viz/) or a Sonnet fan-out (prompt pattern in
the master plan's appendix) → **precision ≥ 70% required to ship**. Record
the measured number and the rule change under "Methods changes" in
[CHANGELOG.md](../CHANGELOG.md); Close's threshold documentation reads that
section monthly. The 2026-09-24 same-entity rule has not yet been measured
against this gate — the table below predates it and describes the loose
matcher.

### Manual overrides

Anything the automated rule still gets wrong — an Overture-only closure the
gate misses, a wrong POI at any threshold, or a POI Close must carry despite
a low score — is fixed by a row in the manual overrides CSV
([src/openpois/conflation/manual_overrides.py](../src/openpois/conflation/manual_overrides.py);
columns `unified_id, overture_id, action, reason, date, report_id`,
`action ∈ exclude | include`). Rows are appended by the Close triage lane
with the report id. `exclude` forces `conf_mean = conf_lower = conf_upper
= 0` with `calibration_flag = 'manual_exclude'`; `include` forces 1 with
`'manual_include'`. The stage runs last and is idempotent.

### Last hand-vetted Seattle A/B (May 2026, 290 reviewed POIs)

| | Baseline | With CD |
|---|---|---|
| Demoted Overture rows | 0 | 293 |
| Vetted true-drops captured | — | 221 |
| Vetted false-drops still penalized | — | 58 |
| Precision (vs vetted truth) | — | 79.2 % |

The remaining ~20 % FPR is the cost of catching the broad "churn at this
address" signal — see the open per-region calibration TODO for the planned
follow-up.

## Known limits

- **Asymmetric blindness.** OSM history captures closures cleanly but is
  silent on new openings. A real closure (e.g., a node tagged "Calvary
  Chapel" was deleted) plus a different current business at the same address
  (Overture shows "Redemption Church") reads as evidence Overture is stale,
  even when Overture is right. This explained ~75 % of the residual false
  positives on the Seattle vetting set. Since 2026-09-24 the same-entity
  rule sidesteps it by never demoting a different-name pair — at the cost
  of the recall on churned addresses that decision rule A was chosen for.
  A real fix still requires data we don't ingest (Overture POI creation
  timestamps, per-region prior calibration, or ground-truth surveys).

- **Single national δ per `shared_label`.** The fitted turnover model is
  national-average, so the penalty magnitude is wrong in regions where OSM
  mapping is sparse or stale. A per-state override mechanism is tracked in
  [.claude/TODO.md](../.claude/TODO.md).

- **DuckDB v1.4.1 has a buggy `ST_Distance_Sphere`.** The bundled spherical
  distance returns values ~25 % too high at continental scale and ~30-50 %
  off at small scales. The pipeline does **not** use `ST_Distance_Sphere`
  anywhere — distance work goes through `sklearn.neighbors.BallTree` with
  the haversine metric — but any new code touching this area should avoid it
  until the DuckDB pin is bumped. Tracked in
  [.claude/TODO.md](../.claude/TODO.md).

## Vetting tool

[vetting_viz/](../vetting_viz/) is a single-page Leaflet app for hand-vetting
the demoted-POI CSV produced by `diff_change_detection.py`. Run via:

```bash
conda run -n openpois python -m http.server --directory vetting_viz 8765
# → http://localhost:8765/ → Load CSV → seattle_demoted_pois_v6.csv
```

Markers are colored by vetting status; clicking a point opens the full row
plus a radio for tagging. Export to CSV when done; reload to resume the
session.

## File map

| Path | Role |
|---|---|
| [src/openpois/io/osm_history_pbf.py](../src/openpois/io/osm_history_pbf.py) | Two-pass filter to retain deletion versions. |
| [src/openpois/conflation/ghost_osm.py](../src/openpois/conflation/ghost_osm.py) | `_scan_all_changes`, event-type detection. |
| [scripts/conflation/build_ghosts.py](../scripts/conflation/build_ghosts.py) | Stage 1 driver. |
| [src/openpois/conflation/change_detection.py](../src/openpois/conflation/change_detection.py) | Shadow matcher, R1 filter, δ penalty. |
| [scripts/conflation/apply_change_detection.py](../scripts/conflation/apply_change_detection.py) | Stage 3 driver. |
| [scripts/conflation/diff_change_detection.py](../scripts/conflation/diff_change_detection.py) | Demoted-POI CSV producer. |
| [src/openpois/conflation/manual_overrides.py](../src/openpois/conflation/manual_overrides.py) | Stage 5: manual exclude / include pins. |
| [scripts/conflation/apply_manual_overrides.py](../scripts/conflation/apply_manual_overrides.py) | Stage 5 driver (in-place, idempotent, runs last). |
| [vetting_viz/](../vetting_viz/) | Manual review UI. |
| [Makefile](../Makefile) | `make conflate` orchestrator + sub-targets. |
| [config.yaml](../config.yaml) → `conflation.change_detection` | All tunables. |

## Scoring configuration is shared with the main matcher

`apply_change_detection.py` reads the same `conflation.distance_weight` /
`name_weight` / `type_weight` / `identifier_weight` keys as `conflate.py` and
calls the same `compute_match_scores`. Retuning the main matcher therefore moves
ghost matching too — the 2026-07 reweighting (0.0/0.50/0.30/0.20 with a constant
identifier stub, to 0.20/0.40/0.40/0.00) took shadow matches from 57,760 to
62,248 (+7.8%), mean penalty factor 0.1720 to 0.1731.

The threshold is separate: `conflation.change_detection.min_shadow_match_score`,
raised 0.50 to 0.70 on 2026-07-27 alongside the main `min_match_score`.

Its scores are **not** on quite the same scale as the main matcher's. The shadow
matcher passes all-zero L0 bit arrays and supplies no type-affinity table, so its
type score falls back to binary exact `shared_label` equality; and it supplies no
identifier arrays, so it always uses the no-identifier weight set. Re-calibrate it
against its own score distribution rather than assuming the main matcher's bands
carry over. See [../.claude/docs/match-scoring.md](../.claude/docs/match-scoring.md).
