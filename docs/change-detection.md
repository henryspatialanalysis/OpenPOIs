# Change detection

OSM edit history is used to downweight Overture POIs whose location has seen a
recent closure / rename / lifecycle event in OSM. This is a post-processing
pass on the conflated dataset; the no-CD baseline is preserved as
``conflated_baseline.parquet`` and the CD-applied result becomes the canonical
``conflated.parquet`` that downstream partition / PMTiles / publish steps
consume.

## Pipeline

Four stages, each separately runnable and individually inspectable:

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
                       shadow-matcher
                  (reuses match.py scoring)
                              │
                              │  matches
                              ▼
                  R1: current-OSM-survivor filter
                  (drop if a live OSM POI with the
                   same name lives within 50 m)
                              │
                              ▼
              new_conf = old_conf × δ_group
              audit columns appended
                              │
                              ▼  conflated.parquet (canonical)
─── 4. downstream (unchanged) ─────────────────────────────────────────
   summarize.py · format_for_upload.py · prepare_pmtiles.py · publish
```

## How to run it

The Makefile target wires the three new sub-steps together and is the
canonical entry point for national runs:

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
| `make conflate` | End-to-end. Runs the three steps above in order. |

For one-off A/B experiments outside the make flow, see
[scripts/conflation/apply_change_detection.py](../scripts/conflation/apply_change_detection.py)
— it accepts `--baseline-suffix`, `--output-suffix`, `--no-survivor-filter`,
and `--min-prior-name-score` for ablation.

## Stage details

### 1. Ghost extraction

[src/openpois/conflation/ghost_osm.py](../src/openpois/conflation/ghost_osm.py)
walks `osm_changes.parquet` in one flat pass, maintaining a per-element rolling
tag dictionary. For each `(element, version)` it emits at most one *ghost* of
the priority-ordered event types:

1. `hard_delete` — `visible` flipped `true → false`. Fires regardless of name.
2. `lifecycle_prefix_added` — a `disused:` / `was:` / `demolished:` /
   `abandoned:` / `removed:` / `razed:` key appeared. Only fires when the
   prior state was un-named (avoids the noise of named lifecycle retagging).
3. `primary_tag_deleted` — a POI tag key was Deleted. Same no-prior-name gate.
4. `substantial_rename` — `name` changed with `rapidfuzz.token_set_ratio < 50`
   **and** neither name is a token-level subset/superset of the other
   (guards "Walgreens" ↔ "Walgreens Pharmacy").

Nodes only in the current implementation: ways and relations would require
geometry reconstruction beyond what the per-version parquets capture.

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
   centroids, haversine metric). Reuses the composite scoring from
   [src/openpois/conflation/match.py](../src/openpois/conflation/match.py)
   with `distance_weight=0`, `name_weight=0.5`, `type_weight=0.3`,
   `identifier_weight=0.2`. Type score is binary on exact `shared_label`
   equality. Greedy one-to-one above `min_shadow_match_score = 0.50`.
   Subset/superset name pairs are dropped as obvious same-entity matches.

2. **R1 current-OSM-survivor filter** — for each surviving shadow match,
   spatial-query the live rated snapshot for OSM POIs within 50 m of the
   Overture centroid. If any has `token_set_ratio ≥ 70` against the Overture
   name, the match is dropped. The POI is still in OSM under different
   geometry / spelling and the primary matcher just missed it. Implemented
   via DuckDB centroid extraction + sklearn `BallTree` haversine query;
   nationwide cost ~90 s, ~3-5 GB peak memory.

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
require no changes. They now consume the CD-applied output by default. The
no-CD archive (`conflated_baseline.parquet`) is left on disk for spot-checks
and ablation.

## Tunables

Under `conflation.change_detection` in [config.yaml](../config.yaml):

| Knob | Default | Effect |
|---|---|---|
| `enabled` | `false` | Reserved; the production gate is the matcher itself, not this flag. |
| `min_shadow_match_score` | `0.50` | Composite score threshold for the shadow matcher. |
| `name_change_similarity_threshold` | `50` | Below this `token_set_ratio`, a name change becomes a `substantial_rename` ghost. |
| `default_delta` | `0.141` | Fallback δ for `shared_label` values absent from the fitted model. Equals `sigmoid(logit_delta_0)` for the current fit (`logit_delta_0 = -1.809`, 20260724 random_effects). |
| `min_prior_name_match_score` | `0` | Hard gate on Overture-vs-prior-name `token_set_ratio` before any composite scoring. **Leave at 0** — values ≥ 70 produce high precision but miss real closures where Overture has updated to a different current business name at a churned address. |
| `suppress_if_current_survivor.enabled` | `true` | R1 filter on/off. |
| `suppress_if_current_survivor.radius_m` | `50` | R1 search radius (meters). |
| `suppress_if_current_survivor.name_similarity_threshold` | `70` | R1 token_set_ratio gate. |

## Validation

Last hand-vetted Seattle A/B (May 2026, 290 reviewed POIs):

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
  even when Overture is right. This explains ~75 % of the residual false
  positives on the Seattle vetting set and is intrinsic — fixing it requires
  data we don't currently ingest (Overture POI creation timestamps,
  per-region prior calibration, or ground-truth surveys).

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
