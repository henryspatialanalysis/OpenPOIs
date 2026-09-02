---
name: full-data-pull
description: Use when the user wants to refresh the independent POI snapshots (OSM, Overture) and rate the OSM snapshot for conflation. Triggers: "refresh all snapshots", "do a new data pull", "download new OSM/Overture", "monthly data refresh", "pull the latest POI data". Does NOT include conflation or Source Cooperative publishing — those live in conflate-snapshots.
---

# Full data pull

Downloads the snapshot sources (50 US states + DC + 5 inhabited territories: PR, VI, GU, MP, AS), refreshes the OSM history that drives ghost reconstruction, and applies the rating model to OSM so conflation can run.

## Prerequisites

- conda env `openpois` active.
- For OSM: `osmium` in env bin (resolved automatically via `Path(sys.executable).parent / "osmium"`).
- Boundary cache at `directories.boundary` (auto-downloads on first use).
- A fitted model exists for the OSM rating step (see [skills/model-history-pipeline](../model-history-pipeline/SKILL.md)).
- For OSM history: a fresh Geofabrik OAuth cookie file at `download.osm.history_cookie_file` (Netscape format; any OSM account works). See [docs/data-sources.md](../../docs/data-sources.md#osm-history-geofabrik-full-history-pbfs).

## Steps

1. **Bump versions in `config.yaml`** — sources release on independent cadences, don't force them to match:
   ```yaml
   versions:
     snapshot_osm: "YYYYMMDD"
     snapshot_overture: "YYYYMMDD"
     osm_data: "YYYYMMDD"     # bumps each month — history is refreshed for ghosts
     ghost_osm: "YYYYMMDD"    # pinned to osm_data; bumps in lockstep
   ```
   `model_output` does **not** bump unless you're re-fitting λ from scratch (see [skills/model-history-pipeline](../model-history-pipeline/SKILL.md)). See [docs/data-versioning.md](../../docs/data-versioning.md).

   **Also advance `download.osm.end_date` each month** to this run's cutoff — the Overture release date is a convenient anchor (e.g. `2026-07-22`). It bounds the osmium time-filter window for the history download, so leaving it pinned to a fixed past date silently drops every edit after that date from **both** ghost reconstruction and any λ refit — the history stops moving forward even though you re-download it. `start_date` stays fixed at `2016-01-01`.

2. **Run the downloads** (independent — order doesn't matter, can run in parallel):

   ```bash
   python scripts/osm_snapshot/download.py     # 4 Geofabrik PBFs → osm_snapshot.parquet + landuse_residential.parquet
   python scripts/overture/download.py         # DuckDB over S3   → overture_snapshot.parquet
   python scripts/osm_data/download_history.py # 4 internal OSH PBFs → osm_versions.parquet + osm_changes.parquet
   ```
   Each loader pulls 4 extracts in sequence: `us`, `pr`, `usvi`, `american_oceania`. Per-source details, auth, and schema quirks are in [docs/data-sources.md](../../docs/data-sources.md).

   **After the Overture download, run the monthly confidence-drift gate** (quick, ~2 min):
   ```bash
   python scripts/overture/compare_confidence.py   # prior month auto-detected
   ```
   It compares confidence on matched GERS ids vs the prior local snapshot and prints the
   decision metrics. The pass/breach rule and what each outcome means for the
   calibration stage live in
   [docs/confidence-calibration.md](../../docs/confidence-calibration.md) ("Curves do
   not transport across releases"). Also run
   `python scripts/overture/compare_taxonomy.py --strict` **before** bumping config /
   downloading — it is the schema + crosswalk-drift pre-flight (see
   [docs/taxonomy-setup.md](../../docs/taxonomy-setup.md)).

   **Gotcha — interrupted snapshot runs**: all 4 extracts share `~/data/openpois/snapshots/osm/<v>/parse_chunks/`. If a run dies between extracts, leftover chunks from extract N may be silently mistaken for extract N+1's parsed output on resume (the parser short-circuits on existing chunks). Before resuming an interrupted snapshot run, nuke the work dir:
   ```bash
   rm -rf ~/data/openpois/snapshots/osm/{version}/parse_chunks/
   ```
   This forces a clean re-parse of whichever extract was in flight; completed extracts (which write their own per-extract intermediate parquet next to the final output) are still skipped.

   **Gotcha — `download_history.py` is for ghost regeneration only**: do **not** re-run `scripts/osm_data/format_tabular.py` or `scripts/models/osm_turnover.py` in the monthly cycle — those are part of the model-fit pipeline, which stays pinned to `versions.model_output`. The monthly history refresh only feeds `build_ghosts.py` (invoked by `make conflate`).

   **Gotcha — per-territory 404 tolerance**: if Geofabrik stops publishing a territory's `*-internal.osh.pbf`, the loader logs a warning, skips that extract, and continues. The territory's POIs still flow through downstream stages but the rater falls back to the global-mean δ for its `shared_label`s.

3. **Apply the rating model to OSM** → `osm_snapshot_rated.parquet`:
   ```bash
   make rate     # scripts/osm_snapshot/apply_model_random_effects.py
   ```
   Rates each POI from its own `(shared_label, MSA, urban_rural)` cell using the
   production `random_effects` fit at `{osm_data.apply_model.model_stub}_by_shared_label`.
   **Do not use `apply_model.py`** — it's the legacy per-group rater and mis-rates
   the random_effects model. The rater refuses to run against a fit trained on a
   POI subsample; see [skills/model-history-pipeline](../model-history-pipeline/SKILL.md).

   > **Both non-POI exclusions run inline at snapshot build** — confirmed on the
   > 2026-09 pull (the first under the new flow), so the old post-hoc scripts
   > (`apply_access_exclusion.py`, `apply_residential_exclusion.py`) are needed
   > only for snapshots built before 2026-07-26/27. Note the access exclusion is
   > **silent** (no log line); verify it with a count instead — unnamed
   > `access=private|no` rows in the snapshot must be 0. The residential
   > exclusion logs `Residential exclusion: N rows kept`. See the "Exclusion"
   > section of [docs/data-sources.md](../../docs/data-sources.md).

4. **Optional schema snapshot** — produces small CSV snippets for spec review:
   ```bash
   python scripts/snapshots/load_samples.py
   ```

## Verification

Hand off to [skills/verify-pipeline-run](../verify-pipeline-run/SKILL.md). Baseline totals (as of 2026-04-17, pre-territory-expansion):
- OSM: ~7.78M POIs
- Overture: ~13.05M POIs (jumped from ~7.23M after widening `download.overture.taxonomy_allowlist` to include `services_and_business` + `lifestyle_services` sub-branches)

**Current baselines (2026-09 refresh).** Both exclusions now run at snapshot
build, so the snapshot and rated files carry the same count:

| stage | rows |
|---|--:|
| parse + inline access exclusion | 4,763,499 |
| after inline residential landuse (final = rated) | 4,508,280 (−5.4%) |

The September count is **−5.4% vs July's 4,764,221** for a one-time reason: it
was the first pull with the value-scoped ingest filter applied at PBF-filter
time, which removed non-destination object values (`tourism=information` −77k,
`tourism=picnic_site` −22k, and kin) that July still ingested and only
unlabelled later. From October onward expect small month-over-month *growth*
from ~4.51M; a >5% swing at the same stage is again a flag. The history
parquets took the same one-time narrowing (`osm_versions` 21.2M → 15.1M rows),
but **labeled ghosts were stable** (596,255 → 602,432) — compare labeled
ghosts, not raw history rows.

Overture 20260819: 13,785,024 (+9.3% on July's 12,606,804; conflation-time
internal dedup removed 864,165 of those). The Overture extraction gained
`brand_wikidata` on 2026-07-26 — a snapshot without it still conflates, but
identifier scoring loses the Wikidata clause.

Conflated output for reference: `20260730` = 14,613,331 rows (matched 1,787,072 /
OSM-only 2,678,337 / Overture-only 10,147,922).

**First territory-inclusive run (≥ 2026-05-21)**: expect ~20–50K additional POIs combined across the 4 new territories (GU/VI/MP/AS). The first such run has no per-territory baseline yet — record actuals in the verify-pipeline-run output so future runs have a comparison point.

Flag >5% drops against the prior run *of the same scope*. Don't compare a territory-inclusive run against a pre-expansion baseline as if a drop has occurred.

## Next

- To publish, continue with [skills/conflate-snapshots](../conflate-snapshots/SKILL.md).
- To update the frontend after publishing, continue with [skills/update-site](../update-site/SKILL.md).
