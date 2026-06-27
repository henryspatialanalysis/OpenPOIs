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

2. **Run the downloads** (independent — order doesn't matter, can run in parallel):

   ```bash
   python scripts/osm_snapshot/download.py     # 4 Geofabrik PBFs → osm_snapshot.parquet
   python scripts/overture/download.py         # DuckDB over S3   → overture_snapshot.parquet
   python scripts/osm_data/download_history.py # 4 internal OSH PBFs → osm_versions.parquet + osm_changes.parquet
   ```
   Each loader pulls 4 extracts in sequence: `us`, `pr`, `usvi`, `american_oceania`. Per-source details, auth, and schema quirks are in [docs/data-sources.md](../../docs/data-sources.md).

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
   the random_effects model.

4. **Optional schema snapshot** — produces small CSV snippets for spec review:
   ```bash
   python scripts/snapshots/load_samples.py
   ```

## Verification

Hand off to [skills/verify-pipeline-run](../verify-pipeline-run/SKILL.md). Baseline totals (as of 2026-04-17, pre-territory-expansion):
- OSM: ~7.78M POIs
- Overture: ~13.05M POIs (jumped from ~7.23M after widening `download.overture.taxonomy_allowlist` to include `services_and_business` + `lifestyle_services` sub-branches)

**First territory-inclusive run (≥ 2026-05-21)**: expect ~20–50K additional POIs combined across the 4 new territories (GU/VI/MP/AS). The first such run has no per-territory baseline yet — record actuals in the verify-pipeline-run output so future runs have a comparison point.

Flag >5% drops against the prior run *of the same scope*. Don't compare a territory-inclusive run against a pre-expansion baseline as if a drop has occurred.

## Next

- To publish, continue with [skills/conflate-snapshots](../conflate-snapshots/SKILL.md).
- To update the frontend after publishing, continue with [skills/update-site](../update-site/SKILL.md).
