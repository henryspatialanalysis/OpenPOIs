---
name: conflate-snapshots
description: Use when the user wants to match rated OSM POIs with Overture POIs into a unified dataset, partition it for web consumption, and push to Source Cooperative. Triggers: "run conflation", "publish new data", "push new conflated data to Source Cooperative", "bump conflation version", "reconflate with new parameters", "re-upload the partitioned parquet".
---

# Conflate snapshots + publish to Source Cooperative

Taxonomy-aware matching between rated OSM and Overture, then partition and
upload for web consumption.

## Prerequisites

- Rated OSM snapshot (`osm_snapshot_rated.parquet`) at `versions.snapshot_osm` — produced by [skills/full-data-pull](../full-data-pull/SKILL.md) step 3.
- Overture snapshot (`overture_snapshot.parquet`) at `versions.snapshot_overture`.
- OSM history parquets (`osm_versions.parquet`, `osm_changes.parquet`) at `versions.osm_data` — produced by [skills/model-history-pipeline](../model-history-pipeline/SKILL.md). Required by the change-detection step in stage 4.
- **Fresh Source Cooperative temp credentials** in `.env.json` at the repo root. Tokens expire in ~1 hour.

> ⚠️ **Credential refresh check.** Source Cooperative uses short-lived AWS
> credentials (`aws_access_key_id` starting with `ASIA…`). **Before** running
> step 7, ask the user to regenerate them at
> <https://source.coop/repositories/henryspatialanalysis/openpois/manage>
> and overwrite `.env.json` at the repo root. The upload script will warn if
> the file looks stale, but it cannot tell whether the token itself has
> expired until it actually fails.

## Steps

1. **Bump `versions.conflation` and `versions.source_coop`** in `config.yaml`.
   `versions.source_coop` is the remote folder name — `YYYY-MM-DD-vN`. Keep
   `vN` at `v0`; only bump `v1`, `v2`, … if you re-upload under the same
   calendar date.

2. **Review conflation parameters** (`config.yaml` → `conflation`):
   - `min_match_score` (default 0.50) — raises/lowers match acceptance
   - `max_radius_m`, `default_radius_m` — per-label radii come from `match_radii.csv`
   - Component weights: `distance_weight`, `name_weight`, `type_weight`, `identifier_weight`
   - Changing these reshapes match counts — run with `--test` first (Seattle bbox).

3. **Sync taxonomy if crosswalks changed** — run the [sync-taxonomy](../sync-taxonomy/SKILL.md) skill. It regenerates `site/public/taxonomy.html` and `site/src/taxonomy.generated.js`, and detects drift in the hand-maintained display labels.

4. **Run the conflation pipeline.** The canonical entry point is `make conflate`, which orchestrates three sub-steps so every national run gets the OSM-history change-detection penalty automatically (see [docs/change-detection.md](../../../docs/change-detection.md) for the design and tunables):
   1. `build_ghosts.py` — reconstruct ghost POIs from OSM history (`ghosts.parquet` under `versions.ghost_osm`).
   2. `conflate.py --output-suffix=baseline` — OSM × Overture matching, writes `conflated_baseline.parquet` (no-CD archive).
   3. `apply_change_detection.py` — shadow-match unmatched Overture against the ghosts and apply the per-`shared_label` δ penalty; writes the canonical `conflated.parquet`.

   ```bash
   make conflate            # full CONUS, ~22M POIs, peak RSS ~10 GB (matching)
                            #                    + ~5 GB (CD step) projected
   make conflate TEST=1     # Seattle bbox dry run

   # Sub-targets for partial re-runs:
   make build_ghosts        # ghosts only
   make conflate_baseline   # matching only (writes conflated_baseline.parquet)
   make apply_cd            # CD pass only (reads baseline, writes conflated.parquet)
   ```

   Outputs:
   - `conflated.parquet` — canonical output that downstream steps consume (CD applied).
   - `conflated_baseline.parquet` — same shape but without the CD penalty; kept on disk for spot-checks.
   - `ghosts.parquet` under `versions.ghost_osm` — see [docs/change-detection.md](../../../docs/change-detection.md).
   - `match_diagnostics.parquet`.

5. **Match-rate sanity check**:
   ```bash
   python scripts/conflation/summarize.py
   ```
   Writes `summary_by_label.csv`.

6. **Partition for web** — geohash-4 partition, geohash-6 sort:
   ```bash
   python scripts/conflation/format_for_upload.py
   ```
   Outputs `conflated_partitioned/` (and OSM-only `osm_snapshot_partitioned/`).

6.5. **Build PMTiles** — single-zoom (z14) archives consumed directly by the
     site via `ol-pmtiles`. Intermediate FlatGeobufs are cleaned up on success.
     ```bash
     python -u scripts/osm_snapshot/prepare_pmtiles.py \
       2>&1 | tee ~/data/openpois/logs/pmtiles_osm_<version>.log
     python -u scripts/conflation/prepare_pmtiles.py \
       2>&1 | tee ~/data/openpois/logs/pmtiles_conflated_<version>.log
     ```
     Properties and zoom range are configured under `publish.pmtiles` in
     `config.yaml`.

7. **Publish to Source Cooperative** — uploads OSM + conflated parquet,
   both PMTiles, and a freshly-rendered per-version `README.md` under
   `<repo>/<versions.source_coop>/`. Confirm the credential check above first.
   ```bash
   # Preview everything that would be uploaded:
   python scripts/publish/upload_to_source_coop.py --dry-run

   # Real upload (datasets + version README):
   python -u scripts/publish/upload_to_source_coop.py \
     2>&1 | tee ~/data/openpois/logs/publish_<version>.log

   # If the top-level README or LICENSE changed:
   python scripts/publish/upload_to_source_coop.py --update-top-level
   ```
   `--skip-osm-parquet`, `--skip-conflated-parquet`, and `--skip-pmtiles`
   allow partial reuploads (e.g. after regenerating PMTiles alone).

## Verification

- `summary_by_label.csv` match rates should resemble the prior run; large drifts mean a parameter or crosswalk regression.
- `match_diagnostics.parquet` for per-pair forensics on surprising matches.
- Spot-check the version landing page at
  <https://source.coop/henryspatialanalysis/openpois/> and confirm the
  per-version `README.md` renders with the expected OSM date, Overture
  release, and row counts.
- See [skills/verify-pipeline-run](../verify-pipeline-run/SKILL.md).

## Next

- Bump the frontend: [skills/update-site](../update-site/SKILL.md).

## Key code

- Matching: [src/openpois/conflation/match.py](../../../src/openpois/conflation/match.py)
- Merging: [src/openpois/conflation/merge.py](../../../src/openpois/conflation/merge.py)
- Taxonomy assignment: [src/openpois/conflation/taxonomy.py](../../../src/openpois/conflation/taxonomy.py)
- Change-detection (ghost emission + shadow matching + R1): [src/openpois/conflation/ghost_osm.py](../../../src/openpois/conflation/ghost_osm.py), [src/openpois/conflation/change_detection.py](../../../src/openpois/conflation/change_detection.py)
- Publish orchestration: [scripts/publish/upload_to_source_coop.py](../../../scripts/publish/upload_to_source_coop.py)
- Source Coop S3 adapter: [src/openpois/io/source_coop.py](../../../src/openpois/io/source_coop.py)
- Conflation algorithm docs: [scripts/conflation/README.md](../../../scripts/conflation/README.md)
- Change-detection design: [docs/change-detection.md](../../../docs/change-detection.md)
