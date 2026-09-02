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
- OSM history parquets (`osm_versions.parquet`, `osm_changes.parquet`) at `versions.osm_data` — **regenerated each month** by [skills/full-data-pull](../full-data-pull/SKILL.md) step 2 (via `scripts/osm_data/download_history.py`). The full re-fit pipeline at [skills/model-history-pipeline](../model-history-pipeline/SKILL.md) is only invoked when re-fitting λ. Required by the change-detection step in stage 4.
- **A live Source Cooperative login.** Credentials come from the `source-coop` CLI, not from a file. Tokens last ~1 hour.

> ⚠️ **Authenticate before step 7.** Source Cooperative moved to OIDC/STS: the
> `source-coop` CLI mints short-lived credentials after a browser login, and
> `openpois.io.credentials` reads them directly via `source-coop creds`. The
> CLI is auth-only — it has no upload or sync command, so our own upload script
> still does the transfer.
>
> ```bash
> # once per machine
> curl --proto '=https' --tlsv1.2 -LsSf https://github.com/source-cooperative/source-coop-cli/releases/latest/download/source-coop-cli-installer.sh | sh
>
> # once per session (opens a browser; ~1 hour of validity)
> source-coop login
> source-coop creds >/dev/null && echo "authenticated"
> ```
>
> The CLI cannot open a browser under WSL, so it prints a URL instead — hand
> that to the user to open. Mirrored networking (`.wslconfig`
> `networkingMode=mirrored`) lets the Windows browser reach the local callback
> port. `source-coop login` cannot be run for the user; it needs their
> interaction.
>
> **Budget the token against the payload.** A full release is ~7 GB, so a
> ~1-hour token needs roughly 2 MB/s sustained. Check the expiry that
> `source-coop creds` reports before starting, and re-`login` if it is close.
> If it does expire mid-transfer, `latest/` is untouched (it mirrors last) and
> the `--skip-*` flags resume at dataset granularity.
>
> `.env.json` remains a fallback for a static credential block, but it is not
> the normal path any more and Source Coop no longer issues keys that way.

## Steps

1. **Bump `versions.conflation` and `versions.source_coop`** in `config.yaml`.
   `versions.source_coop` is the remote folder name — `YYYY-MM-DD-vN`. Keep
   `vN` at `v0`; only bump `v1`, `v2`, … if you re-upload under the same
   calendar date.

2. **Review conflation parameters** (`config.yaml` → `conflation`):
   - `min_match_score` (**0.70** since 2026-07-26) — raises/lowers match acceptance.
     Set from a precision-by-band review; see [docs/match-scoring.md](../../docs/match-scoring.md).
   - `max_radius_m`, `default_radius_m` — per-label radii come from `match_radii.csv`
   - Component weights come in **two sets**, chosen per candidate pair:
     `weights_with_identifier` when both sides carry a website/phone/wikidata,
     else the flat `distance_weight` / `name_weight` / `type_weight` /
     `identifier_weight`. Both sum to 1.0.
   - `type_affinity_k` — shrinkage for the type-affinity table (rebuild after changing).
   - Changing any of these reshapes match counts — run with `--test` first (Seattle
     bbox; writes `conflated_test.parquet`, not the production file).

3. **Sync taxonomy if crosswalks changed** — run the [sync-taxonomy](../sync-taxonomy/SKILL.md) skill. It regenerates `site/public/taxonomy.html` and `site/src/taxonomy.generated.js`, and detects drift in the hand-maintained display labels.

4. **Run the conflation pipeline.** The canonical entry point is `make conflate`, which orchestrates four stages so every national run gets both the OSM-history change-detection penalty and the confidence calibration automatically (see [docs/change-detection.md](../../../docs/change-detection.md) and [docs/confidence-calibration.md](../../docs/confidence-calibration.md)):
   1. `build_ghosts.py` — reconstruct ghost POIs from OSM history (`ghosts.parquet` under `versions.ghost_osm`).
   2. `conflate.py --output-suffix=baseline` — OSM × Overture matching, writes `conflated_baseline.parquet` (no-CD archive).
   3. `apply_change_detection.py` — shadow-match unmatched Overture against the ghosts and apply the per-`shared_label` δ penalty; writes `conflated_cd.parquet`.
   4. `fit_calibration.py` + `apply_calibration.py` + `plot_calibration.py` — fit the per-segment existence-confidence curves from the validation handoff and map every POI through them; writes the canonical `conflated.parquet`.

   ```bash
   make conflate            # full CONUS; peak RSS measured 21.9 GB on 20260902
                            # (pre-merge reload of both frames; 13.8M Overture
                            # rows) against the 24 GB WSL cap — see TODO.md
   make conflate TEST=1     # Seattle bbox dry run

   # Sub-targets for partial re-runs:
   make build_ghosts        # ghosts only
   make conflate_baseline   # matching only (writes conflated_baseline.parquet)
   make apply_cd            # CD pass only (reads baseline, writes conflated_cd.parquet)
   make calibrate           # fit + apply + plot (reads conflated_cd, writes conflated.parquet)
   make fit_calibration     # curves only — safe to iterate, touches no POI data
   ```

   **Calibration must follow change detection, never precede it** — CD multiplies
   `conf_mean` by δ (≈0.14), so calibrating first would leave a calibrated probability
   scaled by δ.

   **Whether to fit at all is governed by the monthly confidence-drift gate**
   (`scripts/overture/compare_confidence.py`, run during the data pull; decision rule in
   [docs/confidence-calibration.md](../../docs/confidence-calibration.md)). On a
   **pass** (the normal monthly case), do **not** run `fit_calibration` — reuse the most
   recent fitted curves verbatim:
   ```bash
   # copy curves + metadata from the prior conflation version, with a provenance note
   python scripts/conflation/apply_calibration.py --input-suffix cd --output-suffix "" \
       --curves-dir ~/data/openpois/conflation/<prior version>/calibration
   ```
   On a **breach**, refresh the validation handoff, pin it, and refit:
   ```bash
   cd ~/repos/openpois-validator && python scripts/08_export_handoff.py
   # then set versions.calibration in config.yaml to that round, and run make calibrate
   ```
   The handoff lands in the gitignored `data/calibration/<round>/`. If it is missing,
   `fit_calibration.py` fails fast rather than shipping uncalibrated data.

   Outputs:
   - `conflated.parquet` — canonical output that downstream steps consume (CD + calibration applied). `conf_mean`/`conf_lower`/`conf_upper` are calibrated P(exists and open); `conf_mean_uncalibrated` archives the post-CD value; `calibration_flag` records edge rules.
   - `conflated_cd.parquet` — post-CD, pre-calibration.
   - `conflated_baseline.parquet` — neither CD nor calibration; kept on disk for spot-checks.
   - `calibration/` — fitted curves, per-segment metadata, and `fit_report.md`.
   - `ghosts.parquet` under `versions.ghost_osm` — see [docs/change-detection.md](../../../docs/change-detection.md).
   - `match_diagnostics.parquet`.

4b. **Rebuild the type-affinity table** if the taxonomy changed, or on the
   first run after a new Overture release:
   ```bash
   python scripts/conflation/build_type_affinity.py --conflated <PREVIOUS run's conflated.parquet>
   ```
   Writes `src/openpois/conflation/data/type_affinity.csv`, the derived type
   score. Calibrated from the *previous* run's identifier-confirmed matches, so
   pass `--conflated` explicitly once `versions.conflation` points at the run
   being produced. See [docs/type-affinity-metric.md](../../docs/type-affinity-metric.md).
   Skip it if neither the taxonomy nor the release changed — the table is
   checked in.

5. **Match-rate sanity check**:
   ```bash
   python scripts/conflation/summarize.py
   ```
   Writes `summary_by_label.csv` and `match_status_by_label.csv` (the published
   per-label table: matched / OSM-only / Overture-only / total / match %). Stdout
   calls out every **single-source label** — a zero in any column means that label
   can never match, which is almost always a crosswalk gap rather than a real
   property of the data. Only `Car Rental` is expected to be single-source
   (Overture has no car-rental category).

6. **Partition for web** — geohash-4 partition, geohash-6 sort:
   ```bash
   python scripts/conflation/format_for_upload.py
   ```
   Outputs `conflated_partitioned/` (and OSM-only `osm_snapshot_partitioned/`).

6.5. **Build PMTiles** — multi-zoom (z10–z14, `drop-densest-as-needed`; see
     `publish.pmtiles` in config.yaml) archives consumed directly by the site
     via `ol-pmtiles`. Intermediate FlatGeobufs are cleaned up on success.
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
   `<repo>/<versions.source_coop>/`, then mirrors the whole version to
   `latest/`. Confirm the authentication check above first.
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
   allow partial reuploads (e.g. after regenerating PMTiles alone), and
   `--skip-latest-mirror` holds `latest/` on the previous release.

   **Writes go through the data proxy, reads do not.** Uploads use
   `endpoint_url = https://data.source.coop` with the **account as the bucket**
   (`henryspatialanalysis`) and the repository as the first key segment
   (`openpois/<version>/…`). Anonymous public *reads* still work against the
   legacy flat bucket `us-west-2.opendata.source.coop` with
   `henryspatialanalysis/openpois/…` keys, which is what the published README's
   quickstart examples use — so do not "fix" those to match the write path.

## Verification

- `summary_by_label.csv` match rates should resemble the prior run; large drifts mean a parameter or crosswalk regression.
- **Score distribution.** `match_score` should span roughly the threshold to 1.0 with a healthy top decile. If nothing exceeds 0.9, or everything piles into one band, a scoring component is dead — that is exactly how the constant-0.5 identifier stub hid for months. See [docs/match-scoring.md](../../docs/match-scoring.md).
- **Precision spot-check after any scoring change.** Sample 30 matches per score band and eyeball them; the bands are only meaningful if someone has looked. Use `ORDER BY hash(unified_id) LIMIT 30` inside the band filter — DuckDB's `USING SAMPLE n ROWS` can apply before the `WHERE` and return a single row per band.
- The conflate log's `OSM: X/N assigned` line should be ~100%. A big shortfall means tag columns are missing from the load — `OSM_MATCH_COLS` must cover every key in `download.osm.filter_keys`, since `assign_osm_shared_label` skips absent columns silently and `drop_unlabeled` then deletes those POIs.
- `match_diagnostics.parquet` for per-pair forensics on surprising matches.
- Spot-check the version landing page at
  <https://source.coop/henryspatialanalysis/openpois/> and confirm the
  per-version `README.md` renders with the expected OSM date, Overture
  release, and row counts.
- See [skills/verify-pipeline-run](../verify-pipeline-run/SKILL.md).

## Next

- **Always bump the frontend after a successful upload** — the site's PMTiles
  URLs are pinned to a version folder and do **not** auto-follow new uploads, so
  a refresh isn't complete until the frontend points at the new
  `versions.source_coop`. Treat this as a required step of every monthly refresh,
  not an optional follow-up: [skills/update-site](../update-site/SKILL.md).

## Key code

- Matching: [src/openpois/conflation/match.py](../../../src/openpois/conflation/match.py)
- Merging: [src/openpois/conflation/merge.py](../../../src/openpois/conflation/merge.py)
- Taxonomy assignment: [src/openpois/conflation/taxonomy.py](../../../src/openpois/conflation/taxonomy.py)
- Change-detection (ghost emission + shadow matching + R1): [src/openpois/conflation/ghost_osm.py](../../../src/openpois/conflation/ghost_osm.py), [src/openpois/conflation/change_detection.py](../../../src/openpois/conflation/change_detection.py)
- Publish orchestration: [scripts/publish/upload_to_source_coop.py](../../../scripts/publish/upload_to_source_coop.py)
- Source Coop S3 adapter: [src/openpois/io/source_coop.py](../../../src/openpois/io/source_coop.py)
- Conflation algorithm docs: [scripts/conflation/README.md](../../../scripts/conflation/README.md)
- Change-detection design: [docs/change-detection.md](../../../docs/change-detection.md)
