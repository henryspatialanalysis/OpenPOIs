---
name: model-history-pipeline
description: Use when the user wants to run the full OSM-history turnover model from scratch — downloading Geofabrik full-history PBFs, formatting observations, fitting lambda, and applying predictions to the current snapshot. Triggers: "run the full history pipeline", "refit the model from scratch", "bump osm_data version and rerun", "new model run", "rerun history modeling for <date>".
---

# Full model history pipeline

End-to-end: Geofabrik full-history PBFs → observations table → fitted λ → rated OSM snapshot.

## Prerequisites

- Geofabrik OAuth cookies at `download.osm.history_cookie_file` (Netscape format). Any OSM account works; export via browser login or `oauth_cookie_client.py`. See [docs/data-sources.md](../../docs/data-sources.md#osm-history-geofabrik-full-history-pbfs).
- conda env `openpois` active; `osmium` is in the env's `bin/` (not PATH).
- `versions.osm_data` in `config.yaml` set to the target `YYYYMMDD` (bump if this is a new run).

## Steps

1. **Download full-history PBFs** → `osm_versions.parquet` + `osm_changes.parquet`
   ```bash
   python scripts/osm_data/download_history.py
   ```
   Runs `osmium tags-filter --omit-referenced` then `osmium time-filter`, then pyosmium streams results. Iterates the 4 extracts (`us`, `pr`, `usvi`, `american_oceania`) and applies N-way iterative `(type, id)` dedup before writing the final concatenated parquets. Controlled by `download.osm.*` in config.yaml.

   **Gotcha — extract order matters for performance, not correctness.** The dedup keeps a `seen_type_ids` set that grows monotonically across extracts: the first extract is stream-copied as-is, each subsequent extract is filtered against the union of all prior extracts. Put the largest extract first (`us` is already first in `scripts/osm_data/download_history.py`). Reordering to put a small territory first forces the huge `us` extract to be filtered against a tiny seen-set — same correctness, much more wasted work.

   **Gotcha — `_concat_history` memory at nationwide scale.** The `seen_type_ids` set holds one `(str, int)` tuple per element version across all extracts. Python sets of small tuples are ~150 bytes each → at ~50M US versions, the set alone can reach **~7 GB RSS**. This isn't a regression from the pre-territory pairwise (US, PR) version (pass-1 already grew the set during US stream-copy), but it's worth watching peak memory during the final concat step. If the box OOMs, the fix is to spill the seen set to a disk-backed `(type, id)` store (e.g. DuckDB or RocksDB) rather than the in-memory `set`.

   **Gotcha — per-extract 404 tolerance**: if Geofabrik ever stops publishing a territory's `*-internal.osh.pbf`, the loader logs a warning, skips that extract, and continues. The territory's POIs still flow through downstream stages but the rater falls back to the global-mean δ for its `shared_label`s. Pre-flight is cheap — see [docs/data-sources.md](../../docs/data-sources.md#osm-history-geofabrik-full-history-pbfs) for the cookie/URL check.

2. **Format tabular observations** → `osm_observations.parquet`
   ```bash
   python scripts/osm_data/format_tabular.py
   ```
   Uses `osm_data.tag_key` (e.g., `name`) to flag change/deletion per POI version, then assigns shared taxonomy labels from the conflation crosswalk and explodes rows per label. One row = (POI version, shared_label). Rows with no matching taxonomy category are dropped.

3. **Pick a modeling config and fit λ** — see [skills/iterate-model-types](../iterate-model-types/SKILL.md) for choosing `model_type` / `group_key`.
   ```bash
   python scripts/models/osm_turnover.py
   ```
   Writes `fitted_params.csv`, `param_draws.csv`, `predictions.csv` to `{date}_by_shared_label` (the unified random-effects model) or `{date}_constant` (single-rate baseline) under `directories.model_output.path`.

4. **Apply predictions to the OSM snapshot** → `osm_snapshot_rated.parquet`
   ```bash
   python scripts/osm_snapshot/apply_model.py
   ```
   Reads the `osm_data.apply_model.model_stub` date, loads the `{stub}_by_shared_label` random-effects model (if present), falls back to `{stub}_constant` for rows with no matching taxonomy label, and rates every POI in `osm_snapshot.parquet`.

## Verification

Hand off to [skills/verify-pipeline-run](../verify-pipeline-run/SKILL.md) — in particular:
- Row counts on `osm_versions.parquet` vs previous run (flag >5% drops).
- `fitted_params.csv`: confirm all expected group values present, λ ranges sensible.
- `predictions.csv`: head/tail spot-check.
- `osm_snapshot_rated.parquet`: confirm `conf_mean`/`conf_lower`/`conf_upper` populated for all rows.

## Key code

- Entry: [src/openpois/io/osm_history_pbf.py](../../../src/openpois/io/osm_history_pbf.py) (`download_osm_history`)
- Entry: [src/openpois/osm/format_observations.py](../../../src/openpois/osm/format_observations.py)
- Entry: [src/openpois/models/](../../../src/openpois/models/) — `ModelFitter` (JAX/BlackJAX), model classes
- Registry: [src/openpois/models/osm_models.py](../../../src/openpois/models/osm_models.py) — `MODEL_REGISTRY`, `get_model_class`
