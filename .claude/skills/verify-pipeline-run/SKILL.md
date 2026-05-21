---
name: verify-pipeline-run
description: Use when the user wants a QA/sanity check on a recently completed pipeline run — row counts, parameter spot-checks, diffs vs. prior versions, or site-side verification. Triggers: "sanity check the run", "verify the new data looks right", "QA the model output", "diff against the last version", "did the upload work".
---

# Verify a pipeline run

Post-run QA runbook. Pick the subsection that matches what just ran.

## Snapshots (OSM / Overture)

Baseline row counts (2026-04-17):
- OSM: ~7.78M
- Overture: ~13.05M (up from ~7.23M after widening `taxonomy_allowlist`; pre-2026-04-17 runs will be lower)

Check:
```python
import pandas as pd
pd.read_parquet(path).shape[0]
```

Flag >5% drops. Known regression patterns:
- **OSM**: PR, USVI, and `american-oceania` are *separate* PBFs from the main `us-latest.osm.pbf` — confirm all 4 got downloaded, filtered, and concat'd. The orchestrator logs `"Processing <name> extract..."` for each; absence of a line in the log = silent skip.
- **Overture**: coarse-bbox pushdown + final DuckDB `ST_Within` — drop means the antimeridian split was lost (Guam/NMI/Aleutians) or the Census boundary failed to load. If the run crashed with "Information loss on integer cast", the DuckDB pin was bumped off 1.4.1 (see [docs/data-sources.md](../../docs/data-sources.md) → Overture Maps).

### Per-territory spot checks

The 2026-05-21 expansion adds GU/VI/MP/AS to the existing 50+DC+PR footprint. Fermi expectations for the *first* territory-inclusive run (no historical baseline yet):
- Combined territory POI count: ~20–50K (well under 1% of the existing total). <1K combined or >100K combined signals boundary clip or american-oceania filter regression.
- Per territory: ~5–15K each. Skew (one territory dominating) is suspicious.

Quick check on the OSM snapshot:
```python
import pandas as pd
osm = pd.read_parquet("~/data/openpois/snapshots/osm/{version}/osm_snapshot.parquet")
print(osm["addr:state"].value_counts().head(20))
# Expect a mix of "GU"/"Guam", "VI"/"USVI", "MP"/"Northern Mariana Islands",
# "AS"/"American Samoa" (OSM doesn't enforce a single value). If any
# territory is missing entirely, the corresponding PBF didn't merge.
print("Longitude extent:", osm.geometry.x.min(), "→", osm.geometry.x.max())
# Should now span ~-171 (American Samoa) to ~+146 (NMI), not just CONUS.
```

Then for the conflated output, also check `shared_label` distribution per territory — if one territory is >50% dominated by a single `shared_label`, suspect either a conflation bug or you accidentally pulled independent Samoa (the country, separate `samoa-latest.osm.pbf` extract — *not* American Samoa).

## Model output

```
~/data/openpois/osm_turnover_model/{version}/
  fitted_params.csv     # λ and σ per group
  param_draws.csv       # uncertainty bounds
  predictions.csv       # predictions per POI
```

Checks:
- Row count in `fitted_params.csv` ≈ number of groups (after `min_value_count` filter).
- λ values in a sensible range (spot-check against prior `fitted_params.csv`).
- `predictions.csv` head/tail — every POI should have a prediction; no NaNs.
- **Post-territory-expansion runs**: territories are <1% of POIs so global `logit_lambda` / `logit_delta_0` shouldn't move much. A >5% shift on the first territory-inclusive run vs the prior `fitted_params.csv` is worth investigating — could be a real mapping-behavior difference (territory POIs have fewer edits per year because smaller mapper communities) or a bug in `format_observations`.

## Rated snapshot

```
~/data/openpois/snapshots/osm/{version}/osm_snapshot_rated.parquet
```

Confirm `conf_mean`, `conf_lower`, `conf_upper` columns are populated for every row. NaNs indicate groups not covered by any `{stub}_by_*` or `{stub}_constant` fallback.

## Conflation

```
~/data/openpois/conflation/{version}/
  conflated.parquet
  match_diagnostics.parquet
  summary_by_label.csv
```

- Match rate per label in `summary_by_label.csv` should resemble prior run. Large drifts → parameter regression or crosswalk edit.
- `match_diagnostics.parquet` for per-pair forensics when specific matches look wrong.
- **Territory match rates**: expect **lower** OSM × Overture match rates in territories than in CONUS — smaller mapper communities on both sides mean more source-only POIs. A territory match rate that looks CONUS-like is suspicious (possibly accepting cross-territory candidates with overly loose thresholds, or the conflation polygon clip missed a region).

## Site

- Open the deployed site (or `npm run dev` locally after a constants.js bump).
- Browser console: no CORS, no 404s on `data.source.coop` URLs.
- Filter dropdown: each source (OSM / Overture / Conflated) loads.
- Popups non-empty; taxonomy legend rendered; PMTiles overlay visible at zoom 14+.
- **Post-territory-expansion runs**: pan to all 4 new territories — Guam (~+144°E) and American Samoa (~-170°W) are the longitudes most likely to expose tile-wrap or PMTiles edge bugs that haven't been exercised before. Verify points render at all 4 territories. Geocoder check: `"Hagåtña"`, `"Charlotte Amalie"`, `"Saipan"`, `"Pago Pago"` should resolve in the search bar (Stadia `boundary.country` now includes `PR,VI,GU,MP,AS`).

## Recording issues

Anything anomalous goes into [.claude/TODO.md](../../TODO.md) under **In progress** so follow-ups don't drop.
