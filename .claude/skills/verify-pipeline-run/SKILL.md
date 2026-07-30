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
- **Was this a full-data fit?** `grep poi_sample_fraction {version}/config.yaml` must show `1.0`. A sampled fit produces a plausible-looking but much weaker model — check this *first*, because every downstream number inherits it. Corroborate in the fit log: `amenity_msa interaction: N cells >= 100 POIs` should read ~4,000, not ~18. `apply_model_random_effects.py` enforces this too, but catch it here rather than 5 h later.
- Row count in `fitted_params.csv` ≈ number of groups (after `min_value_count` filter). A full-data `random_effects` fit is ~9,000 rows; ~1,200 means it was sampled.
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
- **Attribute-coverage check** (added 2026-07 after the merge-phase column bug): non-null share of contact/address columns per `source` class must be far from zero.
  ```python
  import pandas as pd
  df = pd.read_parquet(path, columns = ["source", "phone", "website", "addr_street"])
  print(df.groupby("source").agg(lambda s: s.notna().mean()))
  ```
  Overture-sourced rows should sit near the snapshot's coverage (~90% phone, ~80% website, ~98% street as of 2026-07); OSM-sourced rows are much sparser but still nonzero. **Near-0% on any source class** means merge-phase column plumbing regressed again (see `spill_rows` + `require_all` in `conflate.py`). Caveat: conflated outputs from **before 2026-07-24 carry this bug** (0% on Overture rows) — don't baseline coverage against them.
- `match_diagnostics.parquet` for per-pair forensics when specific matches look wrong.
- **Territory match rates**: expect **lower** OSM × Overture match rates in territories than in CONUS — smaller mapper communities on both sides mean more source-only POIs. A territory match rate that looks CONUS-like is suspicious (possibly accepting cross-territory candidates with overly loose thresholds, or the conflation polygon clip missed a region).

## Confidence calibration

```
~/data/openpois/conflation/{version}/
  conflated_cd.parquet          # pre-calibration (CD applied)
  conflated.parquet             # canonical, calibrated
  calibration/fit_report.md     # read this first
  calibration/{segment}_curve.parquet + _metadata.json
  calibration/biggest_movers.csv, shift_by_label.csv
  viz/calibration_{curves,reliability,shift}.png
```

Read `fit_report.md` first — see the "Reading the fit report" section of
[docs/confidence-calibration.md](../../docs/confidence-calibration.md) for what
each table means. Then check the deployed output:

- **Row count preserved exactly** against `conflated_cd.parquet`. Calibration
  must never add or drop a row.
- **Range and interval invariants** — these should all be zero:
  ```python
  import duckdb
  print(duckdb.sql(f"""
    SELECT COUNT(*) rows,
      SUM(CASE WHEN conf_mean < 0 OR conf_mean > 1 THEN 1 ELSE 0 END) out_of_range,
      SUM(CASE WHEN conf_mean IS NULL OR isnan(conf_mean) THEN 1 ELSE 0 END) null_conf,
      SUM(CASE WHEN conf_lower > conf_mean + 1e-9 THEN 1 ELSE 0 END) lower_gt_mean,
      SUM(CASE WHEN conf_upper < conf_mean - 1e-9 THEN 1 ELSE 0 END) upper_lt_mean
    FROM read_parquet('{path}')"""))
  ```
- **Monotonicity of the deployed map**: within a segment, a higher input score
  must never yield a lower calibrated value. Bin the index and check for
  inversions; there should be none.
- **Flag counts are plausible**: `shadow_cd` should equal the change-detection
  row count exactly, `unnamed_extrapolated` the unnamed-OSM count, and
  `missing_conf` the count of Overture rows at exactly 0.5.
- **Shadow rows untouched**: every `shadow_matched` row must satisfy
  `conf_mean = conf_mean_uncalibrated` with a null interval.
- **Composite vs reference**: the fit report's Horvitz-Thompson reference curve
  should sit inside the composite's band over most of the grid. A systematic gap
  means the working model is wrong — investigate before publishing.
- **Band redistribution is expected and large.** On 20260730 the `>90%` band
  halved and `<30%` nearly emptied. Confirm the shift matches the fit report
  rather than assuming a bug, but do sanity-check `shift_by_label.csv`: the
  biggest movers should be explainable (stable OSM institutions down because the
  OSM curve has a ceiling; Overture-only up because the flat ×0.7 is gone).
- **Curves are release-specific.** If `snapshot_overture` or the turnover model
  moved but `versions.calibration` did not, the curves are stale — re-export the
  handoff from openpois-validator and refit.

## Site

- Open the deployed site (or `npm run dev` locally after a constants.js bump).
- Browser console: no CORS, no 404s on `data.source.coop` URLs.
- Filter dropdown: each source (OSM / Overture / Conflated) loads.
- Popups non-empty; taxonomy legend rendered; PMTiles overlay visible at zoom 14+.
- **Post-territory-expansion runs**: pan to all 4 new territories — Guam (~+144°E) and American Samoa (~-170°W) are the longitudes most likely to expose tile-wrap or PMTiles edge bugs that haven't been exercised before. Verify points render at all 4 territories. Geocoder check: `"Hagåtña"`, `"Charlotte Amalie"`, `"Saipan"`, `"Pago Pago"` should resolve in the search bar (Stadia `boundary.country` now includes `PR,VI,GU,MP,AS`).

## Recording issues

Anything anomalous goes into [.claude/TODO.md](../../TODO.md) under **In progress** so follow-ups don't drop.
