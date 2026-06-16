# Survey-vs-armchair source/provenance tags in OSM POIs

Status: **investigated / parked.** A conference attendee (OSM editor, 2026-06)
noted that mappers don't always observe a business in person — some edits are
traced from aerial/satellite imagery ("armchair" mapping) — and that OSM has tags
marking this, suggesting we might downweight or drop online-only observations in
the turnover model. This note records the descriptive check we ran to decide
whether the signal is worth pursuing. **Conclusion: the relevant element tags
exist but are too sparse (~0.5% of POIs carry an in-person marker), and the
gold-standard armchair signal is changeset-level and not retained by our
pipeline.** No model change was built. The analysis is read-only and additive.

## 1. Which OSM tags distinguish in-person from online?

From the OSM wiki, and crucially *where* each is applied:

| Tag | Applied to | Signal | Reachable in our data? |
|---|---|---|---|
| `survey:date` | **Element** | Strong — "last surveyed or verified **in person**" | Yes (history change log) |
| `source` | **Element** (legacy) + **changeset** (modern, preferred) | Value-dependent: ground (`survey`, `GPS`, `local_knowledge`, `mapillary`) vs online (`Bing`, `aerial`, `Esri`, `Maxar`, `landsat`) vs import (`TIGER`, `GNIS`, agency GIS, URLs) | Element values: yes. Changeset: **no** |
| `check_date` | **Element** | Weak — "last checked"; could be remote or ground (StreetComplete writes it) | Yes |
| `imagery_used` | **Changeset only** | **Strongest** armchair marker; auto-added by iD/JOSM/Vespucci | **No** |

The decisive distinction is element vs changeset. The cleanest "this was traced
from imagery" markers (`imagery_used`, modern changeset `source`) live on
**changesets**. Our full-history PBF pipeline retains changeset *IDs* but not
changeset *tags* (pyosmium does not expose them), so those signals are simply
absent — only 16 `imagery_used` rows have ever leaked onto elements nationwide.

## 2. Where the tags survive in our data

- **`osm_snapshot.parquet` destroys `source`.** `source` *is* in
  `download.osm.extract_keys`, but `src/openpois/io/_osm_poi_handler.py`
  overwrites the column with the constant data-origin label `'osm'`
  (`rec.update({"source": self._source_label, ...})` — a key collision). Every
  snapshot row reads `source='osm'`; the raw OSM value is lost. `survey:date` is
  not extracted at all. Only `check_date` / `check_date:opening_hours` survive
  as usable snapshot columns.
- **`osm_changes.parquet` keeps everything.** It records every tag
  Added/Changed/Deleted across full element history, so raw `source`,
  `survey:date`, and `check_date` *values* are recoverable. The current value of
  a tag is the value at its **highest version that is not a Deletion**.

So the measurement reconstructs current per-element values from `osm_changes`,
then anchors on the live snapshot (`osm_id`/`osm_type` == `id`/`type`) as the POI
universe.

## 3. Method

Script: `scripts/osm_data/source_tag_prevalence.py` (read-only DuckDB, modeled on
`scripts/osm_data/pool_source_breakdown.py`). Steps:

- **A. Discover** all `source%` / `survey%` / `check_date%` keys in `osm_changes`
  (don't assume the list) → confirms `imagery_used` never appears on elements.
- **B. Reconstruct** current `source` / `survey:date` / `check_date` per element
  (window: highest non-Deleted version), join to the snapshot, compute prevalence.
- **C. Bucket** `source` values case-insensitively by substring into
  **ground / online / import_other / uncategorized**, ground-first so a mixed
  value crediting an in-person source counts as ground. `image` is excluded from
  ground terms because it collides with the online "imagery" terms.
- **D. Observation-level** prevalence by joining the per-element flags to
  `osm_observations.parquet` (the model-relevant denominator).
- **E.** Writes a markdown summary + 5 CSVs.

Run it (data version 20260521; ~minutes):

```bash
python -u scripts/osm_data/source_tag_prevalence.py \
  2>&1 | tee ~/data/openpois/logs/source_tag_prevalence.log
```

Outputs land in `~/data/openpois/logs/`: `source_tag_prevalence_SUMMARY.md`,
`source_tag_keys_discovered.csv`, `source_tag_prevalence_live.csv`,
`source_value_breakdown.csv`, `source_bucket_summary.csv`,
`source_tag_prevalence_observations.csv`.

## 4. Findings (version 20260521)

8,799,633 live POIs; 9,963,378 observations; **94.5%** of snapshot POIs appear in
the full-history element set, so the figures are near-complete (the ~5.5% absent
can only deflate them slightly).

**Prevalence among live POIs:**

| signal | % of live POIs |
|---|---:|
| `source` (any value) | 4.49% |
| &nbsp;&nbsp;→ online (Bing / aerial / Esri) | 2.13% |
| &nbsp;&nbsp;→ import_other (TIGER / GNIS / agency GIS / URLs) | 1.64% |
| &nbsp;&nbsp;→ **ground (survey / GPS / local knowledge)** | **0.45%** |
| &nbsp;&nbsp;→ uncategorized (mostly remote GIS imports) | 0.27% |
| `survey:date` | 0.07% |
| `check_date` | 2.59% |
| **any in-person signal (`survey:date` OR `source`==ground)** | **0.52%** |

Observation-level numbers are similar (any in-person = 0.95%). The uncategorized
`source` tail is dominated by agency GIS imports (NPS, NHD, 3DEP, state DNR), i.e.
also not in-person — which reinforces, rather than threatens, the verdict.

## 5. Verdict and why we stopped

The tags **do** distinguish in-person from online, but a *positive in-person
marker* sits on only ~0.5% of POIs — far too sparse to drive observation
downweighting on its own. The only sizeable "remote" slice is online-imagery +
imports (~3.8%), but that conflates armchair tracing with bulk GIS imports and
is a poor proxy for "the mapper didn't see it." The signal one would actually
want — changeset `imagery_used` — is unreachable from the full-history PBFs and
would require a **separate changeset-tag pull** (OSM API or changeset dumps).

A follow-up was deliberately not designed. If revisited, the two levers are:
(1) a changeset-tag pull to recover `imagery_used`/changeset `source`; (2) fixing
the snapshot `source` collision in `_osm_poi_handler.py` so element-level `source`
is at least preserved going forward. See also
[turnover-model-methodology.md](turnover-model-methodology.md) for how
observations feed λ.
