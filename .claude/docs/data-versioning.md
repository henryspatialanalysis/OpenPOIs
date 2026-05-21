# Data versioning

Every pipeline output is versioned via a single `versions:` block in [config.yaml](../../config.yaml). The external `config_versioned` package resolves these into filesystem paths.

## Source of truth

```yaml
versions:
  osm_data: "20260416"                 # historical PBF pipeline outputs
  model_output: "20260416_by_leisure"  # fitted model artifacts (suffix indicates variant)
  snapshot_osm: "20260416"             # OSM current-state snapshot
  snapshot_overture: "20260417"        # Overture snapshot
  conflation: "20260417"               # conflated output
  source_coop: "2026-04-17-v0"         # Source Cooperative upload folder (see below)
```

Each key corresponds to a `directories.<key>` entry in `config.yaml` with `versioned: true`, except `source_coop`, which only names the remote folder.

## Path resolution

External `config_versioned.Config` API:

```python
config.get_dir_path("osm_data")
# → ~/data/openpois/osm_data/20260416/

config.get_file_path("osm_data", "osm_versions")
# → ~/data/openpois/osm_data/20260416/osm_versions.parquet
```

**Prefer `get_file_path` over composing `get_dir_path()` + `get()` manually.**

`.get()` raises `ValueError` on null values — pass `fail_if_none=False` for optional fields like `download.overture.release_date: null`.

`config.write_self(section)` snapshots the effective config into the output directory — used by model and conflation scripts to record the state of a run.

## Naming conventions

- **Local dates**: `YYYYMMDD`, e.g., `20260416`.
- **Model variants**: `{date}_by_{group_key}` (e.g., `20260416_by_leisure`, `20260416_by_amenity`) or `{date}_constant`. See [skills/iterate-model-types](../skills/iterate-model-types/SKILL.md).
- **Source Coop folder**: `YYYY-MM-DD-v<IDX>`. Default `v0` for every fresh publish; only bump `v1`, `v2`, … if republishing under the same calendar date (e.g. a hot-fix). The Source Coop upload script writes the per-version README into this folder, so the suffix must be unique per upload round.
- **Independent cadences**: snapshot versions can (and should) differ across sources — Overture releases ~monthly. Don't force them to match.

## External references (hand-update when bumping)

Version strings appear in these places outside `versions:` — grep before any cross-source version change:

| File | References |
|---|---|
| [site/src/constants.js](../../site/src/constants.js) | `OSM_PMTILES_URL`, `CONFLATED_PMTILES_URL` (full `data.source.coop` URLs) |
| [site/public/about.html](../../site/public/about.html) | Hardcoded Source Coop browse links in the data-access section |
| `osm_data.apply_model.model_stub` (config.yaml) | Which model family [scripts/osm_snapshot/apply_model.py](../../scripts/osm_snapshot/apply_model.py) ingests |

[skills/update-site](../skills/update-site/SKILL.md) covers the frontend side; [skills/conflate-snapshots](../skills/conflate-snapshots/SKILL.md) covers the publish + config side.

## Geographic-scope changes (hand-update on bump)

When the dataset's geographic scope changes (e.g. the 2026-05-21 territory expansion), the *next* Source Cooperative publish is the first to expose the change. The publish step doesn't read the boundary file, so the per-version README's geographic-scope language must be hand-updated alongside the version bump:

| File | Update |
|---|---|
| `publish.version_metadata` in [config.yaml](../../config.yaml) | Any human-readable scope strings surfaced in the per-version README |
| Per-version Source Coop README template (in the publisher) | Geographic scope line — e.g. "50 states + DC + 5 inhabited US territories" |
| [README.md](../../README.md), [docs/data-sources.md](data-sources.md) | Already updated as part of the scope change PR; double-check before publish |

The 2026-05-21 expansion was the move from "50 states + DC + PR" to "50 states + DC + 5 inhabited US territories" (added VI, GU, MP, AS).

## Workflow

1. Bump the relevant `versions.*` keys before running a pipeline. For a public release, also bump `versions.source_coop` to the new `YYYY-MM-DD-v0`.
2. Run the pipeline — outputs land in the versioned directory.
3. After publishing, update the frontend references in `site/src/constants.js` and `site/public/about.html`.
4. Old local versions stay on disk — delete manually when confident nothing references them. Old Source Coop folders stay published indefinitely and serve as an immutable archive.
