# CLAUDE.md

Guidance for Claude Code working in this repository. Deep-dives live in [skills/](skills/) and [docs/](docs/); this file is orientation.

## Environment setup

```bash
make build_env       # Create conda env from environment.yml (name: openpois, Python 3.10+)
make install_package # pip install -e . (editable install)
```

Python executable: `$CONDA_PREFIX/bin/python` after `conda activate openpois`.

> **Note on data paths.** Examples in this directory show resolved paths under `~/data/openpois/...`, which is the maintainer's configured `directories.*.path` value in `config.yaml`. Substitute your own `directories.*.path` root when reading them; the layout under that root is set by `config.yaml`.

## Common commands

```bash
pytest               # Run tests
make export_env      # Export conda env to environment.yml after adding deps
```

Style: Black (format-on-save in VSCode). Lint: flake8 + pylint, configured in `pyproject.toml`.

## Architecture at a glance

**openpois** models POI stability over time from OpenStreetMap history, and produces unified OSM + Overture snapshots for web consumption. Work splits into four pipelines:

| Pipeline | Skill |
|---|---|
| Fit λ from OSM history, rate current snapshots | [skills/model-history-pipeline](skills/model-history-pipeline/SKILL.md) |
| Iterate model variants on a pinned history run | [skills/iterate-model-types](skills/iterate-model-types/SKILL.md) |
| Refresh the POI snapshots (OSM / Overture) | [skills/full-data-pull](skills/full-data-pull/SKILL.md) |
| Conflate OSM + Overture, partition, publish to Source Cooperative | [skills/conflate-snapshots](skills/conflate-snapshots/SKILL.md) |
| Bump the frontend to the new data version | [skills/update-site](skills/update-site/SKILL.md) |
| Post-run QA on any of the above | [skills/verify-pipeline-run](skills/verify-pipeline-run/SKILL.md) |

## Where things live

| Path | Purpose |
|---|---|
| [src/openpois/io/](../src/openpois/io/) | I/O adapters: OSM history/snapshot, Overture, Census boundary |
| [src/openpois/osm/](../src/openpois/osm/) | OSM-specific transforms: `format_observations`, `change_plots` |
| [src/openpois/models/](../src/openpois/models/) | JAX/BlackJAX empirical Bayes: `ModelFitter`, model registry |
| [src/openpois/conflation/](../src/openpois/conflation/) | OSM×Overture matching: `taxonomy`, `match`, `merge` |
| [scripts/](../scripts/) | End-to-end pipelines using config.yaml — not installed, reference only |
| [site/](../site/) | Vue 3 + Vite frontend |

## Reference docs

- [docs/data-sources.md](docs/data-sources.md) — URLs, auth, schema quirks for every source
- [docs/taxonomy-setup.md](docs/taxonomy-setup.md) — crosswalk CSVs, build_taxonomy.py, frontend sync
- [docs/data-versioning.md](docs/data-versioning.md) — `versions:` block, path resolution, external references
- [docs/package-versioning.md](docs/package-versioning.md) — semver bumps for the Python package + Vue site + Sphinx docs (distinct from data versioning)
- [docs/partitioning-strategy.md](docs/partitioning-strategy.md) — Hive layout of the partitioned Parquet (`shared_label` for conflated, `primary_tag` for OSM), query patterns, when each layout applies
- [docs/turnover-model-methodology.md](docs/turnover-model-methodology.md) — statistical derivation of the POI turnover model with ZIE extension
- [docs/change-detection.md](docs/change-detection.md) — OSM-history-derived ghost POIs, shadow matching, and the per-`shared_label` δ penalty applied to Overture. Canonical entry point is `make conflate`, which runs the three-step `build_ghosts` → `conflate.py --output-suffix=baseline` → `apply_change_detection.py` pipeline so all national runs include the CD penalty by default.
- [docs/time-varying-models.md](docs/time-varying-models.md) — experimental/parked `constant_breakpoint` (two-rate, tag-age) turnover model: rationale, key files, how to run, the inconclusive 2026-06-08 nationwide result (t_B collapses to ~days, non-identified), and why the random-effects extension was not built. Production `constant`/`random_effects` models are unchanged.

## Running to-do

[TODO.md](TODO.md) — curated running list. Not auto-synced to git status.

## Config gotcha worth surfacing

`config_versioned.Config.get()` raises `ValueError` on null values. For optional fields (e.g., `release_date: null`), pass `fail_if_none=False`. Prefer `config.get_file_path(section, file_key)` over composing `get_dir_path()` + `get()` manually.

## Running long workflows

When kicking off a long-running pipeline (downloads, rating, conflation, upload), stream stdout to a file you can tail at any time — don't rely on piping through `head`/`tail` or on the background-task's captured-output file, since Python output may stay buffered for long stretches. Prefer:

```bash
python -u scripts/... 2>&1 | tee ~/data/openpois/logs/<step>_<version>.log
```

`python -u` disables output buffering; `tee` lets the shell and a tail watcher both see output in real time.
