# OpenPOIs

A unified, confidence-scored open dataset of U.S. points of interest —
covering the 50 states, DC, and the 5 inhabited U.S. territories (Puerto
Rico, US Virgin Islands, Guam, Northern Mariana Islands, American Samoa) —
built from [OpenStreetMap](https://www.openstreetmap.org) and
[Overture Maps](https://overturemaps.org).

![OpenPOIs interactive map](docs/_static/hero.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: ODbL](https://img.shields.io/badge/Data-ODbL%20v1.0-orange.svg)](https://opendatacommons.org/licenses/odbl/1-0/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](pyproject.toml)
[![Site deploy](https://github.com/henryspatialanalysis/openpois/actions/workflows/deploy-site.yml/badge.svg)](https://github.com/henryspatialanalysis/openpois/actions/workflows/deploy-site.yml)

- 🌐 **Live map:** <https://openpois.org>
- 📘 **Python API docs:** <https://openpois.org/docs/>
- 🗄️ **Dataset on Source Cooperative:** <https://source.coop/henryspatialanalysis/openpois>


## What is OpenPOIs?

OpenPOIs conflates points of interest from OpenStreetMap and Overture Maps
into a single unified dataset, then attaches a per-POI confidence score
estimating the probability that the place still exists. Confidence comes from
a Bayesian turnover model fit on OSM tag-edit history. The published dataset
covers the United States and its 5 inhabited territories, and is refreshed
monthly, following the Overture Maps monthly release cycle.

This repository contains the Python library used to produce the data, the
end-to-end pipelines that download and conflate sources, and the Vue
front-end that powers the live map.


## Quickstart — read the data

No install required. The dataset is hosted anonymously on Source Cooperative;
read it straight from S3:

```python
import pyarrow.dataset as ds
import pyarrow.fs as pafs

BASE = "us-west-2.opendata.source.coop/henryspatialanalysis/openpois"
VERSION = "latest"   # or pin a dated folder, e.g. "2026-04-23-v0"

fs = pafs.S3FileSystem(anonymous = True, region = "us-west-2")
pois = ds.dataset(
    f"{BASE}/{VERSION}/conflated-parquet/",
    filesystem = fs,
    format = "parquet",
    partitioning = "hive",
)
print(pois.schema)
print(f"{pois.count_rows():,} POIs")
```

GeoPandas, DuckDB, and PMTiles examples live in the
[dataset README on Source Cooperative](https://source.coop/henryspatialanalysis/openpois).


## Python package

The full OpenPOIs package API — I/O adapters, the turnover model, conflation
primitives — is documented at <https://openpois.org/docs/>.

### Installation

This package can be installed from source:

```bash
git clone https://github.com/henryspatialanalysis/openpois.git
cd openpois
make build_env          # conda env from environment.yml
conda activate openpois
make install_package    # pip install -e .
```

### Repository layout

| Path | Purpose |
|---|---|
| [src/openpois/](src/openpois/) | Library source: I/O, models, conflation, publishing |
| [scripts/](scripts/) | End-to-end pipelines using `config.yaml` |
| [site/](site/) | Vue 3 + Vite frontend powering openpois.org |
| [docs/](docs/) | Sphinx documentation source |
| [tests/](tests/) | Unit tests |

### Reproduce the dataset yourself

The data is produced by four pipelines under [scripts/](scripts/), each
driven by [config.yaml](config.yaml):

1. Snapshot downloads (OSM + Overture)
2. OSM history download and Bayesian turnover-model fit
3. Apply model to OSM snapshot to get per-POI confidence
4. Conflate OSM × Overture, partition, publish to Source Cooperative

Each pipeline and its scripts are documented in the workflows reference at
<https://openpois.org/docs/workflows.html>.

### Web map

The interactive map at <https://openpois.org> is a Vue 3 + Vite app rendering
PMTiles archives over MapLibre GL. To run it locally:

```bash
make site_dev      # http://localhost:5173, hot reload
make site_build    # production build to site/dist/
```

The site auto-deploys to GitHub Pages via
[.github/workflows/deploy-site.yml](.github/workflows/deploy-site.yml) on
every push to `main` that touches `site/`, `src/`, `docs/`, or `scripts/`.

### Development

```bash
pytest               # run the test suite
make lint            # flake8 + pylint
make export_env      # rewrite environment.yml after adding deps
```

## Licensing

OpenPOIs is dual-licensed:

- **Code** — [MIT License](LICENSE). You can use, modify, and redistribute the
  Python package, scripts, and front-end freely.
- **Data** — [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
  The published parquet and PMTiles archives are derivative works of
  OpenStreetMap and Overture Maps and inherit ODbL terms. Any public use must
  attribute OpenPOIs, [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
  and the [Overture Maps Foundation](https://docs.overturemaps.org/attribution/).
  Derivative databases must be released under the same license.

## Citation

If you use OpenPOIs in research, please cite:

> Henry, N. (2026). *OpenPOIs: a unified, confidence-scored dataset of U.S. points of interest.* Henry Spatial Analysis. <https://openpois.org>

A machine-readable citation is provided in [CITATION.cff](CITATION.cff);
GitHub renders it as a "Cite this repository" button on the repo home page.

## Contact

Bug reports, feature requests, and contributions are welcome via
[GitHub issues](https://github.com/henryspatialanalysis/openpois/issues).
