# OpenPOIs

A unified, open dataset for points of interest across the United States. Built from [OpenStreetMap](https://www.openstreetmap.org) and
[Overture Maps](https://overturemaps.org), with per-POI confidence scores
calibrated against a verified validation sample so they read as probabilities
that a place exists and is currently open.

- 🌐 **Interactive map:** <https://openpois.org/>
- 💻 **Source code:** <https://github.com/henryspatialanalysis/openpois>
- 📘 **Data license:** [Open Database License v.1.0](./LICENSE). For more details, see the [Open Data Commons](https://opendatacommons.org/licenses/odbl/1-0/).

## Repository layout

Each refresh writes a new versioned folder. Inside every version:

```
<YYYY-MM-DD-vN>/
├── README.md                     # version metadata (OSM date, Overture release, model)
├── osm-parquet/                  # OSM-only snapshot, hive-partitioned by primary_tag
├── osm-pmtiles/osm.pmtiles       # OSM snapshot as a single PMTiles archive
├── conflated-parquet/            # OSM × Overture conflated snapshot, hive-partitioned by shared_label
└── conflated-pmtiles/conflated.pmtiles
```

`latest/` is a server-side mirror of the most recently published version —
use it for live demos and tutorials, and pin a dated folder
(e.g. `2026-04-23-v0/`) when you need a stable, reproducible reference.

Browse all versions at
<https://source.coop/henryspatialanalysis/openpois>.

## What's in the conflated dataset

One row per real-world POI after matching OpenStreetMap features against
Overture Maps places. Key columns:

| Column | Description |
|---|---|
| `unified_id` | Stable ID for the conflated POI |
| `source` | `matched` (in both sources), `osm`, or `overture` |
| `osm_id`, `osm_type` | Source OSM feature (when present) |
| `overture_id` | Source Overture ID (when present) |
| `name`, `brand` | Preferred display names |
| `shared_label` | Harmonised category across the two source taxonomies |
| `conf_mean` | **Calibrated probability that the POI exists and is currently open to the public** (see Confidence, below) |
| `conf_lower`, `conf_upper` | 95% interval for the calibrated confidence |
| `conf_mean_uncalibrated` | The pre-calibration model value, retained for comparison |
| `calibration_flag` | Null for the ordinary case; otherwise why this row was handled specially (see Confidence) |
| `match_score`, `match_distance_m` | Diagnostics for the OSM × Overture link |
| `osm_conf_mean`, `overture_confidence` | The two raw per-source inputs to the calibration |
| `bbox` | GeoParquet 1.1 covering struct, for spatial row-group pruning |
| `geohash` | Within-partition sort key (precision 6) |
| `geometry` | WKB point (EPSG:4326) |

The `osm-parquet/` files contain the same OSM rows before conflation. This data retains the original OSM tags.

## Confidence

`conf_mean` on the conflated dataset is a **calibrated probability that the place
exists and is open to the public**, not a raw model score. It is produced by
mapping each POI's source score(s) through a curve estimated from an independent
validation sample of POIs whose real-world status was established by research
and human review. Curves are fitted separately for each detection pattern — in
both sources, OpenStreetMap only, Overture only — because those populations
behave very differently.

Three consequences worth knowing before you filter on it:

- **It is not comparable to previous releases before 2026-07-30.** Earlier
  versions published an uncalibrated blend, and Overture-only rows carried a
  flat downweight. `conf_mean_uncalibrated` holds the old-style value if you
  need to reproduce prior behaviour.
- **The achievable range is narrower than 0–1**, because the calibration
  reports the existence rate actually observed in each score band rather than
  the source's nominal score. Overture-only rows, for example, span roughly
  0.54–0.84.
- **`conf_mean` means something different in `osm-parquet/`.** There it is the
  OpenStreetMap turnover-model posterior — the probability the feature is still
  current given its edit history — and it is *not* calibrated against verified
  ground truth. Do not compare the two columns numerically across the two
  datasets.

`calibration_flag` records the exceptions:

| Flag | Meaning |
|---|---|
| *(null)* | Ordinary case: confidence read from the segment's calibration curve |
| `shadow_cd` | Overture row demoted by OpenStreetMap-history change detection; keeps that value, no interval |
| `missing_conf` | Overture supplied no confidence upstream; a placeholder was imputed before calibration |
| `unnamed_extrapolated` | Unnamed feature, outside the validation sample; calibrated by extrapolation |

## Quickstart

Read directly from Source Cooperative's S3 mirror (no authentication):

- **pyarrow / GeoPandas** use `pyarrow.fs.S3FileSystem(anonymous=True)` and a
  bucket-qualified path (no scheme prefix).
- **DuckDB** uses an `s3://` URL plus an anonymous `SECRET` so its glob
  expansion works over the bucket listing.

Every example uses `VERSION = "latest"`; swap in a dated folder (e.g.
`"2026-04-23-v0"`) when you need a reproducible pin.

### Python: pyarrow

```python
import pyarrow.dataset as ds
import pyarrow.fs as pafs

BASE = "us-west-2.opendata.source.coop/henryspatialanalysis/openpois"
VERSION = "latest"   # or pin a specific dated folder, e.g. "2026-04-23-v0"

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

### Python: DuckDB

```python
import duckdb

BASE = "s3://us-west-2.opendata.source.coop/henryspatialanalysis/openpois"
VERSION = "latest"   # or pin a specific dated folder, e.g. "2026-04-23-v0"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("""
    CREATE OR REPLACE SECRET srccoop (
        TYPE s3, PROVIDER config,
        REGION 'us-west-2', URL_STYLE 'path',
        KEY_ID '', SECRET ''
    );
""")
df = con.execute(f"""
    SELECT shared_label, COUNT(*) AS n
    FROM read_parquet('{BASE}/{VERSION}/conflated-parquet/**/*.parquet',
                      hive_partitioning = true)
    GROUP BY shared_label
    ORDER BY n DESC
    LIMIT 20
""").df()
print(df)
```

### Python: GeoPandas

```python
import geopandas as gpd
import pyarrow.fs as pafs

BASE = "us-west-2.opendata.source.coop/henryspatialanalysis/openpois"
VERSION = "latest"   # or pin a specific dated folder, e.g. "2026-04-23-v0"

fs = pafs.S3FileSystem(anonymous = True, region = "us-west-2")
# conflated-parquet is hive-partitioned by shared_label.
gdf = gpd.read_parquet(
    f"{BASE}/{VERSION}/conflated-parquet/shared_label=Cafe/part-0.parquet",
    filesystem = fs,
)
print(gdf.head())
```

### Browser / vector-tile map

The `*-pmtiles/*.pmtiles` archives can be loaded directly by any PMTiles
client (MapLibre + `pmtiles://`, OpenLayers + `ol-pmtiles`, etc.). See
`site/` in the GitHub repo for a working example.

The snippet below is a self-contained HTML page that renders the conflated
PMTiles over MapLibre, coloured by the model's `conf_mean`. Save it as
`openpois.html` and open it in a browser — no build step, no server needed.
PMTiles are authored at zoom 14, so zoom in past z14 to see points.

```html
<!doctype html>
<meta charset="utf-8" />
<title>OpenPOIs — conflated</title>
<link href="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css" rel="stylesheet" />
<style>html, body, #map { height: 100%; margin: 0; }</style>
<div id="map"></div>
<script src="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/pmtiles@3/dist/pmtiles.js"></script>
<script>
  const BASE = "https://data.source.coop/henryspatialanalysis/openpois";
  const VERSION = "latest";   // or pin a specific dated folder, e.g. "2026-04-23-v0"

  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);

  const map = new maplibregl.Map({
    container: "map",
    style: "https://tiles.openfreemap.org/styles/positron",
    center: [-73.9855, 40.758],   // Times Square
    zoom: 16,
  });

  map.on("load", () => {
    map.addSource("openpois", {
      type: "vector",
      url: `pmtiles://${BASE}/${VERSION}/conflated-pmtiles/conflated.pmtiles`,
      minzoom: 14,
    });
    map.addLayer({
      id: "openpois-points",
      type: "circle",
      source: "openpois",
      "source-layer": "conflated_pois",   // set by publish.pmtiles.conflated_layer_name
      paint: {
        "circle-radius": 4,
        "circle-stroke-width": 1,
        "circle-stroke-color": "#ffffff",
        // Red when stale (conf_mean ≈ 0), green when fresh (≈ 1).
        "circle-color": [
          "interpolate", ["linear"], ["get", "conf_mean"],
          0.0, "#d73027",
          0.3, "#fee08b",
          0.7, "#1a9850",
        ],
      },
    });
    map.on("click", "openpois-points", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(
          `<b>${p.name ?? "(no name)"}</b><br>` +
          `${p.shared_label} · source=${p.source}<br>` +
          `conf_mean = ${Number(p.conf_mean).toFixed(3)}`
        )
        .addTo(map);
    });
  });
</script>
```


## License & attribution

The OpenPOIs dataset is released under the [Open Database License (ODbL) v.1.0](./LICENSE). Any public use must credit OpenStreetMap contributors, the Overture Maps Foundation, and OpenPOIs. Any derivative database must be shared under the same license. See <https://www.openstreetmap.org/copyright> and <https://docs.overturemaps.org/attribution/> for upstream attribution
requirements.

## Citation

If you use this data in research, please cite:

> Henry Spatial Analysis (2026). *OpenPOIs: a unified, confidence-scored
> dataset of U.S. points of interest.* <https://openpois.henryspatialanalysis.com>

## Contact

Questions, bug reports, and contributions welcome via <https://github.com/henryspatialanalysis/openpois/issues>.
