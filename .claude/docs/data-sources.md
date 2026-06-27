# Data sources

Reference for every external data source openpois ingests. For the workflow that orchestrates these, see the skills under [.claude/skills/](../skills/).

## OSM history (Geofabrik full-history PBFs)

**Used by**: the historical modeling pipeline ([skills/model-history-pipeline](../skills/model-history-pipeline/SKILL.md)).

- **URLs** (passed via `HistoryExtract` specs from `scripts/osm_data/download_history.py`):
  - `download.osm.history_pbf_url` → `https://osm-internal.download.geofabrik.de/north-america/us-internal.osh.pbf`
  - `download.osm.pr_history_pbf_url` → `.../north-america/us/puerto-rico-internal.osh.pbf`
  - `download.osm.usvi_history_pbf_url` → `.../north-america/us/us-virgin-islands-internal.osh.pbf`
  - `download.osm.american_oceania_history_pbf_url` → `.../australia-oceania/american-oceania-internal.osh.pbf` (covers Guam, NMI, American Samoa, plus uninhabited US Pacific possessions)
- **Auth**: OAuth — any OSM account works. Produce a Netscape-format cookie jar (browser export or Geofabrik's `oauth_cookie_client.py`). Path: `download.osm.history_cookie_file` (default `~/data/openpois/.creds/geofabrik_cookies.txt`).
- **Cookie expiry symptom**: a stale cookie returns **HTTP 403** on every URL, including known-good ones — *not* 401, *not* a redirect to a login page. Easy to misdiagnose as a server outage. Diagnostic:
  ```bash
  curl -sI -b ~/data/openpois/.creds/geofabrik_cookies.txt \
    https://osm-internal.download.geofabrik.de/north-america/us-internal.osh.pbf | head -1
  # HTTP/1.1 200 OK    → cookie good
  # HTTP/1.1 403       → cookie expired, refresh required
  ```
  Refresh: sign in at <https://osm-internal.download.geofabrik.de/> and export cookies (browser extension or `oauth_cookie_client.py`).
- **Pipeline**: per-extract loop — `osmium tags-filter --omit-referenced` → `osmium time-filter` → pyosmium streams to intermediate `*_versions.parquet` + `*_changes.parquet`. Iterative N-way dedup (`_concat_history`) drops `(type, id)` overlap across extracts before writing the final `osm_versions.parquet` + `osm_changes.parquet`.
- **Per-extract failure tolerance**: if a territory's history PBF returns HTTP 404 (e.g. Geofabrik stops publishing it), the loader logs a warning and continues without that territory's history; the rater then falls back to the global-mean δ for that territory's `shared_label`s.
- **Entry**: [src/openpois/io/osm_history_pbf.py](../../src/openpois/io/osm_history_pbf.py) (`download_osm_history`).
- **Config**: `download.osm.start_date`, `end_date`, `filter_keys`, `extract_keys`.

## OSM snapshot (Geofabrik standard PBFs)

**Used by**: current-state snapshot (`osm_snapshot.parquet`).

- **URLs** (passed via `SnapshotExtract` specs from `scripts/osm_snapshot/download.py`):
  - US: `https://download.geofabrik.de/north-america/us-latest.osm.pbf` (~11 GB, 50 states incl. AK+HI)
  - PR: `https://download.geofabrik.de/north-america/us/puerto-rico-latest.osm.pbf` — **PR is not in the US extract**
  - USVI: `https://download.geofabrik.de/north-america/us/us-virgin-islands-latest.osm.pbf`
  - American Oceania: `https://download.geofabrik.de/australia-oceania/american-oceania-latest.osm.pbf` (covers Guam, NMI, American Samoa, and uninhabited US Pacific possessions; Geofabrik does not publish per-territory PBFs for the inhabited western Pacific territories)
- **Auth**: none (public).
- **Pipeline**: per-extract loop — `osmium tags-filter` → pyosmium parse → write intermediate parquets → concat all intermediates → GeoParquet.
- **Entry**: [src/openpois/io/osm_snapshot.py](../../src/openpois/io/osm_snapshot.py).
- **Quirks**:
  - **Ingest filter is crosswalk-derived & value-scoped** (both OSM snapshot and history). `taxonomy.build_osm_tag_filter_expressions` builds the `osmium tags-filter` expressions from `taxonomy_crosswalk_openstreetmap.csv`: keys with a `*` wildcard row stay key-level; keys without one are restricted to the listed values (`landuse=cemetery,religious`, `craft=<28 values>`). So broad keys like `landuse` no longer drag in every value. See [taxonomy-setup.md](taxonomy-setup.md). Passed via the `tag_filter_exprs` arg; omit it to fall back to the old key-level behavior.
  - `osmium` is in the conda env's `bin/` but **not** on shell PATH. Code resolves via `Path(sys.executable).parent / "osmium"`.
  - Geofabrik extracts are pre-cut to admin boundaries → no polygon post-filter needed. `american-oceania-latest.osm.pbf` ships a few non-target uninhabited US Pacific possessions (Wake, Midway, Howland, Baker, Jarvis, Palmyra, Kingman); they contain near-zero POIs and pass through as bonus coverage. **Note**: these POIs are *not* in the Census boundary polygon, so they pass through OSM but get dropped by Overture's `ST_Within` and won't appear in the conflated output. Any per-state rollup of the OSM-only snapshot will show them with an unrecognized `addr:state` (or no value) — expected, not a bug.
  - **Config key naming asymmetry**: `usvi_*` (lowercase abbrev) and `american_oceania_*` (underscored words). Easy to mistype. If a future extract is added, pick the convention that matches the closest existing key.

## Overture Maps

**Used by**: current-state Overture snapshot (`overture_snapshot.parquet`).

- **URL**: public S3 at `s3://overturemaps-us-west-2/`.
- **Auth**: none (DuckDB + httpfs queries directly).
- **Pipeline**: per-part resumable download → exact-polygon filter, all inside DuckDB. Each of the 16 `part-*.parquet` files streams through a fresh DuckDB connection into a local parquet intermediate under `.parts/<release>/`; coarse-bbox `WHERE` pushes down on Overture's `bbox` struct. Once every part is present, a final `COPY` applies `ST_Within` against the dissolved US + territories polygon and writes the GeoParquet. No pandas materialization; crashed runs resume by skipping existing intermediates.
- **Entry**: [src/openpois/io/overture.py](../../src/openpois/io/overture.py). Returns a `Path`, not a `GeoDataFrame`.
- **DuckDB version pin**: `environment.yml` pins `duckdb==1.4.1`. 1.4.4+ and every 1.5.x crash mid-scan on WSL2 with "Information loss on integer cast" in `HTTPFileSystem::ReadInternal` — tracked as DuckDB issue #21669, fix merged to main but not in any tagged release as of 2026-04-17. See [memory: project_duckdb_pin.md] for the bump checklist.
- **Schema quirks (as of Feb 2026 schema)**:
  - `taxonomy` is a named STRUCT `{primary, hierarchy[], alternates[]}` — use `taxonomy.hierarchy[1]` **not** `taxonomy[1]`.
  - `brand` is a singular struct, **not** a `brands[]` array.
  - L0 category names: `food_and_drink`, `shopping`, `arts_and_entertainment`, `sports_and_recreation`, `health_care`.
  - Geometry is native DuckDB GEOMETRY — must `LOAD spatial;` and use `ST_X()` / `ST_Y()`.
- **Upcoming migration (~June 2026)**: L0/L1 hierarchy → flat `basic_category`. Crosswalk CSV + `assign_overture_shared_label` will need updating.

## Census boundary

**Used by**: both snapshot downloaders (spatial clipping).

- **URL**: `download.general.boundary.source_url` → `https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip` (1:5M cartographic, 50 states + DC + 5 inhabited territories: PR, VI, GU, MP, AS). Note: the 1:20M variant used previously does **not** include territories.
- **Auth**: none.
- **Pipeline**: download ZIP → cache under `directories.boundary` (first-use) → dissolve → buffer outward by `coastline_buffer_m` (default 100 m) in EPSG:6933 (equal-area, so buffer accurate across CONUS / AK / HI / Caribbean territories / western Pacific territories).
- **Entry**: [src/openpois/io/boundary.py](../../src/openpois/io/boundary.py) (`get_us_pr_boundary`).
- **Function-name caveat**: `get_us_pr_boundary` / `load_us_pr_boundary` / `us_pr_unary_polygon` / `us_pr_bboxes` are now misnomers — they cover all 56 STUSPS codes, not just US + PR. Names kept for backwards-compat with existing callers (`scripts/overture/download.py`, etc.); don't rename without coordinating the call sites. Docstrings reflect the actual scope.
- **Returns**: `(boundary_gdf, coarse_bboxes)` — single-row dissolved+buffered polygon (EPSG:4326) plus a list of bboxes for predicate pushdown.
- **Antimeridian**: two bboxes returned, split via per-part centroid at lon=0. The negative-longitude bbox covers CONUS, AK mainland, HI, PR, USVI, and American Samoa (~-170°W). The positive-longitude bbox covers the Aleutian Near Islands (~+172°E), Guam (~+144°E), and the Northern Mariana Islands (~+145°E).
- **Stale cached file**: `directories.boundary` is `versioned: false`. After the 1:20M → 1:5M swap (2026-05-21) the old `cb_2023_us_state_20m.*` files linger in the cache directory. Harmless but worth deleting the next time someone tidies the cache.

