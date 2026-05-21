Workflows
=========

This page describes the four end-to-end pipelines that produce the openpois
dataset, in the order they are executed. Each pipeline is implemented as a
series of scripts in the ``scripts/`` directory; the scripts call library
functions documented in the :doc:`api`.

All scripts read their configuration from ``config.yaml`` via
``config_versioned.Config``. See the individual script docstrings for the
exact config keys each script uses.


Prerequisites
-------------

**Python environment.** Install the conda env from ``environment.yml`` and
the package itself in editable mode:

.. code-block:: bash

   make build_env       # conda env create -f environment.yml (env name: openpois)
   conda activate openpois
   make install_package # pip install -e .

**Geofabrik OAuth (Pipeline 2 only).** Pipeline 2 downloads full-history
PBFs from Geofabrik's OAuth-protected internal server. Any OSM account grants
access. Generate a Netscape-format cookie jar by logging in at
``https://osm-internal.download.geofabrik.de/`` and exporting cookies, or by
running Geofabrik's ``oauth_cookie_client.py``. Save the cookie jar at the
path configured in ``config.yaml`` under ``download.osm.history_cookie_file``.

**Source Cooperative credentials (publishing only).** Publishing the data
back to Source Cooperative requires short-lived AWS-style credentials in a
JSON file at the path configured under ``publish.credentials_file`` (default:
``.env.json`` at the repo root). The format is documented in the
``scripts/publish/upload_to_source_coop.py`` docstring. Replicators who do
not intend to publish can stop after Pipeline 4 Step 3.

----

Pipeline 1: POI Snapshot Downloads
----------------------------------

These two scripts are independent and can be run in any order (or in
parallel). Each downloads a current US-wide POI snapshot from one data
source and saves it as a GeoParquet file.

**OSM snapshot**

.. code-block:: bash

   python scripts/osm_snapshot/download.py

Downloads the Geofabrik North America PBF extract (~11 GB), filters with
osmium, and parses with pyosmium. Output: ``osm_snapshot.parquet``
(~7.8 M POIs).

See :mod:`openpois.io.osm_snapshot`.

**Overture Maps snapshot**

.. code-block:: bash

   python scripts/overture/download.py

Queries Overture Maps GeoParquet files on public S3 via DuckDB. No
credentials required. Output: ``overture_snapshot.parquet`` (~13 M POIs).

See :mod:`openpois.io.overture`.

**Quick schema inspection** *(optional)*

.. code-block:: bash

   python scripts/snapshots/load_samples.py

Reads the first 100 rows of each snapshot without loading the full files,
saving snippet CSVs to the ``testing/`` directory for column inspection.

----

Pipeline 2: OSM Historical Change-Rate Model
--------------------------------------------

This pipeline downloads OpenStreetMap full-history PBFs (US + inhabited
territories: Puerto Rico, US Virgin Islands, plus Guam / NMI / American
Samoa via the ``american-oceania`` extract) and fits a Poisson change-rate
model to estimate how quickly different POI categories become outdated.

**Step 1 — Download full-history PBFs**

.. code-block:: bash

   python scripts/osm_data/download_history.py

Requires the Geofabrik OAuth cookie jar described in *Prerequisites* above.
Downloads each Geofabrik full-history extract in turn, filters each with
``osmium tags-filter`` (POI tag keys only) and ``osmium time-filter`` (the
``download.osm.start_date`` / ``end_date`` window), then parses with
pyosmium into per-extract Parquet tables and concatenates with iterative
``(type, id)`` dedup. If a territory's history PBF is missing on the server
(HTTP 404), the loader logs a warning and continues; that territory's
snapshot/Overture coverage is unaffected and the rater falls back to the
global-mean delta. Outputs: ``osm_versions.parquet`` and
``osm_changes.parquet``.

See :mod:`openpois.io.osm_history_pbf`.

**Step 2 — Reformat into observations**

.. code-block:: bash

   python scripts/osm_data/format_tabular.py

Converts raw version histories into one-row-per-observation records, each
flagged for whether the configured ``osm_data.tag_key`` changed, then
assigns a shared taxonomy label and explodes rows for POIs mapping to
multiple labels. Output: ``osm_observations.parquet``.

See :mod:`openpois.osm.format_observations`.

**Step 3 — Fit the change-rate model**

.. code-block:: bash

   python scripts/models/osm_turnover.py

Fits an empirical Bayes JAX model (constant or random-effects by type)
estimating the Poisson change rate λ per group via BlackJAX NUTS. Outputs
``fitted_params.csv`` and ``predictions.csv`` (and optionally
``param_draws.csv``).

See :mod:`openpois.models.model_fitter`, :mod:`openpois.models.osm_models`,
and :mod:`openpois.models.setup`.

**Step 4 — Visualise stability curves** *(optional)*

.. code-block:: bash

   python scripts/osm_data/data_viz.py

Produces Kaplan-Meier-style survival curve plots saved to
``osm_data/viz/``.

See :mod:`openpois.osm.change_plots`.

----

Pipeline 3: Rate the OSM Snapshot
---------------------------------

This pipeline applies the fitted change-rate model (Pipeline 2) to the OSM
snapshot (Pipeline 1) to assign a confidence score to every POI.

**Prerequisites:** Pipeline 2 (model fitted) and Pipeline 1 OSM snapshot.

**Step 1 — Apply model predictions**

.. code-block:: bash

   python scripts/osm_snapshot/apply_model.py
   python scripts/osm_snapshot/apply_model.py --test   # first 10 k rows only

Matches each POI to its best-fit model group (by tag key priority), then
looks up the predicted change probability at the POI's age. Adds columns
``conf_mean``, ``conf_lower``, ``conf_upper``, ``t2_years``,
``model_version``, and ``model_group``. Output: ``osm_snapshot_rated.parquet``.

See :mod:`openpois.models.apply`.

**Step 2 — Partition for upload**

.. code-block:: bash

   python scripts/osm_snapshot/format_for_upload.py

Adds geohash columns and writes a Hive-style partitioned dataset so the web
map can fetch only the tiles it needs. Output: ``osm_snapshot_partitioned/``.

See :mod:`openpois.io.geohash_partition`.

**Step 3 — Build OSM PMTiles**

.. code-block:: bash

   python scripts/osm_snapshot/prepare_pmtiles.py

Generates a single-zoom (z14) PMTiles archive from the partitioned dataset
for use by the web map. Output: ``osm_snapshot.pmtiles``.

See :mod:`openpois.io.pmtiles`.

----

Pipeline 4: Conflation and Publishing
-------------------------------------

This pipeline conflates the rated OSM snapshot with the Overture Maps
snapshot into a single unified POI dataset and publishes it to Source
Cooperative.

**Prerequisites:** Pipeline 3 rated OSM snapshot and Pipeline 1 Overture
snapshot.

**Step 1 — Conflate**

.. code-block:: bash

   python scripts/conflation/conflate.py
   python scripts/conflation/conflate.py --test   # Seattle bbox only

Assigns shared taxonomy labels, finds spatial candidates via BallTree, scores
on distance + name + type + identifiers, performs greedy one-to-one matching,
and saves a unified GeoParquet. Output: ``conflated.parquet``.

See :mod:`openpois.conflation.match`, :mod:`openpois.conflation.merge`, and
:mod:`openpois.conflation.taxonomy`.

**Step 2 — Summarise** *(optional)*

.. code-block:: bash

   python scripts/conflation/summarize.py

Produces a summary CSV with match counts and average match scores per
shared taxonomy label. Output: ``summary_by_label.csv``.

**Step 3 — Partition and build conflated PMTiles**

.. code-block:: bash

   python scripts/conflation/format_for_upload.py
   python scripts/conflation/prepare_pmtiles.py

Adds geohash columns and writes a Hive-style partitioned dataset, then
builds a single-zoom (z14) PMTiles archive of the conflated points.
Outputs: ``conflated_partitioned/`` and ``conflated.pmtiles``.

See :mod:`openpois.io.geohash_partition` and :mod:`openpois.io.pmtiles`.

**Step 4 — Publish to Source Cooperative** *(optional)*

.. code-block:: bash

   python scripts/publish/upload_to_source_coop.py

Uploads the partitioned conflated dataset, the partitioned OSM dataset,
both PMTiles archives, and a per-version README to Source Cooperative
under the ``versions.source_coop`` folder. Requires the credentials file
described in *Prerequisites*. Skip this step if you only want the data
locally.

See :mod:`openpois.io.source_coop` and :mod:`openpois.publish.build_readme`.
