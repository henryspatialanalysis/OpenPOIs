API Reference
=============

conflation
----------

openpois.conflation.match
~~~~~~~~~~~~~~~~~~~~~~~~~

Spatial candidate matching and composite scoring for POI conflation. Provides
a BallTree-based radius search to find nearby OSM–Overture candidate pairs
within category-specific thresholds, a multi-component scorer (distance, name
similarity, taxonomy agreement, shared identifiers), and a greedy one-to-one
assignment step that filters below a minimum composite score.

.. automodule:: openpois.conflation.match
   :members:
   :undoc-members:
   :show-inheritance:

openpois.conflation.merge
~~~~~~~~~~~~~~~~~~~~~~~~~

Merge matched and unmatched POIs into a unified conflated GeoDataFrame.
Produces a superset containing matched OSM–Overture pairs with blended
confidence scores, unmatched OSM POIs at their original confidence, and
unmatched Overture POIs at downweighted confidence. Uses a disk-backed
split-then-concat pattern to avoid peak memory issues at CONUS scale.

.. automodule:: openpois.conflation.merge
   :members:
   :undoc-members:
   :show-inheritance:

openpois.conflation.taxonomy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Taxonomy crosswalk between OSM tags and the Overture Maps category hierarchy.
Loads four CSV reference files (OSM crosswalk, Overture crosswalk, match radii,
and top-level key-to-L0 mappings) and provides functions to assign each POI a
``shared_label`` string, a per-category spatial match radius, and an L0 bitmask
used for type-agreement scoring.

.. automodule:: openpois.conflation.taxonomy
   :members:
   :undoc-members:
   :show-inheritance:

----

io
--

openpois.io.osm_history_pbf
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download Geofabrik full-history PBFs (US + inhabited territories: Puerto
Rico, US Virgin Islands, plus Guam / NMI / American Samoa via the
``american-oceania`` extract), filter to POI tags with ``osmium tags-filter``,
time-window with ``osmium time-filter``, and parse with pyosmium into
per-version and per-change Parquet tables suitable for the change-rate
model. Uses an OAuth cookie jar against Geofabrik's internal server.
Per-extract failure tolerance: missing-on-server (HTTP 404) territory PBFs
are skipped with a warning rather than aborting the run.

.. automodule:: openpois.io.osm_history_pbf
   :members:
   :undoc-members:
   :show-inheritance:

openpois.io.osm_snapshot
~~~~~~~~~~~~~~~~~~~~~~~~

Download a current US-wide OSM POI snapshot from a Geofabrik PBF extract.
Streams the PBF (~11 GB), runs ``osmium tags-filter`` to reduce it to
matching tag keys, then parses nodes and way centroids with pyosmium into
a GeoParquet file. The osmium binary is resolved from the conda environment
rather than the system PATH.

.. automodule:: openpois.io.osm_snapshot
   :members:
   :undoc-members:
   :show-inheritance:

openpois.io.overture
~~~~~~~~~~~~~~~~~~~~

Download a current US-wide Overture Maps Places snapshot. Uses DuckDB's
``httpfs`` and ``spatial`` extensions to query Overture GeoParquet files
directly from public S3, filtering by bounding box and L0 taxonomy category.
No authentication is required. Auto-detects the latest Overture release date
from S3 if a specific date is not pinned.

.. automodule:: openpois.io.overture
   :members:
   :undoc-members:
   :show-inheritance:

openpois.io.geohash_partition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for spatially partitioning GeoDataFrames by geohash for efficient
web-map viewport queries. Computes geohash columns from geometry centroids,
writes Hive-style partitioned Parquet datasets (``geohash_prefix=XX/``), and
sorts rows within each partition by a finer geohash for spatial locality.

.. automodule:: openpois.io.geohash_partition
   :members:
   :undoc-members:
   :show-inheritance:

openpois.io.source_coop
~~~~~~~~~~~~~~~~~~~~~~~

Upload a locally partitioned dataset to Source Cooperative's S3-compatible
storage. Walks the Hive partition directory, uploads each Parquet file under
a versioned prefix, and reports the public URL on completion. Credentials
come from a JSON file at the repo root (``publish.credentials_file``).

.. automodule:: openpois.io.source_coop
   :members:
   :undoc-members:
   :show-inheritance:

openpois.io.credentials
~~~~~~~~~~~~~~~~~~~~~~~

Load Source Cooperative AWS-compatible credentials from a JSON file. Tokens
are short-lived (~1 hour); the loader logs a clear error pointing at the
credentials regeneration URL when the file is stale or missing.

.. automodule:: openpois.io.credentials
   :members:
   :undoc-members:
   :show-inheritance:

----

models
------

openpois.models.jax_core
~~~~~~~~~~~~~~~~~~~~~~~~

JAX/BlackJAX helpers: a PRNG factory, a jitted Markov-chain scan, a NUTS
sampler with window adaptation, and a vmap-based predictive-draw utility.

.. automodule:: openpois.models.jax_core
   :members:
   :undoc-members:
   :show-inheritance:

openpois.models.model_fitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BlackJAX NUTS fitter for POI change-rate models. Takes an ``event_rate_fun``
plus starting parameters as a pytree, runs window-adapted NUTS to draw from
the posterior, and produces posterior summaries and predictive distributions
of change probability versus time.

.. automodule:: openpois.models.model_fitter
   :members:
   :undoc-members:
   :show-inheritance:

openpois.models.osm_models
~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX model classes for OSM turnover. ``ConstantModel`` and
``RandomByTypeModel`` package their own data, priors, and event-rate
functions to hand to ``ModelFitter``. Selectable via ``get_model_class``.

.. automodule:: openpois.models.osm_models
   :members:
   :undoc-members:
   :show-inheritance:

openpois.models.setup
~~~~~~~~~~~~~~~~~~~~~

Data-preparation helpers. ``prepare_data_for_model`` filters and groups
observation records and computes the ``tag_years`` elapsed column used as
the per-observation interval length in fitting.

.. automodule:: openpois.models.setup
   :members:
   :undoc-members:
   :show-inheritance:

openpois.models.apply
~~~~~~~~~~~~~~~~~~~~~

Apply saved change-rate model predictions to a POI snapshot. Loads
``predictions.csv`` from a versioned model output directory and builds
fast numpy lookup arrays (indexed by group and time step) for both constant
and random-effects model variants.

.. automodule:: openpois.models.apply
   :members:
   :undoc-members:
   :show-inheritance:

----

osm
---

openpois.osm.format_observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert raw OSM version histories into modelling-ready observation records.
Joins version and change tables to produce one row per element version,
with timestamps for the previous and current tag values and a flag indicating
whether the configured tag changed at this version.

.. automodule:: openpois.osm.format_observations
   :members:
   :undoc-members:
   :show-inheritance:

openpois.osm.change_plots
~~~~~~~~~~~~~~~~~~~~~~~~~

Kaplan-Meier-style tag stability plots using plotnine. Computes the
proportion of tag assignments that remain unchanged over time from
observation records, and renders single-panel and faceted multi-panel
figures saved as PNG files.

.. automodule:: openpois.osm.change_plots
   :members:
   :undoc-members:
   :show-inheritance:
