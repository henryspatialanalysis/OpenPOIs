# Export the environment to a yml file
export_env:
	@conda env export > environment.yml;

# Build conda environment from the yml file
build_env:
	@conda env create -f environment.yml;

# Install the package to pip
install_package:
	@pip install -e .;

# Run the unit test suite (no network calls, runs in seconds)
test:
	@python -m pytest tests/ -v;

# Lint source code, exploratory scripts, and tests
# Uses the openpois conda env's binaries regardless of whether it is activated
CONDA_PYTHON := $(shell conda run -n openpois which python 2>/dev/null || echo python)
CONDA_BIN := $(dir $(CONDA_PYTHON))

lint:
	@$(CONDA_BIN)flake8 src/ scripts/ tests/
	@$(CONDA_BIN)pylint src/openpois/

# Build the site for production
site_build:
	@cd site && npm run build;

# Serve the site locally with hot reload
# Note: does not build Sphinx docs; use site_preview for a full build
site_dev:
	@cd site && npm run dev;

# Generate site/public/taxonomy.html from the conflation data CSVs
# Requires the openpois conda env to be active (for pandas)
build_taxonomy:
	@python scripts/build_taxonomy.py;

# Full build + local preview: Sphinx docs, Vite production build, then serve
# Mirrors the GitHub Actions workflow; serves at http://localhost:4173
# Requires the openpois conda env to be active (for sphinx-build)
# Uses Python's HTTP server instead of vite preview so /docs/ is served
# correctly (vite preview uses SPA fallback which swallows directory requests)
site_preview:
	@python scripts/build_taxonomy.py
	@sphinx-build -b html docs docs/_build/html -q
	@cd site && npm run build
	@cp -r docs/_build/html site/dist/docs
	@python -m http.server 4173 --directory site/dist;

# -----------------------------------------------------------------------------
# Conflation pipeline (canonical entry point for all national runs)
#
# `make conflate` runs the three steps that produce the published
# conflated.parquet:
#
#   1. build_ghosts.py            - reconstruct "ghost" POI dataset
#                                    from OSM history (deletions,
#                                    primary-tag removals, lifecycle
#                                    prefixes, substantial renames).
#   2. conflate.py                 - OSM x Overture matching as before,
#                                    written to conflated_baseline.parquet
#                                    so the pre-CD result is archived.
#   3. apply_change_detection.py   - penalize Overture POIs that shadow-
#                                    match a ghost; emits the canonical
#                                    conflated.parquet that downstream
#                                    summarize / format_for_upload /
#                                    prepare_pmtiles / publish steps
#                                    consume.
#
# Each sub-step tees a per-run log under ~/data/openpois/logs/.
#
# Pass TEST=1 to scope to the Seattle bbox:
#     make conflate            # full CONUS
#     make conflate TEST=1     # Seattle bbox dry run
#
# Sub-targets (build_ghosts / conflate_baseline / apply_cd) are exposed
# for partial re-runs when one stage is being iterated on.

TEST ?=
TEST_FLAG := $(if $(TEST),--test,)
LOG_DIR := $(HOME)/data/openpois/logs
LOG_TS := $(shell date +%Y%m%d_%H%M%S)

.PHONY: rate conflate build_ghosts conflate_baseline apply_cd

# Rate the OSM snapshot with the production random_effects model (per-POI cell
# reconstruction). Uses apply_model.model_stub from config; pass MODEL_VERSION=
# to override. NOTE: this is the correct rater for random_effects — the older
# apply_model.py is per-group only and must not be used for it.
rate:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/osm_snapshot/apply_model_random_effects.py \
		$(if $(MODEL_VERSION),--model-version $(MODEL_VERSION),) $(TEST_FLAG) \
		2>&1 | tee $(LOG_DIR)/rate_$(LOG_TS).log

build_ghosts:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/conflation/build_ghosts.py \
		2>&1 | tee $(LOG_DIR)/build_ghosts_$(LOG_TS).log

conflate_baseline:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/conflation/conflate.py \
		--output-suffix=baseline $(TEST_FLAG) \
		2>&1 | tee $(LOG_DIR)/conflate_baseline_$(LOG_TS).log

apply_cd:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/conflation/apply_change_detection.py \
		--baseline-suffix=baseline --output-suffix="" $(TEST_FLAG) \
		2>&1 | tee $(LOG_DIR)/apply_cd_$(LOG_TS).log

conflate: build_ghosts conflate_baseline apply_cd
	@echo
	@echo "Conflation pipeline complete."
	@echo "  Canonical output: ~/data/openpois/conflation/<version>/conflated.parquet"
	@echo "  (no-CD archive:   conflated_baseline.parquet)"
	@echo "  Logs under: $(LOG_DIR)/{build_ghosts,conflate_baseline,apply_cd}_$(LOG_TS).log"

# Convenience target to print all of the available targets in this file
# From https://stackoverflow.com/questions/4219255
.PHONY: list
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | \
		awk -v RS= -F: '/^# File/,/^# Finished Make data base/ \
		{if ($$1 !~ "^[#.]") {print $$1}}' | \
		sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
