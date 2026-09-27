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
# `make conflate` runs the five steps that produce the published
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
#                                    match a same-entity ghost; writes
#                                    conflated_cd.parquet.
#   4. calibrate                   - fit + apply the existence-confidence
#                                    curves; writes the canonical
#                                    conflated.parquet.
#   5. apply_manual_overrides.py   - hand-curated exclude/include pins
#                                    (Close triage CSV), rewritten in
#                                    place over conflated.parquet. Runs
#                                    LAST so a forced conf_mean is never
#                                    re-scaled by calibration.
#
# Each sub-step tees a per-run log under ~/data/openpois/logs/.
#
# Pass TEST=1 to scope to the Seattle bbox:
#     make conflate            # full CONUS
#     make conflate TEST=1     # Seattle bbox dry run
#
# Sub-targets (build_ghosts / conflate_baseline / apply_cd / calibrate /
# apply_manual_overrides) are exposed for partial re-runs when one stage
# is being iterated on.

TEST ?=
TEST_FLAG := $(if $(TEST),--test,)
LOG_DIR := $(HOME)/data/openpois/logs
LOG_TS := $(shell date +%Y%m%d_%H%M%S)

.PHONY: download_history check_history rate conflate build_ghosts conflate_baseline apply_cd \
	fit_calibration apply_calibration calibrate apply_manual_overrides

# Build versions.osm_data: the full-history download (history_mode: full) or a
# roll-forward of download.osm.incremental_history.base_version with Geofabrik's
# daily diffs (history_mode: incremental). PLAN=1 prints the diff sequences an
# incremental run would fetch and exits.
download_history:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/osm_data/download_history.py \
		$(if $(PLAN),--plan-only,) \
		2>&1 | tee $(LOG_DIR)/osm_history_$(LOG_TS).log

# QA: the history's last state of each snapshot node must match the snapshot.
check_history:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/osm_data/check_history_vs_snapshot.py \
		2>&1 | tee $(LOG_DIR)/check_history_$(LOG_TS).log

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
		--baseline-suffix=baseline --output-suffix=cd $(TEST_FLAG) \
		2>&1 | tee $(LOG_DIR)/apply_cd_$(LOG_TS).log

# Fit the per-segment existence-confidence curves from the validation handoff
# pinned by versions.calibration, then map every POI through them. Calibration
# runs AFTER change detection: the CD penalty multiplies conf_mean, so
# calibrating first would leave a calibrated probability scaled by delta.
fit_calibration:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/conflation/fit_calibration.py \
		--input-suffix=cd $(TEST_FLAG) \
		2>&1 | tee $(LOG_DIR)/fit_calibration_$(LOG_TS).log

apply_calibration:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/conflation/apply_calibration.py \
		--input-suffix=cd --output-suffix="" $(TEST_FLAG) \
		2>&1 | tee $(LOG_DIR)/apply_calibration_$(LOG_TS).log

calibrate: fit_calibration apply_calibration
	@$(CONDA_PYTHON) -u scripts/conflation/plot_calibration.py \
		2>&1 | tee $(LOG_DIR)/plot_calibration_$(LOG_TS).log

# Manual exclude/include pins from the Close triage CSV. Must run AFTER
# calibrate: it rewrites conflated.parquet in place and a forced conf_mean
# of 0 / 1 must not be re-scaled by the curves. A missing CSV is a no-op.
apply_manual_overrides:
	@mkdir -p $(LOG_DIR)
	@$(CONDA_PYTHON) -u scripts/conflation/apply_manual_overrides.py \
		$(TEST_FLAG) \
		2>&1 | tee $(LOG_DIR)/apply_manual_overrides_$(LOG_TS).log

conflate: build_ghosts conflate_baseline apply_cd calibrate apply_manual_overrides
	@echo
	@echo "Conflation pipeline complete."
	@echo "  Canonical output: ~/data/openpois/conflation/<version>/conflated.parquet"
	@echo "  (calibrated + manual overrides applied in place)"
	@echo "  (pre-calibration: conflated_cd.parquet)"
	@echo "  (no-CD archive:   conflated_baseline.parquet)"
	@echo "  Curves + fit report: conflation/<version>/calibration/"
	@echo "  Logs under: $(LOG_DIR)/{build_ghosts,conflate_baseline,apply_cd,fit_calibration,apply_calibration,apply_manual_overrides}_$(LOG_TS).log"

# Convenience target to print all of the available targets in this file
# From https://stackoverflow.com/questions/4219255
.PHONY: list
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | \
		awk -v RS= -F: '/^# File/,/^# Finished Make data base/ \
		{if ($$1 !~ "^[#.]") {print $$1}}' | \
		sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
