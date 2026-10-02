"""
Download OSM full-history data for the US + inhabited territories for POI
change-rate modelling.

Downloads four Geofabrik full-history PBFs (US + Puerto Rico + US Virgin
Islands + American Oceania — which bundles Guam, NMI, American Samoa, and
the uninhabited US Pacific possessions), filters them to the configured POI
tag keys, slices to the configured date range with ``osmium time-filter``,
and streams each element's versions into osm_versions.parquet plus one row
per tag-level change into osm_changes.parquet. Those two Parquets feed
format_tabular.py unchanged.

Geofabrik's internal server requires OSM OAuth. Point ``history_cookie_file``
at a Netscape-format cookie jar (export from a browser logged in at
osm-internal.download.geofabrik.de, or use Geofabrik's oauth_cookie_client.py).

Per-extract failure tolerance: if a territory's history PBF is not published
on Geofabrik's internal server (HTTP 404), the loader logs a warning and
continues without that territory's history. Snapshot/Overture coverage is
unaffected; the rater falls back to the global-mean delta for any
``shared_label`` that lacks territory-specific history evidence.

Config keys used (config.yaml):
    download.osm.history_pbf_url                  — Geofabrik US history PBF URL
    download.osm.pr_history_pbf_url               — Geofabrik PR history PBF URL
    download.osm.usvi_history_pbf_url             — Geofabrik USVI history PBF URL
    download.osm.american_oceania_history_pbf_url — Geofabrik American-Oceania
                                                    history PBF URL (covers
                                                    Guam, NMI, American Samoa)
    download.osm.history_cookie_file  — cookie file for Geofabrik OAuth (or null)
    download.osm.filter_keys          — OSM tag keys to retain
    download.osm.start_date           — start of time-filter window
    download.osm.end_date             — end of time-filter window
    download.osm.overwrite_download   — re-download raw PBFs if present
    download.osm.overwrite_filter     — re-run tags-filter/time-filter if present
    download.osm.overwrite_parse      — re-run parse if Parquets are present
    download.osm.chunk_size           — rows per Parquet-writer flush
    download.osm.verbose              — print progress
    directories.osm_data              — output directory (versioned)

Incremental mode (``download.osm.history_mode: incremental``) skips the
full-history download: it rolls ``download.osm.incremental_history.base_version``
forward with Geofabrik's public daily diffs through ``end_date``, writing the
same two Parquets (ghost-grade only; see
``openpois.io.osm_history_incremental``). ``--plan-only`` prints the diff
sequences each feed would fetch and exits without downloading.

Output files (in osm_data directory):
    osm_versions.parquet   — one row per element version
    osm_changes.parquet    — one row per per-version tag change (Added/Changed/Deleted)
    history_coverage.json  — full vs incremental, coverage end, sequences reached

Usage:
    python scripts/osm_data/download_history.py [--plan-only]
"""
import argparse
import datetime

from config_versioned import Config

from openpois.conflation.taxonomy import (
    build_osm_tag_filter_expressions,
    load_osm_crosswalk,
)
from openpois.io.osm_history_incremental import (
    HistoryCoverage,
    plan_sequences,
    read_coverage,
    replication_state_getter,
    roll_osm_history,
    write_coverage,
)
from openpois.io.osm_history_pbf import HistoryExtract, download_osm_history

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

HISTORY_COOKIE_FILE = config.get(
    "download", "osm", "history_cookie_file", fail_if_none = False
)
FILTER_KEYS = config.get("download", "osm", "filter_keys")
# Value-scope keys to the taxonomy crosswalk (e.g. landuse=cemetery,religious)
# so the history POI set matches the snapshot ingest filter.
TAG_FILTER_EXPRS = build_osm_tag_filter_expressions(load_osm_crosswalk())
START_DATE = datetime.datetime.combine(
    config.get("download", "osm", "start_date"), datetime.time.min
)
END_DATE = datetime.datetime.combine(
    config.get("download", "osm", "end_date"), datetime.time.min
)
OVERWRITE_DOWNLOAD = config.get("download", "osm", "overwrite_download")
OVERWRITE_FILTER = config.get("download", "osm", "overwrite_filter")
OVERWRITE_PARSE = config.get("download", "osm", "overwrite_parse")
CHUNK_SIZE = config.get("download", "osm", "chunk_size")
VERBOSE = config.get("download", "osm", "verbose")
HISTORY_MODE = config.get("download", "osm", "history_mode")
INCREMENTAL = config.get("download", "osm", "incremental_history")

SAVE_DIR = config.get_dir_path("osm_data")
SAVE_DIR.mkdir(parents = True, exist_ok = True)

OUTPUT_VERSIONS = config.get_file_path("osm_data", "osm_versions")
OUTPUT_CHANGES = config.get_file_path("osm_data", "osm_changes")
BASE_DIR = config.get_dir_path(
    "osm_data", custom_version = INCREMENTAL["base_version"]
)

# One HistoryExtract per Geofabrik full-history PBF. Order is preserved
# through to the concat step; keep the US-mainland extract first since it
# dominates wall time and ``_concat_history`` only drops rows from later
# extracts.
EXTRACTS = [
    HistoryExtract(
        name = "us",
        url = config.get("download", "osm", "history_pbf_url"),
        raw_pbf_path = config.get_file_path("osm_data", "raw_history_pbf"),
        filtered_pbf_path = config.get_file_path("osm_data", "filtered_history_pbf"),
        time_filtered_pbf_path = config.get_file_path(
            "osm_data", "time_filtered_history_pbf"
        ),
        versions_path = config.get_file_path("osm_data", "us_versions"),
        changes_path = config.get_file_path("osm_data", "us_changes"),
    ),
    HistoryExtract(
        name = "pr",
        url = config.get("download", "osm", "pr_history_pbf_url"),
        raw_pbf_path = config.get_file_path("osm_data", "raw_pr_history_pbf"),
        filtered_pbf_path = config.get_file_path("osm_data", "filtered_pr_history_pbf"),
        time_filtered_pbf_path = config.get_file_path(
            "osm_data", "time_filtered_pr_history_pbf"
        ),
        versions_path = config.get_file_path("osm_data", "pr_versions"),
        changes_path = config.get_file_path("osm_data", "pr_changes"),
    ),
    HistoryExtract(
        name = "usvi",
        url = config.get("download", "osm", "usvi_history_pbf_url"),
        raw_pbf_path = config.get_file_path("osm_data", "raw_usvi_history_pbf"),
        filtered_pbf_path = config.get_file_path(
            "osm_data", "filtered_usvi_history_pbf"
        ),
        time_filtered_pbf_path = config.get_file_path(
            "osm_data", "time_filtered_usvi_history_pbf"
        ),
        versions_path = config.get_file_path("osm_data", "usvi_versions"),
        changes_path = config.get_file_path("osm_data", "usvi_changes"),
    ),
    HistoryExtract(
        name = "american_oceania",
        url = config.get("download", "osm", "american_oceania_history_pbf_url"),
        raw_pbf_path = config.get_file_path(
            "osm_data", "raw_american_oceania_history_pbf"
        ),
        filtered_pbf_path = config.get_file_path(
            "osm_data", "filtered_american_oceania_history_pbf"
        ),
        time_filtered_pbf_path = config.get_file_path(
            "osm_data", "time_filtered_american_oceania_history_pbf"
        ),
        versions_path = config.get_file_path("osm_data", "american_oceania_versions"),
        changes_path = config.get_file_path("osm_data", "american_oceania_changes"),
    ),
]


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run_full() -> None:
    """Download, filter and parse the full-history PBFs (the original path)."""
    download_osm_history(
        extracts = EXTRACTS,
        output_versions_path = OUTPUT_VERSIONS,
        output_changes_path = OUTPUT_CHANGES,
        filter_keys = FILTER_KEYS,
        tag_filter_exprs = TAG_FILTER_EXPRS,
        start_date = START_DATE,
        end_date = END_DATE,
        cookie_file = HISTORY_COOKIE_FILE,
        overwrite_download = OVERWRITE_DOWNLOAD,
        overwrite_filter = OVERWRITE_FILTER,
        overwrite_parse = OVERWRITE_PARSE,
        chunk_size = CHUNK_SIZE,
        verbose = VERBOSE,
    )

    # -------------------------------------------------------------------------
    # Clean up intermediates
    # -------------------------------------------------------------------------
    finals_ok = all(
        p.exists() and p.stat().st_size > 0
        for p in (OUTPUT_VERSIONS, OUTPUT_CHANGES)
    )
    if finals_ok:
        intermediates = [
            p
            for spec in EXTRACTS
            for p in (
                spec.raw_pbf_path,
                spec.filtered_pbf_path,
                spec.time_filtered_pbf_path,
                spec.versions_path,
                spec.changes_path,
            )
        ]
        for p in intermediates:
            if p.exists():
                print(f"Removing intermediate {p} ...")
                p.unlink()
        write_coverage(SAVE_DIR, HistoryCoverage(
            mode = "full",
            coverage_end = END_DATE.replace(tzinfo = datetime.timezone.utc),
            filter_exprs = list(TAG_FILTER_EXPRS),
        ))


def plan_incremental() -> None:
    """Print the diff sequences each feed would fetch; download nothing."""
    coverage = read_coverage(BASE_DIR)
    print(
        f"Base {INCREMENTAL['base_version']}: mode {coverage.mode}, "
        f"coverage_end {coverage.coverage_end.isoformat()}, "
        f"chain length {coverage.chain_length}"
    )
    end_utc = END_DATE.replace(tzinfo = datetime.timezone.utc)
    if end_utc <= coverage.coverage_end:
        print(
            f"WARNING: end_date {end_utc.isoformat()} is not after the base "
            "coverage; advance download.osm.end_date. Planning to the newest "
            "diff instead."
        )
        end_utc = None
    for name, url in INCREMENTAL["replication_urls"].items():
        get_state = replication_state_getter(url)
        newest = get_state(None)
        if newest is None:
            print(f"  {name}: feed unavailable ({url})")
            continue
        plan = plan_sequences(
            get_state, coverage.coverage_end, end_utc,
            last_sequence = coverage.extracts.get(name, {}).get("last_sequence"),
        )
        if plan is None:
            print(f"  {name}: nothing new (server newest {newest.sequence})")
            continue
        print(
            f"  {name}: sequences {plan.start}-{plan.end} "
            f"({plan.end - plan.start + 1} files, through "
            f"{plan.end_timestamp.isoformat()}; server newest "
            f"{newest.sequence} at {newest.timestamp.isoformat()})"
        )


def run_incremental() -> None:
    """Roll the base run's parquets forward with the daily diffs."""
    roll_osm_history(
        base_dir = BASE_DIR,
        base_version = INCREMENTAL["base_version"],
        out_dir = SAVE_DIR,
        out_versions_path = OUTPUT_VERSIONS,
        out_changes_path = OUTPUT_CHANGES,
        replication_urls = INCREMENTAL["replication_urls"],
        tag_filter_exprs = TAG_FILTER_EXPRS,
        end_date = END_DATE.replace(tzinfo = datetime.timezone.utc),
        max_chain_months = INCREMENTAL["max_chain_months"],
        keep_diffs = INCREMENTAL["keep_diffs"],
        verbose = VERBOSE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    parser.add_argument(
        "--plan-only",
        action = "store_true",
        help = "Incremental mode: print the diff sequences to fetch and exit.",
    )
    args = parser.parse_args()

    print(f"History mode: {HISTORY_MODE}")
    if HISTORY_MODE == "full":
        if args.plan_only:
            raise SystemExit("--plan-only applies to history_mode: incremental")
        run_full()
    elif HISTORY_MODE == "incremental":
        if args.plan_only:
            plan_incremental()
            raise SystemExit(0)
        run_incremental()
    else:
        raise ValueError(
            f"download.osm.history_mode must be 'full' or 'incremental', "
            f"got {HISTORY_MODE!r}"
        )
    config.write_self("osm_data")
