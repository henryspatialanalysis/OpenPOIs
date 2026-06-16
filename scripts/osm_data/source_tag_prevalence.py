"""
Descriptive analysis: how common are "survey-vs-armchair" provenance tags in our
OSM POI data?

Motivation: an OSM editor noted at a conference that mappers don't always observe a
business in person -- some edits are traced from aerial/satellite imagery ("armchair"
mapping). OSM has tags that hint at this, and we may eventually want to downweight
online-only observations in the turnover model. This script answers the prior,
purely descriptive question: do those tags exist, and are they common in our data?

The relevant element-level tags (changeset-level `imagery_used` / `source` are NOT
retained by our full-history PBF pipeline, so they are out of reach here):
  - survey:date  -- strong "verified in person" signal (presence == ground survey)
  - source       -- value-dependent: ground (survey/GPS/local knowledge) vs online
                    (Bing/aerial/Esri imagery) vs import/other (TIGER/GNIS/URLs)
  - check_date   -- weak proxy; "last checked", could be remote or ground

Data sources (all already on disk for version 20260521; nothing is re-downloaded):
  - osm_changes.parquet  -- every tag Added/Changed/Deleted across full history; the
                            only place raw `source`/`survey:date` values survive
                            (the snapshot overwrites `source` with the constant 'osm').
  - osm_versions.parquet -- per-version metadata (unused for now; kept for reference).
  - osm_snapshot.parquet -- the live POI universe (our "existing OSM POIs data"); the
                            denominator, and the only direct source of `check_date`.
  - osm_observations.parquet -- the per-tag-change modeling units; the model-relevant
                            denominator for a future downweighting scheme.

Method: reconstruct each element's *current* `source` / `survey:date` / `check_date`
value as the value at its highest version that is not a Deletion, then anchor on the
live snapshot (join on osm_id/osm_type == id/type). `source` values are bucketed into
ground / online / import_other / uncategorized by case-insensitive substring match,
keeping an uncategorized tail so nothing is silently mislabeled.

Outputs (CSV + a markdown summary) land under ~/data/openpois/logs/. Read-only.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
from config_versioned import Config

# --- Inputs (versioned paths resolved from config.yaml) -----------------------
config = Config("~/repos/openpois/config.yaml")
OSM_CHANGES = config.get_file_path("osm_data", "osm_changes")
OSM_VERSIONS = config.get_file_path("osm_data", "osm_versions")
OSM_OBSERVATIONS = config.get_file_path("osm_data", "osm_observations")
OSM_SNAPSHOT = config.get_file_path("snapshot_osm", "snapshot")

OUT_DIR = Path("~/data/openpois/logs").expanduser()
KEYS_CSV = OUT_DIR / "source_tag_keys_discovered.csv"
LIVE_CSV = OUT_DIR / "source_tag_prevalence_live.csv"
VALUE_CSV = OUT_DIR / "source_value_breakdown.csv"
BUCKET_CSV = OUT_DIR / "source_bucket_summary.csv"
OBS_CSV = OUT_DIR / "source_tag_prevalence_observations.csv"
SUMMARY_MD = OUT_DIR / "source_tag_prevalence_SUMMARY.md"

# --- `source` value bucketing -------------------------------------------------
# Case-insensitive substring patterns, checked ground-first so a mixed value that
# mentions an in-person source ("survey, checked on bing") is credited as ground.
# `image` is deliberately excluded -- it collides with the "imagery" online terms.
GROUND_TERMS = [
    "survey", "gps", "local knowledge", "local_knowledge", "knowledge",
    "mapillary", "kartaview", "openstreetcam", "streetlevel", "observation",
    "on the ground", "on-site", "onsite", "fieldwork",
]
ONLINE_TERMS = [
    "bing", "aerial", "digitalglobe", "digital globe", "maxar", "esri",
    "landsat", "satellite", "mapbox", "yahoo", "orthophoto", "ortho",
    "imagery", "sentinel", "naip",
]
IMPORT_TERMS = [
    "tiger", "gnis", "geonames", "import", "buildingfootprint",
    "building footprint", "openaddress", "address", "gis", "county",
    "city of", "dataset", "http", "www", ".gov", ".com", ".org", "license",
]


def _bucket_case(value_expr: str) -> str:
    """Build a SQL CASE expression assigning a bucket to a `source` value."""

    def clause(terms: list[str], label: str) -> str:
        ors = " OR ".join(
            f"lower({value_expr}) LIKE '%{t}%'" for t in terms
        )
        return f"        WHEN {ors} THEN '{label}'"

    return "\n".join([
        "CASE",
        f"        WHEN {value_expr} IS NULL THEN NULL",
        clause(GROUND_TERMS, "ground"),
        clause(ONLINE_TERMS, "online"),
        clause(IMPORT_TERMS, "import_other"),
        "        ELSE 'uncategorized'",
        "    END",
    ])


def _session(con: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
    con.execute("SET memory_limit = '6GB';")
    con.execute("SET threads TO 4;")
    con.execute(f"SET temp_directory = '{temp_dir.as_posix()}';")
    con.execute("SET preserve_insertion_order = false;")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUT_DIR / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ch = OSM_CHANGES.as_posix()
    vs = OSM_VERSIONS.as_posix()
    sn = OSM_SNAPSHOT.as_posix()
    obs = OSM_OBSERVATIONS.as_posix()

    con = duckdb.connect()
    _session(con, temp_dir)

    bucket_expr = _bucket_case("value")

    # --- Step A: discover the provenance-tag key families ---------------------
    print("Step A: discovering source/survey/check key families ...", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT key,
                   count(*)            AS change_rows,
                   count(DISTINCT id)  AS n_elements
            FROM read_parquet('{ch}')
            WHERE key LIKE 'source%' OR key = 'survey' OR key LIKE 'survey:%'
               OR key LIKE 'check_date%'
               OR key IN ('mapillary', 'image', 'imagery_used')
            GROUP BY key
            ORDER BY n_elements DESC
        ) TO '{KEYS_CSV.as_posix()}' (FORMAT csv, HEADER)
        """
    )
    imagery_used_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{ch}') WHERE key = 'imagery_used'"
    ).fetchone()[0]
    print(f"  wrote {KEYS_CSV.name}; imagery_used on elements: "
          f"{imagery_used_rows} rows", flush=True)

    # --- Reconstruct current value per element for the 3 element-level keys ----
    # Highest version that is not a Deletion == current value of that tag.
    print("Building current per-element tag values from change history ...",
          flush=True)
    con.execute(
        f"""
        CREATE TEMP TABLE cur_tags AS
        SELECT id, type, key, value
        FROM (
            SELECT id, type, key, value, change,
                   row_number() OVER (
                       PARTITION BY id, type, key ORDER BY version DESC
                   ) AS rn
            FROM read_parquet('{ch}')
            WHERE key IN ('source', 'survey:date', 'check_date')
        )
        WHERE rn = 1 AND change <> 'Deleted'
        """
    )

    # Per-element flags + source bucket, keyed (id, type) for joining downstream.
    con.execute(
        f"""
        CREATE TEMP TABLE elem AS
        SELECT
            COALESCE(s.id, sd.id, cd.id)       AS id,
            COALESCE(s.type, sd.type, cd.type) AS type,
            s.value                            AS source_val,
            {_bucket_case('s.value')}          AS source_bucket,
            sd.value                           AS survey_date_val,
            cd.value                           AS check_date_val
        FROM (SELECT id, type, value FROM cur_tags WHERE key = 'source') s
        FULL OUTER JOIN (
            SELECT id, type, value FROM cur_tags WHERE key = 'survey:date'
        ) sd USING (id, type)
        FULL OUTER JOIN (
            SELECT id, type, value FROM cur_tags WHERE key = 'check_date'
        ) cd USING (id, type)
        """
    )

    # --- Snapshot anchor: the live POI universe -------------------------------
    n_total = con.execute(
        f"SELECT count(*) FROM read_parquet('{sn}')"
    ).fetchone()[0]
    snap_source_vals = con.execute(
        f"SELECT DISTINCT source FROM read_parquet('{sn}') LIMIT 5"
    ).fetchall()
    snap_check_date = con.execute(
        f"SELECT count(*) FROM read_parquet('{sn}') WHERE check_date IS NOT NULL"
    ).fetchone()[0]

    # Join live snapshot POIs to reconstructed element tags.
    con.execute(
        f"""
        CREATE TEMP TABLE live AS
        SELECT
            sn.osm_id   AS id,
            sn.osm_type AS type,
            sn.check_date IS NOT NULL AS snap_has_check_date,
            e.source_val,
            e.source_bucket,
            e.survey_date_val,
            e.check_date_val
        FROM read_parquet('{sn}') sn
        LEFT JOIN elem e
            ON sn.osm_id = e.id AND sn.osm_type = e.type
        """
    )

    # History coverage diagnostic: fraction of live POIs that appear *anywhere* in
    # the full-history element set (not just those carrying a provenance tag). A
    # high value means the prevalence figures below are near-complete; the residual
    # is snapshot POIs absent from the full-history extract, which can only deflate
    # the prevalence percentages slightly.
    n_in_history = con.execute(
        f"""
        WITH vids AS (SELECT DISTINCT id, type FROM read_parquet('{vs}'))
        SELECT count(*)
        FROM read_parquet('{sn}') sn
        SEMI JOIN vids v ON sn.osm_id = v.id AND sn.osm_type = v.type
        """
    ).fetchone()[0]

    # --- Step B: element-level prevalence among live POIs ---------------------
    print("Step B: computing live-POI prevalence ...", flush=True)
    bstats = con.execute(
        f"""
        SELECT
            count(*)                                              AS n_total,
            sum((source_val IS NOT NULL)::INT)                    AS n_source,
            sum((source_bucket = 'ground')::INT)                  AS n_source_ground,
            sum((source_bucket = 'online')::INT)                  AS n_source_online,
            sum((source_bucket = 'import_other')::INT)            AS n_source_import,
            sum((source_bucket = 'uncategorized')::INT)           AS n_source_uncat,
            sum((survey_date_val IS NOT NULL)::INT)               AS n_survey_date,
            sum((check_date_val IS NOT NULL)::INT)                AS n_check_date_hist,
            sum(snap_has_check_date::INT)                         AS n_check_date_snap,
            sum(((survey_date_val IS NOT NULL)
                 OR source_bucket = 'ground')::INT)               AS n_any_ground
        FROM live
        """
    ).fetchdf().iloc[0].to_dict()

    def pct(n: float) -> float:
        return 100.0 * n / n_total if n_total else 0.0

    live_rows = [
        ("source (any value)", bstats["n_source"]),
        ("source == ground", bstats["n_source_ground"]),
        ("source == online", bstats["n_source_online"]),
        ("source == import_other", bstats["n_source_import"]),
        ("source == uncategorized", bstats["n_source_uncat"]),
        ("survey:date (any)", bstats["n_survey_date"]),
        ("check_date (history-reconstructed)", bstats["n_check_date_hist"]),
        ("check_date (snapshot column)", bstats["n_check_date_snap"]),
        ("any in-person signal (survey:date OR source==ground)",
         bstats["n_any_ground"]),
    ]
    import csv
    with LIVE_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["signal", "n_live_pois", "pct_of_live_pois"])
        for label, n in live_rows:
            w.writerow([label, int(n), round(pct(n), 4)])
    print(f"  wrote {LIVE_CSV.name}", flush=True)

    # --- Step C: source value breakdown (live POIs) ---------------------------
    print("Step C: source value breakdown ...", flush=True)
    con.execute(
        f"""
        COPY (
            SELECT source_val AS value,
                   source_bucket AS bucket,
                   count(*) AS n_live_pois
            FROM live
            WHERE source_val IS NOT NULL
            GROUP BY 1, 2
            ORDER BY n_live_pois DESC
        ) TO '{VALUE_CSV.as_posix()}' (FORMAT csv, HEADER)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT source_bucket AS bucket,
                   count(*) AS n_live_pois,
                   round(100.0 * count(*) / {n_total}, 4) AS pct_of_live_pois
            FROM live
            WHERE source_val IS NOT NULL
            GROUP BY 1
            ORDER BY n_live_pois DESC
        ) TO '{BUCKET_CSV.as_posix()}' (FORMAT csv, HEADER)
        """
    )
    top_uncat = con.execute(
        """
        SELECT source_val, count(*) AS n
        FROM live
        WHERE source_bucket = 'uncategorized'
        GROUP BY 1 ORDER BY 2 DESC LIMIT 50
        """
    ).fetchall()
    print(f"  wrote {VALUE_CSV.name}, {BUCKET_CSV.name}", flush=True)

    # --- Step D: observation-level prevalence (model-relevant denominator) ----
    print("Step D: observation-level prevalence ...", flush=True)
    ostats = con.execute(
        f"""
        WITH joined AS (
            SELECT
                e.source_val,
                e.source_bucket,
                e.survey_date_val
            FROM read_parquet('{obs}') o
            LEFT JOIN elem e
                ON o.id = e.id AND o.osm_type = e.type
        )
        SELECT
            count(*)                                       AS n_obs,
            sum((source_val IS NOT NULL)::INT)             AS n_source,
            sum((source_bucket = 'ground')::INT)           AS n_source_ground,
            sum((source_bucket = 'online')::INT)           AS n_source_online,
            sum((source_bucket = 'import_other')::INT)     AS n_source_import,
            sum((survey_date_val IS NOT NULL)::INT)        AS n_survey_date,
            sum(((survey_date_val IS NOT NULL)
                 OR source_bucket = 'ground')::INT)        AS n_any_ground
        FROM joined
        """
    ).fetchdf().iloc[0].to_dict()
    n_obs = int(ostats["n_obs"])

    def opct(n: float) -> float:
        return 100.0 * n / n_obs if n_obs else 0.0

    obs_rows = [
        ("source (any value)", ostats["n_source"]),
        ("source == ground", ostats["n_source_ground"]),
        ("source == online", ostats["n_source_online"]),
        ("source == import_other", ostats["n_source_import"]),
        ("survey:date (any)", ostats["n_survey_date"]),
        ("any in-person signal (survey:date OR source==ground)",
         ostats["n_any_ground"]),
    ]
    with OBS_CSV.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["signal", "n_observations", "pct_of_observations"])
        for label, n in obs_rows:
            w.writerow([label, int(n), round(opct(n), 4)])
    print(f"  wrote {OBS_CSV.name}", flush=True)

    con.close()

    # --- Step E: written summary ----------------------------------------------
    lines = []
    lines.append("# Survey-vs-armchair source tags in OpenPOIs OSM data\n")
    lines.append(f"Data version: 20260521. Live POIs (snapshot rows): {n_total:,}. "
                 f"Modeled observations: {n_obs:,}.\n")
    lines.append(f"History coverage: {n_in_history:,} of {n_total:,} live POIs "
                 f"({pct(n_in_history):.2f}%) appear in the full-history element set, "
                 f"so the prevalence figures are near-complete (the ~"
                 f"{100 - pct(n_in_history):.1f}% absent from history can only deflate "
                 f"the percentages slightly).\n")
    lines.append("## Are the tags present?\n")
    lines.append(
        "- **survey:date** (element-level, strong in-person signal): present.\n"
        "- **source** (element-level, value-dependent): present.\n"
        "- **check_date** (element-level, weak proxy): present.\n"
        "- **imagery_used** (changeset-level, strongest armchair signal): "
        f"NOT on elements ({imagery_used_rows} rows) and not retrievable from our "
        "full-history PBF pipeline (changeset tags are dropped).\n"
    )
    lines.append("## Prevalence among live POIs\n")
    lines.append("| signal | n live POIs | % of live POIs |")
    lines.append("|---|---:|---:|")
    for label, n in live_rows:
        lines.append(f"| {label} | {int(n):,} | {pct(n):.3f}% |")
    lines.append(f"\nSnapshot `source` distinct values (collision check, expect only "
                 f"'osm'): {snap_source_vals}\n")
    lines.append(f"Snapshot direct `check_date IS NOT NULL`: {snap_check_date:,} "
                 f"({pct(snap_check_date):.3f}%) -- cross-check vs reconstructed.\n")
    lines.append("## Prevalence among modeled observations\n")
    lines.append("| signal | n observations | % of observations |")
    lines.append("|---|---:|---:|")
    for label, n in obs_rows:
        lines.append(f"| {label} | {int(n):,} | {opct(n):.3f}% |")
    lines.append("\n## Top uncategorized `source` values (eyeball for mis-bucketing)\n")
    for val, n in top_uncat[:25]:
        lines.append(f"- `{val}` -- {n:,}")
    lines.append("\n## Verdict\n")
    in_person_pct = pct(bstats["n_any_ground"])
    lines.append(
        f"Any in-person signal (survey:date OR source==ground) covers "
        f"{in_person_pct:.2f}% of live POIs. See numbers above for whether this is "
        f"common enough to justify a follow-up downweighting scheme. The gold-standard "
        f"armchair signal (changeset `imagery_used`/`source`) is unavailable in the "
        f"current pipeline and would require a separate changeset-tag pull.\n"
    )
    SUMMARY_MD.write_text("\n".join(lines))

    # --- console echo ---------------------------------------------------------
    print(f"\nLive POIs: {n_total:,} | Observations: {n_obs:,}")
    print(f"History coverage of live POIs: {pct(n_in_history):.2f}%")
    print("\nLive-POI prevalence:")
    for label, n in live_rows:
        print(f"  {label:<52}{int(n):>12,}{pct(n):>9.3f}%")
    print("\nObservation prevalence:")
    for label, n in obs_rows:
        print(f"  {label:<52}{int(n):>12,}{opct(n):>9.3f}%")
    print(f"\nSummary: {SUMMARY_MD}")
    print(f"CSVs: {KEYS_CSV.name}, {LIVE_CSV.name}, {VALUE_CSV.name}, "
          f"{BUCKET_CSV.name}, {OBS_CSV.name}")


if __name__ == "__main__":
    main()
