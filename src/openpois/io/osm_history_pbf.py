#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module downloads full-history OpenStreetMap data for the US + inhabited
territories for POI change-rate modelling using Geofabrik full-history PBF
extracts, osmium-tool CLI pre-filtering, and pyosmium streaming.

It is broken into the following functions:

- download_history_pbf: Downloads a .osh.pbf file (optionally authenticated via
    a Geofabrik OAuth cookie jar) via streaming HTTP.
- filter_history_pbf: Runs osmium tags-filter --omit-referenced to produce a
    reduced POI-only history PBF.
- time_filter_history_pbf: Runs osmium time-filter FROM TO to slice the history
    PBF to versions active in a given date range.
- parse_history_pbf: Streams a filtered history PBF with pyosmium and writes
    per-version metadata (osm_versions.parquet) and per-version tag diffs
    (osm_changes.parquet).
- download_osm_history: End-to-end orchestrator. Downloads each Geofabrik
    history extract in the provided list, filters and time-filters each,
    parses each, concatenates the results with per-element dedup, and writes
    final versions/changes Parquets. If a per-extract download returns 404
    (e.g. a territory's history is not published on Geofabrik's internal
    server), the loader logs a warning and continues without that extract;
    its snapshot/Overture POIs still flow through downstream stages and the
    rater falls back to the global-mean delta for that territory.

Data sources (Geofabrik internal full-history extracts; full set passed via
the ``extracts`` argument to ``download_osm_history``). Paths are relative
to ``https://osm-internal.download.geofabrik.de/``:
    - US mainland (all 50 states incl. AK + HI, ~50 GB):
      ``north-america/us-internal.osh.pbf``
    - Puerto Rico (separate extract):
      ``north-america/us/puerto-rico-internal.osh.pbf``
    - US Virgin Islands (separate extract):
      ``north-america/us/us-virgin-islands-internal.osh.pbf``
    - American Oceania (Guam + NMI + American Samoa + uninhabited US Pacific
      possessions):
      ``australia-oceania/american-oceania-internal.osh.pbf``

All URLs live on Geofabrik's OAuth-protected internal server and require a
valid OSM-account cookie jar. Any OSM account grants access; see the README
section on cookie acquisition for details.

osmium-tool CLI must be installed (conda install -c conda-forge osmium-tool).

Note: This module is separate from openpois.io.osm_snapshot, which extracts
the current POI snapshot only.
"""
from __future__ import annotations

import datetime
import http.cookiejar
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import osmium
import pyarrow as pa
import pyarrow.parquet as pq
import requests


class HistoryExtract(NamedTuple):
    """One Geofabrik full-history extract to be downloaded, filtered,
    time-filtered, and parsed.

    Attributes:
        name: Short label used for logging and intermediate filenames
            (e.g. ``"us"``, ``"pr"``, ``"usvi"``, ``"american_oceania"``).
            Must be unique within the list passed to ``download_osm_history``.
        url: Geofabrik ``*-internal.osh.pbf`` URL.
        raw_pbf_path: Local path to store the downloaded raw history PBF.
        filtered_pbf_path: Local path to store the tags-filtered PBF.
        time_filtered_pbf_path: Local path to store the time-filtered PBF.
        versions_path: Intermediate per-extract versions Parquet.
        changes_path: Intermediate per-extract changes Parquet.
    """
    name: str
    url: str
    raw_pbf_path: Path
    filtered_pbf_path: Path
    time_filtered_pbf_path: Path
    versions_path: Path
    changes_path: Path


# -----------------------------------------------------------------------------
# Parquet schemas
# -----------------------------------------------------------------------------

VERSIONS_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("version", pa.int64()),
    ("changeset", pa.int64()),
    ("timestamp", pa.string()),
    ("user", pa.string()),
    ("uid", pa.int64()),
    ("type", pa.string()),
])

CHANGES_SCHEMA = pa.schema([
    ("key", pa.string()),
    ("value", pa.string()),
    ("change", pa.string()),
    ("id", pa.int64()),
    ("version", pa.int64()),
    ("type", pa.string()),
])


# -----------------------------------------------------------------------------
# osmium resolution (shared with osm_snapshot; osmium is in conda env bin,
# not necessarily on PATH)
# -----------------------------------------------------------------------------


def _resolve_osmium() -> str:
    """Return the path to the osmium binary (env bin fallback)."""
    env_bin = Path(sys.executable).parent / "osmium"
    return (
        shutil.which("osmium") or (str(env_bin) if env_bin.exists() else "osmium")
    )


# -----------------------------------------------------------------------------
# Tag-diff logic
# -----------------------------------------------------------------------------


def _diff_tag_sets(
    prev_tags: set[tuple[str, str]],
    curr_tags: set[tuple[str, str]],
) -> list[dict]:
    """
    Compute tag-level changes between two versions' tag sets.

    Returns list-of-dicts. The classification rule: Added if the key is only
    in curr_tags, Deleted if only in prev_tags, Changed if the key is in
    both but with different values.

    Args:
        prev_tags: Set of (key, value) tuples from the previous version.
        curr_tags: Set of (key, value) tuples from the current version.

    Returns:
        List of dicts with keys ``key``, ``value``, ``change`` where change is
        one of ``"Added"``, ``"Changed"``, ``"Deleted"``.
    """
    new_tuples = curr_tags - prev_tags
    removed_tuples = prev_tags - curr_tags
    new_keys = {k for k, _ in new_tuples}
    removed_keys = {k for k, _ in removed_tuples}
    rows: list[dict] = []
    for key, value in new_tuples:
        change = "Changed" if key in removed_keys else "Added"
        rows.append({"key": key, "value": value, "change": change})
    for key, value in removed_tuples:
        if key in new_keys:
            continue  # already emitted as "Changed"
        rows.append({"key": key, "value": value, "change": "Deleted"})
    return rows


# -----------------------------------------------------------------------------
# Download helper
# -----------------------------------------------------------------------------


def _load_cookie_session(cookie_file: Path | None) -> requests.Session:
    """
    Build a requests.Session with cookies loaded from a Netscape-format jar.

    Args:
        cookie_file: Path to a Netscape (Mozilla) cookie jar, or None for an
            unauthenticated session.

    Returns:
        Configured requests.Session.

    Raises:
        FileNotFoundError: If cookie_file is given but does not exist.
    """
    session = requests.Session()
    if cookie_file is None:
        return session
    cookie_path = Path(cookie_file).expanduser()
    if not cookie_path.exists():
        raise FileNotFoundError(
            f"Geofabrik cookie file not found: {cookie_path}. Generate one by "
            "logging in at https://osm-internal.download.geofabrik.de/ and "
            "exporting cookies, or run Geofabrik's oauth_cookie_client.py."
        )
    jar = http.cookiejar.MozillaCookieJar(str(cookie_path))
    jar.load(ignore_discard=True, ignore_expires=True)
    session.cookies = jar
    return session


def download_history_pbf(
    url: str,
    output_path: Path,
    cookie_file: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Downloads a full-history PBF file from the given URL via streaming HTTP.

    Writes to a temporary file in the same directory and renames atomically on
    success so a partial download never masquerades as a complete file.

    Args:
        url: URL of the history PBF file to download.
        output_path: Local path to save the downloaded PBF.
        cookie_file: Path to a Netscape-format cookie jar for Geofabrik OAuth,
            or None for an unauthenticated session (fine for public extracts,
            required for the internal server).
        overwrite: If False and output_path already exists, skip the download.

    Returns:
        Path to the downloaded PBF file.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        FileNotFoundError: If cookie_file is given but does not exist.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"History PBF already exists at {output_path}; skipping download.")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = _load_cookie_session(cookie_file)
    print(f"Downloading history PBF from {url} to {output_path}...")
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with session.get(url, stream=True, timeout=(30, None)) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"  {pct:.1f}%", end="\r")
        tmp_path.rename(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"\nDownload complete: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# osmium-tool filters
# -----------------------------------------------------------------------------


def _extract_poi_ids(
    tag_filtered_pbf: Path,
    ids_path: Path,
) -> Path:
    """Write a sorted list of unique ``<type><id>`` strings (e.g.
    ``n12345``, ``w67890``) for every element seen in a tag-filtered
    history PBF.

    Used as the input to ``osmium getid`` in the second filter pass.
    """
    seen: set[tuple[str, int]] = set()
    fp = osmium.FileProcessor(str(tag_filtered_pbf))
    for obj in fp:
        kind = "n" if obj.is_node() else ("w" if obj.is_way() else "r")
        seen.add((kind, int(obj.id)))
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ids_path, "w") as f:
        for kind, oid in sorted(seen):
            f.write(f"{kind}{oid}\n")
    return ids_path


def filter_history_pbf(
    input_pbf: Path,
    output_pbf: Path,
    osm_keys: list[str],
    overwrite: bool = False,
    tag_filter_exprs: list[str] | None = None,
) -> Path:
    """
    Two-pass POI extraction from a full-history PBF.

    A single ``osmium tags-filter`` pass drops every version whose tags
    don't match the filter expression — including **deletion versions**,
    which carry no tags. That's a significant loss for the change-rate
    pipeline: we never see ``visible: true → false`` transitions.

    To recover deletions while still narrowing to POI elements, we run
    two osmium-tool passes:

    1. ``osmium tags-filter --omit-referenced`` on the raw history PBF
       produces an intermediate file (``<output_stem>-tagfilt.osh.pbf``)
       containing only versions whose tags match. We use this only to
       enumerate the set of element IDs that have ever carried a POI
       tag.
    2. ``osmium getid --with-history`` on the **raw** history PBF, given
       that ID list, produces the final filtered history PBF —
       preserving *every* version (including invisible deletion
       versions) of every POI-tagged element.

    The intermediate tag-filter file and the IDs list are removed after
    pass 2 completes successfully.

    Args:
        input_pbf: Path to the raw history PBF.
        output_pbf: Path to write the final two-pass filtered PBF.
        osm_keys: OSM tag keys identifying POIs. Used to build key-level
            ``nwr/<key>`` expressions when ``tag_filter_exprs`` is not given.
        overwrite: If False and ``output_pbf`` exists, skip filtering.
        tag_filter_exprs: Optional pre-built osmium filter expressions (e.g.
            ``["nwr/amenity", "nwr/landuse=cemetery,religious"]``) used
            verbatim instead of ``osm_keys``. Lets callers value-scope keys
            to the taxonomy crosswalk values via
            ``taxonomy.build_osm_tag_filter_expressions``, keeping the
            history POI set aligned with the snapshot.

    Returns:
        Path to the filtered PBF file.

    Raises:
        subprocess.CalledProcessError: If osmium exits with non-zero
            status on either pass.
    """
    output_pbf = Path(output_pbf)
    if output_pbf.exists() and not overwrite:
        print(
            f"Filtered history PBF already exists at {output_pbf};"
            " skipping filter."
        )
        return output_pbf

    output_pbf.parent.mkdir(parents=True, exist_ok=True)
    osmium_bin = _resolve_osmium()
    key_args = (
        list(tag_filter_exprs)
        if tag_filter_exprs is not None
        else [f"nwr/{key}" for key in osm_keys]
    )

    # Intermediate paths under the same directory as the final output.
    stem = output_pbf.stem  # e.g. "us-pois"
    tagfilt_pbf = output_pbf.with_name(f"{stem}-tagfilt.osh.pbf")
    ids_path = output_pbf.with_name(f"{stem}-ids.txt")

    # Pass 1: tag filter to identify POI elements.
    cmd_tagfilt = [
        osmium_bin, "tags-filter",
        "--omit-referenced",
        "--overwrite",
        "--output-format=osh.pbf",
        "-o", str(tagfilt_pbf),
        str(input_pbf),
    ] + key_args
    print(f"Running: {' '.join(cmd_tagfilt)}")
    subprocess.run(cmd_tagfilt, check=True)

    # Extract IDs from the intermediate.
    print(f"Extracting element IDs from {tagfilt_pbf} ...")
    _extract_poi_ids(tagfilt_pbf, ids_path)

    # Pass 2: getid --with-history pulls every version (including
    # deletions) of those elements from the RAW history PBF.
    cmd_getid = [
        osmium_bin, "getid",
        "--with-history",
        "--overwrite",
        "--output-format=osh.pbf",
        "-i", str(ids_path),
        "-o", str(output_pbf),
        str(input_pbf),
    ]
    print(f"Running: {' '.join(cmd_getid)}")
    subprocess.run(cmd_getid, check=True)

    # Clean up intermediates so versioned dirs don't accumulate them.
    try:
        tagfilt_pbf.unlink()
        ids_path.unlink()
    except OSError:
        pass

    print(f"Filtered history PBF written to {output_pbf}")
    return output_pbf


def time_filter_history_pbf(
    input_pbf: Path,
    output_pbf: Path,
    start_date: datetime.datetime | datetime.date,
    end_date: datetime.datetime | datetime.date,
    overwrite: bool = False,
) -> Path:
    """
    Runs osmium time-filter FROM TO on a full-history PBF.

    With two ISO-formatted timestamps, ``osmium time-filter`` preserves every
    version active during the window and keeps the output in history format.
    A single-timestamp call would collapse the file to a snapshot — that is
    not what we want here.

    Args:
        input_pbf: Path to the tag-filtered history PBF.
        output_pbf: Path to write the time-filtered history PBF.
        start_date: Start of the window (inclusive). datetime or date.
        end_date: End of the window (exclusive per osmium semantics).
        overwrite: If False and output_pbf exists, skip the filter.

    Returns:
        Path to the time-filtered PBF file.

    Raises:
        subprocess.CalledProcessError: If osmium exits with non-zero status.
    """
    output_pbf = Path(output_pbf)
    if output_pbf.exists() and not overwrite:
        print(
            f"Time-filtered history PBF already exists at {output_pbf};"
            " skipping time-filter."
        )
        return output_pbf

    output_pbf.parent.mkdir(parents=True, exist_ok=True)
    osmium_bin = _resolve_osmium()
    start_iso = _to_iso_z(start_date)
    end_iso = _to_iso_z(end_date)
    cmd = [
        osmium_bin, "time-filter",
        "--overwrite",
        "--output-format=osh.pbf",
        "-o", str(output_pbf),
        str(input_pbf),
        start_iso,
        end_iso,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Time-filtered history PBF written to {output_pbf}")
    return output_pbf


def _to_iso_z(value: datetime.datetime | datetime.date) -> str:
    """Format a datetime/date as YYYY-MM-DDTHH:MM:SSZ for osmium."""
    if isinstance(value, datetime.datetime):
        dt = value
    else:
        dt = datetime.datetime.combine(value, datetime.time.min)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# -----------------------------------------------------------------------------
# pyosmium streaming parser
# -----------------------------------------------------------------------------


def _flush_parquet(
    buffer: list[dict],
    writer: pq.ParquetWriter,
    schema: pa.Schema,
) -> None:
    """Append rows in ``buffer`` to an open ParquetWriter and clear the list."""
    if not buffer:
        return
    columns = {field.name: [row.get(field.name) for row in buffer] for field in schema}
    table = pa.table(columns, schema=schema)
    writer.write_table(table)
    buffer.clear()


def _tag_set_for_version(obj: osmium.osm.OSMObject) -> set[tuple[str, str]]:
    """
    Build the (key, value) tag set for one element version.

    OSM tags are combined with the pseudo-tags ``visible`` (and ``lat``/``lon``
    for nodes) so that lat/lon edits and visibility changes show up as entries
    in osm_changes.

    Args:
        obj: A pyosmium element version (node / way / relation).

    Returns:
        Set of (key, value) tuples.
    """
    tags: set[tuple[str, str]] = set()
    for tag in obj.tags:
        tags.add((tag.k, tag.v))
    tags.add(("visible", "true" if obj.visible else "false"))
    if obj.is_node():
        location = obj.location
        if location is not None and location.valid():
            tags.add(("lat", str(location.lat)))
            tags.add(("lon", str(location.lon)))
    return tags


def _kind_of(obj: osmium.osm.OSMObject) -> str:
    """Return 'node', 'way', or 'relation' for a pyosmium object."""
    if obj.is_node():
        return "node"
    if obj.is_way():
        return "way"
    return "relation"


def parse_history_pbf(
    pbf_path: Path,
    versions_path: Path,
    changes_path: Path,
    chunk_size: int = 500_000,
    overwrite: bool = False,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """
    Stream a filtered full-history PBF and write versions + changes Parquets.

    The pyosmium FileProcessor emits every version of every element in
    ``(type, id, version)`` order for a history PBF. For each version we
    compare its tag set against the previous version of the same element
    (reset whenever ``(type, id)`` changes) and emit:

    - one row per version to ``versions_path`` with
      ``id, version, changeset, timestamp, user, uid, type``;
    - one row per tag change (Added / Changed / Deleted) to ``changes_path``
      with ``key, value, change, id, version``.

    Includes ``visible``, ``lat``, and ``lon`` as pseudo-tags so that
    visibility toggles (deletions) and coordinate edits are captured in
    osm_changes — matches the behaviour of the existing Overpass-based
    pipeline.

    Args:
        pbf_path: Path to the (tag-filtered and optionally time-filtered)
            history PBF.
        versions_path: Destination Parquet for per-version metadata.
        changes_path: Destination Parquet for per-version tag diffs.
        chunk_size: Number of rows to buffer before each flush. Same value is
            applied independently to the versions and changes buffers.
        overwrite: If False and both destinations already exist, skip parsing.
        verbose: If True, print progress every chunk_size versions.

    Returns:
        Tuple ``(versions_path, changes_path)``.
    """
    versions_path = Path(versions_path)
    changes_path = Path(changes_path)
    if (
        versions_path.exists()
        and changes_path.exists()
        and not overwrite
    ):
        print(
            f"Versions+changes Parquets already exist at {versions_path.parent};"
            " skipping parse."
        )
        return versions_path, changes_path

    versions_path.parent.mkdir(parents=True, exist_ok=True)
    changes_path.parent.mkdir(parents=True, exist_ok=True)

    fp = osmium.FileProcessor(str(pbf_path))

    versions_buf: list[dict] = []
    changes_buf: list[dict] = []
    prev_key: tuple[str, int] | None = None
    prev_tags: set[tuple[str, str]] = set()
    total_versions = 0

    with (
        pq.ParquetWriter(versions_path, VERSIONS_SCHEMA) as v_writer,
        pq.ParquetWriter(changes_path, CHANGES_SCHEMA) as c_writer,
    ):
        for obj in fp:
            kind = _kind_of(obj)
            key = (kind, obj.id)
            if key != prev_key:
                prev_tags = set()

            curr_tags = _tag_set_for_version(obj)

            versions_buf.append({
                "id": int(obj.id),
                "version": int(obj.version),
                "changeset": int(obj.changeset),
                "timestamp": (
                    obj.timestamp.isoformat() if obj.timestamp else None
                ),
                "user": obj.user,
                "uid": int(obj.uid),
                "type": kind,
            })
            for diff_row in _diff_tag_sets(prev_tags, curr_tags):
                diff_row["id"] = int(obj.id)
                diff_row["version"] = int(obj.version)
                diff_row["type"] = kind
                changes_buf.append(diff_row)

            prev_key = key
            prev_tags = curr_tags

            if len(versions_buf) >= chunk_size:
                total_versions += len(versions_buf)
                _flush_parquet(versions_buf, v_writer, VERSIONS_SCHEMA)
                if verbose:
                    print(f"  Flushed versions ({total_versions:,} so far)")
            if len(changes_buf) >= chunk_size:
                _flush_parquet(changes_buf, c_writer, CHANGES_SCHEMA)

        # Final flush
        total_versions += len(versions_buf)
        _flush_parquet(versions_buf, v_writer, VERSIONS_SCHEMA)
        _flush_parquet(changes_buf, c_writer, CHANGES_SCHEMA)

    if verbose:
        print(
            f"Parsed {total_versions:,} versions from {pbf_path} →"
            f" {versions_path}, {changes_path}"
        )
    return versions_path, changes_path


# -----------------------------------------------------------------------------
# Parquet concatenation (N-way: US + territories) with cross-extract deduplication
# -----------------------------------------------------------------------------


def _concat_history(
    intermediates: list[tuple[Path, Path]],
    out_versions_path: Path,
    out_changes_path: Path,
) -> tuple[Path, Path]:
    """
    Stream-concatenate N per-extract versions/changes Parquet pairs into the
    final outputs, dropping rows for any element already written by an
    earlier extract.

    Geofabrik's per-state/-territory extracts share near-boundary elements:
    the same ``(type, id)`` version can legitimately appear in two extracts
    that touch the same area. Concatenating naively would produce duplicate
    rows per ``(id, version, key)`` in the changes Parquet, which breaks
    ``format_observations`` (it calls ``.loc[key, "change"]`` and expects a
    scalar, not a Series).

    Strategy (iterative over ``intermediates`` in the given order):
    - Stream-copy the first extract's versions to the output, collecting
      the set of ``(type, id)`` seen.
    - For each subsequent extract, drop any row whose ``(type, id)`` was
      already written, append the rest, and add the new ``(type, id)`` keys
      to the seen set. Track which ids were dropped per-extract so the
      matching changes rows can be dropped in the second pass.
    - Stream-copy the first extract's changes; for each subsequent extract,
      drop the changes rows tied to dropped versions and append the rest.

    Dedup is ``(type, id)``-keyed in both tables. OSM element ids are only
    unique *within* a type, so an id-only join would incorrectly collapse a
    node and a way that share an integer id — see the change-log for the
    bug that motivated adding ``type`` to ``osm_changes``.

    Putting the largest extract first (e.g. ``us`` before the territories)
    minimises wasted reads since dedup only ever drops rows from later
    extracts.

    Args:
        intermediates: List of ``(versions_path, changes_path)`` pairs, one
            per successfully-processed extract, in concatenation order.
        out_versions_path: Final concatenated versions Parquet.
        out_changes_path: Final concatenated changes Parquet.

    Returns:
        Tuple ``(out_versions_path, out_changes_path)``.
    """
    if not intermediates:
        raise ValueError(
            "`intermediates` must contain at least one (versions, changes) pair."
        )

    out_versions_path.parent.mkdir(parents=True, exist_ok=True)
    out_changes_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: versions. Stream-copy the first extract, then per-extract
    # filter+append each subsequent one. Track per-extract dropped (type, id)
    # so pass 2 can match the changes rows.
    seen_type_ids: set[tuple[str, int]] = set()
    dropped_per_extract: list[set[tuple[str, int]]] = [set()]
    with pq.ParquetWriter(out_versions_path, VERSIONS_SCHEMA) as writer:
        first_versions_path, _ = intermediates[0]
        reader = pq.ParquetFile(str(first_versions_path))
        for batch in reader.iter_batches():
            tbl = pa.Table.from_batches([batch], schema=VERSIONS_SCHEMA)
            seen_type_ids.update(
                zip(
                    tbl.column("type").to_pylist(),
                    tbl.column("id").to_pylist(),
                )
            )
            writer.write_table(tbl)
        for versions_path, _ in intermediates[1:]:
            v_tbl = pq.read_table(str(versions_path), schema=VERSIONS_SCHEMA)
            types = v_tbl.column("type").to_pylist()
            ids = v_tbl.column("id").to_pylist()
            keep_mask = [(t, i) not in seen_type_ids for t, i in zip(types, ids)]
            dropped = {
                (t, i) for (t, i), keep in zip(zip(types, ids), keep_mask)
                if not keep
            }
            dropped_per_extract.append(dropped)
            v_filtered = v_tbl.filter(pa.array(keep_mask, type=pa.bool_()))
            if v_filtered.num_rows > 0:
                seen_type_ids.update(
                    zip(
                        v_filtered.column("type").to_pylist(),
                        v_filtered.column("id").to_pylist(),
                    )
                )
                writer.write_table(v_filtered)

    # Pass 2: changes. Stream-copy the first extract's changes, then for
    # each subsequent extract drop the changes rows whose (type, id) was
    # dropped from versions and append the rest.
    with pq.ParquetWriter(out_changes_path, CHANGES_SCHEMA) as writer:
        _, first_changes_path = intermediates[0]
        reader = pq.ParquetFile(str(first_changes_path))
        for batch in reader.iter_batches():
            writer.write_table(pa.Table.from_batches([batch], schema=CHANGES_SCHEMA))
        for (_, changes_path), dropped in zip(intermediates[1:], dropped_per_extract[1:]):
            c_tbl = pq.read_table(str(changes_path), schema=CHANGES_SCHEMA)
            if dropped and c_tbl.num_rows > 0:
                keep_mask = [
                    (t, i) not in dropped
                    for t, i in zip(
                        c_tbl.column("type").to_pylist(),
                        c_tbl.column("id").to_pylist(),
                    )
                ]
                c_tbl = c_tbl.filter(pa.array(keep_mask, type=pa.bool_()))
            if c_tbl.num_rows > 0:
                writer.write_table(c_tbl)

    return out_versions_path, out_changes_path


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


def _download_filter_timefilter_parse(
    pbf_url: str,
    raw_pbf_path: Path,
    filtered_pbf_path: Path,
    time_filtered_pbf_path: Path,
    versions_path: Path,
    changes_path: Path,
    filter_keys: list[str],
    start_date: datetime.datetime | datetime.date,
    end_date: datetime.datetime | datetime.date,
    cookie_file: Path | None,
    overwrite_download: bool,
    overwrite_filter: bool,
    overwrite_parse: bool,
    chunk_size: int,
    verbose: bool,
    tag_filter_exprs: list[str] | None = None,
) -> tuple[Path, Path]:
    """Download + tags-filter + time-filter + parse one history PBF."""
    download_history_pbf(
        url=pbf_url,
        output_path=raw_pbf_path,
        cookie_file=cookie_file,
        overwrite=overwrite_download,
    )
    filter_history_pbf(
        input_pbf=raw_pbf_path,
        output_pbf=filtered_pbf_path,
        osm_keys=filter_keys,
        overwrite=overwrite_filter,
        tag_filter_exprs=tag_filter_exprs,
    )
    time_filter_history_pbf(
        input_pbf=filtered_pbf_path,
        output_pbf=time_filtered_pbf_path,
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite_filter,
    )
    return parse_history_pbf(
        pbf_path=time_filtered_pbf_path,
        versions_path=versions_path,
        changes_path=changes_path,
        chunk_size=chunk_size,
        overwrite=overwrite_parse,
        verbose=verbose,
    )


def download_osm_history(
    extracts: list[HistoryExtract],
    output_versions_path: Path,
    output_changes_path: Path,
    filter_keys: list[str],
    start_date: datetime.datetime | datetime.date,
    end_date: datetime.datetime | datetime.date,
    cookie_file: Path | None = None,
    overwrite_download: bool = False,
    overwrite_filter: bool = False,
    overwrite_parse: bool = False,
    chunk_size: int = 500_000,
    verbose: bool = True,
    tag_filter_exprs: list[str] | None = None,
) -> tuple[Path, Path]:
    """
    End-to-end orchestrator: download each Geofabrik full-history PBF in
    ``extracts``, filter and time-filter each, parse each to Parquets, and
    concatenate into the final versions + changes files.

    Per-extract failure tolerance: if ``download_history_pbf`` raises an
    ``HTTPError`` with status 404 for an extract (e.g. a territory's history
    is not published on Geofabrik's internal server), the loader logs a
    warning and continues without that extract. Its snapshot/Overture POIs
    still flow through the downstream stages; the rater simply falls back
    to the global-mean delta for that territory's ``shared_label``s. Any
    other download error propagates.

    Putting the largest extract first (e.g. ``us`` before the territories)
    minimises wasted work in ``_concat_history``, which only ever drops
    rows from later extracts.

    Args:
        extracts: List of ``HistoryExtract`` specs (one per Geofabrik
            full-history PBF). Names must be unique within the list.
        output_versions_path: Final concatenated osm_versions.parquet.
        output_changes_path: Final concatenated osm_changes.parquet.
        filter_keys: OSM tag keys passed to ``tags-filter`` (fallback when
            ``tag_filter_exprs`` is not given).
        tag_filter_exprs: Optional pre-built osmium filter expressions (see
            ``filter_history_pbf``) used to value-scope keys to the taxonomy
            crosswalk, keeping the history POI set aligned with the snapshot.
        start_date: Start of the time-filter window.
        end_date: End of the time-filter window.
        cookie_file: Netscape-format cookie jar for Geofabrik OAuth.
        overwrite_download: Re-download raw PBFs even if present.
        overwrite_filter: Re-run tags-filter and time-filter even if present.
        overwrite_parse: Re-run parse even if Parquets are present.
        chunk_size: Rows per Parquet-writer flush in the streaming parser.
        verbose: Print progress during parsing.

    Returns:
        Tuple ``(output_versions_path, output_changes_path)``.

    Raises:
        ValueError: If ``extracts`` is empty, has duplicate names, or every
            extract fails the 404 check (the final concat needs at least
            one successful intermediate).
    """
    if not extracts:
        raise ValueError("`extracts` must contain at least one HistoryExtract.")
    names = [spec.name for spec in extracts]
    if len(set(names)) != len(names):
        raise ValueError(
            f"`extracts` names must be unique; got duplicates in {names}."
        )

    successful: list[tuple[Path, Path]] = []
    for spec in extracts:
        print(f"Processing {spec.name} history extract...")
        try:
            _download_filter_timefilter_parse(
                pbf_url=spec.url,
                raw_pbf_path=spec.raw_pbf_path,
                filtered_pbf_path=spec.filtered_pbf_path,
                time_filtered_pbf_path=spec.time_filtered_pbf_path,
                versions_path=spec.versions_path,
                changes_path=spec.changes_path,
                filter_keys=filter_keys,
                tag_filter_exprs=tag_filter_exprs,
                start_date=start_date,
                end_date=end_date,
                cookie_file=cookie_file,
                overwrite_download=overwrite_download,
                overwrite_filter=overwrite_filter,
                overwrite_parse=overwrite_parse,
                chunk_size=chunk_size,
                verbose=verbose,
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                print(
                    f"WARNING: {spec.name} history PBF not found on Geofabrik "
                    f"({spec.url} -> HTTP 404). Skipping; this territory's "
                    f"POIs will rate against the global-mean delta only."
                )
                continue
            raise
        successful.append((spec.versions_path, spec.changes_path))

    if not successful:
        raise ValueError(
            "All history extracts failed with HTTP 404. Cannot concat. "
            "Check the URLs in config.yaml or refresh the Geofabrik cookie."
        )

    print(
        f"Concatenating {len(successful)} history extract(s) into "
        f"{output_versions_path} / {output_changes_path}..."
    )
    _concat_history(
        intermediates=successful,
        out_versions_path=output_versions_path,
        out_changes_path=output_changes_path,
    )
    print(
        f"Saved OSM history to {output_versions_path} and {output_changes_path}"
    )
    return output_versions_path, output_changes_path
