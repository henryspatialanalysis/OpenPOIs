#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root.
#   -------------------------------------------------------------
"""
OSM-internal same-brand deduplication of the merged output.

OSM often maps one branded business as two valid objects: a drugstore's
building (``shop=chemist``) and the pharmacy counter inside it
(``amenity=pharmacy``), or a store's building way next to a node for the
same store. Both get the same ``shared_label``, and the OSM x Overture
matcher pairs at most one of them with the Overture listing, so the
release carries two POIs for one place (4,427 same-brand Pharmacy pairs
within 50 m in 2026-10-22-v1: Walgreens, CVS, Rite Aid).

Where it runs: on the conflated rows, after the merge and before
``assign_stable_ids.py``. It has to run after matching, because the rule
keeps the row that carries the Overture match, and which row that is is
only known once matching has run. Removing an object before matching
could leave the kept building unmatched and the Overture listing as a
third row. It has to run before the stable ids, so the absorbed
element's prior id can follow it to the kept row as an alias edge (see
:func:`openpois.conflation.stable_ids.load_dedup_aliases`).

Rule, per pair of rows (all conditions required):

* both carry an OSM element (``source`` ``osm`` or ``matched``);
* same non-empty ``shared_label`` (optionally limited to ``labels``);
* centroids within ``radius_m``;
* at least one carries a brand, and the brand keys agree: the brand
  (else the name), normalised by :func:`ghost_osm.normalise_name`, with
  generic category tokens ("pharmacy", "drug", "store", ...) dropped,
  must be non-empty on both sides and reach ``token_set_ratio >=
  min_name_score``;
* at least one side is OSM-only (``source == 'osm'``). An Overture-only
  row is never absorbed, and two matched rows are never merged.

Within each connected group, the kept rows are every matched row plus the
best row by (matched first, way/relation before node, higher
``conf_mean``, OSM key). Each OSM-only row with a qualifying pair to a
kept row is absorbed into the nearest such kept row; an OSM-only row
linked only through another absorbed row stays.

The kept row keeps its own id, source and attributes, and gains two things:

* ``absorbed_osm_keys`` (list<string>, ``node/123`` / ``way/456`` /
  ``relation/789``): every OSM element folded into it. The column is written
  on every row by :func:`dedup_osm_brand` and is NULL where nothing was
  absorbed, so downstream id lookups by OSM element can still find the place.
* The building's geometry: when the kept row is an OSM node and a way or
  relation was absorbed into it, the row takes the geometry of the nearest
  such element (the store building rather than the counter node inside it).
  ``match_distance_m`` still describes the OSM x Overture link as matched,
  before the swap. ``bbox`` and ``geohash`` are derived from ``geometry`` at
  publish time (``format_for_upload.py``), so they follow the new geometry.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import BallTree

from openpois.conflation.ghost_osm import normalise_name
from openpois.conflation.match import EARTH_RADIUS_M

#: Tokens that name a category, not a brand. Dropped from brand keys so a
#: bare "Pharmacy" never matches "CVS Pharmacy" through token subsets.
GENERIC_TOKENS = frozenset({
    "pharmacy", "pharmacies", "drug", "drugs", "drugstore", "rx", "store",
    "stores", "shop", "market", "supermarket", "grocery", "the", "and",
})

ABSORBED_COLUMN = "absorbed_osm_keys"
ABSORBED_TYPE = pa.list_(pa.string())

AUDIT_COLUMNS = [
    "loser_osm_key", "loser_osm_type", "loser_osm_id", "loser_name",
    "winner_osm_key", "winner_osm_type", "winner_overture_id", "winner_source",
    "winner_name",
    "shared_label", "brand_key", "distance_m", "name_score",
    "loser_lon", "loser_lat",
]

_FRAME_COLUMNS = [
    "source", "osm_type", "osm_id", "overture_id", "shared_label",
    "osm_brand", "brand", "osm_name", "name", "conf_mean", "lon", "lat",
]


def brand_key(brand, name) -> str:
    """Brand (else name), normalised, with generic category tokens removed."""
    text = normalise_name(brand) or normalise_name(name)
    return " ".join(t for t in text.split() if t not in GENERIC_TOKENS)


def osm_key(osm_type, osm_id) -> str | None:
    """``osm:n123`` / ``osm:w123`` / ``osm:r123``, as stable_ids keys it."""
    if osm_type is None or osm_id is None:
        return None
    try:
        if pd.isna(osm_id) or pd.isna(osm_type):
            return None
    except (TypeError, ValueError):
        pass
    return f"osm:{str(osm_type).lower()[:1]}{int(osm_id)}"


def _text(values) -> np.ndarray:
    return np.array(
        ["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
         for v in values],
        dtype = object,
    )


def _candidate_pairs(lat: np.ndarray, lon: np.ndarray, radius_m: float):
    """(i, j, distance_m) for every pair i < j within ``radius_m``."""
    if len(lat) < 2:
        empty = np.array([], dtype = np.int64)
        return empty, empty, np.array([], dtype = float)
    rad = np.radians(np.column_stack([lat, lon]))
    tree = BallTree(rad, metric = "haversine")
    neigh, dist = tree.query_radius(
        rad, r = radius_m / EARTH_RADIUS_M, return_distance = True,
    )
    ii, jj, dd = [], [], []
    for i, (nb, ds) in enumerate(zip(neigh, dist)):
        keep = nb > i
        ii.append(np.full(int(keep.sum()), i))
        jj.append(nb[keep])
        dd.append(ds[keep] * EARTH_RADIUS_M)
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(dd)


def find_brand_duplicates(
    frame: pd.DataFrame,
    *,
    radius_m: float = 50.0,
    min_name_score: float = 90.0,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Rows of ``frame`` to absorb, one audit row each (see module docstring).

    ``frame`` needs the columns in ``_FRAME_COLUMNS``; its index labels are
    returned as ``loser_index`` / ``winner_index``.
    """
    missing = [c for c in _FRAME_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"frame lacks columns {missing}")
    out_cols = ["loser_index", "winner_index"] + AUDIT_COLUMNS
    source = _text(frame["source"])
    label = _text(frame["shared_label"])
    osm_brand, brand = _text(frame["osm_brand"]), _text(frame["brand"])
    osm_name, name = _text(frame["osm_name"]), _text(frame["name"])
    has_osm = frame["osm_id"].notna().to_numpy() & np.isin(
        source, ["osm", "matched"],
    )
    eligible = has_osm & (label != "")
    if labels:
        eligible &= np.isin(label, list(labels))
    brand_raw = np.where(osm_brand != "", osm_brand, brand)
    keys = np.array(
        [brand_key(b, n or m) if e else ""
         for b, n, m, e in zip(brand_raw, osm_name, name, eligible)],
        dtype = object,
    )
    has_brand = np.array(
        [bool(normalise_name(b)) for b in brand_raw], dtype = bool,
    )
    eligible &= keys != ""
    pos = np.where(eligible)[0]
    if len(pos) < 2:
        return pd.DataFrame(columns = out_cols)

    lat = frame["lat"].to_numpy(dtype = float)
    lon = frame["lon"].to_numpy(dtype = float)
    conf = pd.to_numeric(frame["conf_mean"], errors = "coerce").fillna(0.0).to_numpy()
    otype = _text(frame["osm_type"])
    okeys = np.array(
        [osm_key(t, i) or "" for t, i in zip(frame["osm_type"], frame["osm_id"])],
        dtype = object,
    )

    # Qualifying pairs, label by label (pairs never cross labels).
    pa_, pb_, pd_, ps_ = [], [], [], []
    for lab in np.unique(label[pos]):
        sub = pos[label[pos] == lab]
        i, j, d = _candidate_pairs(lat[sub], lon[sub], radius_m)
        if len(i) == 0:
            continue
        a, b = sub[i], sub[j]
        ok = (has_brand[a] | has_brand[b]) & (
            (source[a] == "osm") | (source[b] == "osm")
        )
        a, b, d = a[ok], b[ok], d[ok]
        scores = np.array(
            [fuzz.token_set_ratio(keys[x], keys[y]) for x, y in zip(a, b)],
            dtype = float,
        )
        ok = scores >= min_name_score
        pa_.append(a[ok])
        pb_.append(b[ok])
        pd_.append(d[ok])
        ps_.append(scores[ok])
    if not pa_ or sum(len(x) for x in pa_) == 0:
        return pd.DataFrame(columns = out_cols)
    a, b = np.concatenate(pa_), np.concatenate(pb_)
    dist, score = np.concatenate(pd_), np.concatenate(ps_)

    # Groups over the pair graph (positions remapped to a dense range).
    nodes, inv = np.unique(np.concatenate([a, b]), return_inverse = True)
    ia, ib = inv[: len(a)], inv[len(a):]
    graph = csr_matrix(
        (np.ones(len(a), dtype = np.int8), (ia, ib)), shape = (len(nodes),) * 2,
    )
    _, comp = connected_components(graph, directed = False)
    edge = {}
    for x, y, d, s in zip(a, b, dist, score):
        edge[(x, y)] = edge[(y, x)] = (d, s)

    rows = []
    for c in np.unique(comp):
        members = nodes[comp == c]
        rank = sorted(
            members,
            key = lambda m: (
                source[m] != "matched", otype[m].lower() == "node",
                -conf[m], okeys[m],
            ),
        )
        keepers = {rank[0]} | {m for m in members if source[m] == "matched"}
        for m in members:
            if m in keepers or source[m] != "osm":
                continue
            linked = [(edge[(m, k)], k) for k in keepers if (m, k) in edge]
            if not linked:
                continue
            (d, s), k = min(linked, key = lambda t: t[0][0])
            rows.append({
                "loser_index": frame.index[m],
                "winner_index": frame.index[k],
                "loser_osm_key": okeys[m],
                "loser_osm_type": otype[m],
                "loser_osm_id": int(frame["osm_id"].iloc[m]),
                "loser_name": osm_name[m] or name[m],
                "winner_osm_key": okeys[k] or None,
                "winner_osm_type": otype[k],
                "winner_overture_id": (
                    None if pd.isna(frame["overture_id"].iloc[k])
                    else str(frame["overture_id"].iloc[k])
                ),
                "winner_source": source[k],
                "winner_name": osm_name[k] or name[k],
                "shared_label": label[m],
                "brand_key": keys[m],
                "distance_m": float(d),
                "name_score": float(s),
                "loser_lon": float(lon[m]),
                "loser_lat": float(lat[m]),
            })
    return pd.DataFrame(rows, columns = out_cols)


# -----------------------------------------------------------------
# Parquet driver
# -----------------------------------------------------------------


def load_candidate_frame(path: Path | str) -> pd.DataFrame:
    """Rows of a conflated parquet that can take part, with centroids.

    Index = the row's position in the file, so the result of
    :func:`find_brand_duplicates` names the rows to drop.
    """
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    frame = con.execute(f"""
        SELECT file_row_number AS pos, source, osm_type, osm_id, overture_id,
               shared_label, osm_brand, brand, osm_name, name, conf_mean,
               ST_X(ST_Centroid(geometry)) AS lon,
               ST_Y(ST_Centroid(geometry)) AS lat
        FROM read_parquet('{Path(path)}', file_row_number = true)
        WHERE osm_id IS NOT NULL AND source IN ('osm', 'matched')
          AND coalesce(shared_label, '') <> ''
          AND coalesce(nullif(osm_brand, ''), nullif(brand, ''),
                       nullif(osm_name, ''), nullif(name, '')) IS NOT NULL
    """).df()
    con.close()
    return frame.set_index("pos")


def element_ref(osm_key: str) -> str:
    """``osm:n123`` -> ``node/123`` (the published ``absorbed_osm_keys`` form)."""
    kind = {"n": "node", "w": "way", "r": "relation"}[osm_key[4]]
    return f"{kind}/{osm_key[5:]}"


def plan_row_updates(losers: pd.DataFrame) -> tuple[dict, dict]:
    """From :func:`find_brand_duplicates` output, the per-kept-row updates.

    Returns ``(absorbed, geometry_from)``: ``absorbed`` maps a kept row's
    index to its sorted ``absorbed_osm_keys`` list; ``geometry_from`` maps a
    kept node row's index to the index of the nearest absorbed way/relation
    whose geometry it takes.
    """
    absorbed: dict = {}
    geometry_from: dict = {}
    if losers.empty:
        return absorbed, geometry_from
    for win, grp in losers.groupby("winner_index", sort = False):
        absorbed[win] = sorted(element_ref(k) for k in grp["loser_osm_key"])
        if str(grp["winner_osm_type"].iloc[0]).lower() != "node":
            continue
        areas = grp[grp["loser_osm_type"].str.lower().isin(["way", "relation"])]
        if len(areas):
            geometry_from[win] = areas.sort_values("distance_m")["loser_index"].iloc[0]
    return absorbed, geometry_from


def _read_geometries(path: Path, positions) -> dict:
    """Raw geometry values (WKB) of the rows at ``positions`` (file order)."""
    want = np.sort(np.asarray(list(positions), dtype = np.int64))
    out: dict = {}
    if len(want) == 0:
        return out
    pf = pq.ParquetFile(str(path))
    offset = 0
    for batch in pf.iter_batches(batch_size = 2_000_000, columns = ["geometry"]):
        n = batch.num_rows
        lo, hi = np.searchsorted(want, [offset, offset + n])
        if hi > lo:
            col = batch.column(0)
            for pos in want[lo:hi]:
                out[int(pos)] = col[int(pos - offset)].as_py()
        offset += n
    return out


def rewrite_conflated(
    input_path: Path | str,
    output_path: Path | str,
    drop = (),
    absorbed: dict | None = None,
    geometry_from: dict | None = None,
    chunk_rows: int = 2_000_000,
) -> int:
    """Copy ``input_path`` applying the dedup, in file order.

    Drops the rows at ``drop``; writes ``absorbed_osm_keys`` (list<string>,
    NULL where nothing was absorbed) on every row, appending the column when
    the input lacks it; and gives each row in ``geometry_from`` the geometry
    of the row it maps to. Keys of ``absorbed`` / ``geometry_from`` and the
    values of ``geometry_from`` are file row positions. Row order, the other
    columns and the GeoParquet metadata are preserved. In place when the two
    paths are equal (written beside it, then swapped in).
    """
    input_path, output_path = Path(input_path), Path(output_path)
    absorbed = absorbed or {}
    geometry_from = geometry_from or {}
    drop_arr = np.sort(np.asarray(list(drop), dtype = np.int64))
    source_geoms = _read_geometries(input_path, geometry_from.values())
    new_geoms = {int(k): source_geoms[int(v)] for k, v in geometry_from.items()}
    in_place = output_path.resolve() == input_path.resolve()
    write_path = (
        output_path.with_name(output_path.name + ".tmp") if in_place
        else output_path
    )
    pf = pq.ParquetFile(str(input_path))
    base = pf.schema_arrow
    if ABSORBED_COLUMN in base.names:
        out_schema = base.set(
            base.get_field_index(ABSORBED_COLUMN),
            pa.field(ABSORBED_COLUMN, ABSORBED_TYPE),
        )
    else:
        out_schema = pa.schema(
            list(base) + [pa.field(ABSORBED_COLUMN, ABSORBED_TYPE)],
            metadata = base.metadata,
        )
    geom_idx = base.get_field_index("geometry")
    n_out, offset = 0, 0
    with pq.ParquetWriter(str(write_path), out_schema, compression = "zstd") as writer:
        for batch in pf.iter_batches(batch_size = chunk_rows):
            n = batch.num_rows
            table = pa.Table.from_batches([batch], schema = batch.schema)
            positions = range(offset, offset + n)
            values = [absorbed.get(p) for p in positions]
            column = pa.array(values, type = ABSORBED_TYPE)
            if ABSORBED_COLUMN in table.schema.names:
                table = table.set_column(
                    table.schema.get_field_index(ABSORBED_COLUMN),
                    ABSORBED_COLUMN, column,
                )
            else:
                table = table.append_column(
                    pa.field(ABSORBED_COLUMN, ABSORBED_TYPE), column,
                )
            hits = [p for p in positions if p in new_geoms]
            if hits:
                geoms = table.column(geom_idx).to_pylist()
                for p in hits:
                    geoms[p - offset] = new_geoms[p]
                table = table.set_column(
                    geom_idx, base.field(geom_idx),
                    pa.array(geoms, type = base.field(geom_idx).type),
                )
            lo, hi = np.searchsorted(drop_arr, [offset, offset + n])
            if hi > lo:
                keep = np.ones(n, dtype = bool)
                keep[drop_arr[lo:hi] - offset] = False
                table = table.filter(pa.array(keep))
            table = table.cast(out_schema)
            writer.write_table(table)
            n_out += table.num_rows
            offset += n
            del table, batch
            gc.collect()
    if in_place:
        os.replace(write_path, output_path)
    return n_out


def drop_rows(input_path, output_path, positions, chunk_rows: int = 2_000_000) -> int:
    """:func:`rewrite_conflated` with drops only (``absorbed_osm_keys`` all NULL)."""
    return rewrite_conflated(input_path, output_path, drop = positions,
                             chunk_rows = chunk_rows)


def dedup_osm_brand(
    conflated_path: Path | str,
    audit_path: Path | str,
    *,
    radius_m: float = 50.0,
    min_name_score: float = 90.0,
    labels: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """Absorb same-brand OSM duplicates in ``conflated_path`` (in place).

    Always rewrites the file, so ``absorbed_osm_keys`` is present on every
    release the stage runs on. Writes one audit row per absorbed element to
    ``audit_path`` (removed when nothing is absorbed, so a stale audit never
    feeds the id aliases) and returns a summary dict.
    """
    audit_path = Path(audit_path)
    frame = load_candidate_frame(conflated_path)
    losers = find_brand_duplicates(
        frame, radius_m = radius_m, min_name_score = min_name_score,
        labels = labels,
    )
    del frame
    gc.collect()
    absorbed, geometry_from = plan_row_updates(losers)
    summary = {
        "n_absorbed": int(len(losers)),
        "n_kept_rows": int(len(absorbed)),
        "n_geometry_swaps": int(len(geometry_from)),
        "by_label": losers["shared_label"].value_counts().to_dict(),
    }
    if len(losers):
        audit_path.parent.mkdir(parents = True, exist_ok = True)
        losers[AUDIT_COLUMNS].to_parquet(
            audit_path, compression = "zstd", index = False,
        )
    elif audit_path.exists():
        audit_path.unlink()
    summary["n_rows_out"] = int(rewrite_conflated(
        conflated_path, conflated_path,
        drop = losers["loser_index"].tolist(),
        absorbed = absorbed, geometry_from = geometry_from,
    ))
    if verbose:
        top = ", ".join(
            f"{k} {v:,}" for k, v in list(summary["by_label"].items())[:6]
        )
        print(
            f"  OSM same-brand dedup: absorbed {len(losers):,} rows into "
            f"{len(absorbed):,} kept rows ({top}); {len(geometry_from):,} "
            f"kept nodes took a building's geometry; audit -> {audit_path}"
        )
    return summary
