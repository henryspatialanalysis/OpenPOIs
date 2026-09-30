"""Apply fitted existence-confidence calibration curves to conflated POIs.

Deploy side of the v4 calibration (see
``~/data/library/writeups/2026-07-30-openpois-confidence-calibration-v4.md``
and :mod:`openpois.conflation.calibration_fit`). Every production POI's raw
source score is mapped through its detection segment's monotone curve --
arithmetic, with no per-POI verification cost.

**Ordering.** This step runs *after* change detection. The change-detection
penalty multiplies ``conf_mean`` by a per-label delta; calibrating first would
leave a calibrated probability multiplied by ~0.14, which is not a probability
of anything. The curves were themselves fit on the post-CD frame.

Per-segment curve index:

===============  ==========================================================
``matched``      a fitted combination of ``osm_conf_mean`` and
                 ``overture_confidence`` chosen by the curve metadata's
                 ``index_mode``: the monotone bilinear ``interaction``
                 index (production since October 2026), the constrained
                 log-odds pool (curves fit before then), the average, an
                 additive index (parameters under ``index``), or -- in
                 ``surface`` mode -- no index at all but a 2-D cell lookup.
                 No fixed 0.7 downweight and no 0.588/0.412 blend.
``osm``          ``osm_conf_mean`` (the OSM turnover posterior mean)
``overture``     ``overture_confidence`` (the provider score; never missing,
                 which the ingest enforces)
===============  ==========================================================

Edge rules, each recorded in ``calibration_flag``:

``shadow_cd``
    Shadow-matched rows keep the change-detection value untouched. The curve is
    indexed on the un-penalized Overture score, so applying it would silently
    undo the demotion; CD is a separate evidence channel with its own
    validation.
``unnamed_extrapolated``
    Unnamed POIs, excluded from the validation frame because the verification
    instrument needs a name to search on. Calibrated through the osm curve as
    a documented extrapolation.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openpois.conflation.calibration_fit import (POOLED_SEGMENTS,
                                                 apply_step_lookup,
                                                 apply_surface, average_score,
                                                 index_score)

SEGMENTS = ("matched", "osm", "overture")

FLAG_SHADOW = "shadow_cd"
FLAG_UNNAMED = "unnamed_extrapolated"

_NEW_FIELD_SPECS = [
    ("conf_mean_uncalibrated", pa.float64()),
    ("calibration_flag", pa.string()),
]


def read_curves(curves_dir) -> dict:
    """Load ``{segment: lookup}`` from a fitted-curve directory."""
    curves_dir = Path(curves_dir)
    curves = {}
    for segment in SEGMENTS:
        path = curves_dir / f"{segment}_curve.parquet"
        if path.exists():
            curves[segment] = pd.read_parquet(path)
    if not curves:
        raise FileNotFoundError(f"No segment curves found in {curves_dir}")
    return curves


def read_curve_metadata(curves_dir) -> dict:
    """Load ``{segment: metadata}`` for the fitted curves that exist."""
    curves_dir = Path(curves_dir)
    out = {}
    for segment in SEGMENTS:
        path = curves_dir / f"{segment}_metadata.json"
        if path.exists():
            with open(path, encoding = "utf-8") as handle:
                out[segment] = json.load(handle)
    return out


def apply_curve(scores, lookup: pd.DataFrame) -> pd.DataFrame:
    """Step-function lookup of the calibrated triple for raw scores.

    Ported from ``openpois_validator.calibrate.artifacts.apply_curve`` so the
    consumer does not depend on the private package. Scores below the first
    bin clamp to it, and NaN scores yield NaN. The implementation lives in
    :func:`calibration_fit.apply_step_lookup` so fit-time cross-validation
    scores exactly this map.
    """
    return apply_step_lookup(scores, lookup)


def is_surface_lookup(lookup: pd.DataFrame) -> bool:
    """Whether a curve table is a 2-D ``surface`` lookup."""
    return "osm_lo" in lookup.columns and "ov_lo" in lookup.columns


def curve_index(source: np.ndarray, osm_conf_mean: np.ndarray,
                overture_confidence: np.ndarray,
                pool_params: dict = None,
                index_mode: str = "pool") -> np.ndarray:
    """Per-segment curve index score.

    Matched rows combine both source scores: their unweighted mean
    (``index_mode = "average"``) or any fitted index form through the shared
    :func:`calibration_fit.index_score` (``pool_params`` carries the params;
    a ``form`` key selects pool / additive / interaction, and pre-2026-09
    curves without one are pools). Single-source segments use their native
    score. Both arguments come from the matched curve's metadata, so the
    deploy step cannot drift from how the curve was fit. ``surface`` mode has
    no index; its matched rows are NaN here and are scored by
    :func:`calibration_fit.apply_surface` instead.
    """
    scores = np.full(len(source), np.nan, dtype = float)
    matched = source == "matched"
    if matched.any() and index_mode != "surface":
        if index_mode == "average":
            scores[matched] = average_score(
                osm_conf_mean[matched], overture_confidence[matched]
            )
        elif pool_params is None:
            raise ValueError(
                "Matched rows need index parameters from the matched curve "
                "metadata (key 'index' or 'pool'), or index_mode = 'average'"
            )
        else:
            scores[matched] = index_score(
                osm_conf_mean[matched], overture_confidence[matched],
                pool_params,
            )
    is_osm = source == "osm"
    scores[is_osm] = osm_conf_mean[is_osm]
    is_overture = source == "overture"
    scores[is_overture] = overture_confidence[is_overture]
    return scores


def calibration_flags(source: np.ndarray, shadow_matched: np.ndarray = None,
                      name: np.ndarray = None) -> np.ndarray:
    """Per-row edge-rule flag (empty string where the plain curve applies)."""
    flags = np.full(len(source), "", dtype = object)
    if name is not None:
        unnamed = pd.isna(name) | (pd.Series(name).astype(str).str.len() == 0)
        flags[unnamed.to_numpy() & (source == "osm")] = FLAG_UNNAMED
    if shadow_matched is not None:
        flags[np.asarray(shadow_matched, dtype = bool)] = FLAG_SHADOW
    return flags


def pool_params_from_metadata(metadata: dict) -> dict:
    """Index parameters for the pooled segments, keyed by segment.

    Reads ``index`` (any form, 2026-09 on) and falls back to ``pool`` for
    curves written before the index was generalized.
    """
    out = {}
    for segment in POOLED_SEGMENTS:
        entry = metadata.get(segment) or {}
        out[segment] = entry.get("index") or entry.get("pool")
    return out


def score_decimals_from_metadata(metadata: dict) -> dict:
    """Rounding each segment's curve was fit with (``None`` = unrounded)."""
    return {
        segment: (metadata.get(segment) or {}).get("score_decimals")
        for segment in SEGMENTS
    }


def index_modes_from_metadata(metadata: dict) -> dict:
    """How each pooled segment's curve is indexed, keyed by segment.

    Defaults to ``"pool"`` for curves written before the mode was recorded.
    """
    return {
        segment: (metadata.get(segment) or {}).get("index_mode") or "pool"
        for segment in POOLED_SEGMENTS
    }


def calibrate_frame(frame: pd.DataFrame, curves: dict,
                    pool_params: dict = None,
                    index_modes: dict = None,
                    score_decimals: dict = None) -> pd.DataFrame:
    """Calibrated triple + flag for one in-memory batch of conflated rows.

    Returns a frame with ``conf_mean``, ``conf_lower``, ``conf_upper``,
    ``conf_mean_uncalibrated`` and ``calibration_flag``, aligned to ``frame``.
    Shadow-matched rows keep their incoming values and a NaN interval.

    ``score_decimals`` (from the curve metadata) rounds each segment's source
    scores exactly as the fit did; curves fit before rounding existed carry no
    value and are applied to unrounded scores, so their bin edges keep
    meaning what they meant when they were fit.
    """
    source = frame["source"].to_numpy()
    osm_conf = pd.to_numeric(frame["osm_conf_mean"], errors = "coerce"
                             ).to_numpy(dtype = float)
    ov_conf = pd.to_numeric(frame["overture_confidence"], errors = "coerce"
                            ).to_numpy(dtype = float)
    for segment, decimals in (score_decimals or {}).items():
        if decimals is None:
            continue
        in_segment = source == segment
        osm_conf = np.where(in_segment, np.round(osm_conf, int(decimals)),
                            osm_conf)
        ov_conf = np.where(in_segment, np.round(ov_conf, int(decimals)),
                           ov_conf)
    incoming = pd.to_numeric(frame["conf_mean"], errors = "coerce"
                             ).to_numpy(dtype = float)
    shadow = (
        frame["shadow_matched"].to_numpy(dtype = bool)
        if "shadow_matched" in frame.columns else None
    )
    names = frame["name"].to_numpy() if "name" in frame.columns else None

    scores = curve_index(
        source, osm_conf, ov_conf,
        pool_params = (pool_params or {}).get("matched"),
        index_mode = (index_modes or {}).get("matched", "pool"),
    )
    flags = calibration_flags(source, shadow_matched = shadow, name = names)

    conf_mean = np.full(len(frame), np.nan, dtype = float)
    conf_lower = np.full(len(frame), np.nan, dtype = float)
    conf_upper = np.full(len(frame), np.nan, dtype = float)

    for segment, lookup in curves.items():
        # Unnamed OSM POIs ride the osm curve; every other row uses its own
        # segment's curve. Shadow rows are excluded and handled below.
        in_segment = source == segment
        if shadow is not None:
            in_segment = in_segment & ~shadow
        if not in_segment.any():
            continue
        if is_surface_lookup(lookup):
            triple = apply_surface(osm_conf[in_segment], ov_conf[in_segment],
                                   lookup)
        else:
            triple = apply_curve(scores[in_segment], lookup)
        conf_mean[in_segment] = triple["conf_mean"].to_numpy()
        conf_lower[in_segment] = triple["conf_lower"].to_numpy()
        conf_upper[in_segment] = triple["conf_upper"].to_numpy()

    # Shadow-matched rows: keep the change-detection value and its NaN band.
    if shadow is not None and shadow.any():
        conf_mean[shadow] = incoming[shadow]
        conf_lower[shadow] = np.nan
        conf_upper[shadow] = np.nan

    # Any row a curve could not score (missing raw score, absent segment
    # curve) keeps its incoming confidence rather than going null.
    unscored = ~np.isfinite(conf_mean)
    conf_mean[unscored] = incoming[unscored]

    return pd.DataFrame(
        {
            "conf_mean": conf_mean,
            "conf_lower": conf_lower,
            "conf_upper": conf_upper,
            "conf_mean_uncalibrated": incoming,
            "calibration_flag": pd.Series(flags, dtype = object).replace(
                "", None
            ),
        },
        index = frame.index,
    )


def apply_calibration(input_path: Path, output_path: Path, curves: dict,
                      pool_params: dict = None, index_modes: dict = None,
                      score_decimals: dict = None,
                      chunk_rows: int = 2_000_000,
                      verbose: bool = True) -> dict:
    """Stream ``input_path`` to ``output_path``, calibrating confidence.

    Overwrites ``conf_mean`` / ``conf_lower`` / ``conf_upper`` in place (so the
    PMTiles allowlist, the site, and the published schema need no changes) and
    appends ``conf_mean_uncalibrated`` + ``calibration_flag``. GeoParquet
    metadata is preserved. Peak memory is one row-group batch, following
    ``change_detection._write_cd_output``.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    pf = pq.ParquetFile(str(input_path))
    base_schema = pf.schema_arrow
    out_schema = pa.schema(
        list(base_schema)
        + [pa.field(name, typ) for name, typ in _NEW_FIELD_SPECS],
        metadata = base_schema.metadata,
    )

    needed = ["source", "osm_conf_mean", "overture_confidence", "conf_mean"]
    optional = [c for c in ("shadow_matched", "name")
                if c in base_schema.names]
    stats = {
        "rows": 0,
        "flag_counts": {},
        "mean_before": 0.0,
        "mean_after": 0.0,
        "by_segment": {},
    }

    with pq.ParquetWriter(str(output_path), out_schema,
                          compression = "zstd") as writer:
        for batch in pf.iter_batches(batch_size = chunk_rows):
            table = pa.Table.from_batches([batch], schema = batch.schema)
            frame = table.select(needed + optional).to_pandas()
            calibrated = calibrate_frame(frame, curves,
                                         pool_params = pool_params,
                                         index_modes = index_modes,
                                         score_decimals = score_decimals)

            for column in ("conf_mean", "conf_lower", "conf_upper"):
                idx = table.schema.get_field_index(column)
                table = table.set_column(
                    idx, column,
                    pa.array(calibrated[column].to_numpy(dtype = float),
                             type = pa.float64()),
                )
            table = table.append_column(
                "conf_mean_uncalibrated",
                pa.array(
                    calibrated["conf_mean_uncalibrated"].to_numpy(
                        dtype = float
                    ),
                    type = pa.float64(),
                ),
            )
            table = table.append_column(
                "calibration_flag",
                pa.array(calibrated["calibration_flag"].tolist(),
                         type = pa.string()),
            )
            table = table.cast(out_schema)
            writer.write_table(table)

            stats["rows"] += len(frame)
            stats["mean_before"] += float(
                np.nansum(calibrated["conf_mean_uncalibrated"])
            )
            stats["mean_after"] += float(np.nansum(calibrated["conf_mean"]))
            for flag, count in (
                calibrated["calibration_flag"].value_counts().items()
            ):
                stats["flag_counts"][flag] = (
                    stats["flag_counts"].get(flag, 0) + int(count)
                )
            for segment in SEGMENTS:
                mask = (frame["source"] == segment).to_numpy()
                if not mask.any():
                    continue
                entry = stats["by_segment"].setdefault(
                    segment, {"n": 0, "sum_before": 0.0, "sum_after": 0.0}
                )
                entry["n"] += int(mask.sum())
                entry["sum_before"] += float(
                    np.nansum(calibrated["conf_mean_uncalibrated"][mask])
                )
                entry["sum_after"] += float(
                    np.nansum(calibrated["conf_mean"][mask])
                )
            del table, batch, frame, calibrated
            gc.collect()

    if stats["rows"]:
        stats["mean_before"] /= stats["rows"]
        stats["mean_after"] /= stats["rows"]
        for entry in stats["by_segment"].values():
            entry["mean_before"] = entry["sum_before"] / entry["n"]
            entry["mean_after"] = entry["sum_after"] / entry["n"]
    if verbose:
        print(f"  Wrote {stats['rows']:,} rows to {output_path}")
        print(f"  Mean conf_mean: {stats['mean_before']:.4f} -> "
              f"{stats['mean_after']:.4f}")
        for segment, entry in sorted(stats["by_segment"].items()):
            print(f"    {segment}: {entry['n']:,} rows, "
                  f"{entry['mean_before']:.4f} -> {entry['mean_after']:.4f}")
        for flag, count in sorted(stats["flag_counts"].items()):
            print(f"    flag {flag}: {count:,}")
    return stats
