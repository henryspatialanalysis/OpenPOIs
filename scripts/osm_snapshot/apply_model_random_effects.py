#!/usr/bin/env python
"""
Apply a fitted ``random_effects`` turnover model to the OSM POI snapshot.

Unlike ``apply_model.py`` (which looks confidence curves up from a per-group
``predictions.csv``), this rates each POI by **reconstructing its cell's curve
from the fitted posterior draws** — so a POI whose exact
``(shared_label, msa_code, urban_rural)`` cell wasn't in the training data still
gets a partial-pooling estimate from whatever amenity / MSA / interaction /
urbanicity levels *were* seen (unseen levels drop out; see
``openpois.models.reconstruct``).

Per POI:
  1. Enrich with ``msa_code`` / ``urban_rural`` from its snapshot geometry
     (the same ``assign_indicators`` used to build the observations).
  2. Assign a ``shared_label`` (single-label, first-match-wins) — the same
     taxonomy the conflation pipeline uses.
  3. Reconstruct the posterior-predictive curve for that cell and read off the
     confidence (1 − P(change)) at the number of years since the last edit
     (rounded to 0.1, capped at 10).

Output columns appended to the snapshot:
  conf_mean, conf_lower, conf_upper  — confidence estimates (conditional regime)
  t2_years                           — years since last OSM edit
  model_version                      — the random_effects model version used
  model_group                        — the assigned cell "label|msa|urban_rural"

Streams row-groups so peak memory stays bounded; curves are reconstructed once
per unique cell and cached.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from config_versioned import Config

from openpois.conflation.taxonomy import (
    assign_osm_shared_label,
    load_match_radii,
    load_osm_crosswalk,
)
from openpois.io import indicators
from openpois.models import reconstruct


config = Config("~/repos/openpois/config.yaml")

FILTER_KEYS = config.get("download", "osm", "filter_keys")
SNAPSHOT_PATH = config.get_file_path("snapshot_osm", "snapshot")
OUTPUT_PATH = config.get_file_path("snapshot_osm", "rated_snapshot")
MODEL_BASE = Path(config.get_dir_path("model_output")).parent
# Default model to rate with: the by_shared_label random_effects fit at the
# configured apply_model.model_stub (overridable via --model-version).
DEFAULT_MODEL_VERSION = (
    f"{config.get('osm_data', 'apply_model', 'model_stub')}_by_shared_label"
)
DELTA_GROUP = config.get(
    "osm_turnover_model", "random_effects", "delta_group", fail_if_none = False
) or "shared_label"

# Census layers for enrichment.
CBSA_SHP = config.get_file_path("census_areas", "cbsa_shapefile")
PLACE_SHP = config.get_file_path("census_areas", "place_shapefile")
POPULATION_CSV = config.get_file_path("census_areas", "place_population")

TIMES = np.arange(101) / 10.0           # 0.0, 0.1, ..., 10.0 years
BATCH_ROWS = 500_000
ROW_GROUP_SIZE = 50_000
_SEP = "\x1f"


class CurveCache:
    """Reconstruct and cache one confidence curve per unique cell."""

    def __init__(self, draws, maps, delta_group_col):
        self.draws = draws
        self.maps = maps
        self.delta_group_col = delta_group_col
        self.key_to_idx: dict[str, int] = {}
        # 1 - P(change) curves (confidence). conf = 1 - p_cond; the interval
        # flips: conf_lower = 1 - p_cond_upper, conf_upper = 1 - p_cond_lower.
        self._conf_mean: list[np.ndarray] = []
        self._conf_lower: list[np.ndarray] = []
        self._conf_upper: list[np.ndarray] = []

    def ensure(self, cells_df: pd.DataFrame) -> None:
        """Reconstruct curves for any cells not already cached."""
        keys = _cell_keys(cells_df)
        new_mask = np.array([k not in self.key_to_idx for k in keys])
        if not new_mask.any():
            return
        new_cells = cells_df.loc[new_mask].drop_duplicates(
            subset = ["shared_label", "msa_code", "urban_rural"]
        )
        new_keys = _cell_keys(new_cells)
        # Filter to genuinely-new unique keys (drop_duplicates may keep a key
        # already added in this same call's dedup).
        keep = []
        seen = set()
        for i, k in enumerate(new_keys):
            if k not in self.key_to_idx and k not in seen:
                keep.append(i)
                seen.add(k)
        new_cells = new_cells.iloc[keep].reset_index(drop = True)
        new_keys = [new_keys[i] for i in keep]

        curves = reconstruct.reconstruct_cell_curves(
            self.draws, self.maps, new_cells, TIMES,
            delta_group_col = self.delta_group_col,
        )
        for j, k in enumerate(new_keys):
            self.key_to_idx[k] = len(self._conf_mean)
            self._conf_mean.append(1.0 - curves["p_cond_mean"][j])
            self._conf_lower.append(1.0 - curves["p_cond_upper"][j])
            self._conf_upper.append(1.0 - curves["p_cond_lower"][j])

    def lookup(self, cells_df, t2_int):
        keys = _cell_keys(cells_df)
        idx = np.array([self.key_to_idx[k] for k in keys])
        cm = np.stack(self._conf_mean)
        cl = np.stack(self._conf_lower)
        cu = np.stack(self._conf_upper)
        return (
            cm[idx, t2_int], cl[idx, t2_int], cu[idx, t2_int],
            [" | ".join(k.split(_SEP)) for k in keys],
        )


def _cell_keys(cells_df) -> list[str]:
    return (
        cells_df["shared_label"].astype(str) + _SEP
        + cells_df["msa_code"].astype(str) + _SEP
        + cells_df["urban_rural"].astype(str)
    ).tolist()


def _t2_int(last_edited) -> np.ndarray:
    """Years since last edit, rounded to 0.1 and capped at 10 → index 0..100."""
    today = pd.Timestamp.now(tz = "UTC")
    le = last_edited
    if le.dt.tz is None:
        le = le.dt.tz_localize("UTC")
    if le.isna().any():
        raise ValueError("Null last_edited timestamps present; impute first.")
    elapsed_years = (today - le).dt.total_seconds().to_numpy() / (365.25 * 86_400)
    t2_years = np.clip(np.round(elapsed_years * 10) / 10, 0.0, 10.0)
    return t2_years, np.round(t2_years * 10).astype(int)


def _check_full_data_fit(model_dir: Path, allow_sampled: bool) -> None:
    """Refuse to rate from a fit trained on a POI subsample.

    ``osm_turnover.py`` snapshots its resolved config into the model
    directory, so the sample fraction the fit actually used is recorded
    there. A sampled fit keeps only a fraction of the ``amenity_msa``
    interaction cells (~18 of ~4,000 at 1%) and thins the per-label and
    per-MSA levels with it, which quietly produces confidence estimates
    far weaker than — and not comparable to — a full-data run. Rating is
    the point where that becomes invisible, so the check lives here.
    """
    fit_config = model_dir / "config.yaml"
    if not fit_config.exists():
        print(
            f"WARNING: {fit_config.name} missing from {model_dir}; cannot "
            "confirm the fit used full data."
        )
        return

    with open(fit_config) as handle:
        fit_cfg = yaml.safe_load(handle) or {}
    fraction = (
        fit_cfg.get("osm_turnover_model", {}).get("poi_sample_fraction")
    )
    if fraction is None or float(fraction) >= 1.0:
        return

    message = (
        f"The fit in {model_dir.name} used poi_sample_fraction="
        f"{fraction} (a {float(fraction) * 100:g}% POI subsample), not full "
        "data. Its confidence estimates would be materially weaker than a "
        "full-data fit and not comparable to previous runs."
    )
    if allow_sampled:
        print(f"WARNING: {message} Proceeding (--allow-sampled-fit).")
        return
    raise SystemExit(
        f"ERROR: {message}\n"
        "Re-fit with poi_sample_fraction 1.0 (the config default), or pass "
        "--allow-sampled-fit if you knowingly want a sampled rating."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Rate the OSM snapshot with a fitted random_effects model."
    )
    parser.add_argument(
        "--model-version", default = None,
        help = (
            "model_output version directory holding the random_effects fit. "
            f"Defaults to {DEFAULT_MODEL_VERSION} "
            "(apply_model.model_stub + '_by_shared_label')."
        ),
    )
    parser.add_argument(
        "--test", action = "store_true",
        help = "Process only the first 10,000 snapshot rows.",
    )
    parser.add_argument(
        "--allow-sampled-fit", action = "store_true",
        help = (
            "Rate from a fit trained on a POI subsample. Off by default: a "
            "sampled fit loses most of its amenity_msa interaction cells and "
            "per-label levels, so its confidence estimates are much weaker "
            "and are not comparable to a full-data run."
        ),
    )
    args = parser.parse_args()
    model_version = args.model_version or DEFAULT_MODEL_VERSION

    # In --test mode, write beside the real output (don't clobber the
    # production rated snapshot) using a "_test" suffix.
    output_path = OUTPUT_PATH
    if args.test:
        output_path = OUTPUT_PATH.with_name(
            f"{OUTPUT_PATH.stem}_test{OUTPUT_PATH.suffix}"
        )

    model_dir = MODEL_BASE / model_version
    # Resolves Parquet (preferred) or legacy CSV; raises if neither is present.
    reconstruct.resolve_param_draws(model_dir)
    _check_full_data_fit(model_dir, allow_sampled = args.allow_sampled_fit)
    print(f"Loading random_effects artifacts from {model_dir} ...")
    draws = reconstruct.load_random_effects_draws(model_dir)
    maps = reconstruct.load_factor_maps(model_dir)
    cache = CurveCache(draws, maps, DELTA_GROUP)

    print("Loading Census layers ...")
    msa_gdf, classified_places = indicators.load_classified_layers(
        CBSA_SHP, PLACE_SHP, POPULATION_CSV
    )
    osm_crosswalk = load_osm_crosswalk()
    match_radii = load_match_radii()

    pf = pq.ParquetFile(SNAPSHOT_PATH)
    n_total = pf.metadata.num_rows
    print(f"Rating {n_total:,} POIs from {SNAPSHOT_PATH} ...")
    input_schema = pf.schema_arrow
    new_fields = [
        pa.field("t2_years", pa.float64()),
        pa.field("conf_mean", pa.float64()),
        pa.field("conf_lower", pa.float64()),
        pa.field("conf_upper", pa.float64()),
        pa.field("model_version", pa.string()),
        pa.field("model_group", pa.string()),
    ]
    output_schema = pa.schema(
        list(input_schema) + new_fields, metadata = input_schema.metadata
    )
    lookup_cols = ["last_edited"] + [
        k for k in FILTER_KEYS if k in set(input_schema.names)
    ]

    output_path.parent.mkdir(parents = True, exist_ok = True)
    n_written = 0
    with pq.ParquetWriter(output_path, output_schema, compression = "zstd") as writer:
        batches = (
            [next(pf.iter_batches(batch_size = 10_000))] if args.test
            else pf.iter_batches(batch_size = BATCH_ROWS)
        )
        for batch in batches:
            tbl = pa.Table.from_batches([batch])
            df = tbl.select(lookup_cols).to_pandas()

            # 1. spatial indicators from geometry
            geom = gpd.GeoSeries.from_wkb(
                tbl.column("geometry").to_pandas()
            ).representative_point()
            pts = gpd.GeoDataFrame(
                df.copy(), geometry = geom.to_numpy(), crs = "EPSG:4326"
            )
            enriched = indicators.assign_indicators(
                pts, msa_gdf, classified_places
            )

            # 2. shared_label
            labels, _ = assign_osm_shared_label(
                df, osm_crosswalk, match_radii, FILTER_KEYS,
            )
            cells = pd.DataFrame({
                "shared_label": np.asarray(labels, dtype = object),
                "msa_code": enriched["msa_code"].to_numpy(),
                "urban_rural": enriched["urban_rural"].to_numpy(),
            })

            # 3. reconstruct (cached) + read off at t2
            t2_years, t2_int = _t2_int(df["last_edited"])
            cache.ensure(cells)
            cm, cl, cu, groups = cache.lookup(cells, t2_int)

            for field in new_fields:
                values = {
                    "t2_years": t2_years, "conf_mean": cm,
                    "conf_lower": cl, "conf_upper": cu,
                    "model_version": np.full(len(df), model_version, object),
                    "model_group": np.asarray(groups, dtype = object),
                }[field.name]
                tbl = tbl.append_column(
                    field.name, pa.array(values, type = field.type)
                )
            writer.write_table(tbl, row_group_size = ROW_GROUP_SIZE)
            n_written += batch.num_rows
            print(f"  {n_written:,}/{n_total:,} rated", flush = True)

    print(f"\nDone. Rated {n_written:,} POIs → {output_path}")
    print(f"Unique cells reconstructed: {len(cache.key_to_idx):,}")
