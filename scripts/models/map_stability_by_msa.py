"""
Map modeled POI name-stability by metropolitan/micropolitan area (CBSA).

For a fitted ``random_effects`` turnover model, plots the probability that a POI
name remains unchanged after a horizon (default 10 years) for each MSA, as a
choropleth with Alaska / Hawaii / Puerto Rico repositioned in the tidycensus /
tigris style (via ``pygris.shift_geometry``).

Outcome
-------
The mapped value for each MSA is the **fresh-regime survival**

    S(t) = (1 - delta) * exp(-lambda * t)        # t = horizon, default 10 yr

marginalized over the POIs actually observed in that MSA. Concretely, the model
predicts one curve per (shared_label x msa_code x urban_rural) *cell*; for each
MSA we take the observation-count-weighted average of ``1 - p_fresh_mean`` at the
horizon across that MSA's cells:

    S_msa = sum_c [ w_c * (1 - p_fresh_mean_c) ] / sum_c w_c

where ``c`` ranges over the (shared_label x urban_rural) cells in the MSA and
``w_c`` is the number of observations in cell ``c``. This is the same
incorporation of the other random effects used for the per-shared_label stability
overlays (``viz_random_effects.py``): the urban / suburban / rural and
shared_label composition of each MSA is interspersed by prevalence, rather than
zeroing those effects out. (Averaging the survival across cells is the
population-mean probability that a POI remains unchanged.)

Color
-----
ColorBrewer ``RdYlBu`` (full 11-class), red (low stability) -> yellow -> dark blue
(high). The scale is fixed to 20%-60% with both ends extended, so MSAs below 20%
or above 60% clamp to the end colors. The state backdrop (non-metro areas) is
filled with the weighted ``NO_MSA`` value on the same scale; state lines are black.

Usage
-----
    conda activate openpois
    python scripts/models/map_stability_by_msa.py --model-version 2026-06-05-nationwide-full

Output: ``<model_output>/<model-version>/viz/stability_map_by_msa_<H>yr.png``.

Requires the CBSA cartographic-boundary shapefile (``cb_2023_us_cbsa_500k``); uses
the cached copy under the ``census_areas`` directory if present, otherwise fetches
it via pygris. ``pygris.shift_geometry`` downloads a 20m states layer (cached) for
the AK/HI/PR overlay, so first use needs network access.
"""
import argparse
import os
import sys
from pathlib import Path

# Point PROJ at the active env's data dir so reprojection (and GDAL's CRS lookups)
# resolve proj.db on WSL2/conda.
os.environ.setdefault("PROJ_DATA", str(Path(sys.prefix) / "share" / "proj"))
os.environ.setdefault("PROJ_LIB", str(Path(sys.prefix) / "share" / "proj"))

import numpy as np
import pandas as pd
import geopandas as gpd
from config_versioned import Config

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

from pygris.utils import shift_geometry  # noqa: E402


config = Config("~/repos/openpois/config.yaml")

CELL_COLS = ["shared_label", "msa_code", "urban_rural"]
CBSA_FILE = "cb_2023_us_cbsa_500k.shp"

# ColorBrewer RdYlBu, 11-class (low -> high): dark red (low stability) through
# yellow to dark blue (high stability). The colorbar is fixed to [VMIN, VMAX] with
# both ends extended, so values below 20% / above 60% clamp to the end colors.
RDYLBU_11 = [
    "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffbf",
    "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695",
]
STABILITY_CMAP = LinearSegmentedColormap.from_list("RdYlBu_full", RDYLBU_11, N = 256)
VMIN, VMAX = 0.20, 0.60


def msa_stability(model_dir: Path, observations_path: Path, horizon: float) -> pd.DataFrame:
    """Observation-count-weighted fresh survival at ``horizon`` per ``msa_code``."""
    preds = pd.read_csv(
        model_dir / "predictions.csv",
        low_memory = False,
        usecols = ["t2", "p_fresh_mean"] + CELL_COLS,
        dtype = {"msa_code": "str"},
    )
    missing = [c for c in CELL_COLS if c not in preds.columns]
    if missing:
        raise SystemExit(
            f"predictions.csv missing {missing} — not a random_effects fit?"
        )
    preds = preds[np.isclose(preds["t2"], horizon)].copy()
    if preds.empty:
        raise SystemExit(f"No predictions at t2={horizon}.")
    preds["remain"] = 1.0 - preds["p_fresh_mean"]
    preds["msa_code"] = preds["msa_code"].str.replace(r"\.0$", "", regex = True)

    obs = (
        pd.read_parquet(observations_path, columns = CELL_COLS)
        .dropna(subset = CELL_COLS)
    )
    obs["msa_code"] = obs["msa_code"].astype(str).str.replace(r"\.0$", "", regex = True)
    for col in ("shared_label", "urban_rural"):
        obs[col] = obs[col].astype(str)
    weights = obs.groupby(CELL_COLS, observed = True).size().rename("w").reset_index()

    merged = preds.merge(weights, on = CELL_COLS, how = "left")
    merged["w"] = merged["w"].fillna(0.0)

    def _wavg(d: pd.DataFrame) -> float:
        return (
            float(np.average(d["remain"], weights = d["w"]))
            if d["w"].sum() > 0 else float(d["remain"].mean())
        )

    return (
        merged.groupby("msa_code").apply(_wavg, include_groups = False)
        .rename("remain")
        .reset_index()
    )


def load_cbsa(cbsa_shp: Path | None) -> gpd.GeoDataFrame:
    """Load CBSA cartographic boundaries from a cached shapefile or via pygris."""
    if cbsa_shp is not None and cbsa_shp.exists():
        return gpd.read_file(cbsa_shp)[["GEOID", "NAME", "geometry"]]
    from pygris import core_based_statistical_areas
    cbsa = core_based_statistical_areas(cb = True, resolution = "500k", year = 2023)
    return cbsa[["GEOID", "NAME", "geometry"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "MSA choropleth of modeled POI name-stability.",
    )
    parser.add_argument("--model-version", required = True)
    parser.add_argument(
        "--observations", default = None,
        help = "Override the configured osm_data.osm_observations path.",
    )
    parser.add_argument("--horizon", type = float, default = 10.0,
                        help = "Survival horizon in years (default 10).")
    parser.add_argument("--cbsa-shp", default = None,
                        help = "Path to cb_*_us_cbsa_*.shp (else cache / pygris).")
    parser.add_argument("--output", default = None)
    args = parser.parse_args()

    model_dir = config.get_dir_path("model_output").parent / args.model_version
    if not (model_dir / "predictions.csv").exists():
        raise SystemExit(f"predictions.csv not found in {model_dir}")
    obs_path = Path(
        args.observations or config.get_file_path("osm_data", "osm_observations")
    )

    if args.cbsa_shp is not None:
        cbsa_shp = Path(args.cbsa_shp)
    else:
        try:
            cbsa_shp = config.get_dir_path("census_areas") / CBSA_FILE
        except Exception:
            cbsa_shp = Path("~/data/openpois/census_areas").expanduser() / CBSA_FILE

    stab = msa_stability(model_dir, obs_path, args.horizon)
    cbsa = load_cbsa(cbsa_shp)
    cbsa = cbsa.merge(stab, left_on = "GEOID", right_on = "msa_code", how = "left")

    n_val = int(cbsa["remain"].notna().sum())
    print(f"{n_val} / {len(cbsa)} CBSAs have a modeled value")

    # Reposition AK/HI/PR (tidycensus/tigris style). States backdrop uses the
    # identical shift so the layers align.
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip"
    )[["STUSPS", "geometry"]]
    states = states[~states["STUSPS"].isin(["AS", "GU", "MP", "VI"])]
    states_s = shift_geometry(states, preserve_area = False, position = "below")
    cbsa_s = shift_geometry(cbsa, preserve_area = False, position = "below")

    have = cbsa_s[cbsa_s["remain"].notna()]
    # Non-metro default: the weighted survival of the NO_MSA group (POIs outside any
    # CBSA), used as the state-backdrop fill so it reads on the same scale.
    nm = stab.loc[stab["msa_code"] == "NO_MSA", "remain"]
    nonmetro_value = float(nm.iloc[0]) if len(nm) else float(stab["remain"].mean())
    print(f"non-metro (NO_MSA) weighted {args.horizon:.0f}yr survival: {nonmetro_value:.3f}")
    norm = Normalize(vmin = VMIN, vmax = VMAX)

    fig, ax = plt.subplots(1, 1, figsize = (15, 9))
    states_s.plot(
        ax = ax, color = to_hex(STABILITY_CMAP(norm(nonmetro_value))),
        edgecolor = "black", linewidth = 0.5,
    )
    have.plot(
        ax = ax, column = "remain", cmap = STABILITY_CMAP, norm = norm,
        edgecolor = "#444444", linewidth = 0.15,
    )
    # Foreground state lines (dotted, same width as the solid backdrop lines) so
    # state borders stay visible over the MSAs.
    states_s.boundary.plot(ax = ax, color = "black", linewidth = 0.5, linestyle = ":")
    ax.set_axis_off()

    sm = ScalarMappable(cmap = STABILITY_CMAP, norm = norm)
    # Colorbar as an inset anchored flush with the map's bottom edge (zero gap):
    # its top (y0 + height) sits at axes y = 0.
    cax = ax.inset_axes([0.30, -0.02, 0.40, 0.02])
    cbar = fig.colorbar(
        sm, cax = cax, orientation = "horizontal", extend = "both",
        ticks = [0.2, 0.3, 0.4, 0.5, 0.6],
    )
    cbar.ax.set_xticklabels(["<20%", "30%", "40%", "50%", ">60%"])
    cbar.set_label(
        f"Probability a POI name tag is unchanged after {args.horizon:.0f} years"
    )

    out_dir = model_dir / "viz"
    out_dir.mkdir(parents = True, exist_ok = True)
    out_path = Path(args.output) if args.output else (
        out_dir / f"stability_map_by_msa_{args.horizon:.0f}yr.png"
    )
    fig.savefig(out_path, dpi = 200, bbox_inches = "tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
