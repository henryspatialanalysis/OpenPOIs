"""
Build conflated.pmtiles from the conflated POI dataset.

Output is a multi-zoom PMTiles archive keyed by the config's
``publish.pmtiles`` block: z10 up to z14, extended past z14 where the densest
tiles still drop points (``extend_zooms_if_still_dropping``), so every POI is
present at the archive's top zoom. ``drop-densest-as-needed`` thins lower
zooms to keep each tile under ~500 KB; the site scales the point radius down
at lower zooms to match, and over-zooms past the top zoom.

Intermediate FlatGeobuf is staged next to the output and deleted on success.
"""
from config_versioned import Config

from openpois.io.pmtiles import build_pmtiles

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("conflation", "conflated")
OUTPUT_PATH = config.get_file_path("conflation", "pmtiles")

LAYER_NAME = config.get("publish", "pmtiles", "conflated_layer_name")
PROPERTIES = config.get("publish", "pmtiles", "conflated_properties")
MIN_ZOOM = config.get("publish", "pmtiles", "min_zoom")
MAX_ZOOM = config.get("publish", "pmtiles", "max_zoom")
DROP_STRATEGY = config.get("publish", "pmtiles", "drop_strategy")
EXTEND_ZOOMS = config.get("publish", "pmtiles", "extend_zooms_if_still_dropping")

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Building conflated PMTiles from {INPUT_PATH}")
    print(f"  layer: {LAYER_NAME}")
    print(f"  zooms: Z{MIN_ZOOM}-z{MAX_ZOOM}")
    print(f"  drop:  --{DROP_STRATEGY}")
    print(f"  props: {', '.join(PROPERTIES)}")
    print(f"  -> {OUTPUT_PATH}")

    stats = build_pmtiles(
        input_parquet = INPUT_PATH,
        output_pmtiles = OUTPUT_PATH,
        layer_name = LAYER_NAME,
        properties = PROPERTIES,
        min_zoom = MIN_ZOOM,
        max_zoom = MAX_ZOOM,
        drop_strategy = DROP_STRATEGY,
        extend_zooms_if_still_dropping = EXTEND_ZOOMS,
    )

    print(
        f"Done. Wrote {stats['rows_written']:,} features, "
        f"{stats['pmtiles_bytes'] / 1e9:.2f} GB PMTiles."
    )
