"""
Build osm_snapshot.pmtiles from the rated OSM snapshot.

Output is a multi-zoom PMTiles archive (z10-z14 by default) keyed by the
config's ``publish.pmtiles`` block. ``drop-densest-as-needed`` silently
drops features at lower zooms to keep each tile under ~500 KB; the site
scales the point radius down at lower zooms to match. OpenLayers
over-zooms z15+ natively as lossless geometric scale-ups of the z14 tile.

Intermediate FlatGeobuf is staged next to the output and deleted on success.
"""
from config_versioned import Config

from openpois.io.pmtiles import build_pmtiles

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = Config("~/repos/openpois/config.yaml")

INPUT_PATH = config.get_file_path("snapshot_osm", "rated_snapshot")
OUTPUT_PATH = config.get_file_path("snapshot_osm", "pmtiles")

LAYER_NAME = config.get("publish", "pmtiles", "osm_layer_name")
PROPERTIES = config.get("publish", "pmtiles", "osm_properties")
MIN_ZOOM = config.get("publish", "pmtiles", "min_zoom")
MAX_ZOOM = config.get("publish", "pmtiles", "max_zoom")
DROP_STRATEGY = config.get("publish", "pmtiles", "drop_strategy")

# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Building OSM PMTiles from {INPUT_PATH}")
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
    )

    print(
        f"Done. Wrote {stats['rows_written']:,} features, "
        f"{stats['pmtiles_bytes'] / 1e9:.2f} GB PMTiles."
    )
