import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { Style, Circle, Fill, Stroke } from 'ol/style'
import {
  confidenceColor,
  discretizeConf,
  zoomTierFromResolution,
  POI_DOT_BY_ZOOM,
} from '../utils.js'
import { OSM_PMTILES_URL } from '../constants.js'

// OSM filter keys that drive feature visibility. A feature is visible when at
// least one *enabled* key has a non-null value on it. If every filter is off,
// nothing renders.
const OSM_KEYS = [
  'amenity', 'shop', 'leisure', 'healthcare',
  'craft', 'historic', 'landuse', 'office', 'tourism',
]

let layer = null
let enabledKeys = new Set(OSM_KEYS)  // default: all on

const styleCache = {}

export function getOsmLayer() {
  if (layer) return layer

  layer = new VectorTileLayer({
    source: new PMTilesVectorSource({ url: OSM_PMTILES_URL }),
    style: osmTileStyle,
    zIndex: 10,
  })
  return layer
}

/**
 * Update the set of enabled filter keys and redraw the layer in place.
 * Called from MapContainer.vue whenever the OSM filter checkboxes change.
 */
export function updateOsmFilters(filtersObj) {
  enabledKeys = new Set(
    Object.entries(filtersObj).filter(([, v]) => v).map(([k]) => k)
  )
  if (layer) layer.changed()
}

function osmTileStyle(feature, resolution) {
  if (enabledKeys.size === 0) return null

  // Visibility: feature must have at least one enabled key set.
  let visible = false
  for (const k of enabledKeys) {
    if (feature.get(k) != null) {
      visible = true
      break
    }
  }
  if (!visible) return null

  const conf = feature.get('conf_mean')
  const bucket = discretizeConf(conf)
  const tier = zoomTierFromResolution(resolution)
  const key = `${bucket}|${tier}`
  if (!styleCache[key]) {
    const color = confidenceColor(conf == null || isNaN(conf) ? null : conf)
    const { radius, stroke } = POI_DOT_BY_ZOOM[tier]
    styleCache[key] = new Style({
      image: new Circle({
        radius,
        fill: new Fill({ color }),
        stroke: new Stroke({ color: '#fff', width: stroke }),
      }),
    })
  }
  return styleCache[key]
}

/**
 * Wrap an immutable VectorTile RenderFeature in a plain object that exposes
 * the OL-Feature-ish API used by PoiPopup.vue: .get(), .getKeys(), .getGeometry().
 */
export function wrapOsmFeature(rf) {
  const props = {
    _source: 'osm',
    osm_id: rf.get('osm_id'),
    osm_type: rf.get('osm_type'),
    name: rf.get('name'),
    conf_mean: rf.get('conf_mean'),
    source_dataset: 'OpenStreetMap',
  }
  // Copy any tag column that was baked into the tile
  for (const k of OSM_KEYS) {
    const v = rf.get(k)
    if (v != null) props[k] = v
  }
  return {
    get: (k) => props[k],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}
