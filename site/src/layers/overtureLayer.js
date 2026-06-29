import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { createXYZ } from 'ol/tilegrid'
import { get as getProjection } from 'ol/proj'
import { Style, Circle, Fill, Stroke } from 'ol/style'
import {
  confidenceColor,
  discretizeConf,
  zoomTierFromResolution,
  POI_DOT_BY_ZOOM,
} from '../utils.js'
import { OVERTURE_PMTILES_URL } from '../constants.js'

let layer = null

// Style cache keyed by conf bucket (same discretisation as OSM layer)
const styleCache = {}

// Overture's hosted PMTiles archive contains tiles at z14 ONLY (unlike our
// OSM/conflated archives, which carry a z10–z14 pyramid). OpenLayers does not
// down-sample vector tiles, so without intervention Overture renders nothing
// below z14. We pin the source tile grid to a single z14 level so OL reuses
// the z14 tiles at lower view zooms (under-zoom), and floor rendering at
// OVERTURE_MIN_ZOOM so we don't try to load a metro's worth of z14 tiles when
// zoomed way out. Below the floor the UI shows a "zoom in" hint instead.
export const OVERTURE_MIN_ZOOM = 13

const overtureTileGrid = createXYZ({
  extent: getProjection('EPSG:3857').getExtent(),
  minZoom: 14,
  maxZoom: 14,
  tileSize: 512,
})

export function getOvertureLayer() {
  if (layer) return layer

  layer = new VectorTileLayer({
    source: new PMTilesVectorSource({
      url: OVERTURE_PMTILES_URL,
      tileGrid: overtureTileGrid,
    }),
    style: overtureTileStyle,
    zIndex: 10,
    visible: false,
    // minZoom is exclusive: the layer renders only at view zoom > OVERTURE_MIN_ZOOM.
    minZoom: OVERTURE_MIN_ZOOM,
  })
  return layer
}

/**
 * No-op: Overture PMTiles tiles contain only granular L1/L2 subcategories
 * (e.g. "coffee_shop", "gym") — not the L0 labels (e.g. "food_and_drink")
 * used in the filter UI. Category filtering is therefore not applied here.
 */
export function updateOvertureFilters(_filtersObj) {}

function overtureTileStyle(feature, resolution) {
  const cats = tryParse(feature.get('categories'))
  if (cats?.primary === 'parking') return null

  const conf = feature.get('confidence')
  const bucket = discretizeConf(conf)
  const tier = zoomTierFromResolution(resolution)
  const key = `${bucket}|${tier}`
  if (!styleCache[key]) {
    const color = confidenceColor(conf ?? null)
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
 * Wrap a VectorTile RenderFeature (immutable) in a plain object that
 * exposes the same .get() / .getKeys() / .getGeometry() API as an OL Feature,
 * with Overture's JSON-encoded fields pre-parsed into flat properties.
 */
export function wrapOvertureFeature(rf) {
  const cats = tryParse(rf.get('categories'))
  const brand = tryParse(rf.get('brand'))
  const addrs = tryParse(rf.get('addresses'))
  const addr = Array.isArray(addrs) ? addrs[0] : addrs
  const websites = tryParse(rf.get('websites'))
  const phones = tryParse(rf.get('phones'))

  const props = {
    _source: 'overture',
    name: rf.get('@name') || null,
    id: rf.get('id'),
    confidence: rf.get('confidence'),
    l0: cats?.primary ?? null,
    l1: cats?.alternate?.[0] ?? null,
    brand: brand?.names?.primary ?? null,
    website: Array.isArray(websites) ? websites[0] : null,
    phone: Array.isArray(phones) ? phones[0] : null,
    'addr:street': addr?.freeform ?? null,
    'addr:city': addr?.locality ?? null,
    'addr:state': addr?.region ?? null,
    source_dataset: 'Overture Maps',
  }

  return {
    get: (key) => props[key],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}

function tryParse(str) {
  if (!str) return null
  try { return JSON.parse(str) } catch { return null }
}
