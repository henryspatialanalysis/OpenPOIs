import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { Style, Circle, Fill, Stroke } from 'ol/style'
import {
  confidenceColor,
  discretizeConf,
  zoomTierFromResolution,
  POI_DOT_BY_ZOOM,
} from '../utils.js'
import { OVERTURE_PMTILES_URL } from '../constants.js'
import { watchSourceHealth } from './sourceHealth.js'

// Our own archive of the monthly Overture snapshot (scripts/overture/
// prepare_pmtiles.py), with the same zoom pyramid as the OSM and conflated
// archives: z10 up, extended past z14 where dense tiles would drop points.
// Overture's hosted archive at tiles.overturemaps.org blocks openpois.org
// referers.

let layer = null
let hiddenKeys = new Set()  // filter keys switched off; empty = all on

const styleCache = {}

export function getOvertureLayer() {
  if (layer) return layer

  const source = new PMTilesVectorSource({ url: OVERTURE_PMTILES_URL })
  watchSourceHealth(source, 'overture')

  layer = new VectorTileLayer({
    source,
    style: overtureTileStyle,
    zIndex: 10,
    visible: false,
  })
  return layer
}

/**
 * Filter key for an Overture feature: its basic_category, which Overture
 * recommends for map filtering, or "other:<L0>" when it has none.
 */
export function overtureFilterKey(basicCategory, l0) {
  return basicCategory ?? `other:${l0}`
}

/**
 * filtersObj is {filterKey: boolean}. Only keys set to false hide features,
 * so a category missing from the filter panel's list still renders.
 */
export function updateOvertureFilters(filtersObj) {
  hiddenKeys = new Set(
    Object.entries(filtersObj).filter(([, v]) => !v).map(([k]) => k)
  )
  if (layer) layer.changed()
}

function overtureTileStyle(feature, resolution) {
  if (hiddenKeys.size > 0) {
    const key = overtureFilterKey(
      feature.get('basic_category'), feature.get('taxonomy_l0')
    )
    if (hiddenKeys.has(key)) return null
  }

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
 * exposes the same .get() / .getKeys() / .getGeometry() API as an OL Feature.
 */
export function wrapOvertureFeature(rf) {
  const props = {
    _source: 'overture',
    name: rf.get('name') || null,
    id: rf.get('id'),
    confidence: rf.get('confidence'),
    basic_category: rf.get('basic_category') ?? null,
    taxonomy_primary: rf.get('taxonomy_primary') ?? null,
    taxonomy_hierarchy: rf.get('taxonomy_hierarchy') ?? null,
    brand: rf.get('brand') ?? null,
    website: rf.get('website') ?? null,
    phone: rf.get('phone') ?? null,
    'addr:street': rf.get('addr_street') ?? null,
    'addr:city': rf.get('addr_city') ?? null,
    'addr:state': rf.get('addr_state') ?? null,
    source_dataset: 'Overture Maps',
  }

  return {
    get: (key) => props[key],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}
