import { reactive } from 'vue'

// Per-source load failures, keyed by the activeSource name ('osm', 'overture',
// 'conflated'). MapContainer shows a "layer unavailable" banner while the
// active source has an entry here.
export const layerErrors = reactive({})

// A tile error after a good header is usually transient (one bad range
// request); only flag the layer once errors keep coming with no successes.
const TILE_ERROR_THRESHOLD = 5

/**
 * Report a PMTiles source as unavailable when its archive cannot be read.
 *
 * ol-pmtiles 2.x never handles a rejected header fetch: the source stays in
 * the 'loading' state forever and the map renders nothing, with only an
 * "Uncaught (in promise)" in the console. pmtiles caches the header promise,
 * so asking for it again here observes the same failure without a second
 * request. Per-tile failures surface as 'tileloaderror' events.
 */
export function watchSourceHealth(source, key) {
  source.pmtiles_.getHeader().catch((err) => {
    console.error(`${key} PMTiles archive unavailable:`, err)
    layerErrors[key] = String(err?.message ?? err)
  })

  let consecutiveErrors = 0
  source.on('tileloadend', () => { consecutiveErrors = 0 })
  source.on('tileloaderror', () => {
    consecutiveErrors += 1
    if (consecutiveErrors >= TILE_ERROR_THRESHOLD && !layerErrors[key]) {
      layerErrors[key] = 'Repeated tile load failures'
    }
  })
}
