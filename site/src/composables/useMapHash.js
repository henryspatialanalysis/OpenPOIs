import { fromLonLat, toLonLat } from 'ol/proj'

const HASH_RE = /^#(\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?)$/

export function parseMapHash() {
  const m = HASH_RE.exec(window.location.hash)
  if (!m) return null
  return {
    zoom: parseFloat(m[1]),
    center: fromLonLat([parseFloat(m[3]), parseFloat(m[2])]),
  }
}

function writeHash(view) {
  const center = toLonLat(view.getCenter())
  const zoom = Math.round(view.getZoom() * 10) / 10
  const lat = center[1].toFixed(4)
  const lon = center[0].toFixed(4)
  window.history.replaceState(null, '', `#${zoom}/${lat}/${lon}`)
}

export function useMapHash(map) {
  map.on('moveend', () => writeHash(map.getView()))
}
