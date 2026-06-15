/**
 * Discretize confidence to N steps for style cache efficiency.
 */
export function discretizeConf(conf) {
  if (conf == null || isNaN(conf)) return 'null'
  return Math.round(Math.max(0, Math.min(1, conf)) * 20)
}

// Confidence gradient stops: Turbo (reversed so low=red, high=purple),
// resampled so yellow is centered at t=0.5. The warm red->yellow side is
// stretched and the green-blue-purple side compressed. The Turbo range is
// trimmed to its 5%-95% span so the endpoints aren't too dark.
// Source: https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
const CONF_STOPS = [
  { t: 0.0, r: 161, g:  18, b:   1 },  // #a11201 dark red
  { t: 0.1, r: 208, g:  47, b:   5 },  // #d02f05 red
  { t: 0.2, r: 237, g:  85, b:  16 },  // #ed5510 orange-red
  { t: 0.3, r: 253, g: 138, b:  38 },  // #fd8a26 orange
  { t: 0.4, r: 250, g: 186, b:  57 },  // #faba39 yellow-orange
  { t: 0.5, r: 223, g: 223, b:  55 },  // #dfdf37 yellow
  { t: 0.6, r: 153, g: 254, b:  66 },  // #99fe42 yellow-green
  { t: 0.7, r:  50, g: 242, b: 152 },  // #32f298 green
  { t: 0.8, r:  34, g: 197, b: 226 },  // #22c5e2 teal/cyan
  { t: 0.9, r:  70, g: 128, b: 246 },  // #4680f6 blue
  { t: 1.0, r:  61, g:  53, b: 139 },  // #3d358b dark purple
]

function lerpChannel(a, b, t) {
  return Math.round(a + (b - a) * t)
}

/**
 * Map a confidence value [0,1] to a hex color via red→yellow→green gradient.
 */
export function confidenceColor(value) {
  if (value == null || isNaN(value)) return '#999999'
  const v = Math.max(0, Math.min(1, value))

  // Find the two surrounding stops
  let lo = CONF_STOPS[0]
  let hi = CONF_STOPS[CONF_STOPS.length - 1]
  for (let i = 0; i < CONF_STOPS.length - 1; i++) {
    if (v <= CONF_STOPS[i + 1].t) {
      lo = CONF_STOPS[i]
      hi = CONF_STOPS[i + 1]
      break
    }
  }

  const span = hi.t - lo.t
  const t = span === 0 ? 0 : (v - lo.t) / span
  const r = lerpChannel(lo.r, hi.r, t)
  const g = lerpChannel(lo.g, hi.g, t)
  const b = lerpChannel(lo.b, hi.b, t)
  return `rgb(${r},${g},${b})`
}

// Spherical-mercator resolution → integer zoom tier in [10, 14].
// 40075016.6855784 / 256 ≈ 156543.03 m/px at z0; resolution halves per zoom.
// Clamps to the tile range that prepare_pmtiles emits, so z15+ over-zoom and
// any sub-z10 view (shouldn't happen — View.minZoom is 10) both fall through.
export function zoomTierFromResolution(resolution) {
  const z = Math.log2(156543.03392804097 / resolution)
  if (z >= 13.5) return 14
  if (z >= 12.5) return 13
  if (z >= 11.5) return 12
  if (z >= 10.5) return 11
  return 10
}

// Radius / stroke width by zoom tier. Tuned so dense urban areas stop smearing
// at low zoom while keeping z14 visually identical to the prior fixed-5 dots.
export const POI_DOT_BY_ZOOM = {
  14: { radius: 5,   stroke: 1 },
  13: { radius: 4,   stroke: 1 },
  12: { radius: 3,   stroke: 0.75 },
  11: { radius: 2.5, stroke: 0.5 },
  10: { radius: 2,   stroke: 0.5 },
}
