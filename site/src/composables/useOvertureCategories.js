import { shallowRef, ref } from 'vue'
import { OVERTURE_CATEGORIES, OVERTURE_CATEGORIES_URL } from '../constants.js'
import { overtureFilterKey } from '../layers/overtureLayer.js'

// Filter-panel groups for the Overture layer, loaded once from the
// overture_categories.json published beside the Overture PMTiles:
//   [{ l0, label, count, categories: [{ key, label, count }] }]
// Each category key matches overtureFilterKey() on a tile feature.
const groups = shallowRef([])
const error = ref(null)
let loading = null

const L0_ORDER = OVERTURE_CATEGORIES.map(c => c.key)
const L0_LABELS = Object.fromEntries(OVERTURE_CATEGORIES.map(c => [c.key, c.label]))

function humanize(key) {
  const s = key.replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function toGroups(json) {
  const rank = (l0) => {
    const i = L0_ORDER.indexOf(l0)
    return i === -1 ? L0_ORDER.length : i
  }
  return json.groups
    .map(g => ({
      l0: g.l0,
      label: L0_LABELS[g.l0] ?? humanize(g.l0),
      count: g.count,
      categories: g.categories.map(c => ({
        key: overtureFilterKey(c.key, g.l0),
        label: c.key ? humanize(c.key) : 'Other',
        count: c.count,
      })),
    }))
    .sort((a, b) => rank(a.l0) - rank(b.l0))
}

export function useOvertureCategories() {
  if (!loading) {
    loading = fetch(OVERTURE_CATEGORIES_URL)
      .then((resp) => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        return resp.json()
      })
      .then((json) => { groups.value = toGroups(json) })
      .catch((err) => {
        console.error('Overture categories unavailable:', err)
        error.value = String(err?.message ?? err)
      })
  }
  return { groups, error }
}
