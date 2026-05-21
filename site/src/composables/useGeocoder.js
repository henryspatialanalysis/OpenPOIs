import { ref } from 'vue'
import { STADIA_GEOCODING_URL } from '../constants.js'

const results = ref([])
const loading = ref(false)
let debounceTimer = null

async function search(query) {
  if (!query || query.length < 3) {
    results.value = []
    return
  }

  clearTimeout(debounceTimer)
  return new Promise((resolve) => {
    debounceTimer = setTimeout(async () => {
      loading.value = true
      try {
        const params = new URLSearchParams({
          api_key: import.meta.env.VITE_STADIA_API_KEY,
          text: query,
          // Match the dataset's geographic scope: 50 states + DC + the
          // 5 inhabited US territories (PR, VI, GU, MP, AS).
          'boundary.country': 'US,PR,VI,GU,MP,AS',
          size: '5',
        })
        const resp = await fetch(`${STADIA_GEOCODING_URL}?${params}`)
        const data = await resp.json()
        results.value = data.features ?? []
      } catch (err) {
        console.error('Geocoding error:', err)
        results.value = []
      } finally {
        loading.value = false
        resolve()
      }
    }, 300)
  })
}

function clear() {
  results.value = []
}

export function useGeocoder() {
  return { results, loading, search, clear }
}
