<template>
  <div class="map-container">
    <div ref="mapEl" style="width: 100%; height: 100%"></div>

    <button
      class="geolocation-btn"
      title="My location"
      @click="handleGeolocate"
    >
      <span class="material-symbols-outlined">my_location</span>
    </button>

    <ConfidenceLegend
      v-if="props.activeSource === 'osm' || props.activeSource === 'overture' || props.activeSource === 'conflated'"
    />

    <!-- Desktop: native select -->
    <div class="basemap-switcher basemap-switcher-desktop">
      <select v-model="selectedStyle" @change="switchBaseMap">
        <option v-for="s in baseMapStyles" :key="s.key" :value="s.key">
          {{ s.label }}
        </option>
      </select>
    </div>

    <!-- Mobile: button + centered modal -->
    <button class="basemap-switcher basemap-mobile-btn" @click="basemapModalOpen = true">
      {{ baseMapStyles.find(s => s.key === selectedStyle)?.label }} ▾
    </button>
    <Teleport to="body">
      <div v-if="basemapModalOpen" class="basemap-modal-overlay" @click.self="basemapModalOpen = false">
        <div class="basemap-modal">
          <div class="basemap-modal-title">Base Map</div>
          <button
            v-for="s in baseMapStyles"
            :key="s.key"
            :class="['basemap-modal-option', { active: selectedStyle === s.key }]"
            @click="selectBasemap(s.key)"
          >
            {{ s.label }}
          </button>
        </div>
      </div>
    </Teleport>

    <!-- Popup overlay anchor (managed by OL Overlay, not Vue v-if) -->
    <div ref="popupEl">
      <PoiPopup :feature="selectedFeature" @close="closePopup" />
    </div>

    <div class="map-attribution">
      &copy;
      <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap<span class="attr-long"> contributors</span></a>,
      <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener noreferrer">OpenMapTiles</a>,
      <a href="https://overturemaps.org/" target="_blank" rel="noopener noreferrer">Overture Maps<span class="attr-long"> Foundation</span></a>,
      <a href="https://openpois.org/about.html" target="_blank" rel="noopener noreferrer">OpenPOIs</a>
    </div>
  </div>
</template>

<script setup>
import {
  ref, shallowRef, watch, onMounted, onBeforeUnmount, nextTick,
} from 'vue'
import Map from 'ol/Map'
import View from 'ol/View'
import Overlay from 'ol/Overlay'
import { fromLonLat, transformExtent } from 'ol/proj'
import { apply } from 'ol-mapbox-style'
import PoiPopup from './PoiPopup.vue'
import ConfidenceLegend from './ConfidenceLegend.vue'
import { useGeolocation } from '../composables/useGeolocation.js'
import { useMapHash, parseMapHash } from '../composables/useMapHash.js'
import {
  getOsmLayer,
  updateOsmFilters,
  wrapOsmFeature,
} from '../layers/osmLayer.js'
import {
  getOvertureLayer,
  updateOvertureFilters,
  wrapOvertureFeature,
} from '../layers/overtureLayer.js'
import {
  getConflatedLayer,
  updateConflatedFilters,
  wrapConflatedFeature,
} from '../layers/conflatedLayer.js'
import {
  BASE_MAP_STYLES,
  INITIAL_CENTER,
  INITIAL_ZOOM,
  MIN_ZOOM,
} from '../constants.js'

const props = defineProps({
  activeSource: { type: String, required: true },
  osmFilters: { type: Object, required: true },
  overtureFilters: { type: Object, required: true },
  conflatedFilters: { type: Object, required: true },
})

const mapEl = ref(null)
const popupEl = ref(null)
const map = shallowRef(null)
const popupOverlay = shallowRef(null)
const selectedFeature = shallowRef(null)
const selectedStyle = ref('positron')
const basemapModalOpen = ref(false)
const baseMapStyles = BASE_MAP_STYLES

const { locate } = useGeolocation()

let geoOverlay = null
let geocodeMarker = null

// Helper: get all data layers
function getDataLayers() {
  return [getOsmLayer(), getOvertureLayer(), getConflatedLayer()]
}

onMounted(async () => {
  const hashState = parseMapHash()
  const view = new View({
    center: hashState ? hashState.center : fromLonLat(INITIAL_CENTER),
    zoom: hashState ? hashState.zoom : INITIAL_ZOOM,
    minZoom: MIN_ZOOM,
  })

  const osmLyr = getOsmLayer()
  const overtureLyr = getOvertureLayer()
  const conflatedLyr = getConflatedLayer()
  osmLyr.setVisible(props.activeSource === 'osm')
  overtureLyr.setVisible(props.activeSource === 'overture')
  conflatedLyr.setVisible(props.activeSource === 'conflated')

  const olMap = new Map({
    target: mapEl.value,
    view,
    layers: [osmLyr, overtureLyr, conflatedLyr],
  })
  map.value = olMap

  document.getElementById('initial-loader')?.remove()

  // Try to apply vector tile style (replaces fallback raster)
  applyBaseStyle('positron')

  // Popup overlay
  await nextTick()
  popupOverlay.value = new Overlay({
    element: popupEl.value,
    positioning: 'bottom-center',
    offset: [0, -8],
  })
  olMap.addOverlay(popupOverlay.value)
  popupEl.value.addEventListener('pointerdown', (e) => e.stopPropagation())

  olMap.on('singleclick', handleClick)

  // Initialise PMTiles filters from props
  updateOsmFilters(props.osmFilters)
  updateOvertureFilters(props.overtureFilters)
  updateConflatedFilters(props.conflatedFilters)

  useMapHash(olMap)
  if (!hashState) handleGeolocate()
})

onBeforeUnmount(() => {
  if (map.value) map.value.setTarget(null)
})

async function applyBaseStyle(styleKey) {
  const style = BASE_MAP_STYLES.find(s => s.key === styleKey)
  if (!style || !map.value) return

  const olMap = map.value
  const dataLayers = getDataLayers()

  // Save view state
  const center = olMap.getView().getCenter()
  const zoom = olMap.getView().getZoom()

  // Remove data layers BEFORE apply() to prevent "duplicate item" error.
  // apply() replaces/manages base layers but leaves others — removing them
  // first lets us cleanly re-add them on top afterward.
  const layerArr = olMap.getLayers().getArray()
  for (const lyr of dataLayers) {
    if (layerArr.includes(lyr)) olMap.removeLayer(lyr)
  }

  try {
    await apply(olMap, style.url)
    // Re-add data layers on top of the new base style
    for (const lyr of dataLayers) {
      olMap.addLayer(lyr)
    }
    olMap.getView().setCenter(center)
    olMap.getView().setZoom(zoom)
  } catch (err) {
    console.error('Failed to apply base style:', err)
    // Ensure data layers are present even on error
    const arr = olMap.getLayers().getArray()
    for (const lyr of dataLayers) {
      if (!arr.includes(lyr)) olMap.addLayer(lyr)
    }
  }
}

function switchBaseMap() {
  applyBaseStyle(selectedStyle.value)
}

function selectBasemap(key) {
  selectedStyle.value = key
  basemapModalOpen.value = false
  applyBaseStyle(key)
}

// ---- Interaction ----

function handleClick(evt) {
  closePopup()

  map.value.forEachFeatureAtPixel(evt.pixel, (feature, lyr) => {
    // All three data layers are now VectorTile/PMTiles. Route through the
    // matching wrapper so PoiPopup gets a plain OL-Feature-ish object.
    let wrapped = null
    if (lyr === getOsmLayer()) {
      wrapped = wrapOsmFeature(feature)
    } else if (lyr === getOvertureLayer()) {
      wrapped = wrapOvertureFeature(feature)
    } else if (lyr === getConflatedLayer()) {
      wrapped = wrapConflatedFeature(feature)
    }
    if (!wrapped) return false

    // VectorTile RenderFeatures expose getType() but only getFlatCoordinates(),
    // not the full-Feature getCoordinates(). Anchoring the popup to the
    // click location sidesteps the API mismatch and matches the long-standing
    // Overture handler behaviour.
    selectedFeature.value = wrapped
    popupOverlay.value.setPosition(evt.coordinate)
    return true
  })
}

function closePopup() {
  selectedFeature.value = null
  if (popupOverlay.value) {
    popupOverlay.value.setPosition(undefined)
  }
}

// ---- Geolocation ----

async function handleGeolocate() {
  try {
    const pos = await locate()
    const coord = fromLonLat([pos.lon, pos.lat])

    map.value.getView().animate({
      center: coord,
      zoom: 15,
      duration: 500,
    })

    if (!geoOverlay) {
      const el = document.createElement('div')
      el.className = 'geolocation-dot'
      geoOverlay = new Overlay({
        element: el,
        positioning: 'center-center',
      })
      map.value.addOverlay(geoOverlay)
    }
    geoOverlay.setPosition(coord)
  } catch (err) {
    console.warn('Geolocation failed:', err)
  }
}

// ---- Public methods ----

function flyToBbox(bbox) {
  if (!map.value) return
  const extent = transformExtent(
    [bbox.west, bbox.south, bbox.east, bbox.north],
    'EPSG:4326',
    'EPSG:3857'
  )
  map.value.getView().fit(extent, {
    duration: 500,
    maxZoom: 17,
    padding: [50, 50, 50, 50],
  })

  if (bbox.lng != null && bbox.lat != null) {
    const coord = fromLonLat([bbox.lng, bbox.lat])
    if (!geocodeMarker) {
      const el = document.createElement('div')
      el.className = 'geocode-marker'
      geocodeMarker = new Overlay({
        element: el,
        positioning: 'center-center',
        stopEvent: false,
      })
      map.value.addOverlay(geocodeMarker)
    }
    geocodeMarker.setPosition(coord)
  }
}

defineExpose({ flyToBbox })

// ---- Watchers ----

watch(() => props.activeSource, (src) => {
  getOsmLayer().setVisible(src === 'osm')
  getOvertureLayer().setVisible(src === 'overture')
  getConflatedLayer().setVisible(src === 'conflated')
  closePopup()
})

watch(
  () => props.osmFilters,
  (filters) => { updateOsmFilters(filters) },
  { deep: true }
)

watch(
  () => props.overtureFilters,
  (filters) => { updateOvertureFilters(filters) },
  { deep: true }
)

watch(
  () => props.conflatedFilters,
  (filters) => { updateConflatedFilters(filters) },
  { deep: true }
)

</script>
