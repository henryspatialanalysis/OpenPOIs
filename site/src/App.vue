<template>
  <div class="top-bar">
    <SourceToggle
      :active-source="activeSource"
      @update:source="setSource"
    />
    <SearchBar @fly-to="handleFlyTo" />
    <div class="top-bar-right">
      <a href="/about.html" class="about-link">About</a>
      <a
        href="https://henryspatialanalysis.com/"
        target="_blank"
        rel="noopener noreferrer"
        class="brand-logo-link"
      >
        <img src="./assets/logo.png" alt="Henry Spatial Analysis" class="brand-logo" />
      </a>
    </div>
  </div>
  <MapContainer
    ref="mapRef"
    :active-source="activeSource"
    :osm-filters="osmFilters"
    :overture-filters="overtureFilters"
    :conflated-filters="conflatedFilters"
  />
  <AmenityFilter
    :active-source="activeSource"
    :osm-filters="osmFilters"
    :overture-filters="overtureFilters"
    :conflated-filters="conflatedFilters"
    :conflated-labels="CONFLATED_LABELS"
    @update:osm-filters="osmFilters = $event"
    @update:overture-filters="overtureFilters = $event"
    @update:conflated-filters="conflatedFilters = $event"
  />
</template>

<script setup>
import { ref } from 'vue'
import SourceToggle from './components/SourceToggle.vue'
import SearchBar from './components/SearchBar.vue'
import MapContainer from './components/MapContainer.vue'
import AmenityFilter from './components/AmenityFilter.vue'
import {
  OSM_FILTER_KEYS,
  CONFLATED_LABELS,
} from './constants.js'

const activeSource = ref('conflated')
const mapRef = ref(null)

const osmFilters = ref(
  OSM_FILTER_KEYS.reduce((acc, f) => ({ ...acc, [f.key]: true }), {})
)
// {filterKey: false} hides an Overture category; empty = everything on.
const overtureFilters = ref({})
const conflatedFilters = ref(
  CONFLATED_LABELS.reduce((acc, lbl) => ({
    ...acc,
    [lbl]: !lbl.startsWith('Other '),
  }), {})
)

function setSource(src) {
  activeSource.value = src
}

function handleFlyTo(bbox) {
  if (mapRef.value) {
    mapRef.value.flyToBbox(bbox)
  }
}
</script>
