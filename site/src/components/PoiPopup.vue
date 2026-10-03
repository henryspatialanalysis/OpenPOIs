<template>
  <div v-if="feature" class="poi-popup">
    <button class="close-btn" @click="$emit('close')">&times;</button>

    <template v-for="(entry, idx) in entries" :key="idx">
      <hr v-if="idx > 0" class="entry-divider" />

      <h3>{{ entry.name || 'Unnamed' }}</h3>

      <!-- OSM details -->
      <template v-if="source === 'osm'">
        <div v-if="entry.amenity" class="detail-row">
          <span class="detail-label">Amenity</span>
          <span class="detail-value">{{ entry.amenity }}</span>
        </div>
        <div v-if="entry.shop" class="detail-row">
          <span class="detail-label">Shop</span>
          <span class="detail-value">{{ entry.shop }}</span>
        </div>
        <div v-if="entry.leisure" class="detail-row">
          <span class="detail-label">Leisure</span>
          <span class="detail-value">{{ entry.leisure }}</span>
        </div>
        <div v-if="entry.healthcare" class="detail-row">
          <span class="detail-label">Healthcare</span>
          <span class="detail-value">{{ entry.healthcare }}</span>
        </div>
        <div v-if="entry.brand" class="detail-row">
          <span class="detail-label">Brand</span>
          <span class="detail-value">{{ entry.brand }}</span>
        </div>
        <div v-if="entry.cuisine" class="detail-row">
          <span class="detail-label">Cuisine</span>
          <span class="detail-value">{{ entry.cuisine }}</span>
        </div>
        <div v-if="entryAddress(entry)" class="detail-row">
          <span class="detail-label">Address</span>
          <span class="detail-value">{{ entryAddress(entry) }}</span>
        </div>
        <div v-if="entry.phone" class="detail-row">
          <span class="detail-label">Phone</span>
          <span class="detail-value">{{ entry.phone }}</span>
        </div>
        <div v-if="entry.website" class="detail-row">
          <span class="detail-label">Website</span>
          <span class="detail-value">
            <a :href="entry.website" target="_blank" rel="noopener">
              {{ entry.website }}
            </a>
          </span>
        </div>
        <div v-if="entry.opening_hours" class="detail-row">
          <span class="detail-label">Hours</span>
          <span class="detail-value">{{ entry.opening_hours }}</span>
        </div>
        <div v-if="formatLastEdited(entry.last_edited)" class="detail-row">
          <span class="detail-label">Last edited</span>
          <span class="detail-value">{{ formatLastEdited(entry.last_edited) }}</span>
        </div>
        <div v-if="entryConf(entry) != null" class="detail-row">
          <span class="detail-label">Confidence</span>
          <span class="detail-value">
            {{ (entryConf(entry) * 100).toFixed(0) }}%
            <span
              v-if="entry.conf_lower != null && entry.conf_upper != null"
              class="conf-interval"
            >
              ({{ (entry.conf_lower * 100).toFixed(0) }}%–{{ (entry.conf_upper * 100).toFixed(0) }}%)
            </span>
          </span>
        </div>
        <div v-if="entryConf(entry) != null" class="confidence-bar">
          <div
            class="confidence-fill"
            :style="{
              width: (entryConf(entry) * 100) + '%',
              backgroundColor: confidenceColor(entryConf(entry)),
            }"
          />
        </div>
        <div v-if="entry.osm_type" class="detail-row">
          <span class="detail-label">OSM type</span>
          <span class="detail-value">{{ entry.osm_type }}</span>
        </div>
        <a
          v-if="entry.osm_id"
          class="osm-link"
          :href="`https://www.openstreetmap.org/${entry.osm_type || 'node'}/${cleanOsmId(entry.osm_id)}`"
          target="_blank"
          rel="noopener"
        >
          View on OpenStreetMap ↗
        </a>
      </template>

      <!-- Overture details -->
      <template v-if="source === 'overture'">
        <div v-if="entry.basic_category" class="detail-row">
          <span class="detail-label">Category</span>
          <span class="detail-value">{{ formatCategory(entry.basic_category) }}</span>
        </div>
        <div v-if="taxonomyPath(entry)" class="detail-row">
          <span class="detail-label">Taxonomy</span>
          <span class="detail-value">{{ taxonomyPath(entry) }}</span>
        </div>
        <div v-if="entry.brand" class="detail-row">
          <span class="detail-label">Brand</span>
          <span class="detail-value">{{ entry.brand }}</span>
        </div>
        <div v-if="entryAddress(entry)" class="detail-row">
          <span class="detail-label">Address</span>
          <span class="detail-value">{{ entryAddress(entry) }}</span>
        </div>
        <div v-if="entry.phone" class="detail-row">
          <span class="detail-label">Phone</span>
          <span class="detail-value">{{ entry.phone }}</span>
        </div>
        <div v-if="entry.website" class="detail-row">
          <span class="detail-label">Website</span>
          <span class="detail-value">
            <a :href="entry.website" target="_blank" rel="noopener">
              {{ entry.website }}
            </a>
          </span>
        </div>
        <div v-if="entry.source_dataset" class="detail-row">
          <span class="detail-label">Source</span>
          <span class="detail-value">{{ entry.source_dataset }}</span>
        </div>
        <div v-if="entryConf(entry) != null" class="detail-row">
          <span class="detail-label">Confidence</span>
          <span class="detail-value">{{ (entryConf(entry) * 100).toFixed(0) }}%</span>
        </div>
        <div v-if="entryConf(entry) != null" class="confidence-bar">
          <div
            class="confidence-fill"
            :style="{
              width: (entryConf(entry) * 100) + '%',
              backgroundColor: confidenceColor(entryConf(entry)),
            }"
          />
        </div>
        <div v-if="entry.id" class="detail-row detail-row--muted">
          <span class="detail-label">ID</span>
          <span class="detail-value detail-monospace">{{ entry.id }}</span>
        </div>
      </template>

      <!-- Conflated details -->
      <template v-if="source === 'conflated'">
        <div v-if="entry.shared_label" class="detail-row">
          <span class="detail-label">Category</span>
          <span class="detail-value">{{ formatCategory(entry.shared_label) }}</span>
        </div>
        <div v-if="entry.source" class="detail-row">
          <span class="detail-label">Match type</span>
          <span class="detail-value">{{ formatMatchType(entry.source) }}</span>
        </div>
        <div v-if="entry.brand" class="detail-row">
          <span class="detail-label">Brand</span>
          <span class="detail-value">{{ entry.brand }}</span>
        </div>
        <div v-if="showBothNames(entry)" class="detail-row">
          <span class="detail-label">OSM name</span>
          <span class="detail-value">{{ entry.osm_name }}</span>
        </div>
        <div v-if="showBothNames(entry)" class="detail-row">
          <span class="detail-label">Overture name</span>
          <span class="detail-value">{{ entry.overture_name }}</span>
        </div>
        <div v-if="entryConf(entry) != null" class="detail-row">
          <span class="detail-label">Confidence</span>
          <span class="detail-value">
            {{ (entryConf(entry) * 100).toFixed(0) }}%
            <span
              v-if="entry.conf_lower != null && entry.conf_upper != null
                && !isNaN(entry.conf_lower) && !isNaN(entry.conf_upper)"
              class="conf-interval"
            >
              ({{ (entry.conf_lower * 100).toFixed(0) }}%–{{ (entry.conf_upper * 100).toFixed(0) }}%)
            </span>
          </span>
        </div>
        <div v-if="entryConf(entry) != null" class="confidence-bar">
          <div
            class="confidence-fill"
            :style="{
              width: (entryConf(entry) * 100) + '%',
              backgroundColor: confidenceColor(entryConf(entry)),
            }"
          />
        </div>
        <div
          v-if="validNum(entry.osm_conf_mean) && validNum(entry.overture_confidence)"
          class="detail-row"
        >
          <span class="detail-label">Source scores</span>
          <span class="detail-value">
            OSM {{ (entry.osm_conf_mean * 100).toFixed(0) }}%
            / Overture {{ (entry.overture_confidence * 100).toFixed(0) }}%
          </span>
        </div>
        <div v-if="validNum(entry.match_score)" class="detail-row">
          <span class="detail-label">Match score</span>
          <span class="detail-value">
            {{ (entry.match_score * 100).toFixed(0) }}%
            <template v-if="validNum(entry.match_distance_m)">
              ({{ entry.match_distance_m.toFixed(0) }} m)
            </template>
          </span>
        </div>
        <a
          v-if="entry.osm_id"
          class="osm-link"
          :href="`https://www.openstreetmap.org/${entry.osm_type || 'node'}/${cleanOsmId(entry.osm_id)}`"
          target="_blank"
          rel="noopener"
        >
          View on OpenStreetMap ↗
        </a>
        <div v-if="entry.unified_id" class="detail-row detail-row--muted">
          <span class="detail-label">ID</span>
          <span class="detail-value detail-monospace">{{ entry.unified_id }}</span>
        </div>
      </template>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { confidenceColor } from '../utils.js'

const props = defineProps({
  feature: { type: Object, default: null },
})
defineEmits(['close'])

const source = computed(() => props.feature?.get('_source'))

/**
 * Build the list of detail objects to render. If the feature has _colocated
 * (two or more points at the exact same location), show all of them;
 * otherwise wrap the single feature's properties as a one-item array.
 */
const entries = computed(() => {
  if (!props.feature) return []
  const colocated = props.feature.get('_colocated')
  if (colocated && colocated.length > 0) return colocated
  // Build a plain object from the OL feature's properties
  const obj = {}
  props.feature.getKeys().forEach(k => { obj[k] = props.feature.get(k) })
  return [obj]
})

function entryConf(entry) {
  const v = entry.conf_mean ?? entry.confidence
  return v != null && !isNaN(v) ? v : null
}

function entryAddress(entry) {
  const parts = [entry['addr:street'], entry['addr:city'], entry['addr:state']]
    .filter(Boolean)
  return parts.length ? parts.join(', ') : null
}

function formatLastEdited(raw) {
  if (raw == null) return null
  // Parquet timestamps arrive as numbers; detect unit by magnitude:
  //   < 1e10  → seconds (OSM stores ~1.3–1.8e9)  → multiply by 1000
  //   < 1e13  → milliseconds                      → use as-is
  //   else    → microseconds                      → divide by 1000
  const ms = raw < 1e10 ? raw * 1000 : raw > 1e13 ? raw / 1000 : raw
  const d = new Date(ms)
  if (isNaN(d.getTime())) return null
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

function formatCategory(cat) {
  if (!cat) return ''
  return cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// Overture's full taxonomy path, most general first ("Food And Drink ›
// Restaurant › Casual Eatery"); falls back to the primary category alone.
function taxonomyPath(entry) {
  const path = entry.taxonomy_hierarchy
    ? entry.taxonomy_hierarchy.split(' > ')
    : [entry.taxonomy_primary].filter(Boolean)
  return path.map(formatCategory).join(' › ')
}

function formatMatchType(src) {
  if (src === 'matched') return 'Matched (OSM + Overture)'
  if (src === 'osm') return 'OSM only'
  if (src === 'overture') return 'Overture only'
  return src
}

function showBothNames(entry) {
  return entry.source === 'matched'
    && entry.osm_name && entry.overture_name
    && entry.osm_name !== entry.overture_name
}

function validNum(v) {
  return v != null && !isNaN(v)
}

function cleanOsmId(id) {
  if (id == null) return null
  const s = String(id)
  // Strip trailing ".0" from float-cast IDs
  return s.replace(/\.0$/, '')
}
</script>
