<template>
  <div class="amenity-filter">
    <div class="amenity-filter-header" @click="collapsed = !collapsed">
      <span>Filters</span>
      <span>{{ collapsed ? '+' : '-' }}</span>
    </div>
    <div v-if="!collapsed" class="amenity-filter-body">
      <template v-if="activeSource === 'osm'">
        <label v-for="f in osmFilterKeys" :key="f.key">
          <input
            type="checkbox"
            :checked="osmFilters[f.key]"
            @change="toggleOsm(f.key)"
          />
          {{ f.label }}
        </label>
      </template>
      <template v-else-if="activeSource === 'overture'">
        <p v-if="overtureError" class="filter-note">
          Overture categories are unavailable, so filtering is off.
        </p>
        <p v-else-if="overtureGroups.length === 0" class="filter-note">
          Loading categories…
        </p>
        <template v-else>
          <div class="filter-actions">
            <button class="filter-action-btn" @click="setAllOverture(true)">All</button>
            <button class="filter-action-btn" @click="setAllOverture(false)">None</button>
          </div>
          <div class="overture-filter-list">
            <div v-for="g in overtureGroups" :key="g.l0" class="overture-group">
              <div class="overture-group-row">
                <button
                  class="overture-group-toggle"
                  :aria-expanded="expandedL0 === g.l0"
                  :aria-label="`Show ${g.label} categories`"
                  @click="expandedL0 = expandedL0 === g.l0 ? null : g.l0"
                >
                  {{ expandedL0 === g.l0 ? '▾' : '▸' }}
                </button>
                <label>
                  <input
                    type="checkbox"
                    :checked="groupState(g) === 'all'"
                    :indeterminate.prop="groupState(g) === 'some'"
                    @change="setGroupOverture(g, groupState(g) !== 'all')"
                  />
                  {{ g.label }}
                </label>
              </div>
              <div v-if="expandedL0 === g.l0" class="overture-group-members">
                <label v-for="c in g.categories" :key="c.key" :title="`${c.count.toLocaleString()} POIs`">
                  <input
                    type="checkbox"
                    :checked="isOvertureOn(c.key)"
                    @change="toggleOverture(c.key)"
                  />
                  {{ c.label }}
                </label>
              </div>
            </div>
          </div>
        </template>
      </template>
      <template v-else-if="activeSource === 'conflated'">
        <div class="filter-actions">
          <button class="filter-action-btn" @click="selectAllConflated">All</button>
          <button class="filter-action-btn" @click="selectNoneConflated">None</button>
        </div>
        <div class="conflated-filter-list">
          <label v-for="lbl in conflatedLabels" :key="lbl">
            <input
              type="checkbox"
              :checked="conflatedFilters[lbl]"
              @change="toggleConflated(lbl)"
            />
            {{ lbl }}
          </label>
        </div>
      </template>

    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { OSM_FILTER_KEYS } from '../constants.js'
import { useOvertureCategories } from '../composables/useOvertureCategories.js'

const props = defineProps({
  activeSource: { type: String, required: true },
  osmFilters: { type: Object, required: true },
  overtureFilters: { type: Object, required: true },
  conflatedFilters: { type: Object, required: true },
  conflatedLabels: { type: Array, required: true },
})

const emit = defineEmits([
  'update:osm-filters',
  'update:overture-filters',
  'update:conflated-filters',
])
const collapsed = ref(false)
const osmFilterKeys = OSM_FILTER_KEYS
const { groups: overtureGroups, error: overtureError } = useOvertureCategories()
const expandedL0 = ref(null)

function toggleOsm(key) {
  emit('update:osm-filters', { ...props.osmFilters, [key]: !props.osmFilters[key] })
}

// Overture filters hide only keys set to false, so a key not yet in the
// object (every key, before the first click) counts as on.
function isOvertureOn(key) {
  return props.overtureFilters[key] !== false
}

function toggleOverture(key) {
  emit('update:overture-filters', {
    ...props.overtureFilters,
    [key]: !isOvertureOn(key),
  })
}

function groupState(group) {
  const on = group.categories.filter(c => isOvertureOn(c.key)).length
  if (on === group.categories.length) return 'all'
  return on === 0 ? 'none' : 'some'
}

function setGroupOverture(group, value) {
  const next = { ...props.overtureFilters }
  for (const c of group.categories) next[c.key] = value
  emit('update:overture-filters', next)
}

function setAllOverture(value) {
  const next = {}
  for (const g of overtureGroups.value) {
    for (const c of g.categories) next[c.key] = value
  }
  emit('update:overture-filters', next)
}

function toggleConflated(label) {
  emit('update:conflated-filters', {
    ...props.conflatedFilters,
    [label]: !props.conflatedFilters[label],
  })
}

function selectAllConflated() {
  const all = {}
  for (const lbl of props.conflatedLabels) all[lbl] = true
  emit('update:conflated-filters', all)
}

function selectNoneConflated() {
  const none = {}
  for (const lbl of props.conflatedLabels) none[lbl] = false
  emit('update:conflated-filters', none)
}

</script>
