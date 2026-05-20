// vetting_viz — interactive review of demoted POIs from the conflation
// change-detection pipeline.
//
// Loads a CSV with the seattle_demoted_pois.csv shape, plots each row
// as a colored point on a Leaflet map (Yellow=Unvetted, Green=True
// drop, Red=False drop), and lets the reviewer tag each point via a
// popup radio. Exports the updated CSV (with the `vetted` column
// added/preserved) at session end.
//
// Standalone — no build step. Leaflet + Papa Parse are loaded from
// CDN via index.html.

(() => {
  "use strict";

  const VETTED_VALUES = ["Unvetted", "True drop", "False drop"];

  const FILL_COLOR = {
    "Unvetted":   "#facc15",
    "True drop":  "#22c55e",
    "False drop": "#ef4444",
  };

  const STROKE_COLOR = {
    "Unvetted":   "#a87f00",
    "True drop":  "#15803d",
    "False drop": "#991b1b",
  };

  // -------------------------------------------------------------------
  // Map setup
  // -------------------------------------------------------------------

  const map = L.map("map", {
    center: [47.60, -122.33],
    zoom: 11,
    zoomControl: true,
  });

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  // Single feature-group layer so we can iterate / clear easily.
  const pointsLayer = L.layerGroup().addTo(map);

  // -------------------------------------------------------------------
  // State
  // -------------------------------------------------------------------

  const state = {
    rows: [],            // array of CSV row objects (vetted column is
                         // mutable)
    markers: [],         // parallel array of Leaflet circle markers;
                         // markers[i] is null if hidden by filter
    csvHeader: null,     // ordered column list (preserves CSV order)
    typeFilter: new Set(),
    statusFilter: new Set(VETTED_VALUES),
  };

  // -------------------------------------------------------------------
  // CSV load
  // -------------------------------------------------------------------

  document
    .getElementById("csv-input")
    .addEventListener("change", (evt) => {
      const file = evt.target.files[0];
      if (!file) return;
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) =>
          loadRows(results.data, results.meta.fields),
      });
    });

  function loadRows(rows, headerFields) {
    // Preserve input column order; append `vetted` last if missing.
    const header = headerFields.slice();
    if (!header.includes("vetted")) header.push("vetted");

    // Drop rows without valid lon/lat (silently).
    const cleaned = rows.filter((r) => {
      const lon = parseFloat(r.lon);
      const lat = parseFloat(r.lat);
      return Number.isFinite(lon) && Number.isFinite(lat);
    });

    cleaned.forEach((r) => {
      if (!VETTED_VALUES.includes(r.vetted)) {
        r.vetted = "Unvetted";
      }
    });

    state.rows = cleaned;
    state.csvHeader = header;
    state.typeFilter = new Set(
      cleaned.map((r) => r.shared_label || "(no type)"),
    );
    state.statusFilter = new Set(VETTED_VALUES);

    rebuildMarkers();
    rebuildTypeFilterUI();
    applyFilter();
    fitToBounds();

    document.getElementById("export-btn").disabled = cleaned.length === 0;
    document.getElementById("empty-overlay").classList.add("hidden");
  }

  function rebuildMarkers() {
    pointsLayer.clearLayers();
    state.markers = state.rows.map((row, idx) => {
      const lon = parseFloat(row.lon);
      const lat = parseFloat(row.lat);
      const marker = L.circleMarker([lat, lon], stylePropsFor(row));
      marker.on("click", () => openPopup(marker, idx));
      marker.addTo(pointsLayer);
      return marker;
    });
  }

  function stylePropsFor(row) {
    const vetted = VETTED_VALUES.includes(row.vetted)
      ? row.vetted
      : "Unvetted";
    return {
      radius: vetted === "Unvetted" ? 5 : 6,
      fillColor: FILL_COLOR[vetted],
      color: STROKE_COLOR[vetted],
      weight: 1.5,
      opacity: 1,
      fillOpacity: 0.85,
    };
  }

  function fitToBounds() {
    const latlngs = state.markers.map((m) => m.getLatLng());
    if (!latlngs.length) return;
    map.fitBounds(L.latLngBounds(latlngs), { padding: [40, 40] });
  }

  // -------------------------------------------------------------------
  // Type filter UI
  // -------------------------------------------------------------------

  function rebuildTypeFilterUI() {
    const container = document.getElementById("type-filter");
    container.innerHTML = "";
    const counts = {};
    state.rows.forEach((r) => {
      const k = r.shared_label || "(no type)";
      counts[k] = (counts[k] || 0) + 1;
    });
    Object.keys(counts)
      .sort((a, b) => counts[b] - counts[a])
      .forEach((type) => {
        const label = document.createElement("label");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = type;
        cb.checked = true;
        cb.addEventListener("change", () => {
          if (cb.checked) state.typeFilter.add(type);
          else state.typeFilter.delete(type);
          applyFilter();
        });
        const name = document.createElement("span");
        name.textContent = type;
        const count = document.createElement("span");
        count.textContent = `(${counts[type]})`;
        count.style.cssText = "color:#94a3b8;margin-left:auto;";
        label.appendChild(cb);
        label.appendChild(name);
        label.appendChild(count);
        container.appendChild(label);
      });
  }

  document.getElementById("type-all").addEventListener("click", () => {
    document
      .querySelectorAll("#type-filter input[type=checkbox]")
      .forEach((cb) => {
        cb.checked = true;
        state.typeFilter.add(cb.value);
      });
    applyFilter();
  });

  document.getElementById("type-none").addEventListener("click", () => {
    document
      .querySelectorAll("#type-filter input[type=checkbox]")
      .forEach((cb) => {
        cb.checked = false;
        state.typeFilter.delete(cb.value);
      });
    applyFilter();
  });

  document
    .querySelectorAll("#status-filter input[type=checkbox]")
    .forEach((cb) => {
      cb.addEventListener("change", () => {
        if (cb.checked) state.statusFilter.add(cb.value);
        else state.statusFilter.delete(cb.value);
        applyFilter();
      });
    });

  function applyFilter() {
    state.markers.forEach((marker, idx) => {
      const row = state.rows[idx];
      const type = row.shared_label || "(no type)";
      const status = row.vetted;
      const keep =
        state.typeFilter.has(type) && state.statusFilter.has(status);
      if (keep && !pointsLayer.hasLayer(marker)) {
        pointsLayer.addLayer(marker);
      } else if (!keep && pointsLayer.hasLayer(marker)) {
        pointsLayer.removeLayer(marker);
      }
    });
    updateStats();
  }

  // -------------------------------------------------------------------
  // Popup
  // -------------------------------------------------------------------

  function openPopup(marker, idx) {
    const row = state.rows[idx];

    const title =
      row.name && row.name.trim() ? row.name : "(no name)";
    const sub = [row.shared_label, row.shadow_event_type]
      .filter(Boolean)
      .join(" • ");

    const omit = new Set(["name", "shared_label", "vetted"]);
    const rowsHtml = state.csvHeader
      .filter((col) => !omit.has(col))
      .map((col) => {
        const val = row[col];
        if (val === undefined || val === null || val === "") return "";
        let displayVal = escapeHtml(String(val));
        if (col === "shadow_ghost_id") {
          const m = String(val).match(/^(node|way|relation)\/(\d+)/);
          if (m) {
            displayVal =
              `<a href="https://www.openstreetmap.org/${m[1]}/${m[2]}" ` +
              `target="_blank" rel="noopener">${escapeHtml(val)}</a>`;
          }
        }
        return (
          `<dt>${escapeHtml(col)}</dt><dd>${displayVal}</dd>`
        );
      })
      .join("");

    const radiosHtml = VETTED_VALUES.map(
      (v) => `
        <label>
          <input type="radio" name="vet-${idx}" value="${escapeAttr(v)}"
            ${row.vetted === v ? "checked" : ""}>
          ${escapeHtml(v)}
        </label>`,
    ).join("");

    const html = `
      <div class="popup-body">
        <h3 class="popup-title">${escapeHtml(title)}</h3>
        ${sub ? `<p class="popup-subtitle">${escapeHtml(sub)}</p>` : ""}
        <fieldset class="popup-vet">
          <legend>Vetting status</legend>
          ${radiosHtml}
        </fieldset>
        <dl class="popup-rows">${rowsHtml}</dl>
      </div>
    `;

    marker.unbindPopup();
    marker.bindPopup(html, {
      maxWidth: 420,
      minWidth: 320,
      autoPan: true,
      keepInView: false,
      className: "vetting-popup",
    });

    // Wire up the radios once Leaflet has put the popup in the DOM —
    // the ``popupopen`` event fires after the content is attached.
    marker.once("popupopen", (evt) => {
      const popupNode = evt.popup.getElement();
      if (!popupNode) return;
      popupNode
        .querySelectorAll('input[type="radio"]')
        .forEach((radio) => {
          radio.addEventListener("change", () => {
            const newVal = radio.value;
            row.vetted = newVal;
            marker.setStyle(stylePropsFor(row));
            applyFilter(); // status filter may now hide it
          });
        });
    });

    marker.openPopup();
  }

  // -------------------------------------------------------------------
  // Stats + export
  // -------------------------------------------------------------------

  function updateStats() {
    const total = state.rows.length;
    const counts = { "Unvetted": 0, "True drop": 0, "False drop": 0 };
    let visible = 0;
    state.markers.forEach((marker, idx) => {
      const row = state.rows[idx];
      counts[row.vetted] = (counts[row.vetted] || 0) + 1;
      if (pointsLayer.hasLayer(marker)) visible += 1;
    });
    document.getElementById("stat-total").textContent =
      `${visible.toLocaleString()} / ${total.toLocaleString()}`;
    document.getElementById("stat-unvetted").textContent =
      counts["Unvetted"].toLocaleString();
    document.getElementById("stat-true").textContent =
      counts["True drop"].toLocaleString();
    document.getElementById("stat-false").textContent =
      counts["False drop"].toLocaleString();
  }

  document.getElementById("export-btn").addEventListener("click", () => {
    if (!state.rows.length) return;
    const csv = Papa.unparse(state.rows, {
      columns: state.csvHeader,
      quotes: false,
    });
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    const ts = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .slice(0, 19);
    a.href = URL.createObjectURL(blob);
    a.download = `vetted_pois_${ts}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  });

  // -------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeAttr(s) {
    return String(s).replace(/"/g, "&quot;");
  }
})();
