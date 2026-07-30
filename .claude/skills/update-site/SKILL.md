---
name: update-site
description: Use when the user wants to bump the frontend to point at newly uploaded Source Cooperative data, or wants to run/preview/build the site locally. Triggers: "push new data to the site", "bump site to latest data version", "update constants.js", "deploy the site", "preview the site with new data", "rebuild site after data refresh".
---

# Update + verify the site

Vue 3 + Vite frontend lives in [site/](../../../site/). After a data pull +
publish, the site's PMTiles URLs need a manual bump to the new Source
Cooperative version folder.

## Prerequisites

- New data published to Source Cooperative via [skills/conflate-snapshots](../conflate-snapshots/SKILL.md).
- Node + npm available (see `site/package.json` for engine requirements).

## Steps

1. **Sync taxonomy** — run the [sync-taxonomy](../sync-taxonomy/SKILL.md) skill first. It regenerates `site/src/taxonomy.generated.js` and `site/public/taxonomy.html` from the conflation CSVs and checks `constants.js` for missing display labels. Catch drift before touching data URLs.

2. **Update PMTiles URLs in [site/src/constants.js](../../../site/src/constants.js)** — both point at the Source Coop version folder (`versions.source_coop` in `config.yaml`):
   - `OSM_PMTILES_URL` → `https://data.source.coop/henryspatialanalysis/openpois/<YYYY-MM-DD-vN>/osm-pmtiles/osm.pmtiles`
   - `CONFLATED_PMTILES_URL` → `https://data.source.coop/henryspatialanalysis/openpois/<YYYY-MM-DD-vN>/conflated-pmtiles/conflated.pmtiles`
   - `OVERTURE_PMTILES_URL` → bump on monthly Overture release

3. **Local preview**:
   ```bash
   cd site && npm run dev
   ```
   Verify:
   - Map loads POIs at zoom 14+ without CORS/404 errors on `data.source.coop`
   - Source filter dropdown (OSM / Overture / Conflated) toggles data
   - Taxonomy legend renders from `taxonomy.html`
   - POI popups show non-empty name/category/confidence
   - **Post-territory-expansion (≥ 2026-05-21)**: pan to each of GU, VI, MP, AS — Guam (~+144°E) and American Samoa (~-170°W) are the longitudes most likely to expose tile-wrap or PMTiles edge bugs that haven't been exercised on real data. Confirm points render at all 4. Search-bar sanity check: `"Hagåtña"`, `"Charlotte Amalie"`, `"Saipan"`, `"Pago Pago"` should resolve via the Stadia geocoder (`useGeocoder.js` widens `boundary.country` to include `PR,VI,GU,MP,AS`).

4. **Production build**:
   ```bash
   npm run build
   ```
   Inspect `dist/` output; flag large chunk-size increases if dependencies changed.

   **Post-territory-expansion**: tippecanoe (run upstream during `conflate-snapshots`) now widens its tile pyramid to cover both hemispheres now that territory POIs span ~+144°E to ~-170°W. The PMTiles archive should still be small (z14, point data only), but a >2× jump in `osm.pmtiles` / `conflated.pmtiles` size vs the prior version suggests an unwanted full-globe tile pyramid — investigate before deploying.

5. **Deploy — automatic.** [.github/workflows/deploy-site.yml](../../../.github/workflows/deploy-site.yml)
   builds and publishes to GitHub Pages on every push to `main` touching
   `site/**`, `src/**`, `docs/**`, `scripts/**` or the workflows. Merging the
   release PR is the deploy; there is nothing to run by hand. Trigger a rebuild
   without a code change via `workflow_dispatch`, and watch a run with:
   ```bash
   gh run list --workflow=deploy-site.yml --limit 5
   ```
   Note CI regenerates `site/public/taxonomy.html` from the crosswalk CSVs and
   builds the Sphinx docs into `site/dist/docs`, so both can differ from what a
   local `npm run build` produces.

6. **Post-deploy check** — load the deployed site, open browser console, confirm no CORS or 404s on the new Source Coop URLs. After a taxonomy change also check `/taxonomy.html` lists the new labels, and after an `api.rst` change check `/docs`.

## Commit convention

Two separate commits, matching the recent history:
- "Push to new data version" — `config.yaml` and publish-side changes
- "Update to latest data version" — `site/src/constants.js`

## Key files

- [site/src/constants.js](../../../site/src/constants.js) — PMTiles URLs, color ramps, zoom thresholds, CONFLATED_LABELS
- [site/vite.config.js](../../../site/vite.config.js) — code-split chunks (ol, duckdb, arrow, etc.)
- [site/README.md](../../../site/README.md) — maintenance notes on the PMTiles URLs
