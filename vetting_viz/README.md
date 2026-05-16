# vetting_viz

Quick interactive mapping viz for hand-vetting the output of the
change-detection conflation pipeline. Loads a CSV with the
[`seattle_demoted_pois.csv`](../scripts/conflation/diff_change_detection.py)
shape (or any later vetting-session export), drops each row on a
Leaflet map, and lets you tag each point as `True drop`, `False drop`,
or leave it `Unvetted`. Export to CSV when you're done.

## Run

No build step — serve the directory through the ``openpois`` conda
env's Python and open ``http://localhost:8765/`` in a browser:

```bash
conda run -n openpois python -m http.server --directory vetting_viz 8765
```

## Workflow

1. Click **Load CSV** and pick a demoted-POI CSV (e.g.
   `~/data/openpois/logs/seattle_demoted_pois.csv`). If the file has
   no `vetted` column, one is added with every row set to `Unvetted`.
2. Filter by **POI type** (sidebar checkboxes) and by **vetting
   status**.
3. Click a point. The popup shows every column from the CSV plus a
   radio group for the vetting status. Selecting a radio:
   - Immediately recolors the marker (yellow → green / red).
   - Updates the in-memory CSV row.
   - Re-applies the active filter (so e.g. flipping a row to
     `True drop` while filtering only `Unvetted` will hide it).
4. **Export CSV** at any time. The download includes every original
   column plus the updated `vetted` column.
5. Future sessions: load the previously-exported CSV — the `vetted`
   column is recognized and the markers come up pre-colored.

## Column expectations

Required: `lon`, `lat`. Recommended: `name`, `shared_label`,
`shadow_event_type`, `shadow_ghost_id`, `ghost_prior_name`. The popup
renders columns in the order they appeared in the input CSV.

`shadow_ghost_id` values matching `node/<id>` / `way/<id>` /
`relation/<id>` render as clickable osm.org links so you can pull the
original element up for review.

## Color key

| Status     | Fill   |
|------------|--------|
| Unvetted   | Yellow |
| True drop  | Green  |
| False drop | Red    |
