#!/usr/bin/env python3
"""Generate site/public/taxonomy.html from the conflation data CSVs.

Run from the repo root:
    python scripts/build_taxonomy.py
"""

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "src/openpois/conflation/data"
OUTPUT = REPO_ROOT / "site/public/taxonomy.html"
OUTPUT_JS = REPO_ROOT / "site/src/taxonomy.generated.js"

# Reserved sentinel in the OSM crosswalk marking a non-POI (key, value) tag
# that is dropped rather than labelled. Kept in sync with
# ``openpois.conflation.taxonomy.EXCLUDE_LABEL``.
EXCLUDE_LABEL = "EXCLUDE"


def osm_cell(group):
    """Format OSM tags for one shared_label, grouped by key."""
    by_key = {}
    wildcard_keys = []
    for _, row in group.iterrows():
        if row["osm_value"] == "*":
            wildcard_keys.append(row["osm_key"])
        else:
            by_key.setdefault(row["osm_key"], []).append(row["osm_value"])
    parts = []
    for key, vals in by_key.items():
        vals_str = ", ".join(vals)
        wiki = f"https://wiki.openstreetmap.org/wiki/{key.capitalize()}"
        parts.append(
            f'<a href="{wiki}" target="_blank" rel="noopener noreferrer"'
            f' class="tx-key">{key}</a>={vals_str}'
        )
    for key in wildcard_keys:
        wiki = f"https://wiki.openstreetmap.org/wiki/{key.capitalize()}"
        parts.append(
            f'<a href="{wiki}" target="_blank" rel="noopener noreferrer"'
            f' class="tx-key">{key}</a>=* — all other {key} tags'
        )
    return "<br>".join(parts)


def overture_cell(group):
    """Format Overture categories for one shared_label.

    Entries are sorted shallowest-first so parent-only rows precede their
    more specific siblings (e.g. `health_care: acupuncture` above
    `health_care > acupuncture: alternative_medicine`).
    """
    entries = []
    for _, row in group.iterrows():
        l0 = row["overture_l0"]
        l1 = row.get("overture_l1", "")
        l2 = row.get("overture_l2", "")
        l3 = row.get("overture_l3", "")
        l1 = l1 if pd.notna(l1) and l1 != "" else None
        l2 = l2 if pd.notna(l2) and l2 != "" else None
        l3 = l3 if pd.notna(l3) and l3 != "" else None
        # Deepest specified node carries the label; show it after the L0.
        leaf = l3 or l2 or l1
        mid = l1 if (l1 and (l2 or l3)) else None
        depth = (1 if l1 else 0) + (1 if l2 else 0) + (1 if l3 else 0)
        if leaf and mid:
            html = (
                f'<span class="tx-key">{l0} &rsaquo;'
                f' {mid}:</span> {leaf}'
            )
        elif leaf:
            html = f'<span class="tx-key">{l0}:</span> {leaf}'
        else:
            html = f'<span class="tx-key">{l0}</span>'
        entries.append((depth, l0, l1 or "", l2 or "", l3 or "", html))
    entries.sort(key = lambda e: (e[0], e[1], e[2], e[3], e[4]))
    return "<br>".join(e[5] for e in entries)


def build_rows(radii, osm, overture):
    osm_html = (
        osm.groupby("shared_label")
        .apply(osm_cell, include_groups = False)
        .rename("osm_html")
    )
    overture_html = (
        overture.groupby("shared_label")
        .apply(overture_cell, include_groups = False)
        .rename("overture_html")
    )
    df = (
        radii.set_index("shared_label")
        .join(osm_html)
        .join(overture_html)
        .fillna("")
    )
    df = df.iloc[
        sorted(
            range(len(df)),
            key = lambda i: (df.index[i].startswith("Other "), df.index[i]),
        )
    ]
    rows = []
    for label, row in df.iterrows():
        rows.append(
            f"""        <tr>
          <td>{label}</td>
          <td class="tx-tags">{row['osm_html']}</td>
          <td class="tx-tags">{row['overture_html']}</td>
        </tr>"""
        )
    return "\n".join(rows)


def excluded_section(excluded):
    """Build the "Excluded tags" HTML block from EXCLUDE crosswalk rows.

    Returns an empty string when there are no exclusions, so the section
    is omitted entirely.
    """
    if excluded.empty:
        return ""
    by_key = {}
    for _, row in excluded.iterrows():
        by_key.setdefault(row["osm_key"], []).append(row["osm_value"])
    items = []
    for key in sorted(by_key):
        vals_str = ", ".join(sorted(by_key[key]))
        wiki = f"https://wiki.openstreetmap.org/wiki/{key.capitalize()}"
        items.append(
            f'          <li><a href="{wiki}" target="_blank"'
            f' rel="noopener noreferrer" class="tx-key">{key}</a>'
            f"={vals_str}</li>"
        )
    lis = "\n".join(items)
    return f"""
    <h2 class="tx-subhead">Excluded tags</h2>
    <p class="lead">
      These OpenStreetMap tags describe map furniture and infrastructure
      (parking, benches, waste baskets, …) rather than places. They are
      dropped during ingest and never appear as POIs or in an
      &ldquo;Other&rdquo; category.
    </p>
    <ul class="tx-excluded">
{lis}
    </ul>"""


def render(rows, excluded_html = ""):
    return f"""<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Taxonomy – OpenPOIs</title>
  <link rel="icon" type="image/x-icon" href="/favicon.ico" />
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
        Oxygen, Ubuntu, Cantarell, sans-serif;
      font-size: 15px;
      color: #333;
      background: #f8f9fa;
      line-height: 1.6;
    }}

    .header {{
      background: #fff;
      border-bottom: 1px solid #ddd;
      padding: 10px 24px;
      display: flex;
      align-items: center;
      gap: 16px;
    }}

    .header-back {{
      color: #2563eb;
      text-decoration: none;
      font-size: 14px;
      white-space: nowrap;
      flex-shrink: 0;
    }}

    .header-back:hover {{ text-decoration: underline; }}

    .header-spacer {{ flex: 1; }}

    .header-link {{
      color: #555;
      text-decoration: none;
      font-size: 14px;
      white-space: nowrap;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .header-link:hover {{ color: #333; text-decoration: underline; }}

    .brand-logo {{
      height: 28px;
      width: auto;
      display: block;
    }}

    .github-icon {{
      width: 20px;
      height: 20px;
      fill: #555;
      flex-shrink: 0;
    }}

    .header-link:hover .github-icon {{ fill: #333; }}

    .content {{
      max-width: 1100px;
      margin: 48px auto;
      padding: 0 24px;
    }}

    h1 {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 12px;
      color: #111;
    }}

    .lead {{
      font-size: 16px;
      color: #555;
      margin-bottom: 32px;
    }}

    /* Taxonomy table */
    .tx-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      overflow: hidden;
      table-layout: fixed;
    }}

    .tx-table th:nth-child(1),
    .tx-table td:nth-child(1) {{ width: 16%; }}

    .tx-table th:nth-child(2),
    .tx-table td:nth-child(2),
    .tx-table th:nth-child(3),
    .tx-table td:nth-child(3) {{ width: 42%; word-break: break-word; }}

    .tx-table th {{
      background: #f3f4f6;
      padding: 10px 14px;
      text-align: left;
      font-weight: 600;
      font-size: 13px;
      border-bottom: 2px solid #e5e7eb;
      white-space: nowrap;
    }}

    .tx-table td {{
      padding: 8px 14px;
      vertical-align: top;
      border-bottom: 1px solid #f0f0f0;
    }}

    .tx-table tbody tr:last-child td {{ border-bottom: none; }}

    .tx-table tbody tr:hover {{ background: #fafafa; }}

    .tx-tags {{ line-height: 1.8; }}

    .tx-key {{
      color: #888;
      font-size: 12px;
    }}

    .tx-subhead {{
      font-size: 20px;
      font-weight: 700;
      margin: 40px 0 12px;
      color: #111;
    }}

    .tx-excluded {{
      list-style: none;
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 12px 18px;
      font-size: 13px;
      line-height: 1.9;
    }}

    footer {{
      text-align: center;
      padding: 32px 24px;
      color: #999;
      font-size: 13px;
      border-top: 1px solid #e5e7eb;
      margin-top: 48px;
    }}

    footer a {{ color: #2563eb; text-decoration: none; }}
    footer a:hover {{ text-decoration: underline; }}
  </style>
</head>

<body>

  <header class="header">
    <a href="/about.html" class="header-back">&#8592; About</a>
    <div class="header-spacer"></div>
    <a href="https://github.com/henryspatialanalysis/openpois" target="_blank"
      rel="noopener noreferrer" class="header-link" aria-label="GitHub repository">
      <svg class="github-icon" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
          0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
          -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
          .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
          -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
          1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82
          1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
          1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
      </svg>
      GitHub
    </a>
    <a href="https://henryspatialanalysis.com/" target="_blank"
      rel="noopener noreferrer" class="header-link">
      <img src="/logo.png" alt="Henry Spatial Analysis" class="brand-logo" />
    </a>
  </header>

  <main class="content">
    <h1>POI Taxonomy</h1>
    <p class="lead">
      OpenPOIs maps OpenStreetMap and Overture Maps categories to a shared set
      of labels used for conflation and filtering. The match radius controls
      how close two POIs from different sources must be to be considered the
      same place.
    </p>

    <table class="tx-table">
      <thead>
        <tr>
          <th>Shared label</th>
          <th>OpenStreetMap tags</th>
          <th>Overture Maps categories</th>
        </tr>
      </thead>
      <tbody>
{rows}
      </tbody>
    </table>
{excluded_html}
  </main>

  <footer>
    OpenPOIs &mdash;
    by <a href="https://henryspatialanalysis.com/">Henry Spatial Analysis</a>
    &mdash; <a href="https://github.com/henryspatialanalysis/openpois">GitHub</a>
    &mdash; MIT License
  </footer>

</body>
</html>
"""


def sorted_labels(radii):
    """Return shared_labels sorted alphabetically with 'Other *' last."""
    labels = radii["shared_label"].tolist()
    return sorted(labels, key = lambda s: (s.startswith("Other "), s))


def render_js(radii, osm, overture):
    """Build site/src/taxonomy.generated.js with the three raw arrays."""
    labels = sorted_labels(radii)
    osm_keys = sorted(osm["osm_key"].dropna().unique().tolist())
    overture_l0s = sorted(overture["overture_l0"].dropna().unique().tolist())

    def js_array(items):
        inner = ",\n  ".join(f"'{i}'" for i in items)
        return f"[\n  {inner},\n]"

    return (
        "// AUTO-GENERATED by scripts/build_taxonomy.py — do not edit.\n"
        "// Source CSVs: src/openpois/conflation/data/\n"
        "\n"
        f"export const SHARED_LABELS = {js_array(labels)}\n"
        "\n"
        f"export const OSM_KEYS = {js_array(osm_keys)}\n"
        "\n"
        f"export const OVERTURE_L0S = {js_array(overture_l0s)}\n"
    )


def main():
    osm_all = pd.read_csv(
        DATA_DIR / "taxonomy_crosswalk_openstreetmap.csv"
    ).fillna("")
    overture = pd.read_csv(DATA_DIR / "taxonomy_crosswalk_overture_maps.csv")
    radii = pd.read_csv(DATA_DIR / "match_radii.csv")
    # Excluded (non-POI) rows are shown in their own section, never as a
    # shared label / category.
    excluded = osm_all[osm_all["shared_label"] == EXCLUDE_LABEL]
    osm = osm_all[osm_all["shared_label"] != EXCLUDE_LABEL]
    rows = build_rows(radii, osm, overture)
    excluded_html = excluded_section(excluded)
    OUTPUT.write_text(render(rows, excluded_html), encoding = "utf-8")
    print(f"Written: {OUTPUT}")
    OUTPUT_JS.write_text(render_js(radii, osm, overture), encoding = "utf-8")
    print(f"Written: {OUTPUT_JS}")


if __name__ == "__main__":
    main()
