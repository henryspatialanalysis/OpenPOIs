#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------
"""
OSM lifecycle-namespace prefixes shared across the package.

The OSM community marks a feature that no longer functions as mapped by moving
its primary function tag into a *lifecycle* namespace — e.g. a closed shop's
``shop=supermarket`` becomes ``disused:shop=supermarket`` — while leaving the
``name`` tag plain as a re-map safeguard. See the OSM "Lifecycle prefix" wiki.

These prefixes have two consumers in this package, both keyed off this single
source of truth:

- ``openpois.osm.format_observations`` — treats a lifecycle prefix appearing on
  a POI's primary tag as a turnover (closure) event for the λ hazard model.
- ``openpois.conflation.ghost_osm`` — emits a ghost POI for the
  change-detection / Overture-penalty pipeline.
"""
from __future__ import annotations


LIFECYCLE_PREFIXES: tuple[str, ...] = (
    "disused:",
    "abandoned:",
    "demolished:",
    "was:",
    "removed:",
    "razed:",
)


def is_lifecycle_key(key: str) -> bool:
    """True when ``key`` lives in any lifecycle namespace (e.g. ``disused:shop``)."""
    return any(key.startswith(p) for p in LIFECYCLE_PREFIXES)
