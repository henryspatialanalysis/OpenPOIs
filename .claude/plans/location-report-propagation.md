# Change detection: same-entity ghosts + manual overrides (Phase 6 of the approved Close plan)

Master plan: `~/.claude/plans/i-ve-just-switched-into-twinkly-lamport.md`
(approved by Nat 2026-09-24; Appendix = the 57-POI verification sample).
Branch `feature/ghost-closure-evidence`.

## Why

Today's shadow matcher (`src/openpois/conflation/change_detection.py`
`find_shadow_matches`) demotes an Overture-only POI when a nearby OSM node was
deleted or substantially renamed, scored 0.2 distance / 0.4 name / 0.4 exact
label ≥ 0.70, and then a second-stage filter (lines 246-262) DROPS the pair
when the ghost's prior name is a token-subset/superset of the Overture name
(identical names included). Net effect, verified with production weights:
"Joe's Cafe" deleted next to Overture "Joe's Cafe" → no demotion; unnamed or
differently named deletions → demotion ×~0.2. A 57-row sample of
2026-09-02-v0 `shadow_cd` rows, web-verified: different-entity demotions 76%
spurious (listing still open), same-entity 62% spurious (node→way merges,
duplicate cleanups, renames Overture already carries). Nat's decision: demote
ONLY the same-entity case, with guards, and gate the release on precision.

## Changes

### F2a — same-entity rule + guards (`change_detection.py`)

1. **Name gate mandatory.** Keep a candidate pair only if
   `max(token_set_ratio)` over name×name, brand×brand, name×brand, brand×name
   ≥ `min_prior_name_match_score` (config → 70) **or** the names are token
   subset/superset. Compare **normalised** names: NFKD accent strip,
   lowercase, punctuation → space, drop legal suffixes (`inc`, `llc`, `ltd`,
   `co`, `corp`), expand `co.` → `company`, `st.` left alone, and — when the
   remaining tokens still match — drop a trailing category token (pharmacy,
   bank, library, cafe, coffee, restaurant, market, store, shop, salon).
   Sample misses this fixes: "La Madelaine Bakery"/"la Madeleine", "WAPA
   cafe"/"Wapa Café Boutique", "The Hair Co."/"Hair Company",
   "SuperChefs"/"Superchef Brands Llc". Helper `normalise_name()` lives next
   to `_is_token_subset_or_superset` in `ghost_osm.py` (also used by the
   survivor filter).
2. **Delete the second-stage subset/superset drop** (lines 246-262) and the
   comment that justifies it.
3. **Unnamed never matches.** With the gate mandatory, the existing pre-gate
   `if not gname or not oname: continue` already excludes them; keep an
   explicit test.
4. **Rename direction.** `ghost_osm.py` `_flush` records `new_name` on
   `substantial_rename` ghosts (the name after the change). In
   `find_shadow_matches`, skip a rename ghost when
   `token_set_ratio(norm(new_name), norm(overture_name)) ≥ 70` (Overture
   already carries the rename: Plaza Pharmacy → CVS; Kangam → Kangnam);
   demote only when the prior name matches and the new name does not.
5. **Survivor filter widened** (`apply_current_survivor_filter`): radius
   `survivor_radius_m` 50 → 150; the candidate set is every element of the
   full filtered `osm_snapshot.parquet` (nodes, ways, relations by centroid;
   config `directories.snapshot`), not only the rated POIs; names compared
   after `normalise_name()`. Sample cases this catches: Blue Harbor Bank,
   Clark University Campus Store, Girl Scouts of Nassau County, Mendocino
   Masonic Hall.
6. **Ghost age.** Drop ghosts whose `event_timestamp` is older than
   `max_ghost_age_years` (3) at run time (`build_ghosts.py` or the loader in
   `apply_change_detection.py`; the sample's pre-2021 deletions were 90%
   spurious).

`config.yaml` `conflation.change_detection`: `min_prior_name_match_score: 70`
(comment rewritten: this deliberately reverses the May-2026 "decision rule A"
choice — precision over recall at churned addresses; the September sample
measured 31% precision for the loose matcher), `suppress_if_current_survivor.
radius_m: 150`, `suppress_if_current_survivor.use_full_snapshot: true`,
`max_ghost_age_years: 3`.

### F2b — named lifecycle ghosts (`ghost_osm.py:222-242`)

Drop the `not has_prior_name` guard on `lifecycle_prefix_added` and
`primary_tag_deleted`; delete the now-dead "avoids the noise of named
lifecycle retagging" rationale (lines 211-216) and the matching sentence in
`docs/change-detection.md` §1. Nodes only (ways need a position source;
follow-up noted in `.claude/TODO.md`).

### F4 — manual overrides, applied AFTER calibration

New `src/openpois/conflation/manual_overrides.py` + versioned CSV
`manual_overrides.csv` (`unified_id, overture_id, action, reason, date,
report_id`; `action ∈ exclude | include`; one of the two ids required). New
script `scripts/conflation/apply_manual_overrides.py` and Makefile target
`apply_manual_overrides` appended to the `conflate` chain **after**
`apply_calibration` (so a forced `conf_mean = 0` is never re-scaled):
`exclude` → `conf_mean = conf_lower = conf_upper = 0`, `calibration_flag =
'manual_exclude'`; `include` → row kept with `conf_mean = 1`,
`calibration_flag = 'manual_include'` (for POIs Close must carry despite a
low score). Config: `conflation.manual_overrides.path`. Rows are appended by
the Close triage lane (report id in the last column) and are the fix path for
Overture-only closures / wrong POIs at any threshold.

## Tests (`tests/test_change_detection.py`, new)

Synthetic pairs 3 m apart, both `Cafe`, production weights: same name →
match; "Walgreens"/"Walgreens Pharmacy" → match; "Starbucks" brand-only →
match; "Joe's Coffee House"/"Joe's Cafe" → no match; unnamed ghost → no
match; unrelated → no match; "La Madelaine Bakery"/"la Madeleine" → match
(normalisation); rename ghost prior "Plaza Pharmacy" new "CVS Pharmacy" vs
Overture "CVS Pharmacy" → skipped; survivor: a same-name way centroid 120 m
away suppresses; a 4-year-old ghost is dropped. `test_ghost_osm.py`: named
`disused:` retag emits a `lifecycle_prefix_added` ghost with `new_name`.
`test_manual_overrides.py`: exclude/include applied after calibration, flags
set, idempotent.

## Release gate (before the next published run)

`make conflate TEST=1` (Seattle clip) → `scripts/conflation/diff_change_detection.py`
→ sample ≥ 100 demoted POIs (all if fewer) → vet with `vetting_viz/` or a
Sonnet fan-out (prompt pattern in the master plan's Appendix) → precision
(listing actually closed/moved) **≥ 70%** required to ship. Record the number
and the rule change under "Methods changes" in `CHANGELOG.md` — Close's
threshold doc reads that section monthly.

## Docs

`docs/change-detection.md` §1 (ghost types), §3 (matcher rules, remove the
subset/superset sentence, add the guards), §Validation (new gate);
`.claude/CLAUDE.md` gotcha: "apply_manual_overrides runs LAST";
`.claude/TODO.md`: ways as ghost sources.
