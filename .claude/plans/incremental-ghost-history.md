# Incremental OSM history for ghost building (no full-history download)

**Status 2026-09-26: implemented, backtested (PASS), default for the October 2026
refresh.** Branch `feature/matched-index-modes` working tree (uncommitted at time of
writing). Implementation: `src/openpois/io/osm_history_incremental.py`; driver
`scripts/osm_data/download_history.py` (`make download_history`, `PLAN=1`); QA
`make check_history`; backtest `scripts/osm_data/backtest_incremental_history.py`.

**Decisions (Nat, 2026-09-26):**
- **Output:** rolled parquets, not ghost-append.
- **Feed:** public diffs only.
- **Chain limit:** warn at 12 chained months.
- **Deletion timestamp:** the diff file's `state_ts`.
- **Gate:** the backtest had to pass before flipping the default.

**Where the implementation departs from the draft below:**
- The base is `download.osm.incremental_history.base_version`, not a `versions:`
  key; `config_versioned`'s `custom_version` resolves it, so there is no new
  directories entry.
- Diffs are parsed as streaming XML, not with pyosmium. pyosmium's tag iterator
  cost ~60 µs per tag (64 s vs 6 s per US file). The two parsers were row-identical
  on all 152 backtest files.
- Coverage for bases without `history_coverage.json` is inferred from the newest
  version timestamp. `osm_data/20260724` claims `end_date: 2026-07-22` but its
  data ends 2026-07-12 23:59, so the config value would have been wrong.
- The run hard-fails when `end_date` is not after the base coverage, or when no
  feed has a new diff. Either would otherwise silently copy the base.

**Backtest result** (`osm_data/20260818_incremental_backtest/backtest_report.md`).
`20260724` was rolled forward over 38 daily files (2026-07-13 → 2026-08-18 20:20)
and compared with the full `20260902` build on shared nodes:

- **Named + labeled ghosts: recall 99.06%, precision 99.25%**, against a 97% gate.
- `hard_delete`: 99.8% / 99.6%. Rename is the weakest type (98.0% / 95.5%): the
  extra renames are same-day edit bursts (edits seconds apart) that the daily
  diff collapses.
- `hard_delete` timestamp lag: median 13 h, max 24 h.
- History vs. the September snapshot: rolled 99.992%, full build 99.905%
  timestamp match.
- Roll wall time was 40 min: ~10 min of downloads, plus the old pyosmium parser,
  which has since been replaced. The expected monthly time is ~15–20 min.

## Problem

Change detection's ghosts (`scripts/conflation/build_ghosts.py`) are read from
`osm_data/<v>/osm_versions.parquet` + `osm_changes.parquet`. Today those parquets come only
from `scripts/osm_data/download_history.py`, which downloads the 23.41 GB
`us-internal.osh.pbf` (plus the PR / USVI / American Oceania history extracts). It then
runs a two-pass osmium filter, a time-filter and a pyosmium parse. The turnover model is
refit only occasionally (`model_output` stays pinned), so in most months this download
exists only to add about one month of ghost events:

| Month (event_timestamp) | Ghosts | Named |
|---|--:|--:|
| 2026-05 | 6,658 | 3,617 |
| 2026-06 | 6,020 | 4,154 |
| 2026-07 | 5,347 | 4,106 |

(`ghost_osm/20260902/ghosts.parquet`: 665,691 ghosts overall, ~6k new per month.)

## Recommendation

**Roll the previous month's history parquets forward using Geofabrik's public daily
replication diffs.** The output is a new `osm_data/<v>/` directory with the same two
parquets in the same schema, so `build_ghosts.py`, `apply_change_detection.py` and every
later stage run unchanged. This is selected by a pipeline toggle,
`download.osm.history_mode: full | incremental`.

| | Full (today) | Incremental |
|---|---|---|
| Download | 23.41 GB US + territories | ~11–13 MB/day US → **~0.35–0.45 GB/month** (+ small territory diffs) |
| Processing | two osmium passes + time-filter + parse of ~15M versions | parse ~30 daily diffs, one DuckDB fold over the base, parquet rewrite: minutes |
| Auth | Geofabrik OAuth cookie (expires) | none (public server) |
| Ghost fidelity | reference | same event logic; the differences below are bounded and measured in the backtest |
| Usable for a λ refit | yes | **no**: rolled parquets are "ghost grade" (see Fidelity) |

## Evidence

### Geofabrik daily diffs (verified 2026-09-26)

- Layout: `https://download.geofabrik.de/north-america/us-updates/` → `state.txt`
  (sequence 2393, `2026-09-25T20:24:36Z`) and `000/002/NNN.osc.gz` plus `NNN.state.txt`.
  One file per day, cut around 20:24 UTC. Territory feeds exist at
  `north-america/us/puerto-rico-updates/`, `north-america/us/us-virgin-islands-updates/`
  and `australia-oceania/american-oceania-updates/`.
- **Retention is about 131 days.** The oldest file on the server is `000/002/261.osc.gz`
  (2026-05-19); `000/001/` has already been pruned. A monthly cadence therefore has about
  a 3-month cushion. Missing ~4 consecutive months forces a full-history run.
- Public files drop `user`, `uid` and `changeset`
  ([technical.html](https://download.geofabrik.de/technical.html)). The OAuth internal
  server publishes the same feeds with that metadata.
- **Semantics, checked directly on `392.osc.gz` and `393.osc.gz`:**
  - No object appears twice in a file. The diffs are derived extract-to-extract (the
    `osmium derive-changes` convention), not filtered planet osmChange. **Edits made on
    the same day are collapsed**: a create/modify carries only the day's final version.
  - All 25,744 and 31,587 node `<delete>`s carry the prior lat/lon, and tagged ones carry
    the prior tags (keep-details). About 160 POI-tagged node deletes per day, roughly 4.8k
    per month, which is consistent with the observed ghost rate.
  - **A delete's `version`/`timestamp` are the node's last live version, not the
    deletion.** Timestamps on deletes range from 2008 to 2026. The real deletion version
    is ≥ last + 1, and the real deletion time falls inside the file's day window.
  - Modify timestamps fall inside the day window (e.g. 2026-09-24/25 for file 393), so
    they are real.
- Tooling in the `openpois` env: pyosmium 4.3.0 (`ReplicationServer.get_state_info`,
  `get_diff_url`, `timestamp_to_sequence`) and osmium-tool 1.19. `collect_diffs()` /
  `pyosmium-get-changes` **silently return partial data on a mid-run download error** and
  hold diffs in memory. Fetch each daily file ourselves with `download_resilient` and
  check sequence continuity.

### How the parquets encode a deletion (from `osm_data/20260902`)

A deletion version has these rows: `visible` `Changed` → `false`, and every tag plus the
`lat`/`lon` pseudo-tags `Deleted` (e.g. node 30455282 v5, node 62890052 v12). The
incremental writer must emit exactly this shape.

### What the ghost consumer actually reads

`change_detection.find_shadow_matches` uses only `geometry`, `shared_label`,
`prior_name`, `prior_brand`, `new_name`, `event_type`, `event_timestamp` (age filter,
3 y) and `ghost_id` (audit only). Unnamed ghosts never match. So sub-day timestamp error
and version gaps cannot change a demotion. The only thing that can is a missed or extra
**named** event.

## Alternatives considered

| Option | Verdict |
|---|---|
| **B. Snapshot differencing**: prior vs current `osm_snapshot.parquet`, no new download | Rejected as the primary method. A node that disappears could be a deletion, a tag removal, or a filter/taxonomy/exclusion change: the September ingest narrowing alone removed about 77k `tourism=information` rows, which would all read as "deletions". Telling them apart needs the raw current PBF, and timing is only known to the month. Kept as the **monthly consistency QA** below. |
| C. `osmium derive-changes --keep-details` on two retained POI-filtered PBFs | Rejected. It uses the same algorithm as Geofabrik's diffs, but on *filtered* files a tag removal looks like a delete, and it needs two PBFs retained. |
| D. `osmium apply-changes -H` on a retained filtered `.osh.pbf`, then re-run the existing filter/parse | Viable and needs less new event code. Rejected because it needs a multi-GB PBF kept between months (the 20260902 ones were deleted, so bootstrapping would need one more full download) and re-parses 10 years of history each month. |
| E. Planet replication diffs (`planet.openstreetmap.org/replication/day/`) | Real deletion versions and timestamps and long retention, but global volume, and clipping by location is impossible for deletes/ways. Possible future fallback if Geofabrik retention is missed; out of scope. |
| F. ohsome API / ohsome-planet | Unverified whether `@deletion` rows carry pre-deletion tags/geometry; external rate limits and lag. Not pursued. |
| G. AWS `osm-pds` history ORC | Still updated (221 GB, 2026-09-25), but DuckDB has no ORC reader with pushdown. No. |
| H. Overpass `adiff` | Infeasible at US × month scale on public instances. |

## Design

### Config

```yaml
versions:
  osm_data: "20261022"        # new dir, as today
  osm_data_base: "20260902"   # NEW: history run this month rolls forward from
                              # (ignored in full mode)
download:
  osm:
    history_mode: "incremental"   # NEW: "full" | "incremental"
    end_date: 2026-10-21          # unchanged meaning: upper bound on coverage
    incremental_history:
      replication_urls:           # one per HistoryExtract name
        us: "https://download.geofabrik.de/north-america/us-updates/"
        pr: "https://download.geofabrik.de/north-america/us/puerto-rico-updates/"
        usvi: "https://download.geofabrik.de/north-america/us/us-virgin-islands-updates/"
        american_oceania: "https://download.geofabrik.de/australia-oceania/american-oceania-updates/"
      cookie_file: null           # set + internal URLs to recover user/uid/changeset
      keep_diffs: false           # keep the downloaded .osc.gz under osm_data/<v>/diffs/
directories:
  osm_data_base:                  # NEW: same root as osm_data, so config_versioned
    versioned: true               # resolves get_file_path("osm_data_base", ...)
    path: ~/data/openpois/osm_data
    files: {osm_changes: osm_changes.parquet, osm_versions: osm_versions.parquet,
            coverage: history_coverage.json}
```

### Coverage metadata (new file in every `osm_data/<v>/`)

`history_coverage.json`:

```json
{"mode": "incremental", "base_version": "20260902",
 "coverage_end": "2026-10-20T20:24:36Z",
 "extracts": {"us": {"server": "...", "last_sequence": 2418,
                     "last_timestamp": "2026-10-20T20:24:36Z"}, "...": {}},
 "chain_length": 1}
```

A full run writes `{"mode": "full", "coverage_end": <end_date>T00:00:00Z, "chain_length": 0}`
with no sequences. For a base without the file (20260902, 20260724), coverage is derived
from its `config.yaml` `download.osm.end_date`.

### Window and sequence selection (per extract)

1. **Start.** If the base recorded `last_sequence` for this extract, start at
   `last_sequence + 1`: exact continuity, and all later months align to diff boundaries.
   Otherwise (a full-mode base) start at the first sequence whose window contains
   `coverage_end`. That is `timestamp_to_sequence(coverage_end)`, adjusted so that its
   state time is ≥ `coverage_end`.
2. **End.** The last sequence whose state timestamp is ≤ `end_date` (or the server's
   latest, if that is earlier).
3. **Hard fail** if the start sequence is no longer on the server (pruned), or if any
   sequence in the range fails to download after retries. The error message tells the
   user to switch to `history_mode: full`. No silent partial windows.

### Processing (new module `src/openpois/io/osm_history_incremental.py`)

1. **Fetch** each `NNN.osc.gz` and `NNN.state.txt` with `download_resilient`, then verify
   gzip integrity.
2. **Pass 1, per file, pyosmium.** Keep an object if it is `deleted`, **or** has any tag,
   **or** has `version > 1`. The last rule keeps POI nodes that lost every tag; it drops
   the ~300k/day fresh untagged vertices. For each kept object record `seq`,
   `state_ts`, type, id, version, timestamp, visible, lat/lon, the full tag list, and
   `matches_filter`. The filter is the same crosswalk-derived value-scoped expressions as
   the full path: add a `matches_osm_tag_filter(tags, exprs)` helper next to
   `taxonomy.build_osm_tag_filter_expressions` so the two can't drift.
3. **Universe.** Keep window elements whose `(type, id)` is in the base `osm_versions`
   **or** that match the filter in any window version. This mirrors the full path's
   "ever carried a POI tag" `getid` universe. It is a DuckDB semi-join.
4. **Base state fold.** For the kept elements only, fold the base `osm_changes` to the
   last tag state in DuckDB: `arg_max(value, version)` / `arg_max(change, version)` per
   `(type, id, key)`, dropping `Deleted`. Take last version and timestamp from base
   `osm_versions`.
5. **Replay per element in `seq` order.** Build each version's tag set with the existing
   `_tag_set_for_version` semantics (tags + `visible` + node `lat`/`lon`) and diff
   against the prior state with the existing `_diff_tag_sets`. Rules:
   - **Dedupe:** skip a window version whose `(type, id, version)` is already in the base
     (this happens in the first file, which straddles `coverage_end`).
   - **Delete:** emit version `prior_version + 1` with tag set `{visible: false}` and no
     location, which gives exactly the history encoding. Timestamp = the file's
     `state_ts`. If the base's last version is already invisible, skip it (the deletion
     is already in the base).
   - **Delete of an element with no known prior state** (in the universe only via
     `matches_filter` on the delete's carried tags): first emit the carried last-live
     version as a normal version row (its version and timestamp are real), then the
     deletion row.
   - **Create/modify:** emit as-is. First-seen elements diff against the empty set, just
     as the full parser treats an element's first version. `build_ghosts` already skips
     events with no prior lat/lon, so this can never produce a ghost.
   - `changeset`/`user`/`uid` are null unless the internal feed is configured.
6. **Write.** Per extract, stream `base ∪ window rows`, with cross-extract dedupe on
   `(type, id, version)`, into the new `osm_versions.parquet` / `osm_changes.parquet`
   using the existing `VERSIONS_SCHEMA` / `CHANGES_SCHEMA`. Use DuckDB `COPY … UNION ALL`
   or a pyarrow stream; never materialize the base in pandas. Then write
   `history_coverage.json` and `config.write_self("osm_data")`.

### Entry point

`scripts/osm_data/download_history.py` branches on `history_mode`: `full` runs the
existing path unchanged; `incremental` calls
`roll_osm_history(base_dir, out_dir, extracts, end_date, ...)`. Add a
`make download_history` target that tees to `logs/osm_history_<ts>.log`, matching the
other targets. `build_ghosts.py` stays unchanged, apart from one log line that prints the
coverage mode and `coverage_end`.

### Guards

- `scripts/osm_data/format_tabular.py` and `scripts/models/osm_turnover.py` refuse an
  `osm_data` whose coverage mode is `incremental` (override flag
  `--allow-incremental-history`). The reasons: daily-collapsed versions distort the
  per-version turnover likelihood, `user`/`changeset` are null (and
  `format_observations` reads both), and elements that joined in-window lack their
  pre-window history. A refit month always uses `history_mode: full`.
- `download_history.py` in incremental mode warns when `chain_length` ≥
  `max_chain_months` (default 6) and hard-fails when the base's filter expressions are
  **wider** than the current crosswalk's. Newly scoped values would have no history
  before the window, so a widening needs a full run; a narrowing is harmless.

## Fidelity vs. a full-history build

| Difference | Effect on ghosts | Measured in backtest |
|---|---|---|
| Same-day edits collapsed | A→B→C rename in one day becomes A→C (still fires). Tag-strip then delete on the same day becomes `hard_delete` with the pre-day state (the more informative prior). Created and deleted the same day gives no ghost (was never in any snapshot). | yes: named-ghost recall |
| Delete timestamp = file `state_ts` | ≤ ~24 h late; irrelevant to a 3-year age filter | yes: timestamp deltas |
| Delete version = last live + 1 | Lower bound; only `ghost_id` text changes | yes |
| Object moved out of the extract polygon reads as a delete | Negligible for US borders (fixed POIs) | – |
| No `user`/`uid`/`changeset` (public feed) | None for ghosts; blocks refits (guarded) | – |
| Elements that joined the universe in-window lack prior history | None: no prior lat/lon means no ghost | – |

## Validation

1. **Unit tests** (`tests/test_osm_history_incremental.py`) on hand-written `.osc`
   fixtures plus tiny base parquets:
   - a rename modify; a tag-removal modify; a modify that removes all tags (untagged
     node, `version > 1`);
   - a delete with carried tags, checking the exact encoding (`visible` Changed false,
     `lat`/`lon` + tags Deleted, version = prior + 1, timestamp = state_ts);
   - a delete of an unknown element (carried version row + deletion row);
   - dedupe of versions already in the base; a delete skipped when the base is already
     invisible;
   - a new element matching the value-scoped filter vs. one that doesn't;
   - cross-extract duplicates; a pruned start sequence → hard fail; a gap → hard fail;
   - an end-to-end check that `build_ghosts()` on rolled parquets emits the expected four
     event types.
2. **Backtest against a real full build (the release gate for this feature).** Roll
   `osm_data/20260724` (end 2026-07-22) forward to 2026-08-19 with the public diffs, then
   compare with `osm_data/20260902` (a full build, end 2026-08-19). **Time-boxed:** the
   2026-07-21/22 diffs are pruned around 2026-11-29.
   - Restrict to `(type, id)` present in both universes, because 20260724 predates the
     September ingest narrowing.
   - Ghosts: match on `(osm_id, event_type)` with event_timestamp in
     [2026-07-23, 2026-08-18]. Report recall and precision overall and for **named,
     labeled** ghosts; the target is ≥ 97% on named/labeled. Report the |Δt|
     distribution for `hard_delete` (expect ≤ 24 h) and exact equality of
     `prior_name`/`prior_brand`/`new_name`/`shared_label`/geometry on matched pairs.
   - Parquets: for `modify` versions, `(type, id, version)` coverage and
     change-row equality.
   - Downstream: `make apply_cd TEST=1` (Seattle) with each ghost set, then diff the
     demoted `unified_id`s; expect identical or near-identical sets.
3. **Monthly consistency QA (every incremental run, cheap).** For every **node** in the
   new `osm_snapshot.parquet` with `last_edited < coverage_end`, the rolled parquets
   must hold a last version with the same timestamp and name. Report the match rate and
   flag anything below 99.5%: a drop signals missed diffs or a bad fold. Snapshot nodes
   missing from the rolled universe are reported separately. This works in full mode too.
   It goes in `verify-pipeline-run` and as a summary at the end of
   `download_history.py`.

## Docs and skills to update when implemented

- `.claude/skills/full-data-pull/SKILL.md`: step 1 (`history_mode`, `osm_data_base`);
  step 2 (incremental needs no cookie; the ~3-month retention; what to do on a
  pruned-sequence failure); remove "download_history is for ghost regeneration only"
  wording that assumes the 23 GB pull.
- `.claude/skills/model-history-pipeline/SKILL.md`: a refit requires
  `history_mode: full`; set `osm_data_base` to the refit version for the months that
  follow.
- `docs/change-detection.md` stage 1 and `.claude/docs/data-sources.md` (Geofabrik diffs:
  URLs, retention, delete semantics); `.claude/docs/data-versioning.md` (`osm_data_base`,
  `history_coverage.json`); `.claude/CLAUDE.md` gotcha: "Geofabrik diff deletes carry
  the last-edit version/timestamp, not the deletion's".
- `CHANGELOG.md` "Methods changes": first incremental ghost build and the backtest
  numbers.

## Open questions (resolved 2026-09-26; see Decisions above)

1. **Rolled parquets vs. ghost-only append.** Recommended: rolled parquets, which keep
   one ghost code path and let a future way-ghost extension reuse them. The alternative
   is appending new ghosts to the previous `ghosts.parquet`, which is lighter (no ~800 MB
   copy per month) but duplicates event logic and freezes old ghosts under old rules. The
   20260902 ghosts lack `new_name`, which a rebuild from parquets fixes for free.
2. **Public vs. internal diffs.** Recommended: public (no cookie expiry, and ghosts
   don't need the metadata). Internal only matters if we ever want refits on rolled
   data, which the guard forbids anyway.
3. **Chain limit.** Warn at 6 months without a full re-anchor, or hard-fail? Should a
   full run be forced on a schedule even without a refit (e.g. every January)?
4. **Deletion timestamp convention.** File `state_ts` (upper bound, recommended) or the
   midpoint of the day window.
5. **First production use.** Recommended: the October run, rolling from `20260902`
   (coverage_end 2026-08-19; its diffs stay on the server until ~late December), **after**
   the backtest passes. If the backtest slips, October stays on `full`.
