# Release-stable `unified_id`: design record

Status: **implemented 2026-10-26** (branch `stable-unified-ids`). Every published
release was re-issued as `<date>-v1` with these ids. Code:
`src/openpois/conflation/stable_ids.py`, `scripts/conflation/assign_stable_ids.py`
(pipeline stage), `scripts/conflation/backfill_stable_ids.py` and
`scripts/publish/publish_backfill_v1.py` (one-off backfill). Origin: a wtm.ingest
draft (`../wtm.ingest/.claude/plans/openpois-rekey-issue-draft.md`).

## 1. Problem

Before this work `unified_id` was `matched:<osm_id>_<overture_id>`, `osm:<osm_id>`
or `overture:<overture_id>`, rebuilt from each release's pairing.

1. **Not unique.** The OSM part omitted the element type, so a node and a way with
   the same number collided: 148 duplicated ids in 2026-10-22-v0.
2. **Not stable.** Any change to either partner re-keyed an unchanged place.
   20261001 → 20261022 (full files): 258,397 of 16.6M prior ids vanished. 36,226
   were clean one-to-one re-keys, 31,069 merges (a new row whose two keys point to
   two prior rows) and 29,735 splits. Close.City saw each as a delete + add (18%
   of its routed rows that month).

Each `overture_id` and each `(osm_type, osm_id)` occurs at most once per release,
so ids can be carried forward by joining on source keys.

## 2. Findings that shaped the design

- **Guard.** `ghost_osm.names_match` (the CD name gate, threshold 70) on
  source-specific names: OSM renames failing it, 1,425 a month, are almost all real
  replacements (7-Eleven → Phillips 66). Overture renames failing it, 11,241 a
  month, are mostly rewording. So only OSM keys are guarded. A "name *or*
  category" guard does nothing: 97.5% of low-similarity renames keep the same
  `shared_label`.
- **Returning keys.** 39,627 Overture ids absent from 20261001 were back in
  20261022, so the registry remembers every key, not just the prior release's.
- **Overture dedup.** Dedup losers are dropped from the output, so a winner flip
  retires a published `overture_id`; 99,581 of September's 864,165 losers were
  published in October. `overture_dedup_dropped.parquet` (loser → winner) supplies
  alias edges.
- **Not affected.** The validator keys on its own `(osm_type, osm_id,
  overture_id)` string, and CD and calibration do not use `unified_id`.

## 3. Decisions (Nat, 2026-10-26)

Opaque ids, plus a one-time v0 → v1 migration table · `opoi_` + 16 base32 chars
of `blake2b(founding key)`, re-salted on collision · merge/split precedence: the
OSM side · rename guard on OSM keys only · registry keeps keys forever · publish
`ids/` (registry, change log) with every release · churn fixes limited to
deterministic tie-breaks · v1 folders complete · conflated PMTiles rebuilt for
10-22 only · `osm_id` cast to int64 in the backfill · `latest/` and the site move
to 2026-10-22-v1 · `osm_type` keeps `node|way|relation` · `overture_id` added to
tiles and popup · overrides CSV gains `osm_type, osm_id`.

## 4. Algorithm (`stable_ids.reconcile`)

Source keys: `osm:n123` / `osm:w123` / `osm:r123`, `ovt:<gers id>`.

Registry (`id_registry.parquet`): `source_key, unified_id, id_minted,
first_release, last_release, last_name`. One row per key ever seen, plus
**tombstones** (null `source_key`) for ids no key maps to any more.

1. **Edges** from each current row to registry ids: via its OSM key, via its
   Overture key, and via the key of any Overture dedup loser it beat (alias).
2. **Guard.** If an OSM edge's registry `last_name` and the current `osm_name`
   are both non-empty and `names_match` fails, the row may not take that id by any
   edge (`blocked`).
3. **Rank:** a key the id held on its most recent row before a key it held only
   earlier (`stale`); then OSM > Overture > alias; then older `id_minted`; then id
   and key strings.
4. **Greedy one-to-one** in rank order. Isolated edges are assigned in SQL; the
   contested ones (~1.2M edges on a heavy month) in Python.
5. **Mint** the rest: hash of the founding key (OSM key, else Overture key).
   Any id ever in the registry, tombstones included, counts as taken, so a retired
   id is never re-minted for another place.
6. **Registry update:** present keys → their row's id; absent keys kept;
   orphaned ids tombstoned.
7. **`id_changes.parquet`:** `minted`, `merged_into` (a prior id lost to a row
   that kept another id), `split_from` (a minted row whose prior id went
   elsewhere), `retired`, `revived`, `renamed_guard`.

Lifecycle examples:

- Overture-only X (id A) gains OSM node 9 → matched row keeps A; `osm:n9 → A`.
- Matched (n9, X) → (n9, Y) → (n7, Y): A throughout.
- Split of A = (n9, X): the OSM row keeps A; the Overture row mints (`split_from`).
- Merge of OSM-only n9 (B) and Overture-only X (A): the row keeps B;
  `A merged_into B`. If they split again, X mints a new id (A is a tombstone).
- Two Overture places clustered by dedup: the winner keeps its id; the loser's id
  is `merged_into` it through the alias edge.
- Node folded into a way by mappers: the node's id is `retired`, because no edge
  links them (TODO).
- Key absent one release and back the next: the registry still maps it, so the
  id is `revived`.

## 5. Pipeline integration

- `make conflate_baseline` runs `conflate.py`, then `assign_stable_ids.py`, which
  rewrites `conflated_baseline.parquet` in place, streaming record batches and
  matching rows by position. It then writes `id_registry.parquet`,
  `id_changes.parquet` and `id_stats.json` to `conflation/<version>/`. The target
  uses `pipefail`.
- `versions.id_registry` names the **prior** release; the stage refuses a registry
  whose `last_release` is not older than `versions.conflation`.
- `conflate.py`'s own ids are provisional (`matched:n123_<ovt>` / `osm:w123` /
  `overture:<ovt>`).
- Matching ties are broken by `source_id_order` (score, distance, then source-id
  order) in `select_best_matches` and in the cross-chunk dedup.
- The monthly publish uploads `ids/id_registry.parquet` and `ids/id_changes.parquet`.
  `openpois-remote.sh seed` / `prune_run.py` / `pull` carry the registry.
  `run_summary.md` reports the change counts beside last month's.

## 6. Backfill

`backfill_stable_ids.py` on openpois-01 replays the registry from 2026-04-23
through 2026-10-22, reading each `-v0` partitioned release anonymously from Source
Coop. Per release it writes the v1 parquet (only `unified_id` changed; `osm_id`
int64), the three id files, and the stats. Checks: row counts per file, a
per-column hash of every other column, id uniqueness, and every row's id against
its keys. `--apply-to` puts the 10-22 ids onto the canonical `conflated.parquet`
for the PMTiles rebuild. `publish_backfill_v1.py` uploads each v1 folder; the
other v0 objects are copied server-side, the >5 GB Overture PMTiles re-uploaded,
and v0 READMEs get a pointer line.

Results: see the CHANGELOG entry "Stable ids (v1 re-issue)".

## 7. Deferred (TODO.md)

Re-offer the next candidate to cross-chunk dedup losers; stabilise the Overture
dedup winner; link OSM element merges; wtm.ingest consuming `id_changes` and the
migration table.
