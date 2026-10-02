# Resume the October 2026 run: from the validation handoff to release

State as of 2026-10-01 05:40 UTC. You are picking up the October OpenPOIs monthly run
partway through. Read this page, then `.claude/plans/remote-monthly-run.md` (the
procedure, §4 steps 7–10 are what remains) and `.claude/docs/confidence-calibration.md`
(the calibration method, new this month). Follow Nat's waiting rules
(`~/.claude/docs/waiting-antipatterns.md`): launch long stages detached, watch them with
one Monitor, and end your turn while they run.

## Where things stand

- **Done on openpois-01** (all on branch `feature/matched-index-modes`, pushed; the
  remote checkout is at the same commit):
  - Taxonomy pre-flight (pass).
  - OSM snapshot 20261001 (4,535,124 rows).
  - History roll-forward 20261001 (`check_history` 99.906%, pass).
  - Overture snapshot 20260923 (15,003,463 rows).
  - Drift gate (pass).
  - `make rate`.
  - Type-affinity rebuild (committed).
  - `make conflate_to_cd`: `conflation/20261001/conflated_cd.parquet` holds 16,591,754
    rows, with 25,412 shadow-matched.
  - `run_summary.md`, pulled to `~/data/openpois/conflation/20261001/`.
- **Waiting on:** the validation round `20261001`, run by another agent in
  `~/repos/openpois-validator` (its brief:
  `~/repos/openpois-validator/.claude/plans/2026-10-validation-round-handoff.md`). It
  draws from the **July** population. When it is done, Nat gives you the handoff path,
  normally `~/repos/openpois/data/calibration/20261001/`.
- **openpois-01 is stopped.** Its disk keeps the data, the env and the checkout.

## Tools

| What | How |
|---|---|
| Instance | `openpois-01`, `i-07a5cc802480e3c36`, r7i.2xlarge (8 vCPU, 61 GiB). `~/bin/ec2-openpois start\|stop\|status` (start also rewrites the `ec2-openpois` SSH hostname, which changes on every start). Needs a live `aws login` (Nat runs it; `aws sts get-caller-identity` checks it). |
| Remote driver | `bash scripts/remote/openpois-remote.sh <cmd>` from `~/repos/openpois`: `status`, `sync <branch>`, `run <stage> -- <cmd>` (detached; prints the log path), `watch <log> [s]` (heartbeat for a Monitor), `summary`, `pull <v> [--conflated] [--cd]`, `push-handoff <round>`, `creds put\|clear`, `prune [--apply]`, `exec '<cmd>'`. Its header is the usage text. |
| Code changes | Edit and commit locally, push the feature branch, then `sync feature/matched-index-modes`. Never edit on the remote. Nat approved committing and pushing mid-run fixes to this branch; they go into the end-of-run PR. |

## Next steps

1. **Handoff in.** Check that it exists and that its `metadata.json` says round
   `20261001`. Then `openpois-remote.sh push-handoff 20261001`; the remote must be
   running, so start it first.
2. **Config.** Set `versions.calibration: "20261001"` and
   `conflation.calibration.pooled_rounds: ["20260730"]`. Commit, push, then `sync`.
3. **Calibrate.** `run calibrate -- make calibrate`, then watch it (15-minute
   heartbeat).
   - It runs three parallel Bayesian fits (OSM and matched at target-accept 0.99, from
     `conflation.calibration.bayes`), then the export, `apply_calibration`, the plots and
     the HT review.
   - On the July round alone the fits took about 3, 4 and 30 minutes; pooled rounds will
     be slower.
   - **Stop point:** if the export exits 2, a segment failed the acceptance rule. Report
     the failing items to Nat and do not work around it.
4. **Then:**
   - `run overrides -- make apply_manual_overrides` (a logged no-op: no CSV exists yet);
   - `run summarize -- python -u scripts/conflation/summarize.py`;
   - `summary`;
   - the calibration invariants from `.claude/skills/verify-pipeline-run/SKILL.md`, via
     `exec` (DuckDB, streamed);
   - `pull 20261001 --conflated`.
5. **Review gate.** Nat reads `calibration/fit_report.md`,
   `calibration/ht_review_20261001.pdf` and `run_summary.md`, and gives the release
   decision.
6. **After Nat's go:**
   - Package: both `format_for_upload.py` (conflation and osm_snapshot), then both
     `prepare_pmtiles.py`.
   - Publish: Nat runs `source-coop login` locally; then `creds put`, the upload
     `--dry-run`, the upload, the published-release checks, and `creds clear`.
7. **Wrap up:**
   - `update-site` locally;
   - TODO.md bookkeeping;
   - PR `feature/matched-index-modes` to main (Nat approves);
   - `prune`, check the list, `prune --apply`;
   - `ec2-openpois stop`.

## Open items to raise with Nat before release

- **Change-detection release gate** (CHANGELOG, "same-entity ghosts"): at least 70%
  precision on a vetted sample of at least 100 demoted POIs. Not yet measured.
- **Overture alternates.** Since 2026-09-23.1, alternate categories come from
  `taxonomy.alternates`, which only 0.3% of rows carry, so the published
  `overture_categories_alternate` column is now mostly null. Matching never read it.
- **`site/public/about.html` band prose** needs re-reading against the Bayesian values
  at `update-site` (TODO checklist item 10).

## Lessons from this run (already handled; know them if something recurs)

- **Remote shells are not login shells.** `~/.profile` there starts an ssh-agent on every
  login. `scripts/remote/remote_env.sh` mounts EFS, activates conda and preloads the
  env's `libstdc++` (the system one is too old for scipy). Don't put the env's `lib/` on
  `LD_LIBRARY_PATH`: it swaps OpenSSL and breaks ssh for `git fetch`.
- **Geofabrik `-latest` aliases looped** (301 to themselves). `download_pbf` now resolves
  them to the newest dated extract (`resolve_geofabrik_latest`).
- **Overture removed `categories`** in 2026-09-23.1. Alternates now come from
  `taxonomy.alternates`. Check the release schema when a download fails in DuckDB's
  binder.
- **The type-affinity rebuild writes a tracked, gitignored file**
  (`src/openpois/conflation/data/type_affinity.csv`). Run it on the remote, `scp` it
  back, `git add -f` and commit it locally, `git checkout --` it on the remote, push,
  then `sync`.
- **Sampler settings.** Fit separately, OSM and matched diverge at the default 0.80;
  0.99 passes. More draws at 0.95 did not fix matched.
- **The Overture finalize step** (DuckDB polygon filter) took 74 minutes while sharing
  CPUs with other stages. It is CPU-bound at 4 threads; raising it is a follow-up, not
  done.
- **`run_summary.md` quirks:**
  - Superseded failed attempts still show as "Failed".
  - Prior-version comparisons are blank for files the remote doesn't keep.
  - A follow-up fix is not done.
- **Monitors expire after 30 minutes.** Re-arm `watch` on the same log, and don't
  stack watchers.
