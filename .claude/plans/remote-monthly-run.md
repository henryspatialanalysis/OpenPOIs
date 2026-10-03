# Plan: run the monthly OpenPOIs ingestion on openpois-01

Status: adopted 2026-09-30. The instance exists (§3.1); the tooling is
`scripts/remote/` (§3.3); the first run under it is October 2026 (§4). Decisions are
recorded in §6.

Goal: move the compute-heavy monthly stages (downloads, rating, conflation, change
detection, calibration, partitioning, PMTiles, publish) to the AWS instance `openpois-01`
(SSH alias `ec2-openpois`), coordinated from this machine over SSH/SFTP. The
validation round and the site bump stay local. Summary results come back to the local
`~/data/openpois/` tree in the same layout, so the existing skills and verify checks
keep working unchanged.

## 1. What the server survey found

Survey of `xl-01` (`ec2-wtm`), read-only, 2026-09-30. `openpois-01` is a root-disk
clone of it (§3.1), so the EFS, repo, Python and tool rows apply to both.

| Item | Finding | Consequence |
|---|---|---|
| Instance | `t3.xlarge`: 4 vCPU (burstable), 15 GiB RAM, us-west-2b | **Smaller than the WSL box** (12 cores, 24 GB cap). Conflation peaked at 21.9 GB RSS on 20260902. Needs a resize (§3.1). |
| Swap | 32 GB swapfile at `/var/lib/docker/.swap/swapfile` | A backstop, but running the 90-minute dedup phase in swap would take hours. |
| Disks | Root 128 GB (98 GB free); `/var/lib/docker` 80 GB (25 GB free) | **No 64 GB EBS volume is attached.** One has to be created, or the plan has to use a different volume (§3.2). |
| EFS | `172.31.24.184:/` mounted by `~/.profile` only | Non-interactive `ssh ec2-wtm 'cmd'` does not mount it. Every remote command runs under `bash -lc`. |
| Repo | `~/efs-mount/repos/OpenPOIs`, on `feature/ghost-closure-evidence`, clean, behind | Needs a fetch and checkout of the run branch. `openpois-validator` is also on EFS (not needed). |
| Python | miniforge3 with envs `sga`, `wtm`; **no `openpois` env** | Build it from `environment.yml` (§3.3). |
| Tools | `/usr/bin/osmium`, `aws`; no `source-coop`, no `tippecanoe` | osmium 1.19 and tippecanoe 2.79 come from the conda env. |
| IAM | Instance role `wtm-ec2-worker` | Overture reads are same-region S3. Source Coop writes use their own credentials (§3.5). |
| Load | Two local PostgreSQL clusters running; no containers; no crontab | Nothing else competes for memory now. Check `uptime`/`free` before each launch. |
| Old data | `~/data/openpois/census_areas` (84 MB), `~/data/openpois-validator/20260923` | `census_areas` can be moved onto the scratch volume. |

Hardcoded paths that matter: `config.yaml` puts every data directory under
`~/data/openpois/`, the calibration handoff under `~/repos/openpois/data/calibration/`,
and the Source Coop credential fallback at `~/repos/openpois/.env.json`. §3.3 handles
all three with symlinks, so **no config or code change is needed** to run remotely.

## 2. What runs where

| Stage | Where | Why |
|---|---|---|
| Next-run checklist, `config.yaml` version bumps, code changes, tests, commits | Local | Local stays the source of truth for the repo. The remote only pulls. |
| `compare_taxonomy.py --strict` pre-flight | Remote | Reads Overture from S3; same region. |
| OSM snapshot, Overture, incremental history downloads | Remote | Bandwidth-bound. us-west-2 has fast links to Geofabrik and is in-region for Overture. |
| Drift gate, `check_history`, `make rate`, `build_type_affinity.py` | Remote | Needs the snapshots that are already there. |
| `make conflate` (ghosts, baseline, CD, calibrate, overrides) | Remote | The memory problem this plan exists to solve. |
| `summarize.py`, HT review PDF, both `format_for_upload.py`, both PMTiles | Remote | Reads the 2.5 GB conflated file. |
| Bayesian fixed-rate mixture fit (checklist item 4) | Remote | JAX, about 50 minutes (`MODE=mixture` of `run_bayes_phase1.sh`). The inputs are already there. |
| **Validation round** (`openpois-validator`, LLM checks, human census, phone lane, review UI, handoff export) | **Local** | See the paragraph below. |
| Publish to Source Coop | Remote | 7 GB at in-region speed, versus about 1h40m from home last month. |
| Verify: DuckDB invariants on the output | Remote | Streams the parquet there, so nothing large is downloaded to run it. |
| Verify: fit report, figures, summaries, site | Local, on pulled results | Nat reads these. |
| `update-site` | Local | Needs the frontend dev server and a browser. |

**Why validation stays local.** Three reasons. The LLM checks are
dispatched as Claude Code agents from this machine, and the audit needs Nat in the loop
(review UI, phone lane, census of unverifiables). The gold labels are the "validation
moat", and keeping them off shared EFS keeps that separation simple. And the validator's
compute is small: `01_build_frame.py` reads only `conflated.parquet`
(`cfg.conflated_path`), so a single 2.5 GB download feeds it. The only heavy transfer is
that file coming down. The handoff going back up is about 200 KB.

## 3. One-time setup

Each step lists what it needs from Nat.

### 3.1 Instance: `openpois-01` (done 2026-09-30)

A t3 could not hold the conflation peak, and a burstable CPU drains its credits on a
multi-hour job. Instead of resizing xl-01, a new instance was cloned from its root disk:

| Item | Value |
|---|---|
| Instance | `i-07a5cc802480e3c36`, Name `openpois-01`, `r7i.2xlarge` (8 vCPU, 64 GiB, ~$0.53/h) |
| Image | `ami-0e7b2634aad4064ba` (`xl-01-clone-20260930`), root only, the 80 GB docker volume excluded. AMI and snapshot deleted after launch (2026-09-30). |
| Network | Same as xl-01: `subnet-063fde7809f0bc7ac` (us-west-2b), SGs `local-ssh-connect`, `efs-walkthrough1-ec2-sg`, `ec2-rds-1`, role `wtm-ec2-worker`, IMDSv2 required |
| Root disk | 128 GB **gp3** (xl-01's is gp2), `openpois-01-root` |
| First-boot fixes | docker-volume and old swapfile lines removed from fstab (backup `/etc/fstab.bak-xl01`); 16 GB `/swapfile` on root; PostgreSQL disabled (`systemctl enable --now postgresql` restores it) |
| Verified | Clean 10 s boot, both status checks ok, EFS mounts under `bash -lc`, miniforge envs present |

No Elastic IP, so the public DNS changes on every start. `~/bin/ec2-openpois`
(`start`, `stop`, `status`, or no argument to sync) finds the instance by its Name tag
and rewrites the `Host ec2-openpois` Hostname in both the WSL and Windows SSH configs.
The existing `~/bin/ec2` script still owns line 2 (`ec2-wtm`). Stop the instance between
runs: stopped, it costs ~$10 a month for the root disk.

### 3.2 Working storage: the root disk

After the swapfile, the root disk has ~81 GB free, and ~76 GB after the conda env (§3.3).
The runs work directly in `~/data/openpois/` on the root disk, with no separate volume.

Estimated peak footprint of one run:

| Item | GB |
|---|--:|
| Geofabrik PBFs (us + 3 territories) plus filtered copies during the snapshot build | ~12–15 |
| OSM snapshot dir (snapshot, pre-residential, rated, landuse, partitioned, PMTiles) | ~4 |
| Overture snapshot | ~1.2 |
| History: base (carry-forward) + new version + diffs | ~2 |
| Turnover model `20260727_by_shared_label` | 0.8 |
| Conflation dir (baseline, CD, canonical, partitioned, PMTiles, diagnostics) | ~14 |
| PMTiles FlatGeobuf intermediates, tippecanoe temp | ~5 (estimate, check) |
| Prior month's carry-forward (conflated.parquet, curves, Overture) | ~4 |
| **Total** | **~43–46** |

That leaves ~30 GB of slack. `scripts/osm_snapshot/download.py` already unlinks the raw
PBFs once the snapshot and landuse layer are built, and `openpois.io.pmtiles` deletes its
FlatGeobuf intermediates. Temp files land on the same disk, so no `TMPDIR` redirect is
needed. Check `df` after the downloads and after `make conflate` on the first run.

### 3.3 Tooling: `scripts/remote/`

| File | Runs | Role |
|---|---|---|
| `openpois-remote.sh` | local | The driver. Subcommands `status`, `setup`, `sync <branch>`, `seed`, `run <stage> -- <cmd>`, `watch <log> [s]`, `summary`, `pull <v> [--conflated]`, `push-handoff <round>`, `creds put\|clear`, `prune [--apply]`, `exec '<cmd>'`. Its header is the usage text. |
| `remote_env.sh` | remote | Prelude for every remote command: mounts EFS if needed, activates conda, sets a `GIT_SSH_COMMAND` for the GitHub key, `cd`s to the repo. |
| `stage_runner.sh` | remote | Runs one stage under `/usr/bin/time -v` into `~/data/openpois/logs/<stage>_<ts>.log`, between a START line (host, commit, `conflation=`) and a `=== STAGE <name> DONE rc=<n> ===` line. |
| `run_summary.py` | remote | `conflation/<v>/run_summary.md`: wall time and peak RSS per stage, row counts against the prior version, verdict lines, the drift-gate metrics, disk. |
| `prune_run.py` | remote | Deletes everything but the carry-forward. Dry run unless `--apply`; refuses `--apply` under WSL. |

**Remote commands never use a login shell.** `~/.profile` on the host starts a new
`ssh-agent` on every login (and is the only thing that mounts EFS), so a polling loop
through `bash -lc` would pile up agents. `remote_env.sh` does the EFS mount and the conda
activation itself; the driver sends it over stdin ahead of each command.

Layout on the remote, created by `setup`:

```
~/data/openpois/                             # working tree and carry-forward (root disk)
~/data/openpois/logs       -> ~/efs-mount/analysis/openpois/logs   # logs survive a prune
~/repos/openpois           -> ~/efs-mount/repos/OpenPOIs           # handoff + .env.json paths
```

With these links `config.yaml` resolves unchanged. `setup` builds the `openpois` env from
`environment.yml` minus its `openpois==0.0.0` pip line (not on PyPI), installs the
checkout editable, and runs the tests, as a detached stage. The expected result is the 5
known stale-mock failures, plus the ZIE test's ~1-in-12 flake.

### 3.4 Carry-forward

What next month's run reads from this month's stays on the remote root disk between runs
(the disk is billed whether or not the instance runs):

| Kept | Needed by |
|---|---|
| `osm_data/<v>/` | incremental history `base_version` |
| `snapshots/overture/<v>/overture_snapshot.parquet` + `viz/` | `compare_confidence.py` (prior month) |
| `conflation/<v>/conflated.parquet`, `calibration/`, Bayes eval dirs, CSVs, `run_summary.md` | `build_type_affinity.py --conflated`; curve reuse on a gate pass |
| `ghost_osm/<v>/`, `osm_turnover_model/<model_output>/`, `boundary/`, `census_areas/` | ghosts (tiny), `make rate`, clipping |

`seed` uploads the September set once, reading the versions from the local
`config.yaml`, so run it before the October bumps; it also pushes the
`versions.calibration` handoff. `prune --apply` after each publish removes every other
version dir and the rebuilt or published files inside the kept conflation dir (baseline,
CD, partitioned tree, PMTiles, match diagnostics). There is no manual-overrides CSV yet
(the Close lane has not written one), so that stage is a logged no-op.

### 3.5 Source Coop credentials

Nat runs `source-coop login` locally, as before. `creds put` mints `source-coop creds`
locally and writes the four `REQUIRED_KEYS` of `openpois.io.credentials` to
`~/repos/openpois/.env.json` on the remote (mode 0600; gitignored). The remote has no
`source-coop` binary, so the loader falls through to that file. An in-region 7 GB upload
fits well inside the 1-hour token. `creds clear` deletes the file after the publish.

## 4. Monthly run procedure

**October 2026 resume point:** the run is paused for the validation round; pick it up
with [2026-10-run-resume.md](2026-10-run-resume.md), which has the state, the next
commands and this run's lessons.

Each long stage is launched with `openpois-remote.sh run <stage> -- <cmd>` and followed
by one Monitor running `openpois-remote.sh watch <log>` (30-minute heartbeat for
hours-long stages). The watch ends on the DONE line (which carries the exit code, also
after an OOM kill), on the runner vanishing without one, or on three failed SSH polls;
new error lines are reported without ending it. Gates marked **Nat** pause for review.
Stop the instance (`ec2-openpois stop`) whenever the next step waits on a person.

1. **Local.** Raise the next-run checklist in `.claude/TODO.md` with Nat. Cut
   `run/YYYY-MM` from main (October 2026: run on `feature/matched-index-modes` itself,
   so mid-run fixes land in the same PR); make the version bumps from
   `full-data-pull` step 1 and `conflate-snapshots` step 1; commit and push.
   `ec2-openpois start`, then `openpois-remote.sh sync <branch>` and `status`.
2. **Pre-flight.** `compare_taxonomy.py --strict` and `make download_history PLAN=1`.
   **Nat.**
3. **Downloads.** OSM snapshot and history in parallel, then Overture; then
   `compare_confidence.py` and `make check_history`. **Nat:** counts against the
   baselines, the drift verdict, check_history PASS.
4. **Rate and conflate.** `make rate`; `build_type_affinity.py --conflated <prior
   conflated.parquet>`; then `make conflate_to_cd` in a validation month (it stops
   after change detection, so no provisional calibration runs) or `make conflate` in
   any other month. Then `summary`.
5. **Pull.** `pull <v>`, with `--cd` in a validation month for the validator's input
   (`conflated_cd.parquet`). **Nat** reviews the summaries and `run_summary.md`.
6. **Validation month only** (October 2026 holds the publish for it):
   - locally, the `openpois-validator` round draws from the pulled
     `conflated_cd.parquet` (validator `conflated_file: conflated_cd.parquet`; its
     matched `conf_mean` is the uncalibrated blend the frame expects) and exports the
     handoff (Nat's go for the dispatch cost);
   - `push-handoff <round>`; bump `versions.calibration` and set `pooled_rounds` on the
     branch, push, `sync`.
7. **Calibrate.** `make calibrate`: the three Bayesian mixture fits (1-D Overture, 1-D
   OSM, 2-D matched, in parallel), the export to grid curves, `apply_calibration`, the
   plots and the HT review. **Stop point:** the export refuses to write curves when any
   segment fails the §5.1 acceptance rule, and Nat decides. Then
   `make apply_manual_overrides`, `summarize.py`, `summary`, the calibration invariants
   from `verify-pipeline-run` via `exec`, and `pull <v>`. **Nat** reviews the fit
   report, the HT PDF and the summaries, and gives the release decision.
8. **Package.** Both `format_for_upload.py` runs (`scripts/conflation/` and
   `scripts/osm_snapshot/`; the upload reads both partitioned datasets), then the three
   `prepare_pmtiles.py` runs (`osm_snapshot`, `conflation`, `overture`).
9. **Publish.** The upload `--dry-run` first: it needs no credentials, so missing files
   show up before the token clock starts. Then Nat's local `source-coop login` (on
   2026-10-01 the cached token had expired within an hour, so log in just before);
   `creds put`; the
   upload, the published-release checks; `creds clear`. The `latest/` mirror
   re-uploads the run's PMTiles from disk (the proxy has no server-side copy over
   5 GB), so the transfer is about twice the archive total.
10. **Wrap up.** `update-site` locally; TODO.md bookkeeping; PR the run branch to main
    (Nat approves); `prune`, check the list, `prune --apply`; `ec2-openpois stop`.

## 5. What comes back to the laptop

`pull <v>` rsyncs into the same relative paths under local `~/data/openpois/`, so local
skills and verify snippets read them unchanged:

- `conflation/<v>/calibration/` (fit report, curves, metadata, CSVs,
  `ht_review_<round>.pdf`), `viz/`, `summary_by_label.csv`, `match_status_by_label.csv`,
  `config.yaml`, `run_summary.md`, and the Bayes eval dirs minus `*draws*.parquet`;
- `osm_data/*/history_coverage.json` and `snapshots/overture/*/viz/`;
- all stage logs (from EFS);
- with `--conflated`, `conflation/<v>/conflated.parquet`.

Local keeps these summaries plus `conflated.parquet`. The full version dirs are no longer
mirrored locally: the published release on Source Coop is the archive.

## 6. Decisions (2026-09-30)

1. New instance `openpois-01` (r7i.2xlarge), no Elastic IP, stopped between runs.
2. Working storage on the 128 GB root disk; no separate volume.
3. Carry-forward on the root disk, not EFS.
4. October holds the publish for the new validation round.
5. Each month runs on a `run/YYYY-MM` branch cut from main and PR'd at the end. October
   2026 runs on `feature/matched-index-modes` itself, here and on the remote, so the
   PR carries the new code and any mid-run fixes together.
6. Local keeps tier-1 summaries plus `conflated.parquet`.
7. No manual-overrides CSV exists; the stage runs as a no-op until the Close lane writes
   one.
8. Published calibration is three Bayesian monotone-spline models with the fixed-rate
   mixture label layer (1-D Overture, 1-D OSM, 2-D matched), fit on the remote; v4 is
   retired. No provisional calibration in a validation month.
9. Source Coop credentials are minted locally and copied over for the publish.
10. Stages are launched one at a time for now; revisit a single `run_month.sh` after the
    first supervised run.
