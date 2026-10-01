#!/usr/bin/env bash
# Remote command strings are single-quoted on purpose: they expand on the remote host.
# shellcheck disable=SC2016
# Drive the monthly pipeline on the remote host (openpois-01, SSH alias ec2-openpois)
# from this machine. Every remote command runs in a plain, non-login bash that first
# sources scripts/remote/remote_env.sh (mount EFS, activate conda, cd to the repo):
# a login shell there starts a new ssh-agent each time. Start and stop the instance
# with ~/bin/ec2-openpois; the procedure is in .claude/plans/remote-monthly-run.md.
#
#   openpois-remote.sh status                  host load, disk, running stages
#   openpois-remote.sh setup                   symlinks; build the conda env + tests
#                                              (detached; prints the log to watch)
#   openpois-remote.sh sync <branch>           fetch, check out, fast-forward
#   openpois-remote.sh seed                    upload last month's carry-forward
#                                              (versions read from local config.yaml)
#   openpois-remote.sh run <stage> -- <cmd>    launch a detached stage; prints its log
#   openpois-remote.sh watch <log> [seconds]   heartbeat lines until DONE or exit
#   openpois-remote.sh summary [version]       write conflation/<v>/run_summary.md
#   openpois-remote.sh pull <version> [--conflated] [--cd]
#                                              tier-1 results (+ conflated.parquet,
#                                              + conflated_cd.parquet)
#   openpois-remote.sh push-handoff <round>    upload data/calibration/<round>/
#   openpois-remote.sh creds put|clear         Source Coop credentials for publish
#   openpois-remote.sh prune [--apply]         delete all but the carry-forward
#   openpois-remote.sh exec '<command>'        one-off command in the remote env
set -euo pipefail

HOST=${OPENPOIS_REMOTE_HOST:-ec2-openpois}
HERE=$(cd "$(dirname "$0")" && pwd)
LOCAL_REPO=$(cd "$HERE/../.." && pwd)
LOCAL_DATA=~/data/openpois
RSYNC=(rsync -a --partial -e "ssh -q")

die() { echo "error: $*" >&2; exit 1; }

# Run a command string on the remote host after the shared prelude. The script goes
# over stdin, so nothing in it shows up in a remote process listing.
remote() {
    ssh -q "$HOST" bash -s <<< "$(cat "$HERE/remote_env.sh")"$'\n'"$1"
}

# Launch <command string> as a detached stage; print the remote log path.
launch() {
    local stage=$1 cmd=$2 log
    log="\$HOME/data/openpois/logs/${stage}_$(date -u +%Y%m%d_%H%M%S).log"
    remote "
        log=$log
        setsid nohup bash scripts/remote/stage_runner.sh \"\$log\" $(printf %q "$stage") \
            $(printf %q "$cmd") < /dev/null > /dev/null 2>&1 &
        sleep 2
        echo \"\$log\""
}

local_version() {
    python3 -c "import sys, yaml
print(yaml.safe_load(open('$LOCAL_REPO/config.yaml'))['versions'][sys.argv[1]])" "$1"
}

cmd_status() {
    remote '
        echo "host: $(hostname)  $(uptime)"
        free -g | head -2
        df -h / | tail -1
        echo "branch: $(git rev-parse --abbrev-ref HEAD) @ $(git log -1 --oneline)"
        echo "running stages:"
        pgrep -af "[s]tage_runner.sh" || echo "  none"'
}

cmd_setup() {
    remote '
        mkdir -p ~/repos ~/data/openpois ~/efs-mount/analysis/openpois/logs
        [ -e ~/repos/openpois ] || ln -s ~/efs-mount/repos/OpenPOIs ~/repos/openpois
        if [ -d ~/data/openpois/logs ] && [ ! -L ~/data/openpois/logs ]; then
            echo "~/data/openpois/logs is a real directory; move its contents first" >&2
            exit 1
        fi
        [ -L ~/data/openpois/logs ] \
            || ln -s ~/efs-mount/analysis/openpois/logs ~/data/openpois/logs
        ls -ld ~/repos/openpois ~/data/openpois/logs'
    # environment.yml pins the package itself (openpois==0.0.0), which is not on
    # PyPI: drop that line and install the checkout editable instead.
    launch setup '
        if ! conda env list | grep -q "^openpois "; then
            sed "/^ *- openpois==/d" environment.yml > /tmp/openpois-environment.yml
            ~/miniforge3/bin/mamba env create -f /tmp/openpois-environment.yml \
                || exit 1
        fi
        conda activate openpois
        pip install --no-deps -e ~/repos/openpois || exit 1
        python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -20'
}

cmd_sync() {
    local branch=${1:?usage: sync <branch>}
    remote "
        if [ -n \"\$(git status --porcelain --untracked-files=no)\" ]; then
            git status --short
            echo 'remote tree is dirty; refusing to switch' >&2
            exit 1
        fi
        git fetch --prune origin
        git checkout $(printf %q "$branch")
        git pull --ff-only
        git log -1 --oneline"
}

cmd_seed() {
    local osm overture model conflation calibration
    osm=$(local_version osm_data)
    overture=$(local_version snapshot_overture)
    model=$(local_version model_output)
    conflation=$(local_version conflation)
    calibration=$(local_version calibration)
    echo "seeding: osm_data $osm, overture $overture, model $model," \
        "conflation $conflation, calibration $calibration"
    "${RSYNC[@]}" -R --info=progress2 \
        "$LOCAL_DATA/./osm_data/$osm" \
        "$LOCAL_DATA/./snapshots/overture/$overture/overture_snapshot.parquet" \
        "$LOCAL_DATA/./osm_turnover_model/$model" \
        "$LOCAL_DATA/./conflation/$conflation/conflated.parquet" \
        "$LOCAL_DATA/./conflation/$conflation/calibration" \
        "$LOCAL_DATA/./boundary" \
        "$LOCAL_DATA/./census_areas" \
        "$HOST:data/openpois/"
    cmd_push_handoff "$calibration"
}

cmd_run() {
    local stage=${1:?usage: run <stage> -- <command...>}
    shift
    [[ ${1:-} == "--" ]] && shift
    [[ $# -gt 0 ]] || die "no command given"
    [[ $stage =~ ^[A-Za-z0-9_.-]+$ ]] || die "stage names are [A-Za-z0-9_.-]"
    launch "$stage" "$*"
}

# Heartbeat for a Monitor: one line per interval. Ends on the runner's DONE line,
# on the runner exiting without one, or after three failed SSH polls. New error
# lines are reported as they appear but do not end the watch, since the DONE line
# carries the exit code either way.
cmd_watch() {
    local log=${1:?usage: watch <log> [seconds]} interval=${2:-1800}
    local out state errors finished last seen=0 ssh_fails=0 polls=0
    while true; do
        polls=$((polls + 1))
        if out=$(remote "
            f=$(printf %q "$log")
            [ -f \"\$f\" ] || { echo 'NOLOG|0||'; exit 0; }
            pgrep -f \"[s]tage_runner.sh \$f\" > /dev/null && up=up || up=DOWN
            n=\$(grep -cE 'Traceback|Killed|MemoryError|FAILED|Error:' \"\$f\" || true)
            fin=\$(grep -m1 -E '^=== STAGE .* DONE' \"\$f\" || true)
            last=\$(grep -vE '^\s|^$' \"\$f\" | tail -1 | cut -c1-160)
            echo \"\$up|\$n|\$fin|\$last\""); then
            ssh_fails=0
        else
            ssh_fails=$((ssh_fails + 1))
            echo "[$(date -u +%H:%M)] ssh failed ($ssh_fails/3)"
            [[ $ssh_fails -ge 3 ]] && { echo "host unreachable; stopping watch"; break; }
            sleep 60
            continue
        fi
        IFS='|' read -r state errors finished last <<< "$out"
        if [[ $errors -gt $seen ]]; then
            echo "[$(date -u +%H:%M)] $((errors - seen)) new error line(s): see $log"
            seen=$errors
        fi
        if [[ -n $finished ]]; then
            echo "[$(date -u +%H:%M)] $finished"
            break
        fi
        if [[ $state == DOWN ]]; then
            echo "[$(date -u +%H:%M)] runner exited without a DONE line: $last"
            break
        fi
        if [[ $state == NOLOG && $polls -ge 3 ]]; then
            echo "[$(date -u +%H:%M)] log never appeared: $log"
            break
        fi
        echo "[$(date -u +%H:%M)] $state | $last"
        sleep "$interval"
    done
}

cmd_summary() {
    remote "python scripts/remote/run_summary.py ${1:+--version $(printf %q "$1")}"
}

cmd_pull() {
    local version=${1:?usage: pull <conflation version> [--conflated] [--cd]}
    shift
    local base="$HOST:data/openpois/."
    local sources=(
        "$base/conflation/$version/calibration"
        "$base/conflation/$version/viz"
        "$base/conflation/$version/summary_by_label.csv"
        "$base/conflation/$version/match_status_by_label.csv"
        "$base/conflation/$version/config.yaml"
        "$base/conflation/$version/run_summary.md"
        "$base/conflation/$version/calibration_bayes"
        "$base/conflation/$version/calibration_eval_bayes_*"
        "$base/osm_data/*/history_coverage.json"
        "$base/snapshots/overture/*/viz"
    )
    local flag
    for flag in "$@"; do
        case $flag in
            --conflated) sources+=("$base/conflation/$version/conflated.parquet") ;;
            # The validation frame's input in a month whose calibration waits
            # for the round (make conflate_to_cd).
            --cd) sources+=("$base/conflation/$version/conflated_cd.parquet") ;;
            *) die "usage: pull <conflation version> [--conflated] [--cd]" ;;
        esac
    done
    "${RSYNC[@]}" -R --ignore-missing-args --exclude '*draws*.parquet' \
        --info=progress2 "${sources[@]}" "$LOCAL_DATA/"
    "${RSYNC[@]}" "$HOST:data/openpois/logs/" "$LOCAL_DATA/logs/"
    echo "pulled into $LOCAL_DATA/conflation/$version/"
}

cmd_push_handoff() {
    local round=${1:?usage: push-handoff <round>}
    [[ -d $LOCAL_REPO/data/calibration/$round ]] \
        || die "no handoff at $LOCAL_REPO/data/calibration/$round"
    remote "mkdir -p data/calibration"
    "${RSYNC[@]}" "$LOCAL_REPO/data/calibration/$round" \
        "$HOST:efs-mount/repos/OpenPOIs/data/calibration/"
    echo "pushed handoff $round"
}

# Source Coop credentials: minted here (Nat runs `source-coop login` locally first),
# written to the remote .env.json fallback that openpois.io.credentials reads when no
# CLI is installed, and deleted after the publish.
cmd_creds() {
    case ${1:-} in
        put)
            local cli payload
            cli=$(command -v source-coop || echo ~/.cargo/bin/source-coop)
            payload=$("$cli" creds | python3 -c '
import base64, json, sys
c = json.load(sys.stdin)
print(c.get("Expiration"), file = sys.stderr)
env = {
    "aws_access_key_id": c["AccessKeyId"],
    "aws_secret_access_key": c["SecretAccessKey"],
    "aws_session_token": c["SessionToken"],
    "region_name": "us-west-2",
}
print(base64.b64encode(json.dumps(env).encode()).decode())') \
                || die "source-coop creds failed; run 'source-coop login' first"
            remote "umask 077; echo $payload | base64 -d > ~/repos/openpois/.env.json
                    ls -l ~/repos/openpois/.env.json"
            echo "(the line above the listing is the token expiry)"
            ;;
        clear)
            remote "rm -f ~/repos/openpois/.env.json && echo 'removed remote .env.json'"
            ;;
        *) die "usage: creds put|clear" ;;
    esac
}

cmd_prune() {
    remote "python scripts/remote/prune_run.py ${1:-}"
}

main() {
    local sub=${1:-}
    shift || true
    case $sub in
        status) cmd_status ;;
        setup) cmd_setup ;;
        sync) cmd_sync "$@" ;;
        seed) cmd_seed ;;
        run) cmd_run "$@" ;;
        watch) cmd_watch "$@" ;;
        summary) cmd_summary "$@" ;;
        pull) cmd_pull "$@" ;;
        push-handoff) cmd_push_handoff "$@" ;;
        creds) cmd_creds "$@" ;;
        prune) cmd_prune "$@" ;;
        exec) remote "$*" ;;
        *) sed -n '4,26p' "$0"; exit 2 ;;
    esac
}

main "$@"
