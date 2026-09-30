#!/usr/bin/env bash
# Run one pipeline stage on the remote host, detached from the SSH session that
# launched it. Launched by `openpois-remote.sh run`; not meant to be called by hand.
#
#   stage_runner.sh <log path> <stage name> <command...>
#
# The command runs under `/usr/bin/time -v`, so the log records wall time and peak
# RSS (ru_maxrss over every child, which covers each step of a make chain). The log
# opens with a START line carrying the host, the commit and versions.conflation, and
# always closes with `=== STAGE <name> DONE rc=<n> ===`, including after an OOM kill
# of the Python process: the heartbeat in `openpois-remote.sh watch` keys on it.
set -u

log=$1
stage=$2
shift 2

env_file="$(dirname "$(readlink -f "$0")")/remote_env.sh"
# shellcheck source=scripts/remote/remote_env.sh
source "$env_file"

mkdir -p "$(dirname "$log")"
conflation=$(python -c \
    "import yaml; print(yaml.safe_load(open('config.yaml'))['versions']['conflation'])" \
    2> /dev/null || echo unknown)
{
    echo "=== STAGE $stage START $(date -u +%FT%TZ) host=$(hostname)" \
        "commit=$(git rev-parse --short HEAD) conflation=$conflation ==="
    echo "cmd: $*"
    # The child re-sources the prelude: shell functions such as `conda activate` do
    # not cross into a new bash.
    /usr/bin/time -v bash -c "source $(printf %q "$env_file"); $*"
    rc=$?
    echo "=== STAGE $stage DONE rc=$rc $(date -u +%FT%TZ) ==="
} >> "$log" 2>&1
