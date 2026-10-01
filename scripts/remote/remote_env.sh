# shellcheck shell=bash
# Shared prelude for every command run on the remote pipeline host (openpois-01).
#
# Sourced by stage_runner.sh, and sent inline ahead of each command by
# openpois-remote.sh. Deliberately not a login shell: ~/.profile on this host starts
# a new ssh-agent on every login, which a polling loop would pile up. So this file
# does the two things the login shell was needed for: mount EFS and activate conda.

set -o pipefail

OPENPOIS_REPO=~/efs-mount/repos/OpenPOIs

if ! mountpoint -q ~/efs-mount; then
    sudo mount -t nfs \
        -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport \
        172.31.24.184:/ ~/efs-mount
fi

# shellcheck disable=SC1090
source ~/miniforge3/etc/profile.d/conda.sh
# The env is missing until `openpois-remote.sh setup` has run once.
conda activate openpois 2> /dev/null || true
# The env's libstdc++ must win over the system one: Ubuntu 22.04's stops at
# CXXABI_1.3.13, and scipy's compiled extensions need 1.3.15. Whichever copy an
# earlier import loads first is the one every later extension gets.
if [ "${CONDA_DEFAULT_ENV:-}" = openpois ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# GitHub access for `git fetch` without the login shell's agent.
export GIT_SSH_COMMAND="ssh -i ~/.ssh/id_github -o IdentitiesOnly=yes"

cd "$OPENPOIS_REPO" || exit 1
