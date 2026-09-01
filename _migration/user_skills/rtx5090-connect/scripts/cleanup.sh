#!/usr/bin/env bash
# Usage: cleanup.sh <remote_job_name> [user@host]
# Removes a job's directory from the target's persistent project tree.
# Destructive -- asks for confirmation unless CONFIRM=yes is set (e.g. for
# non-interactive/agent use after results have already been fetched).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

JOB_NAME="${1:?usage: cleanup.sh <remote_job_name> [user@host]}"
[ -n "${2:-}" ] && RTX5090_HOST="$2"

if [ "${CONFIRM:-}" != "yes" ]; then
    echo "This will delete $RTX5090_HOST:$RTX5090_PROJECT_DIR/$JOB_NAME permanently."
    echo "Make sure fetch_results.sh has already been run if you need the outputs."
    read -r -p "Type 'yes' to proceed: " ans
    [ "$ans" = "yes" ] || { echo "Aborted."; exit 1; }
fi


# Jobs run as root inside the container (see Dockerfile), so files/dirs they
# create (checkpoints/, etc.) end up root-owned on the host bind mount --
# the host's non-root user can't delete them directly (Permission denied on
# the containing dir). Delete via the container instead, where root actually
# has permission.
rtx_ssh "cd $RTX5090_COMPOSE_DIR && docker compose exec -T train rm -rf ${RTX5090_CONTAINER_PROJECT_DIR:?}/$JOB_NAME"
echo "Removed $JOB_NAME from target."
