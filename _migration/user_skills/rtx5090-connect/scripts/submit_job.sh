#!/usr/bin/env bash
# Usage: submit_job.sh <local_job_dir> <remote_job_name> [entrypoint.py] [user@host]
#
# Copies a local job directory to the target's persistent project tree and
# launches it detached inside the always-on `train` container. Logs land in
# <remote_job_name>/train.log on the target, readable while the job runs.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

LOCAL_DIR="${1:?usage: submit_job.sh <local_job_dir> <remote_job_name> [entrypoint.py] [user@host]}"
JOB_NAME="${2:?usage: submit_job.sh <local_job_dir> <remote_job_name> [entrypoint.py] [user@host]}"
ENTRYPOINT="${3:-train.py}"
[ -n "${4:-}" ] && RTX5090_HOST="$4"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "FAIL: local dir '$LOCAL_DIR' does not exist" >&2
    exit 1
fi

verify_target_gpu || exit 1

echo "== copying $LOCAL_DIR -> $RTX5090_HOST:$RTX5090_PROJECT_DIR/$JOB_NAME =="
rsync -avz --progress "$LOCAL_DIR/" "$RTX5090_HOST:$RTX5090_PROJECT_DIR/$JOB_NAME/"

echo "== launching detached inside train container =="
rtx_ssh "cd $RTX5090_COMPOSE_DIR && docker compose exec -d train bash -c \
  'cd $RTX5090_CONTAINER_PROJECT_DIR/$JOB_NAME && python $ENTRYPOINT > train.log 2>&1'"

sleep 3
echo "== confirming it started =="
rtx_ssh "test -f $RTX5090_PROJECT_DIR/$JOB_NAME/train.log && tail -n 20 $RTX5090_PROJECT_DIR/$JOB_NAME/train.log" \
  || echo "WARN: no train.log yet -- it may still be starting up (large imports, model download, etc). Check again shortly with status.sh."

echo "Job '$JOB_NAME' launched. Monitor with: status.sh $JOB_NAME"
