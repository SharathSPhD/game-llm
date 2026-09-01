#!/usr/bin/env bash
# Usage: fetch_results.sh <remote_job_name> <local_dest_dir> [remote_subpath] [user@host]
# Pulls a job's outputs back to the DGX Spark. Defaults to the job's
# checkpoints/ subdirectory; pass a different remote_subpath (e.g. "." for
# the whole job dir) if the job doesn't follow that convention.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

JOB_NAME="${1:?usage: fetch_results.sh <remote_job_name> <local_dest_dir> [remote_subpath] [user@host]}"
LOCAL_DEST="${2:?usage: fetch_results.sh <remote_job_name> <local_dest_dir> [remote_subpath] [user@host]}"
REMOTE_SUBPATH="${3:-checkpoints}"
[ -n "${4:-}" ] && RTX5090_HOST="$4"

mkdir -p "$LOCAL_DEST"
echo "== pulling $RTX5090_HOST:$RTX5090_PROJECT_DIR/$JOB_NAME/$REMOTE_SUBPATH -> $LOCAL_DEST =="
rsync -avz --progress "$RTX5090_HOST:$RTX5090_PROJECT_DIR/$JOB_NAME/$REMOTE_SUBPATH/" "$LOCAL_DEST/"
echo "Done."
