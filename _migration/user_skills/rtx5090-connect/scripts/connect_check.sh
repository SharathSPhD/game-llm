#!/usr/bin/env bash
# Usage: connect_check.sh [user@host]
# Verifies SSH reaches the target and it is genuinely the RTX 5090 box, then
# confirms Docker + the persistent training container are up.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

[ -n "${1:-}" ] && RTX5090_HOST="$1"

echo "== target: $RTX5090_HOST =="
verify_target_gpu || exit 1

echo "== docker =="
rtx_ssh "systemctl is-active docker" || { echo "FAIL: docker service not active on target" >&2; exit 1; }

echo "== train container =="
container_status=$(rtx_ssh "cd $RTX5090_COMPOSE_DIR && docker compose ps --format '{{.Name}} {{.State}}'" 2>/dev/null)
echo "$container_status"
if ! echo "$container_status" | grep -q "running"; then
    echo "WARN: train container not running -- starting it..." >&2
    rtx_ssh "cd $RTX5090_COMPOSE_DIR && docker compose up -d"
fi

echo "== gpu visible inside container =="
rtx_ssh "cd $RTX5090_COMPOSE_DIR && docker compose exec -T train nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader" || exit 1

echo "All checks passed. Target ready for jobs."
