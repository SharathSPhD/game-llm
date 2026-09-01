#!/usr/bin/env bash
# Usage: smoke_test.sh [steps] [user@host]
# Runs the known-good GPU stability check on the target: a small transformer
# trained for N steps. Use before trusting a long real job to the box, or any
# time something feels off (a prior job hung, connection was flaky, etc).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_common.sh

STEPS="${1:-50}"
[ -n "${2:-}" ] && RTX5090_HOST="$2"

verify_target_gpu || exit 1

echo "== running smoke test ($STEPS steps) =="
rtx_ssh "cd $RTX5090_COMPOSE_DIR && docker compose exec -T train python $RTX5090_CONTAINER_PROJECT_DIR/gpu-smoke-test/train.py --steps $STEPS"
status=$?

if [ $status -ne 0 ]; then
    echo "FAIL: smoke test exited non-zero ($status) -- do not trust a long job to this box right now." >&2
    exit $status
fi
echo "Smoke test passed."
