#!/usr/bin/env bash
# SPEC 0022 launcher: stages the code + pack and submits a job to the 5090.
#
# Usage:
#   launch.sh preflight <pack_dir>          # both arms, 12 measured steps each
#   launch.sh phase1    <pack_dir>          # Arm E then Arm T, 2.5B tokens each
#   launch.sh extend    <pack_dir>          # Arm T alone 2.5B -> 10B with decay
#   launch.sh sync-pack <pack_dir>          # rsync a pack only (idempotent)
#
# The job directory on the target is fusion-project/kinetic-twin; results land
# under kinetic-twin/results/scale/exp39/<arm>/ and are fetched with the
# rtx5090-connect fetch_results.sh script. Wrapping the proven skill scripts
# rather than reimplementing ssh/docker keeps the wrong-machine guard.
set -euo pipefail

MODE="${1:?usage: launch.sh preflight|phase1|extend|interventions|sync-pack <pack_dir>}"
PACK_DIR="${2:?give the local pack directory}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKILL="${RTX5090_SKILL_DIR:-$HOME/.claude/skills/rtx5090-connect/scripts}"
HOST="${RTX5090_HOST:-ss@192.168.0.204}"
JOB=kinetic-twin
STAGE_DIR="$(mktemp -d /tmp/twin5090.XXXXXX)"
trap 'rm -rf "$STAGE_DIR"' EXIT

PACK_BASE="$(basename "$PACK_DIR")"

mkdir -p "$STAGE_DIR/experiments"
cp "$REPO/scripts/twin5090/train.py" "$STAGE_DIR/train.py"
cp "$REPO/experiments/exp39_twin_1b.py" "$STAGE_DIR/experiments/"
rsync -a --exclude __pycache__ "$REPO/kinetic_ai" "$STAGE_DIR/"

python3 - "$MODE" "$PACK_BASE" "$STAGE_DIR" <<'EOF'
import json, sys
mode, pack, stage_dir = sys.argv[1:4]
common = ["--pack-dir", pack, "--device", "cuda:0"]
if mode == "preflight":
    stages = [
        ["--arm", "explicit", "--preflight", "12"] + common,
        ["--arm", "tied", "--preflight", "12"] + common,
    ]
elif mode == "phase1":
    stages = [
        ["--arm", "explicit", "--target-tokens", "2500000000"] + common,
        ["--arm", "tied", "--target-tokens", "2500000000"] + common,
    ]
elif mode == "extend":
    stages = [
        ["--arm", "tied", "--target-tokens", "10000000000",
         "--decay-start-tokens", "8000000000",
         "--decay-end-tokens", "10000000000"] + common,
    ]
elif mode == "interventions":
    stages = [
        ["--arm", "tied", "--target-tokens", "500000000",
         "--block-lr-scale", "0.25",
         "--out-dir", "results/scale/exp39/i1_blocklr",
         "--milestones", "500000000"] + common,
        ["--arm", "tied", "--target-tokens", "500000000",
         "--supervise-final-only",
         "--out-dir", "results/scale/exp39/i2_finalonly",
         "--milestones", "500000000"] + common,
    ]
elif mode == "sync-pack":
    stages = []
else:
    raise SystemExit(f"unknown mode {mode}")
json.dump({"stages": stages}, open(f"{stage_dir}/job.json", "w"), indent=1)
EOF

echo "== syncing pack $PACK_DIR -> $HOST:~/fusion-project/$JOB/$PACK_BASE =="
ssh "$HOST" "mkdir -p ~/fusion-project/$JOB"
rsync -a --info=progress2 "$PACK_DIR/" "$HOST:~/fusion-project/$JOB/$PACK_BASE/"

if [ "$MODE" = "sync-pack" ]; then
    echo "pack synced; no job submitted"
    exit 0
fi

bash "$SKILL/submit_job.sh" "$STAGE_DIR" "$JOB" train.py "$HOST"
