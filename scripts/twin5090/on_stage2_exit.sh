#!/usr/bin/env bash
# SPEC 0022 stage-2 exit watcher, running ON the 5090 (the GB10-side monitor
# left with the GB10 for RMA on 2026-09-02).
#
# Guards the running Arm T trainer BY PID, and when it exits:
#   1. verifies stage 2 exited 0 with the tied 2.5B MILESTONE line, else
#      writes ALERT_stage2_not_clean and stops (resume per HANDOFF-5090.md);
#   2. archives the Phase 1 log/job.json beside the job dir;
#   3. runs the exp40 core ladder on the tied 2.5B checkpoint, on the GPU,
#      between stages (best effort, bounded by EVAL_TIMEOUT) - never while
#      a training stage runs;
#   4. stages the current repo code into the job dir (the job dir's copy
#      predates SPEC 0024's --block-lr-scale / --supervise-final-only) and
#      launches the two SPEC 0024 interventions as one detached job, exactly
#      as launch.sh mode `interventions` builds them.
#
# Usage (detached, survives the session):
#   setsid nohup scripts/twin5090/on_stage2_exit.sh <train_pid> \
#       > /dev/null 2>&1 &
# Progress: tail -f ~/fusion-project/kinetic-twin/stage2_watch.log
set -uo pipefail

PID="${1:?pid of the running 'python train.py' (Arm T stage 2)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB="${JOB_DIR:-/home/ss/fusion-project/kinetic-twin}"
COMPOSE_DIR="${RTX5090_COMPOSE_DIR:-/home/ss/rtx5090setup/docker}"
CONTAINER_JOB=/fusion-project/kinetic-twin
LOG="$JOB/stage2_watch.log"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-90m}"
DRY_RUN="${DRY_RUN:-0}"  # 1: exercise the exit logic, skip the eval and the launch

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

# PID guard: the guarded process must be the real trainer, not a pattern match.
cmd="$(ps -o args= -p "$PID" 2>/dev/null || true)"
case "$cmd" in
    *"python train.py"*) ;;
    *) log "ABORT: pid $PID is not 'python train.py' (got: '${cmd:-<gone>}')"; exit 2 ;;
esac
log "GUARD pid=$PID cmd='$cmd' repo=$(git -C "$REPO" rev-parse --short HEAD) job=$JOB"

while kill -0 "$PID" 2>/dev/null; do sleep 60; done
log "trainer pid $PID exited"
sleep 20  # let the container flush train.log

exit_line="$(grep -E '^=== stage 2 exit' "$JOB/train.log" | tail -1 || true)"
milestone="$(grep -E "^MILESTONE .*'milestone_tokens': 2500000000.*'arm': 'tied'" "$JOB/train.log" | tail -1 || true)"
log "exit line: ${exit_line:-<none>}"
log "milestone: ${milestone:-<none>}"
if [[ "$exit_line" != *" exit 0 after "* || -z "$milestone" ]]; then
    log "ALERT: stage 2 did not exit 0 with the tied 2.5B milestone; NOT launching interventions."
    log "       Resume Arm T with --resume auto from results/scale/exp39/tied/ckpt_latest.pt (HANDOFF-5090.md)."
    touch "$JOB/ALERT_stage2_not_clean"
    exit 1
fi

cp "$JOB/train.log" "$JOB/train_phase1.log"
cp "$JOB/job.json" "$JOB/job_phase1.json"
log "archived Phase 1 log -> train_phase1.log, job.json -> job_phase1.json"

# 3. Ladder eval on the GPU between stages (the one-GPU-job rule holds:
#    nothing is training now, and the interventions launch only after this).
CKPT="$JOB/results/scale/exp39/tied/ckpt_2500000000.pt"
OUT="$REPO/results/scale/exp40/milestone_tied_2p5B.json"
if [ "$DRY_RUN" = 1 ]; then
    log "DRY_RUN: skipping ladder eval"
elif [ -f "$CKPT" ]; then
    log "ladder eval on $CKPT (GPU, timeout $EVAL_TIMEOUT) -> $OUT"
    if (cd "$REPO" && timeout "$EVAL_TIMEOUT" .venv/bin/python experiments/exp40_ladder.py \
            --checkpoint "$CKPT" --tasks core --max-examples 1000 \
            --device cuda:0 --out "$OUT") >> "$JOB/exp40_tied_2p5B.log" 2>&1; then
        log "ladder eval done (uncommitted: commit $OUT with the milestone journal entry)"
    else
        log "WARN: ladder eval rc=$? (best effort; see exp40_tied_2p5B.log); continuing"
    fi
else
    log "WARN: $CKPT missing; skipping ladder eval"
fi

# 4. Stage current code (SPEC 0024 flags) and launch the interventions.
rsync -a --delete --exclude __pycache__ "$REPO/kinetic_ai/" "$JOB/kinetic_ai/"
cp "$REPO/experiments/exp39_twin_1b.py" "$JOB/experiments/exp39_twin_1b.py"
cp "$REPO/scripts/twin5090/train.py" "$JOB/train.py"
python3 - > "$JOB/job.json" <<'EOF'
import json
common = ["--pack-dir", "pack_1b", "--device", "cuda:0"]
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
print(json.dumps({"stages": stages}, indent=1))
EOF
log "staged repo $(git -C "$REPO" rev-parse --short HEAD) into $JOB; job.json = SPEC 0024 interventions (I1 block-lr/4, I2 final-only)"

if [ "$DRY_RUN" = 1 ]; then
    log "DRY_RUN: would launch 'docker compose exec -d train ...' in $COMPOSE_DIR; stopping here"
    exit 0
fi
if pgrep -x -f 'python train.py' > /dev/null; then
    log "ABORT: a 'python train.py' is already running; refusing a second GPU job"
    exit 3
fi
(cd "$COMPOSE_DIR" && docker compose exec -d train bash -c \
    "cd $CONTAINER_JOB && python train.py > train.log 2>&1")
sleep 45
newpid="$(pgrep -x -f 'python train.py' | head -1 || true)"
log "interventions launched: pid=${newpid:-<none>}"
tail -n 5 "$JOB/train.log" 2>/dev/null | tee -a "$LOG"
printf '{"launched_at": "%s", "repo_commit": "%s", "pid": "%s", "spec": "0024", "job": "%s"}\n' \
    "$(date -Is)" "$(git -C "$REPO" rev-parse HEAD)" "${newpid:-}" "$(tr -d '\n' < "$JOB/job.json")" \
    > "$JOB/interventions_launch.json"
[ -n "$newpid" ] || { log "ALERT: no trainer pid after launch; check train.log"; touch "$JOB/ALERT_interventions_not_started"; exit 4; }
log "done; next: I1/I2 0.5B milestones read against SPEC 0024 bars (rescue <=1589, no-rescue >=1780)"
