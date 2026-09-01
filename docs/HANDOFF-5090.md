# Handoff — continuing SPEC 0022/0023/0024 from a session on the RTX 5090 box

Written 2026-09-01 on the GB10 immediately before its shutdown for RMA
return. Audience: a fresh Claude Code session running ON the 5090
workstation (ss-Fusion-75, user `ss`), with no memory of this one. Read
`CLAUDE.md`, `research/memory/state.json`, `research/memory/journal.md`
(last five entries), and specs 0022/0023/0024 before acting. The operator's
standing directives in CLAUDE.md bind unchanged — commits as
`SharathSPhD <qbz506@york.ac.uk>` with no Co-Authored-By, pre-registration
before runs, never two training jobs at once, measured beats extrapolated.

## Environment: use the migration, do not rebuild it

The dedicated migration session already moved everything (its commits and
docs live under `_migration/`): the project tree is at
`/home/ss/projects/game-llm`, the GB10's Claude sessions — including the
session that ran this programme — resume via
`CLAUDE_CONFIG_DIR=/home/ss/.claude-gb10` + `claude --resume`, and user
skills (efe-autoresearch, rtx5090-connect, academic-paper-style) are
exported in `_migration/user_skills/`. Follow `_migration/MIGRATION.md`
for login and paths; nothing environmental needs re-doing. This document
carries only what that one does not: the live research-programme state and
queue. The git remote remains the source of truth for the record.

## Where the run artifacts are

The job directory `~/fusion-project/kinetic-twin/` holds:
`pack_1b/` (10.5B-token GPT-2 pack, hash 973c14c07147), `pack_byte/`
(SPEC 0023's 5.0B-byte pack, hash 08ad3d88bab4, synced today),
`gb10_artifacts/ckpt/` (the 121M anytime, 46M compute-matched, and explicit
baseline checkpoints exp41/exp42 use), `results/scale/exp39/` (all twin
checkpoints and logs), `train.py` + `job.json` (the dispatcher), and the
running container context. Training runs inside the persistent docker
compose service `train` (container `rtx5090-train`); jobs are launched
detached as `docker compose exec -d train bash -c 'cd /fusion-project/
kinetic-twin && python train.py > train.log 2>&1'` from the compose dir
(the old remote wrapper `scripts/twin5090/launch.sh` did exactly this over
ssh — running locally, reuse its job.json-building python block and skip
the rsync/ssh parts, or keep using it with RTX5090_HOST=ss@localhost).

## What is running RIGHT NOW (do not disturb)

SPEC 0022 Phase 1, stage 2: Arm T (tied, 158M resident) training to 2.5B
tokens, ~15.6k tok/s, expected to finish around 2026-09-02 midday. The
dispatcher exits after stage 2 — nothing else is queued in job.json. Check
`tail ~/fusion-project/kinetic-twin/train.log` and
`results/scale/exp39/tied/train_log.jsonl`. Loss was 6.78 at 1B tokens,
descending smoothly. If the process is found dead, `--resume auto` from
`ckpt_latest.pt` continues it exactly (the checkpoint carries optimizer,
cursor, RNG, pack hash).

## The state of the science (one paragraph)

Arm E (913M explicit) completed 2.5B tokens: held-out ppl 1271 → 503 → 260
at 0.5/1/2.5B. Arm T FAILED the pre-registered kill gate at 1B: ppl 785.4
vs bar 604.2 (1.20 × Arm E), ratio 1.560 and widening from 1.543 at 0.5B.
The 10B extension is therefore HALTED (operator-confirmed). Benchmarks are
at chance for both arms at these budgets — the ppl trail is the record.
No NULL is declared yet: SPEC 0024's two interventions must run first.

## The queue (operator-decided, in order)

1. **When stage 2 exits** (Arm T reaches 2.5B): record the final milestone
   (its held-out ppl prints in train.log as `MILESTONE {...}` and in
   `tied/milestones.jsonl`; the trend vs 1.560 distinguishes slow
   convergence from a capacity ceiling — journal it). Then launch
   SPEC 0024: build job.json with the two intervention stages exactly as
   `scripts/twin5090/launch.sh` mode `interventions` constructs them
   (I1: tied 0.5B tokens with `--block-lr-scale 0.25`, out-dir
   `results/scale/exp39/i1_blocklr`; I2: tied 0.5B with
   `--supervise-final-only`, out-dir `results/scale/exp39/i2_finalonly`;
   both `--pack-dir pack_1b --milestones 500000000`). ~9h each.
2. **Read the interventions** against SPEC 0024's fixed bars: held-out ppl
   at 0.5B ≤ 1589 = rescue (send that arm on to 1B and re-apply the kill
   gate bar 604.2; a pass reopens the 10B extension with the fix);
   both ≥ 1780 = the NULL is earned and recorded; between = report the
   numbers and ask the operator.
3. **SPEC 0023 C1** (byte cell) regardless of 2's outcome: preflight both
   arms first (12 steps, `--pack-dir pack_byte --d-model 1536 --n-heads 16
   --d-ff 6144 --vocab-size 256 --seq-len 2048`), GO ≥ 12k bytes/s, then
   Arm E (explicit, 5B bytes) then Arm T, milestones 1B/2.5B/5B bytes,
   kill gate at 1B bytes: tied bpb ≤ 1.20 × explicit (bpb = held-out
   loss / ln 2). Success: bpb ratio ≤ 1.10, or ratio ≥ 12× measured with
   bpb ≤ 1.15.
4. **Utility** (operator: decide after diagnostics): the SFT/MPO pipeline
   is built (`kinetic_ai/train/instruct.py`, `experiments/
   exp42_instruct_pilot.py`, mechanically validated). If an intervention
   rescued tying → extension → SFT+MPO the tied 10B model. If not → the
   operator chooses between SFT+MPO on Arm E's 2.5B checkpoint (MPO keeps
   the MMD spine in the shipped artifact) and shipping tied with honest
   numbers. AskUserQuestion at that fork.

## Milestone eval protocol (the ladder must not lapse)

Every milestone checkpoint gets `experiments/exp40_ladder.py --checkpoint
<ckpt> --tasks core --max-examples 1000` (bf16 path is default for our
checkpoints; results named `results/scale/exp40/milestone_<arm>_<tokens>.
json`, committed). Public rungs are already measured and committed
(`rung_*.json`) — do not re-run them. With the GB10 gone, evals share the
5090: run them between training stages, or on CPU if the queue is hot (a
core eval is ~7h on CPU — acceptable for non-blocking milestones; NEVER
preempt a training stage for an eval). The halting probe
(`experiments/exp41_halting_semantics.py`) runs on tied milestone
checkpoints when convenient; its record so far: effort-expectancy
correlation replicates at deployment scale (rho ~0.27), final-position
early-settling at 121M and 1B geometry, reversed at 46M.

## What the GB10 shutdown takes offline (tell the operator if asked)

The app's model serving (FastAPI + tunnel on the GB10) is down until the
box returns; the Vercel frontend stays up but live inference calls will
fail — the demo page should be treated as replay-only. The Supabase/
Cloudflare gateway configuration is untouched and will reconnect when a
serving host reappears. Do not attempt to re-host serving on the 5090
while it carries the training queue.

## Ledger discipline

Findings only through the closure contract (CLAUDE.md): pre-registered
gates read as written, no NULL without the two interventions (SPEC 0024 is
exactly that), journal + state.json updated in the same cycle as every
result, paper/site sections updated when findings close (F-numbers next:
F55). Push to origin at every milestone — with the GB10 gone, the GitHub
remote is the only off-box copy of the record.
