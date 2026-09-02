# Resuming the Kinetic AI scale programme (if ever)

The programme was closed on 2026-09-02 (ADR 0010, ADR 0011; finding F55).
This note is for whoever reopens it. Read `CLAUDE.md`, `research/memory/
state.json`, the last entries of `research/memory/journal.md`, and ADRs 0008,
0010 and 0011 first.

## Where the science stopped

- SPEC 0022 (twin at 1B, FineWeb-Edu, 2.5B tokens each arm) is COMPLETE.
  Checkpoints and milestone logs are in the 5090 job directory
  `~/fusion-project/kinetic-twin/results/scale/exp39/{explicit,tied}/`
  (`ckpt_{500000000,1000000000,2500000000}.pt`, root-owned, ~1.9 GB each for
  the tied arm). They are NOT in git and NOT on the Hub; publish or back
  them up before the job directory is cleaned. The held-out numbers are in
  `milestones.jsonl` beside them and in F55.
- SPEC 0024 (interventions) was HALTED: I1 stopped at 209.7M / 500M tokens
  with no checkpoint (checkpoints are written every 500M tokens); I2 never
  ran. Partial I1 log: `results/scale/exp39/i1_blocklr_halted/`.
- SPEC 0023 (byte-level cell) never ran. Its pack exists at
  `~/fusion-project/kinetic-twin/pack_byte/` (5.0B bytes, hash 08ad3d88bab4).
- The 10B extension never ran. The pre-registered shape for a scale answer
  is BOTH arms to 5B tokens (~6 GPU-days on a 5090), not the tied arm alone.

## The open scientific question

Does the compute-matched exchange rate (F45/F50, 0.958 at 46–121M) hold at
1B? At 2.5B tokens the tied/explicit perplexity ratio is 1.308 and closing; a
power-law fit puts the 1.10 bar between 5B and 10B tokens. The mechanism of
the early gap (block learning rate, supervision schedule, or capacity) is
unresolved because SPEC 0024 was halted. Answering it costs ~18 GPU-hours
for the two interventions from scratch, or ~6 GPU-days for the 5B twin.

## How to relaunch (5090, docker `rtx5090-train`)

```bash
cd ~/fusion-project/kinetic-twin
# stage current code (the job dir's copy is whatever was last rsynced)
rsync -a --delete --exclude __pycache__ ~/projects/game-llm/kinetic_ai/ kinetic_ai/
cp ~/projects/game-llm/experiments/exp39_twin_1b.py experiments/
cp ~/projects/game-llm/scripts/twin5090/train.py train.py
# job.json: see scripts/twin5090/launch.sh for the exact stage argv of
# preflight | phase1 | extend | interventions; write it by hand, then:
cd ~/rtx5090setup/docker && docker compose exec -d train bash -c \
  'cd /fusion-project/kinetic-twin && python train.py > train.log 2>&1'
```

Set `gpu_lock: true` in `research/memory/state.json` before launching and
`false` after; the serving profile and the executor both read it. Never two
`python train.py` at once. `scripts/twin5090/on_stage2_exit.sh <pid>` shows
how to chain stages and evals unattended (PID-guarded; uses `ps -p`, not
`kill -0`, because the trainer is root inside the container).

## Evaluation

`experiments/exp40_ladder.py --checkpoint <ckpt> --tasks core --max-examples
1000 --device cuda:0 --out results/scale/exp40/<name>.json` — about three
minutes per checkpoint on the 5090. Public rungs are already measured and
committed; do not re-run them.

## What closure changed that a resumer must know

- Serving is profile-driven (`KINETIC_SERVE_PROFILE`, `configs/serve/
  profiles/`). The GB10 profile is `gb10`; the 5090 profile serves from the
  CPU while the lock is held.
- Tests use a private state file (`KINETIC_STATE_FILE`); they never touch the
  repo's lock. If you see the lock flipped to false with a job running,
  something else did it.
- exp32's recorded config hashes do not reproduce from the committed
  configs (see the headers of `configs/exp32_seed4{3,4}.yaml`). F45's numbers
  stand; its hash chain does not.
- The record ends at F55. Next finding number: F56.
