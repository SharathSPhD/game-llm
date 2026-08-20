# Kinetic AI Cycle Runner — execute one research cycle

## Procedure (idempotent, resumable)

1. **Orient.** Read `research/memory/state.json`; tail `journal.md`; `git log --oneline -5`;
   `nvidia-smi` (GPU state). Never trust memory over these files.
2. **Harvest pending results.** If a training/eval run finished: verify seeds md5-distinct,
   run stats gates (bootstrap CI, Wilcoxon/paired bootstrap, Holm if multiple), record.
   Gate on result files EXISTING, not on timeouts.
3. **Select next action** (highest expected information gain toward H1–H4; GPU-aware):
   - GPU BUSY → GPU-free work: paper sections from validated findings, app, specs, Tier A.
   - GPU FREE + experiment designed → launch ONE run (background, `setsid`) + set gpu_lock.
   - No experiment ready → design the RQ (spec first), build its code (TDD), then launch.
4. **Act** via subagents/Workflows. TDD for all code.
5. **Tarka adversarial review** — attack every finding (refute-first) before recording.
6. **Record.** Append `journal.md`; update `findings.md` (status: SIGN-OFF PENDING) and
   `state.json`; commit as SharathSPhD <qbz506@york.ac.uk> (no Co-Authored-By).
7. **Schedule next.** If a long run is pending, ensure a durable poll (cron/ScheduleWakeup).

## Guardrails

- Never two GPU jobs at once. Never report a number you didn't produce.
- Never NULL without ≥2 documented interventions. Never fabricate a citation.
- Keep cycles cheap when idle. Milestone push at each validated finding / phase closure.
