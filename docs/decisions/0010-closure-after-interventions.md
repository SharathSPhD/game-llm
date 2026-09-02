# ADR 0010 — Close the scale programme after SPEC 0024; ship the equilibrium layer as the product

Date: 2026-09-02 · Status: accepted · Amends: [ADR 0008](0008-equilibrium-decoding-paradigm.md), [ADR 0006](0006-scale-program-1b3b.md)

## Context

SPEC 0022 took the compute-matched exchange rate (F45/F50: a weight-tied
block reaches 0.958 of an explicit transformer's quality at equal compute
with 2.70× fewer resident parameters, measured at 46–121M on BabyLM) to 1B
scale on FineWeb-Edu. Held-out perplexity at 0.5B/1B/2.5B tokens: explicit
Arm E 1271/503/260, tied Arm T 1962/785/340. The pre-registered kill gate at
1B (ratio ≤ 1.20) failed at 1.560; the 2.5B success bar (≤ 1.10) was missed
at 1.308. The gap is closing (0.44 → 0.27 nats), and a power-law fit of the
three milestones per arm (R² 0.97/0.99) projects the 1.10 bar between 5B and
10B tokens — a projection, not a measurement. The public ladder is at chance
for both arms (mean of six tasks 0.34 against Pythia-410m's 0.51 and
SmolLM2-360M's 0.61 on the same harness), so perplexity is the only signal.

An adversarial review on 2026-09-02 (five independent reviewers plus a TRIZ
contradiction analysis; journal entry of that date) established: F45's
confidence interval [0.939, 0.977] straddles its 0.95 bar and F50 is a single
seed; findings F45–F54 carry unresolved Tarka reviews; the paper and site are
silent on SPEC 0022; the app is a replay shell whose backend left with the
GB10; and the technical layer has three mypy errors, two missing seed configs
(exp32 seeds 43/44) and six scripts with a hard-coded GB10 home path.

Compute is one RTX 5090. The tied arm trains at 15.6k tok/s: the 10B
extension is 7.4 GPU-days; both arms to 5B is ~6 GPU-days; Pythia-410m's
token budget would be ~220 GPU-days. A 158M model at the projected 10B
perplexity (≈ 70–100) would remain at chance on the ladder.

## Decision

1. **Closure point.** The scale programme closes after SPEC 0024's two
   interventions (I1 block-lr/4, I2 final-only supervision, 0.5B tokens
   each, running since 2026-09-02 09:17). Their readings against the
   pre-registered bars (≤ 1589 rescue, ≥ 1780 no rescue) earn either the
   NULL at 1B or a rescue arm's re-run of the 1B gate. SPEC 0023 C1 runs
   only if an intervention rescues. The 10B extension does not run.
2. **The finding to record.** The exchange rate holds at 46–121M and does
   not transfer unchanged to 1B on web data at 2.5B tokens; the trajectory
   is closing and the mechanism is whichever SPEC 0024 identifies. This is
   stated as a scale boundary, not a paradigm result in either direction.
3. **Utility deliverable.** The product is the equilibrium decoding layer of
   ADR 0008 over public open-weight players — the object F41/F54 measured —
   served through the existing API and app. The tied EqLM checkpoints ship
   as research artifacts with a model card carrying the twin's numbers and
   non-claims. Live inference must not depend on a competitive base model
   of our own or on a specific host.
   Operator refinement (2026-09-02, later the same day): live inference is
   deferred until the GB10 returns; until then the app ships replay-only,
   with the council's telemetry pre-recorded from the F41/F54 measurements
   and every page that once called the backend labelled as replay. The
   default live council is the measured F41 council, so the app's numbers
   are the paper's. The Training Studio is retired to a read-only run
   registry; job submission goes, since the product must never contend
   for a training GPU.
4. **Serving host.** Serving is profile-driven: `configs/serve/profiles/`
   holds one YAML per host (`gb10`, `rtx5090`), selected by
   `KINETIC_SERVE_PROFILE`. The 5090 profile runs CPU-only while a training
   job holds the GPU and flips to GPU when it does not; the GB10 profile is
   restored by changing one variable when the box returns.
5. **Paper.** v2 to arXiv with the scale boundary: SPEC 0022's twin, the
   gate verdict, SPEC 0024's mechanism, and the F45 interval stated as a
   straddle. The Tarka layer for F45–F54 is closed before the tag.

## Consequences

The closure contract's hard rules are honoured: no NULL without two
interventions, no number reported that was not produced. The programme
forgoes the 5B/10B measurement that would turn the projection into a
finding; if the operator later funds it, the pre-registered shape is both
arms to 5B, not the tied arm alone. GPU-free closure work (paper, site,
seed configs, mypy, serving profiles, Tarka pass) starts immediately and
runs alongside the interventions.
