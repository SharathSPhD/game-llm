# Compute estimate — training a Qwen3-class kinetic model to beat baseline from scratch

**Status:** decision support, not an ADR — no decision has been made to pursue this path.
**Date:** 2026-08-30. **Scope:** what it would cost, in dollars and wall-clock, to train a
kinetic-architecture base model large and heavily-trained enough to independently beat
Qwen3-class baselines on public benchmarks — via full from-scratch pretraining, or via teacher
distillation — priced across RunPod GPU options.
**Author context:** produced by an agent session in worktree `kinetic-ai-runpod-estimate-76abc3`
on operator request. Not reviewed or acted on.

## 1. This contradicts two things already on record

`docs/PRD.md`'s stated success criterion is that the system **beats Qwen3-1.7B** (the
active-parameter-matched baseline) and is credited against Qwen3-4B — but the mechanism is the
**equilibrium council aggregating existing smaller models at inference time**, not out-training
Qwen3 via a bigger pretraining run. The same document's Non-goals section states plainly:

> Training a competitive base model from scratch, which the available compute rules out.

That conclusion is not a hand-wave. It is the direct result of two experiments this project
already ran and closed, recorded in `research/memory/findings.md`:

- **F44 / F45** — kinetic tying is a parameter↔compute *trade*, never a compute *saving*. At
  matched compute, tying yields 2.70x fewer resident parameters for ~96% of an explicit model's
  quality. To reach full parity with an explicit model at *matched parameters*, tying costs
  **4.92x more compute** (F44). There is no regime in which tying reduces the FLOPs needed to
  reach a target quality bar.
- **F51** — converting a pretrained (already-trained) model into tied form fails at any
  gentleness tested; no cheap conversion shortcut exists.
- **F53** — the knowledge-distillation pilot (student trained from scratch against
  Qwen2.5-1.5B-Instruct as teacher) made results **worse**, not better: cross-entropy alone
  reached 482.96 held-out perplexity; adding temperature-softened logit distillation reached
  493.39 — a −2.2% change, the wrong direction. This closed "cheap distillation" as a shortcut
  to scale, with two open caveats noted below (§4).

This document quantifies exactly what "requires a pretraining budget" costs, so the scale of
the non-goal is visible in dollars rather than asserted.

## 2. Method

Standard compute-scaling estimate: **C ≈ 6·N·D** FLOPs, where N = parameters and D = training
tokens (the usual forward + backward FLOPs approximation used in scaling-law literature).
Achieved GPU throughput is assumed at **35% model-FLOPs-utilization (MFU)** of each GPU's
public dense-BF16 TFLOPS spec. This is an assumption, not a measurement — this project's own
discipline (SPEC 0022's preflight GO rule) is to verify throughput before trusting it with real
budget, and that discipline should apply here before any of these numbers are spent against.

Kinetic tying gets the **4.92x compute tax** (F44) applied whenever the target is a resident
parameter count matched to a Qwen3 size — because tying does not save compute, it only trades
compute for a parameter saving.

Two token budgets are modeled per target size:

- **Chinchilla floor** (20 tokens/param) — cheap, but not remotely competitive with a
  heavily-trained modern model; this is roughly the regime SPEC 0022's current ~10B-token
  programme already sits in.
- **Competitive / overtrained** (~1,500 tokens/param) — anchored to real modern small-model
  practice (Llama-3-8B: 15T tokens / 8B params ≈ 1,875 tokens/param). This is the realistic bar
  to actually contend with a model like Qwen3, which was itself heavily overtrained relative to
  Chinchilla-optimal.

## 3. Headline numbers — Qwen3-1.7B-class, kinetic-tied, competitive budget

| Scenario | Tokens | FLOPs |
|---|---|---|
| Chinchilla floor, explicit-equivalent | 34B | 3.47e20 |
| Chinchilla floor, kinetic-tied (×4.92 tax) | 34B | 1.71e21 |
| Competitive, explicit-equivalent | 2.55T | 2.60e22 |
| **Competitive, kinetic-tied (×4.92 tax)** | 2.55T | **1.28e23** |

### Cost and time by RunPod GPU — competitive budget, kinetic-tied (the actual ask)

| GPU | Achieved TFLOPS (35% MFU) | GPU-hours | Community $ | Secure $ |
|---|---|---|---|---|
| RTX 5090 | 73 | 486,000 | $335,400 | $481,200 |
| A100 SXM 80GB | 109 | 325,600 | $452,600 | $517,700 |
| **H100 SXM** | 346 | 102,700 | **$276,300** | $337,900 |
| H200 SXM | 346 (assumed equal to H100; extra bandwidth may raise real MFU) | 102,700 | $368,700 | $471,400 |
| B200 (Secure only — 0 Community slots) | 787 | 45,150 | — | $306,500 |

H100 is the cheapest single-GPU-equivalent option at ≈$276K. Even the fastest option (B200)
needs ~45,150 GPU-hours; no RunPod listing offers more than 8 GPUs per pod today, so reaching
this in weeks rather than months would require a genuinely large multi-node cluster — well
beyond anything discussed in the RunPod migration estimate for the current 1B-scale run.

For calibration, the **Chinchilla-floor / kinetic-tied** case (not competitive — a sanity check
only) costs roughly $4–6 and finishes in under 90 minutes on H100, consistent with where SPEC
0022 already sits. The gap between "cheap" and "actually competitive" is about five orders of
magnitude — the concrete shape of the PRD's "available compute rules out" line.

### Explicit-equivalent floor (no kinetic tax) — for calibration only

Divide the table above by 4.92. This does not deliver the parameter-count saving kinetic tying
is for, it only establishes the compute floor for an explicit model at the same budget.

| GPU | Community $ | Secure $ |
|---|---|---|
| RTX 5090 | $68,171 | $97,810 |
| A100 SXM 80GB | $92,000 | $105,226 |
| H100 SXM | $56,150 | $68,675 |
| B200 | — | $62,300 |

### Scaling to Qwen3-4B

Roughly 5.54x the 1.7B FLOPs (N·D scaling at the same token/param ratio). Competitive,
kinetic-tied cost lands around **$1.5M–$2.7M** across the same GPU spread.

## 4. The "via teacher" path

F53 already tested this and found it made things worse, at the token regime tried (16.4M
tokens, an extreme early-stage regime). Two honest caveats the project itself flagged, neither
yet retested:

- Only one KD configuration was tried (distillation weight 1.0, temperature 2.0); a sweep might
  behave differently.
- 16.4M tokens may be too early for a from-scratch student to use a 151k-way soft teacher
  distribution at all — published KD successes distill into pruned-from-teacher
  initializations that start near the teacher, not into random weights.

If pursued regardless, distillation currently has **no proven compute discount** to offset
against, and it adds real cost: running the teacher's forward pass alongside the student's
training adds roughly **2·N_teacher·D** FLOPs on top of the student's own 6·N_student·D. For a
7B teacher against a 1.7B student, that is **≈+137% extra compute** — making the teacher path
more expensive than from-scratch under current evidence, not less.

**Recommendation if this path is still of interest:** run a proper (not "cheap") KD sweep first
— a few configurations, a larger token count than F53's 16.4M, at roughly the same ~1-day
5090 cost as the original pilot — before committing any of the six-figure budgets above to
either path. That is a near-$0 experiment (already-owned hardware) that could overturn F53,
versus a $270K+ commitment that currently assumes it already has.

## 5. What this means for the programme

Under current evidence, neither shortcut (conversion, cheap distillation) survives, and the
full-budget path costs 5+ orders of magnitude more than the current SPEC 0022 programme. The
PRD's actual success criterion — beating Qwen3-1.7B via the equilibrium council rather than via
a bigger base model — remains the validated, affordable path. This document exists so that
choice is visible as a quantified tradeoff rather than an assumption, and so a future decision
to revisit the non-goal (if RunPod economics or a reversed KD finding change the calculus) has
real numbers to start from.

## 6. Before spending against any number here

1. These are extrapolated estimates (6ND scaling law, 35% MFU assumption), not measurements.
   Re-verify with a short preflight on the target GPU before committing real budget, per this
   project's own preflight discipline.
2. Confirm RunPod pricing/availability at execution time — queried live 2026-08-30 for the
   companion `docs/runpod-migration-estimate.md`; prices move.
3. Run the proper KD sweep (§4) before assuming the teacher path is closed for good — F53
   tested one configuration at one very early token count, not the design space.
4. This document evaluates a strategy the PRD lists as a non-goal. Any decision to pursue it
   should be an explicit operator redirection of PRD scope, not an implicit one.
