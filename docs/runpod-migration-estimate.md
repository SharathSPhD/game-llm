# RunPod migration estimate — SPEC 0022 twin run

**Status:** decision support, not an ADR — no migration decision has been made.
**Date:** 2026-08-30. **Scope:** SPEC 0022 (`research/specs/0022-twin-at-1b.md`), Phase 1,
currently running on the local RTX 5090 (launched 2026-08-30, detached job `kinetic-twin`).
**Author context:** produced by an agent session in worktree `kinetic-ai-runpod-estimate-76abc3`
in response to an operator request to evaluate shifting this run to RunPod; not yet
reviewed or acted on. Re-verify pricing/availability before executing any scenario below —
RunPod pricing is live and can move.

## 1. Baseline (measured, local RTX 5090)

Source: `research/memory/journal.md` (2026-08-29 preflight GO, 2026-08-30 Phase 1 launch), `research/memory/state.json`.

| Arm | Params | Measured throughput | Peak VRAM |
|---|---|---|---|
| Arm E (explicit) | 913.0M | 20,330 tok/s | 28.6 GiB / 32 |
| Arm T (tied, anytime-head overhead) | 158M resident | 15,647 tok/s | 28.6 GiB / 32 |

- Twin phase: both arms to 2.5B tokens each (5B tokens total), currently run **sequentially on
  one GPU** (Arm E then Arm T) because only one 5090 is available locally. Measured/projected:
  3.27 days (78.5 GPU-hours).
- Extension: Arm T only continues from 2.5B tokens toward a **10B-token stretch target**
  (floor 6B, per SPEC 0022's budget cap). Single stream, single GPU, no parallelism available
  in the current codebase (see §4). Stretch: +7.5B tokens = 133.1 GPU-hours = 5.54 days. Floor:
  +3.5B tokens = 62.1 GPU-hours = 2.59 days.
- Full programme (stretch target): 211.6 GPU-hours = **8.81 days**, against a 24-day cap.
- Gates: kill gate ppl ratio ≤1.20 at 1B tokens; success bar ≥0.95 ladder ratio at 2.5B tokens.
  The extension only proceeds if these clear — the "8.81 days" figure assumes it does.

Local marginal cost while the 5090 keeps running: effectively sunk (already launched), plus
≈$22–34 in electricity over the full programme (532W measured draw, ~$0.20/kWh, 8.8 days).

## 2. RunPod GPU catalog (live, queried via RunPod MCP `list-gpu-types`, 2026-08-30)

| GPU | VRAM | $/hr Community | $/hr Secure | Community max / Secure max | Availability |
|---|---|---|---|---|---|
| **RTX 5090** (identical to local hardware) | 32GB | $0.69 | $0.99 | 8 / 10 | MEDIUM |
| RTX 4090 | 24GB | $0.34 | $0.74 | 8 / 8 | HIGH — **excluded**, see §5 |
| RTX PRO 4500 Blackwell | 32GB | n/a (community=false) | $0.72 | 0 / 8 | HIGH |
| RTX PRO 6000 Blackwell | 96GB | $1.69 | $2.09 | 9 / 9 | HIGH |
| A100 SXM4 80GB | 80GB | $1.39 | $1.59 | 8 / 8 | HIGH |
| A100 PCIe 80GB | 80GB | $1.19 | $1.39 | 8 / 8 | LOW |
| H100 SXM 80GB | 80GB | $2.69 | $3.29 | **1** / 8 | HIGH |
| H200 SXM 141GB | 141GB | $3.59 | $4.59 | 8 / 8 | HIGH |
| B200 | 180GB | listed $5.98 but **0 community slots** | $6.79 | 0 / 8 | LOW |
| L40S | 48GB | $0.79 | $0.99 | 8 / 8 | MEDIUM |

Network volume storage: $0.07/GB/month (first 1TB), $0.05/GB/month thereafter. Estimated need
for this run (pack + checkpoints) ≈100GB → ≈$2–5 over the programme's ~9 days. Negligible next
to compute cost.

Community Cloud = third-party hosts, no SLA, can be reclaimed without warning (cheapest).
Secure Cloud = RunPod-owned datacenter capacity, uptime commitment (more expensive). Given
checkpoint/resume is already verified bit-exact for this job, Community Cloud's interruption
risk is cheap to absorb unless gate stakes require otherwise.

## 3. Scenario comparison

All GPU-hour figures for non-5090 GPUs are **extrapolated from public spec-sheet FLOPS/memory
bandwidth**, not measured — see §6 before trusting them with real budget.

### 3a. Full migration to RunPod (no local 5090 involvement)

| Plan | Compute cost (Community) | Compute cost (Secure) | Wall-clock |
|---|---|---|---|
| 1x RTX 5090, sequential (mirrors current design) | ≈$146 | ≈$210 | 8.81 days |
| 2x RTX 5090, twin phase parallel + 1x extension | ≈$146 (same GPU-hours, redistributed) | ≈$210 | 7.39 days |

Parallelizing the twin phase costs the same total (each pod is released as soon as its own arm
hits 2.5B tokens — total GPU-hours billed is unchanged), it only buys back wall-clock.

### 3b. Local 5090 + RunPod in parallel (recommended baseline, no code changes needed)

| # | Plan | Marginal $ beyond local | Wall-clock |
|---|---|---|---|
| 1 | Local 5090 only (today's design) | $0 | 8.81 days |
| 2 | Local Arm E + RunPod-5090 Arm T (parallel twin), extension stays local | $31–44 | 7.35 days |
| **3** | **Local Arm E + RunPod-5090 Arm T (parallel twin) + RunPod-H100 extension** | **$115–235** | **3.15–4.85 days** |
| 4 | Fully on RunPod, local idle | $148–215 | 7.4–8.8 days |

**#3 is the recommended plan if migrating without code changes.** Concretely: leave the local
job running Arm E; immediately start a RunPod RTX 5090 pod running Arm T on the same
byte-identical stream/seed so the twin-phase gate lands in ~44h instead of ~78h; once gates
clear, run the 7.5B-token extension on a RunPod H100 pod rather than the local 5090.

Extension-phase cost/time by GPU (7.5B tokens), for reference when choosing where to run it:

| GPU | Hours (optimistic–conservative) | Cost, Community | Cost, Secure |
|---|---|---|---|
| RTX 5090 (measured baseline) | 133.1h fixed | $91.84 | $131.77 |
| RTX PRO 6000 Blackwell | 80–133h | $135–225 | $167–278 |
| A100 SXM 80GB | 89–119h | $124–165 | $142–189 |
| **H100 SXM 80GB** | **31–71h** | **$83–192** | $102–234 |
| H200 SXM 141GB | 31–50h | $111–178 | $142–228 |
| B200 (Secure only) | 12–30h | — | $84–201 |

H100 is the standout: in the optimistic case it beats the 5090 baseline on *both* cost and
time. B200 is higher-risk/higher-reward (newest silicon, widest uncertainty, no Community
capacity).

### 3c. If the training loop gets distributed (DDP) support

Not available today — confirmed by grep, no `torch.distributed`/DDP/FSDP in `kinetic_ai/`.
If added, the 7.5B-token extension (currently a single stream, the dominant 5.5-day chunk of
the programme) could split across GPUs **within one multi-GPU pod**. Total GPU-hours billed
stays roughly constant (same compute, done concurrently) plus a communication efficiency tax
— this is a materially better trade than paying for a faster single chip.

| Config | $/hr (Secure) | Efficiency assumption | Hours | Total cost |
|---|---|---|---|---|
| 1x RTX 5090 (baseline) | $0.99 | — | 133.1h (5.5d) | $131.77 |
| 8x RTX 5090, one pod (PCIe, no NVLink on consumer cards) | $0.99 ea | 85% | ~19.6h (0.8d) | ~$155 |
| 8x H100 SXM, one pod (NVSwitch; Secure only, Community caps at 1 GPU) | $3.29 ea | 92% | 4.2–9.7h | $110–255 |

Combined with §3b's parallel twin phase, a full-programme estimate with distributed extension:
**≈2.3–2.7 days wall-clock, ≈$140–300 marginal spend** — roughly half the wall-clock of
scenario #3 for a comparable dollar range.

Engineering lift to enable this: wrap the model in `DistributedDataParallel` (straightforward
at 913M params — no FSDP/sharding needed, full model fits on one GPU); gradient checkpointing
and the fused AdamW optimizer both compose cleanly with DDP. The real work is making
checkpoint save/resume rank-aware and re-verifying the **bit-exact resume round-trip** the
project already tests for under multi-GPU — new tests, not just a config flag. Per the
project's spec-driven, measured-not-extrapolated discipline, this should get its own small
spec with a preflight GO gate (measure real DDP efficiency at N=2/4/8 before committing an
8-GPU pod's budget), mirroring SPEC 0022's own preflight.

## 4. Constraint: no distributed training today

Confirmed via `grep -rlE "torch\.distributed|DistributedDataParallel|FSDP|deepspeed|accelerate\.Accelerator|init_process_group" kinetic_ai/` — no matches. Parallelism is
currently capped at the two natural streams (Arm E, Arm T); the extension is single-stream and
can only be sped up by picking a faster *single* GPU (§3b), not by adding more pods, unless
DDP/FSDP support is built (§3c).

## 5. Exclusions and flags

- **RTX 4090 (24GB) excluded**: peak VRAM usage (28.6 GiB) exceeds its 24GB, would force a
  smaller batch / more checkpointing, invalidating a direct throughput comparison to the
  measured baseline.
- **RTX PRO 4500 Blackwell**: same 32GB VRAM as the 5090 at a lower Secure-tier price ($0.72/hr),
  but a mid-tier workstation Blackwell card with unmeasured throughput on this workload — could
  beat or lose to the 5090 on $/token; worth a short preflight before trusting it.
- **A100/H100/H200/B200 throughput figures**: extrapolated between a memory-bandwidth-ratio
  lower bound and a dense-BF16-TFLOPS-ratio upper bound relative to the measured 5090 numbers.
  This workload (913M/158M model, gradient checkpointing, SDPA) may be more bandwidth/kernel-
  bound than FLOPS-bound at this scale, so real throughput could land anywhere in — or, less
  likely, outside — the quoted range.

## 6. Before committing spend

1. Re-check RunPod pricing/availability at execution time (queried live 2026-08-30; can move).
2. For any non-RTX-5090 GPU: run a short measured preflight (same GO-rule discipline as SPEC
   0022's local preflight, ≥5.5k tok/s threshold) before trusting the extrapolated numbers in
   §3b/§3c with real budget.
3. Confirm whether the twin-phase kill gate (ppl ratio ≤1.20 at 1B tokens) and success bar
   (≥0.95 ladder ratio at 2.5B tokens) have cleared before spending on the extension phase at
   all — none of the cost estimates above are worth incurring if the gates fail.
4. If pursuing §3c (distributed training), scope it as its own spec (`research/specs/00XX-*.md`)
   with its own preflight gate before renting an 8-GPU pod.
