# SPEC 0012 — RQ-9 (H8): parameter-space magnetic anchoring (PMA) vs DPO/SimPO at 1.7B

Status: ACTIVE · GPU: 5090 · Pre-registered 2026-08-28 · ADR 0006

## Naming (binding, per ADR 0006)

Our method is **PMA — parameter-space magnetic anchoring**: MagneticAdamW pulls
weights toward the FROZEN BASE WEIGHTS while optimizing the standard DPO loss.
This is NOT the published **MPO** (arXiv 2410.16714, ICLR 2025), which applies
magnetic mirror descent in POLICY space under self-play toward the Nash
equilibrium of a preference game. The paper must cite that work and must not
present F21's negative dose-response as evidence against it.

## Question

At real scale and with real preference data, does the parameter-space magnet
buy anything over DPO — and does F21's 121M null (magnet second-order to the
DPO gradient across tau in [1e-3, 10]) hold, or was it an artifact of scale?

## Design

- Base: Qwen3-1.7B (instruct variant), bf16, TRL DPOTrainer with our optimizer
  injected — the loss is identical across arms; only the optimizer differs.
- Data: UltraFeedback-binarized (public standard), ~60k pairs, held-out split.
- Arms: **D1** DPO (MagneticAdamW tau=0 == decoupled AdamW, same code path);
  **D2** PMA tau in {1e-4, 1e-3, 1e-2} (scaled for lr 5e-7..1e-6 and ~2k
  steps — the F21 lesson is that tau must be set by lr*tau*T, not copied);
  **D3** SimPO (published reference-free baseline, TRL implementation).
- Metrics: held-out preference accuracy; IFEval (instruction following);
  KL-to-base drift on held-out text; MMLU/GSM8K regression check
  (alignment tax); training wall-clock.

## Scoring (pre-registered)

- **MET:** PMA >= DPO on held-out accuracy AND lower KL drift AND no worse
  alignment tax, at some tau in the registered range, on >= 2 of 3 seeds.
- **PARTIAL:** one of the two halves.
- **MISSED:** neither — which would CONFIRM F21 at scale and is a publishable
  negative (the magnet's home is pretraining stability, not preference drift).

## Runtime

~4–8 h/arm on the 5090 (LoRA or full bf16 with grad checkpointing).
