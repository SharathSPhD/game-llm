# SPEC 0014 — RQ-11 (H10): Low-Budget KineticLM Recovery (TRIZ-Guided)

Status: PRE-REGISTERED · GPU: RTX 5090 (uptraining) + GB10 (eval) · Pre-registered 2026-08-28
Depends on: ADR 0007 (TRIZ analysis), F25 (damage curve). Closes H10. Carries forward F24/B1 (anytime), anytime
supervision + teacher distillation from exp15.

## Question

The weight-tied KineticLM conversion of Qwen3-1.7B destroys ~95% of model quality (ppl 6.01 → 1909).
Published recursive-uptraining recipes recover this loss on 10–100B tokens. **We have 100–300M tokens.**

Can TRIZ-identified mechanisms (depth-curriculum, LoRA relaxation, inverse-exponential supervision weighting)
recover ≥75% of base quality on a 30–1000x smaller budget than published recipes?
This would establish the binding constraint (token budget) as the problem, not the architecture.

## Hypothesis (H10)

At least one of two principled arms (depth-curriculum, LoRA relaxation) achieves **≥80% quality retention**
(held-out perplexity or lm-eval benchmark retention) relative to base Qwen3-1.7B, on a budget of 150–300M tokens,
within 7–14 wall-clock hours on RTX 5090. A documented miss at <75% retention closes H10 as a valid null
(the contradiction is genuine: this architecture cannot be recovered at this scale of budget).

## Method (Pre-Registered)

### Baselines and Arms

| Arm | Name | Init | Training | Teacher | Loss Formula | Notes |
|---|---|---|---|---|---|---|
| A1 | Post-surgery baseline | Average mean (F25) | None | N/A | Cross-entropy only | Measures damage: ~1909 ppl (starting point) |
| A2 | Depth-curriculum (Sketch 1) | Average mean | Anytime CE + KL distill | Frozen Qwen3-1.7B | α·KL + (1−α)·[CE@K/4 + CE@K/2 + CE@K], depth weights decay exp(−λt) | 150–300M tokens, exp decay schedule λ(t) |
| A3 | LoRA relaxation (Sketch 2) | Average mean + LoRA rank-16 | Anytime CE + KL distill | Frozen Qwen3-1.7B | α·KL + (1−α)·[CE@K/4 + CE@K/2 + CE@K], adapter rank → 0 over training | 150–300M tokens, linear rank decay |

### Surgical Configuration (Fixed for All Arms)

- **Base model:** Qwen3-1.7B (28 layers, d=2048)
- **KineticLM topology:** 8 pre-layers + 1 shared core (iterated 8×) + 8 post-layers
- **Parameter count:** 1.167B (~68% of base), per F25 damage curve
- **Initialization:** Elementwise mean of middle 12 layers → core block (established in F25 as superior to stepwise)
- **Tokenizer:** Native Qwen3 (vocabulary unchanged)

### Training Protocol

**Corpus:** FineWeb-Edu (100B public, streaming). Token budget per arm: **150–300M** (pre-registered), with actual
run to determine final value based on wall-clock time constraint (≤14h on RTX 5090).

**Hardware & precision:** RTX 5090, bfloat16, gradient checkpointing, seq_len=2048.

**Optimizer:** AdamW (lr=1e-4, β₁=0.9, β₂=0.999, wd=0.01), warmup 5% of steps, cosine decay.

**Teacher model:** Frozen Qwen3-1.7B base, forward passed once per batch for logits + intermediate representations
(at depths K/4, K/2 if distillation arm selected).

**Loss weighting (arms A2/A3):**
- α = 0.3 (KL weight)
- CE weight = 0.7, distributed across anytime depths {K/4, K/2, K}
- **A2 (depth-curriculum):** w(d,t) = exp(−λ(t) · d/K), where λ(t) = 1.0 + 0.5·(t/T)
  - Shallow depths {K/4, K/2} start 5× heavier, decay to equal weight by end of training
- **A3 (LoRA relaxation):** uniform w(d) = 1/3 across all depths; adapter rank p(t) = 16 · max(0, 1 − t/T)
  - Linear rank decay from 16 → 0 over training horizon T

**Checkpointing:** Save at 10%, 25%, 50%, 75%, 100% of budget (5 checkpoints per arm) for budget-dial evaluation.

### Evaluation Protocol

**Held-out test set:** 3–5% of FineWeb-Edu sampled at token-level (not biased by sequential corpus effects).

**Primary metrics** (held-out token-level):
- Perplexity (exp of NLL), reported as scalar
- Retention ratio = base_ppl / model_ppl (e.g., 0.80 means 80% recovery of base quality)

**Secondary metrics (lm-eval-harness on 100–200 held-out examples per benchmark, GB10 eval):
- ARC-Challenge (0-shot)
- HellaSwag (0-shot)  
- GSM8K (8-shot, no calculator)
- MMLU (5-shot)
- **Headline:** average across benchmarks, reported as retention vs base scores

**Anytime property (budget dial):** For each checkpoint (10%, 25%, 50%, 75%, 100%), report ppl and retention.
This tests whether training obeys P11 anytime monotonicity (higher budget ≥ better quality).

### Pre-Registered Scoring Gates

| Gate | Criterion | Evidence | Outcome |
|---|---|---|---|
| **MET** | A2 OR A3 achieves ppl ≤ 110 (ret ≥ 0.80) at ≤300M tokens, anytime monotone | ≥80% of base quality on benchmarks (lm-eval) | Win: mechanism works at this budget scale |
| **PARTIAL** | One arm hits 0.75–0.80 retention, or MET on anytime property but not final quality | 75–80% recovery, or budget dial smooth but quality is weak | Honest progress; next iteration identified |
| **MISSED** | Both arms <0.75 retention | <75% recovery on benchmarks | Valid null; contradiction may be genuine (architecture-level problem, not budget) |

**Key covenant:** If both arms MISS, we do NOT claim failure of weight-tying or anytime training (both are validated
on smaller scales in F20–F24). Instead: "The Qwen3-1.7B → KineticLM conversion damage (F25) cannot be recovered
on 100–300M tokens by curriculum or LoRA relaxation alone. Next iteration explores multi-block DEQ or structural
core changes." This is a documented, replicable null.

### Honest Confounds & Caveats

1. **Tokenizer & vocabulary:** Qwen3 differs from BabyLM (our pretraining base). Benchmarks trained on different corpora.
   Retention numbers are adjusted for domain differences (FineWeb-Edu vs BabyLM), not absolute.

2. **Inference vs training computation:** A2/A3 train with anytime supervision (multiple depths), but evaluation uses
   standard 8-iteration solve (F24 eval-path). The mismatch (Anderson vs plain iteration) may favor or penalize
   depending on whether the learned trajectory matches the standard solver path.

3. **Hyperparameter sensitivity:** λ(t) schedule, α, adapter rank decay, and checkpoint timing are pre-registered,
   but minor tuning (±20%) may improve results. Any tuning applied is logged and rescoped as a rider arm.

4. **Budget definition:** "150–300M tokens" includes all forward passes (base teacher forward, student forward),
   not just unique data tokens. Wall-clock time (7–14h) is the actual binding constraint.

5. **Single seed for screen:** Unless extended by operator, initial run is a single seed; multi-seed validation
   (if quality is close to gate) requires separate 3-seed runs.

## Runtime Estimate

- **Arm A1 (post-surgery, no training):** ~30 min (smoke eval on 10k examples)
- **Arm A2 (depth-curriculum, 150–300M tokens):** ~7–14 h (5090, bf16, grad checkpoint, seq 2048)
- **Arm A3 (LoRA relaxation, 150–300M tokens):** ~7–14 h (same as A2)
- **Eval (all arms, GB10):** ~2 h per arm (lm-eval ARC/HellaSwag/GSM8K/MMLU)

**Total wall-clock:** ~24–32 hours (split across 5090 training + GB10 eval).

## Valid Closure

Per CLAUDE.md closure contract, a documented MISSED verdict on H10 (both arms <0.75 retention) closes this
RQ-11 iteration as a valid null. The next iteration pivots to:
- Multi-block DEQ (learning multiple weight-tied cores instead of one)
- Structural re-design of the core block (e.g., gating mechanisms to reduce effective weight-tying)
- Expansion to 500M–1B token budgets (data-availability gate, not architecture gate)

The key outcome is replicable diagnostic: "Curriculum and adapter relaxation do not recover Qwen3-1.7B under
100–300M tokens; the architecture or budget is too tight."

---

**Prepared:** 2026-08-28 · **Authorized by:** ADR 0007 (TRIZ inventory) · **Approval:** Operator sign-off pending
