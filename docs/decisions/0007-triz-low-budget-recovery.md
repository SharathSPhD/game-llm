# ADR 0007 — TRIZ Inventive-Step Analysis: Low-Budget KineticLM Recovery

Date: 2026-08-28 · Status: PRE-REGISTERED · Resolves: F25 → H10 pathway

## Problem Statement (The Binding Constraint)

**Technical Contradiction:**
- **Improving parameter:** Model quality/convergence speed (lower perplexity, faster recovery)
- **Worsening parameter:** Training budget/compute cost (Power: parameter 21)

**Physical Contradiction:**
The weight-tied core MUST have BOTH:
1. High representational quality (recover from ppl 1909 toward 6.01, the base's range)
2. Low training cost (100-300M tokens vs published recipes' 10-100B tokens)

These are contradictory under conventional uptraining approaches because the tying surgery destroys ~68% of base quality, and recovery is empirically logarithmic in compute budget.

**Real numbers (from F25):**
- Base Qwen3-1.7B: ppl 6.01
- Post-surgery (8+8 explicit, 1 core, average init): ppl 1909 (**318x worse**)
- Budget available: 100-300M tokens (~7 hours on RTX 5090)
- Published recovery cost: 10-100B tokens (30-1000x more)
- **Gap:** 30-1000x under-resourced to achieve published results; nevertheless, parameter saving (68% of base) is non-negotiable — it is the entire value proposition.

## TRIZ Session Results

### Contradiction Mapping

Two formulations were analyzed:

| Improving (A) | Worsening (B) | TRIZ Principles | Cell ID |
|---|---|---|---|
| Speed (P9) | Power (P21) | 19, 35, 38, 2 | 9_21 |
| Ease of Repair (P34) | Power (P21) | 15, 10, 32, 2 | 34_21 |

**Key principles returned:**
- **P19: Periodic Action** — replace continuous optimization with batched cycles, match natural problem structure (depths converge at different rates)
- **P35: Parameter Changes** — adjust operating regimes (depth budgets, learning rates, adapter ranks) dynamically
- **P38: Strong Oxidants** — intensify via better initialization and optimized solver environments
- **P2: Taking Out** — isolate recoverable vs non-recoverable components, eliminate wasteful exploration
- **P15: Dynamics** — make system self-tuning; adapt supervision weights and adapter capacity over training
- **P10: Preliminary Action** — pre-compute and pre-arrange best possible initialization before expensive uptraining

## Solution Sketches (Ranked by IFR Score)

All sketches reuse existing repo machinery (F24 anytime, F25 damage curve, exp15 architecture, teacher distillation).

### Sketch 1: Depth-Curriculum Training ⭐ **IFR 3/4**

**Principles:** P19 (Periodic Action), P35 (Parameter Changes)

**Mechanism:**
- Anytime supervision at depths {K/8, K/4, K/2, K} with equal initial weight
- Exponentially decay shallow-depth supervision over training: w(d, t) = exp(−λt) for d < K
- Deep supervision w(K, t) = 1 always
- Early phases force rapid convergence of cheap (few-iteration) solves
- Late phases refine full-depth solves when trajectory is already near-correct

**Why it works:** Matches P19 batching principle — separates high-ROI (shallow, fast convergence) from low-ROI (deep, expensive) work. Front-loads easy wins, eliminates wasted solver budget on low-probability trajectories early on.

**Concrete params:** λ ∈ {0.5, 1.0}, depth budget {K/8, K/4, K/2, K} ⊂ {1, 2, 4, 8} iterations.

**Cost:** Negligible (reuses exp15 anytime infrastructure, same solver budget).

**IFR gap:** Does NOT self-resolve — requires explicit curriculum tuning.

---

### Sketch 2: Per-Layer LoRA Relaxation Adapters ⭐ **IFR 3/4**

**Principles:** P35 (Parameter Changes), P15 (Dynamics)

**Mechanism:**
- Add learnable low-rank modulations (LoRA, rank 8–16) on each of the 8 pre-layers and 8 post-layers
- These adapt outputs to preserve layer-specific routing information that tying destroyed
- During uptraining, gradually reduce adapter rank to zero via scheduled pruning (P(t) = P_init · (1 − t/T))
- At t=T, adapters are zero; model is equivalent to plain tied core
- Gradient flow progressively shifts from adapter paths back through the core

**Why it works:** Matches P35/P15 — dynamically changes parameter count and flexibility, allowing soft interpolation from multi-layer to tied state. The adapters provide "training wheels" that gradually remove themselves, guiding the core to learn what adapters taught.

**Concrete params:** rank ∈ {4, 8, 16}, decay schedule linear or exponential over T steps.

**Cost:** +2–3% final params during uptraining; zero at inference (adapters pruned).

**IFR gap:** Achieves 3/4 (leverages existing LoRA infrastructure, minimal cost, no new problems). Criterion unmet: not fully self-resolving — requires schedule tuning.

---

### Sketch 3: Inverse-Exponential Depth Weighting ⭐ **IFR 3/4**

**Principles:** P19 (Periodic Action), P35 (Parameter Changes)

**Mechanism:**
- Supervise anytime depths {K/4, K/2, K} with CE losses weighted w(d) = exp(−λd/K)
- Initially (λ ≈ 1), shallow depths receive ~5x heavier gradient weight than full depth
- Over training, gradually increase λ (or equivalently, reduce shallow weights) to focus on full-depth refinement
- Allocation: early tokens "unlock" shallow solves (cheap, fast), late tokens refine the entire system

**Why it works:** Matches P19 (batch work into high-ROI phases) and P35 (shift parameter weights). Extracts more value per token from fixed solver budget by front-loading easy wins, then progressively rebalancing.

**Concrete params:** λ(t) = λ_0 + αt, depth weights w(d, t) = exp(−λ(t) · d / K).

**Cost:** Zero (reuses exp15 anytime supervision, no new architecture).

**IFR gap:** Achieves 3/4 (leverages existing, minimal cost, no new problems). Unmet: requires tuning of λ schedule.

---

### Sketch 4: Factorized Token-Budget Allocation ⭐ **IFR 3/4**

**Principles:** P2 (Taking Out), P35 (Parameter Changes)

**Mechanism:**
- **Phase 1 (20%, ~20–60M tokens):** Surgical smoke — verify F25 damage curve for this exact Qwen3-1.7B config (may differ from F25's sample), establish baseline ppl at the 8+8/1 operating point, measure initialization damage for this base
- **Phase 2 (30%, ~30–90M tokens):** Initialization development — learn best convex combination of middle layers (weighted-mean initialization) or best LoRA seed parameters, freeze at end of phase
- **Phase 3 (50%, ~50–150M tokens):** Final uptraining with validated curriculum (from sketch 1/3 above)
- If phase N proves the architecture unmatchable, phases N+1 are cancelled (self-resolver)

**Why it works:** Matches P2 (taking out wasteful exploration) and P35 (parameter allocation). Eliminates 30-100M tokens wasted on hyperparameter sensitivity in published recipes by front-loading cheap validation.

**Concrete allocation:** 20% exploration, 30% initialization learning, 50% uptraining with best-found curriculum.

**Cost:** Uses tokens that would be wasted anyway on exploratory runs; improves efficiency of all subsequent phases.

**IFR gap:** Achieves 3/4 (minimal cost, no new problems, self-resolving). Criterion unmet: does not fully leverage existing (requires new phase structure).

---

### Sketch 5: Hidden-State Intermediate Distillation ⭐ **IFR 2/4**

**Principles:** P2 (Taking Out), P10 (Preliminary Action)

**Mechanism:**
- Distill not just logits but also frozen base's hidden states at depths {K/4, K/2}
- Pre-compute all base layer outputs in a single forward pass (preliminary action; batch-amortized cost)
- Add auxiliary KL loss terms:
  - L_logit = KL(student logits || base logits)
  - L_mid_quarter = MSE(student z_{K/4} || base z_{K/4})
  - L_mid_half = MSE(student z_{K/2} || base z_{K/2})
- Weighted combination: L_total = α·L_CE + β·L_logit + γ·(L_mid_quarter + L_mid_half)

**Why it works:** Matches P10/P2 — intermediate supervision provides explicit guidance at functioning inner layers, accelerating convergence without changing architecture. Pre-computation amortizes cost.

**Concrete params:** β, γ ∈ {0.1, 0.5, 1.0}; layer MSE can substitute for KL if more stable.

**Cost:** Marginal (one teacher forward pass per epoch, easily amortized in batch).

**IFR gap:** Achieves 2/4 (minimal cost, no new problems). Unmet: does not leverage existing LoRA/curriculum infrastructure; requires new loss terms.

---

### Sketch 6: Weighted-Mean Initialization with Learned Interpolation ⭐ **IFR 2/4**

**Principles:** P10 (Preliminary Action), P35 (Parameter Changes)

**Mechanism:**
- For the first 100 gradient steps, initialize core block as learnable convex combination of all 28 middle layers: core(θ) = Σ w_i · layer_i, where w_i are learned weights (normalized to sum to 1)
- This captures heterogeneity before tying destroys it
- After 100 steps, freeze learned weights to their mean values: w̄_i = mean(w_i), then train normally
- The 100-step "interpolation phase" costs negligible compute relative to full uptraining

**Why it works:** Matches P10 (preliminary action: pre-arrangement before expensive training). Leverages the fact that elementwise mean cannot capture layer correlation or importance ordering — learned interpolation discovers it at minimal cost.

**Concrete params:** 100-step interpolation phase, learned softmax(α) weights, then freeze to mean.

**Cost:** ~0.1% of uptraining budget (100 steps out of 100k+).

**IFR gap:** Achieves 2/4 (minimal cost, no new problems). Unmet: does not leverage existing (requires new learned-interpolation code), not self-resolving (requires tuning of interpolation period).

---

## Ranked Recommendation (by IFR and Reusability)

**Tier 1 (Launch first — IFR 3/4, reuse existing machinery):**
1. **Sketch 1: Depth-Curriculum** — P19/P35, no new code, uses exp15 anytime, highest confidence
2. **Sketch 2: LoRA Relaxation Adapters** — P35/P15, reuses Relaxed Recursive Transformers baseline, self-tuning quality dial

**Tier 2 (Parallel arms if compute permits — IFR 3/4):**
3. **Sketch 3: Inverse-Exponential Weighting** — P19/P35, trivial to implement as anytime weight schedule, low-risk variation
4. **Sketch 4: Factorized Budget** — P2/P35, meta-level optimization, improves efficiency of all arms

**Tier 3 (Conditional on Tier 1 results — IFR 2/4):**
5. Sketch 5 & 6: Only if Tier 1 plateaus; both require new machinery with higher implementation risk.

## Pre-Registered Decision

**Experimental lanes (SPEC 0014):**
- **A1 (Baseline):** Qwen3-1.7B post-surgery, no uptraining (measures damage: 1909 ppl)
- **A2 (Sketch 1):** Depth-curriculum training with exponential decay of shallow supervision
- **A3 (Sketch 2):** LoRA relaxation adapters with rank-decay schedule

**Hypothesis:** At least one Tier-1 arm achieves ≥85% quality recovery (ppl ≤ 110 vs base 6.01, or equivalently, lm-eval retention ≥0.80 of base) on 150–300M tokens within 7–14 hours on RTX 5090.

**Success definition:** Honest progress trajectory toward published results (10-100B tokens) using only our budget, with identified mechanism and replicable ablations.

**Null closure:** If both arms miss <0.75 retention, the contradiction is genuine and the next iteration pivots to multi-block DEQ or structural changes to the weight-tied core (future program phase).

---

**Prepared:** 2026-08-28 · **Approved by:** TRIZ analysis + repo machinery audit
