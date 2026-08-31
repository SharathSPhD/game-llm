# Architecture analysis — what prabhasa-samskrutam's configuration says about the tying claim

**Status:** decision support, not an ADR. No decision has been made and none is proposed here.
**Date:** 2026-08-31.
**Scope:** written for the agent working this repository. It concerns kinetic-ai's own exchange-rate
claim first and the prabhasa relationship second.
**Lock respected:** SPEC 0022 §"Independence" states no prabhasa integration work happens in this
repository. Nothing below asks for integration. The proposal in §8 is a **kinetic experiment run in
kinetic's harness**, and it happens to answer the integration question as a side effect.
**Sources read:** `research/specs/0022-twin-at-1b.md`, `kinetic_ai/models/eqlm.py`,
`kinetic_ai/models/deq_layer.py`, `docs/PLAN.md`, git log through `926181e`;
`prabhasa-samskrutam/configs/train/nemotron_h_1b.yaml`,
`research/findings/{vak-yantra-phase1-report,r2-gate-summary}.md`, `research/journal.md` (H-ORD R0–R5),
`docs/vak-yantra-architecture.md`, ADR-0004/0010/0011/0012.

---

## 1. Why this matters to kinetic before it matters to prabhasa

F45/F50 established the exchange rate at 46–121M: at equal compute the tied block reaches
0.958 ± 0.017 of an explicit transformer's quality with 2.70× fewer resident parameters. SPEC 0022
takes that to d=2048, where the geometry predicts ~5.8×.

**Both of those ratios are capped by the tokenizer, and the record does not say so.**

In SPEC 0022 the tied embedding/head is 107M parameters at vocab 50257 × d 2048. That 107M is
present in *both* arms and cannot be tied away — it is already shared. So the ratio is not
"block savings"; it is block savings diluted by a fixed embedding overhead:

```
Arm E = 16 × block + emb        Arm T = 1 × block + emb
ratio = (16·B + E) / (B + E)
```

With B ≈ 50.4M and E ≈ 107M the ratio is 5.8×. With E → 0 the same arithmetic gives 16× — the
depth itself. **The headline number is a function of the vocabulary, and no run in F1–F54 varies
that axis.**

This is a real gap in the record, and it cuts both ways. A skeptical reviewer can argue that part
of the measured advantage is an embedding-sharing artifact common to both arms rather than a
property of tying. A byte-level cell answers that directly, and it is the cell where the claim is
hardest to make — no embedding to hide behind.

Prabhasa's 1B configuration happens to be exactly that cell, already built and already trained,
which is how this came to attention.

---

## 2. Ground truth

**kinetic — `kinetic_ai/models/eqlm.py` (995 LOC) + SPEC 0022**

Pre-LN transformer block with damped residuals; `residual_damping` 0.2; `spectral_norm` True;
Anderson solver (`anderson_m` 5, `anderson_beta` 1.0) at eval; anytime-unrolled supervision at
iterations [6, 11, 16] with weights [0.15, 0.3, 1.0]; LM head weight-tied to the token embedding.
Contraction is argued by Banach: `f(z) = (1−α)z + α(z + g(z))` with spectral norm on the block
damps the derivative so σ(f) < 1.

SPEC 0022 geometry: vocab 50257 (padded 50304), seq 2048, d_model 2048, n_heads 16, d_ff 8192.
Arm E ≈ 0.92B (16 × 50.4M block + 107M tied embedding/head). Arm T ≈ 158M. Disclosed asymmetry:
anytime supervision computes the 103M head at three depths, ≈13% extra training FLOPs charged
to Arm T.

**prabhasa — `configs/train/nemotron_h_1b.yaml`**

```
d_model 1536 · n_layers 32 · attention_every 8 (4 attention, 28 Mamba-2)
n_heads 24 (head_dim 64) · d_ffn 6144 · mamba_d_state 128 · mamba_expand 2
vocab_size 256          <-- byte-level
seq_len 4096 · global_batch_tokens 1048576 · token_budget 5.6e9
max_epochs_per_source 4.0
```

---

## 3. The arithmetic

Approximate decomposition at d=1536 (two-matrix FFN):

| Component | Params |
|---|---|
| Mamba-2 mixer (in_proj + conv + out_proj) | ~15.0M |
| FFN (1536 × 6144 × 2) | ~18.9M |
| Attention mixer (4 × 1536²) | ~9.4M |
| Mamba layer (mixer + FFN) | ~33.9M |
| Attention layer (mixer + FFN) | ~28.3M |
| 28 Mamba + 4 attention | ~1.06B |
| **Embedding + head (2 × 256 × 1536)** | **~0.79M — 0.07% of the model** |

Nemotron-H is heterogeneous, so the natural tying unit is the **8-layer motif** (7 Mamba +
1 attention), repeated four times. Tying the four repeats gives ~266M resident, a **4.0×**
reduction — and unlike SPEC 0022's 5.8×, essentially none of it is lost to embedding overhead.

The general statement is the one that matters here:

> **At byte-level vocabulary the tying ratio converges on the iteration count.** Nothing dilutes it.
> The property that makes kinetic's ratio mediocre at 121M (embedding-dominated, 2.70×) is
> structurally absent.

Second consequence: the disclosed ~13% anytime-supervision overhead is a function of head size.
At 0.39M rather than 103M, computing the head at three depths is free. **The main disclosed cost of
the F24 recipe nearly vanishes at byte level.**

---

## 4. The crux — depth or parameters?

This is the strongest reason to run the experiment and it is a question about kinetic's claim, not
prabhasa's.

Prabhasa's R5 probe ladder (order-rarity representation probe, held-out test correlation):

| Arm | Probe | Delta |
|---|---|---|
| 353M baseline | 0.179 | — |
| 353M maximally objective-trained | 0.241 | +35% from the objective |
| 1.13B baseline | 0.276 | **+54% from scale alone** |
| 1.13B grammar-shaped | 0.287 | +4% from the objective |

The 1.13B *untrained-on-calibration* baseline exceeds the maximally objective-trained 353M model.
Their journal's conclusion: *"the evidenced levers are (1) SCALE and (2) richer signal / GOLD labels."*

But 353M → 1.13B raised **parameters and compute-depth together.** Nobody has isolated which one
bought the +54%.

kinetic's own record has the same confound. F45/F50 compare tied against explicit at matched
compute — which means matched depth. The exchange rate says "tied is nearly as good at this
geometry"; it does not say whether the capability that survives tying tracks depth or tracks
parameter count. That distinction determines whether tying is nearly free or quietly expensive,
and it is the thing a reviewer will ask about a 5.8× claim.

**A tied model with N-layer-equivalent depth and 1/N the parameters is the instrument that
separates them.** Kinetic needs the answer for its own headline; prabhasa needs it to decide
whether the next rung should be wider or deeper. It is the same measurement.

The honest tension, stated for the record: if the +54% came from parameters, tying costs directly
and a byte-level tied model may reproduce a 353M-class ceiling rather than a 1.13B-class one. That
is a possible outcome of the probe in §8 and it should be pre-registered as such.

---

## 5. A halting criterion the record does not contain

EqLM at inference iterates the Anderson solver to a fixed point, halting on a numerical residual
‖z_{k+1} − z_k‖ < ε. ε is a tuned engineering constant with no meaning outside the optimisation.

`prabhasa-samskrutam/docs/vak-yantra-architecture.md` §1.3 independently defines a quantity with the
same shape:

> "Well-formedness has a native score: **residual unsaturated expectancy** at sentence end. 'The model
> finds this ungrammatical' = 'the frontier won't close' — an interpretable, checkable quantity
> (*nirākāṅkṣatva*)."

Substituting one for the other — halt when the expectancy frontier closes rather than when an
activation norm settles — yields:

1. A halting rule with semantics rather than a tuned constant.
2. Adaptive compute that is *interpretable*: ambiguous input (syncretic vibhakti, competing sandhi
   splits) iterates longer; unambiguous input halts early. Observable, not merely asserted.
3. A claim with no precedent that this analysis could find. Implicit-depth models exist;
   grammatically-terminated implicit-depth models do not.
4. Low cost. Prabhasa's kāraka probes already read roles off hidden states at 62–68% with 97–101%
   retention under licit scrambling (their F2). The probe becomes the halting head.

**It also survives their F7 null**, which is the reason it is worth recording rather than dismissing.
F7 killed a *module added to carry structure* — the saṃsarga memory, whose ablation reproduced the
full result to the third decimal. Their own post-mortem: *"the level was the training objective, not
the forward graph."* A halting criterion adds no module and carries no structure; it constrains an
inference procedure that already runs. It sits at a third level, alongside the objective-level organs
that worked rather than the module-level ones that did not.

Recorded as an idea with a cheap first test (does frontier-closure correlate with solver residual on
existing checkpoints?), not as a work item.

---

## 6. What would not transfer

| Component | Assessment |
|---|---|
| **Equilibrium Council (SPEC 0015)** | Do not propose. F41 was a conditional win; F54's capacity audit found the council loses to a single 7B by 19 points at matched capacity. Note their existing structure — one generator plus a symbolic Z3/hetvābhāsa referee — is asymmetric and is *not* the configuration F54 tested, so F54 does not condemn it. It is simply not evidence in favour. |
| **MMD magnetic anchor, QRE decoding, token auctions** | `docs/PLAN.md` O2 records that implicit depth carries the flagship while the other strands were measured without winning. No basis for spending scarce GPU time proposing them elsewhere. |
| **GPT-2 BPE tokenizer** | Byte-level is their morphological premise (their tax-locus finding is measured *per byte*, 56% at word-onset byte-1) and, per §3, also their tying advantage. |

---

## 7. Risks

**R1 — EqLM's block is a transformer; their spine is Mamba-2.** The substantive engineering risk.
The contraction guarantee (spectral norm + α=0.2 damping ⇒ σ(f) < 1, Banach) is derived for a pre-LN
transformer block. Mamba-2 carries its own recurrence over time, with stability governed by the
selective-scan discretisation and the A-matrix parameterisation. **Composing depth-recurrence over
time-recurrence is unstudied**, and spectral norm on `in_proj`/`out_proj` does not obviously bound the
SSM's internal gain. Nothing in either repository speaks to it. *The probe in §8 avoids this entirely.*

**R2 — Stack integration.** They run Megatron-Core/NeMo; EqLM lives here. Cross-layer tying is what
pipeline parallelism dislikes. At 1–3B on a single GPU pipeline parallelism is unnecessary, so this is
survivable, but the anytime-unrolled loss and Anderson solver would need reimplementation in their
trainer. *Also avoided by §8.*

**R3 — Byte-level is unvalidated for this claim.** F1–F54 is entirely BPE-tokenised English. Byte
sequences are ~4× longer for the same content with different local statistics. That tying survives the
transition is a hypothesis. The binding practice here is that measured beats extrapolated — the GB10
lesson, 520 tok/s measured against 11k assumed, is written into SPEC 0022's GO rule for exactly this
reason.

**R4 — GPU contention.** `docs/PLAN.md` O8: both machines used, never contending; GPU lock in
`state.json`; never two training jobs at once. SPEC 0022 owns the 5090 through the twin and extension
phases. Anything in §8 queues behind it or waits for a window.

**R5 — Corpus discipline.** If the Sanskrit cell in §8 is ever run, prabhasa's corpus charter applies:
every source carries a licence tag and train/publish flags, and contamination is audited before GPU
pretraining. Their holdout hashes exist (`data/holdout/holdout_hashes.json`); a kinetic run must use
them rather than re-deriving a split.

---

## 8. Proposed probe — a missing cell in kinetic's own grid

Not an integration. A cell of the SPEC 0022 design that varies the one axis SPEC 0022 holds fixed.

### 8.1 The pre-registered prediction

Holding architecture, recipe and data constant and changing **only the tokenizer**, the resident-parameter
ratio should move from 5.8× to approximately the iteration count, because the embedding term goes to zero:

| Cell | Vocab | d_model | Depth | Arm E | Arm T | Predicted ratio |
|---|---|---|---|---|---|---|
| SPEC 0022 (running) | 50257 | 2048 | 16 | ≈0.92B | ≈158M | 5.8× |
| **Proposed C1** | **256** | **1536** | **16** | **≈453M** | **≈29M** | **≈15.8×** |

`Arm E ≈ 16 × 28.3M + 0.4M`; `Arm T ≈ 28.3M + 0.4M`. The parameter prediction is arithmetic and is
certain. **The quality ratio is the experiment.**

This is a sharper claim than SPEC 0022's, because it is a stress test rather than a favourable case:
if the quality ratio holds at vocab 256, the tying claim is materially stronger and the embedding-artifact
objection is closed. If it fails, F45/F50's ratio was partly embedding-sharing and the record should say so.

### 8.2 Cells

**C1 — vocabulary isolation (the one that matters).** Byte-tokenised FineWeb-Edu, same source shards as
SPEC 0022's pack so the *content* is identical and only the tokenization differs. Twin protocol exactly
as SPEC 0022: byte-identical stream, fixed consumption units from one shuffled order, matched LR schedule
in token space, arms sequential on one GPU. This isolates the vocabulary variable against a result this
repository will already own.

**C2 — small-corpus regime (optional, later).** Same geometry on prabhasa's Sanskrit pack under R5's
corpus discipline. Tests the second hypothesis: that tying helps most where tokens are scarce. Their 1B
sits ~3.8× under Chinchilla (1.06B × 20 ≈ 21B optimal against a 5.6B budget), and a motif-tied 266M would
sit at ~5.3B optimal against the same 5.6B — within ~5%. Suggestive, not decisive: no established scaling
law covers tied models, and the effective capacity of an iterated block lies somewhere between one block
and N blocks. C2 is worth running only after C1 reads positive.

### 8.3 Readouts

Held-out bits-per-byte (not perplexity — byte-level, and bpb is what the comparison corpus reports);
the resident-parameter ratio measured rather than computed; the anytime retention curve at iterations
6/11/16; and, for §4, a capability probe held at fixed depth while parameters vary, which is the only
readout that separates depth from parameter count.

### 8.4 Gates, in SPEC 0022's form

- **Preflight GO rule.** Measured tok/s at final geometry over 30 post-warmup steps. Extrapolated
  throughput is not accepted.
- **Kill gate.** Arm T held-out bpb ≤ 1.20 × Arm E bpb at identical tokens. Failure records a NULL for
  byte-level tying and closes the cell.
- **Success.** bpb ratio ≤ 1.10, or the parameter ratio ≥ 12× with a bpb ratio ≤ 1.15. Meeting either
  upgrades the exchange-rate claim along the vocabulary axis; meeting neither while passing the kill gate
  is reported with the measured ratio, not rounded up.

### 8.5 Cost and placement

d=1536, depth 16, seq 2048 is ~450M explicit — substantially cheaper than SPEC 0022's 0.92B arms. Sized
to a short token budget it is a days-scale run, not a weeks-scale one. It belongs in a future
`research/cycles/cycle<N>_candidates.json` as an EFE candidate, scored against the alternatives, and it
queues behind SPEC 0022 on the 5090 per O8. It is not urgent and it is not a reason to disturb the
running twin.

---

## 9. What this analysis does not establish

1. **The parameter decomposition in §3 is estimated from the config, not measured.** The Mamba-2 figure
   in particular depends on `n_groups` and head layout not pinned in the YAML. Verify against a loaded
   checkpoint before quoting the 4.0× motif figure anywhere.
2. **No tied Mamba-2 block has been trained by anyone, as far as this analysis found.** R1 stands
   unquantified. The §8 probe deliberately does not test it.
3. **The Chinchilla arithmetic in §8.2 is suggestive only.** Tied-model scaling laws do not exist.
4. **The §4 crux is a confound identified, not resolved.** Nothing here answers whether the +54% came
   from depth or parameters; the probe is proposed because the answer is unknown, not because it is
   suspected.
5. **The halting-criterion idea in §5 is untested at any scale**, including the cheap correlation test
   suggested there.
6. Every prabhasa number quoted is from their record under silver labels and, for the contrastive
   results, one seed per scale. Their own bounds section says so.

---

## 10. Recommendation

**Record C1 as an EFE candidate for a future cycle; propose nothing else.**

The reason to run it is not the prabhasa relationship. It is that C1 varies the one axis F1–F54 never
varied, on the claim this programme leads with, in the direction where the claim is hardest to defend.
A positive result closes the embedding-artifact objection and roughly triples the headline ratio. A
negative result is a finding worth having before the ratio appears in a paper.

That it also answers the integration question — cheaply, in this harness, without porting anything or
touching Mamba-2 or Megatron — is a side effect, and is the reason the SPEC 0022 independence lock costs
nothing to keep.

**The reverse flow, which is presently the stronger one.** Prabhasa's quotient and contrastive-margin
objectives are architecture-agnostic loss terms: penalise the squared difference in sentence-NLL across
two licit reorderings, plus a hinge requiring licit permutations to price below matched illicit ones.
Applied as a ~2.5h finishing pass on a pretrained spine they close essentially all of an excess ordering
tax at 353M and 94% at 1.13B, scale-stable to three decimals (P1/P3 0.382 vs 0.385), **while improving
canonical-text bpb** — negative cost on the base task. They would apply to EqLM exactly as they apply to
Nemotron-H. Nothing in F1–F54 makes a model price a formal structure correctly at negative cost.

On today's evidence, if either programme should take something from the other, **it is this repository
adopting that objective** — which needs only a loss term — rather than prabhasa adopting this
architecture, which needs the C1 result first.
