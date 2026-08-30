# Compute estimate — prune-and-distill from an open teacher, with prabhasa's Sanskrit reasoning folded in

**Status:** decision support, not an ADR — no decision has been made to pursue this path.
**Date:** 2026-08-30. **Supersedes** the "via teacher" analysis in
`docs/qwen3-class-pretraining-estimate.md` §4, which modeled distillation as a from-scratch
student trained alongside a teacher (no proven discount, per F53). This document instead prices
NVIDIA's own published **prune-and-distill** technique (Minitron), which is a fundamentally
different and far cheaper regime, and adds a plan to fold `prabhasa-samskrutam`'s
Sanskrit/kāraka/Nyāya reasoning work in as a low-cost second stage rather than discarding it.
**Author context:** produced by an agent session in worktree `kinetic-ai-runpod-estimate-76abc3`
on operator request. Not reviewed or acted on.

## 1. What changed since the last estimate

The prior document (`docs/qwen3-class-pretraining-estimate.md`) priced two paths: full
from-scratch pretraining (six-to-seven figures) and teacher-guided from-scratch training (no
proven discount per F53, likely *more* expensive than from-scratch). Neither gets near the
$20-30K target.

A third path exists and is NVIDIA's own published production technique for exactly this class of
model, not a speculative architecture discount: **structured pruning of an already-trained,
already-competitive larger model, followed by continued training with knowledge distillation
from the original (unpruned) model as teacher.** This is categorically different from training a
student from scratch against a teacher — the student starts with the teacher's weights
(subsetted), not random initialization, so it needs vastly fewer tokens to recover quality.

## 2. Published numbers (not extrapolated)

- **Nemotron-Mini-4B-Instruct** (`nvidia/Nemotron-Mini-4B-Instruct` model card) is a fine-tune of
  `nvidia/Minitron-4B-Base`, itself pruned and distilled from Nemotron-4-15B. It was never
  pretrained from scratch.
- **Minitron** (arXiv 2408.11796, "LLM Pruning and Distillation in Practice"): pruning
  Llama-3.1-8B down to 4B and continuing training with distillation used **94 billion tokens** —
  a **150x reduction** against the 15T tokens the 8B teacher itself consumed. Reported result: up
  to 16% better MMLU than an equivalent-size model trained from scratch. NVIDIA's summary states
  the *family-wide* compute saving (parent + all pruned children) is 1.8x; the *marginal* cost of
  each additional pruned-and-distilled child is the 40-150x-fewer-tokens figure.
- **Nemotron-H-8B** (arXiv 2504.03624), a from-scratch hybrid Mamba-Transformer, trained on
  **1 trillion tokens** — 125 tokens/param, notably below the ~1,875 tokens/param this project
  previously anchored to Llama-3-8B. Reported on-par-or-better accuracy vs Qwen-2.5-7B/72B and
  Llama-3.1-8B/70B.
- **No dollar training cost is published anywhere found for any of these models.** All $ figures
  below are derived from the disclosed token counts using this project's own FLOPs/GPU-cost
  methodology (§3), not sourced from NVIDIA.

Sources: [Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct) ·
[Minitron paper](https://arxiv.org/abs/2408.11796) ·
[SuperAnnotate summary](https://www.superannotate.com/blog/llm-pruning-distillation-minitron-approach) ·
[Nemotron-H paper](https://arxiv.org/abs/2504.03624)

## 3. Method

Same `C ≈ 6·N·D` base as prior estimates, 35% MFU assumption, same RunPod GPU catalog. Compute
for the distillation stage is student-own-training plus teacher-inference overhead:

$$C \approx 6 N_{\text{student}} D + 2 N_{\text{teacher}} D$$

Token budget `D` = 94B, Minitron's own disclosed figure for an 8B→4B pruning — used directly, not
scaled, since it is the only concrete anchor available; §5 flags where this may not transfer.
Teacher assumed at ~7-8B params (an existing open, ideally Apache-licensed, model — see §5).

## 4. Cost by target size and GPU

| Target | Compute (FLOPs) | H100 Community | H100 Secure | RTX 5090 Community | RTX 5090 Secure |
|---|---|---|---|---|---|
| 1B | 2.07e21 | $4,463 | $5,458 | $7,970 | $11,435 |
| 1.7B (PRD primary target) | 2.46e21 | $5,318 | $6,504 | $9,491 | $13,617 |
| 3B | 3.20e21 | $6,900 | $8,439 | $12,317 | $17,671 |

All cells land under the $20-30K target — the first estimate in this line of analysis that
clears the bar, and it rests on a reproducible published technique rather than an assumed
discount.

### Folding in prabhasa-samskrutam's Sanskrit reasoning work

Prabhasa's own successful training stages ran on 350M-5.25B tokens per stage (M2: ~650M, M3:
350M-780M, M4: 5.25B — see the companion investigation in the prior conversation turn). Adding a
Sanskrit/kāraka/Nyāya continued-training stage on top of the pruned-and-distilled model, at a
comparable 5B-token scale, for the 1.7B target:

$$C_{\text{sanskrit stage}} = 6 \times 1.7\text{e}9 \times 5\text{e}9 = 5.1\text{e}19 \text{ FLOPs}$$

roughly **2% of the main 2.46e21 distillation budget** — a few hundred dollars, not a material
addition. This is the mechanism for keeping prabhasa's Sanskrit-grounded reasoning as a genuine
product differentiator rather than discarding it: the model gets pruned-and-distilled English
competence from an existing capable teacher, then a cheap second stage on prabhasa's
kāraka-role/Nyāya-verification material layered on top.

## 5. What this estimate does not prove

1. **Compression ratio risk.** Minitron's tested ratios were 15B→8B (1.9x) and 8B→4B (2x). This
   estimate applies the 94B-token budget to going as far as 8B→1B (8x) — well beyond anything
   published. Quality risk rises with compression ratio; the 3B target (8B→3B, 2.7x) is closer to
   tested territory and the more defensible of the three sizes. The 1B figure should be read as
   optimistic, not established.
2. **Teacher licensing** needs checking before committing: Qwen2.5-7B (Apache 2.0, clean),
   Llama-3.1-8B (Llama license, usage restrictions), Nemotron-H-8B (NVIDIA Open Model License).
3. **Engineering is real but not quantified here.** NVLabs' Minitron pruning code is
   open-sourced (github.com/NVlabs/Minitron) and reusable rather than build-from-scratch, but
   adapting it to this project's stack, plus integrating prabhasa's Nyāya-verifier and
   Megatron-Core/Nemotron-H tooling, is unquantified integration time.
4. **No SFT/RLHF cost included** — pretraining-adjacent compute alone does not ship a
   benchmark-ready instruct model, consistent with every prior estimate in this line of work.
5. **94B tokens is one data point, not a scaling law.** It has not been shown to hold at
   different compression ratios or target sizes; treat it as the best available anchor, not a
   guarantee.
6. These figures are extrapolated from disclosed token counts, not measured on this project's
   own hardware/stack — the same preflight discipline (SPEC 0022's GO-rule pattern) should apply
   before spending against them.

## 6. Recommendation

This is the first path in this analysis line that plausibly meets the $20-30K target on
published, reproducible grounds. Before committing:

1. Pick a teacher (Qwen2.5-7B is the cleanest license fit) and confirm the compression ratio
   against the 3B target first — the best-tested case.
2. Run a short preflight (pruning + a few hundred million tokens of distillation) to measure
   real throughput and early quality signal before committing the full 94B-token budget.
3. Scope the Sanskrit fold-in stage as a separate, cheap, clearly-gated addition — don't let it
   block or complicate the primary pruning-and-distillation validation.
4. Treat the 1B target's cost figure as optimistic given the untested 8x compression ratio; the
   3B target carries the least extrapolation risk of the three sizes modeled.
