# SPEC 0013 — RQ-10 (H9): truthful auction decoding over real 1–4B specialists

Status: ACTIVE · GPU: GB10 (inference-heavy) · Pre-registered 2026-08-28 · ADR 0006

## Question

F22 showed auction SELECTION wins at scoring time with two 30M toy specialists;
F23 showed the advantage inverts in closed-loop generation, under a judge that
proved style-dominated. Both limitations vanish with real specialists and real
benchmarks with objective answers. Does truthful token-auction aggregation beat
the best single specialist when the specialists are genuinely strong and the
metric is task accuracy rather than a judge LM?

## Design

- Specialists (SAME tokenizer family — a hard requirement for token-level
  aggregation): Qwen2.5-Math-1.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct,
  Qwen2.5-1.5B-Instruct (general). Verify vocabulary identity before any run.
- Mechanisms: **second-price auction** (bid = own max-prob confidence; F6
  truthfulness verified) vs **best single** vs **uniform logit-average
  ensemble** vs **oracle router** (upper bound: always the right domain
  expert).
- **Context-aware bids (the F23 follow-up):** an arm where the bid is the
  model's confidence over the PREFIX (its perplexity on the prompt) rather
  than the current token — testing whether closed-loop drift is fixed by
  prompt-level rather than token-level signal.
- Eval: mixed-domain stream of GSM8K (math), HumanEval or MBPP (code), MMLU
  (general) — **objective accuracy**, no judge LM, closing F23's metric flaw.
  Both regimes reported: teacher-forced scoring AND closed-loop generation.

## Scoring (pre-registered)

- **MET:** auction accuracy > best single specialist on the mixed stream, in
  CLOSED-LOOP generation, on 3 seeds (this is the claim F23 failed at toy
  scale).
- **PARTIAL:** MET at scoring time only (reproducing F22 at real scale), or
  MET closed-loop only with context-aware bids.
- **MISSED:** neither — establishing that per-token auction aggregation does
  not survive strong specialists, a clean negative that closes the line.

## Runtime

Inference only: ~2–6 h on GB10 (3 x 1.5B models resident in 121GB unified
memory simultaneously — this is the machine's advantage and needs no training).
