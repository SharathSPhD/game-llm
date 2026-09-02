---
license: apache-2.0
language: [en]
tags: [deep-equilibrium, weight-tying, parameter-efficiency, kinetic-ai, onnx]
datasets: [BabyLM-community/BabyLM-2026-Strict]
---

# kinetic-eqlm-46m-compute-matched

The flagship artifact of the Kinetic AI programme: a weight-tied transformer
block applied twelve times, at the same width as a conventional 12-layer
baseline, so that one iteration costs exactly one layer and compute is equal by
construction.

## The measured claim

| | this model | explicit 12-layer baseline |
|---|---|---|
| parameters | **45.8M** | 123.8M |
| resident weights (bf16) | **183 MB** | 496 MB |
| compute per token | 84.9M units | 84.9M units (equal) |
| BLiMP ratio, 3 seeds | **0.958 ± 0.017** | 1.000 |
| BLiMP ratio, 31 phenomena | 0.954 | 1.000 |

At identical arithmetic it delivers roughly 96% of the baseline's quality with
2.70 times fewer parameters (12 times fewer in the blocks themselves). One
checkpoint serves every budget: at solver depths 4, 8 and 12 quality degrades
gracefully (0.93 of baseline at half depth), which no fixed stack offers.

## What is honestly NOT claimed

It is not better than the baseline on quality, and the gap (≈4%) did not close
under depth modulation (that made it worse — the tying works because repetition
of ONE map contracts to a fixed point). The memory saving is in weights, not
activations: at batch one the Anderson solver's history makes activation peak
2.3× worse than the baseline. Scaling this recipe to open-weight-class models
requires a pretraining budget: converting pretrained models fails at any
gentleness (initial damage 64–270× base perplexity), and cheap logit
distillation into the from-scratch student failed its pre-registered gate
(−2.2% against a required +15%).

## Where the claim stops: the one-billion-parameter twin (F55)

The recipe was taken to deployment scale under pre-registration: an explicit
913M transformer against a tied block of the same width (158M resident),
compute-matched, 2.5B FineWeb-Edu tokens each. The tied arm failed its 1B-token
kill gate (held-out perplexity ratio 1.56 against a bar of 1.20) and was still
closing at 2.5B tokens (1.31 against a success bar of 1.10); both arms score at
chance on public benchmarks at that budget. The programme halted there (ADR
0011). The exchange rate above is therefore established at 46–121M parameters
on BabyLM and does not transfer unchanged to a billion parameters on web data;
whether more tokens recover it is an open question, not a claim.

## Files and formats

`model.safetensors` — exact, weight tying preserved, zero overhead.
`model_depth12.onnx` — fixed 12-iteration graph with the block's tying preserved
as shared initializers; the embedding/head tie is folded into two copies by the
exporter, so the file is 337 MB rather than 183 MB. GGUF is deliberately not
provided: llama.cpp cannot express weight sharing, and the required unrolling
produces a file 4.9× larger than the baseline this model saves against.

## Provenance

Findings F44–F55 in https://github.com/SharathSPhD/game-llm — every number
traces to a results file with config hash and commit; the record includes the
refuted attempts at the same evidentiary standard as the successes.
