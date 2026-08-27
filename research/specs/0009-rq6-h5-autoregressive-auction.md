# SPEC 0009 — RQ-6 (H5): autoregressive auction decoding (closes F22's gap)

Status: ACTIVE · GPU: 5090 only · Pre-registered 2026-08-27 BEFORE any run.

## Question

F22 established that second-price auction SELECTION beats the best single
specialist under teacher forcing. H5: does the advantage survive CLOSED-LOOP
generation, where the auction's own sampled token becomes the next input
(exposure bias compounds across models)?

## Design

- Systems (reusing exp12 specialists, seeds 42/43/44): S_A alone, S_B alone,
  uniform logit-average ensemble, second-price auction (bid = own max-prob
  at the CURRENT self-generated context; winner's distribution emits the
  token; greedy decoding for determinism).
- Prompts: 100 held-out prefixes per seed (50 childes / 50 simple_wiki,
  32-token prefixes from the exp12 held-out split); generate 32 tokens.
- **Primary metric (pre-registered): mean NLL/token of the generated
  continuation under the frozen exp10 seed-42 124M explicit judge**
  (independent of all compared systems, identical application). H5 MET if
  auction < best single specialist on 3/3 seeds; PARTIAL if 2/3; else
  MISSED.
- Secondary: domain consistency (fraction of continuations whose own-domain
  specialist assigns lower ppl than the off-domain one); 3-gram repetition
  rate (degeneration check); per-position winner traces (app tie-in).
- Confound guard: judge scores exclude the prompt tokens; identical prompts
  and generation length for all systems.

## Runtime

Eval-only (checkpoints exist): ~10-20 min on the 5090. Runs BEFORE exp13.
