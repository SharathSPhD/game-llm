# SPEC 0008 — RQ-5 (H4): truthful token-auction decoding of specialist models

Status: ACTIVE (closure program) · GPU: 5090 (specialist training + decode eval)
— decode eval moved off GB10 (2026-08-26): GB10 excluded from ALL GPU workload
per operator directive pending its cooling-defect RMA/repair (see SPEC 0007
for the reassignment rationale).

## Design (pre-registered)

- Specialists: 2 domain-specialist ~30M LMs trained on disjoint BabyLM subdomains
  (child-directed speech vs written/wiki text — the corpus ships domain files);
  short runs (~3k steps each) on the 5090.
- Mechanism: validated second-price auction (F6) over the two specialists per
  decoding step; each agent's bid = its own max-prob (confidence) as valuation;
  winner's distribution emits the token; payments logged.
- Baselines: each specialist alone; uniform logit-average ensemble.
- Eval: held-out mixed-domain stream (50/50 interleaved) perplexity + per-domain
  perplexity.
- **H4 scoring:** auction ensemble beats BEST single specialist on mixed-domain
  perplexity ⇒ MET; beats worst but not best ⇒ PARTIAL; else MISSED.
- App tie-in: the run's decode traces (bids, payments, winners per token) feed
  the Auction playground with REAL model data — science and app feature in one.

## Runtime

Specialists: ~1h total on 5090. Decode eval: CPU or 5090 minutes (not GB10).

## Realization (2026-08-27)

- Harness: experiments/exp12_auction_decoding.py (CPU smoke suite
  tests/test_exp12.py, 5 green incl. vectorized-auction == TokenAuction
  cross-check and second-price trace property).
- Domains: childes.train.txt (child-directed speech, 15.2MB) vs
  simple_wiki.train.txt (written/wiki, 8.9MB) from the BabyLM-2026
  strict-small snapshot; 5% line-level held-out per seed; mixed stream =
  alternating held-out windows (50/50), ~100k eval tokens.
- Specialists: ~30M ExplicitLM (d=384, 6 layers, 6 heads), 3k steps,
  batch 32, lr 3e-4. Seeds 42/43/44. Checkpoints saved for the playground/HF.
- Queued on the 5090 behind exp11 (single-GPU discipline, wait-wrapper).
