# SPEC 0008 — RQ-5 (H4): truthful token-auction decoding of specialist models

Status: ACTIVE (closure program) · GPU: 5090 (specialist training) + GB10 (decode eval)

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

Specialists: ~1h total on 5090. Decode eval: CPU/GB10 minutes.
