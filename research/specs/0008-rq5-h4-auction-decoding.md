# SPEC 0008 — RQ-5 (H4): Truthful token-auction decoding of specialist models

Status: ACTIVE · closure program · pre-registered before any run

## Design

H4 as registered: truthful auction aggregation of 2–3 specialists beats the best
single model on mixed-domain eval. Specialists come from BabyLM-2026-Strict's
natural sub-corpora (e.g. child-directed speech / wiki-style / fiction — exact
split from the dataset's source field, committed as a manifest).

1. Train 3 specialist LMs (identical EqLM-v3 arch, ~110M) each on ONE sub-corpus
   (matched token budgets). 5090 job (fast, parallel with other closure work).
2. Decoding arms on a MIXED eval stream (held-out, all domains interleaved):
   - S1/S2/S3: each specialist alone (best single = max over these).
   - AUC: second-price token auction (validated truthful, F6): each specialist's
     bid = its per-token confidence (max prob or negative entropy — choose in
     smoke, pre-register before full); winner's distribution emits the token;
     payments logged (economic telemetry, not used for selection).
   - MIX: uniform logit-average ensemble (the non-strategic baseline the auction
     must beat or match to be interesting).
3. Metric: mixed-domain perplexity (primary), per-domain breakdown; 3 seeds of
   eval-stream sampling.

## Pre-registered

**H4 met** iff AUC perplexity < best single specialist's mixed-domain perplexity
(paired bootstrap over eval shards); AUC vs MIX reported honestly either way
(auction's value-add claim requires AUC ≤ MIX too, else scoped as "truthful
mechanism at modest cost").

## App tie-in

The same auction decode powers the /auction playground with real models
(bids/payments streamed) — science arm and product feature are one build.
