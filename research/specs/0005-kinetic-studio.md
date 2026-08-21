# SPEC 0005 — Kinetic Studio: the app as the research instrument

Status: ACTIVE · supersedes the showcase framing of the Phase 3 app
Operator intent: beat SAGE Labs; high value for practitioners AND researchers;
train the new architecture end-to-end (corpus → pretraining → fine-tuning);
leverage NVIDIA resources; the app grows with the science like the paper does.

## Product thesis

Kinetic Studio is the place where equilibrium language models are TRAINED, WATCHED,
and USED — not a demo of finished results. Every validated finding ships a surface.

## Capability map (build order)

1. **Runs** (research console, first): submit REAL experiment configs (the exact
   exp05/exp08/exp09 harnesses, YAML-validated) to the GB10 through
   kinetic_ai/serve/executor; single-GPU queue honoring research/memory gpu_lock;
   live loss/residual streaming (SSE from the backend tailing the run's JSONL log);
   run registry = results/ directory + Supabase job_history; every run shows
   config hash + git commit (provenance = the paper's evidence chain).
2. **Models** (checkpoint registry): checkpoints saved by every run (exp09 patch),
   listed with metrics; one-click push to Hugging Face (public, model card
   generated from findings.md entries); load-for-inference on GB10.
3. **Playground + equilibrium lens** (the fruit): chat/complete with any registered
   checkpoint; EqLM vs explicit side-by-side; per-token solver iterations and
   residual-confidence rendered inline (H1'b/c made visible); "think-harder" slider
   = eval-time solver budget (H1'c).
4. **Corpus**: dataset panel — HF datasets by name (BabyLM presets), upload small
   corpora to GB10, tokenizer choice, token-stream cache management.
5. **Fine-tune**: instruction/preference fine-tuning jobs on registered checkpoints
   (H3's MPO vs DPO arms run HERE — Tier C science through the product).
6. **Auction ensemble** (H4 through the product): registered specialist checkpoints
   bid per token via the validated second-price mechanism; live bid/payment stream.

## NVIDIA leverage (pragmatic, staged)

- Now (GB10): PyTorch harnesses as-is; NGC PyTorch container for serving isolation.
- Next: NeMo AutoModel / Megatron-Bridge recipes for scaling EqLM past toy sizes
  (this machine ships nemo-automodel-* and nemo-mbridge-* skills; adopt when a
  run needs >1 device or >100M params); Nemotron datasets as corpus presets.
- Later: RunPod serverless executor (already stubbed) for paid tier.

## Non-goals (now)

Multi-tenant GPU sharing; billing; arbitrary user code execution (configs are
schema-validated, never exec'd).

## Definition of done per capability

Backend endpoint + tests; UI page; E2E smoke through the live gateway; journal
entry; app version note on the site. The closure contract applies.
