# SPEC 0006 — Models registry + HF publication; EqLM-100M scale-up

Status: ACTIVE (operator directives 2026-08-22: registry+HF next; 100M on GB10
before NeMo adoption; paper stays continuously aligned, frozen only at the end)

## Part A — Kinetic Studio capability 2: Models registry + HF push

- Backend:
  - GET /api/models — walk results/**/ *.pt checkpoints saved via
    kinetic_ai.models.eqlm.save_checkpoint; return [{path, config summary,
    params, source run (config hash + git commit), metrics from the sibling
    results.json}].
  - POST /api/models/publish — {checkpoint_path (must be inside results/),
    repo_name} → pushes to Hugging Face under the authenticated account
    (public), generating a model card from research/memory/findings.md entries
    referencing that run + provenance block (config sha, commit, seed). Uses
    huggingface_hub; HF auth = the machine's existing hf login (verified:
    hf_whoami works). Admin-tier only via the app.
- Frontend: /models page — registry table + publish button + link to HF repo.
- First publications (after exp09/100M runs save checkpoints):
  kinetic-ai/eqlm-babylm-10m (A3 arch) and kinetic-ai/explicitlm-babylm-10m
  (baseline) — honest model cards stating BLiMP numbers and the F18 verdict.

## Part B — EqLM-100M on GB10 (scale before NeMo)

- exp10_scale: parameter-matched pair at ~100M params (d_model ~768, explicit
  n_layers 12 vs EqLM widened; BabyLM-2026-Strict [100M words] from HF cache;
  bf16; grad-accum for effective batch; token cache on disk).
- Budget: single seed first (~est. 8-20 GB10-hours/arm — measure in a 1k-step
  timing probe before committing); checkpoints saved every 5k steps + final.
- Prereg: (i) does the EqLM/explicit BLiMP ratio improve vs 10M-scale 0.930?
  (report with CI from BLiMP binomial); (ii) solver stats at width 768;
  (iii) warm-start decode benefit at scale (H1'a measured here).
- NeMo/Megatron adoption deferred until >GB10 scale is required (per operator).

## Ordering

1. exp09 GPU smoke (H1'a/c quick numbers) → 2. timing probe for 100M →
3. exp10 full run (checkpointed) → 4. HF publication of real checkpoints →
5. Studio /models capability → paper sections update continuously.
