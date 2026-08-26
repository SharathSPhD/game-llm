# Kinetic AI Research Journal (append-only)

## [cycle 0 | 2026-08-20] BOOTSTRAP
- Recreated .venv natively (uv, python 3.12, torch 2.13.0+cu130); baseline pytest:
  112 pass / 4 FAIL (all in tests/test_mmd.py — the core MMD convergence claims).
- git init (main), identity SharathSPhD <qbz506@york.ac.uk>, remote wired
  (github.com/SharathSPhD/game-llm — empty), baseline committed (f8c53a8).
- Authored CLAUDE.md (hypotheses H1–H4, closure contract), SPEC 0001, memory files.
- Reference surveys completed: prabhasa-samskrutam (gateway/tunnel/Supabase patterns),
  SAGE (labs UI), PSALM/ACD (paper+site+harness conventions).
- next: Phase 1b ultracode adversarial review workflow → findings → TDD fix loop.

## [cycle 1 | 2026-08-20] PHASE 1b+1c: ADVERSARIAL REVIEW + FIX LOOP CLOSED (TECHNICAL layer)
- Ultracode review workflow (36 agents, 7 dimensions, refute-first 3-lens verification):
  28 confirmed/partial findings, 1 refuted → research/memory/findings-rq1.md.
- Headline adjudication: MMD implementation coherent but ≠ Sokota closed form
  (missing 1/(1+ητ) normalization); the 4 failing tests encoded WRONG theory
  (Nash from fixed magnet at invalid stepsize; algorithm limit-cycles as theory
  predicts). Fixed all three surfaces: closed-form proximal update, stepsize-valid
  tests asserting QRE (fixed magnet) and Nash (RND resets), honest README claims.
- Other fixes: Broyden sign-flip clamp; truthfulness property tests + reserve-price
  payment fix (auctions); SPPO resampled policy-weighted win rates (fixed point
  restored) + semantic calibration wired behind config flag; Kuhn NE value -1/18
  verified; `or True` test bug; Wilcoxon n<10 guard; safe YAML (no eval);
  seeded simulate.py; treeplex reach-weighted DilatedEntropy option.
- Gates: pytest 145/145 (was 112/116), ruff clean, mypy clean, coverage 93% (gate 80).
- next: milestone push #1; Phase 2 Tier A experiments (H2: MMD vs GDA cycling).

## [cycle 2 | 2026-08-20] PHASE 2 TIER A: H2 VALIDATED + DISCOVERY (F1-F8)
- Tier A workflow (4 experiments, 9 Tarka verifications) + refinement workflow
  (3 refinements, 10 Tarka) + damped-QRE fix agent. All findings in findings.md.
- H2 core VALIDATED: MMD linear last-iterate convergence (R² .99/.90) where GDA
  cycles (NashConv ~1.8-1.9); RND resets -> Nash universally (F1, F3).
- DISCOVERY (F2): uniform-anchor MMD fixed points != logit-QRE on asymmetric games;
  context-dependent attractors. Paper-worthy boundary condition on the theory.
- F4: DEQ O(1) activation memory vs O(N) explicit (slope .0168 MB/layer).
- F5: Anderson advantage emerges exactly at stiff fixed points (ratio .888 @ rho=.999).
- F6: second-price exactly truthful (regret 0.0, 16k obs); weighted aggregation
  measurably manipulable (regret .077/.068) - consistent with Phase 1 VCG finding.
- F7/F8: warm-start homotopy 25% faster (asym 2x2); QRE solver now damped
  (converges lambda in {1,10,100}; undamped diverged >0.32). Two honest partials
  (non-monotone path, small path movement) recorded as-is.
- EqLM module built TDD (kinetic_ai/models/eqlm.py, 16 tests): causal DEQ transformer,
  weight-tied, param-matching helper, MMD-compatible. NOTE: JFB param-gradient path
  incomplete in deq_layer - using IFT backward; revisit for Tier B throughput.
- Gates: 162/162 tests, ruff+mypy clean, cov 93%.
- next: milestone push #2; Tier B smoke (SPEC 0004): BabyLM strict-small pipeline
  end-to-end on GB10.

## [cycle 3 | 2026-08-21] TIER B SMOKE + PHASE 3 APP BUILD (+security hardening)
- Session-limit interruption killed two agents mid-flight; recovered, finished
  gates (200 tests green), invalid first smoke rejected (6 steps, no param match).
- Real smoke (exp05 iter 2): pipeline end-to-end on GB10. F9 recorded: EqLM learns
  slowly (init scale bug), raw-MMD arm flat (adaptivity confound) -> ADR 0003
  (MagneticAdamW), BLiMP deltas = noise at these losses. Agent overclaim rejected.
- Phase 3: FastAPI backend (solve/qre_path/auction/jobs, executor abstraction,
  RunPod-portable), Cloudflare gateway scripts, Supabase migration, full Next.js
  frontend (6 pages, replay mode, findings explorer) - built, built-verified, pushed.
- Security reviews (3 rounds) fixed: no seeded secrets + RLS allowlist; fail-closed
  GATEWAY_SECRET; explicit CORS; session-gated allowlisted proxy; no error echo.
- next: cycle 4 - MagneticAdamW (TDD) + init-scale fix + smoke rerun A2/A3;
  then full Tier B run (3 seeds), gateway deploy needs operator (Supabase/Vercel auth).

## [cycles 4-5 | 2026-08-21] MAGNETIC-ADAMW DEBUGGING ARC + REAL DATA STREAM (F10-F12)
- exp05 iter3: init fix verified; EqLM parity with explicit at smoke scale (F10),
  later scoped: corpus was 22.7k unique tokens cycled (data cap; F10-correction).
- Fixed F9 data blocker: build_token_stream streams until max_tokens (regression
  tests); exp06 first runs exposed: agents twice delivered "ready to run" without
  running; multiple imagined APIs caught by CPU dry-runs before GPU spend.
- MagneticAdamW arc: 97.7GB memory leak fixed (no_grad discipline); then the real
  bug — coupled weight decay destroying the tied embedding (F11), found via
  isolated per-process A/B after the sweep showed tau-independent throttling.
- exp06 (fixed): magnetic EMA pull loss-neutral at tau<=1e-2 (F12); drift prereg
  ruled invalid (cross-architecture). exp06b: solver budget loss-neutral, 0%
  convergence at tol 1e-3 across budgets.
- App: Pages site LIVE at sharathsphd.github.io/game-llm; CI green after extras
  fix; frontend + gateway + proxy hardened (3 security review rounds).
- next: FULL Tier B run (A1/A2/A3 20k steps, full stream) launched in background;
  harvest -> Tarka -> paper/site refresh -> operator sign-off pass on F1-F12.

## [cycle 6 | 2026-08-21] TIER B FULL RUN HARVESTED (F13): H1 iter-1 missed, cause identified
- exp05_full (3.5h GB10): A1 BLiMP 0.734 vs EqLM 0.571 / +magnet 0.584 -> H1
  MISSED (78-80% of baseline vs >=95% prereg). Solver 0% convergence at all
  budgets = non-contractive map; spectral norm never wired in (Phase-0 gap).
- next: EqLM-v2 cycle - wire apply_spectral_norm/pcDEQ into EqLM block (TDD),
  verify solver convergence >80% on smoke, rerun matched full comparison;
  refresh paper/site with F10-F13 truth; operator sign-off pass F1-F13.

## [cycle 7 | 2026-08-21] EXP07 + F14: no-fixed-point diagnosis (the real EqLM story)
- exp07 (GPU): v2 spectral-norm+damping preserves loss but 0% convergence.
- Follow-up numerics: residual scales linearly with damping, tail flat ->
  NO fixed point exists (residual map without outer bounding op); absolute-norm
  convergence criterion also unsatisfiable by construction (F14).
- F13 reframed: current EqLM = weight-tied 12-iteration transformer.
- next: EqLM-v3 (outer-LN map, relative residual) queued as H1 iteration 2.

## [cycle 8 | 2026-08-21] EQLM-V3 SMOKE + F15: fixed points exist, contraction is the frontier
- v3 (post-LN map, rel-residual criterion) landed TDD (226 tests); exp07 4-arm
  smoke: loss parity MET, convergence NOT MET at 12 iters; width probe: rel
  residual 1.0->0.105 decaying (F14 drift gone) but contraction ~1 at width.
- H1 status: iteration 2 open with precise v4 arms (alpha schedules, solver-aware
  loss, targeted spectral budget, pcDEQ, tol-sufficiency study).
- next: operator sign-off pass F1-F15; v4 arm selection; paper/site refresh with
  F14/F15 arc; app deploy awaits operator (Supabase/Vercel/Cloudflare auth).

## [cycle 9 | 2026-08-21] OPERATOR SIGN-OFF F1-F15; DEPLOYMENT PHASE OPENED
- Operator signed off findings F1-F15 (closure layer 6 satisfied); Supabase and
  Vercel connectors authorized; directive: deploy + continue goal autonomously.
- Flaky overfit test made deterministic (copy task, single-thread reductions).
- next: Supabase project + migration; Vercel deploy of apps/web; gateway tunnel
  (pending CF credentials on machine); EqLM-v4 arm launch.

## [cycle 9b | 2026-08-21] DEPLOYMENT LIVE (Phase 3 closure, end-to-end verified)
- Supabase project `kinetic-ai` (sevlncqcywaajapqitfk, eu-west-2, $0/mo) created via
  MCP; migration applied (user_tiers + admin-signup trigger for
  sharath.sathish@gmail.com + set_user_tier guest RPC + job_history + runtime_config
  RLS allowlist).
- Vercel project kinetic-ai-web linked to SharathSPhD/game-llm (root apps/web);
  env: Supabase publics + GATEWAY_SECRET (generated, local copy in gitignored
  .gateway.env) + GATEWAY_URL; production READY at https://kinetic-ai-web.vercel.app
- GB10 backend live: uvicorn app.server :8097 (setsid) + cloudflared quick tunnel
  (ephemeral URL - worker/KV permanent gateway pending operator CF_EMAIL/CF_KEY;
  scripts/gateway ready).
- E2E smoke: /api/health -> GB10 gpu_available:true through the full chain;
  /api/proxy/api/solve correctly 401s anonymous (Supabase session + tier row
  required); /lab 200. Operator: sign in once with sharath.sathish@gmail.com to
  auto-receive admin; enable guests via select set_user_tier('<email>','user').
- next: EqLM-v4 arm; worker/KV permanent gateway when CF creds provided.

## [cycle 10b | 2026-08-21] PERMANENT GATEWAY LIVE (operator CF creds)
- Cloudflare account b7f7f1b1 (sharath.ai.colab@gmail.com): workers.dev subdomain
  'kinetic-ai' registered; KV namespace fb379d31; worker 'kinetic' deployed via API
  after fixing 3 worker.js defects (dropped Authorization header, wrong Vercel host,
  wildcard CORS). Public stable URL: https://kinetic.kinetic-ai.workers.dev
- Vercel GATEWAY_URL switched tunnel->worker (redeployed); Supabase runtime_config
  updated; run_gateway.sh defaults point at new account/KV (secrets gitignored).
- E2E verified: worker /health -> GB10 gpu true; UI proxied via worker (200);
  anonymous GPU access still 401. Tunnel restarts now self-heal via run_gateway.sh.

## [cycle 11b | 2026-08-21] H1 VERDICT (F18): 93.0% [89.8, 94.9] - formal miss, tight CI
- 3 seeds complete (7h GB10). Ratio CI upper bound < 0.95: honest miss, huge
  progress from iter-1 (0.78 -> 0.93). Aux rider no BLiMP benefit at 1e-3.
- GPU lock released. next: paper/site F16-F18 refresh; H1 iteration 4 candidates
  (eval-time solver budget, aux annealing, 2-block DEQ) + Tier C (H3 MPO, H4
  auction decoding) remain open in the program backlog.

## [cycle 13 | 2026-08-22] H1' BUILD + STUDIO CONSOLE + PYPI (+two agent-fake catches)
- PyPI v0.2.0 LIVE (pip install kinetic-ai) per operator authorization.
- Caught + fixed agent no-op warm_start (flag ignored); implemented real
  warm-start (DEQLayer z_init -> EqLM.forward -> generate) with mechanism-spy
  tests; discovery: contraction-trained toys converge in 2 iters (floor) so
  warm-start value is a scale question (H1'a, exp09).
- Studio runs console live in code: real experiment submission (allowlist +
  strict override schema), gpu_lock enforcement, log polling, runs registry,
  full UI; 21 tests.
- Root-caused order-dependent test failures + a HANG: mock mode keyed on magic
  GATEWAY_SECRET=='test-secret' at import time - tests could launch REAL
  training subprocesses. Replaced with explicit KINETIC_MOCK_EXPERIMENTS env
  read at submit time. Suite: 266/266 in 65s (was 26 min).
- Checkpoint save/load added (unblocks HF publication). SPEC 0006 (models
  registry + HF push; EqLM-100M on GB10 before NeMo). Operator directives:
  registry+HF next, 100M first, paper continuously aligned.
- next: backend restart; exp09 GPU smoke (H1'a/c); 100M timing probe; exp10.

## [cycle 14 | 2026-08-22] EXP10 PROBE: memory advantage + loss lead appear at 110M scale
- Probe (300 steps, 121-124M params matched): EqLM peak 6.2GB vs explicit 8.0GB
  (-22% — O(1)-depth advantage first visible at LM scale); EqLM AHEAD on train
  loss (6.79 vs 7.40 at step 275, same init) — width reverses the small-scale
  pattern; cost 2.57 vs 0.46 s/step (5.6x; F19 warm-start is the decode answer).
- exp10 full launched: 10k steps x 3 arms (~15.6 GPU-h), 35M-token stream,
  checkpoints -> first HF publications.

## [cycle 14b | 2026-08-22] REGISTRY LIVE; git hygiene for weights
- Models registry + HF publisher deployed to GB10 backend (3 probe checkpoints
  listed); /models UI shipped; known minor bugs: params_estimate math, exp09
  checkpoint missed by scan (fix with capability-3 pass).
- Git hygiene: 1.4GB probe checkpoints were in unpushed commits (408 on push);
  results/**/checkpoints/ + *.pt now gitignored, commits rewritten - HF is the
  weights channel, git is the code/results channel.
- exp10 full (10k steps x 3 arms) training; harvest closes H1-at-scale +
  first HF publications.

## [cycle 15 | 2026-08-26] DGX CRASH INVESTIGATION: platform-level, NOT session memory pressure
- Reboot forensics: hard stops (journal cut mid-write, no shutdown, no panic, no
  OOM-kills) at Aug 22 08:02 and Aug 26 02:15. sar shows ~20% memory used / 99GB
  free at BOTH crash moments -> memory exhaustion ruled out. Crash 2 occurred
  with none of our workloads running (services do not survive reboots).
- Contributing observation: NVRM NV_ERR_NO_MEMORY storm Aug 21 01:47 during seed
  runs (unified-memory pressure) - box survived it by a day; not the killer.
- Verdict: DGX Spark platform instability on 6.17.0-1029-nvidia; the automatic
  kernel update to 6.17.0-1031-nvidia this morning is likely the fix. Operator
  recommendation: run full DGX OS/firmware updates (fwupdmgr / apt) when idle.
- Proposed but NOT installed (needs operator authorization): systemd user units
  for backend+tunnel auto-heal after reboots.
- Recovery: backend + fresh quick tunnel + KV pointer updated (gateway E2E green);
  exp10 relaunched (per-arm resume active); persistent monitor re-armed.
