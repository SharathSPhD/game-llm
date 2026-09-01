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

## [cycle 16 | 2026-08-26] GB10 ROOT CAUSE CONFIRMED: THERMAL TRIP (~91C) + dual-GPU closure program
- Blackbox caught the 15:48 crash: final synced samples show 96% GPU util,
  63-67W, thermal zones climbing 89.9->90.3->91.2C, then instant power cut.
  EC-level thermal trip explains every silent death (no OS logging possible).
  91C at only ~65W = suspected COOLING DEFECT (fan/dust/contact) - operator
  physical check requested; no fan RPM sensors exposed for software check.
- Mitigation: thermal_governor.sh (SIGSTOP >85C / SIGCONT <78C) now guards the
  GB10 seed-42 run; blackbox continues recording.
- Alignment review recorded: H3/H4 starvation + seed discipline drift corrected;
  SPEC 0007 (MPO vs DPO on BLiMP-pairs-as-preferences) + SPEC 0008 (auction
  decoding of domain specialists) written; closure = full program + arXiv paper,
  HF releases, PyPI v1.0+docs (operator).
- Compute split live: 5090 = primary pretraining (seeds 43/44 full power,
  port agent running); GB10 = seed 42 governed + serving + evals + fine-tuning.
- Archive to 5090 relaunched (verified clean slate; checksum-verified moves).

## [cycle 16b | 2026-08-26] HARDWARE DEFECT CONFIRMED (operator investigation) + FieldDiag staged
- Operator telemetry + NVIDIA forums confirm: known DGX Spark GB10 defect - 12C
  die-vs-platform sensor gap at 60W (degraded TIM/mounting), BMC power-throttle
  to ~60W despite no kernel throttle flag, hard trip ~91C. RMA channel exists
  (FieldDiag PowerStress MODS-020000600139).
- GB10 cleared for FieldDiag: governor+trainer stopped cleanly (A1 banked);
  seed-42 requeued to the 5090 (monitor armed to submit after seeds 43/44).
  FieldDiag install/run requires operator sudo - exact commands handed over.
- Archive completed & independently verified: agent over-claimed; I re-verified
  zero-diff per project and finished deletions (incl. oak which was never
  uploaded - archived first). 25 projects on ss@:~/gb10-archive, ~200GB freed
  (2.3T->2.1T), stubs + manifest in place. PSALM-integration/prabhasa pruning
  proposal awaits operator (Section D of ARCHIVE-MANIFEST.md).

## [cycle 16c | 2026-08-26] THERMAL DEFECT RE-VERIFIED (controlled test) + FieldDiag blocker + dashboard fix
- Controlled stress (safety abort 88C, no sudo needed): platform zones 79.6->
  89.4C in 44 SECONDS at ~95W; GPU die read only 67-77C simultaneously (12C+
  die-vs-platform gap = degraded TIM/heat path). DEFECT PRESENT on latest
  kernel+EC. Evidence: results/thermal_verify.log.
- FieldDiag fails under Secure Boot: mods.ko builds but insmod rejected ("Key
  was rejected by service"); box is SSH-only so UEFI/MOK console paths need
  care. Operator options handed over (sign mods.ko with enrolled shim MOK).
- NVIDIA Sync DGX Dashboard down because FieldDiag prep stopped services and
  failed runs never restored them: dgx-dashboard{,-admin} + nvidia-persistenced
  inactive (enabled). One sudo line restores.

## [cycle 16d | 2026-08-26] GB10 EXCLUDED FROM GPU WORKLOAD (operator directive); 5090-only closure program
- FieldDiag PowerStress blocked for real this time: mods.ko builds+signs fine,
  but `insmod` rejected under Secure Boot ("Key was rejected by service");
  official user guide confirms Secure Boot must be disabled in UEFI first.
  Box is SSH-only (no physical/console access), so the disable/re-enable UEFI
  round trip is not safe to attempt remotely - deferred, not abandoned.
  dgx-dashboard{,-admin}.service restarted (was stopped for the attempt).
- Operator directive: leave GB10 out of ALL GPU workload going forward (until
  RMA/repair). SPEC 0007 (H3) reassigned GB10->5090; SPEC 0008 (H4) decode-eval
  reassigned GB10->5090. state.json known_defects[] now carries the GB10
  hardware-defect record (was empty).
- 5090 status verified directly (connect_check + status.sh + container
  inspection, not assumed from memory): seed43 COMPLETE (all 3 arms,
  results.json + loss_curves.pdf, finished 18:43); seed44 running arm A2
  (~step 5800/10000 at check time), 100% util / 351W / 69C - healthy. No
  seed42 monitor was actually alive (prior "monitor armed" note did not
  survive) - built configs/exp10_seed42.yaml (seed44 config + seed:42) and
  seed42_after.sh (polls for train.py exit, then launches seed42; enforces
  never-two-GPU-jobs-at-once) directly in the existing kinetic_exp10 project
  dir on the 5090; launched detached (PID 736), confirmed waiting correctly.
- Next: poll 5090 for seed44 A3 completion + seed42 full run; harvest
  3-seed H1-at-scale verdict (bootstrap CI / Wilcoxon per run.md gates);
  then SPEC 0007 + SPEC 0008 on 5090 (sequential, same queue discipline).

## [cycle 17 | 2026-08-26] 110M SEEDS 43/44 HARVESTED (interim; seed 42 training on 5090)
- 5090 delivered seeds 43/44 (A1 11min/arm, EqLM 92min/arm at batch 32 x 10k):
  A1 BLiMP 0.675/0.693 (loss 3.07/2.73) vs A2 0.538/0.542, A3 0.532/0.544 -
  ratios 0.78-0.80. The 300-step probe loss lead INVERTED by 10k steps.
- Interim honest read (pending seed 42): matched-budget quality ratio DEGRADES
  with scale (0.93 @ 11M -> ~0.79 @ 121M). Paper story sharpens: equilibrium
  LMs buy O(1) depth-memory + 79% cheaper warm decoding at a scale-growing
  matched-budget quality cost. F20 formalized after seed 42 + Tarka.
- Port agent had pre-staged chained seed-42 on the 5090 (auto-started 22:01);
  monitor re-armed. GB10 remains GPU-idle per cooling-defect directive; NVIDIA
  case package delivered (3 PDFs, questionnaire answered from telemetry).

## [cycle 17b | 2026-08-27] F20 CLOSED (Tarka-amended) · H3 LAUNCHED ON 5090
- Seed 42 auto-chained job completed overnight; 3-seed 121M verdict: ratio
  0.7868, CI [0.785, 0.788], paired t=-82.6. Memory -23% confirmed at width.
- Tarka REFUTED the original mechanism (F18 small-scale runs show the same
  0.0 convergence / 12-iter telemetry): rewrote F20 as graded
  contraction-vs-width under a fixed budget; scale trend rescoped for the
  20k-vs-10k step confound. Findings ledger amended before any artifact use.
- H3 shipped: exp11 MPO-vs-DPO harness (TDD, 4 smoke tests incl. tau-only
  controlled comparison via MagneticAdamW ref_mode="fixed"+tau=0 as the DPO
  arm), spec 0007 amended (controlled comparison + phenomenon-level split),
  3 seed configs; container preflight (local kinetic_ai shadows stale wheel,
  offline tokenizer, checkpoint load) then chained seeds launched. ~2h est.

## [cycle 17c | 2026-08-27] H3 HARVESTED (F21 PARTIAL) · exp12 crash diagnosed · queue reordered
- exp11 all 3 seeds green (17 min/seed). F21 recorded: pre-registered verdict
  PARTIAL (accuracy parity exact; KL reduction ns at under-dosed tau).
  Secondary: DPO damages unseen phenomena (0.74->0.61 heldout, KL 1.2);
  EqLM ~1500x drift-resistant under identical updates. Tarka running.
- exp12 first launch crashed instantly on ALL seeds: rsync -a copied HF
  snapshot SYMLINKS (dangling on 5090); AND run-script "exit code $?" lied
  because $(date -Is) reset $? (both now noted as ops lessons). Re-shipped
  with rsync -L (real 15MB/8.9MB files verified by content), relaunched
  chained behind exp11b (which grabbed the GPU when exp12 crashed out).
- Queue now: exp11b tau rider (running) -> exp12 relaunch. Monitors re-armed.
- HF releases live: qbz506/kinetic-eqlm-121m-babylm,
  qbz506/kinetic-explicitlm-124m-babylm. Backend+tunnel restored on GB10;
  registry serves 121M checkpoints end-to-end through the worker.

## [cycle 17d | 2026-08-27] RIDER HARVESTED: magnet second-order in DPO regime
- exp11b: KL 1.2406->1.2256 across tau 0->10; heldout flat. Rider prediction
  refuted; F21 addendum recorded (PARTIAL on letter, MISSED in spirit).
- exp12 (H4) now holds the GPU. F21 fully closed pending operator sign-off.

## [cycle 17e | 2026-08-27] H4 HARVESTED: MET 3/3 (F22) - auction beats best
## single by 23% and ensemble by 12% on mixed-domain perplexity
- exp12 relaunch green all seeds; 0 vectorized-vs-mechanism mismatches;
  F22 recorded, Tarka in flight (teacher-forcing scoping + childes-line
  leakage check requested). App playground serves real traces
  (GET /api/auction/traces -> seeds [42,43,44]) end-to-end on GB10.
- Empirical program now COMPLETE: H1 (miss, F18/F20), H2 (validated),
  H3 (PARTIAL, F21+rider), H4 (MET, F22). Remaining: Tarka F22, paper
  H4 section, artifact freeze (PyPI v1.0, HF done, arXiv polish),
  operator sign-off pass F16-F22.

## [cycle 18 | 2026-08-27] EMPIRICAL PROGRAM CLOSED — all four hypotheses adjudicated
- F22 Tarka-resolved: MET confirmed (paired t=4.98); rescoped to teacher-
  forced scoring-time SELECTION (autoregressive auction decoding = future
  work); childes-overlap caveat; traces labeled as 200-position sample.
  Paper H4 section + site card shipped.
- Final ledger: H1 MISSED with mechanism (F18 0.930 @ 11M, F20 0.787 @ 121M,
  truncation penalty widens with width; -23% memory + 79% warm-decode win
  retained). H2 VALIDATED (F1-F3). H3 PARTIAL letter / missed spirit (F21 +
  rider; DPO damages unseen phenomena; EqLM 1655x drift-resistant). H4 MET
  (F22, +23% over best single).
- Artifacts frozen: PyPI kinetic-ai 1.0.0; HF 121M checkpoint releases;
  paper builds clean with all four hypothesis sections + expanded related
  work; site rebuilt with F20/F21/F22 cards; app serves real 121M playground
  + real auction traces through the gateway.
- OPEN: operator sign-off F16-F22 (contract layer 6); NVIDIA ticket
  submission (operator); optional autoregressive-auction follow-up.

## [cycle 19 | 2026-08-27] OPERATOR SIGN-OFF F16-F22 · NEW /goal: next programs
- Operator signed off all staged findings; GB10/NVIDIA thread explicitly out
  of scope for now. New goal: complete the follow-on programs autonomously
  (TRIZ inventive step + EFE autoresearch + ralph closure):
  RQ-6/H5 autoregressive auction decoding (closes F22's scoping gap);
  RQ-7/H6 contraction-at-width (attacks F20's open problem).

## [cycle 19b | 2026-08-27] NEW PROGRAMS RUNNING: H5 harvested (MISSED 3/3), H6 screen training
- Ledger repair: findings.md had ballooned to 13.7MB (duplicated F20 blocks
  from a replace-script bug); rebuilt clean from the F19 base + final
  amended F20/F21/F22. All F16-F22 SIGNED-OFF per operator.
- TRIZ session (ADR 0005): contraction-at-width -> physical contradiction
  (Lipschitz small AND large) -> separation by condition/space -> three
  arms: B1 anytime (P11), B2 trajectory-local penalty (P35), B3
  bottleneck-core (P24). Model additions TDD'd (11 tests): forward_unrolled,
  local_lipschitz, EqLMCore (+checkpoint support).
- exp14 (H5) ran first on the 5090: MISSED 3/3 — teacher-forced auction
  advantage inverts in closed-loop generation (F23 recorded, Tarka
  running). F22's rescoping empirically vindicated. Auction still lowest
  repetition; ensemble degenerates.
- exp13 seed-42 screen (B1/B2/B3) training on 5090 (~5h). Ship hiccups
  fixed en route: unquoted YAML colon; exp14's 5s/seed verified real
  (batched greedy).

## [cycle 19c | 2026-08-27] H6 SCREEN: each TRIZ arm cracked a different half
- Seed-42 screen (vs A3 control 0.537 / conv 0.0): B1 anytime BLiMP 0.662
  (86% gap closure!) but conv 0.0; B2 trajpen conv 1.00 at 4.0 iters (first
  CERTIFIED 121M equilibrium, mem 5.5GB lowest) but BLiMP 0.577; B3 core
  0.642, conv 0.20, fastest (24min). Bonus discovery: unrolled anytime
  training BEATS IFT-solver training on wall-clock (44 vs 92 min) - the
  solve dominates EqLM training cost, not backprop.
- No arm meets both MET halves -> pre-registered B4 = B1+B2 combo (spec
  amendment BEFORE run) + B1 budget-sweep rider. Queue 2 launched: B4
  screen -> B1 seeds 43/44 -> budget sweeps. B4 note: Lipschitz penalty
  huge at init (loss 35k -> ~250 by step 600, grad-clip protected);
  running as registered; log-scale B4' is the fallback iteration.

## [cycle 19d | 2026-08-27] H6 QUEUE 2 HARVESTED: PARITY AT WIDTH (F24)
- B1 anytime 3 seeds: 0.662/0.697/0.672 -> ratio vs explicit 0.991 mean
  (seed 43 EXCEEDS baseline). F20's widening-gap trend was a property of
  solver-based training, not the architecture. Budget-sweep rider MET
  (0.59-0.62 @4, graceful). B4 combo refuted (0.529, penalty drowned CE).
  F24 recorded; Tarka running (like-for-like recipe deltas, anytime-vs-
  Anderson eval-path mismatch, BLiMP set identity are its attack angles).
- GPU queue drained again. Next: fold Tarka F24, paper H6 section + site
  card + README, HF release of the parity checkpoint (B1 seed43), ralph
  closure pass, sign-off request F23/F24.

## [cycle 20 | 2026-08-27] NEW PROGRAMS CLOSED — H5 (F23) and H6 (F24) adjudicated
- F24 Tarka-resolved: all claims confirmed, like-for-like audited clean;
  Anderson-vs-plain eval-path rescoping + unrolled-memory attribution
  applied. Paper H5+H6 sections in; site F23/F24 cards; README refreshed.
- HF release: qbz506/kinetic-eqlm-anytime-121m-babylm (the parity model,
  B1 seed 43, ratio 1.033 vs its baseline).
- Program state: H1 arc CLOSED-SUPERSEDED (F18/F20 solver-trained gap ->
  F24 anytime parity); H2 validated; H3 partial; H4 met (scoring-time);
  H5 missed (judge-relative, honest); H6 partial-with-parity. Open problem
  for any future program: quality-preserving certification (B4' log-scale
  penalty pre-sketched in spec 0010).
- AWAITING OPERATOR: sign-off F23 + F24 (both Tarka-resolved, staged).

## [cycle 21 | 2026-08-27] PROJECT CLOSED — final elevation pass shipped
- Operator signed off F23/F24 and directed final closure with two elevations.
- PAPER (ActiveCircuitDiscovery-grade rewrite, 16pp): purged the early
  draft's misdescriptions (MMD wrongly glossed as Maximum Mean Discrepancy;
  a vocabulary-space architecture that was never built; Newton-Schulz;
  three bib entries with fabricated author lists). New structure: abstract
  with the six-verdict ledger, contributions, corrected Background with the
  real MMD closed form and post-LN map, methodology section on
  pre-registration + Tarka audit, results as arcs (convergence, mechanism,
  the full H1 diagnostic, parity F24, MPO, auctions), related work over a
  24-entry VERIFIED bibliography, threats-to-validity, reproducibility,
  per-seed appendix tables, new parity+budget figure. Builds clean.
- APP (SAGE-grade): design-token system + dark theme, 7-explainer Learn
  section, findings gallery from results.json, landing stat tiles, full
  nav. Agent draft AUDITED: three Learn sections contained fabricated
  science (magnet credited for F21's 1655x; invented "logic/arithmetic"
  domains; invented F23 story) — rewritten from findings.md verbatim; nav
  restored after agent orphaned five pages. tsc + build green; deploys
  via Vercel on push.
- Suite 330 green. All artifacts final: paper 16pp, site F1-F24, app,
  PyPI 1.0.0, HF x3 checkpoints, findings ledger fully signed off.

## [cycle 22 | 2026-08-28] OPERATOR CAUGHT IT: garbage generation -> decode-path fix
- Operator flagged playground output quality; bisect proved the F24
  Anderson-vs-plain mismatch corrupts absolute next-token distributions
  (BLiMP relative scoring had masked it). decode_mode in checkpoints +
  training-matched generate() + temperature/top-k sampling shipped through
  model, server, and UI (suite 76 green on touched files). Lesson recorded:
  a working app with unexamined outputs is not a tested app — generation is
  the sensitive assay for eval-path bugs.

## [cycle 23 | 2026-08-28] SCALE PROGRAM OPENED: 1-3B open-weight class
- Operator goal: advance beyond BabyLM to the 1-3B instruct class, real LLM
  output, kinetic architecture retained as the core, benchmark vs the same
  open-weight bases, NVIDIA tooling, 5090 trains / GB10 parallel, productionize.
- Research (web): Nemotron 3 Nano is 30B-MoE/3B-active; Qwen3-1.7B posts
  MMLU 62-66 / GSM8K 75-79. From-scratch 1B is out (Qwen3 saw ~36T tokens).
  Established route = uptrain a pretrained model into a looped/recursive form
  (Relaxed Recursive Transformers 2410.20672; Huginn; Ouro). DEQ at >=1B is
  UNEXPLORED (our opening); auction/mechanism-design decoding is the most
  novel of our three directions.
- INTEGRITY (ADR 0006): "Magnetic Preference Optimization" is ALREADY
  PUBLISHED (arXiv 2410.16714, ICLR 2025) as policy-space MMD with self-play.
  Ours is a different mechanism (parameter-space magnet on the DPO loss) but
  the name collides: renamed PMA; paper must cite it and scope F21's negative
  result to PMA. Recorded before any scale claim.
- Built: KineticLM conversion module (object-identity weight tying with
  distinct layer_idx -> KV cache, generate(), and lm-eval all work unmodified;
  block-recursive n_cores; budget dial; anytime forward; HF-standard
  persistence). 23 tests green. Two real bugs caught by tests/integration:
  HF refuses shared-tensor saves; Qwen3 indexes config.layer_types per layer
  (depth changes overflow it).
- F25 damage curve measured before spending GPU-days: average init beats
  stepwise 10-100x; explicit outer layers matter more than core count;
  operating point 8+8/1-core = 68% params, ppl 1909 (base 6.01). SPEC 0011
  parameter gate amended 60%->70% on that evidence, pre-registered.
- RUNNING: exp15 KineticLM uptraining on 5090 (~98M tokens, distillation +
  stochastic anytime supervision; smoke recovered ppl 9649->329 in 12 steps).
  exp16 auction over real Qwen2.5-1.5B specialists on GB10 (40 GSM8K + 40
  MMLU, closed-loop, objective accuracy). Baseline for comparison established
  under lm-eval-harness on GB10: ARC-C 40.7 / HellaSwag 43.0 / GSM8K 45.7.
- Ops lesson repeated and heeded: the first auction smoke showed chance-level
  accuracy for EVERY system; inspecting the actual generations showed a
  96-token truncation, not a science result.

## [cycle 24 | 2026-08-28] EXP15 UPTRAINING COMPLETE; EXP16 2/3 SEEDS MET
- exp15 (H7, distillation + stochastic anytime arm): KineticLM 1.167B (68% of
  Qwen3-1.7B), 98M FineWeb-Edu tokens, 3.88h on the 5090.
  Held-out ppl 13877 -> 20.84. Base Qwen3-1.7B on the SAME tokens: 17.486,
  so the converted model sits at 1.19x base perplexity after a budget
  100-1000x smaller than published recursive-uptraining recipes.
  Budget sweep nearly flat (21.22 at depth 6 vs 20.95 at depth 12) — the
  anytime property (F24/B1) transfers to the 1.7B conversion.
  Benchmark retention (lm-eval, same invocation as the recorded base rates)
  running on the 5090; H7 verdict pending that number.
- exp16 (H9) seed 42: AUC 0.625 vs best single 0.537 (MET).
  seed 43: AUC 0.637 vs best single 0.575 (MET). Seed 44 in progress.
  Both seeds also show ENS >= AUC, so the auction's advantage over uniform
  logit averaging is NOT established; and AUC_CTX (context-aware bids, the
  F23 follow-up) trails plain per-token bidding on both seeds.
- Paper brought to the ACD standard and encoded as the academic-paper-style
  SKILL (~/.claude/skills/), whose check_style.sh caught 8 figures included
  but never referenced from prose — a defect manual review had missed.

## [cycle 25 | 2026-08-28] NEW PARADIGM: equilibrium decoding (ADR 0008)
- Operator redirection: not incremental mimicry of existing architectures — a
  new paradigm that still beats the baselines, with the kinetic strands (MMD,
  QRE, DEQ, auctions) retained as its substance, EFE loop driving the search.
- Paradigm: the next-token distribution is the tau-regularized QRE of an
  influence game among model-players, solved at decode time by MMD under the
  entropy mirror map. Not a blend (averaging) and not a choice (routing) —
  both are its degenerate cases (beta=0 gives logit averaging exactly; large
  beta approaches routing), so the equilibrium strictly generalises them.
- Uses every validated strand: F1 (linear last-iterate convergence of exactly
  this update), F21 (the magnet belongs in POLICY space, which decode time is —
  the parameter-space version was second-order), F6 (truthful bids make the
  influence weights ungameable), F19 (warm start: adjacent equilibria are
  close), F24 (anytime truncation), QRE (tau/beta as rationality parameters).
- Cost: after one forward per player the solve is softmax/dot products over the
  vocabulary — measured under 50ms for 20 solves at vocab 32k, i.e. ensemble
  cost with strictly richer aggregation.
- Implemented with 11 TDD tests; the Euclidean proximal form was tried first
  and rejected (its fixed point is the arithmetic mean of distributions, while
  the simplex geometry and ensembling practice both call for the geometric
  mean).
- Next: EFE-driven (beta,tau) probe on 20 prompts to cut uncertainty cheaply,
  then the decisive 80-prompt comparison against ENS/AUC/best-single/oracle,
  then the ladder.

## Cycle 26 — 2026-08-28 — the answer-level arena, opened and closed

Phase 0 finished: four players measured on one harness (F28). The ladder
corrected an assumption worth naming, since the parameter-matched model was not
the strongest — Qwen2.5-1.5B-Instruct leads MMLU at 0.626 against Qwen3-1.7B's
0.583, so adopting the nominal comparison as the bar would have flattered every
later result by four points. GSM8K was excluded rather than reported: strict
match scored zero for all four models, which measures answer formatting and not
arithmetic.

The decisive design choice of the cycle was cheap rather than clever. Instead of
sweeping the equilibrium's parameters over a handful of generated prompts, one
evaluation pass stored every player's per-option loglikelihoods, after which the
entire grid was swept offline over 8,301 questions at no further GPU cost. That
is what made a 0.0007 margin legible as noise rather than arguable as a win.

Two defects surfaced along the way and both would have been invisible in the
output. Answer-label conventions disagree between tasks — WinoGrande numbers
from one, ARC carries two conventions in one record — and the assumed mapping
scored players at 0.13 where the truth was 0.63; the fix derives each task's
mapping from the harness's own scoring and verifies it reconciles. Separately,
the thermal governor was found guarding the wrong process for the third time,
having SIGSTOPped the 5090 training watchdog while the GB10 job ran to 90C
unprotected; it now takes a PID rather than a pattern, which removes the failure
mode rather than the instance.

The science: solving the influence game is indistinguishable from averaging
(F29), and eleven rules including mechanism design and calibration all land at
or below the mean (F30), while twenty points of per-example complementarity sit
unclaimed. The explanation that survives is that every rule reweights one fixed
body of evidence and the game discards part of it. SPEC 0016 was amended to
record that its own pre-committed response — move to better players — is not
being taken, because the same data shows player quality is not the constraint.

Housekeeping that mattered: the test venv could not construct a Qwen3 config on
transformers 4.45, which had been erroring 22 KineticLM tests; aligning it to
5.16.1 and repairing the two API changes that surfaced restored 383 passing at
87% coverage.

Running at close of cycle: GSM8K re-measured with chat templates on GB10, then
the cross-examination arena at three seeds; the depth-curriculum recovery arm on
the 5090 at step 4440 of 6000.

## Cycle 27 opening — the loop is a computation now

The autoresearch loop had been prose: `run.md` described choosing experiments by
expected free energy and `CLAUDE.md` cited it, while `kinetic_ai/` contained no
implementation and the choosing was done by judgement. A first attempt at the
implementation, produced by a fan-out, scored all five candidates at exactly
-5.607 because its likelihood never referenced the action; its tests passed
because they only checked a number came back. That is the failure mode worth
naming, and it is now guarded twice: the project's agent has a test asserting the
scores are mutually distinct, and the portable script exits non-zero rather than
print a ranking that selects nothing.

Cycle 27's ranking, from a belief state of 2.821 nats over five hypotheses:
cross-examination first at G=-0.911, serving latency second at -0.814, a plain
domain-router baseline third at -0.581, a second model family at -0.320, and
teacher distillation last at -0.058. Distillation carries the second-highest
pragmatic value and still ranks last, because F31 drove the belief that better
players raise the ceiling down to 0.15 and eight GPU-hours is no longer worth
spending on a question that far settled.

Two of those the programme had not planned. Serving latency is owed under ADR
0009, whose quadratic cross-examination cost invalidates the PRD's ensemble-cost
argument until re-measured. The domain-router baseline is the more useful
suggestion: F33 showed ten points of routable headroom on the mixed arena, and
before any council machinery is credited with capturing it, the cheapest possible
mechanism should be measured, so that the equilibrium is compared against a
trivial router rather than only against the best single player.

## Cycle 28 — the inventive step, taken deliberately

Cycle 27 ended with the autoresearch loop reporting every action at near-zero
expected value — selection exhausted, generation required. The operator's
direction was to generate. The TRIZ engine was applied to the standing
contradiction (adaptability of collective evidence against reliability where a
specialist dominates); matrix cells 35/27 and 35/28 returned inversion,
counterbalance, intermediary and preliminary action, which compose into one
mechanism rather than four suggestions: make the router the reference policy of
the game. The magnetically anchored answer vote scores each answer equivalence
class by its votes plus a magnetic bonus on the router's class, so the council
overrides the incumbent only beyond a margin, and at large magnet strength the
mechanism IS the router — the first rule in the programme whose floor is the bar
itself.

Offline over the 360 stored candidates it beats the router in every grid cell
in-sample and on all three held-out folds (mean +0.0597; one fold 21W/1L,
another 14W/0L), gaining on both domains at once. The result evades F30's
impossibility because it is not a reweighting of fixed distributions: extracted
answers from independent chains are new evidence, and the anchor injects the
ladder's prior knowledge of who is reliable where.

Discipline held: SPEC 0017 pre-registered uniform/tau=1.0 on fresh seeds before
any confirmation data existed, and the confirmation is running now. F39 is
recorded as preliminary-positive, gated on it. The mechanism was promoted into
the package with ten property tests and reproduces the experiment's numbers
exactly; the paper and site carry it marked as pending confirmation.

The EFE skill drove the cycle: the belief state ranked the free offline test
ahead of everything costly, and the observation moved
P(anchored_vote_beats_router) from 0.5 to 0.8. One skill-script limitation
surfaced — it cannot express that the confirmation run is conditional on the
offline result — noted for a future revision rather than patched mid-cycle.

## Cycles 32–33 — closure, the failed gate, and the honest way through it

The science froze the way the specs said it would. The deployment story was
measured before being claimed and split in two: the weight saving is real and
large (F48) while the activation story is neutral-to-worse at batch one; the
distribution story split the same way, safetensors exact, GGUF structurally
dishonest at 4.91x the baseline size (F49). The full-suite harness confirmed the
exchange rate at tripled breadth (F50), and the Tarka closure sweep reproduced
every number in F44–F51 from its results file with zero defects.

SPEC 0020's leap-by-conversion died at its own gate in under an hour: the
gentlest possible surgery starts at 64.5x base perplexity against the
pre-registered 5x (F51). The operator's redirection — break the pretraining
budget deadlock rather than accept it — produced SPEC 0021 through the
intermediary principle: the teacher is the compressed corpus, and distilling
into FROM-SCRATCH tied weights is the recipe that works (F45) receiving the
signal conversion could not survive. The pilot gate (15% held-out ppl reduction
at equal tokens) is running; the month and its twins are pre-registered behind
it, with budget-bracketed anchors and a hard 30-day stop while v1 ships.

Two operational catches this cycle worth their scars: the automated commit
review caught a client-side bearer token the app agent had introduced (fixed to
the server-side proxy within the cycle), and mid-training Hub streaming stalled
inside the container, replaced by a materialised token cache that also made the
two pilot arms byte-identical in data order.

## Closure — 2026-08-29

The empirical record ends at F54: fifty-four findings, every number reproduced
from its results file under adversarial review, eleven pre-registered gates of
which four closed programmes their author hoped would pass. The validated core:
at equal compute the kinetic tied architecture delivers ~96% of a conventional
transformer's quality with 2.70x fewer parameters; the saving lives in resident
weights and survives in safetensors and ONNX; one checkpoint serves every
budget. The systems side result: routing with fallback beats its best member by
8.3 points pre-registered, is the best measured use of its generation budget,
and is not the best use of its resident memory. The path to open-weight scale is
budget-gated with both affordable shortcuts measured and closed. Shipped: paper
52pp, app four pages with live anytime dial verified over authenticated HTTP,
three HF model repos and one measurements dataset, all of it pushed.

## 2026-08-29 — Cycle 34 opens: SPEC 0022, the twin at 1B

The operator challenged the budget-gated closure with the prabhasa-samskrutam
precedent: 1.13B parameters pretrained on 5.25B tokens in ~71 hours on the
RTX 5090. The challenge holds — the audit's barrier was tokens toward
Qwen-equivalence, never parameter count — and it reopens the one experiment
the closure left on the table: the F45 exchange rate at deployment scale.
TRIZ (segmentation, parameter change, mediated coupling) reconciled the EFE
ranking's tension between the utility-optimal single arm (G=-0.46) and the
rigorous twin (G=-0.19 at 5B): unequal arm lengths. Twin locked-step to 2.5B
tokens closes the architecture claim; the tied arm alone extends to 10B for
utility. Operator locked scope: GPT-2 50k vocab, SFT+HF+app+harness
deliverable, full independence from prabhasa. Spec registered with kill gate
(ppl ratio <=1.20 at 1B tokens), success bar (>=0.95 ladder ratio at 2.5B),
budget cap (<=24 5090-days, floors 2B/6B), measured-preflight GO rule
(>=5.5k tok/s). Data: FineWeb-Edu, documented substitution for Nemotron-CC-HQ
which has no official HF distribution. 5090 verified idle and healthy.

## 2026-08-29 — SPEC 0022 preflight GO (measured, not extrapolated)

The 5090 preflight at final geometry: Arm E (913.0M params, spec-exact)
20,330 tok/s median; Arm T 15,647 tok/s with the disclosed anytime-head
overhead; both arms peak 28.6 GiB of 32, both save/resume round trips
bit-exact. Against the registered GO rule (>=5.5k) the margin is 3-4x, and
the projections are twin phase 3.3 days, extension 4.5, whole programme ~8
training days against a 24-day cap — the SDPA path, per-block checkpointing
and fused AdamW earned their keep. Three defects were caught before they
could cost anything: the pack builder's stream loop never broke once the
train target was met (27GB RSS hang on sample-100BT, found live, regression
test added), a resumed build dropped completed shards from the manifest, and
a resume landing exactly on the target re-entered the unbounded loop. The
public ladder rungs (Pythia-410M/1B, SmolLM2-360M, TinyLlama-1.1B) are being
measured on the GB10 through the same exp40 harness the milestones will use.
Phase 1 launches when the full 10.5B pack lands.

## 2026-08-30 — Phase 1 launched: the twin is training

The 10.5B-token FineWeb-Edu pack finished at 3.3h (21 shards, sha256
manifest, pack hash 973c14c07147) and Phase 1 went up on the 5090 as one
detached job: Arm E (913M explicit) to 2.5B tokens, then Arm T (158M
resident, tied) on the byte-identical stream. Full geometry confirmed in the
first log line — 1.05M tokens per step, gradient checkpointing, SDPA — GPU at
100%, 532W. A persistent watcher relays milestones, loss spikes, aborts and
stage exits. Before launch the rung sweep audited the eval harness the hard
way: every public model scored exact chance, which unmasked a scoring
off-by-one (logits at a token's own position), a degenerate winogrande
protocol (one pair scored twice — 0.493 everywhere, the gold marginal), an
mmlu field-name error, and BOS injection poisoning Llama-family
continuations. All four fixed, tests rewritten around a bigram oracle that a
misaligned scorer cannot pass, and the harness now reproduces published
numbers for all four rungs. The milestone numbers the twins produce will be
the first from this harness that were never wrong in public.

## 2026-08-31 — Arm E complete at 2.5B tokens

34.66 hours at an unwavering 20.15k tok/s: held-out perplexity 1271 -> 503
-> 260 across the milestones, loss curve smooth throughout, no spikes, no
interventions. The ladder at 2.5B remains near chance (arc_easy 0.294, piqa
0.516) while perplexity halves per doubling — the expected ordering at this
budget, and the reason the registered success criterion carries a perplexity
branch: at chance level a ladder ratio is trivially ~1 and carries no
information, so the twin verdict at 2.5B rests on the ppl ratio (<=1.10)
with the ladder reported alongside. Absolute quality trails a Pythia-recipe
model at matched tokens (rotary, cosine decay, GeLU versus this stack's
learned positions, WSD, ReLU); both arms share every one of those choices,
so the comparison is untouched — stated for the paper's honesty, not as a
caveat to the ratio. Arm T began at 12:24 on the byte-identical stream,
157.6M resident parameters against Arm E's 913.0M.

## 2026-08-31 — Redirection absorbed: SPEC 0023 registered, GB10 loaded, spine re-threaded

The operator's architecture analysis exposed the one axis F1-F54 never
varied: every tying ratio is embedding-diluted, (16B+E)/(B+E), and at byte
vocabulary the same arithmetic gives the iteration count itself. SPEC 0023
registers that cell — d=1536, depth 16, vocab 256, arms 453M vs 29M,
predicted ratio 15.8x, gates in SPEC 0022's form — queued on the 5090 behind
the extension per operator decision (extension first, then C1). The byte
pack builds by decoding the existing GPT-2 shards (BPE is byte-reversible;
content identical by construction). The reverse-flow ordering objectives are
being digested from prabhasa's own record, nulls first, before any design
work; and Phase 3 instruct tuning is now an MPO-versus-SFT comparison, which
returns the MMD anchor to the shipped path and gives RQ-4 its answer at real
scale.

The halting idea (analysis §5) took its cheap test on the idle GB10 today
and gave a two-sided answer worth having: solver effort correlates with
next-token expectancy on both existing checkpoints (Spearman 0.27 +/- 0.02
at 121M, 0.19 +/- 0.02 at 46M, both CIs excluding zero), but the
sentence-final contrast points the required direction only at 121M (-0.77
iterations) and reverses at 46M (+1.12). The semantic signal exists; the
closure-halts-early pattern is regime-dependent. Recorded as an exploratory
observation feeding RQ-3b, not a finding.

## 2026-08-31 — Arm T at 0.5B tokens: the gate's first shadow

Held-out perplexity 1961.6 against Arm E's 1271.4 at identical tokens — a
ratio of 1.543 where the 1B kill gate demands 1.20. The gate is registered
at 1B and applies there, not here; what this milestone establishes is the
size of the close required: the tied arm must descend roughly a quarter of
a nat faster over its second half-billion tokens than the explicit arm did
over the same stretch. The F24 regime's history is that tied models lag
early and close late, but that history is from 46-121M on BabyLM; whether
it repeats at 1B on web data is precisely what the gate exists to decide.
Recorded now so that whatever the verdict, no one can say the trajectory
was hidden.

## 2026-09-01 — The kill gate fails at 1B tokens

Arm T held-out perplexity 785.4 against the registered bar of 604.2 — ratio
1.560, and wider than the 1.543 of the 0.5B milestone, so the tied arm is
not closing on this trajectory. In loss terms: 0.44 nats behind at matched
tokens where the gate allowed 0.18. Per SPEC 0022 the extension does not
launch without a new operator decision. Per the closure contract's hard
rule, this gate verdict is recorded now but the FINDING is not a closed
NULL: no failure is declared on attempt one, and at least two documented
interventions precede any NULL. The candidate diagnoses, stated before any
diagnostic runs: the tied block accumulates ~16 gradient contributions per
token where an explicit layer takes one, so the shared learning rate may be
effectively 16x too hot for the block at this width (F24's regime never saw
d=2048); single-epoch web data at this budget is memorization-heavy and the
158M-parameter arm memorizes 5.8x less — the depth-versus-parameters crux
the architecture analysis pre-registered as a risk; and the anytime
supervision weights, tuned at depth 12 and 46-121M, may misallocate at
depth 16. The byte cell (SPEC 0023, pack ready) bears directly on the
second diagnosis. Operator consulted on Phase 1 completion, diagnostic
sequencing, and the utility path.
