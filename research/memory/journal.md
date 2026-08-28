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
