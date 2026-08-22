# Validated Findings

(Populated only with Tarka-reviewed findings as experiments close. Every number here
traces to a real run with config hash + seeds. All statuses: operator sign-off: F1-F15 SIGNED OFF 2026-08-21.)

## F1 — MMD converges linearly to its magnetic fixed point where GDA cycles (H2 core: VALIDATED on symmetric games)

- **Claim:** On symmetric zero-sum matrix games, fixed-magnet MMD (uniform anchor,
  lr=0.1, τ=0.1) exhibits linear (geometric) last-iterate convergence to its fixed
  point, while simultaneous GDA at the same stepsize cycles, bounded away from
  equilibrium.
- **Evidence (10 md5-distinct seeds, 2000 steps; exp01 iteration 2):** log-linear fit
  on distance-to-fixed-point, last 50%: R²=0.9948 (matching pennies), 0.9015 (RPS).
  GDA final NashConv: 1.93 [1.90,1.96] (MP), 1.76 [1.62,1.88] (RPS), R²<0.09 (no
  decay). Artifacts: results/exp01_mmd_vs_gda/ (config sha e1c1efdd…, commit 9a2cde2f).
- **Tarka:** CONFIRM (independent recomputation).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21)

## F2 — Uniform-anchor MMD fixed points ≠ logit-QRE for asymmetric games (DISCOVERY)

- **Claim:** For asymmetric games (biased RPS), MMD-with-uniform-reference dynamics
  converge deterministically, but to an attractor distinct from the logit-QRE(λ=1/τ);
  the dynamics exhibit context-dependent attractors (long-run point NashConv≈0.018 vs
  one-shot computed FP NashConv=0.353). The symmetric-game identity MMD-FP = QRE does
  not generalize.
- **Evidence:** exp01 iteration 2 fixed-point verification (numerical, 50k-step
  lr=1e-5 ground truth) + zero-variance convergence across 10 seeds.
- **Tarka:** CONFIRM_WITH_CORRECTION (wording adopted above). Iteration-1's R²=0.033
  "failure" was a metric artifact (NashConv plateaus at the QRE(τ) floor).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21) · feeds paper §Convergence Analysis and
  motivates adaptive-reference theory work.

## F3 — Regularized Nash Dynamics (periodic reference resets) reach Nash universally (H2: VALIDATED)

- **Claim:** MMD with periodic reference resets converges to Nash (NashConv < 0.05) on
  symmetric AND asymmetric games.
- **Evidence:** final NashConv 8.48e-6 [4.78e-6,1.26e-5] (MP), 1.07e-6 (RPS), 5.21e-5
  [2.62e-5,9.08e-5] (biased RPS); mean R² 0.7181–0.9825. exp01, 10 seeds.
- **Tarka:** CONFIRM_WITH_CORRECTION (R² range corrected as stated).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21)

## F4 — DEQ peak activation memory is O(1) in effective depth; explicit stacks are O(N) (RQ-6 Tier A: VALIDATED)

- **Claim:** DEQ implicit block: 0.032±0.000 MB peak activation memory, flat across
  effective depth (0% variance); explicit stack: linear, slope 0.0168 MB/layer
  (N=4→0.067 MB … N=32→0.539 MB), like-for-like layers, CPU measurement.
- **Evidence:** results/exp03_deq_solvers/ (config sha a0f8f5c0…, 5 seeds).
- **Tarka:** CONFIRM. (GPU-scale measurement for H1's ≤50% claim happens in Tier B.)
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21)

## F5 — Anderson acceleration beats Picard exactly where theory predicts: stiff fixed points (VALIDATED, iteration 2)

- **Claim:** On contraction maps with controlled spectral radius ρ, Anderson/Picard
  iteration ratio < 0.95 at ρ=0.999 (0.888 at dim 32; 0.940 at dim 128); no advantage
  at easy ρ (iteration-1 "miss" explained by problem easiness). Spectral radii
  empirically verified (max abs error ~2e-4).
- **Evidence:** results/exp03_deq_solvers/ iteration 2, ρ∈{0.9,0.99,0.999}, 10 seeds.
- **Tarka:** CONFIRM / CONFIRM_WITH_CORRECTION (wording adopted).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21)

## F6 — Second-price token auction is exactly truthful; weighted aggregation is measurably manipulable (H4 groundwork: VALIDATED)

- **Claim:** Empirical truthful-bidding regret in second-price auctions is exactly 0.0
  (95% CI [0.0,0.0], 16k observations, misreport grid 0.25v–2v). Weighted-aggregation
  mechanism has positive manipulation gain (mean regret 0.0773 at n=3, 0.0683 at n=5)
  — documented as non-truthful, matching the Phase-1 finding that its payments are
  not VCG.
- **Evidence:** results/exp04_auction_truthfulness/ (config sha 5c458dac…, 10 seeds ×
  200 auctions × {3,5} agents).
- **Tarka:** CONFIRM (both claims).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21)

## F7 — Warm-started homotopy accelerates QRE path tracing (VALIDATED, iteration 3)

- **Claim:** Warm-starting each QRE solve from the previous λ's solution reduces total
  solver iterations vs cold-start: 25.2% on asymmetric 2×2 (5990→4481 avg), 2.6% on
  biased RPS; degenerate control (matching pennies) flat as expected. Exploitability
  along the path is NOT globally monotone for these games (honest partial vs prereg);
  path strategy movement is smooth and small (0.015 < 0.05 prereg threshold).
- **Evidence:** results/exp02_qre_homotopy/ iteration 3, λ∈logspace(0.01,100,50).
- **Status:** VALIDATED (with two honest partials) · SIGNED OFF (operator, 2026-08-21)

## F8 — Undamped logit-QRE fixed-point iteration requires damping beyond small λ (METHOD FINDING)

- **Claim:** Plain s←softmax(λAs) diverges for λ‖A‖ moderately large (biased RPS at
  λ>0.32). Adaptive damped iteration s←(1-γ)s+γ·softmax(λAs) (γ init 1/(1+λ/10),
  halve on residual increase) converges across λ∈{1,10,100} (21 / 700 / 42k iters).
- **Evidence:** tests/test_qre.py::TestQREHighRationality; kinetic_ai/games/qre.py
  damped solver (default on, backward compatible).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21) · exposes future work: Anderson-accelerated
  QRE solves (42k iters at λ=100 is the next bottleneck).

## F9 — Tier B pipeline validated end-to-end; EqLM trains but slowly; raw-MMD arm exposed an adaptivity confound (SMOKE, method finding)

- **What ran (exp05 iteration 2, GB10, 800 steps/arm, param-matched within 5%,
  1000 BLiMP pairs, ~7 min GPU):** A1 ExplicitLM+AdamW loss 125→4.39; A2 EqLM+AdamW
  132→28.5 (learning, ~10x slower); A3 EqLM+raw-MMD 128→125 (no learning).
- **Honest read:** BLiMP values (0.459/0.513/0.537) are noise around chance at these
  loss levels — NOT evidence of an MMD win (contra the run agent's initial claim,
  rejected on review). All arms start at loss ≈125 ≫ ln(vocab)≈10.8 → unscaled
  weight-tied logits at init. A3's flatline is an optimizer-adaptivity confound:
  Euclidean MMD is unpreconditioned SGD+magnet (see ADR 0003).
- **Value:** data→tokenize→train→BLiMP pipeline proven on GB10; three concrete
  defects identified with fixes scheduled (init scale, MagneticAdamW, DEQ solver
  stats capture; data loader currently limited to first 1000 samples — noted).
- **Status:** VALIDATED as method finding · SIGNED OFF (operator, 2026-08-21)

## F10 — EqLM matches the explicit transformer at smoke scale after init fix (H1 signal; SMOKE)

- **Claim:** With correct init (all arms start at CE 10.82–10.83 = ln|V|), EqLM+AdamW
  reaches final loss 5.94 vs ExplicitLM 5.99 over identical 800-step budgets —
  parity — with 6.2% fewer parameters (10.67M vs 11.06M), at 2.7x wall time
  (177.6s vs 66.8s; 12 fixed-point iters/forward). BLiMP still ≈ chance at this
  scale (0.452/0.465) as expected for a 3-minute pretrain.
- **Evidence:** results/exp05_eqlm_pretrain/results.json iteration 3 (config sha
  3ed6e8fc…, commit 2151a19).
- **Caveats found (feed cycle 5):** (a) MagneticAdamW arm throttled learning
  (10.83→9.91) — tau/EMA coupling mis-tuned; (b) MagneticAdamW peak memory 97.7GB
  vs 3.5GB — memory bug; (c) DEQ solver 0% convergence at max_iter=12/tol=1e-3
  during training yet training succeeds (phantom-gradient regime) — needs
  characterization before full run.
- **Status:** VALIDATED (smoke scale) · SIGNED OFF (operator, 2026-08-21)

## F10-correction — exp05 token-budget description was wrong (data cap)

- exp05 iterations 2–3 cycled a 22,703-unique-token corpus (the 1000-sample cap,
  F9 blocker) for ~3.3M token-PRESENTATIONS — not 3.3M unique tokens. The F10
  parity claim remains valid as a like-for-like comparison (identical data both
  arms), but describes an overfitting-regime corpus. Cap fixed in
  kinetic_ai/data/dataset.py (streaming until max_tokens; regression-tested);
  exp06 onward uses the real stream.

## F11 — MagneticAdamW coupled-weight-decay bug: Adam-L2 destroys sparse-gradient parameters (METHOD FINDING, fixed)

- **Claim:** Folding weight decay into the gradient before Adam normalization makes
  wd·θ act as a ~lr-magnitude decay on parameters whose gradients are mostly zero
  (tied embedding rows): EqLM+MagneticAdamW(τ=0) reached only 10.46 vs torch AdamW
  8.82 on identical setup, τ-independent. Decoupling wd (θ←θ(1−lr·wd)−step) restores
  EXACT AdamW equivalence at τ=0 (8.8236 ≡ 8.8236) and τ∈{1e-4,1e-3} preserve
  learning (8.8236/8.8237). Root-caused by isolated per-process A/B; regression test
  on a sparse-grad embedding now enforces equivalence.
- **Lesson:** optimizer-equivalence tests must include sparse-gradient parameters;
  dense-Linear tests masked the defect.
- **Status:** VALIDATED · fixed in kinetic_ai/optim/magnetic_adamw.py · SIGNED OFF (operator, 2026-08-21)

## F12 — Magnetic pull (EMA anchor, τ≤1e-2) is loss-neutral at pretraining scale; solver budget beyond 12 iters buys nothing at 300 steps (exp06/exp06b)

- **Evidence (real data stream, 300 steps, fixed optimizer):** magnetic arms
  8.52–8.61 vs explicit baseline 8.51 (within 10% prereg: MET). Drift prereg was
  MIS-DESIGNED (compared EqLM arms to the explicit-architecture baseline) — recorded
  as invalid, not "missed"; magnet-vs-no-magnet drift at these τ is negligible per
  the closed form (shrink ≤ lr·τ = 3e-6/step vs trailing EMA). exp06b: max_iter
  {12,24,48} → loss {8.60,8.67,8.67}, 0% solver convergence at tol 1e-3 in ALL
  budgets — phantom-gradient training is loss-neutral here; tol 1e-3 appears
  unreachable for this block, needs tol study not iteration study.
- **Tier B decision:** pretraining arms use AdamW (A1/A2); MagneticAdamW with FIXED
  reference is reserved for H3 preference-phase (its theoretical home). DEQ
  max_iter stays 12 (cheapest; no loss penalty).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21)

## F13 — H1 iteration 1: MISSED at scale; diagnosis points to non-contractive fixed-point map (Tier B full run)

- **What ran (exp05_full, 20k steps/arm, full strict-small stream, param-matched,
  2000-pair BLiMP eval on 1000 scored pairs, config sha 8a2fa16e, commit 385f5d4):**
  A1 ExplicitLM: loss 3.90, BLiMP 0.734 (credible BabyLM-class baseline).
  A2 EqLM: 4.42, BLiMP 0.571. A3 EqLM+MagneticAdamW(1e-3): 4.68, BLiMP 0.584.
  EqLM = 78–80% of baseline BLiMP vs pre-registered ≥95% ⇒ **H1 iter-1 MISSED**.
  A3−A2 = +1.3pp ≈ 0.8σ (n=1000) — not significant. Wall time 2.9x; peak memory
  comparable (logit activations dominate at this scale, depth-memory advantage
  not visible).
- **Diagnosis:** DEQ solver converged 0% of forward passes at tol 1e-3 across ALL
  budgets (12/24/48 iters; exp06b) ⇒ the block's fixed-point map is not
  contractive — spectral normalization exists in kinetic_ai/models/deq_layer.py
  (apply_spectral_norm) but was never wired into EqLM (known Phase-0 gap). Smoke
  "parity" (F10) held only in the memorization regime (22.7k-token cycled corpus).
- **Next iteration (EqLM-v2, per operator ethos hypotheses→invention):** enforce
  contraction (spectral norm on block weights / pcDEQ constraints), verify >80%
  solver convergence, then rerun the matched comparison; secondary: solver-tol
  study (1e-2 vs 1e-3), solver-depth warmup.
- **Status:** VALIDATED (honest miss + diagnosis) · SIGNED OFF (operator, 2026-08-21)

## F14 — EqLM's map has no bona fide fixed point: it is a weight-tied iterated transformer, not yet an equilibrium model (DISCOVERY via exp07)

- **Evidence:** Solver residual plateaus at constant value with tail-ratio ≈0.99 for
  100 iterations across v1/v2; residual magnitude scales LINEARLY with damping α
  (32.65 → 5.41 → 1.35 at α = 1 / 0.2 / 0.05) — the signature of z ← z + α·g(z)
  with ‖g‖ constant: iterates drift at speed α‖g‖, no fixed point is approached.
  Spectral norm on sub-layers cannot fix a residual map with no outer bounding
  operation. Additionally the solver's convergence criterion is an ABSOLUTE global
  norm over the batch tensor (deq_layer.py:68) — unsatisfiable at batch scale, so
  convergence % was never meaningful (retro-scopes the F12/F13 "0% convergence").
- **Reframing of F13:** the trained "EqLM" models are 12-iteration weight-tied
  transformers; their BLiMP 0.571/0.584 quantify weight-tying at matched params,
  not equilibrium computation.
- **EqLM-v3 design (next iteration):** (a) put the outer LayerNorm inside the map —
  f(z,x)=LN(z + Attn + MLP + inj(x)) (Bai et al. DEQ-transformer form) so iterates
  are bounded and fixed points exist; (b) record RELATIVE residual
  ‖Δz‖/(‖z‖+eps) in deq_layer and gate convergence on it; (c) re-verify
  contraction empirically, then rerun the matched comparison.
- **exp07 numbers (for the record):** losses A1 8.71 / v2 8.88 / v1 8.84 (300
  steps, real stream) — loss-parity prereg MET, convergence prereg NOT MET.
- **Status:** VALIDATED (diagnosis reproducible in-repo) · SIGNED OFF (operator, 2026-08-21)

## F15 — EqLM-v3 (post-LN map): fixed points now exist but contraction is weak at LM width (H1 frontier identified)

- **Evidence:** exp07 four-arm smoke (300 steps, real stream): v3 loss 8.84 (parity
  prereg MET vs v1 8.82; explicit 8.72); convergence prereg NOT MET at
  max_iter=12. Direct probe at full smoke width (d=204, T=127, 80 iters):
  relative residual decays 1.0 → 0.105 monotonically then creeps (0.1078 → 0.1053
  over the last 24 iters) — the F14 constant-speed-drift signature is ELIMINATED
  (bounded iterates, genuine approach), but the map's effective contraction factor
  is ≈1 at this width, so tol 1e-2 is unreachable in practical iteration budgets.
- **Interpretation:** the architecture question H1 poses is now precise: make the
  post-LN transformer map strongly contractive at width WITHOUT destroying
  capacity. Candidate v4 directions (each a testable arm): damping/α annealing
  schedules; solver-aware auxiliary loss penalizing ‖f(z*)−z*‖; tighter spectral
  budget on attention value/projection paths only; pcDEQ orthant constraints;
  tol relaxation study (does rel 0.1 suffice for representation quality?).
- **Status:** VALIDATED · SIGNED OFF (operator, 2026-08-21) · H1 iteration 2 remains open — this is
  the identified scientific frontier of the program, with the falsifiable next arms
  above.

## F16 — Solver-aware auxiliary loss teaches contraction almost for free (BREAKTHROUGH; resolves F15 frontier)

- **Claim:** Adding L = CE + λ·r with r = ‖f(z*,x)−z*‖/‖z*‖ (one tracked block
  application after the no-grad solve) reduces the exit relative residual 16×
  (0.1277 → 0.0076–0.0078) for every λ ∈ {0.01, 0.1, 1.0}, at ≤1.8% CE cost
  (8.92–8.99 vs control 8.77) and zero wall-time overhead (0.99–1.00×). The
  post-LN map becomes a genuine equilibrium computation (rel residual ≪ 1e-2
  target) — the model LEARNS to be contractive.
- **Evidence:** results/exp08_solver_aware/results.json (300 steps, real stream,
  4 arms; prereg criterion met by all three λ arms).
- **Consequence:** H1 iteration 2 unlocked: full 20k-step matched comparison with
  EqLM-v4 (postln + aux λ=0.01).
- **Status:** VALIDATED (smoke scale) · pending full-run confirmation

## F17 — H1 iteration 2: post-LN EqLM reaches 94.4% of baseline BLiMP (near-miss, within noise); aux loss trades capacity for contraction at scale

- **Evidence (exp05_full_v4, 20k steps, full stream, param-matched, 1000 scored
  BLiMP pairs; config sha 8f3c8969, commit e360e7c5):** A1 ExplicitLM 0.746 (loss
  3.75); A3 EqLM-v3 postln no-aux 0.704 = 94.4% of A1 (loss 4.45) — 0.6pp below
  the pre-registered 95% threshold, within 1σ (binomial σ≈1.4pp at n=1000):
  statistically a TIE with the threshold, seeds required to adjudicate. A2
  EqLM-v4 aux λ=0.01: 0.658 (loss 4.43) — the solver-aware loss, ≤1.8% CE cost
  at 300 steps (F16), costs ~4.6pp BLiMP compounded over 20k steps: contraction
  and task capacity trade off at fixed λ.
- **Iteration-1→2 progress:** EqLM 0.571 → 0.704 (arch fixes alone: post-LN map +
  init scaling). Wall time 2.9x; memory parity.
- **Next (iteration 3, pre-registered):** 2 more seeds of A1/A3 to resolve the
  near-miss with mean±CI (CLAUDE.md ≥3-seed gate); rider hypothesis: λ=1e-3 aux
  arm (10x smaller) preserves BLiMP while keeping residual gains.
- **Status:** VALIDATED · seeds pending for H1 verdict

## F18 — H1 verdict (3 seeds): EqLM reaches 93.0% of the explicit baseline's BLiMP — formally below the 95% threshold (HONEST MISS with a tight CI)

- **Evidence (seeds 42/43/44, 20k steps each, full stream, param-matched;
  results/exp05_full_v4{,_s43,_s44}):** A1 explicit BLiMP 0.7133±0.0294
  (0.746/0.689/0.705); A3 EqLM post-LN 0.6637±0.0365 (0.704/0.654/0.633).
  Per-seed ratio 0.944/0.949/0.898; mean 0.9303, 95% bootstrap CI
  [0.8979, 0.9492] — upper bound < 0.95 ⇒ pre-registered H1 threshold MISSED.
  Paired bootstrap: Δ=+4.97pp for explicit, p<1e-3, effect size 1.84.
  λ=1e-3 aux rider (seeds 43/44): 0.620/0.643 — no BLiMP benefit vs no-aux.
- **Trajectory of the program:** iteration 1 ratio 0.78 → iteration 3 ratio 0.93
  via diagnosed fixes (init scale, post-LN fixed-point map). Memory parity at this
  scale; wall time 2.9x; O(1) depth-memory advantage retained (F4).
- **Paper position:** a weight-tied equilibrium LM attains 93% of a param-matched
  explicit transformer's BLiMP at equal token budget — reported with the full
  diagnostic arc (F13–F17) and the open gap as future work (eval-time solver
  budget, aux annealing, multi-block DEQ).
- **Status:** VALIDATED (formal H1 miss) · closes RQ-3 iteration 3

## F19 — Warm-started equilibrium decoding cuts solver cost 79% at 97.6% token agreement (H1′a: reduction PASS, agreement narrow miss; smoke scale)

- **Evidence (exp09, 500-step aux-trained EqLM d=204, 40 prompts × 25 tokens,
  config sha 54698443; agreement probe on the saved checkpoint, seed 7):**
  cold 7.91 iters/token → warm 1.64 (−79.3%; prereg ≥50% PASS). Greedy token
  agreement 97.6% (976/1000), 39/40 sequences exact — prereg ≥99% narrowly
  missed (one divergent sequence). Knob for the scale run: slightly higher warm
  budget/tighter tol should close agreement while retaining ≫50% savings.
- **Meaning:** the TRIZ P10 prediction holds — sequential equilibria are close,
  so equilibrium decoding pays ~1.6 block applications/token. This inverts the
  wall-clock story: an explicit L-layer stack always pays L; a warm equilibrium
  model pays ~1.6 at smoke scale. Full verdict needs the 100M run (exp10).
- **H1′c (think-harder dial):** BLiMP flat across budgets 6→48
  (0.594/0.600/0.600/0.600) at this training scale — honest null at 500 steps;
  re-measured at 100M scale.
- **Checkpoint saved** (results/exp09_adaptive/checkpoints/eqlm_smoke.pt) — the
  checkpointing path for HF publication is proven.
- **Status:** VALIDATED (smoke) · scale confirmation in exp10
