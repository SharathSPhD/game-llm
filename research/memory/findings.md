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
- **Status:** VALIDATED (smoke scale) · SIGNED-OFF (operator, 2026-08-27)

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
- **Status:** VALIDATED · SIGNED-OFF (operator, 2026-08-27)

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
- **Status:** VALIDATED (formal H1 miss) · SIGNED-OFF (operator, 2026-08-27) · closes RQ-3 iteration 3

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
- **Status:** VALIDATED (smoke) · SIGNED-OFF (operator, 2026-08-27) · scale-confirmed in exp10 (F20)

## F20 — H1-at-scale verdict (121M, 3 seeds): matched-budget quality ratio DROPS to 0.787; mechanism is graded contraction loss at width, not binary convergence failure

- **Evidence (seeds 42/43/44 on RTX 5090, batch 32 x 10k steps, full BabyLM
  stream, param-matched 120.7M vs 123.8M; results/exp10_5090/exp10_seed4{2,3,4}/
  results.json, config hashes 2a92f4f1/9b2f58ad/8944b1f3):**
  A1 explicit BLiMP 0.6833 (0.682/0.675/0.693) vs A3 EqLM-v3 0.5377
  (0.537/0.532/0.544); per-seed ratio 0.7874/0.7881/0.7850, mean **0.7868**,
  bootstrap 95% CI [0.785, 0.788]; paired t = -82.6 (df=2). A2 (v1 residual
  map) statistically identical to A3 at this scale (0.5383). The 300-step
  probe loss lead (EqLM 6.79 vs 7.40) INVERTED by 10k steps (A1 final loss
  2.73-3.07 vs EqLM 3.29-4.00).
- **Mechanism (Tarka-corrected):** solver telemetry shows convergence_rate
  = 0.0 with mean iterations 12.0/12 on every step of all six 121M EqLM runs
  — but Tarka review found the SAME telemetry (0.0, 12/12) in the F18 d=204
  runs. The fixed-point solve was budget-truncated at BOTH scales; what
  changed is how much quality that truncation costs. As width grows 204 ->
  1704 under a fixed 12-iteration budget, the map becomes progressively less
  contractive per iteration and the truncation penalty widens from 7% to 21%
  — a graded contraction-vs-width effect, not a binary convergence failure.
  Certified equilibrium remains unachieved at either scale within feasible
  budgets; that is the honest open problem.
- **Cost frontier (honest):** EqLM keeps the memory edge at width — A2 peak
  6.29GB vs A1 8.13GB (-23%; matches GB10 probe) — but pays 8.4x wall-clock
  per matched step (92 vs 11 min/arm). Matched-compute comparison would favor
  the explicit baseline even more strongly.
- **Scale trend (rescoped per Tarka):** ratio 0.930 [0.898, 0.949] at
  11M/20k steps -> 0.787 [0.785, 0.788] at 121M/10k steps. The step budget
  differs (20k vs 10k, applied identically to both arms), so the width-driven
  component cannot be fully isolated from the budget component without a
  matched-step run; the direction of the trend is robust (probe loss lead
  inverted well before 10k), the magnitude is confounded. Memory caveat:
  peak_memory_mb is bit-identical across seeds (deterministic allocator peak
  for a fixed graph) — treated as an architecture-level measurement.
- **Paper position:** equilibrium LMs buy O(1) depth-memory (-23% measured at
  121M) and 79% cheaper warm-started decoding (F19), paying a quality gap
  that widens from 7% to 21% as width scales past the solver's contraction
  budget. Open problem: contraction that survives width (better fixed-point
  maps / adaptive iteration budgets), not more parameters.
- **Tarka review:** claims 1 (scores) and 3 (memory/wall-clock) CONFIRMED;
  original mechanistic claim REFUTED and rewritten above; scale trend
  rescoped for the step-budget confound. Review resolved 2026-08-27.
- **Status:** VALIDATED (3 seeds, tight CI) · SIGNED-OFF (operator,
  2026-08-27) · closes the H1/H1' empirical program

## F21 — H3 verdict (3 seeds): PARTIAL — at the pre-registered tau range MPO is indistinguishable from DPO; the magnet is real but under-dosed. DPO damages unseen phenomena; EqLM is ~1655x more drift-resistant

- **Protocol (SPEC 0007 + amendments):** DPO loss on BLiMP preference pairs
  (600 train / 400 held-out, phenomenon-level split), bases = each seed's own
  exp10 121M checkpoints, arms differ ONLY in MagneticAdamW tau (P1 tau=0 ==
  decoupled AdamW; P2a 1e-3; P2b 1e-2; ref_mode="fixed" = frozen base),
  beta=0.1, lr=1e-5, 3 epochs (~675 steps). Seeds 42/43/44; bases
  md5-distinct (verified); results/exp11_seed4{2,3,4}/results.json.
- **Pre-registered verdict: PARTIAL.** Held-out accuracy: P2 == P1 exactly
  (all seeds, both bases) — the ">= DPO" half holds by equality. Lower KL
  drift: NOT demonstrated — P2b-P1 on the explicit base is -0.0018/-0.0009/
  +0.0004 (mean -0.0008, ns); P2a mixed. Diagnosis: total magnetic
  displacement ~ lr*tau*T ~ 2e-5 relative at the pre-registered tau.
- **Secondary finding A (3 seeds): preference optimization damages unseen
  phenomena.** Explicit base: train-phenomena accuracy 0.646->0.877 while
  held-out phenomena DROP 0.740->0.612 mean, KL-to-base ~1.24 nats/token.
- **Secondary finding B: the equilibrium architecture is intrinsically
  drift-resistant.** Identical loss/lr/steps: EqLM train acc 0.493->0.516,
  KL drift 0.00075 — 1655x less than explicit. Cause: damped gradients
  through the 12-iteration weight-tied solve. Double-edged and reported as
  such: stability against preference drift AND insensitivity to preference
  fine-tuning at matched lr.
- **Tarka review (resolved 2026-08-27):** all claims CONFIRMED; bug
  hypothesis (tau ignored) excluded — KL trajectories diverge across arms
  and per-arm losses differ; eqlm base train acc LOW (0.493) rules out
  already-satisfied preferences.
- **Rider result (exp11b, 3 seeds):** tau dose-response {0.1, 1, 10}:
  explicit KL 1.2406 -> 1.2414 -> 1.2398 -> 1.2256; held-out accuracy
  unchanged at every tau; EqLM flat at 0.00074. Rider prediction ("visible
  by tau=1") REFUTED: the magnetic proximal pull is second-order to the DPO
  gradient across tau in [1e-3, 10] at this budget. H3 stands at PARTIAL on
  letter, MISSED in spirit (mirrors ADR 0003: the magnet's home is
  pretraining stability, not preference-phase drift control).
- **Status:** VALIDATED (pre-registered arms, Tarka-resolved) · verdict
  PARTIAL · SIGNED-OFF (operator, 2026-08-27)

## F22 — H4 verdict (3 seeds): MET — truthful second-price token-auction selection beats the best single specialist by 23% and the logit-average ensemble by 12% on mixed-domain perplexity

- **Protocol (SPEC 0008):** two 30.0M-param explicit specialists per seed,
  trained 3k steps on disjoint BabyLM subdomains (A: childes child-directed
  speech; B: simple_wiki written text; line-level 5% held-out per seed);
  eval on a 50/50 interleaved held-out mixed stream (~100k tokens). Per
  token: bid_i = own max-prob confidence (target-independent); winner's
  distribution scores the token; winner pays second price (F6-validated;
  vectorized path cross-checked against TokenAuction — 0 mismatches all
  seeds). Seeds 42/43/44; results/exp12/results_seed4{2,3,4}.json.
- **Pre-registered verdict: MET (3/3 seeds).** Mixed-domain perplexity
  (means): auction 182.8 < uniform logit-average ensemble 207.9 < best
  single specialist 236.8 (S_A) < worst 1242.4 (S_B). Auction beats BEST
  single on every seed (158.5/189.4/200.5 vs 234.3/232.8/243.4; paired
  t=4.98) and also beats the ensemble on every seed.
- **How it wins:** confidence-bid selection is imperfect within-domain (on
  domain A the auction's 26-33 ppl trails pure S_A's 13.8; win-frac_A ~0.61
  on a 50/50 stream) but per-token specialist selection dominates any fixed
  commitment when domains mix: S_A collapses on domain B (3979-4280 ppl)
  and S_B on domain A (4477-4856).
- **Tarka review (resolved 2026-08-27): MET confirmed; oracle leakage
  CLEARED (bids target-independent); fairness CLEARED. Three rescopings
  applied:** (1) teacher-forced per-token SELECTION at scoring time, not
  autoregressive generation — claim scoped to "token-auction selection";
  autoregressive decoding = follow-up program. (2) S_A's absolute 13.8
  childes ppl may be inflated by n-gram overlap across the line-level split
  (repetitive child-directed speech); affects absolute numbers only — all
  systems share the same eval stream. (3) traces_seed*.json are a SAMPLE of
  the first 200 positions (win-frac_A ~0.80 there vs 0.61 full-stream) —
  labeled as such.
- **App tie-in delivered:** real bid/winner/payment traces served by
  GET /api/auction/traces for the Auction playground.
- **Status:** VALIDATED (3 seeds, Tarka-resolved with rescoping) · verdict
  MET · SIGNED-OFF (operator, 2026-08-27)

## F23 — H5 verdict (3 seeds): MISSED — the auction's teacher-forced selection advantage INVERTS in closed-loop generation, empirically vindicating F22's Tarka rescoping

- **Protocol (SPEC 0009, pre-registered):** exp12 specialists per seed, 100
  held-out mixed prefixes (50/50 childes/simple_wiki, 32 tokens), 32
  generated tokens, greedy; judge = frozen exp10 seed-42 124M explicit LM,
  NLL/token of generated text only. results/exp14/results_seed4{2,3,4}.json.
- **Verdict: MISSED (3/3).** Judge NLL: best single S_A 3.39/3.37/3.69 <
  auction 4.74/4.23/4.60 on every seed — the pre-registered comparison.
  Ordering beyond that is variable (seeds 42/43: AUC < ENS < S_B; seed 44:
  S_B 5.51 < ENS 5.70). The auction beats the uniform ensemble on 2/3
  seeds and the worse specialist on 2/3, but loses to the best single
  model once each system generates its own context — per-token model
  switching creates style hand-offs that the fixed model never pays.
- **Why (from secondaries):** the auction drifts toward S_A everywhere
  (domain consistency ~1.0 on childes prompts but ~0.02 on wiki prompts —
  on wiki prefixes its continuations end up S_A-flavored), while under
  teacher forcing the TRUE context kept pulling selection back to the
  right specialist each token. Closed-loop removes that anchor. Notable
  positive: the auction has the LOWEST degeneration of all systems
  (3-gram repetition 0.387-0.483 vs S_A 0.517-0.601; the ensemble collapses
  to 0.78-0.83) — bid competition acts as an implicit anti-repetition
  regularizer, worth its own study.
- **Per-domain judge decomposition (Tarka-required; CPU rerun on GB10
  reproduced the 5090 aggregates to 4 decimals, results/exp14_local/):**
  on wiki-only prompts S_A still scores best (3.27/3.34/3.78 vs AUC
  4.58/4.54/4.62) — but so does everything childes-flavored: S_B generating
  its own home style on its own domain scores 5.10-5.49, worse than S_A
  off-domain. The judge (trained on child-speech-heavy BabyLM strict-small)
  carries a ~2-nat style prior for childes-flavored text regardless of
  prompt — larger than any between-system effect measured. The primary
  metric is therefore style-dominated: the pre-registered MISSED verdict
  stands as registered, but the mechanism claim is RESCOPED to
  judge-relative — "the closed-loop auction advantage is NOT demonstrated
  under a BabyLM-distribution judge"; whether it exists under a
  style-neutral judge is untested (no unbiased judge exists at this
  scale in-repo; external-LM judging = future work).
- **What survives cleanly regardless of judge:** (a) the auction's output
  distribution genuinely diverges from its teacher-forced behavior (drift
  to S_A style measured independently of the judge); (b) the auction has
  the lowest degeneration of all systems and the ensemble collapses (
  repetition is judge-free); (c) the pre-registered closed-loop claim
  failed to be met — F22 remains correctly scoped to scoring time.
- **Caveats:** domain-consistency uses the specialists themselves as
  scorers (self-aware circularity; appropriate for drift detection, noted
  for transparency). Greedy-only decoding; sampling could change hand-off
  dynamics.
- **Meaning for the program:** F22 (scoring-time selection wins) and F23
  (closed-loop generation loses) are BOTH true; the F22 Tarka rescoping
  that insisted on the distinction is empirically vindicated. Auction
  aggregation belongs at scoring/reranking time, or needs context-aware
  bids (bid on the PREFIX's domain, not own confidence) as future work.
- **Tarka review (resolved 2026-08-27):** core MISSED CONFIRMED all seeds;
  ordering-tail and repetition ranges corrected; judge-bias confound
  investigated via the per-domain decomposition above and the finding
  rescoped to judge-relative accordingly.
- **Status:** VALIDATED (3 seeds, Tarka-resolved with rescoping) · verdict
  MISSED under the pre-registered judge · SIGNED-OFF (operator, 2026-08-27)

## F24 — H6 verdict (3 seeds): PARTIAL, with the quality half met at PARITY — anytime-unrolled training closes the entire width gap (ratio 0.991), certification achieved only in a separate arm, and the naive combination refuted

- **Protocol (SPEC 0010 + amendment, exp13 on RTX 5090):** TRIZ arms vs the
  exp10 A3 control (0.537/0.532/0.544 per seed; explicit A1 0.682/0.675/
  0.693), identical data/steps/batch (10k x 32, exp10 token cache), 121M
  param-matched. results/exp13_seed4{2,3,4}, exp13b_seed42.
- **B1 anytime (3 seeds): BLiMP 0.662/0.697/0.672, ratio vs explicit
  0.971/1.033/0.970 — mean 0.991.** Gap closure 86%/115%/86% (mean 96%);
  seed 43 EXCEEDS its explicit baseline. The weight-tied single block,
  trained unrolled with CE on iterates z4/z8/z12 and evaluated with the
  standard 12-iteration solver, reaches statistical parity with a
  12-layer explicit transformer at matched budget. Eval note (Tarka): the
  solver is Anderson-accelerated — training supervises PLAIN iterates
  z4/z8/z12 while eval budgets cap Anderson iterations, a different
  algorithm; the parity and degradation results are claims about the
  Anderson-eval path, not about plain-iteration equivalence. This supersedes F20's
  pessimistic scale trend: the widening gap was a property of
  IFT-solver-based training, not of the weight-tied architecture.
- **Budget-sweep rider: MET as predicted** (pre-registered: >=0.60 @8,
  >=0.55 @4): budgets 4/8/12 -> 0.62/0.64/0.66 (s42), 0.60/0.67/0.70
  (s43), 0.59/0.66/0.67 (s44). One model evaluated under the Anderson
  solver at three iteration budgets degrades gracefully — the P11 anytime
  property demonstrated on the deployed eval path.
- **B2 (seed 42, from the screen): certified convergence 1.00 at 4.0 mean
  iterations** — the program's first certified 121M equilibrium — at a
  quality cost (0.577, still above control) and the lowest memory (5.5GB).
- **B4 combination: pre-registered prediction REFUTED.** Anytime + raw
  trajectory penalty scored 0.529 (below control) with conv 0.0 and
  L-hat stuck ~2.5-3.9; the un-normalized penalty (3.5e4 at init) drowned
  the CE signal under grad-clip. Recorded as a documented failed
  intervention; a log-scale penalty is the natural next iteration (not
  run — program timeboxed at the operator's closure directive).
- **Verdict vs pre-registered gates: PARTIAL** — no single arm has BLiMP
  >= 0.610 AND certified convergence > 0.5 (B1: quality without
  certification; B2: certification without sufficient quality). Both
  PARTIAL criteria are individually exceeded. The open problem is now
  sharper and better-posed: QUALITY-PRESERVING certification — the two
  halves are separately solved and their naive sum provably fails.
- **Honest costs:** B1's unrolled TRAINING pays explicit-like activation
  memory (16.4GB vs A3's 7.8GB; a property of storing the unrolled
  iterates for backprop, not of the architecture) — the training-memory advantage is traded
  for quality; the serving-time properties (O(1) depth-memory, warm-start
  decoding F19, think-harder dial) are unchanged since eval still runs
  the solver. Training wall-clock BONUS: unrolled is 2.1x faster than
  IFT-solver training (44 vs 92 min/arm) — the solve, not backprop,
  dominates EqLM training cost.
- **Tarka review (resolved 2026-08-27):** all five claims CONFIRMED;
  like-for-like audited (identical optimizer/lr/grad-clip/steps/batch/
  cache; params within 2.5%); seeds md5-distinct; rescoped the eval-path
  wording (Anderson vs plain iteration) and the memory attribution as
  applied above.
- **Status:** VALIDATED (3 seeds for B1; screen-level for B2/B3/B4) ·
  verdict PARTIAL (quality half at parity) · SIGNED-OFF (operator, 2026-08-27)

### F24 addendum — generation exposes the decode-path mismatch (resolved 2026-08-28)

Operator review of the live playground showed the anytime (B1) checkpoint
generating degenerate text (punctuation/word loops) while scoring 0.697
BLiMP. Bisect (results in-session, reproducible via B1 seed-43 checkpoint):
explicit baseline through the identical greedy path generates coherent
English (exonerates tokenizer/serving); B1 through Anderson-12 produces a
near-flat next-token distribution (top-5 logits 6.87..6.62); B1 through its
TRAINING-TIME computation (plain 12x unrolled from z0=x) produces a sharp
distribution (8.44 top) and baseline-class text. Conclusion: the
Anderson-vs-plain eval-path mismatch Tarka scoped in F24 corrupts ABSOLUTE
distributions while leaving RELATIVE (BLiMP) comparisons intact —
generation is the sensitive assay. Resolution shipped: (a) checkpoints
carry decode_mode ("solver" | "unrolled") and EqLM.generate routes through
the training-matched computation (B1 checkpoints patched); (b) sampling
(temperature/top-k, shared helper) added to generate(), the serving layer,
and the playground UI — greedy loops are ordinary small-LM degeneration
(both architectures loop; Holtzman et al.). Sampled unrolled B1 output is
grammatical BabyLM-register English. Published F24 BLiMP numbers are
unchanged (they attach to the Anderson eval path, as scoped).

## F25 — Conversion damage curve: which layers can be tied, and at what cost (SPEC 0011 design measurement)

- **Setup:** Qwen3-1.7B (28 layers, d=2048, 1.721B params) converted to the
  KineticLM topology — explicit outer layers around block-recursive shared
  cores — and measured WITHOUT any uptraining. Perplexity on a 3-passage
  general sample; base ppl 6.01. Sweep over {n_pre=n_post in 6,8} x
  {n_cores 1,2,4} x {average, stepwise} init. Artifact:
  results/scale/damage_sweep.json (5090, bf16).
- **(1) Average init dominates stepwise by 10-100x** at every configuration
  (e.g. 8+8/1 core: 1.9e3 vs 2.2e5). Collapsing a group of layers to their
  elementwise mean preserves far more function than adopting any single
  representative layer. This settles an open choice in the published
  recursive-uptraining recipes for this model family.
- **(2) Explicit outer layers matter more than the number of cores.** Going
  6+6 -> 8+8 with a single core improves ppl 10x (18465 -> 1909), while going
  1 -> 2 cores at 8+8 changes nothing (1909 vs 1933). The first and last two
  layers carry function that tying destroys; the middle is far more
  redundant. Parameter saving is therefore best bought in the middle.
- **(3) Every configuration is heavily damaged pre-uptraining** (300-3000x
  base ppl). Conversion is not a free lunch at any tying level tested; the
  uptraining budget, not the surgery, is the binding constraint — consistent
  with published recipes using 10-100B tokens.
- **Operating point selected (pre-registered before uptraining):**
  n_pre=n_post=8, n_cores=1, average init -> 1.167B params (**68% of base**),
  starting ppl 1909. SPEC 0011's parameter gate is amended from <=60% to
  <=70% on this evidence: the 56-59% configs start 3-10x more damaged and
  would not be recoverable within our token budget. Amendment recorded BEFORE
  any uptraining run.
- **Status:** VALIDATED (design measurement, single seed — it selects a
  configuration, it does not claim a scientific result) · feeds SPEC 0011

## F26 — H7 verdict (1.7B conversion): PARTIAL. Perplexity nearly recovers (1.19x base) while benchmark capability does not (0.533 retention), and the loss is strongly task-dependent — multi-step arithmetic collapses while commonsense likelihood survives

- **Protocol (SPEC 0011, exp15 + exp15_eval):** Qwen3-1.7B converted to the
  KineticLM topology at the F25 operating point (8 explicit layers at each end,
  the middle 12 collapsed to one shared core applied 12 times) = 1.167B unique
  parameters, **68% of base**. Uptrained on 98M FineWeb-Edu tokens (3.88 h on
  one RTX 5090) with teacher distillation from the frozen base plus stochastic
  anytime supervision at reduced depth. Evaluated with the SAME
  lm-evaluation-harness invocation that produced the recorded base rates
  (0-shot, 300-sample limit, batch 8, same machine).
  Artifacts: results/scale/exp15_kinetic/, results/scale/exp15_eval_results.json.
- **Perplexity recovers almost fully.** Held-out perplexity 13877 -> 20.84
  against a base measured at **17.486 on the identical tokens** — a ratio of
  **1.19x** after a budget 100-1000x smaller than published recursive-uptraining
  recipes (10-100B tokens).
- **Benchmark capability does not follow.** acc_norm ARC-Challenge 0.2900 vs
  base 0.4433 (retention 0.654); HellaSwag 0.4633 vs 0.5067 (**0.914**);
  GSM8K flexible-extract 0.0133 vs 0.4567 (**0.029**). Mean headline retention
  **0.533**, far below the pre-registered 0.90 gate.
- **The dissociation is the finding, and it is binary rather than graded.**
  Language-modelling loss and downstream capability come apart sharply under
  conversion: next-token prediction is restored to within 19% while two of the
  three benchmarks collapse. Chance-floor analysis (added after review) is
  essential to reading these numbers: ARC-Challenge and HellaSwag are four-way
  multiple choice, so a model that guesses uniformly scores 0.25, which maps to
  apparent "retention" of 0.56 and 0.49 respectively. Against that floor,
  HellaSwag at 0.4633 is genuinely retained ($z = 8.5$ above chance), whereas
  **ARC-Challenge at 0.2900 is NOT distinguishable from chance** ($z = 1.60$,
  binomial SE 0.025 at $n = 300$) — its 0.654 retention figure is an artifact
  of the floor, not evidence of partial capability. GSM8K, being open-ended and
  therefore floorless, collapses to 0.0133. The corrected picture is that
  commonsense likelihood survives conversion while multi-step reasoning and
  arithmetic are destroyed outright, which is what collapsing twelve distinct
  layers into one repeated block would predict. Perplexity on general web text
  is an unreliable progress signal for conversion work: a practitioner watching
  only perplexity would have declared this run a success.
- **Retention resolution.** At 300 samples per task the retention ratios carry
  standard errors of ±0.073 (ARC-C), ±0.077 (HellaSwag) and ±0.015 (GSM8K), so
  the mean headline figure of 0.533 should not be read to three digits, and the
  mean itself mixes two floored metrics with one unfloored one — it is reported
  because SPEC 0011 pre-registered it, not because it is the most informative
  summary.
- **The budget dial transfers.** Held-out perplexity 21.78 / 21.11 / 20.84 at
  recursion depths 3 / 6 / 12: halving inference computation costs about 1% of
  perplexity, reproducing the F24/B1 anytime property on a real pretrained model.
- **Verdict vs SPEC 0011 gates: PARTIAL** — the parameter target (68% <= 70%)
  and the graceful-degradation requirement are met; the >= 0.90 retention half
  is missed at 0.533. Scoped honestly to the budget: 98M tokens is 100-1000x
  below the recipes this method derives from, so the result bounds what one
  GPU-day buys rather than what the architecture can reach.
- **Consequence:** motivates the pre-registered H10 arms (SPEC 0014) —
  depth-curriculum weighting and LoRA relaxation with rank annealed to zero —
  which target recovery specifically rather than perplexity, and should be
  scored on benchmark retention rather than loss.
- **Tarka review (resolved 2026-08-28):** parameter accounting, perplexity
  provenance and harness parity CONFIRMED; the held-out slice was verified
  excluded from training (no contamination). Two corrections added by the
  author after review, which the review had not surfaced: the ARC-Challenge
  chance-floor analysis above, and the retention standard errors.
- **Status:** VALIDATED (single seed, design-screen scale) · verdict PARTIAL ·
  SIGN-OFF PENDING

## F27 — H9 verdict (3 seeds, real 1.5B specialists): MET. Truthful token-auction selection beats the best single specialist in CLOSED-LOOP generation on objective accuracy, reversing F23 — but it does not beat uniform logit averaging, and context-aware bidding is refuted

- **Protocol (SPEC 0013, exp16):** three Qwen2.5-1.5B-Instruct specialists
  (Math, Coder, general), verified to share one tokenizer (vocabulary 151665 /
  151643 identical across all three — a hard requirement for token-level
  aggregation). 80 prompts per seed (40 GSM8K, 40 MMLU) under one shared chat
  template, so the only difference between systems is who chooses each token.
  Fully closed-loop greedy generation; scoring is objective task accuracy
  (numeric match for GSM8K, letter match for MMLU), which removes the
  judge-LM style prior that made F23's verdict judge-relative. Seeds 42/43/44
  on GB10 under the thermal governor. results/scale/exp16/.
- **Verdict MET, 3/3 seeds, with the margin only marginally significant.**
  The second-price auction beats the best single specialist on every seed:
  0.625 vs 0.537, 0.637 vs 0.575, 0.738 vs 0.700 (margins +0.088 / +0.062 /
  +0.038; means 0.667 vs 0.604). A paired test across seeds gives
  $t = 4.33$ at $df = 2$ against a critical value of 4.30 — significant, but
  only just, and the smallest per-seed margin (+0.038) sits below the
  single-seed binomial standard error of 0.055 at $n = 80$. The direction is
  consistent across all three seeds; the effect size is not established to
  better than this. Additional seeds would be required to tighten it. This is
  the pre-registered claim that F23 missed at 121M with toy specialists, and it
  holds once the specialists are genuinely strong and the metric is objective.
- **The auction and uniform logit averaging are statistically
  indistinguishable.** ENS scores 0.637 / 0.637 / 0.762 (mean 0.679) against
  the auction's 0.667; per-seed differences are -0.012 / 0.000 / -0.025, and a
  paired test gives $t = -1.73$ at $df = 2$, far short of significance. The
  correct statement is therefore that the mechanism buys nothing measurable
  over simple averaging on this benchmark, not that averaging is superior.
  Aggregation is what wins; the choice of aggregation rule is not resolved by
  this experiment. Reported as the primary limitation of the result.
- **Both aggregators match or beat a domain-correct routing baseline**
  (mean 0.679). This arm is a routing heuristic that always selects the
  specialist owning the prompt's domain; it is NOT a theoretical upper bound,
  and calling it one would overstate the result — uniform averaging matches or
  exceeds it on two of three seeds. Per-token aggregation nonetheless
  contributes something routing cannot: on math the auction reaches 0.800,
  above the routing baseline's 0.775, because it can take a token from the
  general model when the specialist is locally unsure.
- **Context-aware bidding is refuted.** Bidding on prompt-level confidence
  rather than per-token confidence — the fix F23 itself proposed — is worse on
  every seed (-0.100 / -0.025 / -0.100 against per-token bidding). The
  hypothesis that closed-loop drift is fixed by prompt-level signal is
  therefore rejected on its first real test.
- **Domain decomposition:** the auction is strongest exactly where specialists
  differ most (math 0.800 mean, versus 0.533 on general knowledge where the
  three models are closer and confidence is a weaker routing signal).
- **Tarka review (resolved 2026-08-28):** all per-seed numbers, the
  tokenizer-identity guard and the fairness of the best-single definition
  (a hindsight maximum, which makes the comparison harder for the auction)
  CONFIRMED; the "oracle upper bound" wording was rescoped as above. The
  review's claim that the margins greatly exceed the standard error did not
  survive author recomputation and has been replaced with the paired test.
- **Status:** VALIDATED (3 seeds, Tarka-resolved with rescoping) · verdict MET
  (marginal) · SIGN-OFF PENDING

## F28 — The baseline ladder, and the thinness of subject-level headroom

**Cycle 26 · 2026-08-28 · Phase 0 (SPEC 0016, PLAN O3/O4) · GB10**

Four candidate players evaluated on one harness, one invocation, identical
limits (`lm_eval --limit 200`, bfloat16, `device_map=cuda:0`): Qwen3-1.7B,
Qwen2.5-1.5B-Instruct, Qwen2.5-Math-1.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct.

- **The parameter-matched baseline is not the best player.**
  Qwen2.5-1.5B-Instruct leads on MMLU (0.626 weighted over 57 subjects) ahead of
  Qwen3-1.7B (0.583), the Coder model (0.520) and the Math model (0.391). The
  designated baseline for the headline claim is therefore Qwen2.5-1.5B-Instruct,
  not the nominally comparable Qwen3-1.7B; using the weaker model as the bar
  would have flattered every later result by four points.
- **The subjects each player wins are disjoint, but the ceiling is thin.**
  Subject wins split 42 / 10 / 3 / 2 across base, Qwen3, Math and Coder, and the
  weaker players win by real margins where they win — the Math model takes
  abstract algebra (0.420 vs 0.380), college mathematics (0.440 vs 0.400) and
  elementary mathematics (0.520 vs 0.480); Qwen3 takes conceptual physics
  (0.670 vs 0.575) and college chemistry (0.430 vs 0.350); the Coder model takes
  global facts (0.340 vs 0.210). Yet a per-subject oracle — an aggregator
  granted advance knowledge of which player is best on each subject — reaches
  only 0.635, **+0.96 points over simply always using the best single player.**
- **Interpretation, and what it changes.** Subject-granularity routing has
  almost no room on this council, so MMLU-by-subject is the wrong arena in which
  to demonstrate the paradigm: an aggregator could route perfectly and still win
  by under a point. This does not bound token-level aggregation, whose ceiling
  is the per-example any-correct rate rather than the per-subject maximum, and
  which F27 already showed can exceed a domain-correct router. It does mean the
  next measurement must establish the per-example ceiling before any effort goes
  into building players, and the arena must be one where the players actually
  disagree per example.
- **Integrity flag — GSM8K is not measured by this run and is excluded.**
  Strict-match scores 0.000 for all four models, which is a format artifact
  rather than a capability measurement: the task's regex expects the `#### `
  answer convention that instruction-tuned models do not emit. The
  flexible-extract figures that remain (0.455 / 0.095 / 0.290 / 0.340) are
  mutually inconsistent with the published capabilities of these checkpoints —
  Qwen2.5-Math-1.5B-Instruct in particular is reported far higher — which is the
  signature of the generation-budget truncation already diagnosed in the exp16
  smoke. No mathematics claim is drawn from this run, and generative tasks are
  deferred until the harness applies chat templates and a per-domain generation
  budget. The loglikelihood tasks are unaffected, since they involve no
  generation.
- **Status:** VALIDATED (single seed per model; loglikelihood tasks only) ·
  informs O3 (baseline identity) and O4 (domain selection) · Tarka PENDING

## F29 — The influence game does not beat averaging at answer level, and confidence is the reason

**Cycle 26 · 2026-08-28 · SPEC 0016, ADR 0008 · exp18/exp19/exp20 · GB10**

Four players (Qwen3-1.7B, Qwen2.5-1.5B/Math/Coder-Instruct) were scored once on
8,301 questions across 61 tasks with per-example option loglikelihoods logged,
after which every aggregation rule was computed offline on the same stored
scores. Each task's answer-label convention was recovered from the harness's own
per-record scoring rather than assumed; all 61 tasks reconciled exactly, which
matters because two conventions silently disagreed (WinoGrande labels from one,
and ARC carries `target` and `answerKey` under different conventions) and the
uncalibrated reading scored players at 0.13 where the truth was 0.63.

- **Solving the equilibrium is indistinguishable from averaging.** Uniform
  averaging scores 0.6304. The best of 45 grid points over the influence
  rationality and magnet strength scores 0.6311 at $\beta = 0.25$, a margin of
  0.0007 against a standard error of 0.0053 — and that margin is a hindsight
  maximum over the whole grid, so the true expected margin is smaller still. Per
  task the split is 16 better, 12 worse, 33 unchanged.
- **Raising the influence rationality is monotonically harmful.** Accuracy falls
  from 0.6304 at $\beta = 0$ to 0.5486 at $\beta = 8$, an eight-point loss. The
  magnet mitigates but never reverses it: at $\tau = 0.5$ the same setting
  recovers only to 0.6119. Routing-like behaviour, which the construction
  approaches as $\beta$ grows, is therefore actively worse than blending here.
- **The diagnosis is the payoff, not the solver.** Influence follows
  $\langle y, \ell_i \rangle$, which rewards a player for agreeing with the
  emerging consensus — that is confidence, and confidence is not competence. The
  Math specialist is right on 40% of these questions and no less emphatic for it,
  so the game hands it weight exactly when it is emphatic. Majority vote, which
  is the same error in cruder form, scores 0.5905, *below* the best single player.
- **Three interventions, one of which works and none of which is the game.**
  Reversing the payoff so dissent earns influence: 0.6280–0.6316, no gain.
  Influence from per-example sharpness instead of agreement: 0.6289–0.6318, no
  gain. Weighting by each player's reliability measured on a held-out half:
  0.6415, **+1.14 points**, the only arm that moves. Adding the solved game on
  top of those competence weights reaches 0.6422, a further 0.07 points — within
  noise of the weights alone, so the aggregation is doing the work and the game
  is not.
- **Context-dependent competence was tried and does not help either.** A gate
  predicting per-question, per-player correctness from label-free descriptors of
  the score field (entropy, top-two margin, divergence from the field, field
  entropy) reaches 0.6345 averaged over three fits — *below* the constant
  competence weights it was meant to improve on. Whether a player is right on a
  given question is not legible from the shape of the scores.
- **Twenty points sit unclaimed.** Some player answers correctly on 83.1% of the
  questions while the best single player manages 62.6% and the best aggregation
  64.2%. **[Superseded by F32: this ceiling is not extractable. Gating on the
  correct player reaching even modest confidence reduces it to 0.658, so the
  real headroom over the best single player is about three points and the best
  aggregation already sits within 1.6 of it.]**
- **Interpretation.** In this arena the paradigm's central claim fails: solving
  the game is not better than averaging, and the mechanism by which it was
  supposed to win — reallocating influence per position — is the mechanism that
  hurts. The result is confined to answer-level aggregation, where every player
  sees the same prompt once and no consensus is carried forward; it does not
  test the sequential setting in which the consensus prefix changes what each
  player conditions on, which is the only part of the construction an ensemble
  cannot replicate and where F27 measured a gain. That setting is now the
  decisive one, and the requirement it inherits from this finding is that
  influence must be driven by something other than confidence.
- **Status:** VALIDATED (8,301 examples, 3 gate seeds, held-out fitting) ·
  verdict NOT MET for answer-level aggregation · Tarka PENDING

## F30 — Eleven aggregation rules, none better than the mean, and the reason

**Cycle 26 · 2026-08-28 · exp21/exp22 · same 8,301 questions as F29**

F29 attributed the influence game's failure to a payoff that rewards confidence
rather than competence. Two families of repair follow from that diagnosis, and
both were tried.

- **Mechanism design does not repair it.** Rules that refuse to let a player
  vote for its own proposal — the second-price intuition of F6 applied to
  answers — do not beat the mean. Valuing each player's proposal by what it is
  worth to everyone else scores 0.6257 against the mean's 0.6304 ($z = -2.63$ on
  the 320 questions where the two disagree). Dropping each option's keenest
  supporter before averaging is a statistical tie (0.6302, $z = -0.23$). The
  median scores 0.6144 ($z = -5.14$) and Borda count 0.5922 ($z = -10.65$), so
  discarding magnitude and keeping only preference is worse still.
- **Calibration does not repair it either, though it does calibrate.** Fitting
  one temperature per player on a held-out half cuts expected calibration error
  from 0.151 to 0.037 for the worst-calibrated player and improves all four, so
  the players' stated confidences afterwards mean close to what they claim. The
  aggregate does not follow: calibrated averaging scores 0.6277 against raw
  averaging's 0.6301, and the influence game still degrades monotonically in
  $\beta$ (0.6270 at 0.25, 0.5960 at 8). Miscalibration was therefore not the
  cause.
- **The mechanism, stated plainly.** Every rule tested is a reweighting of one
  fixed body of evidence: four distributions over the same options, produced
  from the same prompt. The influence game concentrates weight on a subset of
  players, which *discards* part of that evidence, and discarding evidence from
  a council whose members are jointly right 83% of the time while individually
  right 63% is the wrong direction. Averaging is the rule that keeps all of it.
  No reweighting can recover information that reweighting removed, which is why
  eleven rules across three experiments — six voting and market rules, three
  confidence signals, competence weighting, and calibration — all land at or
  below the mean.
- **What this leaves open, and it is the whole paradigm.** New information, not
  new weights, is the only route to the unclaimed twenty points. The sequential
  setting supplies it and answer-level aggregation cannot: when players generate,
  each produces a reasoning chain the others never saw, so scoring a peer's
  chain is genuine evidence rather than a re-reading of one's own. Cross-
  examination over generated candidates is the mechanism the next arm tests, and
  the answer-level version of it (players pricing each other's proposed options)
  correctly fails here precisely because nothing new is on the table.
- **Status:** VALIDATED (8,301 examples, held-out fitting for both repairs) ·
  verdict NOT MET · closes the answer-level arena · Tarka PENDING

## F31 — Training players to be calibrated would make the game worse, not better

**Cycle 26 · 2026-08-28 · exp24 · simulation anchored to the measured council**

F29 attributed the influence game's failure to confidence not tracking
competence. The natural remedy is to train players so that it does, which is
expensive, so the premise was tested first by simulating a council with two
dials: how tightly each player's confidence tracks its chance of being right on
that question, and how often players that are wrong are wrong in the *same* way.
Sixteen regimes, five seeds each, 8,000 questions per seed, with the players'
standing competences set to the four measured in F28.

- **Both dials were measured on the real council before being swept.** Confidence
  correlates with correctness at 0.26 to 0.44 across the four players, placing
  the council near a coupling of 0.35. Errors are strongly shared: when two
  players are both wrong they choose the same wrong option 56.6% of the time
  against the 34.2% expected if errors were independent, a ratio of 1.66, rising
  to 0.68 between the base and coder variants.
- **The remedy is refuted, and the sign is the opposite of the hypothesis.** The
  game's margin over averaging falls monotonically as coupling rises: $+0.041$ at
  coupling 0, $+0.010$ at 0.35, $-0.006$ at 0.70 and $-0.010$ at 1.0, against a
  cell standard error of 0.0025. Solving the game helps only when confidence is
  nearly uninformative and hurts once confidence becomes informative. Training
  players to report competence faithfully would therefore move the council
  *away* from the regime in which the game pays.
- **The reason is the one F30 identified, seen from the other side.** When
  confidence is informative, averaging is the larger beneficiary — the mean rises
  from 0.709 to 0.757 at fixed error correlation as coupling goes from 0.35 to
  1.0 — because informative confidences make every player's contribution worth
  more, and averaging keeps all of them. Concentrating influence discards
  corroboration, which is only a good trade when the discarded evidence was
  worthless to begin with.
- **Error correlation lowers every rule without changing which one wins.** Raising
  it from 0 to 0.8 costs the mean between 12 and 17 points at every coupling, but
  the sign of the game-minus-mean margin is set by coupling alone. Correlated
  errors are therefore a problem for the council's ceiling, not an explanation of
  the game's failure.
- **Limitation, and it is material.** The simulation reproduces the *sign* of the
  measured comparison at the measured regime — a tie, $+0.0036$ simulated against
  $+0.0007$ observed — but not the magnitude of the degradation as influence
  concentrates: eight points on the real council between $\beta = 0$ and
  $\beta = 8$, against half a point in simulation. Something about real players'
  score geometry punishes concentration far harder than this model does, so the
  quantitative predictions here are not load-bearing and only the direction of
  the coupling effect is claimed.
- **Consequence.** The training-for-calibration branch is closed before any GPU
  was spent on it, which is what the probe existed to decide. ADR 0009's
  direction is unaffected and, if anything, reinforced: cross-examination adds
  candidates rather than concentrating influence over fixed ones, and every
  measurement in this cycle points the same way — a council is helped by having
  more evidence weighed and harmed by having evidence discarded.
- **Status:** VALIDATED (16 regimes, 5 seeds) · closes the calibration-training
  branch · simulation, not measurement, and scoped accordingly · Tarka PENDING

## F32 — Correction: the twenty-point oracle gap is mostly not extractable

**Cycle 26 · 2026-08-28 · audit of F29–F31 · same 8,301 questions**

F29, F30 and F31 each reported that some player answers correctly on 82.6% of
the questions against a best single player's 62.5%, and drew the inference that
twenty points of complementary knowledge sat unclaimed and the whole difficulty
was extraction. That inference does not survive an audit of the statistic and is
withdrawn.

A mechanism cannot use a correct answer it has no way of identifying. Requiring
the correct player to be at least somewhat confident collapses the ceiling: at a
confidence gate of 0.5 — only twice chance on a four-option question — the oracle
falls from 0.826 to 0.658, which is 3.3 points above the best single player
rather than 20. At a gate of 0.7 it falls to 0.503, below the best single player
outright. Most of the apparent headroom consists of questions where the one
correct player is correct at low confidence, which is indistinguishable from
being correct by chance and carries no signal any rule could key on.

- **The realistic ceiling is about 0.66, and aggregation has nearly reached it.**
  The best measured rule, competence-weighted averaging, scores 0.6415 against a
  0.658 confidence-gated ceiling. The programme's aggregation rules were
  therefore operating within roughly 1.6 points of what this council's answer
  distributions can support, not 20 points short of it.
- **This explains the earlier results rather than contradicting them.** Eleven
  rules landed within noise of averaging because there was little left to
  extract, and F31's finding that concentration only helps when confidence is
  uninformative is the same fact seen from the other side. The measurements were
  right; the ceiling they were compared against was wrong.
- **Error correlation is corroborated independently.** Were the players' errors
  independent, the oracle would be 0.955 given the measured per-player
  accuracies; the observed 0.826 falls well short, consistent with the 1.66-times-
  chance error agreement measured in F31.
- **Consequence for the direction, which is sharpened rather than reversed.**
  The case for ADR 0009 no longer rests on a large unclaimed reserve. It rests
  on the opposite: the fixed-evidence ceiling is close to reached, so improving
  the aggregation rule over these four distributions cannot produce a
  competitive system, and the only remaining routes are players that are right
  more often or a mechanism that puts new candidates on the table.
  Cross-examination over generated solutions is the latter, which is why it
  remains the live arm.
- **Artifacts corrected in this cycle:** the paper's aggregation section and
  Table~\ref{tab:ladder}, the site's F29 and F30 entries, and the interpretation
  paragraphs of F29, F30 and F31 above, each of which asserted the twenty-point
  reading.
- **Status:** VALIDATED · supersedes the headroom claim in F29–F31 · Tarka PENDING

## F33 — The arena was homogeneous, and that is why nothing beat averaging

**Cycle 26 · 2026-08-28 · corrected GSM8K measurement · GB10**

F28 excluded GSM8K because strict match scored zero for all four models, which
measured the answer convention an instruction-tuned model emits rather than its
arithmetic. Re-measuring with each model's chat template applied and a 512-token
generation budget settles what the exclusion was hiding.

- **The correction is large and selective.** Qwen2.5-Math-1.5B-Instruct goes from
  0.290 to **0.795**, Qwen2.5-1.5B-Instruct from 0.095 to 0.595, and
  Qwen2.5-Coder-1.5B-Instruct from 0.340 to 0.510, while Qwen3-1.7B is unmoved at
  0.455 to 0.450. The chat template is decisive for the Qwen2.5-Instruct family
  and irrelevant for Qwen3, so a single harness configuration was silently
  penalising three of four players by between 17 and 51 points.
- **The "weak" player is the council's strongest specialist.** The mathematics
  variant leads GSM8K by 20 points over the generalist and 35 over Qwen3, while
  scoring 0.391 on MMLU where it is last. Its competence is real, large and
  entirely invisible in multiple choice.
- **The arena that produced F29 and F30 was nearly homogeneous, and GSM8K was the
  one task excluded from it.** Across the 61 loglikelihood tasks the four players
  differ mainly in overall strength rather than in what they know, which is why a
  per-subject oracle bought only 0.96 points there. On a mixed arena of equal
  parts mathematics and knowledge the same four players give a best single score
  of 0.611 against a perfect domain router's 0.711 — **ten points of routable
  headroom, an order of magnitude more than the arena where every aggregation
  rule was tested.**
- **Interpretation, and what it does and does not overturn.** F29 and F30 remain
  correct about what they measured: over those distributions no rule beats
  averaging, and F32's audit shows little was left to extract. What they cannot
  support is the general claim that aggregation is worthless for this council.
  The negative was obtained where the players are close to interchangeable, and
  the measurement that would have shown otherwise was the one the broken harness
  forced out of the set. The chain is worth stating plainly because each link was
  individually correct: a genuine integrity flag removed the task, removing the
  task homogenised the arena, and a homogeneous arena cannot separate aggregation
  rules.
- **Consequence.** Selection has real headroom on heterogeneous work and close to
  none on homogeneous work, so any claim about the paradigm must state which
  regime it was measured in. The cross-examination arm uses an equal mix of
  mathematics and knowledge for exactly this reason, and its comparison against
  best-single now has a meaningful 10-point ceiling rather than a 1-point one.
- **Status:** VALIDATED (single seed per model, 200 questions, limit-bounded) ·
  scopes F29–F31 without overturning them · Tarka PENDING

## F34 — The bar is a domain router, not the best single player

**Cycle 27 · 2026-08-28 · exp23_router_baseline over the exp23 records**

F33 reported ten points of routable headroom on a mixed arena and treated it as
an opportunity for the council. It is better read as an obligation. The headroom
is reachable by the cheapest mechanism that exploits the same structure: classify
the prompt's domain and send it to whichever player the ladder already showed to
be best there. On a mix of mathematics word problems and multiple-choice
knowledge that classification is decidable from format alone, so the router costs
one forward pass, needs no aggregation, no solve and no second model, and is what
a competent engineer would build before considering a council at all.

Measured on the first cross-examination run's 120 questions, with the per-domain
champions fixed in advance from the ladder rather than derived from the run being
scored:

- **Domain router 0.6667**, against a best single player of 0.5917 — the router
  takes 7.5 points for free.
- Every council rule measured falls below it: self-preference 0.6250,
  the equilibrium over candidates 0.5000, leave-one-out pricing and
  cross-examination 0.4917. The best of them is more than four points short of a
  method with no machinery in it.
- The oracle over players stands at 0.8333, so the council's members do hold the
  answers; nothing measured extracts them better than asking which subject the
  question is about.

**What this changes.** Every comparison in this programme has been reported
against the best single player, and that bar is too low to be informative,
because it is not what anyone would actually deploy. A council earns its
complexity only by beating the router, and the margin it needs is the router's
7.5 points plus whatever it adds. Results reported against best-single are
retained but are no longer the headline comparison; the router is.

**Scope.** The general half of this particular run was compromised by a prompt
that forbade explanation, which is why the cross-examination figures here are not
a fair test of that mechanism and are being re-measured. The router figure is
unaffected by that defect, since it uses only each player's own answer, and the
margin over every council rule is far too large to be an artefact of it.

**Status:** VALIDATED (single seed; re-measured across three seeds in the
corrected run) · supersedes the headroom framing in F33 · Tarka PENDING

## F35 — Conversion damage is not recoverable at this budget, and reasoning is what is lost

**Cycle 27 · 2026-08-28 · exp15 H10 arms · RTX 5090 · ~4.2 h and 98M tokens per arm**

H10 asked whether the conversion damage measured in F25 could be recovered on a
low budget by either of two TRIZ-derived interventions: a depth curriculum that
introduces recursion gradually, and a rank-annealed depth-LoRA that relaxes
weight tying during uptraining and merges the adapter away at the end. Both were
run to completion on Qwen3-1.7B converted to 1.167B unique parameters at
recursion depth 12, evaluated against the base model under an identical harness
invocation on the same machine.

- **The two interventions are indistinguishable.** Mean headline retention 0.5316
  for the depth curriculum against 0.5314 for the rank-annealed LoRA, a
  difference of 0.0002. Two structurally different repairs arriving at the same
  number is stronger evidence that the budget binds than either arm alone.
- **The headline retention figure overstates what survives, and is corrected
  here.** Retention measured as a ratio of raw accuracies credits the converted
  model for the chance floor: ARC-Challenge is four-way, so a model answering at
  random scores 0.25 and appears to retain 56% of a 0.443 baseline while knowing
  nothing. Measured above chance, the depth curriculum retains 0.259 of
  ARC-Challenge, 0.766 of HellaSwag and 0.036 of GSM8K, for a mean of **0.354
  rather than 0.532**. The corrected figure is the one that should be quoted.
- **Reasoning is destroyed while fluency survives.** GSM8K falls from 0.457 to
  0.017 and 0.010 across the two arms — between two and four percent of the base
  model's arithmetic — while HellaSwag retains over three quarters and held-out
  perplexity sits at 1.19 to 1.20 times base. A model 20% worse by perplexity is
  27 times worse at grade-school arithmetic.
- **This is the perplexity–capability dissociation of F26 at its most extreme,**
  and it is the reason a perplexity-only report of this experiment would have
  read as a near-success. Anyone selecting an operating point from the damage
  curve on perplexity alone, as F25 did by necessity before capability
  measurements existed, would have chosen a configuration whose reasoning was
  already gone.
- **Interpretation.** Weight-tied recursive conversion of a pretrained model
  preserves the statistics of text and loses the computation that multi-step
  reasoning needs, and 98M tokens of uptraining under either repair does not
  bring it back. The verdict on H10 is NOT MET on both arms, and the broader
  reading is that the conversion route to a competitive small model is not viable
  at budgets available here — published recursive-uptraining recipes use 10 to
  100 billion tokens, two to three orders of magnitude beyond what was spent.
- **Consequence for the programme.** This strengthens the case for building on
  strong models as they are rather than converting them, which is the council
  direction, and it removes conversion from the candidate set the autoresearch
  loop ranks. It does not rescue the council: F34's domain router remains the bar
  and nothing has yet beaten it.
- **Provenance note.** The RTX 5090's system clock is skewed by roughly three
  hours against GB10's, so timestamps in these logs are not comparable across
  machines; durations and exit codes are self-consistent and are what the record
  relies on.
- **Status:** VALIDATED (both arms to completion, single seed each) · verdict
  NOT MET · Tarka PENDING

## F36 — Cross-examination ties the domain router and beats nothing

**Cycle 27 · 2026-08-28 · exp23, three seeds, 360 questions · GB10**

ADR 0009 moved the paradigm from answer-level aggregation to cross-examination
over generated candidates, on the argument that a peer's chain of reasoning is
evidence its reader did not have and is therefore the one channel by which a
council can learn something at inference time. Each of four players wrote a full
solution to every prompt and priced every peer's; five selection rules operated
on the resulting valuations. The arena was an equal mix of grade-school
mathematics and multiple-choice knowledge, both halves eliciting reasoning after
a first run showed the inherited multiple-choice prompt forbade explanation and
so left nothing to cross-examine.

Measured against the domain router that F34 established as the bar, paired over
the 360 questions:

- **The equilibrium over candidates ties the router exactly**, 32 wins against 32
  losses, $z = 0.00$, both at 0.5611.
- **Cross-examination is a coin flip against it**, 31 against 32, $z = -0.13$,
  at 0.5583. Leave-one-out pricing scores 0.5444 ($z = -0.77$) and
  self-preference 0.5250 ($z = -1.50$). Self-consistency by majority vote reaches
  0.5722, nominally 1.1 points above the router but 0.4 standard errors on a
  standard error of 2.6 points.
- **The best single player scores 0.5472**, so the rules do clear that bar. They
  clear it by less than the router does, which is why the router and not the
  single player is the comparison that matters.
- **The verbosity confound is gone.** Winning candidates average 731 characters
  against a field average of 810, so the rules are no longer selecting the
  longest answer; the earlier catastrophic reading was the prompt artefact it
  appeared to be, and correcting it moved cross-examination from ten points below
  the best single player to slightly above it.

**The one structure in the result.** Split by domain, the council rules beat the
router where no specialist dominates and lose where one does: on general
knowledge the equilibrium scores 0.333 against the router's 0.300, while on
mathematics the router's 0.822 beats the equilibrium's 0.789. Routing wins when
there is a clear champion and aggregation wins when there is not, and over an
equal mix the two effects cancel almost exactly. A hybrid that routes on
mathematics and solves on general knowledge reaches 0.5778, but which rule serves
which domain was chosen after seeing these numbers, so that figure is a hindsight
maximum ($z = 0.83$ against the router) and is recorded as a hypothesis for a
held-out test rather than as a result.

**Interpretation.** Three routes have now been tested and none has beaten a
trivial baseline. Answer-level aggregation could not beat uniform averaging and
was operating near its achievable ceiling (F29, F30, F32). Conversion to the
recursive topology destroys reasoning at any budget available here (F35).
Cross-examination over generated candidates ties a domain router built from a
twenty-line classifier (this finding). The paradigm as constructed does not beat
what a competent engineer would build, and saying otherwise would require
comparing against the best single player, which is the wrong bar.

**Status:** VALIDATED (3 seeds, 360 questions, paired against the router) ·
verdict NOT MET · closes ADR 0009's Phase 1b · Tarka PENDING

## F37 — The solve is cheap; the council is not, and it buys nothing

**Cycle 27 · 2026-08-28 · exp26 · GB10 · 12 prompts, 1,152 generated tokens**

The PRD's cost argument holds that after one forward pass per player, the
equilibrium solve is softmax and dot products over the vocabulary and therefore
adds negligible time, so a council runs at ensemble cost. ADR 0009 flagged that
cross-examination breaks the argument by being quadratic in council size. The
autoresearch agent ranked measuring this first once the quality questions closed,
and it is the only hypothesis in the belief state that this cycle confirmed.

- **The solve is genuinely cheap.** Of 68.15 s of wall-clock for token-level
  council decoding over three players, forward passes account for 65.94 s
  (96.8%) and the equilibrium solve for 2.16 s (3.2%), at 1.875 ms per token
  against 57.240 ms for the forwards. The claim that the solve adds negligible
  time against the passes themselves is correct as stated.
- **The council costs what an ensemble costs, which is the whole point and also
  the problem.** Three players decode at 5.679 s per request against a single
  model's 1.898 s, a factor of 2.99 — linear in council size, exactly as
  designed. Peak memory 9.4 GB, well inside GB10's unified memory, so residency
  is not the constraint.
- **The comparison that matters is against the router, not the single model.**
  F34's domain router pays one forward pass, the same as a single model, and F36
  showed nothing beats it on quality. So the council's true position is
  **three times the latency of the thing it fails to beat**. A negative result
  that quantifies its own overhead is worth more than one reporting a tie, and
  this is the number a practitioner would want first.
- **Cross-examination is worse than linear and was not run.** It pays one full
  decode per player to generate, then one forward pass over prompt and candidate
  for every reader-writer pair — nine passes for three players, sixteen for four.
  Its generation cost alone is the 3x above, before any scoring.
- **A structural finding surfaced by the guard.** The four-player council does
  not share a tokenizer: Qwen3-1.7B carries 151,669 tokens against Qwen2.5's
  151,665, differing by four control tokens. Token-level aggregation across them
  is therefore undefined, and the measurement above uses the three Qwen2.5
  players that do agree. No earlier result is invalidated, since the answer-level
  experiments aggregated over options and the cross-examination experiments
  scored text, neither of which needs a shared vocabulary — but ADR 0008's
  token-level decoder has never been runnable on the nominal council, and the
  PRD's requirement that a shared tokenizer be enforced at load time rather than
  assumed is vindicated by having caught it here rather than in a fluent-nonsense
  output.
- **Status:** VALIDATED · the cost hypothesis is MET and is the one positive
  result of this cycle · Tarka PENDING

## F38 — The hybrid does not survive held-out testing, and the action set is exhausted

**Cycle 27 · 2026-08-28 · offline over exp23's 360 records**

F36 noted that council rules beat the router where no specialist dominates and
lose where one does, and that a hybrid routing mathematics while solving general
knowledge reached 0.5778 against the router's 0.5611. That figure was flagged as
a hindsight maximum, since which rule served which domain was chosen after seeing
the numbers. Tested properly by fitting the per-domain rule choice on one seed
and evaluating on the other two, across all three folds:

- Fitting on seed 42 gives mathematics to the router and general knowledge to
  self-preference, and scores 0.5583 held-out against the router's 0.5792, a
  margin of $-0.0208$ ($z = -0.73$).
- Fitting on seed 43 gives both domains to the equilibrium, scoring 0.5333
  against 0.5500, margin $-0.0167$ ($z = -0.60$).
- Fitting on seed 44 gives mathematics to the router and general knowledge to the
  equilibrium, scoring 0.5625 against 0.5542, margin $+0.0083$ ($z = 0.37$).

The mean margin is $-0.0097$: the hybrid is no better than the router, and the
0.0167 gain reported in F36 was entirely hindsight. The rule selected differs in
every fold, which is the more telling observation — if there were a stable
per-domain signal, three folds drawn from the same distribution would agree about
what it was.

**The state of the action set.** With this closed, every candidate the
autoresearch agent holds has near-zero or positive expected free energy:
re-measuring latency 0.089 nats of information, a second model family 0.168, the
offline simulation 0.022, cross-examination 0.060, distillation 0.033 against a
cost of 0.333. The two hypotheses that would justify building anything sit at
0.042. The agent is not stuck in the sense of being unable to choose; it is
telling us that nothing in its action set is worth running, which is the correct
output when the routes on offer have all been closed and is a different situation
from an unresolved question.

That is a limit of the method worth stating plainly. Expected Free Energy ranks
actions; it does not invent them. When every action scores near zero the
constraint has moved from selection to generation, and generating genuinely new
candidates is an inventive step the loop cannot perform on its own.

**Status:** VALIDATED (3 held-out folds) · closes the hybrid lead · Tarka PENDING

## F39 — The anchored answer vote beats the router held-out (offline; confirmation pre-registered)

**Cycle 28 · 2026-08-28 · exp27 over exp23's stored candidates · zero GPU · TRIZ-generated**

With the action set exhausted (F38), the TRIZ engine was applied to the standing
contradiction: per-question collective evidence improves adaptability but
dilutes the dominant specialist exactly where the router wins. The matrix cells
35/27 and 35/28 recommend inversion, counterbalance, an intermediary and
preliminary action, which compose into one mechanism: make the router the
*reference policy* of the game rather than its competitor. Players' generated
answers are collapsed into equivalence classes — the intermediary that
neutralises both the tokenizer mismatch (F37) and the verbosity confound — and a
class's score is its vote count plus a magnetic bonus tau on the router's class.
The council moves the answer only when its net margin exceeds tau; at large tau
the mechanism is exactly the router, so its floor is the bar by construction, a
property no previously tested rule had. This is the kinetic core aimed at the
incumbent: the MMD magnet in policy space with the baseline as the magnet, the
QRE argmax over discrete classes reducing to a thresholded vote, and weights
fixed in advance from the ladder as truthful bids.

- **Every cell of the grid beats the router in-sample.** Fourteen
  (weighting, tau) settings score 0.6222 to 0.6333 against the router's 0.5611
  over the pooled 360 questions. Parameter choice barely moves the result, which
  distinguishes this from F38's hybrid, whose selected rule flipped every fold.
- **All three held-out folds are positive.** Fitting (weighting, tau) on one
  seed and evaluating on the other two: +0.0833 (z = 4.26, 21W/1L), +0.0375
  (z = 1.73, 18W/9L), +0.0583 (z = 3.74, 14W/0L). Mean held-out margin +0.0597.
- **It wins on both domains rather than trading one for the other.** Held-out,
  mathematics 0.833 against the router's 0.800 in the strongest fold and never
  materially below; general knowledge 0.492 against 0.358. The anchor protects
  the specialist where it dominates while the vote repairs the domain where no
  one does — which is exactly the division of labour F36 observed and F38 failed
  to exploit by rule selection.
- **Why this evades the F30 impossibility.** The earlier result held that
  reweighting one fixed body of evidence cannot beat retaining all of it. The
  vote is not a reweighting of distributions: extracted answers from independent
  reasoning chains are new evidence (self-consistency), and the anchor injects a
  second signal — the ladder's measurement of who is reliable where — that no
  purely per-question rule contained.
- **Caution, and the reason for SPEC 0017.** The 360 questions used here are the
  same data that produced F34 and F36, so the arena, though not the parameters,
  has been seen. A confirmation at pre-registered uniform/tau = 1.0 on fresh
  seeds 45-47 was registered before any confirmation data existed and is running
  now; the claim stands or falls on it.
- **Status:** SUPERSEDED BY F40 — the Tarka review found the comparison
  structurally unfair before the confirmation reported; the margin decomposes
  into extraction redundancy, not anchoring

## F40 — The anchored vote's margin is redundancy, not anchoring

**Cycle 28 · 2026-08-28 · Tarka review of F39, author recount, fair-bar sweep · zero GPU**

The Tarka review of the offline analysis, run while the confirmation was still
generating, found the comparison structurally unfair: the router row abstains
whenever its single champion emits an unparseable answer — 58 of 360 questions,
16% — while the mechanism has four extraction attempts. The author recount at
the pre-registered cell confirms the defect and corrects one reviewer figure:
24 of the mechanism's 26 paired wins are abstention rescues, and on questions
both systems answered the record is 2W/1L ($z = 0.58$), noise rather than the
net-negative override the review reported from fold-fitted cells.

- **Against the fair bar the margin evaporates.** A router with a majority-vote
  fallback on extraction failure — the one-line repair a competent engineer would
  ship — scores 0.6278. No cell of the fourteen-point grid separates from it:
  the best is 3W/1L ($z = 1.00$), and at $\tau \geq 2$ the mechanism reproduces
  the fallback router exactly, 0W/0L. The anchored vote contains the fair bar as
  a limit and adds nothing measurable to it.
- **The magnet threshold does exactly what F29 predicted, which is why it
  neither helps nor hurts.** At $\tau = 1$ the council overrides an answering
  champion three times in 360 questions. The anchor makes overriding rare enough
  to be harmless and thereby rare enough to be worthless; the earlier finding
  that per-question signals cannot identify when a council should overrule a
  competent incumbent holds at the answer level too.
- **What genuinely survives, stated at its honest size.** The council system —
  ladder-prior routing plus redundancy against single-model extraction failure —
  beats the strongest single baseline model 0.6278 to 0.5472 on the mixed arena,
  eight points, at a measured 1.25 expected generations per request (worst case
  four), since the fallback generates further candidates in ladder order only on
  the 16.1% of champion extraction failures. Both ingredients are this
  programme's constructions and the second is the $\tau \to \infty$ limit of the
  anchored vote. What no tested mechanism contributes is extraction of knowledge
  complementarity beyond redundancy: the per-example oracle stands twelve points
  above the fair bar and remains unclaimed, consistent with F30 and F32 at
  generation level.
- **Process note.** SPEC 0017 was amended before any confirmation data was
  scored: primary comparison moved to the fallback router, threshold raised to
  the Bonferroni $z \geq 2.807$ for the roughly fifteen mechanisms examined on
  the development arena, and every margin now reported with its
  abstention-versus-override decomposition. The amended scorer reports
  `success: False` on the development data, which is the honest reading of F39.
  The confirmation's fresh-seed generation continues unchanged and will test
  whether the decomposition replicates.
- **Status:** VALIDATED (author-verified recount and sweep) · supersedes F39's
  claim · Tarka RESOLVED with one reviewer figure corrected

## F41 — Pre-registered confirmation: anchoring refuted, the system beats the baseline

**Cycle 28 · 2026-08-29 · SPEC 0017 with Amendment 1 · seeds 45-47, 360 fresh questions · GB10**

The confirmation ran on questions drawn after the protocol was frozen and scored
by the amended scorer without further choices. Both halves of the amended
protocol returned a clear answer, and they point in opposite directions.

- **The anchored vote fails its criterion, as amended.** Mean margin against the
  fallback router $-0.0028$; pooled paired 1 win to 2 losses, $z = -0.58$,
  against a required $z \geq 2.807$. Per seed the margins are $-0.0083$,
  $0.0000$ and $0.0000$: on two of three seeds the mechanism and the fair bar
  select identically on every question. Across 360 fresh questions the council
  overrode an answering champion exactly once. F39's offline margin was
  extraction redundancy, F40 diagnosed it correctly, and the confirmation
  settles it on data nothing was fitted to. The magnetic anchor at the token
  level (F29) and at the answer level (here) is refuted by the same mechanism:
  no per-question signal identifies when a council should overrule a competent
  incumbent.
- **The system beats the baseline model, confirmed and replicated.** Against the
  single model the ladder designated in advance, the deployed system — route on
  per-domain priors, fall back to a majority vote of the council when the
  champion's answer cannot be parsed — scores **0.6194 against 0.5361**, a margin
  of **+8.33 points**, paired **38 wins to 8 losses, $z = 4.42$**, clearing the
  Bonferroni-corrected threshold of 2.807 that Amendment 1 imposed. Every seed is
  positive: $+0.0750$, $+0.0917$, $+0.0416$. The development arena gave
  $+8.06$ points, so the effect replicates at its measured size rather than
  shrinking, which is the usual fate of a margin selected on its own data.
- **What the kinetic core contributed, stated exactly.** The winning system is
  the $\tau \to \infty$ limit of the magnetically anchored vote: the construction
  supplied the form — an incumbent policy with a council held at a threshold —
  and the confirmation says the useful setting of that threshold is the one that
  never overrides. Magnetic mirror descent contributed the anchor's shape and the
  proof that the limit contains the baseline, which is why the system cannot lose
  to the model it routes on; it did not contribute an operating point strictly
  inside the interval. Claiming more than that would misdescribe the measurement.
- **Cost.** 1.258 expected generations per request on the confirmation set (1.25 on the
  development set), since the fallback generates further candidates only on the
  champion extraction failures.
- **Status:** VALIDATED (pre-registered, 3 fresh seeds, 360 questions,
  Bonferroni-corrected) · anchoring NOT MET · **system-versus-baseline MET** ·
  Tarka RESOLVED (its correction is what made this comparison fair)

## F42 — What the win is made of, and the condition under which it exists

**Cycle 28 · 2026-08-29 · decomposition of F41 plus the second-family interim · zero GPU**

F41's confirmed margin was reported as one number. Decomposing it on the same
360 confirmation questions, and auditing it against a stronger extractor,
establishes what produces it and when it would not.

- **Routing supplies most of it, redundancy little.** Against the baseline
  single model at 0.5361: adding the council's redundancy alone reaches 0.5500
  ($+1.39$ points), routing on ladder priors alone reaches 0.6000 ($+6.39$), and
  the deployed combination reaches 0.6194 ($+8.33$). Isolating routing by giving
  the baseline the same redundancy leaves the system ahead 33 wins to 8,
  $z = 3.90$. The result is a routing result with a redundancy top-up, not the
  reverse.
- **It is not an artefact of weak answer parsing.** A deliberately more
  forgiving extractor — accepting a trailing bare letter or number without the
  boxed convention — cuts champion abstention from 8.6% to 3.3% and raises both
  systems. The margin holds at $+7.78$ points, 36 wins to 8, $z = 4.22$. The
  self-audit was run because a gain resting on the extractor failing would be an
  engineering artefact rather than a result; it does not.
- **The precondition is measurable in advance, and the second council lacks it.**
  Routing pays only when different players are best on different domains. On the
  Qwen council the mathematics specialist leads mathematics at 0.817 while the
  general model leads knowledge at 0.383, so the router has something to choose.
  On the second council — SmolLM2-1.7B, deepseek-math-7B, Falcon3-3B and
  Falcon3-1B, four families and four tokenizers — calibration selects
  Falcon3-3B as champion of **both** domains, and it is also the best single
  model. The first evaluation seed accordingly gives routing exactly nothing:
  system 0.6417 against best single 0.6417, 0 wins and 0 losses. Champion
  abstention is 1.7% against the Qwen council's 8.6%, so the redundancy term has
  almost nothing to work on either.
- **Interpretation.** The eight-point win is real, pre-registered and replicated,
  but it is contingent on a property of the council rather than universal: a
  council helps when no single member dominates every domain, and reduces to its
  best member when one does. That condition is checkable from the ladder before
  any council is built, which makes this an engineering criterion rather than a
  caveat. It also explains every earlier negative in this programme from a single
  cause: mechanisms were being asked to discover per-question which member to
  trust, when the only signal that ever paid was the per-domain prior measurable
  in advance.
- **The unclaimed headroom persists in both councils.** The oracle stands at
  0.75 on the Qwen confirmation set and 0.79 on the second council's first
  evaluation seed, some 13 to 15 points above the best single model in each. No
  mechanism tested in this programme reaches it.
- **Status:** VALIDATED for the decomposition and the extractor audit; second
  council INTERIM (one of three evaluation seeds) · Tarka PENDING

## F43 — The council advantage does not generalise; it is conditional on non-domination

**Cycle 29 · 2026-08-29 · exp28 · 360 evaluation questions, three seeds · GB10**

A second council was built to be maximally unlike the first: SmolLM2-1.7B,
deepseek-math-7B, Falcon3-3B and Falcon3-1B — four families, four tokenizers,
sizes from 1B to 7B. Champions were calibrated on a held-out seed standing in
for the ladder, and the same systems were compared on three further seeds.

- **The system reduces exactly to its best member.** Falcon3-3B is champion of
  both domains and is also the best single model at 0.6083; the plain router
  therefore scores 0.6083 identically, and the fallback router 0.6111 — one win,
  no losses across 360 questions. The eight-point advantage measured on the Qwen
  council is absent, and its absence is not noise but arithmetic: routing between
  identical destinations cannot move an answer.
- **The precondition F42 identified is confirmed as the governing condition.**
  Routing pays when different members are best on different domains and pays
  nothing when one member dominates. This is checkable from the ladder before a
  council is assembled, which turns the finding into a design rule rather than a
  disappointment: assemble councils only where the ladder shows no dominant
  member, and expect a council to be worth its cost in exactly that case.
- **Complementarity is present and remains unreachable.** The oracle stands at
  0.7917 against the best member's 0.6083, nineteen points, on a council where
  no mechanism extracted any of it. Together with the Qwen council's thirteen
  points this is now measured twice on disjoint model families, which makes the
  unclaimed headroom the most robust quantity in this programme and the strongest
  argument that the remaining problem is identification rather than knowledge.
- **A caution against the one nominally positive number.** The anchored vote
  scores 0.6194 here, above the fallback router's 0.6111. The margin is eight
  questions in 360 on a mechanism refuted at pre-registration (F41), measured
  post hoc on data selected for a different purpose, and it is not claimed. It
  is recorded so that a later reader encountering it in the results file has the
  provenance rather than a surprise.
- **Consequence for the programme.** The claim that survives is narrower and
  more useful than the one that would have been made without this council: a
  council of non-dominated members beats the strongest single model by eight
  points at 1.26 times its cost, and a council containing a dominant member
  should not be built at all. The autoresearch loop's remaining live hypothesis
  is whether a second observation per player — stability under perturbation —
  can reach the nineteen points that reweighting a single observation cannot.
- **Status:** VALIDATED (3 seeds, 360 questions, champions calibrated held-out) ·
  complementarity_generalises NOT MET · Tarka PENDING

## F44 — The parity claim was never compute-matched; weight-tying trades parameters for compute

**Cycle 30 · 2026-08-29 · exp31 seeds 43/44 on the 5090, plus a compute audit · CORRECTS F24**

F24 reported that the anytime-trained tied block reaches a ratio of 0.991 against
a param-matched explicit twelve-layer transformer "at matched budget", and that
figure has anchored the architecture line ever since. An adversarial review of
the adaptive-depth experiment, and an author audit prompted by it, establish that
the budget matched was parameters and iteration count, not compute.

- **The two models are genuinely parameter-matched, and inversely composed.**
  Direct measurement from the checkpoints gives 120.7M against 123.8M. But the
  tied model spends 85.9M of that on embeddings and 34.8M on its single block,
  while the explicit model spends 38.7M on embeddings and 85.1M across twelve
  layers. Matching parameters with one tied block forces width — $d = 1704$
  against $768$ — and the embedding table grows with it. (An adversarial review
  reported 206.4M against 162.5M from a stale results file; the checkpoint
  measurement above supersedes it.)
- **Width costs quadratic compute, so twelve iterations is not twelve layers.**
  Per token per block, $4d^2 + 2 d\,d_{ff}$ gives 34.81M units for the tied block
  against 7.08M for an explicit layer, a factor of **4.92**. Twelve iterations of
  the tied block therefore cost 4.92 times twelve explicit layers. The tied model
  reaches parity while spending nearly five times the arithmetic.
- **At genuinely matched compute it loses heavily.** Equal FLOPs are reached at
  **2.44 iterations**, and the tolerance sweep brackets that point: at a mean
  depth of 2.08 the ratio against the explicit baseline is 0.72 and 0.73 on the
  two seeds. At 3.5 iterations — still 1.43 times the explicit compute — the
  ratio is 0.89 and 0.86. Parity requires roughly five times the compute, and
  compute parity costs roughly a quarter of the quality.
- **Adaptive depth does not rescue it.** Allocating depth per token by residual,
  calibrated to a mean of 12, scores 0.681 against uniform-depth 0.684 and
  explicit 0.684 — ratios 0.9965 and 1.0011. Uneven spending is neither better
  nor worse than uniform at the same mean; the mechanism works, and buys nothing.
  What it does buy is graceful degradation: quality falls smoothly to 0.93 of
  baseline at half the depth and 0.72 at a sixth, which is a usable
  anytime property rather than an advantage.
- **The architectural reading, which is the point.** Weight-tying is a
  parameter-compression technique, not a compute-compression one. Removing
  eleven of twelve layers saves parameters; keeping capacity then demands width;
  and width is quadratic in the operation that dominates inference. A tied model
  is therefore attractive exactly where parameters are the binding constraint —
  memory-limited deployment, or a fixed parameter budget — and unattractive where
  arithmetic is. Every earlier result in this line is consistent with that and
  none of them stated it.
- **What this obliges.** The paper's parity claim must be restated as parity at
  matched parameters and matched iteration count, with the compute ratio given
  alongside; reporting it as "matched budget" without that qualifier overstates
  it. The correction is the author's, prompted by a review that found the
  discrepancy independently.
- **Status:** VALIDATED (2 seeds, third running; FLOP arithmetic from checkpoint
  configs) · CORRECTS F24's headline framing · Tarka RESOLVED with one of its
  five findings (the parameter claim) refuted by direct measurement

## F45 — At equal compute, weight tying costs 3% of quality and saves 63% of parameters

**Cycle 30 · 2026-08-29 · exp32, SPEC 0018 · 2 seeds · RTX 5090**

F44 measured the exchange rate between parameters and arithmetic at one point on
the curve — a tied block widened to $d = 1704$ so that its parameter count would
match a twelve-layer explicit model — and found parity purchased with 4.92 times
the arithmetic. That is the wrong operating point for a practitioner whose
constraint is compute, and SPEC 0018 registered the right one before running it:
set the tied block to the baseline's own width, so one iteration is exactly one
layer and compute is equal by construction rather than by calibration.

- **Equal compute, near-equal quality, a third of the parameters.** The tied
  block at $d = 768$ with twelve iterations scores BLiMP 0.663, 0.650 and 0.651
  against the explicit baseline's 0.682, 0.675 and 0.693 on the same three seeds,
  a mean ratio of **0.9582** with a standard deviation of 0.0169 (95% interval
  $[0.939, 0.977]$). The third seed is the weakest at 0.9394 and pulls the mean
  down from the 0.9676 the first two suggested, which is recorded because the
  two-seed figure was reported before it and was optimistic. It does so with 45.8M parameters against 123.8M — **2.70 times
  fewer** — and 7.1M block parameters against 85.1M, **twelve times fewer** —
  while executing exactly 84.9M compute units per token in both cases.
- **The pre-registered prediction is met.** SPEC 0018 stated before the run that
  a ratio above 0.95 would establish weight tying as a strong
  parameter-compression technique at zero compute cost, and below 0.85 would
  close the line as a quality proposition. The measured mean of 0.958 falls in
  the first regime, though the confidence interval straddles the threshold, so
  the honest reading is that the point estimate clears the bar and the evidence
  does not yet exclude falling just below it.
- **This corrects the pessimistic reading of F44 without retracting its
  arithmetic.** Both findings are true and they describe different points: made
  wide enough to match parameters, the tied model reaches full parity at 4.92
  times the compute; held at the baseline's width, it reaches 96.8% of the
  quality at equal compute and 37% of the parameters. The earlier design was
  answering "how much compute buys parity" when the useful question was "how many
  parameters does equal compute buy", and only the second has an answer a
  practitioner can act on.
- **What is claimed, and what is not.** This is not a quality win; the tied model
  is three points short on BLiMP and that gap is consistent across both seeds. It
  is a Pareto improvement in the parameter dimension at fixed compute, which is
  what an equilibrium formulation should be expected to buy, since storing depth
  as iteration trades memory for arithmetic and here the arithmetic is held
  fixed. The honest statement of the architecture's contribution is the exchange
  rate: roughly three percent of quality for roughly two thirds of the
  parameters, at no compute cost.
- **Where it is worth making.** Memory-limited deployment, fixed parameter
  budgets, and any setting where model size rather than latency is the binding
  constraint. It is not worth making where arithmetic binds, which F44 already
  established from the other direction.
- **Status:** VALIDATED (3 seeds, compute equal by construction, pre-registered
  prediction met on the point estimate) · Tarka PENDING

## F46 — Stability separates right from wrong, and still does not beat the best member

**Cycle 31 · 2026-08-29 · exp29, 60 questions, 5 samples per player · GB10**

Every mechanism refuted in this programme re-read one observation per model, and
F30's impossibility argument left exactly one route open: a second observation.
Stability under perturbation is that route — ask each model the same question
five times under resampling and paraphrase, and see whether it says the same
thing. The probe was gated so that nothing further would be spent if stability
carried no signal.

- **The precondition is met, and decisively.** Pooled over the four members,
  mean stability is 0.679 when the model's modal answer is right and 0.472 when
  it is wrong, a difference of 0.206 at Welch $t = 6.69$. Every member shows the
  separation individually: 0.615 against 0.455, 0.684 against 0.517, 0.777
  against 0.506, and 0.600 against 0.423. This is the first per-question signal
  in the programme that distinguishes correct from incorrect, and it confirms
  F32's diagnosis from the other side — correctness is not legible in the score
  field, and is legible in the agreement between independent samples.
- **It does not convert into a system that beats the best member.** Weighting
  each member's vote by its own measured stability scores 0.700 at the best
  setting against the strongest member's 0.717, losing 3 questions to 4 on
  disagreements ($z = -0.38$). Sharpening the weighting does not help (0.683 at
  every larger setting), and removing it entirely gives plain majority voting at
  0.600, which is worse still.
- **The arithmetic of why is simple and was checkable in advance.** The best
  member is wrong on 17 of 60 questions, and some other member is right on only
  8 of those. The ceiling for any rescue mechanism is therefore 0.717 plus 8/60,
  and the observed oracle of 0.850 confirms it. Stability would have to identify
  nearly all 8 while never displacing a correct answer among the other 43, and it
  identifies about half while displacing a comparable number.
- **What this establishes and what it costs.** The signal is real, which is worth
  knowing because eleven prior mechanisms found nothing at all; a second
  observation succeeds where every re-reading of the first failed. But it is not
  strong enough to overcome a dominant member, which is the same condition F43
  identified from a different direction. Five samples per member is five times
  the compute of asking the best member once, for a result 1.7 points below it.
- **Consequence.** The full stability-weighted council, which the autoresearch
  agent had ranked second and gated behind this probe, is not run. The gate
  worked as intended: four GPU-hours were saved by a one-hour probe whose
  purpose was to make exactly this call.
- **Status:** VALIDATED (60 questions, 5 samples, single seed) · precondition MET
  · system-versus-best-member NOT MET · Tarka PENDING

## F47 — Depth conditioning makes weight tying worse, and the prediction was wrong in direction

**Cycle 31 · 2026-08-29 · exp34, SPEC 0019 · 3 seeds · RTX 5090**

SPEC 0019 predicted, before the run, that per-iteration modulation of the shared
block would recover one to two of the four points that separate compute-matched
tying from the explicit baseline, on the reasoning that a scale and shift can
differentiate each depth's output distribution. Three seeds refute the prediction
and refute it in the opposite direction to the one anticipated.

- **Conditioning costs quality rather than recovering it.** The depth-conditioned
  block scores 0.637, 0.621 and 0.640 against the plain tied block's 0.663, 0.650
  and 0.651 on the same three seeds — worse on every seed, by 0.026, 0.029 and
  0.011. As a ratio against the explicit baseline the mean falls from 0.9582 to
  **0.9258**, a change of $-0.0323$ where SPEC 0019 required $+0.0169$ to count
  as working. The direction is consistent and the effect exceeds the seed spread.
- **The cost was as advertised and is not the issue.** The conditioned model adds
  18.4K parameters, 0.04% of the 45.8M total, and no arithmetic. Nothing about
  the expense explains the loss.
- **Why the reasoning failed, as far as this measurement shows.** The prediction
  assumed that letting the map differ across depths could only add expressiveness,
  since the modulation initialises as the identity and could in principle learn to
  stay there. What it evidently does instead is break the property the tied model
  depends on: applying one map repeatedly makes the iteration a contraction toward
  a fixed point, and a map that changes at every step is no longer iterating
  toward anything. Twelve modulated applications are twelve different functions
  composed once each, which is a twelve-layer network with twelve times too few
  parameters rather than an equilibrium solved twelve times. The tying result was
  never about reusing weights; it was about reusing the *same map* so that
  repetition converges.
- **What this establishes about the architecture.** The four-point gap is not a
  deficit of depth-specific expressiveness, because supplying that expressiveness
  made things worse. It is more likely the capacity of one block against twelve,
  which no modulation scheme addresses. SPEC 0019 stated that indistinguishability
  would close modulation as a direction; a consistent loss closes it more firmly,
  and closes with it the family of remedies that differentiate the map in time.
- **The honest position of the architecture line.** Compute-matched weight tying
  stands at 0.958 of the explicit baseline with 2.70 times fewer parameters
  (F45). That is the result, and this attempt to improve it failed. Reporting the
  failed attempt at the same length as the success is the point of recording it.
- **Status:** VALIDATED (3 seeds, pre-registered prediction refuted in direction)
  · verdict NOT MET · closes depth modulation · Tarka PENDING

## F48 — The memory saving is in weights, not activations, and the deployment claim must say so

**Cycle 32 · 2026-08-29 · exp35 · RTX 5090 · measured, not asserted**

The natural pitch for a fixed-point model on a small device is that it saves
memory twice: one block of weights instead of twelve, and an iterate overwritten
in place instead of activations accumulating with depth. The first is true and
large. The second is not true as implemented, and measuring it before claiming it
is the reason to measure at all.

- **Weights: a genuine and large saving.** The compute-matched tied model holds
  183.2 MB against the explicit baseline's 495.7 MB in bfloat16 — a ratio of
  **0.370**, saving **312.5 MB**. On a device where the model must be resident,
  this is the difference between fitting and not fitting, and it is the direct
  consequence of holding one block's parameters rather than twelve.
- **Activations: no saving at realistic batch sizes.** Peak allocation during a
  forward pass is essentially identical once the batch is above one — ratios of
  1.006 at batch 4 and batch 16 across both sequence lengths tested. At batch 1
  the tied model is actually **2.3 to 2.7 times worse** (122.0 MB against 52.6 MB
  at sequence 128), because the Anderson-accelerated solver retains a history of
  past iterates in order to accelerate, and that history is a memory cost the
  explicit stack does not pay.
- **The consequence for deployment, stated plainly.** Within a 512 MB activation
  budget both models reach the same largest batch of 10, so the tied model buys
  nothing in throughput at a fixed activation budget. What it buys is 312.5 MB of
  permanently reclaimed weight memory, which on a small device is the scarcer
  resource. The honest claim is therefore about model footprint, not about
  running larger batches, and any material saying otherwise would be
  contradicted by this measurement.
- **An avenue the measurement itself suggests.** The batch-1 regression is an
  artefact of solver history rather than of the architecture: plain Picard
  iteration overwrites its iterate and would carry no such history. If the
  single-stream case matters for a device deployment, the solver is the thing to
  change, and the quality cost of doing so is measurable with the checkpoints
  already in hand.
- **Status:** VALIDATED (direct allocation measurement, both models, same card,
  same dtype) · qualifies the low-memory claim to weights only · Tarka PENDING

## F49 — GGUF cannot represent this architecture without destroying what makes it worth shipping

**Cycle 32 · 2026-08-29 · export feasibility, verified by author recount**

Distribution through llama.cpp-based tooling was investigated because it is the
route to consumer devices, which is where a parameter-efficient model should
matter most. The finding is that the format and the architecture are
incompatible in a way no amount of engineering resolves.

- **The mechanism of the incompatibility.** A GGUF graph is a fixed sequence of
  layers, each with its own tensors. A weight-tied fixed-point model is one block
  applied repeatedly under a convergence criterion. Representing the second in
  the first requires unrolling the block into twelve layers, and llama.cpp
  provides no tensor aliasing, so the twelve layers each carry their own copy of
  the weights.
- **The cost, which is the whole point inverted.** The tied block holds 34.8M
  parameters. Unrolled twelvefold for GGUF it holds 417.6M — **4.91 times larger
  than the 85.1M of explicit layers it was supposed to save against.** A format
  conversion undertaken to reach small devices produces a file five times bigger
  than the conventional model, and additionally discards the convergence
  criterion, so the exported artefact is neither smaller nor the same model.
- **The honest export is safetensors, and it is exact.** Weight tying is
  preserved, overhead is zero against the checkpoint, and round-trip fidelity is
  verified to 1e-5 by tests that pass. ONNX at a fixed iteration count is
  possible with modest overhead and loses adaptivity, which is a defensible
  trade for a fixed-budget deployment and is documented as such.
- **Platform eligibility, reported rather than assumed.** OpenRouter's catalogue
  is instruction-tuned models at 7B and above; a 121M pretrained research model
  does not qualify and no review path exists for one. LM Studio and Ollama accept
  arbitrary GGUF and impose no size floor, so submission is technically possible
  — but only via the GGUF whose cost is stated above, which means the only route
  onto those platforms is the one that misrepresents the model. Hugging Face
  imposes no such constraint and carries the architecture honestly, which is why
  it is where the artefacts are published.
- **The wider lesson for the deployment claim.** F48 already narrowed the
  low-memory story from activations to weights. This narrows it again: the weight
  saving is real in a framework that can express weight tying, and evaporates in
  one that cannot. A parameter-efficiency result is only as portable as the
  formats that can represent it, and that constraint belongs beside the number
  rather than in a footnote.
- **Status:** VALIDATED (export tests pass, unrolling cost recomputed by the
  author) · GGUF rejected on integrity grounds · Tarka PENDING

## F50 — The equal-compute ratio holds on a threefold-wider benchmark

**Cycle 33 · 2026-08-29 · exp36 full suite · 31 BLiMP phenomena · RTX 5090**

The F45 ratio was measured on a 12-phenomenon subset. The full-suite harness
evaluates every phenomenon available in the local cache — 31, with the per-file
perplexity arm returning empty inside the container owing to a cache-path
difference, recorded rather than hidden — and the ratio is stable: the
compute-matched tied model scores 0.5731 against the explicit baseline's 0.6005,
a ratio of **0.9544** against F45's 0.9582, with a per-phenomenon record of 12
wins, 17 losses and 2 ties. The claim survives tripling the breadth of the
evaluation, and per-phenomenon the two models are close rather than the ratio
being an average over divergent behaviours.

**Status:** VALIDATED (single seed, 31 phenomena, paired per-phenomenon) ·
consistent with F45 · Tarka PENDING
