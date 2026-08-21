# Validated Findings

(Populated only with Tarka-reviewed findings as experiments close. Every number here
traces to a real run with config hash + seeds. All statuses: SIGN-OFF PENDING until
operator sign-off.)

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
- **Status:** VALIDATED · sign-off pending

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
- **Status:** VALIDATED · sign-off pending · feeds paper §Convergence Analysis and
  motivates adaptive-reference theory work.

## F3 — Regularized Nash Dynamics (periodic reference resets) reach Nash universally (H2: VALIDATED)

- **Claim:** MMD with periodic reference resets converges to Nash (NashConv < 0.05) on
  symmetric AND asymmetric games.
- **Evidence:** final NashConv 8.48e-6 [4.78e-6,1.26e-5] (MP), 1.07e-6 (RPS), 5.21e-5
  [2.62e-5,9.08e-5] (biased RPS); mean R² 0.7181–0.9825. exp01, 10 seeds.
- **Tarka:** CONFIRM_WITH_CORRECTION (R² range corrected as stated).
- **Status:** VALIDATED · sign-off pending

## F4 — DEQ peak activation memory is O(1) in effective depth; explicit stacks are O(N) (RQ-6 Tier A: VALIDATED)

- **Claim:** DEQ implicit block: 0.032±0.000 MB peak activation memory, flat across
  effective depth (0% variance); explicit stack: linear, slope 0.0168 MB/layer
  (N=4→0.067 MB … N=32→0.539 MB), like-for-like layers, CPU measurement.
- **Evidence:** results/exp03_deq_solvers/ (config sha a0f8f5c0…, 5 seeds).
- **Tarka:** CONFIRM. (GPU-scale measurement for H1's ≤50% claim happens in Tier B.)
- **Status:** VALIDATED · sign-off pending

## F5 — Anderson acceleration beats Picard exactly where theory predicts: stiff fixed points (VALIDATED, iteration 2)

- **Claim:** On contraction maps with controlled spectral radius ρ, Anderson/Picard
  iteration ratio < 0.95 at ρ=0.999 (0.888 at dim 32; 0.940 at dim 128); no advantage
  at easy ρ (iteration-1 "miss" explained by problem easiness). Spectral radii
  empirically verified (max abs error ~2e-4).
- **Evidence:** results/exp03_deq_solvers/ iteration 2, ρ∈{0.9,0.99,0.999}, 10 seeds.
- **Tarka:** CONFIRM / CONFIRM_WITH_CORRECTION (wording adopted).
- **Status:** VALIDATED · sign-off pending

## F6 — Second-price token auction is exactly truthful; weighted aggregation is measurably manipulable (H4 groundwork: VALIDATED)

- **Claim:** Empirical truthful-bidding regret in second-price auctions is exactly 0.0
  (95% CI [0.0,0.0], 16k observations, misreport grid 0.25v–2v). Weighted-aggregation
  mechanism has positive manipulation gain (mean regret 0.0773 at n=3, 0.0683 at n=5)
  — documented as non-truthful, matching the Phase-1 finding that its payments are
  not VCG.
- **Evidence:** results/exp04_auction_truthfulness/ (config sha 5c458dac…, 10 seeds ×
  200 auctions × {3,5} agents).
- **Tarka:** CONFIRM (both claims).
- **Status:** VALIDATED · sign-off pending

## F7 — Warm-started homotopy accelerates QRE path tracing (VALIDATED, iteration 3)

- **Claim:** Warm-starting each QRE solve from the previous λ's solution reduces total
  solver iterations vs cold-start: 25.2% on asymmetric 2×2 (5990→4481 avg), 2.6% on
  biased RPS; degenerate control (matching pennies) flat as expected. Exploitability
  along the path is NOT globally monotone for these games (honest partial vs prereg);
  path strategy movement is smooth and small (0.015 < 0.05 prereg threshold).
- **Evidence:** results/exp02_qre_homotopy/ iteration 3, λ∈logspace(0.01,100,50).
- **Status:** VALIDATED (with two honest partials) · sign-off pending

## F8 — Undamped logit-QRE fixed-point iteration requires damping beyond small λ (METHOD FINDING)

- **Claim:** Plain s←softmax(λAs) diverges for λ‖A‖ moderately large (biased RPS at
  λ>0.32). Adaptive damped iteration s←(1-γ)s+γ·softmax(λAs) (γ init 1/(1+λ/10),
  halve on residual increase) converges across λ∈{1,10,100} (21 / 700 / 42k iters).
- **Evidence:** tests/test_qre.py::TestQREHighRationality; kinetic_ai/games/qre.py
  damped solver (default on, backward compatible).
- **Status:** VALIDATED · sign-off pending · exposes future work: Anderson-accelerated
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
- **Status:** VALIDATED as method finding · sign-off pending
