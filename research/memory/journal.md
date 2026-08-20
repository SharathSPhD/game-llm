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
