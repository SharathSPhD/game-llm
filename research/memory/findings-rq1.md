# RQ-1 Findings — Adversarial review of kinetic_ai vs research doc

Date: 2026-08-20 · Method: 7 dimension-sliced reviewers + refute-first 3-lens
verification (36 agents; numerical checks run in-venv) · Status: CONFIRMED, feeding
Phase 1c fix loop · Full detail: workflow artifact `review_findings.json` (session
scratchpad; regenerate via SPEC 0002 protocol).

**28 confirmed/partial, 1 refuted.**

## Critical

- **mmd-001** [mmd.py:206] Dual update ≠ Sokota closed form (missing 1/(1+ητ)
  normalization). PARTIAL: real formula divergence (~1.9% L2 at lr=.5,τ=.1), but the
  test failures' root cause is stepsize violation + fixed-magnet Nash expectation.
- **deq-001** [deq_layer.py:257] Broyden denominator clamp flips sign for negative
  denominators — corrupts Jacobian update.
- **auctions-001** [auctions.py:157] Weighted-aggregation + "VCG" payments is NOT
  truthful — overbidding is profitable (counterexample derived).
- **sppo-001** [self_play.py:255] S-SPPO semantic gate/latent repulsion implemented but
  never wired into run_self_play — claimed feature inert.
- **meta-mmd-test-wrong-expectations** [test_mmd.py:162] Tests demand NashConv→0 under
  a FIXED magnet; theory guarantees τ-regularized QRE only (Nash needs periodic
  reference resets, i.e. Regularized Nash Dynamics).

## Major (summary)

- MMD: convergence tests use invalid stepsizes and wrong equilibrium target
  (mmd-002/003/004); empirically the fixed-magnet lr=0.5 run limit-cycles
  (NashConv≈1.53) exactly as theory predicts — the algorithm cycles, the tests are wrong.
- DilatedEntropy weights 1/(depth·max_branching) lack reach-probability weighting
  (Hoda/Kroer treeplex theory) (mmd-005, meta-dilated-entropy).
- Kuhn poker hardcoded NE strategy evaluates to wrong game value (qre-002);
  test_nash_equilibrium_value doesn't test the value (qre-001).
- test_anderson_faster_than_picard has `or True` — always passes (deq-002);
  test_relu_is_not_concave asserts nothing (deq-003).
- Second-price payment ignores reserve when 2nd bid < reserve (auctions-002); no
  truthfulness property tests (auctions-003).
- SPPO win rates never resampled from current policy → constant advantages, divergent
  log-weights, no fixed point (sppo-002/003); smoke-test-only coverage (sppo-004).
- Wilcoxon normal approximation unguarded for n<10 (eval-wilcoxon); "linear
  convergence" = R²>0.9 heuristic conflates fit quality with rate (meta-convergence).
- README example promises NashConv≈0 for a configuration theory says cycles
  (meta-readme). simulate.py unseeded (meta-simulate).

## Minor

- qre-003 weak monotonicity assertion; sppo-005 docstring wrong (force vs loss);
  eval-bootstrap-ci flaky-by-design (~5%); eval-support-size boundary `>` vs `>=`;
  meta-config eval() on annotations (unsafe + warning-suppressing).

## Refuted

- sppo-006 (O(n²) win-rate scalability "undocumented") — documented behavior,
  intended scale.

## Adjudication of the headline question (the 4 failing tests)

The MMD *implementation* is a mathematically coherent dual-space magnetic update; the
*tests* encode a wrong theoretical expectation (Nash from a fixed magnet, oversized
steps), and the *docstring/README* overclaim. Fix = all three surfaces: implement the
Sokota closed form (normalized), add stepsize-condition validation, rewrite tests to
assert (a) QRE convergence with fixed magnet at valid stepsizes and (b) Nash
convergence under periodic reference resets (RND), and correct the README claims.
