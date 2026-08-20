# SPEC 0002 — RQ-1: Adversarial review of kinetic_ai vs the research doc

Status: ACTIVE · Phase 1b

## Ground truth (condensed from res-docs/)

The research doc specifies, per component:

- **QRE**: σ_i(a) ∝ exp(λ·u_i(a, σ*_{-i})); λ→∞ Nash, λ→0 uniform; temperature
  τ=1/λ is a rationality parameter, not post-hoc sampling.
- **MMD** (Sokota et al. 2023): mirror-descent update with magnetic proximal term
  toward reference; **converges linearly (last-iterate) to the τ-regularized QRE**,
  NOT to Nash unless the reference is periodically reset (Regularized Nash Dynamics /
  FTRL discipline traces QRE path → Nash). Stepsize conditions matter; simultaneous
  GDA without the magnet cycles (imaginary Jacobian eigenvalues).
  Closed form on simplex: x_{t+1} ∝ [x_t^(1/(1+ητ)) · x_ref^(ητ/(1+ητ)) · e^(ηg/(1+ητ))].
- **Dilated entropy** for treeplex/sequence-form games: per-info-set entropy weighted
  by reach; 1-strongly-convex w.r.t. L1 on treeplex (Hoda et al./Kroer et al. weights).
- **DEQ**: z* = f_θ(z*,x); Anderson/Broyden solvers; backward via IFT or JFB; O(1)
  depth-memory. pcDEQ (nonneg weights + orthant-concave activations) ⇒ unique fixed
  point, geometric convergence.
- **Token auctions** (Duetting et al. 2024): second-price truthful; aggregation must be
  strictly monotone for incentive compatibility; VCG payment = externality imposed on
  others (welfare of others without i − welfare of others with i), NOT individual reward.
- **SPPO** (Wu et al. 2024): constant-sum self-play game; multiplicative-weights /
  exponential-weights update converges to Nash of the preference game; S-SPPO adds
  semantic calibration.

## Known suspects (from Phase 0 exploration — adjudicate, don't assume)

1. `tests/test_mmd.py` 4 failures — three-way adjudication REQUIRED: is the
   implementation wrong (kinetic_ai/optim/mmd.py explicit dual update vs closed form),
   are the tests wrong (they expect NashConv→0 under a FIXED magnet, which theory says
   converges to regularized QRE, not Nash; lr=0.5 τ=0.1 may violate stepsize
   conditions), or are the doc claims overstated?
2. `kinetic_ai/mechanisms/auctions.py:231-283` VCG payments.
3. `kinetic_ai/config.py:188` eval() on untrusted YAML.
4. `kinetic_ai/optim/bregman.py:216` DilatedEntropy weights `1/(depth·max_branching)`.

## Review protocol

- Dimensions: MMD/Bregman math · QRE · DEQ/pcDEQ · auctions/mechanisms · SPPO ·
  eval/statistics · test-suite validity.
- Each finding: file:line, doc claim vs code behavior, concrete failure scenario,
  severity (critical/major/minor), suggested fix.
- Every finding adversarially verified (refute-first) through three lenses:
  mathematical correctness, doc fidelity, test validity.
- Output → `research/memory/findings-rq1.md`; fixes proceed TDD (failing test first).
