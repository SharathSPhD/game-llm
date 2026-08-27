/* Learn section content — sourced directly from findings.md and results.json */

export const learnSections = [
  {
    id: "why-game-theory",
    title: "Why Game-Theoretic Training",
    slug: "why-game-theory",
    description: "The foundational thesis: equilibrium learning dynamics in multi-agent systems.",
    takeaway: "When multiple agents optimize simultaneously, convergence is not guaranteed—but equilibrium computation is.",
    content: `
The classical approach to training language models treats the learner as a solitary optimizer descending a loss landscape.
But in multi-agent systems—from distributed training to preference learning to mechanism design—no agent optimizes in isolation.

**The Problem:** Standard gradient descent (GDA) exhibits cycling behavior on games, never converging to equilibrium.
This matters because language model training increasingly involves multiple objectives: aligning to human preference (through RLHF),
aggregating expert model outputs (via auctions), and coordinating across distributed agents.

**The Insight:** Game-theoretic dynamics offer convergence guarantees. By reframing training as a game where the model and
reference distribution play against each other, we can apply equilibrium solution concepts instead of pure loss minimization.
Magnetic Mirror Descent (MMD) provably converges to equilibrium where GDA cycles, and quantal response equilibria (QRE)
interpolate between randomness and best-response play.

**Why It Matters:** Convergence is the foundation of reproducibility. A training algorithm that cycles or drifts unpredictably
makes it impossible to reason about final performance or debug failures. Equilibrium-based training offers a stable target
and a mathematically grounded framework for understanding multi-objective learning.
    `.trim(),
  },
  {
    id: "mmd-convergence",
    title: "Magnetic Mirror Descent and Last-Iterate Convergence",
    slug: "mmd-convergence",
    description: "How MMD reaches equilibrium where standard GDA cycles.",
    takeaway: "MMD with a fixed magnetic anchor achieves linear convergence to equilibrium on zero-sum games (F1–F3).",
    content: `
**The Finding:** On symmetric zero-sum games (Matching Pennies, Rock-Paper-Scissors), Magnetic Mirror Descent with a
fixed-point anchor exhibits linear (geometric) last-iterate convergence to equilibrium while simultaneous gradient descent
cycles, bounded away.

**The Numbers (F1–F3):**
- Matching Pennies: MMD log-linear fit R²=0.9948 (linear convergence), GDA NashConv 1.93±[1.90,1.96] (cycling)
- Rock-Paper-Scissors: MMD R²=0.9015, GDA NashConv 1.76±[1.62,1.88]
- Regularized Nash Dynamics with periodic resets: reaches NashConv < 0.05 on both symmetric and asymmetric games

**How It Works:** MMD adds a "magnetic" term that pulls the player strategy toward a reference distribution (the anchor).
On symmetric games, this anchor is the uniform distribution; on asymmetric games, it becomes context-dependent.
The magnetic pull prevents the oscillatory behavior of GDA and forces convergence. The physics is straightforward:
the fixed point becomes attractive instead of a saddle point.

**Asymmetric Games (F2):** The uniform-anchor MMD dynamics do NOT converge to the logit-QRE; instead, they reach
a distinct context-dependent attractor. This is a discovery finding: the natural equilibrium notion for MMD differs
from QRE on asymmetric settings.

**Implication for Training:** If preference learning or multi-model aggregation is a zero-sum game between model and
reference, MMD provides convergence where Adam + GDA would cycle. The magnetic pull can be tuned (via τ, the temperature)
to control exploration vs. exploitation at training time.
    `.trim(),
  },
  {
    id: "eq-depth",
    title: "Equilibrium Language Models — Depth as a Fixed Point",
    slug: "eq-depth",
    description: "How DEQ transformers enable equilibrium computation inside the model.",
    takeaway: "DEQ blocks reduce activation memory from O(N) to O(1) in depth, but require contractive fixed-point maps (F4, F14).",
    content: `
**The Core Idea:** Instead of stacking N transformer layers (each storing activations for backprop), we iterate a single
layer to convergence, computing depth as time rather than space. Deep Equilibrium (DEQ) models are weight-tied transformers
that run in a solver loop until they reach a fixed point.

**The Memory Win (F4):** A DEQ implicit block occupies 0.032±0.000 MB of peak activation memory, flat across effective depth.
An explicit 32-layer stack requires 0.539 MB (16.8× more). On GPU-memory-constrained problems, this is transformative.

**The Challenge (F14):** The EqLM v1 trained models' solver residuals plateau at a constant value with tail-ratio ≈0.99
over 100 iterations. The signature shows z ← z + α·g(z) with ‖g‖ constant: iterates drift at speed α‖g‖. There is no fixed point
being approached—the model is a weight-tied 12-iteration transformer, not an equilibrium model.

**The Root Cause:** The layernorm is outside the fixed-point map. Without a bounding operation inside the map, the residuals
are unbounded and convergence is impossible.

**EqLM v3 Design (Next):** Put the outer LayerNorm inside the map — f(z,x) = LN(z + Attn + MLP + inj(x)).
This follows the DEQ-transformer form (Bai et al.) and ensures iterates are bounded. With bounded iterates and spectral
normalization on sub-layers, a true fixed point can exist and a solver can converge.

**Why Equilibrium Depth Matters:** If we can train models that solve for a fixed point at each forward pass, we unlock
a fundamentally different way to think about model capacity. The model becomes a solution to an equilibrium problem,
not a stack of transformations. This is relevant to efficiency (a single iterated layer vs. N big layers) and to interpretability
(what is the equilibrium these iterates are finding?).
    `.trim(),
  },
  {
    id: "honest-scaling",
    title: "The Honest Scaling Story: Two Misses, Then Parity",
    slug: "honest-scaling",
    description: "How the width gap appeared, widened, and was closed by changing the training regime.",
    takeaway: "The width gap belonged to the training regime, not the architecture: solver-trained EqLM scored 0.930 (11M) then 0.787 (121M) of baseline; anytime-unrolled training closed it to 0.991 — and trained 2.1× faster (F18, F20, F24).",
    content: `
**The Pre-Registered Claim (H1):** a weight-tied equilibrium LM reaches ≥95% of a parameter-matched explicit
transformer's BLiMP at matched token budget. This was formally **missed — twice** — before it was effectively resolved.

**Miss one (F18, 11M params, 3 seeds):** solver-trained EqLM reached ratio 0.930 with a tight bootstrap CI
[0.898, 0.949] — genuinely below the 95% bar, no rounding rescue. **Miss two, worse (F20, 121M params, 3 seeds):**
the ratio *fell* to 0.787 [0.785, 0.788]. Adversarial review refuted our first mechanism (a binary solver failure at
width) — the solver exhausted its 12-iteration budget at *every* scale; what grew with width was the *price* of that
truncation.

**The turn (F24):** train the same weight-tied block *unrolled* — twelve explicit applications of the map with
cross-entropy supervision at iterates z4, z8, z12 — and evaluate with the standard Anderson solver. Three seeds:
BLiMP 0.662 / 0.697 / 0.672 → **ratio 0.991**, with one seed *above* its explicit baseline. Identical data, steps,
batch, optimizer, and learning rate as the control (audited). Bonus: unrolled training is **2.1× faster** than
implicit-differentiation training — the equilibrium solve, not backprop, dominates EqLM's training cost.

**Anytime property (F24 rider, pre-registered):** the same checkpoint evaluated at solver budgets {4, 8, 12} scores
gracefully (e.g. 0.601 / 0.674 / 0.697 on seed 43). One model, three inference budgets.

**Honest costs:** unrolled training stores its iterates, so it pays explicit-like activation memory *during training*
(16.4 GB vs 7.8 GB); serving keeps the implicit model's O(1) depth-memory and warm-start decoding. And the solver
still exits uncertified — quality and certification were solved in *different* arms (see the findings page, F24).
    `.trim(),
  },
  {
    id: "token-auctions",
    title: "Token Auctions: Where They Win and Where They Don't",
    slug: "token-auctions",
    description: "Truthful per-token model selection — a scoring-time win with an honest closed-loop boundary.",
    takeaway: "Second-price token auctions are exactly truthful (regret 0.0) and beat the best single specialist by 23% at scoring time (F22) — but the advantage inverts in closed-loop generation (F23), where the auction is still the least repetitive system.",
    content: `
**The Setup:** two 30M-parameter specialists trained on disjoint BabyLM subdomains — child-directed speech (childes)
vs written text (simple Wikipedia). At each token, each model bids its own confidence (max next-token probability);
the higher bidder's distribution is used and it pays the second price.

**Truthfulness (F6):** empirical misreporting regret in the second-price mechanism is exactly 0.0 (95% CI [0.0, 0.0],
16k observations); weighted logit aggregation is measurably manipulable (mean gain 0.077 at n=3).

**Scoring time — MET (F22, 3 seeds):** on a 50/50 interleaved held-out stream, mixed-domain perplexity ranks
auction **182.8** < uniform logit-average ensemble 207.9 < best single specialist 236.8 < worst 1242.4. Per-token
selection dominates any fixed commitment because each specialist collapses off-domain (~4000+ ppl). Adversarial review
scoped this precisely: it is teacher-forced *selection* at scoring time, not autoregressive generation.

**Closed loop — MISSED (F23, 3 seeds):** when each system generates its own continuation, the best single specialist
beats the auction under an independent judge (3.4–3.7 vs 4.2–4.7 NLL/token, all seeds). Teacher forcing had been
re-anchoring the selection to the true context every token; remove the anchor and the auction drifts toward one
specialist's style regardless of the prompt. The review's insistence on the scoring-time scope was vindicated by the
follow-up experiment.

**Two judge-independent facts survive:** the auction is the *least repetitive* system (3-gram repetition 0.39–0.48
while the uniform ensemble degenerates at 0.78–0.83) — bid competition acts as an implicit anti-repetition
regularizer — and the judge metric itself turned out style-dominated (a ~2-nat prior for child-speech-flavored text),
a caution for any judge-based generation eval at this scale.
    `.trim(),
  },
  {
    id: "preference-optimization",
    title: "Preference Optimization: an Under-Dosed Magnet and an Unexpectedly Stable Architecture",
    slug: "preference-optimization",
    description: "What DPO does to unseen phenomena, and why the equilibrium model barely moves.",
    takeaway: "The magnetic anchor is second-order to the DPO gradient at every tested dose (H3 PARTIAL) — but DPO damages unseen phenomena (0.740→0.612), and the equilibrium architecture is 1655× more drift-resistant than the explicit model under identical updates (F21).",
    content: `
**The Test (H3):** BLiMP minimal pairs ARE preference pairs (good ≻ bad sentence) — no reward model, no judge.
Train DPO on three linguistic phenomena, evaluate on two unseen ones. The magnetic arm (MPO) is the *same* optimizer
code path with a proximal pull toward the frozen base; arms differ only in the magnet strength τ.

**The magnet result — honest PARTIAL:** at pre-registered τ ∈ {1e-3, 1e-2}, MPO's held-out accuracy equals DPO's
*exactly* (all seeds, both base models) and the KL reduction is insignificant. A pre-registered dose-response rider
refuted its own prediction: even τ = 10 — a thousand times the registered maximum — reduces KL drift only 1.2% with
accuracy unchanged. The magnetic pull is second-order to the DPO gradient across four orders of magnitude at this
budget. (Consistent with pretraining finding F12: the magnet's natural home is stability, not preference-phase
drift control.)

**Secondary finding A — DPO damages unseen phenomena:** on the explicit base model, trained-phenomena accuracy climbs
0.646 → 0.877 while *unseen* phenomena drop 0.740 → 0.612, at ~1.24 nats/token KL drift from the base. The
reward-hacking worry, observed directly in the linguistic domain.

**Secondary finding B — the equilibrium architecture barely moves:** under the *identical* loss, learning rate, and
steps, EqLM shifts its train accuracy only 0.493 → 0.516 with **1655× less** KL drift (0.00075 vs 1.24 nats/token).
Adversarial review excluded the trivial explanation (its base accuracy is low, so the preferences were not already
satisfied): the cause is damped gradients through the 12-iteration weight-tied solve. Double-edged and reported as
such — stability against preference-induced drift, and insensitivity to preference fine-tuning at matched learning
rate. This is a property of the *architecture*, not of the magnetic term.
    `.trim(),
  },
  {
    id: "closure-contract",
    title: "The Closure Contract and Reproducibility",
    slug: "closure-contract",
    description: "How findings are verified and signed off.",
    takeaway: "Every finding must satisfy six layers: technical (tests green), empirical (all arms run), integrity (Tarka review), artifacts (results pushed), memory (state.json), sign-off (operator).",
    content: `
**The Contract:** A phase closes only when it satisfies all six layers:

1. **Technical:** pytest green, ruff+mypy clean, coverage ≥80%.
2. **Empirical:** All experiment arms run; finding declared with interpretation.
3. **Integrity:** Adversarial (Tarka) review of the finding resolved; comparisons like-for-like (matched params/tokens/compute); ≥3 md5-distinct seeds.
4. **Artifacts:** Code+results pushed; paper/site sections updated from the finding.
5. **Memory:** research/memory/ (state.json, journal.md, findings.md) and ADRs updated.
6. **Sign-Off:** Operator sign-off on interpretation before merge to main.

**Hard Rules:**
- Never declare "failed" on attempt 1; double-check the setup.
- Never a NULL finding without ≥2 documented interventions; you must know why it didn't work.
- Never report a number you didn't produce; all claims trace to a run with config hash + seeds.
- Never fabricate a citation or invent evidence.
- Never two GPU jobs at once (GPU lock in state.json prevents race conditions).

**Why It Matters:** Science requires reproducibility. Every finding published here includes:
- Git commit hash
- Config file SHA256
- Random seeds (≥3 distinct)
- Experiment folder with full results
- Tarka verification notes (adversarial review)

This is binding. A null finding is valid if the intervention is documented and reviewed.
A contradicted claim is value (it updates the hypothesis set). Only fabrication and rushed publication are out of scope.

**Operator Ethos:** Hypotheses inform the next iteration/invention toward operator intent.
A well-documented null is a closure state, not an endpoint. The experiment ends when you've learned something,
either because you succeeded or because you discovered a defect and fixed it.
    `.trim(),
  },
];
