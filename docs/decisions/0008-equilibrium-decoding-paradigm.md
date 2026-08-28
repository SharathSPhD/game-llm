# ADR 0008 — Equilibrium Decoding: the output distribution as a solved game

Date: 2026-08-28 · Status: accepted · Supersedes the "council of specialists"
framing in SPEC 0015 with a stronger claim about what the system computes.

## The problem with the obvious system

Assembling open specialists behind a router or an averager reproduces the
existing architecture and competes for incremental gains on its terms. Every
such system — dense model, mixture of experts, ensemble, router — performs one
forward pass and reads a distribution off it. Averaging blends; routing selects.
Neither solves anything, so neither can extract more from a set of models than
the models already contain in superposition.

## The paradigm

The next-token distribution is defined as the **equilibrium of a game among
model-players**, computed at decode time.

Let players $i = 1..N$ hold logits $\ell_i$ at the current position, with
$p_i = \mathrm{softmax}(\ell_i)$. A general-purpose model supplies the magnet
$p_{\mathrm{ref}}$. The council seeks a distribution $y$ over the vocabulary
that is a fixed point of the influence game: each player's weight grows with
how well the consensus serves its payoff,

$$w_i(y) \;\propto\; \exp\!\big(\beta \langle y, \ell_i \rangle\big),$$

and the consensus moves toward the weighted objective under the Magnetic Mirror
Descent update in closed form,

$$y_{t+1} \;=\; \frac{y_t + \eta g_t + \eta\tau\, p_{\mathrm{ref}}}{1 + \eta\tau},
\qquad g_t = \sum_i w_i(y_t)\, \ell_i .$$

The fixed point is the $\tau$-regularized quantal response equilibrium of that
game. Sampling happens from the equilibrium, not from any player and not from
their average.

## Why this is the kinetic programme rather than a departure from it

Every component is a result this project already established. F1 supplies linear
last-iterate convergence for exactly this update where simultaneous play cycles.
F21 supplies the placement: a parameter-space magnet proved second-order to the
optimization gradient, whereas the magnet's natural home is policy space, which
decode time is. F6 makes confidence bids truthful, so $w_i$ cannot be gamed by a
miscalibrated player. F19 supplies warm starting, since consecutive positions
have nearby equilibria. F24 supplies the anytime budget dial: stopping the
iteration early yields a usable distribution, so equilibrium refinement is
compute-adaptive. QRE makes $\tau$ and $\beta$ rationality parameters with
meaning rather than tuned constants.

## Why it can beat a baseline rather than tie

Uniform averaging is the degenerate case of this system at fixed equal weights;
routing is the degenerate case at a one-hot weight vector. The equilibrium
strictly generalises both and adapts per token, which is why F27's one-shot
auction already beat a domain-correct oracle router: aggregation can take a
token from a non-specialist when the specialist is locally unsure. Solving for
the equilibrium rather than taking one bid is the headroom above that result.

The cost argument decides feasibility. After one forward pass per player, the
iterations are softmax and dot products over the vocabulary — microseconds
against the milliseconds of a transformer forward. The system therefore runs at
ensemble cost while computing something an ensemble cannot express.

## Consequences

The council becomes the substrate; equilibrium decoding is the architecture.
Teachers are still built and distilled per domain, but their purpose is to be
good *players*, which is a different objective from being good stand-alone
models: a player earns its place by being decisive where others are unsure.
That reframing is what the next round of teacher construction optimises for.
