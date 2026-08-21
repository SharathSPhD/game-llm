/**
 * Research findings extracted and formatted from research/memory/findings.md
 * Each finding includes claim, evidence, Tarka status, and related artifacts.
 */

export interface Finding {
  id: string;
  title: string;
  category: string;
  claim: string;
  evidence: string;
  tarkaStatus: string;
  deploymentStatus: string;
  keyNumbers?: Record<string, string>;
  artifactPath?: string;
  linkedFinding?: string;
}

export const findings: Finding[] = [
  {
    id: "F1",
    title: "MMD converges linearly to its magnetic fixed point where GDA cycles",
    category: "Convergence Analysis",
    claim:
      "On symmetric zero-sum matrix games, fixed-magnet MMD (uniform anchor, lr=0.1, τ=0.1) exhibits linear (geometric) last-iterate convergence to its fixed point, while simultaneous GDA at the same stepsize cycles, bounded away from equilibrium.",
    evidence:
      "10 md5-distinct seeds, 2000 steps (exp01 iteration 2): log-linear fit on distance-to-fixed-point, last 50%: R²=0.9948 (matching pennies), 0.9015 (RPS). GDA final NashConv: 1.93 [1.90,1.96] (MP), 1.76 [1.62,1.88] (RPS), R²<0.09 (no decay).",
    tarkaStatus: "CONFIRM (independent recomputation)",
    deploymentStatus: "VALIDATED · sign-off pending",
    keyNumbers: {
      "MP R²": "0.9948",
      "RPS R²": "0.9015",
      "GDA final NashConv (MP)": "1.93 [1.90,1.96]",
      "GDA final NashConv (RPS)": "1.76 [1.62,1.88]",
    },
    artifactPath: "results/exp01_mmd_vs_gda/",
  },
  {
    id: "F2",
    title: "Uniform-anchor MMD fixed points ≠ logit-QRE for asymmetric games",
    category: "Mechanism Theory",
    claim:
      "For asymmetric games (biased RPS), MMD-with-uniform-reference dynamics converge deterministically, but to an attractor distinct from the logit-QRE(λ=1/τ); the dynamics exhibit context-dependent attractors (long-run point NashConv≈0.018 vs one-shot computed FP NashConv=0.353). The symmetric-game identity MMD-FP = QRE does not generalize.",
    evidence:
      "exp01 iteration 2 fixed-point verification (numerical, 50k-step lr=1e-5 ground truth) + zero-variance convergence across 10 seeds.",
    tarkaStatus: "CONFIRM_WITH_CORRECTION (wording adopted above). Iteration-1's R²=0.033 'failure' was a metric artifact.",
    deploymentStatus: "VALIDATED · sign-off pending · feeds paper §Convergence Analysis",
    keyNumbers: {
      "Long-run NashConv": "≈0.018",
      "One-shot FP NashConv": "0.353",
    },
    artifactPath: "results/exp01_mmd_vs_gda/",
  },
  {
    id: "F3",
    title: "Regularized Nash Dynamics reach Nash universally with periodic reference resets",
    category: "Convergence Analysis",
    claim:
      "MMD with periodic reference resets converges to Nash (NashConv < 0.05) on symmetric AND asymmetric games.",
    evidence:
      "Final NashConv 8.48e-6 [4.78e-6,1.26e-5] (MP), 1.07e-6 (RPS), 5.21e-5 [2.62e-5,9.08e-5] (biased RPS); mean R² 0.7181–0.9825. exp01, 10 seeds.",
    tarkaStatus: "CONFIRM_WITH_CORRECTION (R² range corrected as stated)",
    deploymentStatus: "VALIDATED · sign-off pending",
    keyNumbers: {
      "MP NashConv": "8.48e-6",
      "RPS NashConv": "1.07e-6",
      "Biased RPS NashConv": "5.21e-5",
      "Mean R² range": "0.7181–0.9825",
    },
    artifactPath: "results/exp01_mmd_vs_gda/",
  },
  {
    id: "F4",
    title: "DEQ peak activation memory is O(1) in effective depth",
    category: "Optimization",
    claim:
      "DEQ implicit block: 0.032±0.000 MB peak activation memory, flat across effective depth (0% variance); explicit stack: linear, slope 0.0168 MB/layer (N=4→0.067 MB … N=32→0.539 MB), like-for-like layers, CPU measurement.",
    evidence: "results/exp03_deq_solvers/ (config sha a0f8f5c0…, 5 seeds).",
    tarkaStatus: "CONFIRM. (GPU-scale measurement for H1's ≤50% claim happens in Tier B.)",
    deploymentStatus: "VALIDATED · sign-off pending",
    keyNumbers: {
      "DEQ peak memory": "0.032±0.000 MB",
      "Explicit stack slope": "0.0168 MB/layer",
      "Explicit N=4": "0.067 MB",
      "Explicit N=32": "0.539 MB",
    },
    artifactPath: "results/exp03_deq_solvers/",
  },
  {
    id: "F5",
    title: "Anderson acceleration beats Picard exactly where theory predicts: stiff fixed points",
    category: "Optimization",
    claim:
      "On contraction maps with controlled spectral radius ρ, Anderson/Picard iteration ratio < 0.95 at ρ=0.999 (0.888 at dim 32; 0.940 at dim 128); no advantage at easy ρ (iteration-1 'miss' explained by problem easiness). Spectral radii empirically verified (max abs error ~2e-4).",
    evidence:
      "results/exp03_deq_solvers/ iteration 2, ρ∈{0.9,0.99,0.999}, 10 seeds.",
    tarkaStatus: "CONFIRM / CONFIRM_WITH_CORRECTION (wording adopted)",
    deploymentStatus: "VALIDATED · sign-off pending",
    keyNumbers: {
      "Ratio at ρ=0.999": "< 0.95",
      "Anderson/Picard ratio (dim 32, ρ=0.999)": "0.888",
      "Anderson/Picard ratio (dim 128, ρ=0.999)": "0.940",
      "Max spectral error": "~2e-4",
    },
    artifactPath: "results/exp03_deq_solvers/",
  },
  {
    id: "F6",
    title: "Second-price token auction is exactly truthful; weighted aggregation is manipulable",
    category: "Mechanism Theory",
    claim:
      "Empirical truthful-bidding regret in second-price auctions is exactly 0.0 (95% CI [0.0,0.0], 16k observations, misreport grid 0.25v–2v). Weighted-aggregation mechanism has positive manipulation gain (mean regret 0.0773 at n=3, 0.0683 at n=5) — documented as non-truthful, matching the Phase-1 finding that its payments are not VCG.",
    evidence:
      "results/exp04_auction_truthfulness/ (config sha 5c458dac…, 10 seeds × 200 auctions × {3,5} agents).",
    tarkaStatus: "CONFIRM (both claims)",
    deploymentStatus: "VALIDATED · sign-off pending",
    keyNumbers: {
      "Second-price regret": "0.0",
      "95% CI": "[0.0,0.0]",
      "Observations": "16k",
      "Weighted aggregation regret (n=3)": "0.0773",
      "Weighted aggregation regret (n=5)": "0.0683",
    },
    artifactPath: "results/exp04_auction_truthfulness/",
  },
  {
    id: "F7",
    title: "Warm-started homotopy accelerates QRE path tracing",
    category: "Optimization",
    claim:
      "Warm-starting each QRE solve from the previous λ's solution reduces total solver iterations vs cold-start: 25.2% on asymmetric 2×2 (5990→4481 avg), 2.6% on biased RPS; degenerate control (matching pennies) flat as expected. Exploitability along the path is NOT globally monotone for these games; path strategy movement is smooth and small (0.015 < 0.05 prereg threshold).",
    evidence:
      "results/exp02_qre_homotopy/ iteration 3, λ∈logspace(0.01,100,50).",
    tarkaStatus: "Not explicitly stated",
    deploymentStatus: "VALIDATED (with two honest partials) · sign-off pending",
    keyNumbers: {
      "Asymmetric 2×2 speedup": "25.2%",
      "Asymmetric 2×2 iters": "5990→4481 avg",
      "Biased RPS speedup": "2.6%",
      "Strategy movement": "0.015 < 0.05",
    },
    artifactPath: "results/exp02_qre_homotopy/",
  },
  {
    id: "F8",
    title: "Undamped logit-QRE fixed-point iteration requires damping beyond small λ",
    category: "Method Finding",
    claim:
      "Plain s←softmax(λAs) diverges for λ‖A‖ moderately large (biased RPS at λ>0.32). Adaptive damped iteration s←(1-γ)s+γ·softmax(λAs) (γ init 1/(1+λ/10), halve on residual increase) converges across λ∈{1,10,100}.",
    evidence:
      "tests/test_qre.py::TestQREHighRationality; kinetic_ai/games/qre.py damped solver (default on, backward compatible). Convergence: 21 iters at λ=1, 700 at λ=10, 42k at λ=100.",
    tarkaStatus: "Not explicitly stated",
    deploymentStatus: "VALIDATED · sign-off pending · exposes future work: Anderson-accelerated QRE solves",
    keyNumbers: {
      "Divergence threshold": "λ > 0.32 (biased RPS)",
      "Iterations at λ=1": "21",
      "Iterations at λ=10": "700",
      "Iterations at λ=100": "42k",
    },
    artifactPath: "kinetic_ai/games/qre.py",
  },
];

/**
 * Get finding by ID
 */
export function getFinding(id: string): Finding | undefined {
  return findings.find((f) => f.id === id);
}

/**
 * Get all findings for a given category
 */
export function getFindingsByCategory(category: string): Finding[] {
  return findings.filter((f) => f.category === category);
}

/**
 * Get all unique categories
 */
export function getCategories(): string[] {
  const cats = new Set<string>();
  findings.forEach((f) => cats.add(f.category));
  return Array.from(cats).sort();
}
