/**
 * Replay data: realistic canned responses for demo/offline mode.
 * Used when GATEWAY_URL is unset. All responses are flagged replay:true.
 */

export interface TrajectoryPoint {
  step: number;
  nash_conv: number;
  utility_1: number;
  utility_2: number;
  strategy_1: number[];
  strategy_2: number[];
}

export interface SolveResponse {
  game: string;
  method: string;
  steps_run: number;
  trajectory: TrajectoryPoint[];
  final_strategy_1: number[];
  final_strategy_2: number[];
  final_nash_conv: number;
  final_utility_1: number;
  final_utility_2: number;
  replay?: boolean;
}

export interface QREPathPoint {
  rationality: number;
  strategy_1: number[];
  strategy_2: number[];
  nash_conv: number;
}

export interface QREPathResponse {
  game: string;
  lambda_min: number;
  lambda_max: number;
  path: QREPathPoint[];
  replay?: boolean;
}

export interface AuctionResponse {
  winner_id: number;
  output_distribution: number[];
  payments: number[];
  sampled_token: number;
  replay?: boolean;
}

// Realistic trajectory for RPS with MMD_FIXED method (log-linear convergence)
function generateMMDTrajectory(steps: number): TrajectoryPoint[] {
  const trajectory: TrajectoryPoint[] = [];
  const uniform = [1/3, 1/3, 1/3];

  for (let i = 0; i < steps; i++) {
    const t = i / steps;
    // Simulate log-linear convergence to Nash equilibrium
    const nashConv = 2.0 * Math.exp(-3.0 * t) + 0.001; // Decays to near-zero
    const perturbation = 0.05 * (1 - t);

    trajectory.push({
      step: i,
      nash_conv: nashConv,
      utility_1: 0 + t * 0.01,
      utility_2: 0 + t * 0.01,
      strategy_1: [
        uniform[0] + perturbation * Math.sin(t * Math.PI * 2),
        uniform[1] + perturbation * Math.cos(t * Math.PI * 3),
        uniform[2] + perturbation * Math.sin(t * Math.PI * 4),
      ],
      strategy_2: [
        uniform[0] + perturbation * Math.cos(t * Math.PI * 2),
        uniform[1] + perturbation * Math.sin(t * Math.PI * 3),
        uniform[2] + perturbation * Math.cos(t * Math.PI * 4),
      ],
    });
  }

  // Ensure last point is near equilibrium
  trajectory[trajectory.length - 1].nash_conv = 0.0005;
  trajectory[trajectory.length - 1].strategy_1 = [...uniform];
  trajectory[trajectory.length - 1].strategy_2 = [...uniform];

  return trajectory;
}

// Realistic QRE path (smooth strategy evolution)
function generateQREPath(lambdaMin: number, lambdaMax: number, nPoints: number): QREPathPoint[] {
  const path: QREPathPoint[] = [];

  for (let i = 0; i < nPoints; i++) {
    const logRatio = (i / (nPoints - 1));
    const lambda = lambdaMin * Math.pow(lambdaMax / lambdaMin, logRatio);

    // For RPS, QRE moves continuously from uniform toward Nash
    const uniform = [1/3, 1/3, 1/3];
    const perturbAmount = Math.min(lambda / 10, 0.15);

    const s1 = [
      uniform[0] + perturbAmount * 0.05,
      uniform[1] - perturbAmount * 0.025,
      uniform[2] - perturbAmount * 0.025,
    ];

    const nashConv = 5.0 / (1 + lambda * 2);

    path.push({
      rationality: lambda,
      strategy_1: s1,
      strategy_2: [...s1].reverse(),
      nash_conv: nashConv,
    });
  }

  return path;
}

export function getReplaySolveResponse(params: {
  game: string;
  method: string;
  steps: number;
}): SolveResponse {
  const { game, method, steps } = params;
  const trajectory = generateMMDTrajectory(Math.min(steps, 200));
  const final = trajectory[trajectory.length - 1];

  return {
    game,
    method,
    steps_run: steps,
    trajectory,
    final_strategy_1: final.strategy_1,
    final_strategy_2: final.strategy_2,
    final_nash_conv: final.nash_conv,
    final_utility_1: final.utility_1,
    final_utility_2: final.utility_2,
    replay: true,
  };
}

export function getReplayQREPathResponse(params: {
  game: string;
  lambda_min: number;
  lambda_max: number;
  n_points: number;
}): QREPathResponse {
  const { game, lambda_min, lambda_max, n_points } = params;
  const path = generateQREPath(lambda_min, lambda_max, Math.min(n_points, 30));

  return {
    game,
    lambda_min,
    lambda_max,
    path,
    replay: true,
  };
}

export function getReplayAuctionResponse(params: {
  bids: number[];
  agent_distributions: number[][];
  auction_type: string;
}): AuctionResponse {
  const { bids, agent_distributions, auction_type } = params;
  const n_agents = bids.length;
  const vocab_size = agent_distributions[0]?.length || 10;

  // Simple: highest bidder wins in second-price
  let winnerId = 0;
  let maxBid = bids[0];
  for (let i = 1; i < bids.length; i++) {
    if (bids[i] > maxBid) {
      maxBid = bids[i];
      winnerId = i;
    }
  }

  // Payments: second-price is the second-highest bid
  const sortedBids = [...bids].sort((a, b) => b - a);
  const payment = n_agents > 1 ? sortedBids[1] : sortedBids[0];

  const payments = bids.map((_, i) => (i === winnerId ? payment : 0));

  // Output distribution: winner's distribution
  const output = agent_distributions[winnerId] || new Array(vocab_size).fill(1/vocab_size);

  // Sampled token: argmax of output distribution
  const sampledToken = output.indexOf(Math.max(...output));

  return {
    winner_id: winnerId,
    output_distribution: output,
    payments,
    sampled_token: sampledToken,
    replay: true,
  };
}

// ── Demo replay for unauthorized visitors ────────────────────────────────────
// Served (clearly flagged with replay: true) when a visitor is not signed in
// or not tier-enabled. The auction trace sample is REAL data from the F22
// evaluation (exp12 seed 42, first 12 positions), embedded verbatim.

export function getReplayPlaygroundResponse(params: {
  prompt: string;
  max_new_tokens: number;
  solver_budget: number;
}) {
  const words = [
    " the", " equilibrium", " of", " the", " game", " is", " a", " fixed",
    " point", " of", " the", " learning", " dynamics", ".", " Each", " player",
  ];
  const n = Math.max(1, Math.min(params.max_new_tokens, words.length));
  const tokens = words.slice(0, n).map((w) => ({
    token_str: w,
    solver_iters: params.solver_budget,
  }));
  return {
    text: params.prompt + tokens.map((t) => t.token_str).join(""),
    tokens,
    mean_iters: params.solver_budget,
    wall_ms: 42.0,
    replay: true,
  };
}

export function getReplayModelsResponse() {
  return {
    // Shape mirrors app/server.py scan_models_registry(): the models, chat and
    // playground pages read path, model_class, params_estimate, size_mb,
    // config.{d_model,n_heads} and run.{config_sha,git_commit,dir}.
    models: [
      {
        path: "results/scale/exp32/checkpoints/C1_seed42.pt",
        model_class: "EqLM",
        num_params: 45785857,
        params_estimate: 45785857,
        size_mb: 183.2,
        config: { d_model: 768, n_heads: 12, d_ff: 3072, deq_max_iter: 12 },
        run: { config_sha: "485fdc4218a9", git_commit: "unknown", dir: "scale/exp32" },
        label: "kinetic-eqlm-46m-compute-matched (F45; replay — not a live model)",
      },
      {
        path: "results/exp13_seed42/checkpoints/B1_anytime.pt",
        model_class: "EqLM",
        num_params: 120696016,
        params_estimate: 120696016,
        size_mb: 482.8,
        config: { d_model: 1704, n_heads: 12, d_ff: 6816, deq_max_iter: 12 },
        run: { config_sha: "40b471ce0a", git_commit: "unknown", dir: "exp13_seed42" },
        label: "kinetic-eqlm-anytime-121m-babylm (F24; replay — not a live model)",
      },
    ],
    replay: true,
  };
}

const F22_DEMO_TRACES = [
  { position: 0, bids: [0.3934, 0.0771], winner: 0, payment: 0.0771, target_token: 3398 },
  { position: 1, bids: [0.9703, 0.0547], winner: 0, payment: 0.0547, target_token: 40 },
  { position: 2, bids: [0.978, 0.1264], winner: 0, payment: 0.1264, target_token: 25 },
  { position: 3, bids: [0.9748, 0.2044], winner: 0, payment: 0.2044, target_token: 197 },
  { position: 4, bids: [0.1133, 0.0278], winner: 0, payment: 0.0278, target_token: 259 },
  { position: 5, bids: [0.305, 0.2079], winner: 0, payment: 0.2079, target_token: 262 },
  { position: 6, bids: [0.0196, 0.0088], winner: 0, payment: 0.0088, target_token: 2119 },
  { position: 7, bids: [0.6711, 0.1347], winner: 0, payment: 0.1347, target_token: 351 },
  { position: 8, bids: [0.9782, 0.2333], winner: 0, payment: 0.2333, target_token: 257 },
  { position: 9, bids: [0.8769, 0.2156], winner: 0, payment: 0.2156, target_token: 345 },
  { position: 10, bids: [0.0957, 0.0461], winner: 0, payment: 0.0461, target_token: 1849 },
  { position: 11, bids: [0.1884, 0.0711], winner: 0, payment: 0.0711, target_token: 287 },
];

export function getReplayTracesListResponse() {
  return { seeds: [42], replay: true };
}

export function getReplayTracesResponse(seed: number) {
  return {
    seed,
    traces: F22_DEMO_TRACES,
    summary: {
      h4_score: "MET",
      domains: { a: "childes", b: "simple_wiki" },
      perplexity_mixed: { S_A: 234.3, S_B: 1241.8, ENS: 208.3, AUC: 158.5 },
      auction_win_frac_a: 0.598,
    },
    replay: true,
  };
}
