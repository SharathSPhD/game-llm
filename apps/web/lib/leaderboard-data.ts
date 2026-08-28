import fs from "fs";
import path from "path";

export interface BenchmarkResult {
  model_name: string;
  size_b: number;
  source: string;
  mmlu_acc: number | null;
  arc_challenge_acc: number | null;
  hellaswag_acc: number | null;
  gsm8k_flexible: number | null;
  mixed_arena: number | null;
  config_sha: string;
  git_commit: string;
  machine: string;
}

/**
 * Load baseline ladder data from results/scale/ladder/.
 * Returns processed benchmark results for display.
 */
export async function getLeaderboardData(): Promise<BenchmarkResult[]> {
  const results: BenchmarkResult[] = [];

  // Define the models to load with their metadata
  const models = [
    {
      dir: "Qwen_Qwen2.5-1.5B-Instruct",
      name: "Qwen2.5-1.5B-Instruct",
      size: 1.5,
      isBaseline: true,
    },
    {
      dir: "Qwen_Qwen2.5-Coder-1.5B-Instruct",
      name: "Qwen2.5-Coder-1.5B-Instruct",
      size: 1.5,
      isBaseline: false,
    },
    {
      dir: "Qwen_Qwen2.5-Math-1.5B-Instruct",
      name: "Qwen2.5-Math-1.5B-Instruct",
      size: 1.5,
      isBaseline: false,
    },
    {
      dir: "Qwen_Qwen3-1.7B",
      name: "Qwen3-1.7B",
      size: 1.7,
      isBaseline: false,
    },
  ];

  const basePath = path.join(
    process.cwd(),
    "..",
    "..",
    "results",
    "scale",
    "ladder"
  );

  for (const model of models) {
    const modelPath = path.join(basePath, model.dir);

    if (!fs.existsSync(modelPath)) {
      continue; // Skip if directory doesn't exist
    }

    const resultData = loadModelResults(modelPath, model.name, model.size);
    if (resultData) {
      results.push(resultData);
    }
  }

  // Also load GSM8K fixed results if available
  const gsm8kBasePath = path.join(
    process.cwd(),
    "..",
    "..",
    "results",
    "scale",
    "gsm8k_fixed"
  );
  if (fs.existsSync(gsm8kBasePath)) {
    // Augment results with GSM8K data
    augmentWithGSM8K(results, gsm8kBasePath);
  }

  return results;
}

/**
 * Load results from a single model directory.
 */
function loadModelResults(
  modelPath: string,
  modelName: string,
  size: number
): BenchmarkResult | null {
  // Find the results JSON file
  const dirs = fs.readdirSync(modelPath);
  let resultsFile: string | null = null;

  for (const dir of dirs) {
    const fullPath = path.join(modelPath, dir);
    if (fs.statSync(fullPath).isDirectory()) {
      const files = fs.readdirSync(fullPath);
      const jsonFile = files.find((f) => f.startsWith("results_"));
      if (jsonFile) {
        resultsFile = path.join(fullPath, jsonFile);
        break;
      }
    }
  }

  if (!resultsFile || !fs.existsSync(resultsFile)) {
    return null;
  }

  const rawData = JSON.parse(fs.readFileSync(resultsFile, "utf-8"));
  return parseResults(rawData, modelName, size);
}

/**
 * Parse lm-eval results JSON to extract key metrics.
 */
function parseResults(
  rawData: any,
  modelName: string,
  size: number
): BenchmarkResult {
  const results = rawData.results || {};

  // Extract MMLU (average across all MMLU subtasks)
  let mmluScores: number[] = [];
  for (const [key, value] of Object.entries(results)) {
    if (key.startsWith("mmlu_")) {
      const acc = (value as any)["acc,none"];
      if (typeof acc === "number") {
        mmluScores.push(acc);
      }
    }
  }
  const mmluAcc =
    mmluScores.length > 0
      ? mmluScores.reduce((a, b) => a + b, 0) / mmluScores.length
      : null;

  // Extract ARC-Challenge
  const arcAcc = (results.arc_challenge as any)?.["acc,none"] ?? null;

  // Extract HellaSwag
  const hellaswagAcc = (results.hellaswag as any)?.["acc,none"] ?? null;

  // Extract GSM8K (flexible)
  const gsm8kAcc = (results.gsm8k as any)?.["exact_match,flexible-extract"] ?? null;

  // Calculate mixed arena (50% MMLU + 50% GSM8K) if both available
  const mixedArena =
    mmluAcc && gsm8kAcc ? (mmluAcc + gsm8kAcc) / 2 : null;

  // Extract metadata
  const configSha = (rawData.config?.sha || rawData.git_hash || "unknown").slice(
    0,
    8
  );
  const gitCommit = (rawData.git_hash || "unknown").slice(0, 8);

  return {
    model_name: modelName,
    size_b: size,
    source: "Alibaba Qwen",
    mmlu_acc: mmluAcc,
    arc_challenge_acc: arcAcc,
    hellaswag_acc: hellaswagAcc,
    gsm8k_flexible: gsm8kAcc,
    mixed_arena: mixedArena,
    config_sha: configSha,
    git_commit: gitCommit,
    machine: "GB10",
  };
}

/**
 * Augment results with GSM8K fixed data where available.
 */
function augmentWithGSM8K(results: BenchmarkResult[], basePath: string): void {
  const models = [
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-Coder-1.5B-Instruct",
    "Qwen_Qwen2.5-Math-1.5B-Instruct",
    "Qwen_Qwen3-1.7B",
  ];

  for (const model of models) {
    const modelPath = path.join(basePath, model);
    if (!fs.existsSync(modelPath)) {
      continue;
    }

    const dirs = fs.readdirSync(modelPath);
    for (const dir of dirs) {
      const fullPath = path.join(modelPath, dir);
      if (fs.statSync(fullPath).isDirectory()) {
        const files = fs.readdirSync(fullPath);
        const jsonFile = files.find((f) => f.startsWith("results_"));
        if (jsonFile) {
          const resultsFile = path.join(fullPath, jsonFile);
          const rawData = JSON.parse(fs.readFileSync(resultsFile, "utf-8"));
          const gsm8kAcc = (rawData.results?.gsm8k as any)?.["exact_match,flexible-extract"] ?? null;

          if (gsm8kAcc !== null) {
            const result = results.find((r) =>
              r.model_name.includes(model.split("_")[1])
            );
            if (result) {
              result.gsm8k_flexible = gsm8kAcc;
              result.mixed_arena =
                result.mmlu_acc && gsm8kAcc
                  ? (result.mmlu_acc + gsm8kAcc) / 2
                  : null;
            }
          }
        }
      }
    }
  }
}
