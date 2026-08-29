import { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "API Documentation",
};

const API_BASE = "https://kinetic.kinetic-ai.workers.dev";

export default function ApiPage() {
  return (
    <div className="page wrap">
      <section className="landing-hero">
        <div>
          <p className="eyebrow">API Reference</p>
          <h1>Kinetic AI OpenAI-Compatible API</h1>
          <p className="lede">
            The Kinetic AI API exposes EqLM and the council of Qwen2.5 specialists. All endpoints
            support the OpenAI-compatible chat completion format, plus optional Kinetic controls
            (rationality λ, solver budget, magnet strength). Auth required for all endpoints.
          </p>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Authentication</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          Include an <code>Authorization</code> header with a bearer token:
        </p>
        <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`Authorization: Bearer GATEWAY_SECRET

# Export as env var for curl:
export KINETIC_AUTH="Bearer $(echo -n $GATEWAY_SECRET)"
curl -H "$KINETIC_AUTH" https://kinetic.kinetic-ai.workers.dev/api/generate`}
        </pre>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Core Endpoints</h2>

        <div style={{ marginTop: "var(--space-5)" }}>
          <h3>/api/eqlm/generate — Single-Model Generation</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            Generate text with EqLM at a specified anytime depth. The model performs adaptive
            fixed-point solving up to the requested budget.
          </p>
          <div className="params-table">
            <strong>GET Parameters:</strong>
            <ul style={{ marginTop: "var(--space-2)" }}>
              <li><code>prompt</code> (string, required): Input text to generate from.</li>
              <li><code>depth</code> (int, default 12): Solver budget (1–12 iterations). Controls quality/speed tradeoff.</li>
              <li><code>max_new_tokens</code> (int, default 48, max 48): Maximum tokens to generate.</li>
              <li><code>device</code> (string, default &quot;auto&quot;): &quot;auto&quot; (GPU if available), &quot;cpu&quot; (CPU-only).</li>
            </ul>
          </div>
          <strong>Response:</strong>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`{
  "status": "ok" | "model_not_loaded" | "error",
  "text": "generated text",
  "tokens_generated": 12,
  "depth_used": 12,
  "mean_solver_iters": 11.3,
  "error": null | "error message"
}`}
          </pre>
          <strong>Example:</strong>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`curl -H "$KINETIC_AUTH" \\
  "${API_BASE}/api/eqlm/generate?prompt=Hello%20world&depth=8&max_new_tokens=32"

# Response:
{
  "status": "ok",
  "text": "This is a test response.",
  "tokens_generated": 4,
  "depth_used": 8,
  "mean_solver_iters": 8.0
}`}
          </pre>
        </div>

        <div style={{ marginTop: "var(--space-8)" }}>
          <h3>/api/eqlm/results — Architecture Results</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            Get EqLM single-model paradigm results (F24: parity at matched params). Includes arm configurations,
            BLiMP accuracy, loss, config hashes, and pre-registration status.
          </p>
          <strong>GET (no parameters)</strong>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`curl -H "$KINETIC_AUTH" "${API_BASE}/api/eqlm/results"

# Response includes:
{
  "paradigm_claim": "EqLM: depth, training, decoding are equilibrium computations",
  "finding": "F24: parity ratio 0.991 at 121M",
  "arms": [
    {
      "seed": 42,
      "arm": "B1",
      "kind": "anytime",
      "num_params": 120696016,
      "blimp_accuracy": 0.662,
      "final_loss": 2.800,
      "config_hash": "..."
    }
  ]
}`}
          </pre>
        </div>

        <div style={{ marginTop: "var(--space-8)" }}>
          <h3>/health — System Health Check</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            Check API availability and GPU status (no auth required).
          </p>
          <strong>GET (no parameters)</strong>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`curl "${API_BASE}/health"

# Response:
{
  "status": "ok",
  "version": "0.1.0-phase3",
  "gpu_available": true
}`}
          </pre>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Research Endpoints (Admin Auth)</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          These endpoints are available to authenticated users and expose research data.
        </p>

        <div style={{ marginTop: "var(--space-5)" }}>
          <h3>/api/leaderboard — Benchmark Ladder</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            Get the full benchmark ladder (Qwen2.5 variants, MMLU, ARC-Challenge, HellaSwag, GSM8K).
            All results trace to config hashes, seeds, and lm-eval invocations.
          </p>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`curl -H "$KINETIC_AUTH" "${API_BASE}/api/leaderboard"

# Returns model scores across benchmarks with full provenance`}
          </pre>
        </div>

        <div style={{ marginTop: "var(--space-5)" }}>
          <h3>/api/auction/traces — Token Auction Traces</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            Real bid/winner/payment traces from the token auction (F22). Per-token specialist selection
            at scoring time. Query by seed or get a sample.
          </p>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`curl -H "$KINETIC_AUTH" "${API_BASE}/api/auction/traces"
curl -H "$KINETIC_AUTH" "${API_BASE}/api/auction/traces/seed42"

# Returns auction traces with bids, winners, payments per token`}
          </pre>
        </div>

        <div style={{ marginTop: "var(--space-5)" }}>
          <h3>/api/solve — Equilibrium Lab Solver</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            Solve matrix games with MMD (Magnetic Mirror Descent) or GDA (Gradient Descent Ascent).
            Returns trajectory and final strategies.
          </p>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`curl -X POST -H "$KINETIC_AUTH" -H "Content-Type: application/json" \\
  -d '{
    "game": "matching_pennies",
    "method": "mmd_fixed",
    "lr": 0.1,
    "tau": 0.1,
    "steps": 100,
    "seed": 42
  }' "${API_BASE}/api/solve"

# Returns trajectory with strategies and NashConv per step`}
          </pre>
        </div>

        <div style={{ marginTop: "var(--space-5)" }}>
          <h3>/api/qre_path — QRE Homotopy Path</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            Trace the quantal response equilibrium (QRE) path as rationality λ varies.
            Smooth interpolation from uniform to Nash.
          </p>
          <pre className="code-block" style={{ marginTop: "var(--space-3)" }}>
{`curl -X POST -H "$KINETIC_AUTH" -H "Content-Type: application/json" \\
  -d '{
    "game": "rps",
    "lambda_min": 0.1,
    "lambda_max": 10.0,
    "n_points": 20
  }' "${API_BASE}/api/qre_path"

# Returns path of (λ, strategy_1, strategy_2) pairs`}
          </pre>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Error Handling</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          All endpoints return standard HTTP status codes and JSON error responses:
        </p>
        <ul style={{ marginTop: "var(--space-3)" }}>
          <li><strong>200 OK</strong>: Request succeeded.</li>
          <li><strong>400 Bad Request</strong>: Invalid parameters.</li>
          <li><strong>401 Unauthorized</strong>: Missing or invalid Authorization header.</li>
          <li><strong>422 Unprocessable Entity</strong>: Invalid game or method name.</li>
          <li><strong>500 Internal Server Error</strong>: Server error (check /health).</li>
        </ul>
        <p style={{ marginTop: "var(--space-3)", color: "var(--text-secondary)" }}>
          Error responses include a <code>detail</code> field:
        </p>
        <pre className="code-block" style={{ marginTop: "var(--space-2)" }}>
{`{"detail": "Unknown game: invalid_game"}`}
        </pre>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Rate Limits & Quotas</h2>
        <ul style={{ color: "var(--text-secondary)" }}>
          <li>Generation endpoints: GPU bandwidth limited (see /health for availability).</li>
          <li>Solver endpoints: CPU-based, no strict limit; large steps ({">"}5000) clamped.</li>
          <li>Batch requests: One job at a time (GPU lock in research/memory/state.json).</li>
        </ul>
      </section>

      <section style={{ marginTop: "var(--space-8)", marginBottom: "var(--space-8)" }}>
        <h2>Quick Start</h2>
        <pre className="code-block">
{`#!/bin/bash
export KINETIC_AUTH="Bearer $GATEWAY_SECRET"

# Test API
curl "$API_BASE/health"

# Generate text
curl -H "$KINETIC_AUTH" \\
  "${API_BASE}/api/eqlm/generate?prompt=Hello&depth=8"

# Solve a game
curl -X POST -H "$KINETIC_AUTH" -H "Content-Type: application/json" \\
  -d '{"game":"rps","method":"mmd_fixed","lr":0.1,"tau":0.1,"steps":100}' \\
  "${API_BASE}/api/solve"`}
        </pre>
        <Link href="/demo" className="btn" style={{ marginTop: "var(--space-5)" }}>
          Try the Interactive Demo
        </Link>
      </section>
    </div>
  );
}
