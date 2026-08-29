"use client";

import Link from "next/link";
import { useState } from "react";

export default function DemoPage() {
  const [depth, setDepth] = useState(8);
  const [prompt, setPrompt] = useState("The future of AI is");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<{
    text: string;
    status: string;
    tokens_generated?: number;
    error?: string;
  } | null>(null);

  const handleGenerate = async () => {
    setIsLoading(true);
    setResult(null);

    try {
      // Routed through the server-side proxy, which injects the gateway
      // credential under the tiered-user gate. A NEXT_PUBLIC_* variable ships
      // to every visitor's browser, so no secret may ever be referenced here.
      const response = await fetch(
        `/api/proxy/api/eqlm/generate?prompt=${encodeURIComponent(prompt)}&depth=${depth}&max_new_tokens=48`
      ).catch(() => null);

      if (response?.ok) {
        const data = await response.json();
        setResult(data);
      } else {
        setResult({
          status: "model_not_loaded",
          text: "",
          error: "Model not yet available. ONNX artifact pending.",
        });
      }
    } catch (err) {
      setResult({
        status: "error",
        text: "",
        error: String(err),
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page wrap">
      <section className="landing-hero">
        <div>
          <p className="eyebrow">Interactive Demo</p>
          <h1>EqLM Anytime Inference</h1>
          <p className="lede">
            Adjust the depth slider to control the solver budget. Watch how quality degrades gracefully
            as you reduce iterations — EqLM adapts to any compute budget, from 4 to 12 iterations.
            The in-browser model scaffold awaits the ONNX artifact.
          </p>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Anytime Depth Dial</h2>
        <div className="card" style={{ marginTop: "var(--space-5)" }}>
          <div style={{ marginBottom: "var(--space-5)" }}>
            <label htmlFor="depth-slider" style={{ display: "block", marginBottom: "var(--space-2)" }}>
              <strong>Solver Budget (Iterations): {depth}</strong>
            </label>
            <input
              id="depth-slider"
              type="range"
              min="4"
              max="12"
              step="1"
              value={depth}
              onChange={(e) => setDepth(Number(e.target.value))}
              disabled={isLoading}
              style={{ width: "100%", marginBottom: "var(--space-3)" }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.875rem", color: "var(--text-secondary)" }}>
              <span>Fast (4)</span>
              <span>Balanced (8)</span>
              <span>Quality (12)</span>
            </div>
          </div>

          <div style={{ marginTop: "var(--space-5)", marginBottom: "var(--space-5)" }}>
            <label htmlFor="prompt-input" style={{ display: "block", marginBottom: "var(--space-2)" }}>
              <strong>Prompt</strong>
            </label>
            <textarea
              id="prompt-input"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={isLoading}
              rows={3}
              style={{
                width: "100%",
                padding: "var(--space-2)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-small)",
                fontFamily: "monospace",
                fontSize: "0.875rem",
                backgroundColor: "var(--bg-subtle)",
              }}
              placeholder="Enter a prompt..."
            />
          </div>

          <button
            onClick={handleGenerate}
            disabled={isLoading}
            style={{ marginTop: "var(--space-3)" }}
            className="btn"
            data-primary="true"
          >
            {isLoading ? "Generating..." : "Generate"}
          </button>

          {result && (
            <div style={{ marginTop: "var(--space-5)", padding: "var(--space-3)", backgroundColor: "var(--bg-subtle)", borderRadius: "var(--radius-small)" }}>
              {result.status === "model_not_loaded" ? (
                <div>
                  <p style={{ color: "var(--text-secondary)", marginBottom: "var(--space-2)" }}>
                    <strong>Model Status: Pending</strong>
                  </p>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.875rem" }}>
                    The in-browser ONNX artifact for EqLM is not yet available. Once it lands in public/model/eqlm.onnx,
                    this demo will run inference locally without network calls.
                  </p>
                </div>
              ) : result.status === "ok" ? (
                <div>
                  <p style={{ marginBottom: "var(--space-3)" }}>
                    <strong>Generated Text:</strong>
                  </p>
                  <p style={{ fontStyle: "italic", color: "var(--text-secondary)", marginBottom: "var(--space-3)" }}>
                    {prompt}{result.text}
                  </p>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "var(--space-2)", fontSize: "0.875rem" }}>
                    <div>
                      <span style={{ color: "var(--text-secondary)" }}>Tokens Generated:</span>
                      <br />
                      <strong>{result.tokens_generated}</strong>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-secondary)" }}>Depth Used:</span>
                      <br />
                      <strong>{depth} iterations</strong>
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <p style={{ color: "var(--error-text)" }}>
                    <strong>Error:</strong> {result.error}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>How It Works</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "var(--space-5)", marginTop: "var(--space-5)" }}>
          <div className="card">
            <h3>Anytime Inference</h3>
            <p>
              The depth slider controls the fixed-point solver budget. At depth=4, the model generates
              quickly but with lower quality. At depth=12, quality peaks but costs 3x the compute.
            </p>
          </div>
          <div className="card">
            <h3>Fixed-Point Computation</h3>
            <p>
              Unlike a 12-layer stack that always computes 12 layers, EqLM solves iteratively until convergence
              or the budget runs out. Each iteration refines the token prediction.
            </p>
          </div>
          <div className="card">
            <h3>Adaptive Per-Token Depth</h3>
            <p>
              The model learns to spend few iterations on easy tokens and many on hard ones.
              At the same mean depth, adaptive reaches parity with fixed depth.
            </p>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)", marginBottom: "var(--space-8)" }}>
        <h2>Explore</h2>
        <ul style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          <li><Link href="/benchmarks" className="btn">Benchmarks</Link></li>
          <li><Link href="/api" className="btn">API Docs</Link></li>
          <li><Link href="/findings" className="btn">Findings</Link></li>
          <li><Link href="/" className="btn" data-secondary="true">Home</Link></li>
        </ul>
      </section>
    </div>
  );
}
