"use client";

import { useState, useEffect } from "react";
import { DemoBadge } from "../components/DemoBadge";

interface ModelEntry {
  path: string;
  size_mb: number;
  config: {
    d_model: number | null;
    n_heads: number | null;
    d_ff: number | null;
    vocab_size: number | null;
    map_form: string | null;
  };
  model_class: string;
  params_estimate: number;
  run: {
    config_sha: string;
    git_commit: string;
  };
}

interface TokenInfo {
  token_str: string;
  solver_iters: number | null;
}

interface GenerationResult {
  text: string;
  tokens: TokenInfo[];
  mean_iters: number;
  wall_ms: number;
}

export default function PlaygroundPage() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [selectedModel, setSelectedModel] = useState<string>("");
  const [prompt, setPrompt] = useState(
    "The future of artificial intelligence is"
  );
  const [maxNewTokens, setMaxNewTokens] = useState(16);
  const [warmStart, setWarmStart] = useState(false);
  const [solverBudget, setSolverBudget] = useState(6);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);

  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerationResult | null>(null);

  // Fetch available models
  useEffect(() => {
    async function fetchModels() {
      try {
        const resp = await fetch("/api/proxy/api/models");
        if (!resp.ok) {
          setModelsError(`Failed to fetch models: ${resp.statusText}`);
          return;
        }
        const data = await resp.json();
        setModels(data);
        if (data.length > 0) {
          setSelectedModel(data[0].path);
        }
      } catch (err) {
        setModelsError(`Error fetching models: ${err}`);
      } finally {
        setModelsLoading(false);
      }
    }

    fetchModels();
  }, []);

  const handleGenerate = async () => {
    if (!selectedModel.trim()) {
      setError("Please select a model");
      return;
    }

    if (!prompt.trim()) {
      setError("Please enter a prompt");
      return;
    }

    setGenerating(true);
    setError(null);
    setResult(null);

    try {
      const resp = await fetch("/api/proxy/api/playground/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_path: selectedModel,
          prompt: prompt,
          max_new_tokens: maxNewTokens,
          warm_start: warmStart,
          solver_budget: solverBudget,
          temperature,
          top_k: topK,
        }),
      });

      const data = await resp.json();

      if (!resp.ok) {
        setError(data.detail || "Generation failed");
        return;
      }

      setResult(data as GenerationResult);
    } catch (err) {
      setError(`Error: ${err}`);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="page">
      <h1>Playground</h1>
      <p className="subtitle">
        Generate text with equilibrium language models. Watch solver iterations
        unfold per token.
      </p>

      <div className="container">
        <div className="control-panel">
          <div className="form-group">
            <label htmlFor="model-select">Model:</label>
            {modelsLoading ? (
              <p className="loading">Loading models...</p>
            ) : modelsError ? (
              <p className="error">{modelsError}</p>
            ) : models.length === 0 ? (
              <p className="error">No models available</p>
            ) : (
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={generating}
              >
                {models.map((model) => (
                  <option key={model.path} value={model.path}>
                    {model.path} ({model.model_class}, {(
                      model.params_estimate / 1e6
                    ).toFixed(1)}M)
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="prompt">Prompt (max 500 chars):</label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value.slice(0, 500))}
              disabled={generating}
              rows={3}
              placeholder="Enter your prompt here..."
            />
            <div className="char-count">
              {prompt.length}/500 characters
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="max-tokens">Max New Tokens (≤64):</label>
              <input
                id="max-tokens"
                type="number"
                min="1"
                max="64"
                value={maxNewTokens}
                onChange={(e) =>
                  setMaxNewTokens(
                    Math.min(Math.max(parseInt(e.target.value) || 1, 1), 64)
                  )
                }
                disabled={generating}
              />
            </div>

            <div className="form-group">
              <label htmlFor="solver-budget">Solver Budget (4-64):</label>
              <div className="slider-container">
                <input
                  id="solver-budget"
                  type="range"
                  min="4"
                  max="64"
                  value={solverBudget}
                  onChange={(e) => setSolverBudget(parseInt(e.target.value))}
                  disabled={generating}
                  className="slider"
                />
                <div className="slider-value">{solverBudget}</div>
              </div>

          <div>
            <label>
              Temperature ({temperature.toFixed(2)}):
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </label>
            <div style={{ color: "var(--text-tertiary)", fontSize: "0.75rem" }}>
              0 = greedy (deterministic, loops); 0.8 recommended
            </div>
          </div>

          <div>
            <label>
              Top-k ({topK}):
              <input
                type="range"
                min={0}
                max={200}
                step={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </label>
          </div>
            </div>
          </div>

          <div className="form-group checkbox">
            <label>
              <input
                type="checkbox"
                checked={warmStart}
                onChange={(e) => setWarmStart(e.target.checked)}
                disabled={generating}
              />
              Warm Start (DEQ initialization)
            </label>
          </div>

          <button
            className="btn-generate"
            onClick={handleGenerate}
            disabled={generating || !selectedModel || !prompt}
          >
            {generating ? "Generating..." : "Generate"}
          </button>

          {error && <div className="error-box">{error}</div>}
        </div>

        {result && (
          <div className="output-panel">
            <h2>Output</h2>
            {(result as { replay?: boolean }).replay && <DemoBadge />}

            <div className="stats">
              <div className="stat">
                <span className="stat-label">Mean Solver Iters:</span>
                <span className="stat-value">{result.mean_iters.toFixed(2)}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Generation Time:</span>
                <span className="stat-value">{result.wall_ms.toFixed(1)}ms</span>
              </div>
              <div className="stat">
                <span className="stat-label">Generated Tokens:</span>
                <span className="stat-value">{result.tokens.length}</span>
              </div>
            </div>

            <div className="generated-text">
              <p className="prompt-text">{prompt}</p>
              <span className="generated-span">
                {result.tokens.map((token, idx) => (
                  <span
                    key={idx}
                    className="token"
                    title={
                      token.solver_iters !== null
                        ? `${token.solver_iters} solver iterations`
                        : "Solver iterations not available"
                    }
                    style={{
                      backgroundColor: getTokenColor(
                        token.solver_iters,
                        result.mean_iters
                      ),
                    }}
                  >
                    {token.token_str}
                  </span>
                ))}
              </span>
            </div>

            <div className="token-table">
              <h3>Token Breakdown</h3>
              <table>
                <thead>
                  <tr>
                    <th>Token</th>
                    <th>Solver Iters</th>
                  </tr>
                </thead>
                <tbody>
                  {result.tokens.map((token, idx) => (
                    <tr key={idx}>
                      <td className="mono">{escapeHtml(token.token_str)}</td>
                      <td>
                        {token.solver_iters !== null
                          ? token.solver_iters
                          : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .container {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
          margin-top: 2rem;
        }

        @media (max-width: 1200px) {
          .container {
            grid-template-columns: 1fr;
          }
        }

        .control-panel {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .form-group label {
          font-weight: 500;
          font-size: 0.95rem;
        }

        .form-group select,
        .form-group textarea,
        .form-group input[type="number"] {
          padding: 0.75rem;
          border: 1px solid var(--border-color);
          border-radius: 4px;
          background: var(--bg-input);
          color: var(--text-primary);
          font-family: inherit;
          font-size: 0.95rem;
        }

        .form-group select:disabled,
        .form-group textarea:disabled,
        .form-group input:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .form-group textarea {
          resize: vertical;
          min-height: 100px;
        }

        .char-count {
          font-size: 0.8rem;
          color: var(--text-secondary);
          margin-top: 0.25rem;
        }

        .form-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
        }

        .slider-container {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .slider {
          flex: 1;
          height: 6px;
          border-radius: 3px;
          background: var(--border-color);
          outline: none;
          -webkit-appearance: none;
          appearance: none;
        }

        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: var(--accent);
          cursor: pointer;
          transition: background 0.2s;
        }

        .slider::-moz-range-thumb {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: var(--accent);
          cursor: pointer;
          border: none;
          transition: background 0.2s;
        }

        .slider-value {
          font-weight: 600;
          min-width: 40px;
          text-align: center;
        }

        .form-group.checkbox label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-weight: normal;
        }

        .form-group.checkbox input[type="checkbox"] {
          width: auto;
        }

        .btn-generate {
          padding: 0.75rem 1.5rem;
          background: var(--accent);
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 1rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-generate:hover:not(:disabled) {
          background: var(--accent-hover);
        }

        .btn-generate:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .error-box {
          background: var(--error-bg);
          color: var(--error);
          padding: 1rem;
          border-radius: 4px;
          font-size: 0.9rem;
        }

        .loading,
        .error {
          font-size: 0.9rem;
          color: var(--text-secondary);
        }

        .error {
          color: var(--error);
        }

        .output-panel {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .output-panel h2 {
          margin-top: 0;
        }

        .stats {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1rem;
        }

        .stat {
          background: var(--bg-secondary);
          padding: 1rem;
          border-radius: 4px;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .stat-label {
          font-size: 0.85rem;
          color: var(--text-secondary);
        }

        .stat-value {
          font-size: 1.5rem;
          font-weight: 600;
          color: var(--accent);
        }

        .generated-text {
          background: var(--bg-secondary);
          padding: 1.5rem;
          border-radius: 4px;
          line-height: 1.8;
          word-wrap: break-word;
        }

        .prompt-text {
          color: var(--text-secondary);
          margin: 0 0 0.5rem 0;
        }

        .generated-span {
          display: flex;
          flex-wrap: wrap;
          gap: 0.1rem;
        }

        .token {
          padding: 2px 4px;
          border-radius: 3px;
          cursor: help;
          transition: opacity 0.2s;
        }

        .token:hover {
          opacity: 0.8;
        }

        .token-table {
          overflow-x: auto;
        }

        .token-table h3 {
          margin-top: 0;
          margin-bottom: 1rem;
        }

        .token-table table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.9rem;
        }

        .token-table th,
        .token-table td {
          padding: 0.75rem;
          text-align: left;
          border-bottom: 1px solid var(--border-color);
        }

        .token-table th {
          background-color: var(--bg-secondary);
          font-weight: 600;
          position: sticky;
          top: 0;
        }

        .token-table tbody tr:hover {
          background-color: var(--bg-secondary);
        }

        .mono {
          font-family: monospace;
          word-break: break-all;
        }

        .subtitle {
          color: var(--text-secondary);
          margin-bottom: 1.5rem;
        }
      `}</style>
    </div>
  );
}

function getTokenColor(
  solverIters: number | null,
  meanIters: number
): string {
  if (solverIters === null) return "transparent";

  // Color scale: cool (low) to warm (high)
  // Blue for low iters, red for high iters
  const intensity = Math.min(solverIters / (meanIters * 1.5), 1.0);

  // HSL scale: 240 (blue) to 0 (red)
  const hue = Math.round(240 * (1 - intensity));
  const saturation = Math.round(60 + 40 * intensity);
  const lightness = Math.round(85 - 20 * intensity);

  return `hsla(${hue}, ${saturation}%, ${lightness}%, 0.6)`;
}

function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}
