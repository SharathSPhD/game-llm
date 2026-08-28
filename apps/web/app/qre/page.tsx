"use client";

import { useState } from "react";
import { DemoBadge } from "../components/DemoBadge";
import { QREPathResponse } from "@/lib/replay-data";
import { Loader2 } from "lucide-react";

export default function QREPage() {
  const [game, setGame] = useState("rps");
  const [lambdaMin, setLambdaMin] = useState(0.1);
  const [lambdaMax, setLambdaMax] = useState(10.0);
  const [nPoints, setNPoints] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QREPathResponse | null>(null);
  const [replay, setReplay] = useState(false);

  const handleTrace = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/proxy/api/qre_path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          game,
          lambda_min: lambdaMin,
          lambda_max: lambdaMax,
          n_points: nPoints,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = (await response.json()) as QREPathResponse & { replay?: boolean };
      setResult(data);
      setReplay(!!data.replay);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="wrap">
      <section>
        <h1>QRE Homotopy Explorer</h1>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Trace quantal response equilibria as rationality (λ) varies from high entropy (low λ)
          to near-best-response (high λ). Measure exploitability and strategy smoothness along the path.
        </p>
      </section>

      <div style={{ marginTop: "2rem", display: "grid", gridTemplateColumns: "1fr 2fr", gap: "2rem" }}>
        {/* Controls */}
        <div className="card">
          <h3>QRE Configuration</h3>

          <div style={{ marginTop: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div>
              <label>Game</label>
              <select value={game} onChange={(e) => setGame(e.target.value)}>
                <option value="rps">Rock-Paper-Scissors</option>
                <option value="matching_pennies">Matching Pennies</option>
                <option value="biased_rps">Biased RPS</option>
                <option value="kuhn">Kuhn Poker</option>
              </select>
            </div>

            <div>
              <label>
                λ Min (log scale): <code>{lambdaMin.toFixed(2)}</code>
              </label>
              <input
                type="range"
                min="0.01"
                max="10"
                step="0.1"
                value={lambdaMin}
                onChange={(e) => setLambdaMin(parseFloat(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <div>
              <label>
                λ Max (log scale): <code>{lambdaMax.toFixed(2)}</code>
              </label>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={lambdaMax}
                onChange={(e) => setLambdaMax(parseFloat(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <div>
              <label>
                Path Points: <code>{nPoints}</code>
              </label>
              <input
                type="range"
                min="5"
                max="50"
                step="1"
                value={nPoints}
                onChange={(e) => setNPoints(parseInt(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <button
              className="btn"
              onClick={handleTrace}
              disabled={loading}
              style={{
                marginTop: "1rem",
                justifyContent: "center",
                opacity: loading ? 0.6 : 1,
              }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} style={{ animation: "spin 0.8s linear infinite" }} />
                  Tracing path...
                </>
              ) : (
                "Trace QRE Path"
              )}
            </button>

            {error && (
              <div
                style={{
                  backgroundColor: "rgba(248, 113, 113, 0.1)",
                  border: "1px solid var(--error)",
                  color: "var(--error)",
                  padding: "0.75rem",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem",
                }}
              >
                {error}
              </div>
            )}

            {replay && (
              <div
                style={{
                  backgroundColor: "rgba(250, 204, 21, 0.1)",
                  border: "1px solid var(--warning)",
                  color: "var(--warning)",
                  padding: "0.75rem",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem",
                }}
              >
                <strong>Demo mode:</strong> Sample replay data — sign in for live runs
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        <div>
          {result ? (
            <div className="card">
              <h3>QRE Path Results</h3>
              {(result as { replay?: boolean }).replay && <DemoBadge />}

              <div style={{ marginTop: "1.5rem" }}>
                <h4>Exploitability vs. Rationality</h4>
                <svg
                  width="100%"
                  height="200"
                  style={{
                    border: "1px solid var(--border)",
                    borderRadius: "0.375rem",
                    backgroundColor: "var(--bg-tertiary)",
                  }}
                  viewBox="0 0 400 200"
                >
                  {/* Axes */}
                  <line
                    x1="30"
                    y1="170"
                    x2="390"
                    y2="170"
                    stroke="var(--border)"
                    strokeWidth="1"
                  />
                  <line
                    x1="30"
                    y1="10"
                    x2="30"
                    y2="170"
                    stroke="var(--border)"
                    strokeWidth="1"
                  />

                  {/* Data line */}
                  <polyline
                    points={result.path
                      .map((point, i) => {
                        const x = 30 + (i / (result.path.length - 1)) * 360;
                        const y = 170 - (point.nash_conv / 5) * 160;
                        return `${x},${y}`;
                      })
                      .join(" ")}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />

                  {/* Labels */}
                  <text
                    x="5"
                    y="15"
                    fontSize="10"
                    fill="var(--text-tertiary)"
                  >
                    NC
                  </text>
                  <text
                    x="360"
                    y="185"
                    fontSize="10"
                    fill="var(--text-tertiary)"
                    textAnchor="end"
                  >
                    λ
                  </text>
                </svg>
              </div>

              <div style={{ marginTop: "1.5rem" }}>
                <h4>Path Data</h4>
                <div
                  style={{
                    maxHeight: "200px",
                    overflowY: "auto",
                    fontSize: "0.75rem",
                  }}
                >
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                    }}
                  >
                    <thead>
                      <tr
                        style={{
                          borderBottom: "1px solid var(--border)",
                          backgroundColor: "var(--bg-tertiary)",
                        }}
                      >
                        <th style={{ padding: "0.5rem", textAlign: "left" }}>λ</th>
                        <th style={{ padding: "0.5rem", textAlign: "left" }}>NashConv</th>
                        <th style={{ padding: "0.5rem", textAlign: "left" }}>σ₁</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.path.map((point, i) => (
                        <tr
                          key={i}
                          style={{
                            borderBottom: "1px solid var(--border-subtle)",
                          }}
                        >
                          <td style={{ padding: "0.5rem" }}>
                            {point.rationality.toFixed(2)}
                          </td>
                          <td style={{ padding: "0.5rem" }}>
                            {point.nash_conv.toExponential(2)}
                          </td>
                          <td style={{ padding: "0.5rem" }}>
                            [
                            {point.strategy_1
                              .slice(0, 3)
                              .map((v) => v.toFixed(2))
                              .join(", ")}
                            ]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  minHeight: "200px",
                  gap: "1rem",
                  color: "var(--text-tertiary)",
                }}
              >
                <svg width="60" height="60" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M3 12c0-4.97 4.03-9 9-9s9 4.03 9 9-4.03 9-9 9-9-4.03-9-9z"
                    stroke="var(--border)"
                    strokeWidth="1.6"
                  />
                  <path
                    d="M12 7v5h4"
                    stroke="var(--border)"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                  />
                </svg>
                <p>Configure range and click &quot;Trace QRE Path&quot; to see homotopy</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
