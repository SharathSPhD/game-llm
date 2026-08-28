"use client";

import { useState } from "react";
import { DemoBadge } from "../../components/DemoBadge";
import { SolveResponse, TrajectoryPoint } from "@/lib/replay-data";
import { Loader2 } from "lucide-react";

export function EquilibriumLab() {
  const [game, setGame] = useState("rps");
  const [method, setMethod] = useState("mmd_fixed");
  const [lr, setLr] = useState(0.1);
  const [tau, setTau] = useState(0.1);
  const [steps, setSteps] = useState(500);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [replay, setReplay] = useState(false);

  const handleSolve = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/proxy/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          game,
          method,
          lr,
          tau,
          steps,
          seed: 42,
        }),
      });

      if (!response.ok) {
        throw new Error(
          response.status === 401
            ? "Sign in required: live solver runs execute on the research backend. Use the Sign in page."
            : `HTTP ${response.status}`
        );
      }

      const data = (await response.json()) as SolveResponse & { replay?: boolean };
      setResult(data);
      setReplay(!!data.replay);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: "2rem" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "2rem" }}>
        {/* Controls */}
        <div className="card">
          <h3>Solver Configuration</h3>

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
              <label>Method</label>
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                <option value="mmd_fixed">MMD Fixed Reference</option>
                <option value="mmd_rnd">MMD Random Reference</option>
                <option value="gda">Gradient Descent Ascent</option>
              </select>
            </div>

            <div>
              <label>
                Learning Rate: <code>{lr.toFixed(3)}</code>
              </label>
              <input
                type="range"
                min="0.001"
                max="1"
                step="0.01"
                value={lr}
                onChange={(e) => setLr(parseFloat(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <div>
              <label>
                Magnetic Strength (τ): <code>{tau.toFixed(3)}</code>
              </label>
              <input
                type="range"
                min="0.001"
                max="0.5"
                step="0.01"
                value={tau}
                onChange={(e) => setTau(parseFloat(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <div>
              <label>
                Steps: <code>{steps}</code>
              </label>
              <input
                type="range"
                min="10"
                max="5000"
                step="50"
                value={steps}
                onChange={(e) => setSteps(parseInt(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <button
              className="btn"
              onClick={handleSolve}
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
                  Running solver...
                </>
              ) : (
                "Run Solver"
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
                <strong>Demo mode:</strong> Using replay data (gateway offline)
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        <div>
          {result ? (
            <div className="card">
              <h3>Convergence Results</h3>
              {(result as { replay?: boolean }).replay && <DemoBadge />}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(2, 1fr)",
                  gap: "1rem",
                  marginTop: "1.5rem",
                }}
              >
                <div className="panel">
                  <div className="panel-label">Game</div>
                  <div className="reading">{result.game.toUpperCase()}</div>
                </div>

                <div className="panel">
                  <div className="panel-label">Method</div>
                  <div className="reading">{result.method.toUpperCase()}</div>
                </div>

                <div className="panel">
                  <div className="panel-label">Steps Run</div>
                  <div className="reading">{result.steps_run}</div>
                </div>

                <div className="panel">
                  <div className="panel-label">Final NashConv</div>
                  <div className="reading" style={{ fontSize: "1.25rem" }}>
                    {result.final_nash_conv.toExponential(2)}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-label">Player 1 Utility</div>
                  <div className="reading">{result.final_utility_1.toFixed(4)}</div>
                </div>

                <div className="panel">
                  <div className="panel-label">Player 2 Utility</div>
                  <div className="reading">{result.final_utility_2.toFixed(4)}</div>
                </div>
              </div>

              <div style={{ marginTop: "1.5rem" }}>
                <h4>Final Strategies</h4>
                <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
                  <p>
                    Player 1: [
                    {result.final_strategy_1.map((v) => v.toFixed(3)).join(", ")}]
                  </p>
                  <p>
                    Player 2: [
                    {result.final_strategy_2.map((v) => v.toFixed(3)).join(", ")}]
                  </p>
                </div>
              </div>

              {/* Simple convergence chart */}
              <div style={{ marginTop: "1.5rem" }}>
                <h4>NashConv Trajectory (log scale)</h4>
                <ConvergenceChart trajectory={result.trajectory} />
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
                  <circle
                    cx="12"
                    cy="12"
                    r="9.2"
                    stroke="var(--border)"
                    strokeWidth="1.6"
                  />
                  <path
                    d="M7 12l5 5 5-7"
                    stroke="var(--border)"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p>Configure solver and click &quot;Run Solver&quot; to see results</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ConvergenceChart({ trajectory }: { trajectory: TrajectoryPoint[] }) {
  if (!trajectory || trajectory.length === 0) {
    return <div>No data</div>;
  }

  // Simple ASCII plot (would use recharts in production)
  const maxStep = trajectory[trajectory.length - 1].step;
  const minNC = Math.min(...trajectory.map((t) => Math.log10(t.nash_conv + 1e-10)));
  const maxNC = Math.max(...trajectory.map((t) => Math.log10(t.nash_conv + 1e-10)));

  const height = 150;
  const width = 100 * (trajectory.length > 50 ? 1 : 2);

  return (
    <svg
      width="100%"
      height={height}
      style={{
        border: "1px solid var(--border)",
        borderRadius: "0.375rem",
        backgroundColor: "var(--bg-tertiary)",
      }}
      viewBox={`0 0 ${width} ${height}`}
    >
      {/* Axes */}
      <line
        x1="30"
        y1={height - 20}
        x2={width - 10}
        y2={height - 20}
        stroke="var(--border)"
        strokeWidth="1"
      />
      <line
        x1="30"
        y1="10"
        x2="30"
        y2={height - 20}
        stroke="var(--border)"
        strokeWidth="1"
      />

      {/* Data line */}
      <polyline
        points={trajectory
          .map((point, i) => {
            const x =
              30 +
              (i / (trajectory.length - 1)) * (width - 40);
            const logNC = Math.log10(point.nash_conv + 1e-10);
            const y =
              height -
              20 -
              ((logNC - minNC) / (maxNC - minNC)) * (height - 30);
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
        log10(NC)
      </text>
      <text
        x={width - 30}
        y={height - 5}
        fontSize="10"
        fill="var(--text-tertiary)"
        textAnchor="end"
      >
        Step
      </text>
    </svg>
  );
}
