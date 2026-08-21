"use client";

import { useState } from "react";
import { AuctionResponse } from "@/lib/replay-data";
import { Loader2 } from "lucide-react";

interface Agent {
  bid: number;
  distribution: number[];
}

export default function AuctionPage() {
  const [numAgents, setNumAgents] = useState(3);
  const [auctionType, setAuctionType] = useState("second_price");
  const [agents, setAgents] = useState<Agent[]>([
    { bid: 1.0, distribution: [1, 0, 0, 0, 0] },
    { bid: 0.8, distribution: [0, 1, 0, 0, 0] },
    { bid: 0.6, distribution: [0, 0, 1, 0, 0] },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AuctionResponse | null>(null);
  const [replay, setReplay] = useState(false);

  const updateAgent = (idx: number, field: keyof Agent, value: unknown) => {
    const newAgents = [...agents];
    if (field === "bid") {
      newAgents[idx].bid = value as number;
    } else if (field === "distribution") {
      newAgents[idx].distribution = value as number[];
    }
    setAgents(newAgents);
  };

  const addAgent = () => {
    setAgents([
      ...agents,
      { bid: 0.5, distribution: Array(5).fill(0).map((_, i) => (i === agents.length ? 1 : 0)) },
    ]);
    setNumAgents(agents.length + 1);
  };

  const removeAgent = (idx: number) => {
    if (agents.length > 2) {
      setAgents(agents.filter((_, i) => i !== idx));
      setNumAgents(agents.length - 1);
    }
  };

  const handleAuction = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/proxy/api/auction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bids: agents.map((a) => a.bid),
          agent_distributions: agents.map((a) => a.distribution),
          auction_type: auctionType,
          vocab_size: 5,
          seed: 42,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = (await response.json()) as AuctionResponse & { replay?: boolean };
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
        <h1>Token Auction Playground</h1>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Design multi-agent token auctions with configurable bids and output distributions.
          Compare second-price (truthful, VCG-exact) vs. weighted aggregation (measurably manipulable).
          See F6 in <a href="/findings">Findings</a> for truthfulness validation.
        </p>
      </section>

      <div style={{ marginTop: "2rem" }}>
        <div className="card">
          <h3>Auction Configuration</h3>

          <div style={{ marginTop: "1.5rem", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
            {/* Type selector */}
            <div>
              <label>Auction Type</label>
              <select value={auctionType} onChange={(e) => setAuctionType(e.target.value)}>
                <option value="second_price">Second-Price (Truthful)</option>
                <option value="weighted_aggregation">Weighted Aggregation (Manipulable)</option>
              </select>
              <p style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginTop: "0.5rem" }}>
                {auctionType === "second_price"
                  ? "Proven truthful: regret = 0.0 (95% CI [0.0, 0.0])"
                  : "Non-truthful: mean regret ≈ 0.077"}
              </p>
            </div>

            {/* Stats */}
            <div>
              <div className="panel">
                <div className="panel-label">Total Agents</div>
                <div className="reading">{agents.length}</div>
              </div>
            </div>
          </div>

          {/* Agent configuration */}
          <div style={{ marginTop: "2rem" }}>
            <h4>Agent Bids & Output Distributions</h4>
            <div
              style={{
                display: "grid",
                gap: "1rem",
                marginTop: "1rem",
                maxHeight: "300px",
                overflowY: "auto",
              }}
            >
              {agents.map((agent, idx) => (
                <div
                  key={idx}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "0.5fr 1fr 1.5fr auto",
                    gap: "1rem",
                    alignItems: "end",
                    padding: "1rem",
                    backgroundColor: "var(--bg-tertiary)",
                    borderRadius: "0.375rem",
                  }}
                >
                  <div>
                    <label style={{ fontSize: "0.75rem" }}>Bid</label>
                    <input
                      type="number"
                      min="0"
                      max="10"
                      step="0.1"
                      value={agent.bid}
                      onChange={(e) => updateAgent(idx, "bid", parseFloat(e.target.value))}
                    />
                  </div>

                  <div>
                    <label style={{ fontSize: "0.75rem" }}>Distribution (vocab=5)</label>
                    <div style={{ display: "flex", gap: "0.25rem" }}>
                      {agent.distribution.map((v, i) => (
                        <input
                          key={i}
                          type="number"
                          min="0"
                          max="1"
                          step="0.1"
                          value={v}
                          onChange={(e) => {
                            const newDist = [...agent.distribution];
                            newDist[i] = parseFloat(e.target.value);
                            updateAgent(idx, "distribution", newDist);
                          }}
                          style={{ width: "3rem" }}
                        />
                      ))}
                    </div>
                  </div>

                  <div>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-tertiary)" }}>
                      sum = {agent.distribution.reduce((a, b) => a + b, 0).toFixed(2)}
                    </span>
                  </div>

                  <button
                    className="btn"
                    onClick={() => removeAgent(idx)}
                    disabled={agents.length <= 2}
                    style={{
                      padding: "0.5rem 0.75rem",
                      opacity: agents.length <= 2 ? 0.4 : 1,
                    }}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>

            <button
              className="btn"
              onClick={addAgent}
              style={{ marginTop: "1rem" }}
            >
              + Add Agent
            </button>
          </div>

          <button
            className="btn"
            onClick={handleAuction}
            disabled={loading}
            style={{
              marginTop: "1.5rem",
              justifyContent: "center",
              opacity: loading ? 0.6 : 1,
              width: "100%",
            }}
          >
            {loading ? (
              <>
                <Loader2 size={16} style={{ animation: "spin 0.8s linear infinite" }} />
                Running auction...
              </>
            ) : (
              "Run Auction"
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
                marginTop: "1rem",
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
                marginTop: "1rem",
              }}
            >
              <strong>Demo mode:</strong> Using replay data (gateway offline)
            </div>
          )}
        </div>

        {result && (
          <div className="card" style={{ marginTop: "2rem" }}>
            <h3>Auction Results</h3>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: "1rem",
                marginTop: "1.5rem",
              }}
            >
              <div className="panel">
                <div className="panel-label">Winner</div>
                <div className="reading">Agent {result.winner_id}</div>
              </div>

              <div className="panel">
                <div className="panel-label">Sampled Token</div>
                <div className="reading">{result.sampled_token}</div>
              </div>
            </div>

            <div style={{ marginTop: "1.5rem" }}>
              <h4>Payments (truthfulness validation)</h4>
              <div style={{ fontSize: "0.875rem" }}>
                {result.payments.map((payment, i) => (
                  <div key={i} style={{ marginBottom: "0.5rem" }}>
                    <span style={{ color: "var(--text-secondary)" }}>Agent {i}:</span> {payment.toFixed(4)}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ marginTop: "1.5rem" }}>
              <h4>Output Distribution (winner&apos;s choice)</h4>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(5, 1fr)",
                  gap: "0.5rem",
                }}
              >
                {result.output_distribution.map((prob, i) => (
                  <div
                    key={i}
                    style={{
                      backgroundColor: "var(--bg-tertiary)",
                      padding: "0.75rem",
                      borderRadius: "0.375rem",
                      textAlign: "center",
                      fontSize: "0.875rem",
                    }}
                  >
                    <div style={{ color: "var(--text-tertiary)", fontSize: "0.75rem" }}>Token {i}</div>
                    <div style={{ fontWeight: 600 }}>{prob.toFixed(3)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
