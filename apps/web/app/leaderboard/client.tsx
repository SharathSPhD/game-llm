"use client";

import { useState, useEffect } from "react";

interface BenchmarkRow {
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

export function LeaderboardTableClient() {
  const [benchmarks, setBenchmarks] = useState<BenchmarkRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch("/api/leaderboard");
        if (!res.ok) {
          throw new Error("Failed to fetch leaderboard data");
        }
        const data = await res.json();
        setBenchmarks(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  if (loading) {
    return <p style={{ color: "var(--text-secondary)" }}>Loading leaderboard...</p>;
  }

  if (error) {
    return <p style={{ color: "var(--error)" }}>Error loading leaderboard: {error}</p>;
  }

  if (benchmarks.length === 0) {
    return <p style={{ color: "var(--text-secondary)" }}>No benchmark data available yet.</p>;
  }

  return (
    <div style={{ marginTop: "2rem", marginBottom: "2rem" }}>
      <div
        style={{
          overflowX: "auto",
          border: "1px solid var(--border-color)",
          borderRadius: "4px",
          background: "var(--bg-secondary)",
        }}
      >
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.9rem",
          }}
        >
          <thead
            style={{
              background: "var(--bg-input)",
              position: "sticky",
              top: 0,
            }}
          >
            <tr>
              <th
                style={{
                  padding: "1rem 0.75rem",
                  textAlign: "left",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  borderBottom: "2px solid var(--border-color)",
                  whiteSpace: "nowrap",
                }}
              >
                Model
              </th>
              <th
                style={{
                  padding: "1rem 0.75rem",
                  textAlign: "right",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  borderBottom: "2px solid var(--border-color)",
                  whiteSpace: "nowrap",
                }}
              >
                Size (B)
              </th>
              <th
                style={{
                  padding: "1rem 0.75rem",
                  textAlign: "right",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  borderBottom: "2px solid var(--border-color)",
                  whiteSpace: "nowrap",
                }}
              >
                MMLU
              </th>
              <th
                style={{
                  padding: "1rem 0.75rem",
                  textAlign: "right",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  borderBottom: "2px solid var(--border-color)",
                  whiteSpace: "nowrap",
                }}
              >
                GSM8K
              </th>
              <th
                style={{
                  padding: "1rem 0.75rem",
                  textAlign: "right",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  borderBottom: "2px solid var(--border-color)",
                  whiteSpace: "nowrap",
                }}
              >
                Mixed (50/50)
              </th>
              <th
                style={{
                  padding: "1rem 0.75rem",
                  textAlign: "left",
                  fontWeight: 600,
                  color: "var(--text-primary)",
                  borderBottom: "2px solid var(--border-color)",
                  whiteSpace: "nowrap",
                }}
              >
                Config
              </th>
            </tr>
          </thead>
          <tbody>
            {benchmarks.map((row, idx) => (
              <tr key={idx}>
                <td
                  style={{
                    padding: "0.875rem 0.75rem",
                    borderBottom: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                    fontWeight: 600,
                    minWidth: "220px",
                  }}
                >
                  {row.model_name}
                </td>
                <td
                  style={{
                    padding: "0.875rem 0.75rem",
                    borderBottom: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                    textAlign: "right",
                    fontFamily: "monospace",
                    fontSize: "0.85rem",
                  }}
                >
                  {row.size_b.toFixed(1)}
                </td>
                <td
                  style={{
                    padding: "0.875rem 0.75rem",
                    borderBottom: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                    textAlign: "right",
                    fontFamily: "monospace",
                    fontWeight: 500,
                    minWidth: "80px",
                  }}
                >
                  {row.mmlu_acc !== null ? (row.mmlu_acc * 100).toFixed(1) : "–"}%
                </td>
                <td
                  style={{
                    padding: "0.875rem 0.75rem",
                    borderBottom: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                    textAlign: "right",
                    fontFamily: "monospace",
                    fontWeight: 500,
                    minWidth: "80px",
                  }}
                >
                  {row.gsm8k_flexible !== null
                    ? (row.gsm8k_flexible * 100).toFixed(1)
                    : "–"}
                  %
                </td>
                <td
                  style={{
                    padding: "0.875rem 0.75rem",
                    borderBottom: "1px solid var(--border-color)",
                    color: "var(--text-primary)",
                    textAlign: "right",
                    fontFamily: "monospace",
                    fontWeight: 500,
                    minWidth: "80px",
                  }}
                >
                  {row.mixed_arena !== null
                    ? (row.mixed_arena * 100).toFixed(1)
                    : "–"}
                  %
                </td>
                <td
                  style={{
                    padding: "0.875rem 0.75rem",
                    borderBottom: "1px solid var(--border-color)",
                    color: "var(--text-secondary)",
                    fontSize: "0.8rem",
                    fontFamily: "monospace",
                    minWidth: "100px",
                  }}
                  title={row.config_sha}
                >
                  {row.config_sha}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
