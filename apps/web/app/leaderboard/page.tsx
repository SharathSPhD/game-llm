import benchmarks from "@/data/benchmarks.json";

export const metadata = {
  title: "Leaderboard — EqLM",
  description: "Benchmark leaderboard comparing EqLM models against open-weight baselines.",
};

export default function LeaderboardPage() {
  return (
    <div className="page">
      <h1>Benchmark Leaderboard</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
        Comparing EqLM models against open-weight baselines on standard benchmarks.
        All evaluations use 0-shot, 300-sample limit on the lm-eval harness.
      </p>

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
                    textAlign: "left",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    borderBottom: "2px solid var(--border-color)",
                    whiteSpace: "nowrap",
                  }}
                >
                  Size
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
                  Source
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
                  ARC-C (acc)
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
                  ARC-C (norm)
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
                  HellaSwag (acc)
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
                  HellaSwag (norm)
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
                  GSM8K (flex)
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
                  Harness
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
                  Machine
                </th>
                <th
                  style={{
                    padding: "1rem 0.75rem",
                    textAlign: "center",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                    borderBottom: "2px solid var(--border-color)",
                    whiteSpace: "nowrap",
                  }}
                >
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.benchmarks.map((row) => (
                <tr
                  key={row.id}
                  style={{
                    opacity: row.status === "pending" ? 0.75 : 1,
                  }}
                >
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
                    {row.size_b.toFixed(2)}B
                  </td>
                  <td
                    style={{
                      padding: "0.875rem 0.75rem",
                      borderBottom: "1px solid var(--border-color)",
                      color: "var(--text-secondary)",
                      fontSize: "0.85rem",
                      minWidth: "100px",
                    }}
                  >
                    {row.source}
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
                    {row.arc_challenge_acc !== null
                      ? (row.arc_challenge_acc * 100).toFixed(2)
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
                    {row.arc_challenge_acc_norm !== null
                      ? (row.arc_challenge_acc_norm * 100).toFixed(2)
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
                    {row.hellaswag_acc !== null
                      ? (row.hellaswag_acc * 100).toFixed(2)
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
                    {row.hellaswag_acc_norm !== null
                      ? (row.hellaswag_acc_norm * 100).toFixed(2)
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
                    {row.gsm8k_flexible !== null
                      ? (row.gsm8k_flexible * 100).toFixed(2)
                      : "–"}
                    %
                  </td>
                  <td
                    style={{
                      padding: "0.875rem 0.75rem",
                      borderBottom: "1px solid var(--border-color)",
                      color: "var(--text-secondary)",
                      fontSize: "0.85rem",
                      minWidth: "100px",
                    }}
                  >
                    {row.harness}
                  </td>
                  <td
                    style={{
                      padding: "0.875rem 0.75rem",
                      borderBottom: "1px solid var(--border-color)",
                      color: "var(--text-secondary)",
                      fontSize: "0.85rem",
                      minWidth: "100px",
                    }}
                  >
                    {row.machine}
                  </td>
                  <td
                    style={{
                      padding: "0.875rem 0.75rem",
                      borderBottom: "1px solid var(--border-color)",
                      color: "var(--text-primary)",
                      textAlign: "center",
                      minWidth: "90px",
                    }}
                  >
                    <span
                      style={{
                        display: "inline-block",
                        padding: "0.35rem 0.75rem",
                        borderRadius: "3px",
                        fontSize: "0.75rem",
                        fontWeight: 600,
                        textTransform: "uppercase",
                        letterSpacing: "0.5px",
                        background: row.status === "baseline" ? "var(--accent)" : "var(--bg-input)",
                        color: row.status === "baseline" ? "white" : "var(--text-secondary)",
                        border: row.status === "baseline" ? "none" : "1px solid var(--border-color)",
                      }}
                    >
                      {row.status === "baseline" ? "MEASURED" : "PENDING"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <section style={{ marginTop: "var(--space-8)", marginBottom: "var(--space-5)" }}>
        <h2>Benchmark Details</h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "var(--space-4)",
            marginTop: "var(--space-4)",
          }}
        >
          <div className="card">
            <h3>ARC-Challenge</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              25 challenging multiple-choice science questions. Measures reasoning and
              knowledge. Accuracy and normalized variants measured.
            </p>
          </div>

          <div className="card">
            <h3>HellaSwag</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              10k multiple-choice commonsense inference tasks from movie descriptions.
              Accuracy and normalized variants measured.
            </p>
          </div>

          <div className="card">
            <h3>GSM8K</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              Grade school math word problems. Flexible-extract variant extracts numerics
              from free-form model responses without strict format matching.
            </p>
          </div>

          <div className="card">
            <h3>Evaluation Setup</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              <strong>Harness:</strong> lm-eval v0.4.12 (PyTorch, bfloat16)
              <br />
              <strong>Shots:</strong> 0-shot (no in-context examples)
              <br />
              <strong>Sample limit:</strong> 300 effective samples per benchmark
              <br />
              <strong>Machine:</strong> GB10 (NVIDIA, 20-core ARM CPU)
            </p>
          </div>
        </div>
      </section>

      <section
        style={{
          marginTop: "var(--space-8)",
          paddingTop: "var(--space-6)",
          borderTop: "1px solid var(--border)",
        }}
      >
        <h2>About the Data</h2>
        <p
          style={{
            maxWidth: "50rem",
            color: "var(--text-2)",
            lineHeight: "1.8",
            marginTop: "var(--space-4)",
          }}
        >
          <strong>Baseline (Qwen3-1.7B):</strong> Open-weight model from Alibaba Qwen,
          evaluated on GB10 with lm-eval harness. All numbers are confirmed from{" "}
          <code
            style={{
              fontFamily: "monospace",
              background: "var(--bg-input)",
              padding: "0.2em 0.4em",
              borderRadius: "3px",
              fontSize: "0.9em",
              color: "var(--text-primary)",
            }}
          >
            results/scale/baselines/qwen3_1p7b_screen/
          </code>
          .
          <br />
          <br />
          <strong>EqLM models:</strong> marked{" "}
          <span
            style={{
              display: "inline-block",
              padding: "0.35rem 0.75rem",
              borderRadius: "3px",
              fontSize: "0.75rem",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              background: "var(--bg-input)",
              color: "var(--text-secondary)",
              border: "1px solid var(--border-color)",
            }}
          >
            PENDING
          </span>{" "}
          until measured. Each row represents a planned evaluation target: anytime-unrolled
          121M (F24), Qwen3 conversion (F25), and specialist-ensemble token auctions (F22).
          Numbers will be populated as experiments complete and results are Tarka-reviewed
          and operator-signed.
          <br />
          <br />
          <strong>Provenance:</strong> Every measured result traces to a config hash and ≥3
          distinct random seeds. All findings are documented in{" "}
          <code
            style={{
              fontFamily: "monospace",
              background: "var(--bg-input)",
              padding: "0.2em 0.4em",
              borderRadius: "3px",
              fontSize: "0.9em",
              color: "var(--text-primary)",
            }}
          >
            research/memory/findings.md
          </code>{" "}
          with full experiment details.
        </p>
      </section>
    </div>
  );
}
