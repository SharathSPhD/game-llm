import { LeaderboardTableClient } from "./client";
import { TwinLadder } from "./TwinLadder";
import { CouncilRecord } from "./CouncilRecord";

export const metadata = {
  title: "Leaderboard — Kinetic AI",
  description: "Benchmark leaderboard: Qwen2.5-1.5B baselines measured on our harness with real provenance.",
};

export default function LeaderboardPage() {
  return (
    <div className="page">
      <h1>Benchmark Leaderboard</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
        Qwen2.5-1.5B baselines (generalist, coder, math specialist) and Qwen3-1.7B measured on the GB10 harness in August 2026 (F28).
        MMLU (0-shot), GSM8K (chat template, flexible extract), and mixed arena (50/50 blend).
        All numbers trace to lm-eval config hash, seed set, and reproducible invocation.
      </p>

      <LeaderboardTableClient />

      <TwinLadder />

      <CouncilRecord />

      <section style={{ marginTop: "var(--space-8)", marginBottom: "var(--space-5)" }}>
        <h2>Domain Strengths &amp; Aggregation Headroom</h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "var(--space-4)",
            marginTop: "var(--space-4)",
          }}
        >
          <div className="card">
            <h3>MMLU (Knowledge)</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              57 knowledge domains. Generalist Qwen2.5-1.5B scores 0.626.
              Math specialist drops to 0.391 (last place). Coder and Qwen3 similar to generalist.
              No routing headroom here — players are nearly interchangeable.
            </p>
          </div>

          <div className="card">
            <h3>GSM8K (Mathematics)</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              Grade school math with chat template + flexible-extract scoring. Math specialist
              scores 0.795, generalist 0.595 — 20-point gap. This is real complementarity (F33).
              Perfect domain router reaches 0.711 on mixed arena.
            </p>
          </div>

          <div className="card">
            <h3>Mixed Arena (50/50)</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              Equal blend of MMLU and GSM8K. Best single model scores 0.611. Perfect router
              reaches 0.711 — 10-point routable headroom, where the aggregation game plays out.
              This is where domain selection matters (F33).
            </p>
          </div>

          <div className="card">
            <h3>Evaluation Protocol</h3>
            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
              <strong>Harness:</strong> lm-eval (PyTorch, bfloat16, 0-shot)
              <br />
              <strong>Machine:</strong> GB10 (NVIDIA DGX Spark)
              <br />
              <strong>Accuracy:</strong> MMLU strict-match, GSM8K flexible-extract
              <br />
              <strong>Provenance:</strong> config hash, seed set, lm-eval invocation
            </p>
          </div>
        </div>
      </section>

      <section
        style={{
          marginTop: "var(--space-8)",
          paddingTop: "var(--space-6)",
          borderTop: "1px solid var(--border-color)",
        }}
      >
        <h2>Findings &amp; Implications</h2>
        <p
          style={{
            maxWidth: "50rem",
            color: "var(--text-secondary)",
            lineHeight: "1.8",
            marginTop: "var(--space-4)",
          }}
        >
          <strong>F28 (Baseline Ladder):</strong> These four models measured on our harness,
          establishing the ground truth against which all later claims rest. All numbers public
          and reproducible from{" "}
          <code style={{ fontFamily: "monospace", background: "var(--bg-input)", padding: "0.2em 0.4em", borderRadius: "3px", fontSize: "0.9em", color: "var(--text-primary)" }}>
            results/scale/ladder/
          </code>
          .
          <br />
          <br />
          <strong>F33 (Arena Correction):</strong> GSM8K was masked by harness fault (chat template
          not applied). With fix: Math variant 0.795 vs 0.595 generalist. Mixed arena reveals
          10-point routable headroom where answer-level aggregation was silent (F29–F31 tested
          only homogeneous MMLU). Cross-examination tests whether verification-driven influence
          can extract that headroom (Phase 1b).
          <br />
          <br />
          <strong>F32 (Oracle Audit):</strong> Gating the oracle on confidence ≥0.5 drops the
          apparent ceiling from 0.826 to 0.658 — 1.6 points above best single, not 20. Aggregation
          rules are operating within reach of what these distributions support.
          <br />
          <br />
          <strong>Next Iteration (Phase 2):</strong> Build teachers from measured eval gaps.
          Math specialist already exists and is strong; replicate that pattern for other weak
          domains. Then test council on heterogeneous mixed arena.
        </p>
      </section>
    </div>
  );
}
