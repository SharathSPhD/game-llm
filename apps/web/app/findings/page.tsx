import results from "@/data/results.json";

export const metadata = {
  title: "Findings — EqLM",
  description: "Validated research findings from equilibrium learning experiments.",
};

export default function FindingsPage() {
  return (
    <div className="page wrap">
      <section style={{ marginBottom: "var(--space-7)" }}>
        <h1 style={{ marginBottom: "var(--space-3)" }}>Research Findings</h1>
        <p className="lede" style={{ marginBottom: "var(--space-5)" }}>
          All findings are Tarka-reviewed and operator-signed. Each entry includes experiment ID, config hash, seed count,
          and links to evidence artifacts. Status is VALIDATED or SIGNED OFF.
        </p>
      </section>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(22rem, 1fr))",
          gap: "var(--space-4)",
        }}
      >
        {results.findings.map((finding: any) => (
          <div key={finding.id} className="card" style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ marginBottom: "var(--space-3)" }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: "var(--space-2)", marginBottom: "var(--space-2)", flexWrap: "wrap" }}>
                <span className="panel-label" style={{ margin: 0 }}>{finding.id}</span>
                <span className="badge" data-tone={finding.status === "VALIDATED" ? "ok" : "accent"}>
                  {finding.status}
                </span>
              </div>
              <h3 style={{ margin: "var(--space-2) 0 0" }}>{finding.title}</h3>
            </div>

            <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)", marginTop: "var(--space-2)", marginBottom: "var(--space-3)", flex: 1 }}>
              {finding.claim}
            </p>

            {finding.evidence && (
              <div
                style={{
                  background: "var(--surface-2)",
                  padding: "var(--space-3)",
                  borderRadius: "var(--radius-sm)",
                  marginBottom: "var(--space-3)",
                  fontSize: "var(--text-sm)",
                  color: "var(--text-2)",
                }}
              >
                {finding.evidence.exp && (
                  <div>
                    <span style={{ fontFamily: "var(--mono)", fontSize: "var(--text-xs)", color: "var(--text-3)" }}>EXP</span>{" "}
                    {finding.evidence.exp}
                  </div>
                )}
                {finding.evidence.seeds && <div>Seeds: {finding.evidence.seeds}</div>}
                {finding.evidence.config_hash && (
                  <div style={{ fontFamily: "var(--mono)", fontSize: "var(--text-xs)" }}>
                    Config: {finding.evidence.config_hash}
                  </div>
                )}
              </div>
            )}

            {finding.artifact_path && (
              <a
                href={`#${finding.artifact_path}`}
                style={{
                  display: "inline-flex",
                  gap: "0.35rem",
                  fontSize: "var(--text-base)",
                  color: "var(--accent)",
                  fontWeight: 550,
                  marginTop: "auto",
                }}
              >
                View evidence →
              </a>
            )}
          </div>
        ))}
      </div>

      <section style={{ marginTop: "var(--space-8)", paddingTop: "var(--space-6)", borderTop: "1px solid var(--border)" }}>
        <h2>About These Findings</h2>
        <p style={{ maxWidth: "50rem", color: "var(--text-2)", lineHeight: "1.8", marginTop: "var(--space-4)" }}>
          Every finding listed here is bound by the closure contract: all six layers must be satisfied before sign-off.
          This means each claim includes reproducible code, measured evidence, Tarka adversarial review, artifact paths,
          and operator approval. To read the full context for any finding, see the research journal and pre-registration
          in research/memory/. To understand the science behind these results, start with{" "}
          <a href="/learn">Learn EqLM</a>.
        </p>
      </section>
    </div>
  );
}
