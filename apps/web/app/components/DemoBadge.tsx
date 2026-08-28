export function DemoBadge() {
  return (
    <p
      style={{
        display: "inline-block",
        padding: "0.3rem 0.6rem",
        borderRadius: "6px",
        border: "1px solid var(--accent-mid)",
        color: "var(--text-secondary)",
        fontSize: "0.8rem",
        marginBottom: "var(--space-3)",
      }}
    >
      Demo replay — sample data, not a live run.{" "}
      <a href="/login" style={{ color: "var(--accent-mid)" }}>
        Sign in
      </a>{" "}
      to execute on the research backend.
    </p>
  );
}
