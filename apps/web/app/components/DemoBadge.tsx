/**
 * Marks a page whose results are pre-recorded. The programme closed on
 * 2026-09-02 (ADR 0011) with its serving host away; every interactive page
 * replays measured or synthetic data and says so. Live inference returns when
 * the GB10 is back, through the serving profile, without a code change here.
 */
export function DemoBadge({ what = "sample data" }: { what?: string }) {
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
      Replay — {what}, not a live run. Live inference returns when the serving host is back
      (see <a href="/findings" style={{ color: "var(--accent-mid)" }}>F55</a>).
    </p>
  );
}
