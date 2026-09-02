/**
 * Site-wide notice rendered by the layout while REPLAY_MODE is on. Server
 * component: the flag is read at build time from lib/config, which is also
 * when /api/health is prerendered, so the banner and the health dot agree.
 */
export function ReplayBanner() {
  return (
    <div
      role="status"
      style={{
        background: "rgba(214, 158, 46, 0.12)",
        borderBottom: "1px solid rgba(214, 158, 46, 0.5)",
        color: "var(--text-secondary)",
        fontSize: "0.8rem",
        padding: "0.4rem 1rem",
        textAlign: "center",
      }}
    >
      The programme closed on 2026-09-02 at finding F55. Every result shown is pre-recorded from the
      published record; nothing here runs on a GPU. Live inference returns with the serving host.{" "}
      <a href="/findings" style={{ color: "var(--accent-mid)" }}>
        Read the record
      </a>
      .
    </div>
  );
}
