"use client";

import { useEffect, useState } from "react";

/**
 * Site-wide notice while the app is replaying. Client-driven by /api/health so
 * it is right whether replay comes from an unset GATEWAY_URL at build time or
 * from a configured gateway that is unreachable at request time (ADR 0011).
 */
export function ReplayBanner({ initial = false }: { initial?: boolean }) {
  const [replay, setReplay] = useState(initial);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/health", { headers: { Accept: "application/json" } })
      .then((r) => (r.ok ? r.json() : { replay: true }))
      .then((d) => {
        if (!cancelled) setReplay(Boolean(d.replay));
      })
      .catch(() => {
        if (!cancelled) setReplay(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!replay) return null;
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
