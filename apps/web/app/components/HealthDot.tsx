"use client";

import { useEffect, useState } from "react";

interface HealthStatus {
  online: boolean;
  replay?: boolean;
  version?: string;
  gpu_available?: boolean;
}

/**
 * Three honest states, not two. "Live" means a backend answered and it was
 * not the replay stub; "Replay" means the app is serving pre-recorded results
 * (the serving host is away, ADR 0011); "Offline" means a configured backend
 * did not answer. The old dot read "Live" whenever /api/health returned 200,
 * which in replay mode it always did.
 */
export function HealthDot() {
  const [health, setHealth] = useState<HealthStatus>({ online: false });
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch("/api/health", {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      if (response.ok) {
        const data = await response.json();
        setHealth({ online: true, ...data });
      } else {
        setHealth({ online: false });
      }
    } catch {
      setHealth({ online: false });
    }
  };

  if (!mounted) {
    return null;
  }

  const state = health.replay ? "replay" : health.online ? "live" : "offline";
  const label = state === "live" ? "Live" : state === "replay" ? "Replay" : "Offline";
  const color =
    state === "live" ? "var(--success, #2e9e5b)" : state === "replay" ? "#d69e2e" : "var(--error, #c0392b)";
  const title =
    state === "live"
      ? `Backend online (v${health.version}, GPU: ${health.gpu_available ? "yes" : "no"})`
      : state === "replay"
        ? "Replay: pre-recorded results. Live inference returns when the serving host is back (ADR 0011)."
        : "Backend configured but unreachable";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        fontSize: "0.75rem",
        color: "var(--text-tertiary)",
      }}
      title={title}
    >
      <div
        className={`health-dot ${state === "offline" ? "offline" : ""}`}
        style={{ background: color, width: 8, height: 8, borderRadius: 999 }}
      />
      <span>{label}</span>
    </div>
  );
}
