"use client";

import { useEffect, useState } from "react";

interface HealthStatus {
  online: boolean;
  version?: string;
  gpu_available?: boolean;
}

export function HealthDot() {
  const [health, setHealth] = useState<HealthStatus>({ online: false });
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch("/api/health", {
        method: "GET",
        headers: { "Accept": "application/json" },
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

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        fontSize: "0.75rem",
        color: "var(--text-tertiary)",
      }}
      title={
        health.online
          ? `Backend online (v${health.version}, GPU: ${health.gpu_available ? "yes" : "no"})`
          : "Backend offline — using replay demo data"
      }
    >
      <div className={`health-dot ${!health.online ? "offline" : ""}`} />
      <span>{health.online ? "Live" : "Demo"}</span>
    </div>
  );
}
