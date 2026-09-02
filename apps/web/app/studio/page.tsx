"use client";

import { useEffect, useMemo, useState } from "react";
import findings from "@/data/results.json";

/**
 * Run Registry (formerly the Training Studio).
 *
 * Job submission was retired at closure (ADR 0011): the product must never
 * contend for a training GPU, and the serving host is away. What remains is
 * the read-only registry the Studio always carried — one row per results.json
 * in the repository, with its configuration hash, commit and headline numbers,
 * joined to the finding that cites it. In replay mode the rows come from
 * apps/web/data/runs.json, built by scripts/build_app_data.py; with a live
 * gateway the backend walks the same tree.
 */
interface Run {
  dir: string;
  experiment: string;
  spec?: string | null;
  seed?: number | null;
  config_hash: string;
  git_commit: string;
  headline?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
}

interface Finding {
  id: string;
  title: string;
  evidence?: { exp?: string; path?: string };
}

export default function RegistryPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/proxy/api/runs");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setRuns(Array.isArray(data.runs) ? data.runs : []);
      } catch (err) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Join each run to the findings that cite its experiment id.
  const findingsByExp = useMemo(() => {
    const map = new Map<string, Finding[]>();
    for (const f of (findings as { findings: Finding[] }).findings) {
      const exp = f.evidence?.exp;
      if (!exp) continue;
      for (const key of exp.split(/[\s,/]+/).filter(Boolean)) {
        map.set(key, [...(map.get(key) ?? []), f]);
      }
    }
    return map;
  }, []);

  const citing = (run: Run): Finding[] => {
    const keys = new Set([run.experiment, ...run.dir.split("/")]);
    const out: Finding[] = [];
    for (const k of keys) for (const f of findingsByExp.get(k) ?? []) if (!out.includes(f)) out.push(f);
    return out;
  };

  const shown = runs.filter(
    (r) => !filter || r.dir.includes(filter) || r.experiment.includes(filter)
  );

  return (
    <div className="page">
      <h1>Run Registry</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "1rem", maxWidth: "50rem" }}>
        Every completed run in the repository, one row per <code>results.json</code>, with the
        configuration hash and commit that produced it and the findings that cite it. Job
        submission was retired when the programme closed (ADR 0011); the record is read-only.
      </p>
      <input
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter by experiment or path"
        style={{
          padding: "0.4rem 0.6rem",
          borderRadius: 6,
          border: "1px solid var(--border-color)",
          background: "var(--bg-input)",
          color: "var(--text-primary)",
          marginBottom: "1rem",
          minWidth: "18rem",
        }}
      />

      {loading && <p style={{ color: "var(--text-secondary)" }}>Loading registry…</p>}
      {error && <p style={{ color: "var(--error)" }}>Could not load the registry: {error}</p>}
      {!loading && !error && shown.length === 0 && (
        <p style={{ color: "var(--text-tertiary)" }}>No runs match.</p>
      )}

      {shown.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-color)" }}>
                {["Run", "Spec", "Seed", "Config hash", "Commit", "Headline", "Findings"].map((h) => (
                  <th key={h} style={{ padding: "0.6rem", textAlign: "left", fontWeight: 600 }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shown.map((run) => {
                const head = run.headline ?? run.metrics ?? {};
                const cites = citing(run);
                return (
                  <tr key={run.dir} style={{ borderBottom: "1px solid var(--border-color)" }}>
                    <td style={{ padding: "0.6rem", fontFamily: "monospace" }}>{run.dir}</td>
                    <td style={{ padding: "0.6rem" }}>{run.spec ?? "—"}</td>
                    <td style={{ padding: "0.6rem" }}>{run.seed ?? "—"}</td>
                    <td style={{ padding: "0.6rem", fontFamily: "monospace", color: "var(--text-secondary)" }}>
                      {run.config_hash === "unknown" ? "—" : run.config_hash.slice(0, 10)}
                    </td>
                    <td style={{ padding: "0.6rem", fontFamily: "monospace", color: "var(--text-secondary)" }}>
                      {run.git_commit === "unknown" ? "—" : run.git_commit.slice(0, 7)}
                    </td>
                    <td style={{ padding: "0.6rem", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                      {Object.entries(head)
                        .slice(0, 3)
                        .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : String(v)}`)
                        .join(", ") || "—"}
                    </td>
                    <td style={{ padding: "0.6rem" }}>
                      {cites.length === 0
                        ? "—"
                        : cites.map((f) => (
                            <a key={f.id} href={`/findings#${f.id}`} title={f.title} style={{ marginRight: "0.4rem" }}>
                              {f.id}
                            </a>
                          ))}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
