import council from "@/data/council.json";

interface SeedRecord {
  seed: number;
  n_tasks: number;
  players: string[];
  accuracy: Record<string, number>;
}

interface FairBaselines {
  n: number;
  accuracy: Record<string, number>;
  generations_per_request?: Record<string, number>;
  council_vs?: Record<string, { wins: number; losses: number; z: number }>;
}

const RULES = ["majority", "equilibrium", "cross_exam", "leave_one_out", "self_preference", "oracle"];

/**
 * The council as it was measured (F41 confirmation seeds, F54 fair baselines).
 * The equilibrium view the PRD asked for — per-token influence weights — was
 * never recorded: exp23 persisted winners only. So this panel shows what
 * exists, per seed and rule, and says what it is.
 */
export function CouncilRecord() {
  const data = council as {
    seeds: SeedRecord[];
    fair_baselines: FairBaselines | null;
    note: string;
  };
  const short = (p: string) => p.replace("Qwen/", "").replace("-Instruct", "");
  return (
    <section style={{ marginTop: "var(--space-8)" }}>
      <h2>The Council as Measured (F41, F54)</h2>
      <p style={{ color: "var(--text-secondary)", maxWidth: "50rem" }}>{data.note}</p>

      <div style={{ overflowX: "auto", marginTop: "1rem" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)" }}>
              <th style={{ padding: "0.5rem", textAlign: "left" }}>Seed</th>
              <th style={{ padding: "0.5rem", textAlign: "right" }}>n</th>
              {RULES.map((r) => (
                <th key={r} style={{ padding: "0.5rem", textAlign: "right" }}>
                  {r}
                </th>
              ))}
              <th style={{ padding: "0.5rem", textAlign: "right" }}>best single</th>
            </tr>
          </thead>
          <tbody>
            {data.seeds.map((s) => {
              const singles = Object.entries(s.accuracy).filter(([k]) => k.startsWith("single::"));
              const best = singles.sort((a, b) => b[1] - a[1])[0];
              return (
                <tr key={s.seed} style={{ borderBottom: "1px solid var(--border-color)" }}>
                  <td style={{ padding: "0.5rem" }}>{s.seed}</td>
                  <td style={{ padding: "0.5rem", textAlign: "right" }}>{s.n_tasks}</td>
                  {RULES.map((r) => (
                    <td key={r} style={{ padding: "0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                      {typeof s.accuracy[r] === "number" ? s.accuracy[r].toFixed(3) : "—"}
                    </td>
                  ))}
                  <td style={{ padding: "0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                    {best ? `${best[1].toFixed(3)} (${short(best[0].slice(8))})` : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {data.fair_baselines && (
        <div className="card" style={{ marginTop: "1rem", maxWidth: "50rem" }}>
          <h3>Fair baselines on the {data.fair_baselines.n} confirmation questions (F54)</h3>
          <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
            {Object.entries(data.fair_baselines.accuracy)
              .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v.toFixed(4)}`)
              .join(" · ")}
          </p>
          <p style={{ fontSize: "var(--text-sm)", color: "var(--text-2)" }}>
            The routing council is the most accurate use of its own generation budget and is beaten
            by nineteen points by a single 7B model of comparable resident memory.
          </p>
        </div>
      )}
    </section>
  );
}
