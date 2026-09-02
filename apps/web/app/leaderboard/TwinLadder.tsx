import ladder from "@/data/ladder_exp40.json";

interface Row {
  model: string;
  tokens: string;
  ours: boolean;
  mean6: number | null;
  arc_easy?: number;
  arc_challenge?: number;
  hellaswag?: number;
  piqa?: number;
  winogrande?: number;
  sciq?: number;
  lambada_openai?: number;
}

const COLS: (keyof Row)[] = [
  "arc_easy",
  "arc_challenge",
  "hellaswag",
  "piqa",
  "winogrande",
  "sciq",
  "lambada_openai",
];

/**
 * The 1B twin against public rungs on the same harness (F55). Server
 * component over a committed snapshot: the honest picture is that both arms
 * are at chance at 2.5B tokens, and that is the reason the programme halted.
 */
export function TwinLadder() {
  const rows = (ladder as { rows: Row[] }).rows;
  return (
    <section style={{ marginTop: "var(--space-8)" }}>
      <h2>The Twin at One Billion Parameters (F55)</h2>
      <p style={{ color: "var(--text-secondary)", maxWidth: "50rem" }}>
        Explicit 913M against tied 158M-resident, compute-matched, 2.5B FineWeb-Edu tokens each,
        scored on 1,000 examples per task beside public rungs. {(ladder as { chance_note: string }).chance_note}
      </p>
      <div style={{ overflowX: "auto", marginTop: "1rem" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)" }}>
              <th style={{ padding: "0.5rem", textAlign: "left" }}>Model</th>
              <th style={{ padding: "0.5rem", textAlign: "left" }}>Tokens</th>
              <th style={{ padding: "0.5rem", textAlign: "right" }}>mean6</th>
              {COLS.map((c) => (
                <th key={c} style={{ padding: "0.5rem", textAlign: "right" }}>
                  {c.replace("_openai", "")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={`${r.model}-${r.tokens}`}
                style={{
                  borderBottom: "1px solid var(--border-color)",
                  fontWeight: r.ours ? 600 : 400,
                }}
              >
                <td style={{ padding: "0.5rem" }}>{r.model}</td>
                <td style={{ padding: "0.5rem" }}>{r.tokens}</td>
                <td style={{ padding: "0.5rem", textAlign: "right" }}>
                  {r.mean6 === null ? "—" : r.mean6.toFixed(3)}
                </td>
                {COLS.map((c) => (
                  <td key={c} style={{ padding: "0.5rem", textAlign: "right", fontFamily: "monospace" }}>
                    {typeof r[c] === "number" ? (r[c] as number).toFixed(3) : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
