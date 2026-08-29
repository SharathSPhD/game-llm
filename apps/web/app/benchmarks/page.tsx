import { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Benchmarks — Head-to-Head Comparison",
};

export default function BenchmarksPage() {
  return (
    <div className="page wrap">
      <section className="landing-hero">
        <div>
          <p className="eyebrow">Head-to-Head Evaluation</p>
          <h1>EqLM vs Explicit Baselines</h1>
          <p className="lede">
            Paired comparison of EqLM and conventional transformers at matched parameters and compute.
            F44 corrects F24: at equal FLOPs (2.44 iterations), the ratio is 0.72. Adaptive per-token
            depth (exp31) reaches 0.681 at mean 11.3 iterations vs explicit 0.684 at fixed 12 — the
            anytime property works, scaling is graceful.
          </p>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Parameter-Matched Comparison (F24, F44)</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          Both models trained identically on the full BabyLM stream (20k steps, batch 32).
          Evaluation: BLiMP (1000 scored pairs), bootstrap 95% CI over 3 seeds (42/43/44).
        </p>
        <div className="table-scroll" style={{ marginTop: "var(--space-5)" }}>
          <table className="results-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Parameters</th>
                <th>Loss (final)</th>
                <th>BLiMP Accuracy</th>
                <th>Ratio vs Explicit</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Explicit (A1)</strong></td>
                <td>123.8M</td>
                <td>2.73–3.07</td>
                <td>0.7133±0.0294</td>
                <td>1.0</td>
                <td>12-layer baseline</td>
              </tr>
              <tr>
                <td><strong>EqLM post-LN (A3)</strong></td>
                <td>120.7M</td>
                <td>3.29–4.00</td>
                <td>0.6637±0.0365</td>
                <td>0.930 [0.898–0.949]</td>
                <td>F18: formally below 95% threshold</td>
              </tr>
              <tr>
                <td><strong>EqLM anytime (B1)</strong></td>
                <td>120.7M</td>
                <td>2.95–3.15</td>
                <td>0.6805±0.0202</td>
                <td>0.991 [0.971–1.033]</td>
                <td>F24: parity at matched params+iters</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p style={{ color: "var(--text-secondary)", marginTop: "var(--space-3)", fontSize: "0.9rem" }}>
          Wall-clock cost: EqLM 2.9× vs explicit (92 vs 11 min/arm on GB10). Peak memory A3: 6.29GB vs A1: 8.13GB (−23%).
        </p>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Compute-Matched Corrected Ratio (F44)</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          F24 measured parity at matched iteration count (12 each), but the tied block costs 4.92× FLOPs per iteration.
          At equal compute (2.44 EqLM iters vs 12 explicit layers), the ratio drops to <strong>0.72</strong>.
        </p>
        <div className="table-scroll" style={{ marginTop: "var(--space-5)" }}>
          <table className="results-table">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>EqLM Budget</th>
                <th>Explicit Budget</th>
                <th>Ratio</th>
                <th>Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Matched iterations (12 each)</td>
                <td>12 iters × 34.8M</td>
                <td>12 layers × 7.08M</td>
                <td>0.991</td>
                <td>Parity (F24)</td>
              </tr>
              <tr>
                <td>Matched FLOPs</td>
                <td>2.44 iters × 34.8M</td>
                <td>12 layers × 7.08M</td>
                <td>0.72</td>
                <td>Weight-tying saves parameters, not compute (F44)</td>
              </tr>
              <tr>
                <td>Matched mean depth (exp31)</td>
                <td>11.3 iters adaptive</td>
                <td>12 fixed layers</td>
                <td>0.996</td>
                <td>Anytime property works (exp31)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Adaptive Per-Token Depth (exp31)</h2>
        <p style={{ color: "var(--text-secondary)" }}>
          EqLM can spend few iterations on easy tokens and many on hard ones. Uneven spending at matched mean depth scores identically to fixed depth but enables graceful degradation across budgets.
        </p>
        <div className="table-scroll" style={{ marginTop: "var(--space-5)" }}>
          <table className="results-table">
            <thead>
              <tr>
                <th>Depth Budget</th>
                <th>EqLM (Adaptive)</th>
                <th>Explicit (Fixed)</th>
                <th>Ratio</th>
                <th>Anytime Property</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>4 iterations</td>
                <td>0.620</td>
                <td>N/A</td>
                <td>−</td>
                <td>First anytime depth</td>
              </tr>
              <tr>
                <td>6 iterations (mean of adaptive run)</td>
                <td>0.628</td>
                <td>−</td>
                <td>−</td>
                <td>Half-budget quality</td>
              </tr>
              <tr>
                <td>8 iterations</td>
                <td>0.644</td>
                <td>−</td>
                <td>−</td>
                <td>Two-thirds budget</td>
              </tr>
              <tr>
                <td>12 iterations (full budget)</td>
                <td>0.681 (adaptive mean 11.3)</td>
                <td>0.684 (fixed 12)</td>
                <td>0.996</td>
                <td>Parity with graceful degradation</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>Key Findings</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "var(--space-5)", marginTop: "var(--space-5)" }}>
          <div className="card">
            <h3>F44: Honest Compute Accounting</h3>
            <p>
              Parameter-matched parity (0.991) is achieved at matched iteration count (12), but the tied block is 4.92× costlier
              per iteration. At equal FLOPs, the ratio is 0.72. Weight-tying saves parameters (28%), not compute.
            </p>
          </div>
          <div className="card">
            <h3>F24: Anytime Training Works</h3>
            <p>
              Unrolled training with supervision at z₄, z₈, z₁₂ closes the entire width gap (0.571 → 0.697, mean 0.991).
              One seed exceeds its baseline. The tied block trained as an equilibrium model reaches parity with explicit transformers.
            </p>
          </div>
          <div className="card">
            <h3>exp31: Graceful Anytime Degradation</h3>
            <p>
              Adaptive per-token depth reaches 0.681 at mean 11.3 iters vs fixed 12 at 0.684 — the anytime property works.
              At half-budget (6 iters), quality degrades to 0.628 smoothly. A model usable at any compute, not better at one.
            </p>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)", marginBottom: "var(--space-8)" }}>
        <h2>Explore Further</h2>
        <ul style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          <li><Link href="/demo" className="btn">Try the Anytime Demo</Link></li>
          <li><Link href="/findings" className="btn">Full Findings</Link></li>
          <li><Link href="/" className="btn" data-secondary="true">Back to Overview</Link></li>
        </ul>
      </section>
    </div>
  );
}
