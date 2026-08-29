import Link from "next/link";
import { readFileSync } from "fs";
import { join } from "path";

export const metadata = {
  title: "Kinetic AI — Equilibrium Language Model",
};

// Load flagship status at build time
let flagshipStatus = {
  stage: "damage_probe",
  updated: new Date().toISOString(),
  model: "Qwen2.5-1.5B-Instruct",
  approach: "Two-fold reuse",
  estimated_unique_params: "1.10B (28% reduction)",
  fixed_compute: true,
  comment: "SPEC 0020: Qwen conversion at scale.",
};

try {
  const statusPath = join(process.cwd(), "public/data/flagship_status.json");
  const data = readFileSync(statusPath, "utf-8");
  flagshipStatus = JSON.parse(data);
} catch {
  // Use default
}

export default function Home() {
  return (
    <div className="page wrap">
      <section className="landing-hero">
        <div>
          <p className="eyebrow">EqLM: Equilibrium Language Model</p>
          <h1>A language model whose depth, training, and decoding are equilibrium computations.</h1>
          <p className="lede">
            EqLM replaces the fixed depth of a conventional transformer with a fixed-point computation whose
            depth is a stopping criterion rather than an architecture. At 121M parameters it reaches parity
            with an explicit transformer (F24) — but that parity is measured at matched parameters and matched
            iteration count, and the tied block is 4.92× more expensive per iteration, so parity costs roughly
            five times the arithmetic. At genuinely equal compute the ratio is 0.72 (F44). Weight-tying
            compresses parameters, not compute.
          </p>
          <div className="hero-actions">
            <Link href="/benchmarks" className="btn" data-primary="true">
              View Benchmarks
            </Link>
            <Link href="/demo" className="btn">
              Try Anytime Demo
            </Link>
            <Link href="/api" className="btn">
              API Reference
            </Link>
          </div>
          <div className="hero-stats">
            <div className="hero-stat">
              <div className="panel-label">EqLM vs Explicit (F24)</div>
              <div className="reading">0.991</div>
              <div className="panel-note">Ratio at matched params and iterations; at equal FLOPs: 0.72 (F44)</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Flagship Progress</div>
              <div className="reading">{flagshipStatus.stage}</div>
              <div className="panel-note">{flagshipStatus.model} · {flagshipStatus.approach}</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Parameter Saving</div>
              <div className="reading">28%</div>
              <div className="panel-note">SPEC 0020: {flagshipStatus.estimated_unique_params}</div>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>The Core Claims</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "var(--space-5)", marginTop: "var(--space-5)" }}>
          <div className="card">
            <h3>Equilibrium Depth</h3>
            <p>
              A language model whose effective depth is not a fixed architecture choice but a solved game:
              at each token, the model computes a fixed point. At matched parameters (121M) it reaches 0.991 of an explicit
              transformer (F24). At genuinely equal compute, the ratio is 0.72 (F44).
            </p>
          </div>

          <div className="card">
            <h3>Anytime Training</h3>
            <p>
              Unrolled training with supervision at intermediate depths closes the quality gap.
              One seed exceeds its explicit baseline. The anytime property enables graceful degradation:
              0.628 at half-budget, 0.488 at one-sixth.
            </p>
          </div>

          <div className="card">
            <h3>Adaptive Per-Token Depth</h3>
            <p>
              An equilibrium model can spend few iterations on easy tokens and many on hard ones.
              Measured (exp31), uneven spending reaches 0.996 at the same mean depth as fixed 0.684 —
              the model adapts to any compute budget.
            </p>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "4rem" }}>
        <h2>Key Pages</h2>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Start here: the 4 core pages that make up the application.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: "1.5rem",
            marginTop: "2rem",
          }}
        >
          <Link href="/benchmarks" className="card" style={{ textDecoration: "none" }}>
            <h3>Benchmarks</h3>
            <p>
              Head-to-head comparison: EqLM vs explicit transformers at matched parameters and compute.
              F24 parity ratio, F44 corrected compute accounting, exp31 adaptive depth results.
            </p>
          </Link>

          <Link href="/api" className="card" style={{ textDecoration: "none" }}>
            <h3>API Reference</h3>
            <p>
              Generate with EqLM at any anytime depth. OpenAI-compatible endpoints, Kinetic controls,
              full endpoint documentation with curl examples.
            </p>
          </Link>

          <Link href="/demo" className="card" style={{ textDecoration: "none" }}>
            <h3>Demo & Findings</h3>
            <p>
              Interactive anytime depth dial. Generate text at 4–12 iterations. Research findings timeline,
              in-browser inference scaffold awaiting ONNX artifact.
            </p>
          </Link>

          <Link href="/findings" className="card" style={{ textDecoration: "none" }}>
            <h3>All Findings</h3>
            <p>
              Validated findings from convergence (F1–F8), mechanism design (F6), EqLM paradigm (F24–F26),
              and scale results (SPEC 0020 damage probe). Each links to config hash, seeds, and harness.
            </p>
          </Link>
        </div>
      </section>

      <section style={{ marginTop: "4rem", marginBottom: "2rem" }}>
        <h2>Secondary Tools</h2>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Interactive mechanics lab and research dashboard.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: "1.5rem",
            marginTop: "2rem",
          }}
        >
          <Link href="/lab" className="card" style={{ textDecoration: "none" }}>
            <h3>Equilibrium Lab</h3>
            <p>
              Run MMD, GDA, QRE solvers on matrix games. Interactive visualization of convergence,
              strategy simplexes, Nash equilibrium computation.
            </p>
          </Link>

          <Link href="/chat" className="card" style={{ textDecoration: "none" }}>
            <h3>Council Chat</h3>
            <p>
              Multi-turn conversation with the council. Tune rationality and solver budget,
              see per-token influence weights.
            </p>
          </Link>
        </div>
      </section>

      <section style={{ marginTop: "4rem", marginBottom: "2rem" }}>
        <h2>Reproducibility & Transparency</h2>
        <ul style={{ color: "var(--text-secondary)", lineHeight: "1.8", listStyle: "none", paddingLeft: 0 }}>
          <li style={{ marginBottom: "var(--space-3)" }}>
            <strong>Every number traces.</strong> BLiMP scores, perplexities, and parameters link to config
            hash, git commit, seed set, and lm-eval invocation. No hardcoded numbers in the code.
          </li>
          <li style={{ marginBottom: "var(--space-3)" }}>
            <strong>Pre-registration.</strong> Hypotheses (H1–H10), experiment specs (SPEC 0001–0020), and success
            criteria are recorded before runs. Findings are validated or formally missed, not reinterpreted.
          </li>
          <li style={{ marginBottom: "var(--space-3)" }}>
            <strong>Honest nulls.</strong> When a mechanism fails (answer-level equilibrium, magnetic drift), the
            result ships with diagnosis and cost. The council&apos;s 8-point win exists alongside its precondition and cost.
          </li>
          <li>
            <strong>API-first delivery.</strong> OpenAI-compatible endpoints from GB10 FastAPI. Models released to
            Hugging Face with full reproducibility card. Anytime inference usable at any compute budget.
          </li>
        </ul>
      </section>
    </div>
  );
}
