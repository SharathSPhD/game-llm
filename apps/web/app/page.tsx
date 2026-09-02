import Link from "next/link";
import { readFileSync } from "fs";
import { join } from "path";

export const metadata = {
  title: "Kinetic AI — Equilibrium Language Model",
};

// Load flagship status at build time
let flagshipStatus = {
  stage: "closed",
  spec: "ADR 0011",
  updated: "2026-09-02",
  description: "",
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
            depth is a stopping criterion rather than an architecture. The validated result: at equal compute,
            a weight-tied block reaches 0.958 of an explicit transformer&apos;s quality with 2.70× fewer
            parameters at 46–121M (F45, three seeds). Taken to a billion parameters on web data, the tied arm
            failed its pre-registered 1B-token gate (perplexity ratio 1.56 vs 1.20) and was still closing at
            2.5B tokens (1.31); both arms score at chance on public benchmarks, and the programme closed there
            (F55). Weight-tying compresses parameters, not compute, and the record says where that stops.
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
              <div className="panel-label">Equal compute, three seeds (F45)</div>
              <div className="reading">0.958</div>
              <div className="panel-note">BLiMP ratio of the tied block to the explicit baseline, CI [0.939, 0.977]</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Programme</div>
              <div className="reading">{flagshipStatus.stage}</div>
              <div className="panel-note">{flagshipStatus.spec} · {flagshipStatus.updated} · record ends at F55</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Parameter Saving</div>
              <div className="reading">2.70×</div>
              <div className="panel-note">45.8M against 123.8M at identical arithmetic (F45)</div>
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
              transformer (F24); at equal compute with the block at the baseline&apos;s width, 0.958 with 2.70× fewer
              parameters (F45). At a billion parameters the exchange rate did not transfer unchanged (F55).
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
              Measured (exp31), uneven spending scores 0.681 against 0.684 at the same mean depth — it buys
              nothing at matched depth and supplies the anytime property, not an advantage.
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
              The complete record F1–F55, each Tarka-reviewed: convergence, mechanism design, the EqLM
              paradigm, the council, the exchange rate and the billion-parameter boundary.
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
              Replays the measured council record (F41, F54). Live council decoding returns with the
              serving host; no per-token influence traces were recorded, and the page says so.
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
            <strong>Pre-registration.</strong> Hypotheses (H1–H10), experiment specs (SPEC 0001–0024), decisions (ADR 0001–0011), and success
            criteria are recorded before runs. Findings are validated or formally missed, not reinterpreted.
          </li>
          <li style={{ marginBottom: "var(--space-3)" }}>
            <strong>Honest nulls.</strong> When a mechanism fails (answer-level equilibrium, magnetic drift), the
            result ships with diagnosis and cost. The council&apos;s 8-point win exists alongside its precondition and cost.
          </li>
          <li>
            <strong>Artifacts.</strong> Four models and the council dataset on Hugging Face under qbz506, with cards
            carrying the claims and the non-claims. The API backend is profile-driven and offline while the serving
            host is away; the app replays the record until it returns.
          </li>
        </ul>
      </section>
    </div>
  );
}
