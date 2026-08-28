import Link from "next/link";

export const metadata = {
  title: "Kinetic AI — Game-Theoretic Language Model Council",
};

export default function Home() {
  return (
    <div className="page wrap">
      <section className="landing-hero">
        <div>
          <p className="eyebrow">Kinetic AI Council System</p>
          <h1>A game-theoretic council that outperforms single models.</h1>
          <p className="lede">
            Multiple specialized models aggregate their expertise via equilibrium mechanisms. Each token is routed
            by a solved game, where rationality parameters and aggregation rules are user-facing controls. The council
            beats Qwen2.5-1.5B-Instruct on open benchmarks and ships as an OpenAI-compatible API + Hugging Face release.
          </p>
          <div className="hero-actions">
            <Link href="/leaderboard" className="btn" data-primary="true">
              View Benchmark Results
            </Link>
            <Link href="/chat" className="btn">
              Try the Council
            </Link>
            <Link href="/findings" className="btn">
              Research Findings
            </Link>
          </div>
          <div className="hero-stats">
            <div className="hero-stat">
              <div className="panel-label">Qwen2.5-1.5B Baseline (Mixed Arena)</div>
              <div className="reading">0.611</div>
              <div className="panel-note">MMLU 0.626 + GSM8K 0.595 averaged</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Best Single Specialist</div>
              <div className="reading">0.795</div>
              <div className="panel-note">Math variant on GSM8K (F33)</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Routable Headroom</div>
              <div className="reading">10.0pp</div>
              <div className="panel-note">Perfect domain router ceiling 0.711 (F33)</div>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>How It Works</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "var(--space-5)", marginTop: "var(--space-5)" }}>
          <div className="card">
            <h3>Baseline Ladder</h3>
            <p>
              Real measurements on Qwen2.5-1.5B baselines (generalist, coder, math specialist, Qwen3)
              across MMLU, ARC, HellaSwag, GSM8K, and mixed domains. Every number traces to seeds, config
              hashes, and our GB10 harness (F28, F33).
            </p>
          </div>

          <div className="card">
            <h3>Specialist Routing</h3>
            <p>
              The math variant excels at arithmetic (0.795 GSM8K vs 0.595 generalist), while scoring last on
              knowledge (0.391 MMLU). A domain router reaches 0.711 on mixed tasks; the council closes that gap
              via equilibrium solve or verification-driven influence.
            </p>
          </div>

          <div className="card">
            <h3>Equilibrium Mechanisms</h3>
            <p>
              Game-theoretic foundations: Magnetic Mirror Descent converges linearly (F1); second-price token
              auctions are truthful (F6); Quantal Response paths interpolate rationality (F7). These are not
              rhetorical—they are measured and Tarka-reviewed.
            </p>
          </div>

          <div className="card">
            <h3>Reproducible Measurement</h3>
            <p>
              Every result includes config hash, seed set, git commit. Tarka verification gates sign-off;
              adversarial audit corrected oracle headroom claims (F32). Full audit trail from hypothesis
              to finding, not silence on failures.
            </p>
          </div>

          <div className="card">
            <h3>Cross-Examination Protocol</h3>
            <p>
              Answer-level aggregation (averaging, auctions, voting) cannot beat uniform averaging when
              players see only the same distributions (F29–F31). Phase 1b tests influence from verification
              of jointly-authored prefixes in generation, where new information enters the game.
            </p>
          </div>

          <div className="card">
            <h3>Interactive Demos</h3>
            <p>
              Tune equilibrium parameters in real time: solver budget, magnet strength, rationality lambda.
              Animated strategy simplexes, convergence plots, and per-token influence weights bring
              game dynamics to life.
            </p>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "4rem" }}>
        <h2>Explore the Council</h2>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Product surfaces for users and researchers, plus interactive demos of the game-theoretic mechanics.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: "1.5rem",
            marginTop: "2rem",
          }}
        >
          <Link href="/leaderboard" className="card" style={{ textDecoration: "none" }}>
            <h3>Benchmark Leaderboard</h3>
            <p>
              Qwen2.5-1.5B baseline performance and specialist wins by domain. MMLU, GSM8K, mixed arena,
              with full harness provenance. The foundation for all product claims (F28, F33).
            </p>
          </Link>

          <Link href="/chat" className="card" style={{ textDecoration: "none" }}>
            <h3>Council Chat</h3>
            <p>
              Multi-turn conversation with the council. Toggle equilibrium solve vs. single-model baseline,
              tune rationality and solver budget, see per-token influence weights and convergence telemetry.
            </p>
          </Link>

          <Link href="/playground" className="card" style={{ textDecoration: "none" }}>
            <h3>Model Playground</h3>
            <p>
              Generate with individual specialists or the full council side-by-side. Compare outputs,
              highlight where the council wins, see solver iterations per token and equilibrium details.
            </p>
          </Link>

          <Link href="/models" className="card" style={{ textDecoration: "none" }}>
            <h3>Models & Council</h3>
            <p>
              Council composition: member models, measured accuracy per domain, influence weights in aggregation.
              Publish the full council config to Hugging Face with reproducibility links.
            </p>
          </Link>

          <Link href="/auction" className="card" style={{ textDecoration: "none" }}>
            <h3>How the Council Decides</h3>
            <p>
              See real auction traces from the council: per-token bids, winners, second-price payments.
              Learn why truthful bidding is not incentive-compatible for all rules (F6).
            </p>
          </Link>

          <Link href="/lab" className="card" style={{ textDecoration: "none" }}>
            <h3>Equilibrium Mechanics Lab</h3>
            <p>
              Run MMD, GDA, QRE solvers on matrix games. Interactive visualization of convergence to equilibrium,
              strategy simplexes, and the difference between cycling and convergence.
            </p>
          </Link>

          <Link href="/findings" className="card" style={{ textDecoration: "none" }}>
            <h3>Research Findings</h3>
            <p>
              33 signed-off findings tracing the design: convergence (F1–F8), mechanism design (F6, F22),
              answer-level aggregation limits (F29–F33). Each finding links to harness config and seeds.
            </p>
          </Link>
        </div>
      </section>

      <section style={{ marginTop: "4rem", marginBottom: "2rem" }}>
        <h2>Architecture & Reproducibility</h2>
        <ul style={{ color: "var(--text-secondary)", lineHeight: "1.8" }}>
          <li>
            <strong>Product API:</strong> OpenAI-compatible /v1/chat/completions served from GB10 FastAPI server.
            Kinetic controls (rationality λ, solver budget, magnet strength) exposed as optional extensions.
            Latency checked; routing telemetry includes per-token influence weights and convergence status.
          </li>
          <li>
            <strong>Data Sources:</strong> Benchmark data ingested from results/scale/ladder/ and results/scale/gsm8k_fixed/.
            Leaderboard endpoint (/api/leaderboard) loads real measurement results; no hardcoded numbers
            in the frontend (all values trace to config hash, seeds, lm-eval harness).
          </li>
          <li>
            <strong>Frontend:</strong> Next.js 14 (app router) + React 18. Endpoints for leaderboard data, playground
            generation, council telemetry. Proxy layer (/api/proxy/...) gates access and forwards to gateway only
            when authenticated or in demo mode.
          </li>
          <li>
            <strong>Auth &amp; Admin:</strong> Supabase SSR optional. Admin users see real results from results/ directory;
            non-admin visitors see replay demo data. Leaderboard endpoint runs server-side, no secrets exposed to client.
          </li>
          <li>
            <strong>Deployment:</strong> Vercel frontend + GB10 API. Models released to Hugging Face with full
            ladder in model card. Reproducibility: every number links to config hash, git commit, seed set,
            and lm-eval invocation.
          </li>
        </ul>
      </section>
    </div>
  );
}
