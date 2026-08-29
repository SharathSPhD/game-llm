import Link from "next/link";

export const metadata = {
  title: "Kinetic AI — Equilibrium Language Model",
};

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
            compresses parameters, not compute. A council of routed models is a separate systems result,
            beating single models by 8 points at 1.26× cost and only when its members have complementary
            strengths (F41–F43).
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
              <div className="panel-label">EqLM vs explicit (F24, corrected F44)</div>
              <div className="reading">0.991</div>
              <div className="panel-note">Ratio at matched params and iterations — but 4.92× the compute; at equal FLOPs, 0.72</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Council vs Baseline (F41)</div>
              <div className="reading">+8.33pp</div>
              <div className="panel-note">0.6194 vs 0.5361, z=4.42, pre-registered, conditional on non-domination</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Council Cost (F41)</div>
              <div className="reading">1.26×</div>
              <div className="panel-note">Expected generations per request; 4.1× resident memory for four models</div>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "var(--space-8)" }}>
        <h2>The Paradigm</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "var(--space-5)", marginTop: "var(--space-5)" }}>
          <div className="card">
            <h3>EqLM: Equilibrium Depth</h3>
            <p>
              A language model whose effective depth is not a fixed architecture choice but a solved game:
              at each token, the model computes a fixed point, spending as many iterations as that token&apos;s
              prediction difficulty requires. At matched parameters (121M) it reaches 0.991 of an explicit
              transformer (F24). The audit that followed (F44) found that budget matched parameters and
              iteration count, not arithmetic: the tied block is 4.92× costlier per iteration because
              param-matching forces width, and at equal FLOPs the ratio falls to 0.72. The honest reading is
              that weight-tying buys parameter efficiency and pays for it in compute.
            </p>
          </div>

          <div className="card">
            <h3>Adaptive Per-Token Depth</h3>
            <p>
              An explicit stack spends fixed depth on every token; an equilibrium model can spend few
              iterations on an easy token and many on a hard one. Measured (exp31, F44), uneven spending
              scores 0.681 against uniform 0.684 at the same mean depth — it works and buys nothing. What it
              does buy is graceful degradation: 0.93 of baseline at half the depth, 0.72 at a sixth, which
              makes the model usable at any compute budget rather than better at one.
            </p>
          </div>

          <div className="card">
            <h3>Council as Systems Result (Conditional)</h3>
            <p>
              A council of four Qwen2.5 variants routed by a calibrated lookup table beats the baseline
              0.6194 vs 0.5361 (+8.33pp, F41). But this is a systems advantage, not an architecture advance,
              and it is conditional: it holds only when different members dominate different domains (F42).
              On a second council with a dominant member, the system reduces exactly to that member (F43).
              Cost: 1.26× expected generations and 4.1× resident memory.
            </p>
          </div>

          <div className="card">
            <h3>Baseline Ladder (Reproducible)</h3>
            <p>
              Real measurements on Qwen2.5-1.5B (generalist, coder, math specialist, Qwen3) across MMLU,
              ARC, HellaSwag, GSM8K, and mixed domains. Every number traces to config hashes, seeds, and
              our GB10 harness (F28, F33). Pre-registration and adversarial audit corrected false claims
              about oracle headroom before publication (F32).
            </p>
          </div>

          <div className="card">
            <h3>Game-Theoretic Foundations</h3>
            <p>
              Magnetic Mirror Descent converges linearly where gradient descent cycles (F1); second-price
              token auctions are exactly truthful (F6); Quantal Response equilibrium paths smoothly
              interpolate rationality (F7). These mechanisms are not rhetorical flourish—they are measured,
              Tarka-reviewed, and integrated into the core.
            </p>
          </div>

          <div className="card">
            <h3>Honest Null Findings</h3>
            <p>
              When a measurement fails, it ships with diagnosis and interpretation. Answer-level equilibrium
              mechanisms proved indistinguishable from averaging (F29–F31). Magnetic thresholds do not help
              (F40). Mechanisms are documented at their real size, not silence or reinterpretation. The council&apos;s
              eight-point win exists alongside its precondition and cost (F41–F43).
            </p>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "4rem" }}>
        <h2>Explore Results &amp; Tools</h2>
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
            <h3>Architecture Results</h3>
            <p>
              EqLM single-model results (F24: parity at matched params) and council composition (member models,
              measured per-domain accuracy). See where routing helps (F42: non-dominated members) and where it
              reduces to the best member (F43). Full configs release to Hugging Face with reproducibility links.
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
              Validated findings from convergence (F1–F8), mechanism design (F6), answer-level limits (F29–F31),
              EqLM paradigm (F24: parity at matched params), and council systems result (F41–F43: pre-registered
              confirmation, decomposition, generalization failure). Each links to config hash, seeds, and harness.
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
