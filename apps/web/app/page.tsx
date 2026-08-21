import Link from "next/link";

export default function Home() {
  return (
    <div className="wrap">
      <section className="landing-hero">
        <div>
          <p className="eyebrow">Equilibrium Learning Mechanisms (EqLM)</p>
          <h1>When both players optimize, convergence matters.</h1>
          <p className="lede">
            Game-theoretic learning dynamics lie at the core of multi-agent systems: from mechanism design
            to neural network training. EqLM provides a research platform to study convergence via
            Magnetic Mirror Descent (MMD), Quantal Response Equilibria (QRE), mechanism truthfulness,
            and GPU-accelerated job orchestration — with full reproducibility tracking.
          </p>
          <div className="hero-actions">
            <Link href="/lab" className="btn" data-primary="true">
              Open Equilibrium Lab
            </Link>
            <Link href="/findings" className="btn">
              View Research Findings
            </Link>
            <Link href="/studio" className="btn">
              Launch Training Studio
            </Link>
          </div>
          <div className="hero-stats">
            <div className="hero-stat">
              <div className="panel-label">Core Games</div>
              <div className="reading">4</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Validated Findings</div>
              <div className="reading">8</div>
            </div>
            <div className="hero-stat">
              <div className="panel-label">Reproducibility</div>
              <div className="reading">
                <Link href="/findings">Tarka</Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "4rem" }}>
        <h2>The EqLM thesis</h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "2rem",
            marginTop: "2rem",
          }}
        >
          <div className="card">
            <h3>Convergence to Equilibrium</h3>
            <p>
              Beyond GDA&apos;s cycling behavior: Magnetic Mirror Descent with fixed or adaptive references
              reaches equilibrium on symmetric and asymmetric games. We trace the geometry of fixed
              points and validate log-linear convergence.
            </p>
          </div>

          <div className="card">
            <h3>QRE Homotopy</h3>
            <p>
              Quantal response equilibria interpolate between randomness and perfect rationality. Warm-started
              homotopy paths accelerate solver convergence; adaptive damping handles high-rationality limits.
            </p>
          </div>

          <div className="card">
            <h3>Mechanism Truthfulness</h3>
            <p>
              Token auctions studied via second-price vs. weighted aggregation. Empirical regret validates
              truthfulness: second-price is VCG-exact, weighted aggregation enables profitable manipulation.
            </p>
          </div>

          <div className="card">
            <h3>Reproducible Science</h3>
            <p>
              Every result includes config hash, seed, git commit. Tarka verification gates findings; intermediate
              artifacts are stored and indexed. Full audit trail from hypothesis to sign-off.
            </p>
          </div>

          <div className="card">
            <h3>GPU Acceleration</h3>
            <p>
              Training Studio orchestrates long-running jobs on CUDA. Experiments marshal PyTorch tensors,
              leverage implicit differentiation (DEQ), and track memory-efficient solver stacks.
            </p>
          </div>

          <div className="card">
            <h3>Interactive Exploration</h3>
            <p>
              Tune hyperparameters in real time: method, learning rate, magnetic strength, rationality range.
              Animated convergence plots and strategy simplexes bring game dynamics to life.
            </p>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "4rem" }}>
        <h2>Explore EqLM</h2>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Choose your path below to dive into equilibrium learning:
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
              Run MMD, GDA, and QRE solvers on RPS, matching pennies, and biased games.
              Tune learning rate, magnetic strength, and step count; watch strategies converge.
            </p>
          </Link>

          <Link href="/qre" className="card" style={{ textDecoration: "none" }}>
            <h3>QRE Explorer</h3>
            <p>
              Trace quantal response paths as rationality varies. See how agents shift from
              uniform randomness to best-response play; measure exploitability along the path.
            </p>
          </Link>

          <Link href="/auction" className="card" style={{ textDecoration: "none" }}>
            <h3>Auction Playground</h3>
            <p>
              Design multi-agent token auctions. Compare second-price (truthful) vs.
              weighted aggregation; measure manipulation regret and output distributions.
            </p>
          </Link>

          <Link href="/studio" className="card" style={{ textDecoration: "none" }}>
            <h3>Training Studio</h3>
            <p>
              Submit long-running jobs: solve-and-audit, ablations, hyperparameter sweeps.
              Monitor status, download results, and trace config hashes to reproducible runs.
            </p>
          </Link>

          <Link href="/findings" className="card" style={{ textDecoration: "none" }}>
            <h3>Research Findings</h3>
            <p>
              Browse validated findings (F1–F8): convergence claims, mechanism properties,
              solver accelerations. All Tarka-reviewed with artifact paths and key numbers.
            </p>
          </Link>
        </div>
      </section>

      <section style={{ marginTop: "4rem", marginBottom: "2rem" }}>
        <h2>Technical Notes</h2>
        <ul style={{ color: "var(--text-secondary)", lineHeight: "1.8" }}>
          <li>
            <strong>Backend:</strong> FastAPI server (app/server.py) handles /api/solve, /api/qre_path,
            /api/auction, /api/jobs endpoints. Authentication via bearer token (GATEWAY_SECRET).
          </li>
          <li>
            <strong>Frontend:</strong> Next.js 14 (app router) + React 18 + TailwindCSS. API proxy routes
            forward requests to gateway; replay mode serves canned demo data when offline.
          </li>
          <li>
            <strong>Auth:</strong> Supabase SSR (optional). When NEXT_PUBLIC_SUPABASE_URL is set, sessions
            are maintained; otherwise auth softly disables and all pages are readable.
          </li>
          <li>
            <strong>Reproducibility:</strong> Results stored in results/ directory; each experiment includes
            config hash, seed, and git commit. Tarka verification gates sign-off.
          </li>
          <li>
            <strong>Deployment:</strong> Vercel (root dir apps/web). Environment variables:
            NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY, GATEWAY_URL, GATEWAY_SECRET.
          </li>
        </ul>
      </section>
    </div>
  );
}
