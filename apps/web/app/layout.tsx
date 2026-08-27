import type { Metadata } from "next";
import Link from "next/link";
import "../styles/globals.css";
import { NavLinks } from "./components/NavLinks";
import { ThemeToggle } from "./components/ThemeToggle";
import { HealthDot } from "./components/HealthDot";

export const metadata: Metadata = {
  title: {
    default: "EqLM — Equilibrium Language Model Research Platform",
    template: "%s — EqLM",
  },
  description:
    "Equilibrium Lab explores game-theoretic learning dynamics: MMD convergence, QRE homotopy, mechanism auctions, and training studio for reproducible research.",
};

// Stamps the stored theme onto <html> before first paint
const THEME_BOOT = `try{var t=localStorage.getItem('eqlm-theme');if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t)}catch(e){}`;

function BrandMark() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="9.2" stroke="var(--accent)" strokeWidth="1.6" />
      <path
        d="M 8 16 Q 12 12 16 16"
        stroke="var(--accent)"
        strokeWidth="1.6"
        fill="none"
        strokeLinecap="round"
      />
      <circle cx="10" cy="10" r="1.2" fill="var(--accent)" />
      <circle cx="14" cy="10" r="1.2" fill="var(--accent)" />
    </svg>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body>
        <a href="#main" className="skip-link">
          Skip to main content
        </a>
        <nav className="site-nav" aria-label="Primary">
          <Link href="/" className="brand">
            <BrandMark />
            EqLM
          </Link>
          <NavLinks />
          <div className="nav-end">
            <HealthDot />
            <ThemeToggle />
          </div>
        </nav>
        <main id="main">{children}</main>
        <footer className="site-footer">
          <div className="footer-inner">
            <div>
              <p className="footer-blurb">
                <strong>EqLM</strong> is a research platform for equilibrium learning dynamics,
                quantal response equilibria, mechanism design verification, and GPU-accelerated training.
                All results include reproducible configs and are tracked via Tarka verification.
              </p>
            </div>
            <div>
              <h3>Lab</h3>
              <ul>
                <li><Link href="/lab">Equilibrium Lab</Link></li>
                <li><Link href="/qre">QRE Explorer</Link></li>
                <li><Link href="/auction">Auction Playground</Link></li>
              </ul>
            </div>
            <div>
              <h3>Research</h3>
              <ul>
                <li><Link href="/studio">Training Studio</Link></li>
                <li><Link href="/findings">Findings</Link></li>
                <li><a href="https://github.com/sharaths/game-llm">Source</a></li>
              </ul>
            </div>
            <div>
              <h3>Info</h3>
              <ul>
                <li><Link href="/">Overview</Link></li>
                <li><a href="mailto:sharath.ai.colab@gmail.com">Contact</a></li>
              </ul>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
