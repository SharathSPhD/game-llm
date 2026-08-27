import Link from "next/link";
import { learnSections } from "@/lib/learn-content";

export const metadata = {
  title: "Learn — EqLM",
  description: "Understanding equilibrium learning dynamics, convergence theory, and reproducible research.",
};

export default function LearnIndex() {
  return (
    <div className="page wrap">
      <section style={{ marginBottom: "var(--space-7)" }}>
        <h1 style={{ marginBottom: "var(--space-3)" }}>Learn EqLM</h1>
        <p className="lede" style={{ marginBottom: "var(--space-5)" }}>
          Seven short explainers that bridge game theory, equilibrium computation, and practical training.
          Every number comes from validated findings. Start anywhere.
        </p>
      </section>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(18rem, 1fr))", gap: "var(--space-4)" }}>
        {learnSections.map((section) => (
          <Link
            key={section.id}
            href={`/learn/${section.slug}`}
            className="entry-card"
          >
            <h3>{section.title}</h3>
            <p>{section.description}</p>
            <div className="entry-go">Read more →</div>
          </Link>
        ))}
      </div>

      <section style={{ marginTop: "var(--space-7)", paddingTop: "var(--space-6)", borderTop: "1px solid var(--border)" }}>
        <h2 style={{ marginBottom: "var(--space-3)" }}>How to Read</h2>
        <div style={{ maxWidth: "50rem", color: "var(--text-2)", lineHeight: "1.8" }}>
          <p>
            Each section pairs a real finding (or set of findings) with its meaning. We don&apos;t explain the theory first
            and then show the evidence; instead, we start with the discovery and build backward.
          </p>
          <p>
            <strong>What you&apos;ll find:</strong> The claim, the numbers (with experiment IDs like F1, F22), and the implication
            for future work or understanding. No invented examples. Every number is traceable to research/memory/findings.md.
          </p>
          <p>
            <strong>One sentence per section:</strong> Each ends with a takeaway&mdash;the one thing that should stick.
            Use these to navigate and decide where to dive deeper.
          </p>
        </div>
      </section>
    </div>
  );
}
