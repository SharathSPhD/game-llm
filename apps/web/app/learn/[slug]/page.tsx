import Link from "next/link";
import { learnSections } from "@/lib/learn-content";
import { notFound } from "next/navigation";

export const metadata = {
  title: "Learn — EqLM",
};

export async function generateStaticParams() {
  return learnSections.map((section) => ({
    slug: section.slug,
  }));
}

export default function LearnArticle({ params }: { params: { slug: string } }) {
  const section = learnSections.find((s) => s.slug === params.slug);
  if (!section) notFound();

  const currentIndex = learnSections.findIndex((s) => s.slug === params.slug);
  const prevSection = currentIndex > 0 ? learnSections[currentIndex - 1] : null;
  const nextSection = currentIndex < learnSections.length - 1 ? learnSections[currentIndex + 1] : null;

  return (
    <div className="page wrap-narrow">
      <Link href="/learn" style={{ color: "var(--text-2)", fontSize: "var(--text-base)", marginBottom: "var(--space-4)", display: "inline-block" }}>
        ← Back to Learn
      </Link>

      <article style={{ marginTop: "var(--space-5)" }}>
        <header style={{ marginBottom: "var(--space-6)" }}>
          <p className="eyebrow">{section.id.toUpperCase().replace(/-/g, " ")}</p>
          <h1 style={{ marginTop: "var(--space-2)", marginBottom: "var(--space-4)" }}>{section.title}</h1>
          <div
            style={{
              background: "var(--accent-soft)",
              border: "1px solid var(--accent-line)",
              borderRadius: "var(--radius)",
              padding: "var(--space-4)",
              marginBottom: "var(--space-6)",
            }}
          >
            <p style={{ margin: 0, fontSize: "var(--text-md)", color: "var(--text)", lineHeight: "1.5" }}>
              <strong>Takeaway:</strong> {section.takeaway}
            </p>
          </div>
        </header>

        <div
          style={{
            fontSize: "var(--text-base)",
            lineHeight: "1.75",
            color: "var(--text)",
            marginBottom: "var(--space-7)",
          }}
        >
          {section.content.split("\n\n").map((paragraph, idx) => (
            <p key={idx} style={{ marginBottom: "var(--space-4)" }}>
              {paragraph.split("\n").map((line, lineIdx) => {
                if (line.startsWith("**") && line.endsWith(":**")) {
                  return (
                    <div key={lineIdx}>
                      <strong style={{ fontSize: "var(--text-md)" }}>{line.replace(/\*\*/g, "")}</strong>
                    </div>
                  );
                }
                if (line.startsWith("- ")) {
                  return (
                    <div key={lineIdx} style={{ marginLeft: "1rem", marginBottom: "0.5rem" }}>
                      {line.slice(2)}
                    </div>
                  );
                }
                return (
                  <div key={lineIdx}>
                    {line.split(/(\*\*[^*]+\*\*)/g).map((part, partIdx) => {
                      if (part.startsWith("**") && part.endsWith("**")) {
                        return <strong key={partIdx}>{part.slice(2, -2)}</strong>;
                      }
                      return <span key={partIdx}>{part}</span>;
                    })}
                  </div>
                );
              })}
            </p>
          ))}
        </div>
      </article>

      <nav
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "var(--space-4)",
          marginTop: "var(--space-7)",
          paddingTop: "var(--space-6)",
          borderTop: "1px solid var(--border)",
        }}
      >
        {prevSection ? (
          <Link href={`/learn/${prevSection.slug}`} className="card" style={{ textDecoration: "none" }}>
            <p style={{ margin: 0, fontSize: "var(--text-xs)", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--space-2)" }}>
              Previous
            </p>
            <p style={{ margin: 0, fontSize: "var(--text-base)", color: "var(--accent)", fontWeight: 550 }}>
              {prevSection.title}
            </p>
          </Link>
        ) : (
          <div />
        )}
        {nextSection ? (
          <Link
            href={`/learn/${nextSection.slug}`}
            className="card"
            style={{ textDecoration: "none", textAlign: nextSection ? "right" : "left" }}
          >
            <p style={{ margin: 0, fontSize: "var(--text-xs)", color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "var(--space-2)" }}>
              Next
            </p>
            <p style={{ margin: 0, fontSize: "var(--text-base)", color: "var(--accent)", fontWeight: 550 }}>
              {nextSection.title}
            </p>
          </Link>
        ) : (
          <div />
        )}
      </nav>
    </div>
  );
}
