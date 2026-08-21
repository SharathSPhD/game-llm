"use client";

import { useState } from "react";
import { findings, getCategories, getFindingsByCategory } from "@/lib/findings";
import { ChevronDown } from "lucide-react";

export default function FindingsPage() {
  const [expandedId, setExpandedId] = useState<string | null>("F1");
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const categories = getCategories();
  const displayedFindings = selectedCategory
    ? getFindingsByCategory(selectedCategory)
    : findings;

  return (
    <div className="wrap">
      <section>
        <h1>Research Findings</h1>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Validated findings from Phase 3 experiments (F1–F8). Each finding includes the claim,
          supporting evidence, Tarka verification status, and artifact paths. Results are gated by
          independent recomputation and sign-off.
        </p>
      </section>

      <div style={{ marginTop: "2rem" }}>
        {/* Category filter */}
        <div style={{ marginBottom: "2rem" }}>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem" }}>
            Filter by category:
          </p>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <button
              className="btn"
              onClick={() => setSelectedCategory(null)}
              style={{
                backgroundColor: selectedCategory === null ? "var(--accent)" : "transparent",
                color: selectedCategory === null ? "#0f172a" : "var(--text)",
                borderColor: selectedCategory === null ? "var(--accent)" : "var(--border)",
                padding: "0.5rem 1rem",
                fontSize: "0.875rem",
              }}
            >
              All ({findings.length})
            </button>
            {categories.map((cat) => {
              const count = findings.filter((f) => f.category === cat).length;
              return (
                <button
                  key={cat}
                  className="btn"
                  onClick={() => setSelectedCategory(cat)}
                  style={{
                    backgroundColor: selectedCategory === cat ? "var(--accent)" : "transparent",
                    color: selectedCategory === cat ? "#0f172a" : "var(--text)",
                    borderColor: selectedCategory === cat ? "var(--accent)" : "var(--border)",
                    padding: "0.5rem 1rem",
                    fontSize: "0.875rem",
                  }}
                >
                  {cat} ({count})
                </button>
              );
            })}
          </div>
        </div>

        {/* Findings list */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {displayedFindings.map((finding) => (
            <div
              key={finding.id}
              className="card"
              style={{
                cursor: "pointer",
              }}
              onClick={() =>
                setExpandedId(expandedId === finding.id ? null : finding.id)
              }
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "start",
                  gap: "1rem",
                }}
              >
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "1rem",
                      marginBottom: "0.5rem",
                    }}
                  >
                    <div
                      style={{
                        fontWeight: 700,
                        fontSize: "1.25rem",
                        color: "var(--accent)",
                      }}
                    >
                      {finding.id}
                    </div>
                    <h3 style={{ margin: 0, fontSize: "1.125rem" }}>
                      {finding.title}
                    </h3>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      gap: "1rem",
                      fontSize: "0.75rem",
                      color: "var(--text-tertiary)",
                      marginTop: "0.5rem",
                      flexWrap: "wrap",
                    }}
                  >
                    <span
                      style={{
                        backgroundColor: "var(--bg-tertiary)",
                        padding: "0.25rem 0.75rem",
                        borderRadius: "0.25rem",
                      }}
                    >
                      {finding.category}
                    </span>
                    <span
                      style={{
                        backgroundColor: "rgba(74, 222, 128, 0.1)",
                        padding: "0.25rem 0.75rem",
                        borderRadius: "0.25rem",
                        color: "var(--success)",
                      }}
                    >
                      {finding.deploymentStatus.split(" · ")[0]}
                    </span>
                  </div>
                </div>

                <ChevronDown
                  size={20}
                  style={{
                    transform:
                      expandedId === finding.id
                        ? "rotate(180deg)"
                        : "rotate(0deg)",
                    transition: "transform 0.2s",
                    color: "var(--text-secondary)",
                    flexShrink: 0,
                  }}
                />
              </div>

              {/* Expanded content */}
              {expandedId === finding.id && (
                <div
                  style={{
                    marginTop: "1.5rem",
                    paddingTop: "1.5rem",
                    borderTop: "1px solid var(--border)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "1.5rem",
                  }}
                >
                  <div>
                    <h4>Claim</h4>
                    <p style={{ color: "var(--text-secondary)" }}>
                      {finding.claim}
                    </p>
                  </div>

                  <div>
                    <h4>Evidence</h4>
                    <p style={{ color: "var(--text-secondary)" }}>
                      {finding.evidence}
                    </p>
                  </div>

                  {finding.keyNumbers && Object.keys(finding.keyNumbers).length > 0 && (
                    <div>
                      <h4>Key Numbers</h4>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                          gap: "1rem",
                        }}
                      >
                        {Object.entries(finding.keyNumbers).map(([label, value]) => (
                          <div
                            key={label}
                            style={{
                              backgroundColor: "var(--bg-tertiary)",
                              padding: "1rem",
                              borderRadius: "0.375rem",
                            }}
                          >
                            <div
                              style={{
                                fontSize: "0.75rem",
                                color: "var(--text-tertiary)",
                                fontWeight: 600,
                                textTransform: "uppercase",
                              }}
                            >
                              {label}
                            </div>
                            <div
                              style={{
                                fontSize: "1rem",
                                fontWeight: 600,
                                marginTop: "0.25rem",
                              }}
                            >
                              {value}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div>
                    <h4>Verification & Status</h4>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: "1rem",
                        fontSize: "0.875rem",
                      }}
                    >
                      <div className="panel">
                        <div className="panel-label">Tarka Status</div>
                        <div style={{ marginTop: "0.5rem" }}>
                          {finding.tarkaStatus}
                        </div>
                      </div>
                      <div className="panel">
                        <div className="panel-label">Deployment Status</div>
                        <div style={{ marginTop: "0.5rem" }}>
                          {finding.deploymentStatus}
                        </div>
                      </div>
                    </div>
                  </div>

                  {finding.artifactPath && (
                    <div>
                      <h4>Artifacts</h4>
                      <div
                        style={{
                          backgroundColor: "var(--bg-tertiary)",
                          padding: "0.75rem",
                          borderRadius: "0.375rem",
                          fontFamily: "monospace",
                          fontSize: "0.875rem",
                          color: "var(--accent)",
                          wordBreak: "break-all",
                        }}
                      >
                        {finding.artifactPath}
                      </div>
                    </div>
                  )}

                  {finding.linkedFinding && (
                    <div style={{ fontSize: "0.875rem" }}>
                      <strong>Related:</strong> See also{" "}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setExpandedId(finding.linkedFinding || null);
                        }}
                        style={{
                          background: "none",
                          border: "none",
                          color: "var(--accent)",
                          cursor: "pointer",
                          textDecoration: "underline",
                        }}
                      >
                        {finding.linkedFinding}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <section style={{ marginTop: "3rem" }}>
        <h2>About Tarka Verification</h2>
        <div
          style={{
            backgroundColor: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: "0.5rem",
            padding: "1.5rem",
            lineHeight: "1.8",
          }}
        >
          <p>
            <strong>Tarka</strong> is an independent verification system that recomputes
            findings and audits experimental config hashes, seeds, and git commits. All
            findings in this explorer have undergone Tarka review; <code>VALIDATED</code> status
            means the claim has been independently reproduced. <code>SIGN-OFF PENDING</code> means
            the verification is complete but awaits final operator signature.
          </p>
          <p style={{ marginTop: "1rem" }}>
            Every result includes:
          </p>
          <ul>
            <li>Config hash (SHA-256) for parameter reproducibility</li>
            <li>Random seed pinning for deterministic runs</li>
            <li>Git commit reference for code versioning</li>
            <li>Artifact paths for accessing raw data and logs</li>
          </ul>
        </div>
      </section>
    </div>
  );
}
