"use client";

import { useEffect, useState } from "react";
import { Loader2, Plus, RefreshCw, ChevronDown } from "lucide-react";

interface ExperimentTemplate {
  id: string;
  name: string;
  description: string;
  config_yaml?: string;
  script: string;
}

interface Job {
  job_id: string;
  template_id: string;
  status: "queued" | "running" | "completed" | "failed";
  submitted_at: string;
  config_hash?: string;
}

interface LogLine {
  text: string;
}

interface Run {
  dir: string;
  experiment: string;
  config_hash: string;
  git_commit: string;
  metrics: Record<string, any>;
}

export default function StudioPage() {
  const [templates, setTemplates] = useState<ExperimentTemplate[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>("");
  const [overrides, setOverrides] = useState<Record<string, any>>({});
  const [jobs, setJobs] = useState<Job[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [expandedJobId, setExpandedJobId] = useState<string | null>(null);
  const [jobLogs, setJobLogs] = useState<Record<string, string[]>>({});

  useEffect(() => {
    fetchTemplates();
    fetchRuns();
  }, []);

  const fetchTemplates = async () => {
    try {
      const resp = await fetch("/api/proxy/api/experiments");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = (await resp.json()) as { templates: ExperimentTemplate[] };
      setTemplates(data.templates);
      if (data.templates.length > 0) {
        setSelectedTemplate(data.templates[0].id);
      }
    } catch (err) {
      setError(`Failed to load templates: ${String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchRuns = async () => {
    try {
      const resp = await fetch("/api/proxy/api/runs");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = (await resp.json()) as { runs: Run[] };
      setRuns(data.runs);
    } catch (err) {
      console.error("Failed to load runs:", err);
    }
  };

  const fetchJobLog = async (jobId: string) => {
    try {
      const resp = await fetch(`/api/proxy/api/jobs/${jobId}/log?offset=0`);
      if (!resp.ok) return;
      const data = (await resp.json()) as { lines: string[] };
      setJobLogs((prev) => ({ ...prev, [jobId]: data.lines }));
    } catch (err) {
      console.error("Failed to load job log:", err);
    }
  };

  const handleSubmit = async () => {
    if (!selectedTemplate) {
      setError("Please select a template");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const resp = await fetch("/api/proxy/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "experiment",
          template_id: selectedTemplate,
          overrides,
        }),
      });

      if (!resp.ok) {
        const errData = (await resp.json()) as { detail?: string };
        throw new Error(errData.detail || `HTTP ${resp.status}`);
      }

      const data = (await resp.json()) as { job_id: string };
      const newJob: Job = {
        job_id: data.job_id,
        template_id: selectedTemplate,
        status: "queued",
        submitted_at: new Date().toISOString(),
      };
      setJobs([newJob, ...jobs]);
      setOverrides({});
      fetchJobLog(data.job_id);
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleOverrideChange = (key: string, value: any) => {
    setOverrides((prev) => ({
      ...prev,
      [key]: value === "" ? undefined : value,
    }));
  };

  const template = templates.find((t) => t.id === selectedTemplate);

  return (
    <div className="wrap">
      <section>
        <h1>Kinetic Studio — Runs</h1>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Submit experiment jobs to the queue with validated config overrides. Monitor progress in real time,
          access logs, and track reproducibility via config hash and git commit.
        </p>
      </section>

      <div style={{ marginTop: "2rem", display: "grid", gridTemplateColumns: "1fr 2fr", gap: "2rem" }}>
        {/* Submit form */}
        <div className="card">
          <h3>Submit Experiment</h3>

          {loading && <p style={{ color: "var(--text-secondary)" }}>Loading templates...</p>}

          {!loading && (
            <div style={{ marginTop: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div>
                <label htmlFor="template-select">Experiment Template</label>
                <select
                  id="template-select"
                  value={selectedTemplate}
                  onChange={(e) => setSelectedTemplate(e.target.value)}
                >
                  {templates.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </div>

              {template && (
                <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
                  <p>{template.description}</p>
                </div>
              )}

              {/* Editable overrides */}
              <div style={{ paddingTop: "1rem", borderTop: "1px solid var(--border)" }}>
                <p style={{ fontSize: "0.875rem", fontWeight: 600, marginBottom: "0.75rem" }}>Overrides</p>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                  <div>
                    <label htmlFor="num-steps" style={{ fontSize: "0.875rem" }}>
                      Training Steps (≤25000)
                    </label>
                    <input
                      id="num-steps"
                      type="number"
                      min={1}
                      max={25000}
                      placeholder="e.g., 100"
                      value={overrides["training.num_steps"] ?? ""}
                      onChange={(e) => handleOverrideChange("training.num_steps", parseInt(e.target.value))}
                      style={{ width: "100%" }}
                    />
                  </div>

                  <div>
                    <label htmlFor="seed" style={{ fontSize: "0.875rem" }}>
                      Random Seed
                    </label>
                    <input
                      id="seed"
                      type="number"
                      min={1}
                      placeholder="e.g., 42"
                      value={overrides["training.seed"] ?? ""}
                      onChange={(e) => handleOverrideChange("training.seed", parseInt(e.target.value))}
                      style={{ width: "100%" }}
                    />
                  </div>

                  <div>
                    <label htmlFor="subset-size" style={{ fontSize: "0.875rem" }}>
                      Data Subset Size
                    </label>
                    <input
                      id="subset-size"
                      type="number"
                      min={1000}
                      placeholder="e.g., 3300000"
                      value={overrides["data.subset_size"] ?? ""}
                      onChange={(e) => handleOverrideChange("data.subset_size", parseInt(e.target.value))}
                      style={{ width: "100%" }}
                    />
                  </div>
                </div>
              </div>

              <button
                className="btn"
                onClick={handleSubmit}
                disabled={submitting || !selectedTemplate}
                style={{
                  marginTop: "1rem",
                  justifyContent: "center",
                  opacity: submitting || !selectedTemplate ? 0.6 : 1,
                }}
              >
                {submitting ? (
                  <>
                    <Loader2 size={16} style={{ animation: "spin 0.8s linear infinite" }} />
                    Submitting...
                  </>
                ) : (
                  <>
                    <Plus size={16} />
                    Submit Job
                  </>
                )}
              </button>

              {error && (
                <div
                  style={{
                    backgroundColor: "rgba(248, 113, 113, 0.1)",
                    border: "1px solid var(--error)",
                    color: "var(--error)",
                    padding: "0.75rem",
                    borderRadius: "0.375rem",
                    fontSize: "0.875rem",
                  }}
                >
                  {error}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Job list */}
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3>Job Queue</h3>
            <button className="btn" onClick={() => fetchTemplates()} style={{ padding: "0.5rem 0.75rem" }}>
              <RefreshCw size={16} />
            </button>
          </div>

          {jobs.length === 0 ? (
            <div
              style={{
                marginTop: "1.5rem",
                padding: "2rem",
                textAlign: "center",
                color: "var(--text-tertiary)",
                backgroundColor: "var(--bg-tertiary)",
                borderRadius: "0.375rem",
              }}
            >
              <p>No jobs submitted yet</p>
            </div>
          ) : (
            <div
              style={{
                marginTop: "1.5rem",
                display: "flex",
                flexDirection: "column",
                gap: "1rem",
                maxHeight: "600px",
                overflowY: "auto",
              }}
            >
              {jobs.map((job) => (
                <div key={job.job_id}>
                  <div
                    onClick={() => {
                      setExpandedJobId(expandedJobId === job.job_id ? null : job.job_id);
                      if (expandedJobId !== job.job_id && !jobLogs[job.job_id]) {
                        fetchJobLog(job.job_id);
                      }
                    }}
                    style={{
                      padding: "1rem",
                      backgroundColor: "var(--bg-tertiary)",
                      borderRadius: "0.375rem",
                      borderLeft: `4px solid ${
                        job.status === "completed"
                          ? "var(--success)"
                          : job.status === "failed"
                            ? "var(--error)"
                            : job.status === "running"
                              ? "var(--info)"
                              : "var(--warning)"
                      }`,
                      cursor: "pointer",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "start",
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: "0.875rem", fontFamily: "monospace" }}>
                        {job.job_id.slice(0, 8)}
                      </div>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "0.25rem" }}>
                        {job.template_id}
                      </div>
                      <div
                        style={{
                          fontSize: "0.75rem",
                          color: "var(--text-tertiary)",
                          marginTop: "0.25rem",
                        }}
                      >
                        {new Date(job.submitted_at).toLocaleString()}
                      </div>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <div
                        style={{
                          padding: "0.375rem 0.75rem",
                          backgroundColor: "var(--bg-secondary)",
                          borderRadius: "0.25rem",
                          fontSize: "0.75rem",
                          fontWeight: 600,
                          textTransform: "uppercase",
                          color:
                            job.status === "completed"
                              ? "var(--success)"
                              : job.status === "failed"
                                ? "var(--error)"
                                : "var(--text-secondary)",
                        }}
                      >
                        {job.status === "running" && (
                          <Loader2 size={12} style={{ display: "inline-block", marginRight: "0.25rem" }} />
                        )}
                        {job.status}
                      </div>
                      <ChevronDown
                        size={16}
                        style={{
                          transform: expandedJobId === job.job_id ? "rotate(180deg)" : "rotate(0deg)",
                          transition: "transform 0.2s",
                        }}
                      />
                    </div>
                  </div>

                  {expandedJobId === job.job_id && jobLogs[job.job_id] && (
                    <div
                      style={{
                        marginTop: "0.5rem",
                        padding: "1rem",
                        backgroundColor: "var(--bg-tertiary)",
                        borderRadius: "0.375rem",
                        fontSize: "0.75rem",
                        fontFamily: "monospace",
                        maxHeight: "300px",
                        overflowY: "auto",
                        color: "var(--text-secondary)",
                      }}
                    >
                      {jobLogs[job.job_id].map((line, i) => (
                        <div key={i}>{line || " "}</div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Runs registry */}
      <section style={{ marginTop: "3rem" }}>
        <h2>Run Registry</h2>
        <p style={{ color: "var(--text-secondary)", marginBottom: "1.5rem" }}>
          Completed runs with config hashes for reproducibility tracking.
        </p>

        {runs.length === 0 ? (
          <div style={{ color: "var(--text-tertiary)" }}>No runs found</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "0.875rem",
              }}
            >
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ padding: "0.75rem", textAlign: "left", fontWeight: 600 }}>Experiment</th>
                  <th style={{ padding: "0.75rem", textAlign: "left", fontWeight: 600 }}>Config Hash</th>
                  <th style={{ padding: "0.75rem", textAlign: "left", fontWeight: 600 }}>Git Commit</th>
                  <th style={{ padding: "0.75rem", textAlign: "left", fontWeight: 600 }}>Key Metrics</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 10).map((run, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "0.75rem" }}>{run.experiment}</td>
                    <td
                      style={{
                        padding: "0.75rem",
                        fontFamily: "monospace",
                        color: "var(--text-secondary)",
                      }}
                    >
                      {run.config_hash.slice(0, 8)}...
                    </td>
                    <td
                      style={{
                        padding: "0.75rem",
                        fontFamily: "monospace",
                        color: "var(--text-secondary)",
                      }}
                    >
                      {run.git_commit.slice(0, 7)}
                    </td>
                    <td style={{ padding: "0.75rem", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                      {Object.entries(run.metrics)
                        .slice(0, 2)
                        .map(([k, v]) => `${k}=${typeof v === "number" ? v.toFixed(3) : v}`)
                        .join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
