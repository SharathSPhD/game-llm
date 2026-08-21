"use client";

import { useState } from "react";
import { Loader2, Plus, RefreshCw } from "lucide-react";

interface Job {
  job_id: string;
  type: string;
  params: Record<string, unknown>;
  status: "queued" | "running" | "completed" | "failed";
  submitted_at: string;
}

export default function StudioPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobType, setJobType] = useState("noop_demo");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [replay, setReplay] = useState(false);

  const handleSubmitJob = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/proxy/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: jobType,
          params: {},
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = (await response.json()) as { job_id: string; replay?: boolean };
      const newJob: Job = {
        job_id: data.job_id,
        type: jobType,
        params: {},
        status: "queued",
        submitted_at: new Date().toISOString(),
      };
      setJobs([newJob, ...jobs]);
      setReplay(!!data.replay);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshJobs = async () => {
    // In a real scenario, we'd fetch job statuses here
    // For now, this is a placeholder
  };

  return (
    <div className="wrap">
      <section>
        <h1>Training Studio</h1>
        <p style={{ color: "var(--text-secondary)", maxWidth: "600px" }}>
          Submit long-running experiments to the job queue. Support for solve-and-audit, hyperparameter
          sweeps, and ablation studies. Monitor status, download results, and trace config hashes to
          reproducible runs on GPU (GB10 live or offline).
        </p>
      </section>

      <div style={{ marginTop: "2rem", display: "grid", gridTemplateColumns: "1fr 2fr", gap: "2rem" }}>
        {/* Submit form */}
        <div className="card">
          <h3>Submit Job</h3>

          <div style={{ marginTop: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div>
              <label>Job Type</label>
              <select value={jobType} onChange={(e) => setJobType(e.target.value)}>
                <option value="noop_demo">No-op Demo</option>
                <option value="solve">Solve & Audit</option>
                <option value="qre_sweep">QRE Sweep</option>
                <option value="auction_eval">Auction Evaluation</option>
              </select>
            </div>

            <button
              className="btn"
              onClick={handleSubmitJob}
              disabled={loading}
              style={{
                marginTop: "1rem",
                justifyContent: "center",
                opacity: loading ? 0.6 : 1,
              }}
            >
              {loading ? (
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

            {replay && (
              <div
                style={{
                  backgroundColor: "rgba(250, 204, 21, 0.1)",
                  border: "1px solid var(--warning)",
                  color: "var(--warning)",
                  padding: "0.75rem",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem",
                }}
              >
                <strong>Demo mode:</strong> Gateway offline, job queued locally
              </div>
            )}
          </div>
        </div>

        {/* Job list */}
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3>Job Queue</h3>
            <button
              className="btn"
              onClick={handleRefreshJobs}
              style={{ padding: "0.5rem 0.75rem" }}
            >
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
                maxHeight: "400px",
                overflowY: "auto",
              }}
            >
              {jobs.map((job) => (
                <div
                  key={job.job_id}
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
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start" }}>
                    <div>
                      <div style={{ fontWeight: 600 }}>{job.job_id.slice(0, 8)}...</div>
                      <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)" }}>
                        Type: <code>{job.type}</code>
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
                        <Loader2
                          size={12}
                          style={{
                            display: "inline-block",
                            marginRight: "0.25rem",
                            animation: "spin 0.8s linear infinite",
                          }}
                        />
                      )}
                      {job.status}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <section style={{ marginTop: "3rem" }}>
        <h2>Studio Features (Roadmap)</h2>
        <ul style={{ color: "var(--text-secondary)", lineHeight: "1.8" }}>
          <li>
            <strong>Job submission:</strong> Submit parameterized experiments with reproducible config hashes.
          </li>
          <li>
            <strong>Status polling:</strong> Real-time or polling-based progress tracking (queued → running → completed).
          </li>
          <li>
            <strong>Result download:</strong> Access outputs, logs, and artifact metadata once jobs complete.
          </li>
          <li>
            <strong>Audit trail:</strong> Full git commit, seed, and parameter tracking for each run.
          </li>
          <li>
            <strong>GPU acceleration:</strong> Jobs orchestrated via SLURM or local executor; PyTorch tensors,
            implicit differentiation (DEQ), memory-efficient solvers.
          </li>
        </ul>
      </section>
    </div>
  );
}
