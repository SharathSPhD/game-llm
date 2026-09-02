"use client";

import { DemoBadge } from "@/app/components/DemoBadge";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface CheckpointConfig {
  d_model: number | null;
  n_heads: number | null;
  d_ff: number | null;
  vocab_size: number | null;
  map_form: string | null;
}

interface RunMetadata {
  config_sha: string;
  git_commit: string;
}

interface ModelEntry {
  path: string;
  size_mb: number;
  config: CheckpointConfig;
  model_class: string;
  params_estimate: number;
  run: RunMetadata;
}

interface PublishState {
  loading: boolean;
  error: string | null;
  success: boolean;
  repo_url?: string;
  selectedCheckpoint?: string;
}

export default function ModelsPage() {
  const router = useRouter();
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [publishState, setPublishState] = useState<PublishState>({
    loading: false,
    error: null,
    success: false,
  });
  const [repoId, setRepoId] = useState("");
  const [showPublishDialog, setShowPublishDialog] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  // Fetch models registry
  useEffect(() => {
    async function fetchModels() {
      try {
        const resp = await fetch("/api/proxy/api/models");
        if (!resp.ok) {
          if (resp.status === 401) {
            setError("Authentication required. Please sign in.");
          } else {
            setError(`Failed to fetch models: ${resp.statusText}`);
          }
          return;
        }
        const raw = await resp.json();
        // The backend returns a bare array; the replay stub wraps it as {models}.
        setModels(Array.isArray(raw) ? raw : Array.isArray(raw?.models) ? raw.models : []);
      } catch (err) {
        setError(`Error fetching models: ${err}`);
      } finally {
        setLoading(false);
      }
    }

    fetchModels();
  }, []);

  const handlePublishClick = (path: string) => {
    setSelectedPath(path);
    setRepoId("");
    setPublishState({ loading: false, error: null, success: false });
    setShowPublishDialog(true);
  };

  const handlePublish = async () => {
    if (!selectedPath || !repoId.trim()) {
      setPublishState({
        loading: false,
        error: "Please enter a repo ID",
        success: false,
      });
      return;
    }

    if (!repoId.includes("/")) {
      setPublishState({
        loading: false,
        error: "Repo ID must be in format: owner/name",
        success: false,
      });
      return;
    }

    setPublishState({ loading: true, error: null, success: false });

    try {
      const resp = await fetch("/api/proxy/api/models/publish", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_path: selectedPath,
          repo_id: repoId,
        }),
      });

      const data = await resp.json();

      if (!resp.ok) {
        setPublishState({
          loading: false,
          error: data.detail || "Failed to publish",
          success: false,
        });
        return;
      }

      setPublishState({
        loading: false,
        error: null,
        success: true,
        repo_url: data.repo_url,
        selectedCheckpoint: selectedPath,
      });

      // Auto-close dialog after 3 seconds
      setTimeout(() => {
        setShowPublishDialog(false);
      }, 3000);
    } catch (err) {
      setPublishState({
        loading: false,
        error: `Error: ${err}`,
        success: false,
      });
    }
  };

  if (loading) {
    return (
      <div className="page">
        <h1>Models Registry</h1>
        <DemoBadge what="the registry snapshot" />
      <p>Loading...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <h1>Models Registry</h1>
        <div className="error-box">{error}</div>
      </div>
    );
  }

  return (
    <div className="page">
      <h1>Models Registry</h1>
      <p className="subtitle">
        Checkpoints from completed training runs. Publishing from the app was retired at closure (ADR 0011); the released models are on Hugging Face under qbz506.
      </p>

      {models.length === 0 ? (
        <p>No checkpoints found in results/</p>
      ) : (
        <div className="models-table-container">
          <table className="models-table">
            <thead>
              <tr>
                <th>Path</th>
                <th>Model Class</th>
                <th>Architecture</th>
                <th>Params</th>
                <th>Size (MB)</th>
                <th>Config</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, idx) => (
                <tr key={idx}>
                  <td className="mono">{model.path}</td>
                  <td>{model.model_class}</td>
                  <td>
                    {model.config.d_model && (
                      <>
                        d={model.config.d_model}
                        {model.config.n_heads && `, h=${model.config.n_heads}`}
                      </>
                    )}
                  </td>
                  <td className="align-right">
                    {(model.params_estimate / 1e6).toFixed(1)}M
                  </td>
                  <td className="align-right">{model.size_mb}</td>
                  <td className="mono small">
                    {model.run.config_sha.slice(0, 8)}
                  </td>
                  <td>
                    <span style={{ color: "var(--text-tertiary)", fontSize: "0.8rem" }}>publishing retired</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Publish Dialog */}
      {showPublishDialog && (
        <div className="dialog-overlay" onClick={() => setShowPublishDialog(false)}>
          <div className="dialog" onClick={(e) => e.stopPropagation()}>
            <h2>Publish to Hugging Face</h2>
            {publishState.success ? (
              <div className="success-box">
                <p>✓ Published successfully!</p>
                {publishState.repo_url && (
                  <p>
                    <a
                      href={publishState.repo_url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      View on HF Hub →
                    </a>
                  </p>
                )}
              </div>
            ) : (
              <>
                <p className="dialog-text">Checkpoint: {selectedPath}</p>
                <div className="form-group">
                  <label>Repo ID (owner/name):</label>
                  <input
                    type="text"
                    placeholder="e.g., kinetic-ai/eqlm-babylm-10m"
                    value={repoId}
                    onChange={(e) => setRepoId(e.target.value)}
                    disabled={publishState.loading}
                  />
                </div>
                {publishState.error && (
                  <p className="error-text">{publishState.error}</p>
                )}
                <div className="dialog-actions">
                  <button
                    className="btn-cancel"
                    onClick={() => setShowPublishDialog(false)}
                    disabled={publishState.loading}
                  >
                    Cancel
                  </button>
                  <button
                    className="btn-primary"
                    onClick={handlePublish}
                    disabled={publishState.loading}
                  >
                    {publishState.loading ? "Publishing..." : "Publish"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <style jsx>{`
        .models-table-container {
          overflow-x: auto;
          margin: 1.5rem 0;
        }

        .models-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.9rem;
        }

        .models-table th,
        .models-table td {
          padding: 0.75rem;
          text-align: left;
          border-bottom: 1px solid var(--border-color);
        }

        .models-table th {
          background-color: var(--bg-secondary);
          font-weight: 600;
          position: sticky;
          top: 0;
        }

        .models-table tbody tr:hover {
          background-color: var(--bg-secondary);
        }

        .mono {
          font-family: monospace;
          font-size: 0.85rem;
          word-break: break-all;
        }

        .small {
          font-size: 0.8rem;
        }

        .align-right {
          text-align: right;
        }

        .btn-small {
          padding: 0.4rem 0.8rem;
          font-size: 0.85rem;
          background: var(--accent);
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-small:hover {
          background: var(--accent-hover);
        }

        .btn-small:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .dialog-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .dialog {
          background: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: 8px;
          padding: 2rem;
          max-width: 500px;
          width: 90%;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .dialog h2 {
          margin-top: 0;
          margin-bottom: 1rem;
        }

        .dialog-text {
          font-size: 0.9rem;
          color: var(--text-secondary);
          word-break: break-all;
          margin: 0.5rem 0;
        }

        .form-group {
          margin: 1rem 0;
        }

        .form-group label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: 500;
        }

        .form-group input {
          width: 100%;
          padding: 0.5rem;
          border: 1px solid var(--border-color);
          border-radius: 4px;
          font-size: 0.95rem;
          background: var(--bg-input);
          color: var(--text-primary);
        }

        .form-group input:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .error-text {
          color: var(--error);
          font-size: 0.9rem;
          margin: 0.5rem 0;
        }

        .success-box {
          background: var(--success-bg);
          color: var(--success);
          padding: 1rem;
          border-radius: 4px;
          margin: 1rem 0;
        }

        .success-box p {
          margin: 0.5rem 0;
        }

        .success-box a {
          color: var(--success);
          text-decoration: underline;
        }

        .error-box {
          background: var(--error-bg);
          color: var(--error);
          padding: 1rem;
          border-radius: 4px;
          margin: 1rem 0;
        }

        .dialog-actions {
          display: flex;
          gap: 1rem;
          margin-top: 1.5rem;
          justify-content: flex-end;
        }

        .btn-cancel {
          padding: 0.5rem 1rem;
          background: var(--bg-secondary);
          color: var(--text-primary);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-cancel:hover {
          background: var(--bg-tertiary);
        }

        .btn-cancel:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-primary {
          padding: 0.5rem 1rem;
          background: var(--accent);
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-primary:hover {
          background: var(--accent-hover);
        }

        .btn-primary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .subtitle {
          color: var(--text-secondary);
          margin-bottom: 1.5rem;
        }
      `}</style>
    </div>
  );
}
