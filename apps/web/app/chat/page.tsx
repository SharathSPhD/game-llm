"use client";

import { useState, useEffect, useRef } from "react";
import { DemoBadge } from "../components/DemoBadge";

interface ModelEntry {
  path: string;
  size_mb: number;
  config: {
    d_model: number | null;
    n_heads: number | null;
    d_ff: number | null;
    vocab_size: number | null;
    map_form: string | null;
  };
  model_class: string;
  params_estimate: number;
  run: {
    config_sha: string;
    git_commit: string;
  };
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface GenerationResult {
  text: string;
  tokens: Array<{
    token_str: string;
    solver_iters: number | null;
  }>;
  mean_iters: number;
  wall_ms: number;
}

export default function ChatPage() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [selectedModel, setSelectedModel] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful AI assistant. Answer questions clearly and concisely."
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [userInput, setUserInput] = useState("");

  const [maxNewTokens, setMaxNewTokens] = useState(32);
  const [solverBudget, setSolverBudget] = useState(6);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);

  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch available models
  useEffect(() => {
    async function fetchModels() {
      try {
        const resp = await fetch("/api/proxy/api/models");
        if (!resp.ok) {
          setModelsError(`Failed to fetch models: ${resp.statusText}`);
          return;
        }
        const raw = await resp.json();
        // The backend returns a bare array; the replay stub wraps it as {models}.
        const data = Array.isArray(raw) ? raw : Array.isArray(raw?.models) ? raw.models : [];
        setModels(data);
        if (data.length > 0) {
          setSelectedModel(data[0].path);
        }
      } catch (err) {
        setModelsError(`Error fetching models: ${err}`);
      } finally {
        setModelsLoading(false);
      }
    }

    fetchModels();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const buildPrompt = (): string => {
    let prompt = systemPrompt.trim();
    if (prompt && !prompt.endsWith("\n")) {
      prompt += "\n";
    }

    for (const msg of messages) {
      if (msg.role === "user") {
        prompt += `\nUser: ${msg.content}`;
      } else {
        prompt += `\nAssistant: ${msg.content}`;
      }
    }

    prompt += "\n\nAssistant:";
    return prompt;
  };

  const handleSend = async () => {
    const inputTrimmed = userInput.trim();
    if (!inputTrimmed) return;

    if (!selectedModel) {
      setError("Please select a model");
      return;
    }

    const newMessages: Message[] = [...messages, { role: "user", content: inputTrimmed }];
    setMessages(newMessages);
    setUserInput("");
    setGenerating(true);
    setError(null);

    try {
      const prompt = buildPrompt();

      const resp = await fetch("/api/proxy/api/playground/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_path: selectedModel,
          prompt: prompt,
          max_new_tokens: maxNewTokens,
          solver_budget: solverBudget,
          temperature,
          top_k: topK,
        }),
      });

      const data = await resp.json();

      if (!resp.ok) {
        setError(data.detail || "Generation failed");
        setMessages(newMessages.slice(0, -1)); // Remove the user message on error
        return;
      }

      const result = data as GenerationResult;
      // Extract just the generated tokens as text, stripping the prompt
      const assistantResponse = result.tokens
        .map((t) => t.token_str)
        .join("")
        .trim();

      setMessages([...newMessages, { role: "assistant", content: assistantResponse }]);
    } catch (err) {
      setError(`Error: ${err}`);
      setMessages(newMessages.slice(0, -1)); // Remove the user message on error
    } finally {
      setGenerating(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setUserInput("");
    setError(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="page">
      <h1>Chat</h1>
      <p className="subtitle">
        Multi-turn conversation with equilibrium language models. Tune solver budget,
        temperature, and top-k for flexible generation.
      </p>

      <div className="chat-container">
        <div className="chat-sidebar">
          <div className="control-panel">
            <div className="form-group">
              <label htmlFor="model-select">Model:</label>
              {modelsLoading ? (
                <p className="loading">Loading models...</p>
              ) : modelsError ? (
                <p className="error">{modelsError}</p>
              ) : models.length === 0 ? (
                <p className="error">No models available</p>
              ) : (
                <select
                  id="model-select"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={generating}
                >
                  {models.map((model) => (
                    <option key={model.path} value={model.path}>
                      {model.path} ({model.model_class}, {(model.params_estimate / 1e6).toFixed(1)}M)
                    </option>
                  ))}
                </select>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="system-prompt">System Prompt:</label>
              <textarea
                id="system-prompt"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                disabled={generating}
                rows={3}
                placeholder="Enter system instructions..."
              />
            </div>

            <div className="form-group">
              <label htmlFor="max-tokens">Max Tokens (≤64):</label>
              <input
                id="max-tokens"
                type="number"
                min="1"
                max="64"
                value={maxNewTokens}
                onChange={(e) =>
                  setMaxNewTokens(
                    Math.min(Math.max(parseInt(e.target.value) || 1, 1), 64)
                  )
                }
                disabled={generating}
              />
            </div>

            <div className="form-group">
              <label>Solver Budget ({solverBudget}):</label>
              <div className="slider-container">
                <input
                  type="range"
                  min="4"
                  max="64"
                  value={solverBudget}
                  onChange={(e) => setSolverBudget(parseInt(e.target.value))}
                  disabled={generating}
                  className="slider"
                />
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginTop: "0.25rem" }}>
                Higher budget = more equilibrium solver iterations per token
              </div>
            </div>

            <div className="form-group">
              <label>Temperature ({temperature.toFixed(2)}):</label>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                disabled={generating}
                className="slider"
              />
              <div style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginTop: "0.25rem" }}>
                0 = deterministic; 0.8 recommended
              </div>
            </div>

            <div className="form-group">
              <label>Top-k ({topK}):</label>
              <input
                type="range"
                min={0}
                max={200}
                step={10}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={generating}
                className="slider"
              />
              <div style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", marginTop: "0.25rem" }}>
                Restrict sampling to top-k most likely tokens
              </div>
            </div>

            {messages.length > 0 && (
              <button
                className="btn-secondary"
                onClick={handleClearChat}
                disabled={generating}
                style={{ width: "100%" }}
              >
                Clear Chat
              </button>
            )}

            {error && <div className="error-box">{error}</div>}
          </div>
        </div>

        <div className="chat-main">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <p>Start a conversation by typing a message below.</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className="message" data-role={msg.role}>
                  <div className="message-role">{msg.role === "user" ? "You" : "Assistant"}</div>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-area">
            <textarea
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={generating}
              placeholder="Type your message (Shift+Enter for new line, Enter to send)..."
              rows={3}
            />
            <button
              className="btn-send"
              onClick={handleSend}
              disabled={generating || !userInput.trim() || !selectedModel}
            >
              {generating ? "Generating..." : "Send"}
            </button>
          </div>
        </div>
      </div>

      <style jsx>{`
        .subtitle {
          color: var(--text-secondary);
          margin-bottom: 1.5rem;
        }

        .chat-container {
          display: grid;
          grid-template-columns: 280px 1fr;
          gap: 1.5rem;
          margin-top: 1.5rem;
          height: calc(100vh - 200px);
          min-height: 600px;
        }

        @media (max-width: 1024px) {
          .chat-container {
            grid-template-columns: 1fr;
            height: auto;
          }
        }

        .chat-sidebar {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          overflow-y: auto;
          max-height: 100%;
        }

        .control-panel {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          padding: 1rem;
          background: var(--bg-secondary);
          border-radius: 4px;
          border: 1px solid var(--border-color);
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .form-group label {
          font-weight: 500;
          font-size: 0.9rem;
        }

        .form-group select,
        .form-group textarea,
        .form-group input[type="number"] {
          padding: 0.5rem;
          border: 1px solid var(--border-color);
          border-radius: 3px;
          background: var(--bg-input);
          color: var(--text-primary);
          font-family: inherit;
          font-size: 0.9rem;
        }

        .form-group textarea {
          resize: vertical;
          min-height: 80px;
        }

        .form-group select:disabled,
        .form-group textarea:disabled,
        .form-group input:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .slider-container {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .slider {
          flex: 1;
          height: 6px;
          border-radius: 3px;
          background: var(--border-color);
          outline: none;
          -webkit-appearance: none;
          appearance: none;
        }

        .slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--accent);
          cursor: pointer;
        }

        .slider::-moz-range-thumb {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: var(--accent);
          cursor: pointer;
          border: none;
        }

        .error-box {
          background: var(--error-bg);
          color: var(--error);
          padding: 0.75rem;
          border-radius: 3px;
          font-size: 0.85rem;
        }

        .loading,
        .error {
          font-size: 0.85rem;
          color: var(--text-secondary);
        }

        .error {
          color: var(--error);
        }

        .chat-main {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          background: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: 4px;
          overflow: hidden;
        }

        .messages {
          flex: 1;
          overflow-y: auto;
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .empty-state {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: var(--text-secondary);
          font-size: 0.95rem;
        }

        .message {
          display: flex;
          flex-direction: column;
          gap: 0.35rem;
          margin-bottom: 0.5rem;
        }

        .message[data-role="user"] {
          align-items: flex-end;
        }

        .message[data-role="assistant"] {
          align-items: flex-start;
        }

        .message-role {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .message-content {
          max-width: 80%;
          padding: 0.75rem 1rem;
          border-radius: 4px;
          line-height: 1.5;
          word-wrap: break-word;
        }

        .message[data-role="user"] .message-content {
          background: var(--accent);
          color: white;
        }

        .message[data-role="assistant"] .message-content {
          background: var(--bg-input);
          color: var(--text-primary);
          border: 1px solid var(--border-color);
        }

        .chat-input-area {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          padding: 1rem;
          border-top: 1px solid var(--border-color);
          background: var(--bg-secondary);
        }

        .chat-input-area textarea {
          padding: 0.75rem;
          border: 1px solid var(--border-color);
          border-radius: 3px;
          background: var(--bg-input);
          color: var(--text-primary);
          font-family: inherit;
          font-size: 0.9rem;
          resize: vertical;
          max-height: 120px;
        }

        .chat-input-area textarea:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .btn-send {
          padding: 0.65rem 1.25rem;
          background: var(--accent);
          color: white;
          border: none;
          border-radius: 3px;
          font-size: 0.9rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-send:hover:not(:disabled) {
          background: var(--accent-hover);
        }

        .btn-send:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-secondary {
          padding: 0.65rem 1rem;
          background: var(--bg-input);
          color: var(--text-primary);
          border: 1px solid var(--border-color);
          border-radius: 3px;
          font-size: 0.9rem;
          font-weight: 500;
          cursor: pointer;
          transition: background 0.2s;
        }

        .btn-secondary:hover:not(:disabled) {
          background: var(--border-color);
        }

        .btn-secondary:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
