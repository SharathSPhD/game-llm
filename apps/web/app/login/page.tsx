"use client";

import { useEffect, useState } from "react";
import { createBrowserClient } from "@supabase/ssr";

function getClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return null;
  return createBrowserClient(url, key);
}

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sessionEmail, setSessionEmail] = useState<string | null>(null);
  const supabase = getClient();

  useEffect(() => {
    if (!supabase) return;
    supabase.auth.getUser().then(({ data }) => {
      setSessionEmail(data.user?.email ?? null);
    });
  }, []);

  if (!supabase) {
    return (
      <div className="wrap-narrow" style={{ paddingTop: "var(--space-7)" }}>
        <h1>Sign in</h1>
        <p className="card" style={{ padding: "var(--space-4)" }}>
          Authentication is not configured in this deployment
          (NEXT_PUBLIC_SUPABASE_URL is unset). Pages remain readable; live
          solver and training actions are disabled.
        </p>
      </div>
    );
  }

  const signInPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setStatus(null);
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    setBusy(false);
    if (error) {
      setStatus(`Sign-in failed: ${error.message}`);
    } else {
      setStatus("Signed in.");
      window.location.href = "/lab";
    }
  };

  const sendMagicLink = async () => {
    if (!email) {
      setStatus("Enter your email first, then request a magic link.");
      return;
    }
    setBusy(true);
    setStatus(null);
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: `${window.location.origin}/lab` },
    });
    setBusy(false);
    setStatus(
      error ? `Magic link failed: ${error.message}` : "Magic link sent — check your email."
    );
  };

  const signOut = async () => {
    await supabase.auth.signOut();
    setSessionEmail(null);
    setStatus("Signed out.");
  };

  return (
    <div className="wrap-narrow" style={{ paddingTop: "var(--space-7)", maxWidth: "26rem" }}>
      <h1>Sign in</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "var(--space-5)" }}>
        Live solver runs, training jobs, and the playground execute on the GB10
        research backend and require an authorized account (admin, or a guest
        the admin has enabled). Everything else on this site is readable
        without signing in.
      </p>

      {sessionEmail ? (
        <div className="card" style={{ padding: "var(--space-4)" }}>
          <p>
            Signed in as <strong>{sessionEmail}</strong>
          </p>
          <button className="btn" onClick={signOut} style={{ marginTop: "var(--space-3)" }}>
            Sign out
          </button>
        </div>
      ) : (
        <form onSubmit={signInPassword} className="card" style={{ padding: "var(--space-4)" }}>
          <label style={{ display: "block", marginBottom: "var(--space-3)" }}>
            Email
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
              style={{ width: "100%", marginTop: "0.3rem" }}
            />
          </label>
          <label style={{ display: "block", marginBottom: "var(--space-4)" }}>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              style={{ width: "100%", marginTop: "0.3rem" }}
            />
          </label>
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
          <button
            className="btn"
            type="button"
            onClick={sendMagicLink}
            disabled={busy}
            style={{ marginLeft: "var(--space-3)" }}
          >
            Email me a magic link
          </button>
        </form>
      )}

      {status && (
        <p style={{ marginTop: "var(--space-4)", color: "var(--text-secondary)" }}>{status}</p>
      )}
    </div>
  );
}
