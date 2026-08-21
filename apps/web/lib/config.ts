/**
 * Configuration and environment loading.
 * Public vars are prefixed NEXT_PUBLIC_; server-only vars loaded at runtime.
 */

export const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
export const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";
export const GATEWAY_URL = process.env.GATEWAY_URL || "";
export const GATEWAY_SECRET = process.env.GATEWAY_SECRET || "";

// Replay mode: when gateway is unavailable, serve canned demo responses
export const REPLAY_MODE = !GATEWAY_URL;

// Auth soft-disabled if no Supabase config
export const AUTH_ENABLED = !!(SUPABASE_URL && SUPABASE_ANON_KEY);
