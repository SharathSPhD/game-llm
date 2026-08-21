import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import {
  AUTH_ENABLED,
  GATEWAY_SECRET,
  GATEWAY_URL,
  REPLAY_MODE,
  SUPABASE_ANON_KEY,
  SUPABASE_URL,
} from "@/lib/config";
import {
  getReplayAuctionResponse,
  getReplayQREPathResponse,
  getReplaySolveResponse,
} from "@/lib/replay-data";

/**
 * Proxy for the GB10 gateway. Security model:
 *  - Endpoint allowlist: only known gateway routes are forwarded.
 *  - When Supabase is configured (production), a signed-in user with a
 *    user_tiers row (admin or admin-invited guest) is required before any
 *    request is forwarded with the server-side GATEWAY_SECRET.
 *  - Upstream error bodies are logged server-side, never echoed to callers.
 *  - Replay mode (no GATEWAY_URL) serves canned demo data, no auth needed.
 */

const POST_ALLOWLIST = new Set(["/api/solve", "/api/qre_path", "/api/auction", "/api/jobs"]);
const GET_ALLOWLIST_EXACT = new Set(["/api/results", "/api/jobs"]);
const GET_ALLOWLIST_PREFIX = ["/api/jobs/"];

function isAllowed(method: "GET" | "POST", endpoint: string): boolean {
  if (method === "POST") return POST_ALLOWLIST.has(endpoint);
  return (
    GET_ALLOWLIST_EXACT.has(endpoint) ||
    GET_ALLOWLIST_PREFIX.some(
      (p) => endpoint.startsWith(p) && !endpoint.slice(p.length).includes("/")
    )
  );
}

/** Returns null when authorized, or an error response. */
async function requireTieredUser(): Promise<NextResponse | null> {
  if (!AUTH_ENABLED) {
    // Standalone/dev mode: Supabase unset. Forwarding is allowed only for
    // local development; production deployments must configure Supabase.
    return null;
  }
  const cookieStore = cookies();
  const supabase = createServerClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    cookies: {
      getAll() {
        return cookieStore.getAll();
      },
      setAll() {
        // Read-only in route handlers; middleware refreshes sessions.
      },
    },
  });
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: "Sign-in required" }, { status: 401 });
  }
  const { data: tier } = await supabase
    .from("user_tiers")
    .select("tier")
    .eq("user_id", user.id)
    .maybeSingle();
  if (!tier) {
    return NextResponse.json(
      { error: "Access not enabled for this account" },
      { status: 403 }
    );
  }
  return null;
}

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const endpoint = `/${params.path.join("/")}`;
  if (!isAllowed("POST", endpoint)) {
    return NextResponse.json({ error: "Unknown endpoint" }, { status: 404 });
  }

  let body: Record<string, unknown> = {};
  try {
    const text = await request.text();
    if (text) {
      body = JSON.parse(text);
    }
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  // Replay mode: serve canned responses (demo data, no auth needed)
  if (REPLAY_MODE) {
    if (endpoint === "/api/solve") {
      return NextResponse.json(
        getReplaySolveResponse({
          game: (body.game as string) || "rps",
          method: (body.method as string) || "mmd_fixed",
          steps: (body.steps as number) || 100,
        })
      );
    }
    if (endpoint === "/api/qre_path") {
      return NextResponse.json(
        getReplayQREPathResponse({
          game: (body.game as string) || "rps",
          lambda_min: (body.lambda_min as number) || 0.1,
          lambda_max: (body.lambda_max as number) || 10.0,
          n_points: (body.n_points as number) || 20,
        })
      );
    }
    if (endpoint === "/api/auction") {
      return NextResponse.json(
        getReplayAuctionResponse({
          bids: (body.bids as number[]) || [1.0, 1.0],
          agent_distributions: (body.agent_distributions as number[][]) || [[1.0], [0.0]],
          auction_type: (body.auction_type as string) || "second_price",
        })
      );
    }
    return NextResponse.json(
      { error: "Endpoint not available in replay mode", replay: true },
      { status: 503 }
    );
  }

  const denied = await requireTieredUser();
  if (denied) return denied;

  if (!GATEWAY_URL || !GATEWAY_SECRET) {
    return NextResponse.json({ error: "Gateway not configured" }, { status: 503 });
  }

  try {
    const response = await fetch(`${GATEWAY_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${GATEWAY_SECRET}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      console.error(
        `gateway POST ${endpoint} -> ${response.status}: ${(await response.text()).slice(0, 500)}`
      );
      return NextResponse.json({ error: "Upstream error" }, { status: response.status });
    }
    return NextResponse.json(await response.json());
  } catch (error) {
    console.error(`gateway POST ${endpoint} unreachable:`, error);
    return NextResponse.json({ error: "Gateway unreachable" }, { status: 503 });
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const endpoint = `/${params.path.join("/")}`;
  if (!isAllowed("GET", endpoint)) {
    return NextResponse.json({ error: "Unknown endpoint" }, { status: 404 });
  }

  if (REPLAY_MODE) {
    return NextResponse.json(
      { error: "Endpoint not available in replay mode", replay: true },
      { status: 503 }
    );
  }

  const denied = await requireTieredUser();
  if (denied) return denied;

  if (!GATEWAY_URL || !GATEWAY_SECRET) {
    return NextResponse.json({ error: "Gateway not configured" }, { status: 503 });
  }

  try {
    const url = new URL(`${GATEWAY_URL}${endpoint}`);
    request.nextUrl.searchParams.forEach((value, key) => {
      url.searchParams.append(key, value);
    });

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${GATEWAY_SECRET}`,
      },
    });

    if (!response.ok) {
      console.error(
        `gateway GET ${endpoint} -> ${response.status}: ${(await response.text()).slice(0, 500)}`
      );
      return NextResponse.json({ error: "Upstream error" }, { status: response.status });
    }
    return NextResponse.json(await response.json());
  } catch (error) {
    console.error(`gateway GET ${endpoint} unreachable:`, error);
    return NextResponse.json({ error: "Gateway unreachable" }, { status: 503 });
  }
}
