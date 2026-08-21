import { NextRequest, NextResponse } from "next/server";
import { GATEWAY_URL, GATEWAY_SECRET, REPLAY_MODE } from "@/lib/config";
import {
  getReplaySolveResponse,
  getReplayQREPathResponse,
  getReplayAuctionResponse,
} from "@/lib/replay-data";

/**
 * Generic proxy handler for all /api/proxy/* routes.
 * Forwards requests to GATEWAY_URL with Authorization header.
 * Falls back to replay mode when GATEWAY_URL is unset.
 */
export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const pathSegments = params.path;
  const endpoint = `/${pathSegments.join("/")}`;

  // Parse request body
  let body: Record<string, unknown> = {};
  try {
    const text = await request.text();
    if (text) {
      body = JSON.parse(text);
    }
  } catch {
    return NextResponse.json(
      { error: "Invalid JSON body" },
      { status: 400 }
    );
  }

  // Replay mode: serve canned responses
  if (REPLAY_MODE) {
    if (endpoint === "/api/solve") {
      return NextResponse.json(
        getReplaySolveResponse({
          game: body.game as string || "rps",
          method: body.method as string || "mmd_fixed",
          steps: body.steps as number || 100,
        })
      );
    }

    if (endpoint === "/api/qre_path") {
      return NextResponse.json(
        getReplayQREPathResponse({
          game: body.game as string || "rps",
          lambda_min: body.lambda_min as number || 0.1,
          lambda_max: body.lambda_max as number || 10.0,
          n_points: body.n_points as number || 20,
        })
      );
    }

    if (endpoint === "/api/auction") {
      return NextResponse.json(
        getReplayAuctionResponse({
          bids: body.bids as number[] || [1.0, 1.0],
          agent_distributions: body.agent_distributions as number[][] || [[1.0], [0.0]],
          auction_type: body.auction_type as string || "second_price",
        })
      );
    }

    // For other endpoints in replay mode, return error
    return NextResponse.json(
      { error: "Endpoint not available in replay mode", replay: true },
      { status: 503 }
    );
  }

  // Live mode: forward to gateway
  if (!GATEWAY_URL || !GATEWAY_SECRET) {
    return NextResponse.json(
      { error: "Gateway not configured" },
      { status: 503 }
    );
  }

  try {
    const url = `${GATEWAY_URL}${endpoint}`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${GATEWAY_SECRET}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: "Gateway error", details: errorText },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: "Gateway unreachable", message: String(error) },
      { status: 503 }
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const pathSegments = params.path;
  const endpoint = `/${pathSegments.join("/")}`;

  // For GET endpoints like /api/results and /api/jobs/{id}

  if (REPLAY_MODE) {
    // Replay mode doesn't support GET endpoints yet
    return NextResponse.json(
      { error: "Endpoint not available in replay mode", replay: true },
      { status: 503 }
    );
  }

  if (!GATEWAY_URL || !GATEWAY_SECRET) {
    return NextResponse.json(
      { error: "Gateway not configured" },
      { status: 503 }
    );
  }

  try {
    const url = new URL(`${GATEWAY_URL}${endpoint}`);
    // Copy query params
    request.nextUrl.searchParams.forEach((value, key) => {
      url.searchParams.append(key, value);
    });

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: {
        "Accept": "application/json",
        "Authorization": `Bearer ${GATEWAY_SECRET}`,
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Gateway error" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { error: "Gateway unreachable", message: String(error) },
      { status: 503 }
    );
  }
}
