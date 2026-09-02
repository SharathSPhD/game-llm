import { NextResponse } from "next/server";
import { GATEWAY_URL, REPLAY_MODE } from "@/lib/config";

export async function GET() {
  if (REPLAY_MODE) {
    // Demo mode: return canned health response
    return NextResponse.json({
      status: "ok",
      version: "0.1.0-demo",
      gpu_available: false,
      replay: true,
    });
  }

  try {
    const response = await fetch(`${GATEWAY_URL}/health`, {
      method: "GET",
      headers: { "Accept": "application/json" },
    });

    if (!response.ok) {
      // Configured but unhealthy: the app replays (ADR 0011); say so.
      return NextResponse.json({
        status: "ok", version: "0.1.0-replay", gpu_available: false, replay: true,
        gateway: `unhealthy (${response.status})`,
      });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    // Configured but unreachable — the serving host is away: replay mode.
    return NextResponse.json({
      status: "ok", version: "0.1.0-replay", gpu_available: false, replay: true,
      gateway: "unreachable",
    });
  }
}
