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
      return NextResponse.json(
        { status: "error", message: "Backend health check failed" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { status: "error", message: "Gateway unreachable", error: String(error) },
      { status: 503 }
    );
  }
}
