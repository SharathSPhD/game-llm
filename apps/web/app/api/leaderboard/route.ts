import { NextResponse } from "next/server";
import { getLeaderboardData } from "@/lib/leaderboard-data";
import snapshot from "@/data/leaderboard.json";

/**
 * GET /api/leaderboard
 * Returns processed baseline ladder data with all benchmark metrics.
 */
export async function GET() {
  try {
    const data = await getLeaderboardData();
    // On Vercel the results tree is outside the root directory, so the live
    // walk finds nothing; serve the committed snapshot instead (ADR 0011).
    return NextResponse.json(data.length > 0 ? data : snapshot);
  } catch (error) {
    console.error("Failed to load leaderboard data:", error);
    // Return empty array rather than error, so UI can handle gracefully
    return NextResponse.json(snapshot);
  }
}
