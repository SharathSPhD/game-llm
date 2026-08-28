import { NextResponse } from "next/server";
import { getLeaderboardData } from "@/lib/leaderboard-data";

/**
 * GET /api/leaderboard
 * Returns processed baseline ladder data with all benchmark metrics.
 */
export async function GET() {
  try {
    const data = await getLeaderboardData();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to load leaderboard data:", error);
    // Return empty array rather than error, so UI can handle gracefully
    return NextResponse.json([]);
  }
}
