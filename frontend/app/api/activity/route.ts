/**
 * Activity API - Activity Feed
 *
 * GET /api/activity - Get activity feed
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getActivityFeed,
  getLabActivityFeed,
  getUserActivityFeed,
  getNewActivities,
  ActivityType,
} from "@/lib/social/activity";

/**
 * GET /api/activity
 * Get activity feed with optional filters
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Check for "since" parameter (for polling new activities)
    const since = searchParams.get("since");
    if (since) {
      const labId = searchParams.get("labId") || undefined;
      const userId = searchParams.get("userId") || undefined;
      const publicOnly = searchParams.get("publicOnly") !== "false";

      const activities = await getNewActivities(since, {
        labId,
        userId,
        publicOnly,
      });

      return NextResponse.json({
        activities,
        count: activities.length,
      });
    }

    // Regular feed request
    const labId = searchParams.get("labId") || undefined;
    const userId = searchParams.get("userId") || undefined;
    const targetId = searchParams.get("targetId") || undefined;
    const typesParam = searchParams.get("types");
    const types = typesParam
      ? (typesParam.split(",") as ActivityType[])
      : undefined;
    const publicOnly = searchParams.get("publicOnly") !== "false";
    const aggregate = searchParams.get("aggregate") !== "false";
    const aggregateWindow = parseInt(
      searchParams.get("aggregateWindow") || "60",
      10
    );
    const page = parseInt(searchParams.get("page") || "1", 10);
    const limit = parseInt(searchParams.get("limit") || "20", 10);

    // Use specialized feeds if filtering by lab or user
    let result;
    if (labId) {
      result = await getLabActivityFeed(labId, {
        types,
        targetId,
        publicOnly,
        aggregate,
        aggregateWindow,
        page,
        limit,
      });
    } else if (userId) {
      result = await getUserActivityFeed(userId, {
        types,
        targetId,
        publicOnly,
        aggregate,
        aggregateWindow,
        page,
        limit,
      });
    } else {
      result = await getActivityFeed({
        types,
        targetId,
        publicOnly,
        aggregate,
        aggregateWindow,
        page,
        limit,
      });
    }

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error getting activity feed:", error);
    return NextResponse.json(
      { error: "Failed to get activity feed" },
      { status: 500 }
    );
  }
}
