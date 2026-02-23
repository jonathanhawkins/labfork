/**
 * Activity Seed API
 *
 * POST /api/activity/seed - Disabled (no fake seeding)
 * GET /api/activity/seed - Check if seed data exists
 */

import { NextResponse } from "next/server";
import { getActivityFeed } from "@/lib/social/activity";

/**
 * POST /api/activity/seed
 * Seeding disabled — real activities only
 */
export async function POST() {
  return NextResponse.json({
    success: true,
    seeded: false,
    count: 0,
    message: "Activity seeding disabled — real activities only",
  });
}

/**
 * GET /api/activity/seed
 * Check seed status
 */
export async function GET() {
  try {
    const existing = await getActivityFeed({ limit: 1 });
    return NextResponse.json({
      hasData: existing.activities.length > 0,
      totalActivities: existing.total,
    });
  } catch {
    return NextResponse.json({
      hasData: false,
      totalActivities: 0,
    });
  }
}
