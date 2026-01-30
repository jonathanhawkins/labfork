/**
 * Credits API Endpoint
 *
 * GET /api/credits - Get user credit balance
 */

import { NextRequest, NextResponse } from "next/server";
import { getUserCredits } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    // Get user ID from query params (in production, get from auth session)
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");

    if (!userId) {
      return NextResponse.json(
        { error: "Missing userId parameter" },
        { status: 400 }
      );
    }

    const credits = await getUserCredits(userId);

    if (!credits) {
      return NextResponse.json(
        { error: "Failed to fetch user credits" },
        { status: 500 }
      );
    }

    return NextResponse.json(credits, { status: 200 });
  } catch (error) {
    console.error("Error in GET /api/credits:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
