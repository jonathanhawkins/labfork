/**
 * User Labs API
 *
 * GET /api/labs/user/[username] - Get labs by username
 */

import { NextRequest, NextResponse } from "next/server";
import { getLabsByUser } from "@/lib/labs/repository";
import { getServerUser } from "@/lib/auth/server";
import { getUserByUsername } from "@/lib/auth/mock-user";

interface RouteParams {
  params: Promise<{ username: string }>;
}

/**
 * GET /api/labs/user/[username]
 * Get all labs by a user
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { username } = await params;

    // Check if user exists
    const targetUser = getUserByUsername(username);
    if (!targetUser) {
      return NextResponse.json(
        { success: false, error: "User not found" },
        { status: 404 }
      );
    }

    // Get current user to check if viewing own labs
    const currentUser = await getServerUser();
    const isOwnProfile = currentUser?.username === username;

    // Get labs (include private if viewing own profile)
    const labs = await getLabsByUser(username, isOwnProfile);

    return NextResponse.json({
      success: true,
      user: {
        id: targetUser.id,
        username: targetUser.username,
        displayName: targetUser.displayName,
        avatar: targetUser.avatar,
        bio: targetUser.bio,
      },
      labs,
      total: labs.length,
    });
  } catch (error) {
    console.error("Failed to get user labs:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get user labs",
      },
      { status: 500 }
    );
  }
}
