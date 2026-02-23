/**
 * Results API - Like Operations
 *
 * POST /api/results/[id]/like - Like a result
 * DELETE /api/results/[id]/like - Unlike a result
 * GET /api/results/[id]/like - Check if user has liked
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getResultById,
  likeResult,
  unlikeResult,
  hasLikedResult,
} from "@/lib/social/results";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/results/[id]/like
 * Check if current user has liked the result
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");

    if (!userId) {
      return NextResponse.json(
        { error: "userId is required" },
        { status: 400 }
      );
    }

    const result = await getResultById(id);
    if (!result) {
      return NextResponse.json(
        { error: "Result not found" },
        { status: 404 }
      );
    }

    const liked = await hasLikedResult(id, userId);

    return NextResponse.json({ liked });
  } catch (error) {
    console.error("Error checking like status:", error);
    return NextResponse.json(
      { error: "Failed to check like status" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/results/[id]/like
 * Like a result
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { userId } = body;

    if (!userId) {
      return NextResponse.json(
        { error: "userId is required" },
        { status: 400 }
      );
    }

    const result = await getResultById(id);
    if (!result) {
      return NextResponse.json(
        { error: "Result not found" },
        { status: 404 }
      );
    }

    const success = await likeResult(id, userId);

    if (!success) {
      return NextResponse.json(
        { error: "Already liked" },
        { status: 409 }
      );
    }

    // Get updated result
    const updated = await getResultById(id);

    return NextResponse.json({
      success: true,
      likes: updated?.stats.likes || result.stats.likes + 1,
    });
  } catch (error) {
    console.error("Error liking result:", error);
    return NextResponse.json(
      { error: "Failed to like result" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/results/[id]/like
 * Unlike a result
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");

    if (!userId) {
      return NextResponse.json(
        { error: "userId is required" },
        { status: 400 }
      );
    }

    const result = await getResultById(id);
    if (!result) {
      return NextResponse.json(
        { error: "Result not found" },
        { status: 404 }
      );
    }

    const success = await unlikeResult(id, userId);

    if (!success) {
      return NextResponse.json(
        { error: "Not liked" },
        { status: 409 }
      );
    }

    // Get updated result
    const updated = await getResultById(id);

    return NextResponse.json({
      success: true,
      likes: updated?.stats.likes || Math.max(0, result.stats.likes - 1),
    });
  } catch (error) {
    console.error("Error unliking result:", error);
    return NextResponse.json(
      { error: "Failed to unlike result" },
      { status: 500 }
    );
  }
}
