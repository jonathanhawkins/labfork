/**
 * Results API - Save/Bookmark Operations
 *
 * POST /api/results/[id]/save - Save a result
 * DELETE /api/results/[id]/save - Unsave a result
 * GET /api/results/[id]/save - Check if user has saved
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getResultById,
  saveResult,
  unsaveResult,
  hasSavedResult,
} from "@/lib/social/results";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/results/[id]/save
 * Check if current user has saved the result
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

    const saved = await hasSavedResult(id, userId);

    return NextResponse.json({ saved });
  } catch (error) {
    console.error("Error checking save status:", error);
    return NextResponse.json(
      { error: "Failed to check save status" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/results/[id]/save
 * Save a result to user's collection
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { userId, collection } = body;

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

    const success = await saveResult(id, userId, collection);

    if (!success) {
      return NextResponse.json(
        { error: "Already saved" },
        { status: 409 }
      );
    }

    // Get updated result
    const updated = await getResultById(id);

    return NextResponse.json({
      success: true,
      saves: updated?.stats.saves || result.stats.saves + 1,
    });
  } catch (error) {
    console.error("Error saving result:", error);
    return NextResponse.json(
      { error: "Failed to save result" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/results/[id]/save
 * Unsave a result
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

    const success = await unsaveResult(id, userId);

    if (!success) {
      return NextResponse.json(
        { error: "Not saved" },
        { status: 409 }
      );
    }

    // Get updated result
    const updated = await getResultById(id);

    return NextResponse.json({
      success: true,
      saves: updated?.stats.saves || Math.max(0, result.stats.saves - 1),
    });
  } catch (error) {
    console.error("Error unsaving result:", error);
    return NextResponse.json(
      { error: "Failed to unsave result" },
      { status: 500 }
    );
  }
}
