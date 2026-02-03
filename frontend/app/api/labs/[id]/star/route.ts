/**
 * Star API
 *
 * POST /api/labs/[id]/star - Star a lab
 * DELETE /api/labs/[id]/star - Unstar a lab
 * GET /api/labs/[id]/star - Check if starred
 */

import { NextRequest, NextResponse } from "next/server";
import { getLabById } from "@/lib/labs/repository";
import {
  starLab,
  unstarLab,
  isLabStarred,
  getStarCount,
  toggleStar,
} from "@/lib/labs/social";
import { canViewLab } from "@/lib/labs/types";
import { getServerUser } from "@/lib/auth/server";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/labs/[id]/star
 * Check if current user has starred the lab
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    // Get lab
    const lab = await getLabById(id);
    if (!lab) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Check visibility
    const user = await getServerUser();
    if (!canViewLab(lab, user?.id)) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Check star status
    const isStarred = user ? await isLabStarred(user.id, id) : false;
    const count = await getStarCount(id);

    return NextResponse.json({
      success: true,
      starred: isStarred,
      count,
    });
  } catch (error) {
    console.error("Failed to check star status:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to check star status",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/labs/[id]/star
 * Star a lab (or toggle)
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    // Get user
    const user = await getServerUser();
    if (!user) {
      return NextResponse.json(
        { success: false, error: "Authentication required" },
        { status: 401 }
      );
    }

    // Get lab
    const lab = await getLabById(id);
    if (!lab) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Check visibility
    if (!canViewLab(lab, user.id)) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Check for toggle mode
    const url = new URL(request.url);
    const toggle = url.searchParams.get("toggle") === "true";

    if (toggle) {
      // Toggle star
      const result = await toggleStar(user.id, id);
      return NextResponse.json({
        success: true,
        starred: result.starred,
        count: result.count,
      });
    }

    // Star the lab
    const starred = await starLab(user.id, id);
    const count = await getStarCount(id);

    return NextResponse.json({
      success: true,
      starred: true,
      alreadyStarred: !starred,
      count,
    });
  } catch (error) {
    console.error("Failed to star lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to star lab",
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/labs/[id]/star
 * Unstar a lab
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    // Get user
    const user = await getServerUser();
    if (!user) {
      return NextResponse.json(
        { success: false, error: "Authentication required" },
        { status: 401 }
      );
    }

    // Unstar the lab
    const unstarred = await unstarLab(user.id, id);
    const count = await getStarCount(id);

    return NextResponse.json({
      success: true,
      starred: false,
      wasStarred: unstarred,
      count,
    });
  } catch (error) {
    console.error("Failed to unstar lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to unstar lab",
      },
      { status: 500 }
    );
  }
}
