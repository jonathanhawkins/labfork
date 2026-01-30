/**
 * Lab API - Single Lab Operations
 *
 * GET /api/labs/[id] - Get lab by ID
 * PATCH /api/labs/[id] - Update lab
 * DELETE /api/labs/[id] - Delete lab
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getLabById,
  updateLab,
  deleteLab,
  incrementLabViews,
} from "@/lib/labs/repository";
import { getLabSocialStats } from "@/lib/labs/social";
import type { UpdateLabInput } from "@/lib/labs/types";
import { canEditLab, canViewLab } from "@/lib/labs/types";
import { getServerUser } from "@/lib/auth/mock-user";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/labs/[id]
 * Get a lab by ID
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
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

    // Get social stats
    const socialStats = await getLabSocialStats(id, user?.id);

    // Increment view count (async, don't wait)
    incrementLabViews(id).catch(console.error);

    return NextResponse.json({
      success: true,
      lab,
      social: socialStats,
    });
  } catch (error) {
    console.error("Failed to get lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get lab",
      },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/labs/[id]
 * Update a lab
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();

    // Get lab
    const lab = await getLabById(id);
    if (!lab) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Check permissions
    const user = await getServerUser();
    if (!canEditLab(lab, user?.id)) {
      return NextResponse.json(
        { success: false, error: "Unauthorized" },
        { status: 403 }
      );
    }

    // Build updates
    const updates: UpdateLabInput = {};

    if (body.name !== undefined) updates.name = body.name;
    if (body.description !== undefined) updates.description = body.description;
    if (body.visibility !== undefined) updates.visibility = body.visibility;
    if (body.status !== undefined) updates.status = body.status;
    if (body.tags !== undefined) updates.tags = body.tags;
    if (body.primaryColor !== undefined) updates.primaryColor = body.primaryColor;
    if (body.readme !== undefined) updates.readme = body.readme;

    // Update lab
    const updated = await updateLab(id, updates);

    if (!updated) {
      return NextResponse.json(
        { success: false, error: "Failed to update lab" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      lab: updated,
    });
  } catch (error) {
    console.error("Failed to update lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to update lab",
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/labs/[id]
 * Delete a lab
 */
export async function DELETE(
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

    // Check permissions
    const user = await getServerUser();
    if (!canEditLab(lab, user?.id)) {
      return NextResponse.json(
        { success: false, error: "Unauthorized" },
        { status: 403 }
      );
    }

    // Delete lab
    const deleted = await deleteLab(id);

    if (!deleted) {
      return NextResponse.json(
        { success: false, error: "Failed to delete lab" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: "Lab deleted",
    });
  } catch (error) {
    console.error("Failed to delete lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to delete lab",
      },
      { status: 500 }
    );
  }
}
