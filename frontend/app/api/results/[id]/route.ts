/**
 * Results API - Single Result Operations
 *
 * GET /api/results/[id] - Get a result by ID
 * PATCH /api/results/[id] - Update a result
 * DELETE /api/results/[id] - Delete a result
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getResultById,
  updateResult,
  deleteResult,
  incrementResultViews,
  UpdateResultInput,
} from "@/lib/social/results";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/results/[id]
 * Get a single result by ID
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    const result = await getResultById(id);

    if (!result) {
      return NextResponse.json(
        { error: "Result not found" },
        { status: 404 }
      );
    }

    // Increment view count (don't wait)
    incrementResultViews(id).catch(console.error);

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error getting result:", error);
    return NextResponse.json(
      { error: "Failed to get result" },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/results/[id]
 * Update a result
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();

    // Get existing result
    const existing = await getResultById(id);
    if (!existing) {
      return NextResponse.json(
        { error: "Result not found" },
        { status: 404 }
      );
    }

    // TODO: Verify user has permission to edit

    const updates: UpdateResultInput = {};

    if (body.title !== undefined) updates.title = body.title;
    if (body.description !== undefined) updates.description = body.description;
    if (body.content !== undefined) updates.content = body.content;
    if (body.visibility !== undefined) updates.visibility = body.visibility;
    if (body.status !== undefined) updates.status = body.status;
    if (body.tags !== undefined) updates.tags = body.tags;
    if (body.isPinned !== undefined) updates.isPinned = body.isPinned;
    if (body.metrics !== undefined) updates.metrics = body.metrics;
    if (body.metadata !== undefined) updates.metadata = body.metadata;

    const result = await updateResult(id, updates);

    if (!result) {
      return NextResponse.json(
        { error: "Failed to update result" },
        { status: 500 }
      );
    }

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error updating result:", error);
    return NextResponse.json(
      { error: "Failed to update result" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/results/[id]
 * Delete a result
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    // Get existing result
    const existing = await getResultById(id);
    if (!existing) {
      return NextResponse.json(
        { error: "Result not found" },
        { status: 404 }
      );
    }

    // TODO: Verify user has permission to delete

    const deleted = await deleteResult(id);

    if (!deleted) {
      return NextResponse.json(
        { error: "Failed to delete result" },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting result:", error);
    return NextResponse.json(
      { error: "Failed to delete result" },
      { status: 500 }
    );
  }
}
