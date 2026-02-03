/**
 * Social Suggestions API - Single Suggestion Operations
 *
 * GET /api/social/suggestions/[id] - Get a suggestion
 * PATCH /api/social/suggestions/[id] - Update a suggestion
 * DELETE /api/social/suggestions/[id] - Delete a suggestion
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getSuggestionById,
  updateSuggestion,
  deleteSuggestion,
  canEditSuggestion,
} from "@/lib/social/suggestions";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/social/suggestions/[id]
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    const suggestion = await getSuggestionById(id);

    if (!suggestion) {
      return NextResponse.json(
        { error: "Suggestion not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(suggestion);
  } catch (error) {
    console.error("Error getting suggestion:", error);
    return NextResponse.json(
      { error: "Failed to get suggestion" },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/social/suggestions/[id]
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();

    const suggestion = await getSuggestionById(id);
    if (!suggestion) {
      return NextResponse.json(
        { error: "Suggestion not found" },
        { status: 404 }
      );
    }

    // Verify user has permission to update
    const userId = request.headers.get("x-user-id");
    if (!canEditSuggestion(suggestion, userId || undefined)) {
      return NextResponse.json(
        { error: "You do not have permission to update this suggestion" },
        { status: 403 }
      );
    }

    const updated = await updateSuggestion(id, body, body.changedBy);

    if (!updated) {
      return NextResponse.json(
        { error: "Failed to update suggestion" },
        { status: 500 }
      );
    }

    return NextResponse.json(updated);
  } catch (error) {
    console.error("Error updating suggestion:", error);
    return NextResponse.json(
      { error: "Failed to update suggestion" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/social/suggestions/[id]
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    const suggestion = await getSuggestionById(id);
    if (!suggestion) {
      return NextResponse.json(
        { error: "Suggestion not found" },
        { status: 404 }
      );
    }

    // Verify user has permission to delete
    const userId = request.headers.get("x-user-id");
    if (!canEditSuggestion(suggestion, userId || undefined)) {
      return NextResponse.json(
        { error: "You do not have permission to delete this suggestion" },
        { status: 403 }
      );
    }

    const deleted = await deleteSuggestion(id);

    if (!deleted) {
      return NextResponse.json(
        { error: "Failed to delete suggestion" },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error deleting suggestion:", error);
    return NextResponse.json(
      { error: "Failed to delete suggestion" },
      { status: 500 }
    );
  }
}
