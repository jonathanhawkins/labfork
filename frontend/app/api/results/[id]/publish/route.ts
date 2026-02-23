/**
 * Results API - Publish Operations
 *
 * POST /api/results/[id]/publish - Publish a result
 */

import { NextRequest, NextResponse } from "next/server";
import { getResultById, publishResult } from "@/lib/social/results";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * POST /api/results/[id]/publish
 * Publish a draft result
 */
export async function POST(
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

    if (result.status === "published") {
      return NextResponse.json(
        { error: "Result is already published" },
        { status: 409 }
      );
    }

    if (result.status === "archived") {
      return NextResponse.json(
        { error: "Cannot publish an archived result" },
        { status: 400 }
      );
    }

    // TODO: Verify user has permission to publish

    const published = await publishResult(id);

    if (!published) {
      return NextResponse.json(
        { error: "Failed to publish result" },
        { status: 500 }
      );
    }

    return NextResponse.json(published);
  } catch (error) {
    console.error("Error publishing result:", error);
    return NextResponse.json(
      { error: "Failed to publish result" },
      { status: 500 }
    );
  }
}
