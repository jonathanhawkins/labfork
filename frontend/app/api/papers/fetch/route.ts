/**
 * Paper Fetch API
 *
 * POST - Fetch paper metadata from input (arXiv ID, URL, DOI, etc.)
 */

import { NextRequest, NextResponse } from "next/server";
import { fetchPaper, validateInput, detectInputSource } from "@/lib/papers/parser";

export const dynamic = "force-dynamic";

/**
 * POST /api/papers/fetch - Fetch paper metadata
 *
 * Body: { input: string }
 * Returns: { success: boolean, paper?: Paper, detection?: PaperInputDetection, error?: string }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { input } = body;

    if (!input) {
      return NextResponse.json(
        { success: false, error: "Input is required" },
        { status: 400 }
      );
    }

    // Validate and detect input type
    const validation = validateInput(input);

    if (!validation.valid) {
      return NextResponse.json(
        {
          success: false,
          error: validation.error,
          detection: null,
        },
        { status: 400 }
      );
    }

    // Detect source for early feedback
    const detection = detectInputSource(input);

    // Fetch paper metadata
    const result = await fetchPaper(input);

    if (!result.success) {
      return NextResponse.json(
        {
          success: false,
          error: result.error,
          detection,
        },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      paper: result.paper,
      detection,
      fromCache: result.fromCache || false,
    });
  } catch (error) {
    console.error("Error fetching paper:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to fetch paper",
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/papers/fetch - Detect input type without fetching
 *
 * Query: ?input=...
 * Returns: { detection?: PaperInputDetection, valid: boolean, error?: string }
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const input = searchParams.get("input");

    if (!input) {
      return NextResponse.json(
        { valid: false, error: "Input is required" },
        { status: 400 }
      );
    }

    const validation = validateInput(input);

    return NextResponse.json({
      valid: validation.valid,
      detection: validation.detection,
      error: validation.error,
    });
  } catch (error) {
    console.error("Error detecting input:", error);
    return NextResponse.json(
      { valid: false, error: "Failed to detect input type" },
      { status: 500 }
    );
  }
}
