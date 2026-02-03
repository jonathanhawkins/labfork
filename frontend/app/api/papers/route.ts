/**
 * Papers List API
 *
 * GET - List all papers with optional filters
 * POST - Add a new paper (fetch and store)
 * PATCH - Update paper status
 * DELETE - Remove paper from queue
 */

import { NextRequest, NextResponse } from "next/server";
import type {
  Paper,
  PaperListFilters,
  PaperListResponse,
  PaperStatus,
} from "@/lib/papers/types";
import { fetchPaper, validateInput } from "@/lib/papers/parser";
import {
  listPapers,
  getPaperByMetadataId,
  getPaperByUrl,
  createPaper,
  updatePaper,
  deletePaper,
} from "@/lib/papers/repository";

export const dynamic = "force-dynamic";

/**
 * GET /api/papers - List papers
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Parse filters from query params
    const filters: PaperListFilters = {};

    const status = searchParams.get("status");
    if (status) {
      filters.status = status.includes(",")
        ? (status.split(",") as PaperStatus[])
        : (status as PaperStatus);
    }

    const domainSlug = searchParams.get("domain");
    if (domainSlug) filters.domainSlug = domainSlug;

    const source = searchParams.get("source");
    if (source) filters.source = source as Paper["metadata"]["source"];

    const search = searchParams.get("search");
    if (search) filters.search = search;

    const minRelevance = searchParams.get("minRelevance");
    if (minRelevance) filters.minRelevance = parseInt(minRelevance, 10);

    const sortBy = searchParams.get("sortBy");
    if (sortBy)
      filters.sortBy = sortBy as PaperListFilters["sortBy"];

    const sortOrder = searchParams.get("sortOrder");
    if (sortOrder)
      filters.sortOrder = sortOrder as PaperListFilters["sortOrder"];

    const limit = searchParams.get("limit");
    if (limit) filters.limit = parseInt(limit, 10);

    const offset = searchParams.get("offset");
    if (offset) filters.offset = parseInt(offset, 10);

    // List papers using repository
    const response: PaperListResponse = await listPapers(filters);

    return NextResponse.json(response);
  } catch (error) {
    console.error("Error listing papers:", error);
    return NextResponse.json(
      { error: "Failed to list papers" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/papers - Add a new paper
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { input, domainSlug } = body;

    if (!input) {
      return NextResponse.json(
        { error: "Input is required" },
        { status: 400 }
      );
    }

    // Validate input
    const validation = validateInput(input);
    if (!validation.valid) {
      return NextResponse.json(
        { error: validation.error },
        { status: 400 }
      );
    }

    // Check if paper already exists by metadata ID or URL
    let existing = null;
    if (validation.detection?.identifier) {
      existing = await getPaperByMetadataId(validation.detection.identifier);
    }
    if (!existing) {
      existing = await getPaperByUrl(input);
    }

    if (existing) {
      return NextResponse.json({
        success: true,
        paper: existing,
        fromCache: true,
        message: "Paper already exists in queue",
      });
    }

    // Fetch paper metadata
    const result = await fetchPaper(input);

    if (!result.success || !result.paper) {
      return NextResponse.json(
        { error: result.error || "Failed to fetch paper" },
        { status: 400 }
      );
    }

    // Add domain context
    if (domainSlug) {
      result.paper.domainSlug = domainSlug;
    }

    // Save to database
    const savedPaper = await createPaper(result.paper);

    return NextResponse.json({
      success: true,
      paper: savedPaper,
      fromCache: false,
    });
  } catch (error) {
    console.error("Error adding paper:", error);
    return NextResponse.json(
      { error: "Failed to add paper" },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/papers - Update paper
 */
export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json();
    const { id, status, notes } = body;

    if (!id) {
      return NextResponse.json(
        { error: "Paper ID is required" },
        { status: 400 }
      );
    }

    // Build updates object
    const updates: Partial<Paper> = {};
    if (status) {
      updates.status = status;
    }
    if (notes !== undefined) {
      updates.notes = notes;
    }

    // Update paper using repository
    const updatedPaper = await updatePaper(id, updates);

    if (!updatedPaper) {
      return NextResponse.json(
        { error: "Paper not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      paper: updatedPaper,
    });
  } catch (error) {
    console.error("Error updating paper:", error);
    return NextResponse.json(
      { error: "Failed to update paper" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/papers - Remove paper
 */
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get("id");

    if (!id) {
      return NextResponse.json(
        { error: "Paper ID is required" },
        { status: 400 }
      );
    }

    // Delete paper using repository
    const deleted = await deletePaper(id);

    if (!deleted) {
      return NextResponse.json(
        { error: "Paper not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: "Paper removed",
    });
  } catch (error) {
    console.error("Error deleting paper:", error);
    return NextResponse.json(
      { error: "Failed to delete paper" },
      { status: 500 }
    );
  }
}
