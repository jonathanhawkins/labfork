/**
 * Papers List API
 *
 * GET - List all papers with optional filters
 * POST - Add a new paper (fetch and store)
 * PATCH - Update paper status
 * DELETE - Remove paper from queue
 */

import { NextRequest, NextResponse } from "next/server";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import type {
  Paper,
  PaperListFilters,
  PaperListResponse,
  PaperStatus,
} from "@/lib/papers/types";
import { fetchPaper, validateInput } from "@/lib/papers/parser";

export const dynamic = "force-dynamic";

// Storage path for papers
const getStoragePath = () => {
  const projectRoot = join(process.cwd(), "..");
  const papersDir = join(projectRoot, "data", "papers");

  // Ensure directory exists
  if (!existsSync(papersDir)) {
    mkdirSync(papersDir, { recursive: true });
  }

  return join(papersDir, "papers.json");
};

// Load papers from storage
function loadPapers(): Paper[] {
  const path = getStoragePath();
  if (!existsSync(path)) {
    return [];
  }
  try {
    const content = readFileSync(path, "utf-8");
    return JSON.parse(content);
  } catch {
    return [];
  }
}

// Save papers to storage
function savePapers(papers: Paper[]): void {
  const path = getStoragePath();
  writeFileSync(path, JSON.stringify(papers, null, 2));
}

// Filter papers based on criteria
function filterPapers(papers: Paper[], filters: PaperListFilters): Paper[] {
  let filtered = [...papers];

  // Status filter
  if (filters.status) {
    const statuses = Array.isArray(filters.status)
      ? filters.status
      : [filters.status];
    filtered = filtered.filter((p) => statuses.includes(p.status));
  }

  // Domain filter
  if (filters.domainSlug) {
    filtered = filtered.filter((p) => p.domainSlug === filters.domainSlug);
  }

  // Source filter
  if (filters.source) {
    filtered = filtered.filter((p) => p.metadata.source === filters.source);
  }

  // Search filter
  if (filters.search) {
    const searchLower = filters.search.toLowerCase();
    filtered = filtered.filter((p) => {
      const titleMatch = p.metadata.title.toLowerCase().includes(searchLower);
      const abstractMatch = p.metadata.abstract
        .toLowerCase()
        .includes(searchLower);
      const authorMatch = p.metadata.authors.some((a) =>
        a.name.toLowerCase().includes(searchLower)
      );
      return titleMatch || abstractMatch || authorMatch;
    });
  }

  // Minimum relevance filter
  if (filters.minRelevance !== undefined) {
    filtered = filtered.filter(
      (p) =>
        p.analysis && p.analysis.relevanceScore >= filters.minRelevance!
    );
  }

  // Sorting
  const sortBy = filters.sortBy || "addedAt";
  const sortOrder = filters.sortOrder || "desc";
  const multiplier = sortOrder === "desc" ? -1 : 1;

  filtered.sort((a, b) => {
    switch (sortBy) {
      case "relevanceScore":
        const scoreA = a.analysis?.relevanceScore ?? 0;
        const scoreB = b.analysis?.relevanceScore ?? 0;
        return (scoreA - scoreB) * multiplier;
      case "citationCount":
        const citesA = a.metadata.citationCount ?? 0;
        const citesB = b.metadata.citationCount ?? 0;
        return (citesA - citesB) * multiplier;
      case "publishedDate":
        const dateA = a.metadata.publishedDate || "";
        const dateB = b.metadata.publishedDate || "";
        return dateA.localeCompare(dateB) * multiplier;
      case "addedAt":
      default:
        return a.addedAt.localeCompare(b.addedAt) * multiplier;
    }
  });

  return filtered;
}

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

    // Load and filter papers
    const allPapers = loadPapers();
    const filtered = filterPapers(allPapers, filters);
    const total = filtered.length;

    // Apply pagination
    let papers = filtered;
    if (filters.offset) {
      papers = papers.slice(filters.offset);
    }
    if (filters.limit) {
      papers = papers.slice(0, filters.limit);
    }

    const response: PaperListResponse = {
      papers,
      total,
      filters,
    };

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

    // Check if paper already exists
    const papers = loadPapers();
    const existing = papers.find(
      (p) =>
        p.metadata.id === validation.detection?.identifier ||
        p.metadata.url === input
    );

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

    // Save to storage
    papers.push(result.paper);
    savePapers(papers);

    return NextResponse.json({
      success: true,
      paper: result.paper,
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

    const papers = loadPapers();
    const index = papers.findIndex((p) => p.id === id);

    if (index === -1) {
      return NextResponse.json(
        { error: "Paper not found" },
        { status: 404 }
      );
    }

    // Update fields
    if (status) {
      papers[index].status = status;
    }
    if (notes !== undefined) {
      papers[index].notes = notes;
    }
    papers[index].updatedAt = new Date().toISOString();

    savePapers(papers);

    return NextResponse.json({
      success: true,
      paper: papers[index],
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

    const papers = loadPapers();
    const index = papers.findIndex((p) => p.id === id);

    if (index === -1) {
      return NextResponse.json(
        { error: "Paper not found" },
        { status: 404 }
      );
    }

    papers.splice(index, 1);
    savePapers(papers);

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
