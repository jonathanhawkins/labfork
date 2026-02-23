/**
 * Labs API - List and Create
 *
 * GET /api/labs - List labs with filtering
 * POST /api/labs - Create a new lab (and sync to Workers)
 */

import { NextRequest, NextResponse } from "next/server";
import {
  listLabs,
  createLab,
  searchLabs,
} from "@/lib/labs/repository";
import type { Lab, LabListOptions, CreateLabInput } from "@/lib/labs/types";
import { getServerUser } from "@/lib/auth/server";
import { userToLabOwner } from "@/lib/auth/mock-user";

// Workers API URL (uses env var or defaults to production)
const WORKERS_API_URL = process.env.WORKERS_API_URL || "https://labfork-agents.workers.dev";

/**
 * Sync a single lab to Workers compute network
 */
async function syncLabToWorkers(lab: Lab): Promise<boolean> {
  try {
    const response = await fetch(`${WORKERS_API_URL}/api/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: lab.id,
        slug: lab.slug,
        name: lab.name,
        description: lab.description,
        domainSlug: lab.domainSlug,
        domainName: lab.domainName,
        tags: lab.tags,
        status: lab.status === "active" ? "active" : "paused",
      }),
    });

    if (!response.ok) {
      console.error("[Labs] Workers sync failed:", response.status);
      return false;
    }

    console.log(`[Labs] Synced lab ${lab.slug} to Workers`);
    return true;
  } catch (error) {
    console.error("[Labs] Workers sync error:", error);
    return false;
  }
}

/**
 * GET /api/labs
 * List labs with optional filters
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Build options from query params
    const options: LabListOptions = {};

    const owner = searchParams.get("owner");
    if (owner) options.owner = owner;

    const domain = searchParams.get("domain");
    if (domain) options.domain = domain;

    const visibility = searchParams.get("visibility");
    if (visibility === "public" || visibility === "private" || visibility === "unlisted") {
      options.visibility = visibility;
    }

    const status = searchParams.get("status");
    if (status === "active" || status === "archived" || status === "suspended") {
      options.status = status;
    }

    const tags = searchParams.get("tags");
    if (tags) options.tags = tags.split(",");

    const search = searchParams.get("search") || searchParams.get("q");
    if (search) options.search = search;

    const sortBy = searchParams.get("sortBy") || searchParams.get("sort");
    // Map frontend sort options to repository sort options
    const sortMapping: Record<string, "stars" | "forks" | "activity" | "created" | "name"> = {
      popular: "stars",
      trending: "activity",
      recent: "created",
      active: "activity",
      stars: "stars",
      forks: "forks",
      activity: "activity",
      created: "created",
      name: "name",
    };
    if (sortBy && sortMapping[sortBy]) {
      options.sortBy = sortMapping[sortBy];
    }

    const sortDir = searchParams.get("sortDir") || searchParams.get("order");
    if (sortDir === "asc" || sortDir === "desc") {
      options.sortDir = sortDir;
    }

    const page = searchParams.get("page");
    if (page) options.page = parseInt(page, 10);

    const limit = searchParams.get("limit");
    if (limit) options.limit = Math.min(parseInt(limit, 10), 100);

    // Execute query
    const result = await listLabs(options);

    // If no labs found, return empty state with helpful message
    if (result.labs.length === 0 && !options.search && !options.owner) {
      return NextResponse.json({
        success: true,
        labs: [],
        total: 0,
        page: 1,
        totalPages: 0,
        hasMore: false,
        isEmpty: true,
        message: "No labs yet. Be the first to create one!",
      });
    }

    return NextResponse.json({
      success: true,
      ...result,
    });
  } catch (error) {
    console.error("Failed to list labs:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to list labs",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/labs
 * Create a new lab
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.name || typeof body.name !== "string") {
      return NextResponse.json(
        { success: false, error: "Name is required" },
        { status: 400 }
      );
    }

    if (!body.domainSlug || typeof body.domainSlug !== "string") {
      return NextResponse.json(
        { success: false, error: "Domain slug is required" },
        { status: 400 }
      );
    }

    // Get current user
    const user = await getServerUser();
    if (!user) {
      return NextResponse.json(
        { success: false, error: "Authentication required" },
        { status: 401 }
      );
    }
    const owner = userToLabOwner(user);

    // Build input
    const input: CreateLabInput = {
      name: body.name,
      slug: body.slug || undefined,
      description: body.description || "",
      domainSlug: body.domainSlug,
      visibility: body.visibility || "public",
      tags: body.tags || [],
      primaryColor: body.primaryColor,
      readme: body.readme,
    };

    // Create lab
    const lab = await createLab(input, owner);

    // Sync to Workers compute network (non-blocking, don't fail if this fails)
    syncLabToWorkers(lab).catch((err) => {
      console.error("[Labs] Background sync failed:", err);
    });

    return NextResponse.json({
      success: true,
      lab,
    });
  } catch (error) {
    console.error("Failed to create lab:", error);

    // Handle duplicate slug error
    if (error instanceof Error && error.message.includes("already exists")) {
      return NextResponse.json(
        { success: false, error: error.message },
        { status: 409 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to create lab",
      },
      { status: 500 }
    );
  }
}
