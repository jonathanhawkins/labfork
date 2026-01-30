/**
 * Labs API - List and Create
 *
 * GET /api/labs - List labs with filtering
 * POST /api/labs - Create a new lab
 */

import { NextRequest, NextResponse } from "next/server";
import {
  listLabs,
  createLab,
  searchLabs,
} from "@/lib/labs/repository";
import type { LabListOptions, CreateLabInput } from "@/lib/labs/types";
import { getServerUser, userToLabOwner } from "@/lib/auth/mock-user";

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
    if (
      sortBy === "stars" ||
      sortBy === "forks" ||
      sortBy === "activity" ||
      sortBy === "created" ||
      sortBy === "name"
    ) {
      options.sortBy = sortBy;
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
