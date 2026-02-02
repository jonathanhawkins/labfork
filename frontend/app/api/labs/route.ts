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
import type { Lab, LabListOptions, CreateLabInput } from "@/lib/labs/types";
import { getServerUser, userToLabOwner } from "@/lib/auth/mock-user";

// Demo labs for production fallback (when no DB data)
const DEMO_LABS: Lab[] = [
  {
    id: "demo_1",
    slug: "protein-folding",
    name: "Protein Folding Explorer",
    description: "AlphaFold-based analysis and prediction for novel protein structures",
    domainSlug: "biotech",
    domainName: "Biotech",
    owner: { id: "demo_user_1", username: "spark_research", displayName: "Spark Research", avatar: "" },
    visibility: "public",
    status: "active",
    stats: { stars: 234, forks: 67, tasks: 22, papers: 28, experiments: 15, views: 1200, viewers: 3 },
    tags: ["alphafold", "protein", "structure"],
    primaryColor: "#ec4899",
    createdAt: "2026-01-15T10:00:00Z",
    updatedAt: "2026-02-01T14:30:00Z",
    lastActivityAt: "2026-02-01T14:30:00Z",
  },
  {
    id: "demo_2",
    slug: "transformer-signals",
    name: "Transformer Signals",
    description: "Momentum signal generation using transformer architectures for algorithmic trading",
    domainSlug: "quant-trading",
    domainName: "Quantitative Trading",
    owner: { id: "demo_user_2", username: "ai_researcher", displayName: "AI Researcher", avatar: "" },
    visibility: "public",
    status: "active",
    stats: { stars: 203, forks: 56, tasks: 18, papers: 15, experiments: 12, views: 980, viewers: 2 },
    tags: ["transformer", "momentum", "signals", "ml"],
    primaryColor: "#22c55e",
    createdAt: "2026-01-10T08:00:00Z",
    updatedAt: "2026-02-02T09:00:00Z",
    lastActivityAt: "2026-02-02T09:00:00Z",
  },
  {
    id: "demo_3",
    slug: "zero-shot-clone",
    name: "Zero-Shot Voice Cloning",
    description: "Speaker adaptation with minimal reference audio using VALL-E and Pocket TTS",
    domainSlug: "voice-clone",
    domainName: "Voice Cloning",
    owner: { id: "demo_user_3", username: "voice_pioneer", displayName: "Voice Pioneer", avatar: "" },
    visibility: "public",
    status: "active",
    stats: { stars: 156, forks: 42, tasks: 9, papers: 7, experiments: 6, views: 750, viewers: 1 },
    tags: ["zero-shot", "vall-e", "speaker-adaptation"],
    primaryColor: "#3b82f6",
    createdAt: "2026-01-20T12:00:00Z",
    updatedAt: "2026-01-30T16:00:00Z",
    lastActivityAt: "2026-01-30T16:00:00Z",
  },
  {
    id: "demo_4",
    slug: "firefly-core",
    name: "Firefly Core",
    description: "Core MPPT algorithms and hardware integration for the Firefly mesh network",
    domainSlug: "firefly-network",
    domainName: "Firefly Network",
    owner: { id: "demo_user_1", username: "spark_research", displayName: "Spark Research", avatar: "" },
    visibility: "public",
    status: "active",
    stats: { stars: 128, forks: 34, tasks: 12, papers: 8, experiments: 8, views: 620, viewers: 0 },
    tags: ["mppt", "solar", "mesh", "hardware"],
    primaryColor: "#f59e0b",
    createdAt: "2026-01-05T09:00:00Z",
    updatedAt: "2026-01-28T11:00:00Z",
    lastActivityAt: "2026-01-28T11:00:00Z",
  },
  {
    id: "demo_5",
    slug: "sim2real",
    name: "Sim2Real Transfer",
    description: "Domain randomization and transfer learning for robotic manipulation",
    domainSlug: "robotics",
    domainName: "Robotics",
    owner: { id: "demo_user_4", username: "open_hardware", displayName: "Open Hardware", avatar: "" },
    visibility: "public",
    status: "active",
    stats: { stars: 112, forks: 31, tasks: 11, papers: 14, experiments: 9, views: 540, viewers: 0 },
    tags: ["sim2real", "manipulation", "transfer-learning"],
    primaryColor: "#a855f7",
    createdAt: "2026-01-08T15:00:00Z",
    updatedAt: "2026-01-25T10:00:00Z",
    lastActivityAt: "2026-01-25T10:00:00Z",
  },
  {
    id: "demo_6",
    slug: "emotion-tts",
    name: "Emotion TTS Research",
    description: "Prosody and emotion control in text-to-speech synthesis using neural networks",
    domainSlug: "voice-clone",
    domainName: "Voice Cloning",
    owner: { id: "demo_user_3", username: "voice_pioneer", displayName: "Voice Pioneer", avatar: "" },
    visibility: "public",
    status: "active",
    stats: { stars: 89, forks: 27, tasks: 15, papers: 12, experiments: 10, views: 430, viewers: 0 },
    tags: ["tts", "prosody", "emotion", "neural"],
    primaryColor: "#3b82f6",
    createdAt: "2026-01-12T14:00:00Z",
    updatedAt: "2026-01-22T08:00:00Z",
    lastActivityAt: "2026-01-22T08:00:00Z",
  },
];

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

    // If no labs found, return demo labs as fallback
    if (result.labs.length === 0 && !options.search && !options.owner) {
      // Sort demo labs by the requested sort option
      let sortedDemoLabs = [...DEMO_LABS];
      if (options.sortBy === "stars") {
        sortedDemoLabs.sort((a, b) => (b.stats?.stars || 0) - (a.stats?.stars || 0));
      } else if (options.sortBy === "forks") {
        sortedDemoLabs.sort((a, b) => (b.stats?.forks || 0) - (a.stats?.forks || 0));
      } else if (options.sortBy === "activity" || options.sortBy === "created") {
        sortedDemoLabs.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
      }

      // Filter by domain if specified
      if (options.domain) {
        sortedDemoLabs = sortedDemoLabs.filter(lab => lab.domainSlug === options.domain);
      }

      return NextResponse.json({
        success: true,
        labs: sortedDemoLabs.slice(0, options.limit || 20),
        total: sortedDemoLabs.length,
        page: 1,
        totalPages: 1,
        hasMore: false,
        isDemo: true,
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
