/**
 * Fork API
 *
 * POST /api/labs/[id]/fork - Fork a lab
 * GET /api/labs/[id]/fork - Get fork info
 */

import { NextRequest, NextResponse } from "next/server";
import { getLabById } from "@/lib/labs/repository";
import {
  forkLab,
  getForkCount,
  getLabForks,
  getLabLineage,
} from "@/lib/labs/social";
import { canViewLab } from "@/lib/labs/types";
import { getServerUser } from "@/lib/auth/server";
import { userToLabOwner } from "@/lib/auth/mock-user";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/labs/[id]/fork
 * Get fork information for a lab
 */
export async function GET(
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

    // Check visibility (user may be null if not authenticated)
    const user = await getServerUser();
    if (!canViewLab(lab, user?.id)) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Get fork info
    const forkCount = await getForkCount(id);
    const forks = await getLabForks(id);
    const lineage = await getLabLineage(id);

    return NextResponse.json({
      success: true,
      forkCount,
      forks: forks.map((f) => ({
        id: f.id,
        name: f.name,
        slug: f.slug,
        owner: f.owner,
        createdAt: f.createdAt,
      })),
      lineage: lineage.map((l) => ({
        id: l.id,
        name: l.name,
        slug: l.slug,
        owner: l.owner,
      })),
      forkedFrom: lab.forkedFrom,
    });
  } catch (error) {
    console.error("Failed to get fork info:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get fork info",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/labs/[id]/fork
 * Fork a lab
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    // Get user
    const user = await getServerUser();
    if (!user) {
      return NextResponse.json(
        { success: false, error: "Authentication required" },
        { status: 401 }
      );
    }

    // Get lab
    const lab = await getLabById(id);
    if (!lab) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Check visibility
    if (!canViewLab(lab, user.id)) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Get optional new slug from body
    let newSlug: string | undefined;
    try {
      const body = await request.json();
      newSlug = body.slug;
    } catch {
      // No body, use default slug
    }

    // Fork the lab
    const owner = userToLabOwner(user);
    const forkedLab = await forkLab(id, owner, newSlug);

    return NextResponse.json({
      success: true,
      lab: forkedLab,
      message: `Forked "${lab.name}" successfully`,
    });
  } catch (error) {
    console.error("Failed to fork lab:", error);

    // Handle duplicate slug error
    if (error instanceof Error && error.message.includes("already exists")) {
      return NextResponse.json(
        { success: false, error: "You already have a lab with this name. Please choose a different slug." },
        { status: 409 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to fork lab",
      },
      { status: 500 }
    );
  }
}
