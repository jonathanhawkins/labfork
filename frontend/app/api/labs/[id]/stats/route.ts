/**
 * Lab Stats API
 *
 * GET /api/labs/[id]/stats - Get lab statistics
 */

import { NextRequest, NextResponse } from "next/server";
import { getLabById } from "@/lib/labs/repository";
import { getLabSocialStats, getLabForks, getLabLineage } from "@/lib/labs/social";
import { canViewLab } from "@/lib/labs/types";
import { getServerUser } from "@/lib/auth/server";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/labs/[id]/stats
 * Get comprehensive lab statistics
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

    // Check visibility
    const user = await getServerUser();
    if (!canViewLab(lab, user?.id)) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    // Get social stats
    const socialStats = await getLabSocialStats(id, user?.id);

    // Get forks and lineage
    const forks = await getLabForks(id);
    const lineage = await getLabLineage(id);

    return NextResponse.json({
      success: true,
      stats: {
        ...lab.stats,
        ...socialStats,
      },
      social: {
        stars: socialStats.stars,
        forks: socialStats.forks,
        isStarred: socialStats.isStarred,
        isFork: socialStats.isFork,
      },
      forkInfo: {
        count: forks.length,
        forkedFrom: lab.forkedFrom,
        forks: forks.slice(0, 5).map((f) => ({
          id: f.id,
          name: f.name,
          slug: f.slug,
          owner: f.owner.username,
        })),
        lineageDepth: lineage.length,
      },
      activity: {
        lastActivity: lab.lastActivityAt,
        createdAt: lab.createdAt,
        updatedAt: lab.updatedAt,
      },
    });
  } catch (error) {
    console.error("Failed to get lab stats:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get lab stats",
      },
      { status: 500 }
    );
  }
}
