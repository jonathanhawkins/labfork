/**
 * Firefly Lab API
 *
 * GET /api/labs/firefly - Get complete Firefly Lab data
 * POST /api/labs/firefly - Seed the Firefly Lab with initial data
 */

import { NextRequest, NextResponse } from "next/server";
import {
  insertOne,
  findOne,
  writeCollection,
  readCollection,
  COLLECTIONS,
} from "@/lib/db/json-store";
import type { Lab } from "@/lib/labs/types";
import {
  FIREFLY_LAB,
  FIREFLY_LAB_ID,
  FIREFLY_PAPERS,
  FIREFLY_TASKS,
  FIREFLY_AGENTS,
  FIREFLY_ACTIVITIES,
  FIREFLY_RESULTS,
  FIREFLY_BOM,
  BOM_SUMMARY,
  type LabPaper,
  type LabTask,
  type LabAgent,
  type LabPublishedResult,
  type BOMItem,
} from "@/lib/labs/firefly-seed-data";

// Additional collections for Firefly data
const FIREFLY_COLLECTIONS = {
  PAPERS: "firefly_papers",
  TASKS: "firefly_tasks",
  AGENTS: "firefly_agents",
  RESULTS: "firefly_results",
  BOM: "firefly_bom",
};

/**
 * GET /api/labs/firefly
 * Get complete Firefly Lab data including papers, tasks, agents, results
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const include = searchParams.get("include")?.split(",") || [
      "lab",
      "papers",
      "tasks",
      "agents",
      "activities",
      "results",
      "bom",
    ];

    const response: Record<string, unknown> = {
      success: true,
    };

    // Get lab
    if (include.includes("lab")) {
      const lab = await findOne<Lab>(
        COLLECTIONS.LABS,
        (l) => l.id === FIREFLY_LAB_ID
      );
      response.lab = lab || FIREFLY_LAB;
    }

    // Get papers
    if (include.includes("papers")) {
      const papers = await readCollection<LabPaper>(FIREFLY_COLLECTIONS.PAPERS);
      response.papers = papers.length > 0 ? papers : FIREFLY_PAPERS;
      response.paperStats = {
        total: (response.papers as LabPaper[]).length,
        implemented: (response.papers as LabPaper[]).filter(
          (p) => p.status === "implemented"
        ).length,
        implementing: (response.papers as LabPaper[]).filter(
          (p) => p.status === "implementing"
        ).length,
        reading: (response.papers as LabPaper[]).filter(
          (p) => p.status === "reading"
        ).length,
        toRead: (response.papers as LabPaper[]).filter(
          (p) => p.status === "to-read"
        ).length,
      };
    }

    // Get tasks
    if (include.includes("tasks")) {
      const tasks = await readCollection<LabTask>(FIREFLY_COLLECTIONS.TASKS);
      response.tasks = tasks.length > 0 ? tasks : FIREFLY_TASKS;
      response.taskStats = {
        total: (response.tasks as LabTask[]).length,
        completed: (response.tasks as LabTask[]).filter(
          (t) => t.status === "completed"
        ).length,
        inProgress: (response.tasks as LabTask[]).filter(
          (t) => t.status === "in_progress"
        ).length,
        pending: (response.tasks as LabTask[]).filter(
          (t) => t.status === "pending"
        ).length,
        blocked: (response.tasks as LabTask[]).filter(
          (t) => t.status === "blocked"
        ).length,
      };
    }

    // Get agents
    if (include.includes("agents")) {
      const agents = await readCollection<LabAgent>(FIREFLY_COLLECTIONS.AGENTS);
      response.agents = agents.length > 0 ? agents : FIREFLY_AGENTS;
      response.agentStats = {
        total: (response.agents as LabAgent[]).length,
        working: (response.agents as LabAgent[]).filter(
          (a) => a.status === "working"
        ).length,
        totalCost: (response.agents as LabAgent[]).reduce(
          (sum, a) => sum + a.costEstimate,
          0
        ),
        totalTokens: (response.agents as LabAgent[]).reduce(
          (sum, a) => sum + a.tokensGenerated,
          0
        ),
      };
    }

    // Get activities
    if (include.includes("activities")) {
      const activities = await readCollection(COLLECTIONS.ACTIVITIES);
      const fireflyActivities = (activities as { labId?: string }[]).filter(
        (a) => a.labId === FIREFLY_LAB_ID || !a.labId
      );
      response.activities =
        fireflyActivities.length > 0 ? fireflyActivities : FIREFLY_ACTIVITIES;
    }

    // Get results
    if (include.includes("results")) {
      const results = await readCollection<LabPublishedResult>(
        FIREFLY_COLLECTIONS.RESULTS
      );
      response.results = results.length > 0 ? results : FIREFLY_RESULTS;
      response.resultStats = {
        total: (response.results as LabPublishedResult[]).length,
        totalLikes: (response.results as LabPublishedResult[]).reduce(
          (sum, r) => sum + r.likes,
          0
        ),
        totalComments: (response.results as LabPublishedResult[]).reduce(
          (sum, r) => sum + r.comments.length,
          0
        ),
      };
    }

    // Get BOM
    if (include.includes("bom")) {
      const bom = await readCollection<BOMItem>(FIREFLY_COLLECTIONS.BOM);
      response.bom = bom.length > 0 ? bom : FIREFLY_BOM;
      response.bomSummary = BOM_SUMMARY;
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error("Failed to get Firefly Lab data:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get data",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/labs/firefly
 * Seed the Firefly Lab with initial data
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    const force = body.force === true;

    // Check if lab already exists
    const existingLab = await findOne<Lab>(
      COLLECTIONS.LABS,
      (l) => l.id === FIREFLY_LAB_ID
    );

    if (existingLab && !force) {
      return NextResponse.json({
        success: true,
        message: "Firefly Lab already exists",
        lab: existingLab,
        seeded: false,
      });
    }

    // Seed lab
    if (!existingLab) {
      await insertOne(COLLECTIONS.LABS, FIREFLY_LAB);
    }

    // Seed papers
    await writeCollection(FIREFLY_COLLECTIONS.PAPERS, FIREFLY_PAPERS);

    // Seed tasks
    await writeCollection(FIREFLY_COLLECTIONS.TASKS, FIREFLY_TASKS);

    // Seed agents
    await writeCollection(FIREFLY_COLLECTIONS.AGENTS, FIREFLY_AGENTS);

    // Seed activities
    const existingActivities = await readCollection(COLLECTIONS.ACTIVITIES);
    const fireflyActivityIds = new Set(FIREFLY_ACTIVITIES.map((a) => a.id));
    const otherActivities = (existingActivities as { id: string }[]).filter(
      (a) => !fireflyActivityIds.has(a.id)
    );
    await writeCollection(COLLECTIONS.ACTIVITIES, [
      ...otherActivities,
      ...FIREFLY_ACTIVITIES,
    ]);

    // Seed results
    await writeCollection(FIREFLY_COLLECTIONS.RESULTS, FIREFLY_RESULTS);

    // Seed BOM
    await writeCollection(FIREFLY_COLLECTIONS.BOM, FIREFLY_BOM);

    return NextResponse.json({
      success: true,
      message: "Firefly Lab seeded successfully",
      lab: FIREFLY_LAB,
      seeded: true,
      counts: {
        papers: FIREFLY_PAPERS.length,
        tasks: FIREFLY_TASKS.length,
        agents: FIREFLY_AGENTS.length,
        activities: FIREFLY_ACTIVITIES.length,
        results: FIREFLY_RESULTS.length,
        bomItems: FIREFLY_BOM.length,
      },
    });
  } catch (error) {
    console.error("Failed to seed Firefly Lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to seed lab",
      },
      { status: 500 }
    );
  }
}
