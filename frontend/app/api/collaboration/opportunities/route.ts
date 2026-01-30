/**
 * Collaboration Opportunities API
 *
 * GET /api/collaboration/opportunities - List/generate collaboration opportunities
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalCollaborationMatcher,
  getGlobalMetaTaskManager,
  getOpenTasks,
  createDefaultLabAvailability,
  createDefaultCollaborationPreferences,
  LabProfile,
} from "@/lib/meta/collaboration";

/**
 * GET /api/collaboration/opportunities
 * List collaboration opportunities for a lab
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const labId = searchParams.get("labId");
    const limit = parseInt(searchParams.get("limit") || "10", 10);
    const minScore = parseFloat(searchParams.get("minScore") || "0.5");

    if (!labId) {
      return NextResponse.json(
        { success: false, error: "labId is required" },
        { status: 400 }
      );
    }

    const matcher = getGlobalCollaborationMatcher();
    const manager = getGlobalMetaTaskManager();

    // Get lab profile if registered
    let labProfile = matcher.labs.get(labId);
    if (!labProfile) {
      // Create a basic profile for unregistered labs
      labProfile = {
        id: labId,
        name: labId,
        description: "",
        expertise: [],
        domains: [],
        resources: [],
        pastCollaborations: [],
        availability: createDefaultLabAvailability(),
        preferences: createDefaultCollaborationPreferences(),
      };
    }

    // Generate opportunities
    const opportunities = matcher.generateOpportunities(
      labProfile,
      manager,
      limit
    );

    // Filter by minimum score
    const filtered = opportunities.filter((o) => o.matchScore >= minScore);

    // Also include open tasks that might be relevant
    const openTasks = getOpenTasks(manager);
    const openTaskOpportunities = openTasks.map((task) => ({
      id: task.id,
      type: "meta_task" as const,
      title: task.title,
      description: task.description,
      matchScore: 0.7, // Default score for open tasks
      benefits: [],
      requirements: task.requirements.requiredExpertise,
      estimatedEffort: {
        hours: 10,
        weeks: Math.ceil(
          task.timeline.phases.length > 0
            ? task.timeline.phases.reduce((sum, p) => {
                const start = new Date(p.startDate);
                const end = new Date(p.endDate);
                return sum + (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 7);
              }, 0)
            : 4
        ),
      },
      deadline: task.timeline.recruitmentDeadline,
      metaTask: task,
      createdAt: task.createdAt,
    }));

    return NextResponse.json({
      success: true,
      opportunities: [...filtered, ...openTaskOpportunities].slice(0, limit),
      total: filtered.length + openTaskOpportunities.length,
    });
  } catch (error) {
    console.error("Failed to get opportunities:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get opportunities",
      },
      { status: 500 }
    );
  }
}
