/**
 * Lab Matching API
 *
 * GET /api/collaboration/matches - Find matching labs for collaboration
 * POST /api/collaboration/matches/register - Register lab profile
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalCollaborationMatcher,
  getGlobalMetaTaskManager,
  getTask,
  LabProfile,
  MetaTaskId,
  createDefaultLabAvailability,
  createDefaultCollaborationPreferences,
} from "@/lib/meta/collaboration";

/**
 * GET /api/collaboration/matches
 * Find matching labs for a task or general collaboration
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const matcher = getGlobalCollaborationMatcher();
    const manager = getGlobalMetaTaskManager();

    const metaTaskId = searchParams.get("metaTaskId") as MetaTaskId | null;
    const labId = searchParams.get("labId");
    const limit = parseInt(searchParams.get("limit") || "10", 10);
    const minScore = parseFloat(searchParams.get("minScore") || "0.3");

    if (metaTaskId) {
      // Find labs for a specific task
      const task = getTask(manager, metaTaskId);
      if (!task) {
        return NextResponse.json(
          { success: false, error: "Task not found" },
          { status: 404 }
        );
      }

      const matches = matcher.findLabsForTask(task, limit);
      const filtered = matches.filter((m) => m.matchScore >= minScore);

      // Compose optimal team if requested
      const compose = searchParams.get("compose") === "true";
      let teamComposition = null;
      if (compose) {
        teamComposition = matcher.composeOptimalTeam(task, limit);
      }

      return NextResponse.json({
        success: true,
        taskId: metaTaskId,
        taskTitle: task.title,
        matches: filtered,
        total: filtered.length,
        teamComposition,
      });
    } else if (labId) {
      // Find matching labs for general collaboration
      const labProfile = matcher.labs.get(labId);
      if (!labProfile) {
        return NextResponse.json(
          { success: false, error: "Lab not registered. Please register first." },
          { status: 404 }
        );
      }

      const matches = matcher.findMatchingLabs(labProfile, limit);
      const filtered = matches.filter((m) => m.matchScore >= minScore);

      return NextResponse.json({
        success: true,
        labId,
        matches: filtered,
        total: filtered.length,
      });
    } else {
      // List all registered labs (public profiles)
      const labs = matcher.getAllLabs().map((lab) => ({
        id: lab.id,
        name: lab.name,
        description: lab.description,
        expertise: lab.expertise,
        domains: lab.domains,
        availability: lab.availability.hoursPerWeek,
        openToCollaboration: lab.preferences.openToNewCollaborators,
      }));

      return NextResponse.json({
        success: true,
        labs,
        total: labs.length,
      });
    }
  } catch (error) {
    console.error("Failed to find matches:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to find matches",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/collaboration/matches
 * Register or update lab profile for matching
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const matcher = getGlobalCollaborationMatcher();

    // Validate required fields
    if (!body.id) {
      return NextResponse.json(
        { success: false, error: "Lab id is required" },
        { status: 400 }
      );
    }

    if (!body.name) {
      return NextResponse.json(
        { success: false, error: "Lab name is required" },
        { status: 400 }
      );
    }

    // Build lab profile
    const profile: LabProfile = {
      id: body.id,
      name: body.name,
      description: body.description || "",
      expertise: body.expertise || [],
      domains: body.domains || [],
      resources: body.resources || [],
      pastCollaborations: body.pastCollaborations || [],
      availability: body.availability || createDefaultLabAvailability(),
      preferences: body.preferences || createDefaultCollaborationPreferences(),
    };

    // Validate availability
    if (profile.availability.hoursPerWeek < 0) {
      return NextResponse.json(
        { success: false, error: "hoursPerWeek must be non-negative" },
        { status: 400 }
      );
    }

    // Register or update
    const isUpdate = matcher.labs.has(body.id);
    matcher.registerLab(profile);

    return NextResponse.json({
      success: true,
      profile,
      message: isUpdate ? "Lab profile updated" : "Lab profile registered",
    });
  } catch (error) {
    console.error("Failed to register lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to register lab",
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/collaboration/matches
 * Unregister lab from matching
 */
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const matcher = getGlobalCollaborationMatcher();

    const labId = searchParams.get("labId");
    if (!labId) {
      return NextResponse.json(
        { success: false, error: "labId is required" },
        { status: 400 }
      );
    }

    if (!matcher.labs.has(labId)) {
      return NextResponse.json(
        { success: false, error: "Lab not found" },
        { status: 404 }
      );
    }

    matcher.labs.delete(labId);

    return NextResponse.json({
      success: true,
      message: "Lab unregistered from matching",
    });
  } catch (error) {
    console.error("Failed to unregister lab:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to unregister lab",
      },
      { status: 500 }
    );
  }
}
