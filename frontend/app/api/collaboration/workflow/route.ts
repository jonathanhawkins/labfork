/**
 * Workflow API
 *
 * GET /api/collaboration/workflow - Get workflow summary
 * GET /api/collaboration/workflow?taskId=... - Get task progress
 * POST /api/collaboration/workflow/conflicts - Report conflict
 * PATCH /api/collaboration/workflow/conflicts - Resolve conflict
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalWorkflowCoordinator,
  getGlobalMetaTaskManager,
  getWorkflowSummary,
  getProgressTracker,
  initializeProgressTracker,
  updateProgress,
  reportConflict,
  proposeResolution,
  voteOnResolution,
  getConflictsForTask,
  getOpenConflicts,
  getOverdueAssignments,
  getBlockedAssignments,
  MetaTaskId,
  ParticipantId,
  ConflictType,
  CreateConflictInput,
} from "@/lib/meta/collaboration";

/**
 * GET /api/collaboration/workflow
 * Get workflow summary or task progress
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const coordinator = getGlobalWorkflowCoordinator();
    const manager = getGlobalMetaTaskManager();

    const metaTaskId = searchParams.get("taskId") as MetaTaskId | null;
    const view = searchParams.get("view"); // 'summary', 'conflicts', 'assignments'

    if (metaTaskId) {
      // Get or initialize progress tracker for specific task
      let tracker = getProgressTracker(coordinator, metaTaskId);
      if (!tracker) {
        tracker = initializeProgressTracker(coordinator, manager, metaTaskId);
      }

      if (!tracker) {
        return NextResponse.json(
          { success: false, error: "Task not found" },
          { status: 404 }
        );
      }

      // Get conflicts for this task
      const conflicts = getConflictsForTask(coordinator, metaTaskId);

      return NextResponse.json({
        success: true,
        taskId: metaTaskId,
        progress: tracker,
        conflicts: conflicts.filter((c) => c.status !== "resolved"),
      });
    } else if (view === "conflicts") {
      // Get all open conflicts
      const conflicts = getOpenConflicts(coordinator);
      return NextResponse.json({
        success: true,
        conflicts,
        total: conflicts.length,
      });
    } else if (view === "assignments") {
      // Get problematic assignments
      const overdue = getOverdueAssignments(coordinator);
      const blocked = getBlockedAssignments(coordinator);
      return NextResponse.json({
        success: true,
        overdue,
        blocked,
        totalOverdue: overdue.length,
        totalBlocked: blocked.length,
      });
    } else {
      // Get overall workflow summary
      const summary = getWorkflowSummary(coordinator);
      return NextResponse.json({
        success: true,
        summary,
      });
    }
  } catch (error) {
    console.error("Failed to get workflow data:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get workflow data",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/collaboration/workflow
 * Report a conflict or update progress
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const coordinator = getGlobalWorkflowCoordinator();
    const manager = getGlobalMetaTaskManager();

    const { action } = body;

    switch (action) {
      case "report_conflict": {
        // Validate required fields
        if (!body.metaTaskId) {
          return NextResponse.json(
            { success: false, error: "metaTaskId is required" },
            { status: 400 }
          );
        }

        if (!body.type || !body.description) {
          return NextResponse.json(
            { success: false, error: "type and description are required" },
            { status: 400 }
          );
        }

        if (!body.parties || !Array.isArray(body.parties) || body.parties.length < 2) {
          return NextResponse.json(
            { success: false, error: "At least 2 parties are required" },
            { status: 400 }
          );
        }

        if (!body.reportedBy) {
          return NextResponse.json(
            { success: false, error: "reportedBy is required" },
            { status: 400 }
          );
        }

        // Validate conflict type
        const validTypes: ConflictType[] = [
          "resource_allocation",
          "priority_disagreement",
          "methodology_dispute",
          "result_interpretation",
          "credit_attribution",
          "timeline_conflict",
          "scope_creep",
        ];
        if (!validTypes.includes(body.type)) {
          return NextResponse.json(
            { success: false, error: `Invalid conflict type: ${body.type}` },
            { status: 400 }
          );
        }

        const input: CreateConflictInput = {
          metaTaskId: body.metaTaskId as MetaTaskId,
          type: body.type as ConflictType,
          description: body.description,
          severity: body.severity || "medium",
          parties: body.parties as ParticipantId[],
          reportedBy: body.reportedBy as ParticipantId,
        };

        const conflict = reportConflict(coordinator, manager, input);

        return NextResponse.json({
          success: true,
          conflict,
        });
      }

      case "propose_resolution": {
        if (!body.conflictId) {
          return NextResponse.json(
            { success: false, error: "conflictId is required" },
            { status: 400 }
          );
        }

        if (!body.description || !body.actions) {
          return NextResponse.json(
            { success: false, error: "description and actions are required" },
            { status: 400 }
          );
        }

        if (!body.proposedBy) {
          return NextResponse.json(
            { success: false, error: "proposedBy is required" },
            { status: 400 }
          );
        }

        const resolution = proposeResolution(
          coordinator,
          body.conflictId,
          body.proposedBy as ParticipantId,
          body.description,
          body.actions
        );

        if (!resolution) {
          return NextResponse.json(
            { success: false, error: "Conflict not found" },
            { status: 404 }
          );
        }

        return NextResponse.json({
          success: true,
          resolution,
        });
      }

      case "vote_resolution": {
        if (!body.conflictId || !body.resolutionId) {
          return NextResponse.json(
            { success: false, error: "conflictId and resolutionId are required" },
            { status: 400 }
          );
        }

        if (!body.participantId) {
          return NextResponse.json(
            { success: false, error: "participantId is required" },
            { status: 400 }
          );
        }

        if (!["approve", "reject", "abstain"].includes(body.vote)) {
          return NextResponse.json(
            { success: false, error: "vote must be approve, reject, or abstain" },
            { status: 400 }
          );
        }

        const resolution = voteOnResolution(
          coordinator,
          body.conflictId,
          body.resolutionId,
          body.participantId as ParticipantId,
          body.vote
        );

        if (!resolution) {
          return NextResponse.json(
            { success: false, error: "Conflict or resolution not found" },
            { status: 404 }
          );
        }

        return NextResponse.json({
          success: true,
          resolution,
        });
      }

      case "update_progress": {
        if (!body.metaTaskId || !body.objectiveId) {
          return NextResponse.json(
            { success: false, error: "metaTaskId and objectiveId are required" },
            { status: 400 }
          );
        }

        if (body.progress === undefined) {
          return NextResponse.json(
            { success: false, error: "progress is required" },
            { status: 400 }
          );
        }

        if (!body.actorId) {
          return NextResponse.json(
            { success: false, error: "actorId is required" },
            { status: 400 }
          );
        }

        const tracker = updateProgress(
          coordinator,
          manager,
          body.metaTaskId as MetaTaskId,
          body.objectiveId,
          body.progress,
          body.actorId as ParticipantId
        );

        if (!tracker) {
          return NextResponse.json(
            { success: false, error: "Task not found" },
            { status: 404 }
          );
        }

        return NextResponse.json({
          success: true,
          progress: tracker,
        });
      }

      default:
        return NextResponse.json(
          { success: false, error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error("Failed to process workflow action:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to process workflow action",
      },
      { status: 500 }
    );
  }
}
