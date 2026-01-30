/**
 * Meta-Task Detail API
 *
 * GET /api/collaboration/meta-tasks/[id] - Get task details
 * PATCH /api/collaboration/meta-tasks/[id] - Update task
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalMetaTaskManager,
  getTask,
  transitionStatus,
  addObjective,
  updateObjectiveProgress,
  submitResult,
  addCompletionCriterion,
  getTaskEvents,
  MetaTaskId,
  MetaTaskStatus,
  ParticipantId,
  CreateObjectiveInput,
  CreateResultInput,
} from "@/lib/meta/collaboration";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * GET /api/collaboration/meta-tasks/[id]
 * Get full task details
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const manager = getGlobalMetaTaskManager();
    const taskId = id as MetaTaskId;

    const task = getTask(manager, taskId);
    if (!task) {
      return NextResponse.json(
        { success: false, error: "Task not found" },
        { status: 404 }
      );
    }

    // Get recent events
    const events = getTaskEvents(manager, taskId).slice(-20);

    // Calculate progress
    const totalObjectives = task.objectives.length;
    const completedObjectives = task.objectives.filter(
      (o) => o.status === "completed"
    ).length;
    const overallProgress =
      totalObjectives > 0
        ? Math.round((completedObjectives / totalObjectives) * 100)
        : 0;

    return NextResponse.json({
      success: true,
      task: {
        ...task,
        progress: overallProgress,
        events,
      },
    });
  } catch (error) {
    console.error("Failed to get meta-task:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get meta-task",
      },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/collaboration/meta-tasks/[id]
 * Update task (status, objectives, results, etc.)
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;
    const body = await request.json();
    const manager = getGlobalMetaTaskManager();
    const taskId = id as MetaTaskId;

    const task = getTask(manager, taskId);
    if (!task) {
      return NextResponse.json(
        { success: false, error: "Task not found" },
        { status: 404 }
      );
    }

    // Require actorId for all mutations
    const actorId = body.actorId as ParticipantId;
    if (!actorId) {
      return NextResponse.json(
        { success: false, error: "actorId is required" },
        { status: 400 }
      );
    }

    // Handle different update types
    const { action } = body;

    switch (action) {
      case "transition_status": {
        const newStatus = body.status as MetaTaskStatus;
        if (!newStatus) {
          return NextResponse.json(
            { success: false, error: "status is required" },
            { status: 400 }
          );
        }
        const updated = transitionStatus(manager, taskId, newStatus, actorId);
        return NextResponse.json({
          success: true,
          task: updated,
        });
      }

      case "add_objective": {
        const objectiveInput: CreateObjectiveInput = {
          title: body.title,
          description: body.description,
          priority: body.priority || "medium",
          assignedTo: body.assignedTo,
          dependencies: body.dependencies,
          deadline: body.deadline,
        };

        if (!objectiveInput.title || !objectiveInput.description) {
          return NextResponse.json(
            { success: false, error: "title and description are required" },
            { status: 400 }
          );
        }

        const objective = addObjective(manager, taskId, objectiveInput, actorId);
        return NextResponse.json({
          success: true,
          objective,
        });
      }

      case "update_objective_progress": {
        const { objectiveId, progress } = body;
        if (!objectiveId || progress === undefined) {
          return NextResponse.json(
            { success: false, error: "objectiveId and progress are required" },
            { status: 400 }
          );
        }
        const updated = updateObjectiveProgress(
          manager,
          taskId,
          objectiveId,
          progress,
          actorId
        );
        return NextResponse.json({
          success: true,
          objective: updated,
        });
      }

      case "submit_result": {
        const resultInput: CreateResultInput = {
          objectiveId: body.objectiveId,
          title: body.title,
          description: body.description,
          type: body.type,
          contributors: body.contributors || [actorId],
          data: body.data,
        };

        if (!resultInput.objectiveId || !resultInput.title) {
          return NextResponse.json(
            { success: false, error: "objectiveId and title are required" },
            { status: 400 }
          );
        }

        const result = submitResult(manager, taskId, resultInput, actorId);
        return NextResponse.json({
          success: true,
          result,
        });
      }

      case "add_criterion": {
        const { description, type, target } = body;
        if (!description || !type) {
          return NextResponse.json(
            { success: false, error: "description and type are required" },
            { status: 400 }
          );
        }
        const criterion = addCompletionCriterion(
          manager,
          taskId,
          { description, type, target },
          actorId
        );
        return NextResponse.json({
          success: true,
          criterion,
        });
      }

      default:
        return NextResponse.json(
          { success: false, error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error("Failed to update meta-task:", error);

    // Handle transition errors gracefully
    if (error instanceof Error && error.message.includes("Cannot transition")) {
      return NextResponse.json(
        { success: false, error: error.message },
        { status: 400 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to update meta-task",
      },
      { status: 500 }
    );
  }
}
