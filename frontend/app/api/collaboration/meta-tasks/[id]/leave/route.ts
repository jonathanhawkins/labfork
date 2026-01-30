/**
 * Leave Meta-Task API
 *
 * POST /api/collaboration/meta-tasks/[id]/leave - Leave a meta-task
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalMetaTaskManager,
  getTask,
  removeParticipant,
  MetaTaskId,
  ParticipantId,
} from "@/lib/meta/collaboration";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * POST /api/collaboration/meta-tasks/[id]/leave
 * Leave a meta-task
 */
export async function POST(
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

    // Validate required fields
    if (!body.participantId) {
      return NextResponse.json(
        { success: false, error: "participantId is required" },
        { status: 400 }
      );
    }

    const participantId = body.participantId as ParticipantId;
    const reason = body.reason as string | undefined;

    // Find the participant
    const participant = task.participants.find((p) => p.id === participantId);
    if (!participant) {
      return NextResponse.json(
        { success: false, error: "Participant not found in this task" },
        { status: 404 }
      );
    }

    // Check if already withdrawn
    if (participant.status === "withdrawn") {
      return NextResponse.json(
        { success: false, error: "Participant has already left the task" },
        { status: 400 }
      );
    }

    // Remove participant
    const removed = removeParticipant(manager, taskId, participantId, reason);

    if (!removed) {
      return NextResponse.json(
        { success: false, error: "Failed to leave task" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: `Successfully left "${task.title}"`,
      unassignedObjectives: participant.assignedObjectives,
    });
  } catch (error) {
    console.error("Failed to leave meta-task:", error);

    // Handle lead cannot leave error
    if (error instanceof Error && error.message.includes("Cannot remove task lead")) {
      return NextResponse.json(
        {
          success: false,
          error: "Task lead cannot leave. Please transfer leadership first.",
        },
        { status: 400 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to leave meta-task",
      },
      { status: 500 }
    );
  }
}
