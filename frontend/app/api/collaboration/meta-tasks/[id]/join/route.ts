/**
 * Join Meta-Task API
 *
 * POST /api/collaboration/meta-tasks/[id]/join - Join a meta-task
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalMetaTaskManager,
  getTask,
  addParticipant,
  MetaTaskId,
  JoinTaskInput,
  ParticipantRole,
  createEmptyParticipantCommitment,
} from "@/lib/meta/collaboration";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * POST /api/collaboration/meta-tasks/[id]/join
 * Join a meta-task as a participant
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
    if (!body.labId || !body.labName) {
      return NextResponse.json(
        { success: false, error: "labId and labName are required" },
        { status: 400 }
      );
    }

    // Check if task is open for joining
    const openStatuses = ["proposed", "recruiting"];
    if (!openStatuses.includes(task.status)) {
      return NextResponse.json(
        {
          success: false,
          error: `Cannot join task in ${task.status} status`,
        },
        { status: 400 }
      );
    }

    // Validate role
    const validRoles: ParticipantRole[] = [
      "contributor",
      "advisor",
      "reviewer",
      "observer",
    ];
    const role = (body.role as ParticipantRole) || "contributor";
    if (!validRoles.includes(role)) {
      return NextResponse.json(
        { success: false, error: `Invalid role: ${role}. Cannot join as lead or co-lead.` },
        { status: 400 }
      );
    }

    // Build commitment
    const commitment = body.commitment || createEmptyParticipantCommitment();

    const joinInput: JoinTaskInput = {
      labId: body.labId,
      labName: body.labName,
      expertise: body.expertise || [],
      role,
      commitment: {
        hoursPerWeek: commitment.hoursPerWeek || 5,
        durationWeeks: commitment.durationWeeks || 4,
        resources: commitment.resources || [],
        responsibilities: commitment.responsibilities || [],
      },
    };

    const participant = addParticipant(manager, taskId, joinInput);

    if (!participant) {
      return NextResponse.json(
        { success: false, error: "Failed to join task" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      participant: {
        id: participant.id,
        labId: participant.labId,
        labName: participant.labName,
        role: participant.role,
        status: participant.status,
        joinedAt: participant.joinedAt,
      },
      message: `Successfully joined "${task.title}" as ${role}`,
    });
  } catch (error) {
    console.error("Failed to join meta-task:", error);

    // Handle specific errors
    if (error instanceof Error) {
      if (error.message.includes("already participating")) {
        return NextResponse.json(
          { success: false, error: "Lab is already participating in this task" },
          { status: 409 }
        );
      }
      if (error.message.includes("Maximum participants")) {
        return NextResponse.json(
          { success: false, error: "Task has reached maximum participants" },
          { status: 400 }
        );
      }
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to join meta-task",
      },
      { status: 500 }
    );
  }
}
