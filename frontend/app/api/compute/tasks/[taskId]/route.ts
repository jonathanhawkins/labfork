/**
 * Task-specific API
 *
 * GET /api/compute/tasks/[taskId] - Get task info
 * POST /api/compute/tasks/[taskId]/complete - Complete a task
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";
import type { CompleteTaskRequest } from "@/lib/compute/types";

interface RouteParams {
  params: {
    taskId: string;
  };
}

/**
 * GET /api/compute/tasks/[taskId]
 * Get task information
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { taskId } = params;
    const orchestrator = getOrchestrator();
    const task = orchestrator.getTask(taskId);

    if (!task) {
      return NextResponse.json(
        { success: false, error: "Task not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      task: {
        id: task.id,
        type: task.type,
        status: task.status,
        priority: task.priority,
        reward: task.reward,
        input: task.input,
        config: task.config,
        createdAt: task.createdAt,
        assignedAt: task.assignedAt,
        completedAt: task.completedAt,
        assignedDeviceId: task.assignedDeviceId,
      },
    });
  } catch (error) {
    console.error("Get task error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to get task" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/compute/tasks/[taskId]
 * Complete a task (expects body with deviceId, success, result/error)
 */
export async function POST(request: NextRequest, { params }: RouteParams) {
  try {
    const { taskId } = params;
    const body = await request.json();

    // Validate required fields
    if (!body.deviceId) {
      return NextResponse.json(
        { success: false, error: "Missing deviceId" },
        { status: 400 }
      );
    }

    if (typeof body.success !== "boolean") {
      return NextResponse.json(
        { success: false, error: "Missing success status" },
        { status: 400 }
      );
    }

    const orchestrator = getOrchestrator();

    const completeRequest: CompleteTaskRequest = {
      taskId,
      deviceId: body.deviceId,
      success: body.success,
      result: body.result,
      error: body.error,
    };

    const result = orchestrator.completeTask(completeRequest);

    if (!result.success) {
      return NextResponse.json(
        { success: false, error: "Failed to complete task" },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      creditsAwarded: result.credits || 0,
    });
  } catch (error) {
    console.error("Complete task error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to complete task" },
      { status: 500 }
    );
  }
}
