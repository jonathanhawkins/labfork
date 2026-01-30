/**
 * Compute Tasks API
 *
 * POST /api/compute/tasks - Submit a new task
 * GET /api/compute/tasks - List tasks (with filters)
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";
import type { SubmitTaskRequest } from "@/lib/compute/types";

/**
 * POST /api/compute/tasks
 * Submit a new compute task
 */
export async function POST(request: NextRequest) {
  try {
    const body: SubmitTaskRequest = await request.json();

    // Validate required fields
    if (!body.type || !body.input || !body.config) {
      return NextResponse.json(
        { success: false, error: "Missing required fields: type, input, config" },
        { status: 400 }
      );
    }

    if (!body.config.modelId) {
      return NextResponse.json(
        { success: false, error: "Missing config.modelId" },
        { status: 400 }
      );
    }

    // Get submitter ID from auth header (mock for now)
    const submitterId = request.headers.get("x-user-id") || "anonymous";

    const orchestrator = getOrchestrator();

    // Check if user has enough credits (skip for now in MVP)
    // const credits = orchestrator.getUserCredits(submitterId);
    // const cost = calculateTaskCost(body.type, body.config);
    // if (!credits || credits.balance < cost) {
    //   return NextResponse.json(
    //     { success: false, error: "Insufficient credits" },
    //     { status: 402 }
    //   );
    // }

    const task = orchestrator.submitTask(body, submitterId);

    return NextResponse.json({
      success: true,
      task: {
        id: task.id,
        type: task.type,
        status: task.status,
        priority: task.priority,
        reward: task.reward,
        createdAt: task.createdAt,
        assignedDeviceId: task.assignedDeviceId,
      },
    });
  } catch (error) {
    console.error("Submit task error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to submit task" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/compute/tasks
 * List tasks with optional filters
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const submitterId = searchParams.get("submitter");
    const status = searchParams.get("status");

    const orchestrator = getOrchestrator();

    let tasks;
    if (submitterId) {
      tasks = orchestrator.getTasksBySubmitter(submitterId);
    } else {
      // Return all recent tasks
      tasks = orchestrator.getAllTasks(50);
    }

    // Filter by status if specified
    if (status) {
      tasks = tasks.filter((t) => t.status === status);
    }

    return NextResponse.json({
      success: true,
      tasks: tasks.map((t) => ({
        id: t.id,
        type: t.type,
        status: t.status,
        priority: t.priority,
        reward: t.reward,
        createdAt: t.createdAt,
        assignedAt: t.assignedAt,
        completedAt: t.completedAt,
        assignedDeviceId: t.assignedDeviceId,
      })),
      count: tasks.length,
    });
  } catch (error) {
    console.error("List tasks error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to list tasks" },
      { status: 500 }
    );
  }
}
