/**
 * Meta-Tasks API
 *
 * GET /api/collaboration/meta-tasks - List meta-tasks
 * POST /api/collaboration/meta-tasks - Create a new meta-task
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalMetaTaskManager,
  createMetaTask,
  getAllTasks,
  getTasksByStatus,
  getTasksByCategory,
  getTasksByParticipant,
  getOpenTasks,
  CreateMetaTaskInput,
  MetaTaskStatus,
  MetaTaskCategory,
} from "@/lib/meta/collaboration";

/**
 * GET /api/collaboration/meta-tasks
 * List meta-tasks with optional filters
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const manager = getGlobalMetaTaskManager();

    // Get filter params
    const status = searchParams.get("status") as MetaTaskStatus | null;
    const category = searchParams.get("category") as MetaTaskCategory | null;
    const labId = searchParams.get("labId");
    const openOnly = searchParams.get("open") === "true";
    const page = parseInt(searchParams.get("page") || "1", 10);
    const limit = Math.min(parseInt(searchParams.get("limit") || "20", 10), 100);

    let tasks;

    if (openOnly) {
      tasks = getOpenTasks(manager);
    } else if (status) {
      tasks = getTasksByStatus(manager, status);
    } else if (category) {
      tasks = getTasksByCategory(manager, category);
    } else if (labId) {
      tasks = getTasksByParticipant(manager, labId);
    } else {
      tasks = getAllTasks(manager);
    }

    // Sort by creation date (newest first)
    tasks.sort((a, b) =>
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );

    // Paginate
    const total = tasks.length;
    const offset = (page - 1) * limit;
    const paginatedTasks = tasks.slice(offset, offset + limit);

    // Return summary data (not full task details for list view)
    const taskSummaries = paginatedTasks.map((task) => ({
      id: task.id,
      title: task.title,
      description: task.description,
      category: task.category,
      status: task.status,
      leadLab: task.lead.labName,
      participantCount: task.participants.filter((p) => p.status === "active").length,
      objectiveCount: task.objectives.length,
      completedObjectives: task.objectives.filter((o) => o.status === "completed").length,
      visibility: task.visibility,
      tags: task.tags,
      domains: task.domains,
      createdAt: task.createdAt,
      updatedAt: task.updatedAt,
    }));

    return NextResponse.json({
      success: true,
      tasks: taskSummaries,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error("Failed to list meta-tasks:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to list meta-tasks",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/collaboration/meta-tasks
 * Create a new meta-task
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const manager = getGlobalMetaTaskManager();

    // Validate required fields
    if (!body.title || typeof body.title !== "string") {
      return NextResponse.json(
        { success: false, error: "Title is required" },
        { status: 400 }
      );
    }

    if (!body.description || typeof body.description !== "string") {
      return NextResponse.json(
        { success: false, error: "Description is required" },
        { status: 400 }
      );
    }

    if (!body.category) {
      return NextResponse.json(
        { success: false, error: "Category is required" },
        { status: 400 }
      );
    }

    if (!body.leadLabId || !body.leadLabName) {
      return NextResponse.json(
        { success: false, error: "Lead lab information is required" },
        { status: 400 }
      );
    }

    // Validate category
    const validCategories: MetaTaskCategory[] = [
      "exploration",
      "integration",
      "benchmark",
      "dataset",
      "replication",
      "extension",
      "application",
    ];
    if (!validCategories.includes(body.category)) {
      return NextResponse.json(
        { success: false, error: `Invalid category: ${body.category}` },
        { status: 400 }
      );
    }

    // Build input
    const input: CreateMetaTaskInput = {
      title: body.title,
      description: body.description,
      category: body.category,
      leadLabId: body.leadLabId,
      leadLabName: body.leadLabName,
      leadExpertise: body.leadExpertise || [],
      requirements: body.requirements,
      timeline: body.timeline,
      tags: body.tags || [],
      domains: body.domains || [],
      visibility: body.visibility || "public",
    };

    // Create meta-task
    const task = createMetaTask(manager, input);

    return NextResponse.json({
      success: true,
      task: {
        id: task.id,
        title: task.title,
        description: task.description,
        category: task.category,
        status: task.status,
        createdAt: task.createdAt,
      },
    });
  } catch (error) {
    console.error("Failed to create meta-task:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to create meta-task",
      },
      { status: 500 }
    );
  }
}
