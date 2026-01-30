/**
 * Task Assignment API
 *
 * POST /api/compute/tasks/assign - Request work assignment for a device
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";
import { TaskRouter } from "@/lib/compute/task-router";
import { getScheduler } from "@/lib/compute/scheduling";
import type { ComputeDevice, ComputeTask } from "@/lib/compute/types";

/**
 * Task assignment request
 */
interface AssignTaskRequest {
  /** Device ID requesting work */
  deviceId: string;
  /** Device capabilities (optional, for matching) */
  capabilities?: {
    /** Available compute (TFLOPS) */
    compute?: number;
    /** Available memory (GB) */
    memory?: number;
    /** Cached model IDs */
    cachedModels?: string[];
  };
  /** Preferred task types (optional) */
  preferredTaskTypes?: string[];
}

/**
 * POST /api/compute/tasks/assign
 * Assign optimal task to requesting device
 */
export async function POST(request: NextRequest) {
  try {
    const body: AssignTaskRequest = await request.json();

    // Validate request
    if (!body.deviceId) {
      return NextResponse.json(
        { success: false, error: "Missing deviceId" },
        { status: 400 }
      );
    }

    const orchestrator = getOrchestrator();
    const router = new TaskRouter();
    const scheduler = getScheduler();

    // Get device
    const device = orchestrator.getDevice(body.deviceId);
    if (!device) {
      return NextResponse.json(
        { success: false, error: "Device not found" },
        { status: 404 }
      );
    }

    // Check device status
    if (device.status !== "online") {
      return NextResponse.json(
        {
          success: false,
          error: `Device not available (status: ${device.status})`,
        },
        { status: 400 }
      );
    }

    // Check if device already has a task
    if (device.currentTaskId) {
      const currentTask = orchestrator.getTask(device.currentTaskId);
      if (currentTask && currentTask.status !== "completed" && currentTask.status !== "failed") {
        return NextResponse.json({
          success: true,
          hasWork: true,
          task: currentTask,
          message: "Device already has assigned task",
        });
      }
    }

    // Get all pending tasks (simplified - in production would use TaskQueue)
    const allTasks = Array.from((orchestrator as any).tasks.values()) as ComputeTask[];
    const pendingTasks = allTasks.filter((t) => t.status === "pending");

    // No work available
    if (pendingTasks.length === 0) {
      return NextResponse.json({
        success: true,
        hasWork: false,
        message: "No tasks available",
        queueStats: {
          pendingTasks: 0,
          totalTasks: allTasks.length,
        },
      });
    }

    // Filter tasks that device can handle based on tier
    const tierOrder: Record<string, number> = { crowd: 0, standard: 1, power: 2 };
    const deviceTierLevel = tierOrder[device.tier] ?? 0;

    const eligibleTasks = pendingTasks.filter((task) => {
      if (!task.config.minTier) return true;
      const taskTierLevel = tierOrder[task.config.minTier] ?? 0;
      return deviceTierLevel >= taskTierLevel;
    });

    if (eligibleTasks.length === 0) {
      return NextResponse.json({
        success: true,
        hasWork: false,
        message: "No eligible tasks for device tier",
        queueStats: {
          pendingTasks: pendingTasks.length,
          totalTasks: allTasks.length,
        },
      });
    }

    // Use fair scheduler to select best task for this device
    // Create mock device list for scheduling fairness calculation
    const onlineDevices = orchestrator.getOnlineDevices();

    // Score all eligible tasks using routing and scheduling
    const taskScores = eligibleTasks.map((task) => {
      // Get router score (task-device compatibility)
      const routerScore = router.findBestDevice(task, [device]) ? 100 : 0;

      // Get scheduler priority (fairness)
      const schedulerPriority = scheduler.calculateDevicePriority(
        device,
        task,
        onlineDevices
      );

      // Combine scores (weighted)
      const totalScore = routerScore * 0.6 + schedulerPriority * 0.4;

      return {
        task,
        score: totalScore,
      };
    });

    // Sort by score (highest first)
    taskScores.sort((a, b) => b.score - a.score);

    // Select best task
    const selectedTask = taskScores[0]?.task;

    if (!selectedTask) {
      return NextResponse.json({
        success: true,
        hasWork: false,
        message: "No suitable task found",
        queueStats: {
          pendingTasks: pendingTasks.length,
          totalTasks: allTasks.length,
        },
      });
    }

    // Assign task to device
    selectedTask.status = "assigned";
    selectedTask.assignedDeviceId = device.id;
    selectedTask.assignedAt = new Date().toISOString();

    device.status = "busy";
    device.currentTaskId = selectedTask.id;

    // Record assignment in scheduler
    scheduler.recordAssignment(device.id, selectedTask.id);

    // Get fairness metrics
    const fairnessMetrics = scheduler.getFairnessMetrics(onlineDevices);

    return NextResponse.json({
      success: true,
      hasWork: true,
      task: {
        id: selectedTask.id,
        type: selectedTask.type,
        input: selectedTask.input,
        config: selectedTask.config,
        priority: selectedTask.priority,
        reward: selectedTask.reward,
        createdAt: selectedTask.createdAt,
        assignedAt: selectedTask.assignedAt,
      },
      queueStats: {
        pendingTasks: pendingTasks.length - 1,
        totalTasks: allTasks.length,
      },
      fairnessMetrics: {
        starvingDevices: fairnessMetrics.starvingDevices,
        avgWaitTime: Math.round(fairnessMetrics.avgWaitTime),
        giniCoefficient: Math.round(fairnessMetrics.giniCoefficient * 100) / 100,
      },
    });
  } catch (error) {
    console.error("Task assignment error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to assign task",
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/compute/tasks/assign
 * Get assignment statistics (for monitoring)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const deviceId = searchParams.get("deviceId");

    const orchestrator = getOrchestrator();
    const scheduler = getScheduler();

    // Get all tasks
    const allTasks = Array.from((orchestrator as any).tasks.values()) as ComputeTask[];
    const pendingTasks = allTasks.filter((t) => t.status === "pending");
    const assignedTasks = allTasks.filter(
      (t) => t.status === "assigned" || t.status === "processing"
    );

    // Get device metrics if device ID provided
    let deviceMetrics = null;
    if (deviceId) {
      deviceMetrics = scheduler.getDeviceMetrics(deviceId);
    }

    // Get fairness metrics
    const onlineDevices = orchestrator.getOnlineDevices();
    const fairnessMetrics = scheduler.getFairnessMetrics(onlineDevices);

    return NextResponse.json({
      success: true,
      stats: {
        pendingTasks: pendingTasks.length,
        assignedTasks: assignedTasks.length,
        totalTasks: allTasks.length,
        onlineDevices: onlineDevices.length,
      },
      fairness: {
        giniCoefficient: Math.round(fairnessMetrics.giniCoefficient * 100) / 100,
        starvingDevices: fairnessMetrics.starvingDevices,
        avgWaitTime: Math.round(fairnessMetrics.avgWaitTime),
        maxWaitTime: Math.round(fairnessMetrics.maxWaitTime),
        taskDistributionStdDev: Math.round(fairnessMetrics.taskDistributionStdDev * 100) / 100,
      },
      deviceMetrics: deviceMetrics
        ? {
            tasksCompleted: deviceMetrics.tasksCompleted,
            creditsEarned: deviceMetrics.creditsEarned,
            waitTime: Math.round(deviceMetrics.waitTime),
            timeSinceLastTask: Math.round(deviceMetrics.timeSinceLastTask),
          }
        : null,
    });
  } catch (error) {
    console.error("Get assignment stats error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to get assignment statistics" },
      { status: 500 }
    );
  }
}
