/**
 * Device-specific API
 *
 * GET /api/compute/devices/[deviceId] - Get device info
 * POST /api/compute/devices/[deviceId]/heartbeat - Send heartbeat
 * GET /api/compute/devices/[deviceId]/task - Get next task assignment
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";
import type { HeartbeatRequest } from "@/lib/compute/types";

interface RouteParams {
  params: {
    deviceId: string;
  };
}

/**
 * GET /api/compute/devices/[deviceId]
 * Get device information
 */
export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { deviceId } = params;
    const orchestrator = getOrchestrator();
    const device = orchestrator.getDevice(deviceId);

    if (!device) {
      return NextResponse.json(
        { success: false, error: "Device not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      device,
    });
  } catch (error) {
    console.error("Get device error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to get device" },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/compute/devices/[deviceId]
 * Update device (heartbeat)
 */
export async function PATCH(request: NextRequest, { params }: RouteParams) {
  try {
    const { deviceId } = params;
    const body = await request.json();

    const orchestrator = getOrchestrator();

    const heartbeatRequest: HeartbeatRequest = {
      deviceId,
      status: body.status || "online",
      taskProgress: body.taskProgress,
    };

    const device = orchestrator.heartbeat(heartbeatRequest);

    if (!device) {
      return NextResponse.json(
        { success: false, error: "Device not found" },
        { status: 404 }
      );
    }

    // Check if there's a task assignment waiting
    const assignment = orchestrator.getNextTask(deviceId);

    return NextResponse.json({
      success: true,
      device: {
        id: device.id,
        status: device.status,
        currentTaskId: device.currentTaskId,
      },
      // Include task assignment if available
      task: assignment?.task || null,
    });
  } catch (error) {
    console.error("Heartbeat error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to process heartbeat" },
      { status: 500 }
    );
  }
}
