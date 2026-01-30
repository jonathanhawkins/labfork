/**
 * Compute Devices API
 *
 * POST /api/compute/devices - Register a new device
 * GET /api/compute/devices - List online devices
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";
import type { RegisterDeviceRequest } from "@/lib/compute/types";
import { initializeContributorProfile } from "@/lib/supabase/contributors";

/**
 * POST /api/compute/devices
 * Register a new compute device
 */
export async function POST(request: NextRequest) {
  try {
    const body: RegisterDeviceRequest = await request.json();

    // Validate required fields
    if (!body.name || !body.capabilities) {
      return NextResponse.json(
        { success: false, error: "Missing required fields: name, capabilities" },
        { status: 400 }
      );
    }

    if (typeof body.capabilities.compute !== "number" || body.capabilities.compute <= 0) {
      return NextResponse.json(
        { success: false, error: "Invalid compute capability" },
        { status: 400 }
      );
    }

    const orchestrator = getOrchestrator();

    // Get user ID from auth header if available (mock for now)
    const userId = request.headers.get("x-user-id") || undefined;

    const device = orchestrator.registerDevice(body, userId);

    // Initialize contributor profile if user is authenticated
    if (userId) {
      await initializeContributorProfile(userId);
    }

    return NextResponse.json({
      success: true,
      device,
      message: `Device registered as ${device.tier} tier`,
    });
  } catch (error) {
    console.error("Device registration error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to register device" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/compute/devices
 * List devices for visualization (includes recently offline)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const onlineOnly = searchParams.get("onlineOnly") === "true";

    const orchestrator = getOrchestrator();
    // Use visualization method to include recently offline devices
    // This prevents devices from vanishing immediately from the 3D view
    const devices = onlineOnly
      ? orchestrator.getOnlineDevices()
      : orchestrator.getAllDevicesForVisualization();

    return NextResponse.json({
      success: true,
      devices: devices.map((d) => ({
        id: d.id,
        name: d.name,
        tier: d.tier,
        status: d.status,
        compute: d.capabilities.compute,
        platform: d.capabilities.platform,
        stats: d.stats,
      })),
      count: devices.length,
    });
  } catch (error) {
    console.error("List devices error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to list devices" },
      { status: 500 }
    );
  }
}
