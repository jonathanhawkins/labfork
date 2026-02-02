/**
 * Compute Devices API
 *
 * POST /api/compute/devices - Register a new device (proxies to Workers)
 * GET /api/compute/devices - List devices (proxies to Workers)
 */

import { NextRequest, NextResponse } from "next/server";

const WORKERS_API = "https://labfork-agents.jonathan-hawkins.workers.dev/api/compute";

/**
 * POST /api/compute/devices
 * Register a new compute device
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${WORKERS_API}/devices`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Device registration error:", error);
    return NextResponse.json(
      { error: "Failed to register device" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/compute/devices
 * List devices
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const activeOnly = searchParams.get("activeOnly") || "false";

    const response = await fetch(`${WORKERS_API}/devices?activeOnly=${activeOnly}`, {
      headers: {
        "Content-Type": "application/json",
      },
      cache: "no-store",
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("List devices error:", error);
    return NextResponse.json(
      { error: "Failed to list devices" },
      { status: 500 }
    );
  }
}
