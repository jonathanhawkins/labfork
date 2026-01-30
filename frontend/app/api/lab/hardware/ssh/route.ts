import { NextRequest, NextResponse } from "next/server";

/**
 * SSH Hardware API
 *
 * POST /api/lab/hardware/ssh - Test SSH connection and detect remote hardware
 *
 * Actions:
 * - test: Test SSH connection
 * - detect-gpu: Detect GPU on remote machine
 * - system-info: Get full system info from remote
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, host, port = 22, user, keyPath } = body;

    if (!action) {
      return NextResponse.json(
        { success: false, error: "Action is required" },
        { status: 400 }
      );
    }

    if (!host || !user) {
      return NextResponse.json(
        { success: false, error: "Host and user are required" },
        { status: 400 }
      );
    }

    // In a real implementation, this would use a backend SSH service
    // For now, we simulate responses based on known hosts
    const knownGpuHost = process.env.REMOTE_GPU_HOST || '';
    const knownGpuUser = process.env.REMOTE_GPU_USER || 'doc';
    const isKnownHost = knownGpuHost && host === knownGpuHost && user === knownGpuUser;

    switch (action) {
      case "test": {
        // Simulate connection test
        if (isKnownHost) {
          return NextResponse.json({
            success: true,
            latency: 45,
            hostname: "DESKTOP-GPU",
            os: "Ubuntu 22.04 LTS (WSL)",
          });
        }

        // Simulate timeout for unknown hosts in dev
        if (process.env.NODE_ENV === "development") {
          // In dev, return success for any connection attempt
          return NextResponse.json({
            success: true,
            latency: Math.floor(Math.random() * 100) + 20,
            hostname: "remote-server",
            os: "Linux",
          });
        }

        return NextResponse.json({
          success: false,
          error: "Connection timed out",
          errorCode: "timeout",
        });
      }

      case "detect-gpu": {
        if (isKnownHost) {
          return NextResponse.json({
            success: true,
            gpu: {
              name: "NVIDIA GeForce RTX 4090",
              vram: 24,
              cudaVersion: "12.1",
              driverVersion: "535.104.05",
              available: true,
              computeCapability: "8.9",
            },
          });
        }

        if (process.env.NODE_ENV === "development") {
          return NextResponse.json({
            success: true,
            gpu: {
              name: "NVIDIA GPU",
              vram: 16,
              cudaVersion: "11.8",
              available: true,
            },
          });
        }

        return NextResponse.json({
          success: false,
          error: "Could not detect GPU on remote machine",
        });
      }

      case "system-info": {
        if (isKnownHost) {
          return NextResponse.json({
            success: true,
            info: {
              hostname: "DESKTOP-GPU",
              os: "Ubuntu 22.04 LTS (WSL)",
              kernel: "5.15.90.1-microsoft-standard-WSL2",
              gpu: {
                name: "NVIDIA GeForce RTX 4090",
                vram: 24,
                cudaVersion: "12.1",
                driverVersion: "535.104.05",
                available: true,
              },
              ram: 64,
              cpuCores: 16,
              pythonVersion: "3.10.12",
              cudaAvailable: true,
              torchVersion: "2.1.0",
              condaEnvs: ["base", "voice"],
            },
          });
        }

        if (process.env.NODE_ENV === "development") {
          return NextResponse.json({
            success: true,
            info: {
              hostname: "remote-server",
              os: "Linux",
              kernel: "5.15.0",
              ram: 32,
              cpuCores: 8,
              pythonVersion: "3.10",
            },
          });
        }

        return NextResponse.json({
          success: false,
          error: "Could not get system info",
        });
      }

      default:
        return NextResponse.json(
          { success: false, error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "SSH operation failed",
      },
      { status: 500 }
    );
  }
}
