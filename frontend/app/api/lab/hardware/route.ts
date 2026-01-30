import { NextRequest, NextResponse } from "next/server";
import type { GpuInfo } from "@/lib/lab-wizard/types";

/**
 * Hardware detection API
 *
 * GET /api/lab/hardware - Detect local GPU and system info
 * GET /api/lab/hardware?full=true - Include Ollama and full system info
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const full = searchParams.get("full") === "true";

    // In a real implementation, this would call native code or a backend service
    // For now, we return mock data based on environment
    const isDev = process.env.NODE_ENV === "development";
    const hasMockGpu = process.env.MOCK_GPU === "true";

    // Simulated GPU detection
    let gpu: GpuInfo | undefined;

    if (hasMockGpu || isDev) {
      // Return mock GPU for development
      gpu = {
        name: "NVIDIA GeForce RTX 4090",
        vram: 24,
        cudaVersion: "12.1",
        driverVersion: "535.104.05",
        available: true,
        computeCapability: "8.9",
      };
    } else {
      // In production without mock, try to detect actual GPU
      // This would require a native module or backend service
      gpu = {
        name: "No GPU detected",
        vram: 0,
        available: false,
      };
    }

    if (!full) {
      return NextResponse.json({
        success: true,
        gpu,
      });
    }

    // Full system info
    const platform = process.platform;
    const ram = 64; // Would need native module to detect

    // Ollama detection (check if ollama is running)
    let ollamaInstalled = false;
    let ollamaVersion: string | undefined;
    let ollamaModels: string[] = [];

    try {
      const ollamaResponse = await fetch("http://localhost:11434/api/version", {
        signal: AbortSignal.timeout(2000),
      });
      if (ollamaResponse.ok) {
        const versionData = await ollamaResponse.json();
        ollamaInstalled = true;
        ollamaVersion = versionData.version;

        // Get models
        const modelsResponse = await fetch("http://localhost:11434/api/tags", {
          signal: AbortSignal.timeout(2000),
        });
        if (modelsResponse.ok) {
          const modelsData = await modelsResponse.json();
          ollamaModels = modelsData.models?.map((m: { name: string }) => m.name) || [];
        }
      }
    } catch {
      // Ollama not running
    }

    return NextResponse.json({
      success: true,
      gpu,
      ram,
      platform,
      ollamaInstalled,
      ollamaVersion,
      ollamaModels,
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Hardware detection failed",
      },
      { status: 500 }
    );
  }
}
