/**
 * Health Check API Endpoint
 *
 * GET /api/health - Returns system health status
 */

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  version: string;
  uptime: number;
  checks: {
    frontend: ServiceCheck;
    backend?: ServiceCheck;
    ollama?: ServiceCheck;
    database?: ServiceCheck;
  };
}

interface ServiceCheck {
  status: "up" | "down" | "unknown";
  latency?: number;
  error?: string;
}

const startTime = Date.now();

async function checkService(
  url: string,
  timeout = 5000
): Promise<ServiceCheck> {
  const startCheck = Date.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(url, {
      signal: controller.signal,
      cache: "no-store",
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      return {
        status: "up",
        latency: Date.now() - startCheck,
      };
    }

    return {
      status: "down",
      latency: Date.now() - startCheck,
      error: `HTTP ${response.status}`,
    };
  } catch (error) {
    return {
      status: "down",
      latency: Date.now() - startCheck,
      error: error instanceof Error ? error.message : "Connection failed",
    };
  }
}

export async function GET() {
  const timestamp = new Date().toISOString();
  const uptime = Math.floor((Date.now() - startTime) / 1000);

  const checks: HealthStatus["checks"] = {
    frontend: { status: "up", latency: 0 },
  };

  // Check backend
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003";
  checks.backend = await checkService(`${backendUrl}/health`);

  // Check Ollama
  const ollamaUrl = process.env.NEXT_PUBLIC_OLLAMA_URL || "http://localhost:11434";
  checks.ollama = await checkService(`${ollamaUrl}/api/tags`);

  // Determine overall status
  let status: HealthStatus["status"] = "healthy";

  if (checks.backend?.status === "down" || checks.ollama?.status === "down") {
    status = "degraded";
  }

  if (checks.backend?.status === "down" && checks.ollama?.status === "down") {
    status = "unhealthy";
  }

  const health: HealthStatus = {
    status,
    timestamp,
    version: process.env.npm_package_version || "1.0.0",
    uptime,
    checks,
  };

  const statusCode = status === "healthy" ? 200 : status === "degraded" ? 200 : 503;

  return NextResponse.json(health, { status: statusCode });
}
