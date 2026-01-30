/**
 * Meta-Agent Status API
 *
 * GET /api/meta-agents/status - Get meta-agent dashboard
 * POST /api/meta-agents/status - Control agents (pause/resume)
 */

import { NextRequest, NextResponse } from "next/server";
import {
  createAgentDashboard,
  generateDashboard,
  getAllAgentStatuses,
  getAgentStatus,
  enableAgent,
  disableAgent,
  updateAgentStatus,
  recordDiscovery,
  logActivity,
  MetaAgentName,
  AgentStatus,
  SignificanceLevel,
} from "@/lib/meta/community";

// Singleton agent dashboard
let agentDashboard = createAgentDashboard();

// Initialize with some activity
initializeDemoActivity();

function initializeDemoActivity() {
  const agents: MetaAgentName[] = [
    "synergy-detector",
    "pattern-recognizer",
    "gap-analyzer",
    "evolution-engine",
    "transfer-agent",
  ];

  // Set some agents as running
  updateAgentStatus(agentDashboard, "synergy-detector", "running", "Scanning for synergies...");
  updateAgentStatus(agentDashboard, "pattern-recognizer", "running", "Analyzing patterns...");
  updateAgentStatus(agentDashboard, "gap-analyzer", "idle");
  updateAgentStatus(agentDashboard, "evolution-engine", "running", "Evolving techniques...");
  updateAgentStatus(agentDashboard, "transfer-agent", "idle");

  // Add some discoveries
  recordDiscovery(
    agentDashboard,
    "synergy-detector",
    "synergy",
    "Found synergy between attention prosody and codec models",
    "high",
    { techniqueIds: ["tech-1", "tech-2"], score: 0.85 }
  );

  recordDiscovery(
    agentDashboard,
    "pattern-recognizer",
    "pattern",
    "Identified recurring transfer learning pattern",
    "medium",
    { patternType: "transfer-learning", frequency: 15 }
  );

  recordDiscovery(
    agentDashboard,
    "evolution-engine",
    "evolution",
    "Evolved attention mechanism with 18% fitness improvement",
    "high",
    { generation: 3, fitnessImprovement: 18 }
  );

  // Add some activity logs
  for (const agent of agents) {
    logActivity(agentDashboard, agent, "initialized", "Agent started successfully", "success", 150);
    logActivity(agentDashboard, agent, "scan", "Completed routine scan", "success", 2500);
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const agentName = searchParams.get("agent") as MetaAgentName | null;
    const detailed = searchParams.get("detailed") === "true";

    if (agentName) {
      const status = getAgentStatus(agentDashboard, agentName);
      if (!status) {
        return NextResponse.json(
          { error: "Agent not found" },
          { status: 404 }
        );
      }
      return NextResponse.json(status);
    }

    if (detailed) {
      const dashboard = generateDashboard(agentDashboard);
      return NextResponse.json(dashboard);
    }

    const agents = getAllAgentStatuses(agentDashboard);
    return NextResponse.json({ agents });
  } catch (error) {
    console.error("Error fetching agent status:", error);
    return NextResponse.json(
      { error: "Failed to fetch agent status" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { agentName, action, reason, status, currentTask } = body as {
      agentName: MetaAgentName;
      action: "enable" | "disable" | "pause" | "resume" | "update";
      reason?: string;
      status?: AgentStatus;
      currentTask?: string;
    };

    if (!agentName || !action) {
      return NextResponse.json(
        { error: "agentName and action are required" },
        { status: 400 }
      );
    }

    let result;

    switch (action) {
      case "enable":
      case "resume":
        result = enableAgent(agentDashboard, agentName);
        break;

      case "disable":
      case "pause":
        result = disableAgent(agentDashboard, agentName, reason);
        break;

      case "update":
        if (!status) {
          return NextResponse.json(
            { error: "status is required for update action" },
            { status: 400 }
          );
        }
        result = updateAgentStatus(agentDashboard, agentName, status, currentTask);
        break;

      default:
        return NextResponse.json(
          { error: "Invalid action" },
          { status: 400 }
        );
    }

    if (!result) {
      return NextResponse.json(
        { error: "Agent not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error controlling agent:", error);
    return NextResponse.json(
      { error: "Failed to control agent" },
      { status: 500 }
    );
  }
}

// Record a discovery (for testing/simulation)
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();
    const { agentName, type, description, significance, metadata } = body as {
      agentName: MetaAgentName;
      type: string;
      description: string;
      significance: SignificanceLevel;
      metadata?: Record<string, unknown>;
    };

    if (!agentName || !type || !description || !significance) {
      return NextResponse.json(
        { error: "agentName, type, description, and significance are required" },
        { status: 400 }
      );
    }

    const discovery = recordDiscovery(
      agentDashboard,
      agentName,
      type,
      description,
      significance,
      metadata
    );

    return NextResponse.json(discovery, { status: 201 });
  } catch (error) {
    console.error("Error recording discovery:", error);
    return NextResponse.json(
      { error: "Failed to record discovery" },
      { status: 500 }
    );
  }
}
