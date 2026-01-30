/**
 * Demo Task Generator API
 *
 * POST /api/compute/demo - Generate demo tasks for testing the compute network
 * GET /api/compute/demo - Get demo system status
 *
 * This endpoint creates real tasks that can be processed by contributor devices.
 */

import { NextRequest, NextResponse } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";
import type { SubmitTaskRequest, TaskType } from "@/lib/compute/types";

/**
 * Demo task templates
 */
const DEMO_TASKS: Array<{
  name: string;
  type: TaskType;
  input: { prompt: string };
  config: { modelId: string; maxTokens: number; temperature: number; minTier?: "crowd" | "standard" | "power" };
  description: string;
}> = [
  {
    name: "Text Summarization",
    type: "full_inference",
    input: {
      prompt: "Summarize the key findings of this research abstract: Machine learning models have shown remarkable progress in natural language understanding. Recent advances in transformer architectures have enabled models to capture long-range dependencies and contextual relationships more effectively. This paper presents a novel approach to improving model efficiency through sparse attention mechanisms.",
    },
    config: {
      modelId: "Qwen2-0.5B",
      maxTokens: 100,
      temperature: 0.7,
      minTier: "crowd",
    },
    description: "Summarize a research abstract",
  },
  {
    name: "Code Explanation",
    type: "full_inference",
    input: {
      prompt: "Explain what this code does:\n\nfunction fibonacci(n) {\n  if (n <= 1) return n;\n  return fibonacci(n - 1) + fibonacci(n - 2);\n}",
    },
    config: {
      modelId: "Phi-3-mini",
      maxTokens: 150,
      temperature: 0.5,
      minTier: "standard",
    },
    description: "Explain a code snippet",
  },
  {
    name: "Draft Token Generation",
    type: "draft_tokens",
    input: {
      prompt: "The future of artificial intelligence in healthcare includes",
    },
    config: {
      modelId: "Qwen2-0.5B",
      maxTokens: 8,
      temperature: 0.8,
      minTier: "crowd",
    },
    description: "Generate draft tokens for speculative decoding",
  },
  {
    name: "Embedding Generation",
    type: "embedding",
    input: {
      prompt: "Distributed computing enables parallel processing across multiple machines to solve complex problems faster.",
    },
    config: {
      modelId: "Qwen2-0.5B",
      maxTokens: 1,
      temperature: 0,
      minTier: "crowd",
    },
    description: "Generate text embeddings",
  },
  {
    name: "Research Question",
    type: "full_inference",
    input: {
      prompt: "What are the main challenges in training large language models on consumer hardware?",
    },
    config: {
      modelId: "Llama-3.2-3B",
      maxTokens: 200,
      temperature: 0.7,
      minTier: "standard",
    },
    description: "Answer a research question",
  },
];

/**
 * POST /api/compute/demo
 * Generate demo tasks
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    const count = Math.min(body.count || 5, 20); // Max 20 tasks at once
    const taskType = body.type as TaskType | undefined;
    const tier = body.tier as "crowd" | "standard" | "power" | undefined;

    const orchestrator = getOrchestrator();
    const submittedTasks: Array<{ id: string; type: string; name: string }> = [];

    for (let i = 0; i < count; i++) {
      // Pick a random task template or filter by type/tier
      let templates = DEMO_TASKS;

      if (taskType) {
        templates = templates.filter((t) => t.type === taskType);
      }

      if (tier) {
        templates = templates.filter((t) => !t.config.minTier || t.config.minTier === tier);
      }

      if (templates.length === 0) {
        templates = DEMO_TASKS; // Fallback to all if no match
      }

      const template = templates[Math.floor(Math.random() * templates.length)];

      const taskRequest: SubmitTaskRequest = {
        type: template.type,
        input: template.input,
        config: template.config,
        priority: Math.floor(Math.random() * 3) + 1, // Priority 1-3
      };

      const task = orchestrator.submitTask(taskRequest, "demo-generator");

      submittedTasks.push({
        id: task.id,
        type: task.type,
        name: template.name,
      });
    }

    return NextResponse.json({
      success: true,
      message: `Generated ${submittedTasks.length} demo tasks`,
      tasks: submittedTasks,
      queueStats: orchestrator.getNetworkStats(),
    });
  } catch (error) {
    console.error("Demo task generation error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to generate demo tasks" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/compute/demo
 * Get demo system status
 */
export async function GET() {
  try {
    const orchestrator = getOrchestrator();
    const stats = orchestrator.getNetworkStats();
    const leaderboard = orchestrator.getLeaderboard(5);

    return NextResponse.json({
      success: true,
      status: {
        networkStats: stats,
        leaderboard,
        availableTaskTypes: DEMO_TASKS.map((t) => ({
          name: t.name,
          type: t.type,
          description: t.description,
          minTier: t.config.minTier || "crowd",
        })),
      },
      instructions: {
        generateTasks: "POST /api/compute/demo with { count: 5, type?: TaskType, tier?: Tier }",
        exampleTypes: ["full_inference", "draft_tokens", "embedding"],
        exampleTiers: ["crowd", "standard", "power"],
      },
    });
  } catch (error) {
    console.error("Demo status error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to get demo status" },
      { status: 500 }
    );
  }
}
