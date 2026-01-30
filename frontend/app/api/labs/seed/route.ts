/**
 * Lab Seed API
 *
 * POST /api/labs/seed - Seed demo labs
 * GET /api/labs/seed - Check if seed data exists
 */

import { NextResponse } from "next/server";
import { createLab, listLabs } from "@/lib/labs/repository";
import type { LabOwner, CreateLabInput } from "@/lib/labs/types";

// Demo users (matching activity seed)
const DEMO_OWNERS: LabOwner[] = [
  { id: "user_demo1", username: "spark_research", displayName: "Spark Research", avatar: "" },
  { id: "user_demo2", username: "voice_pioneer", displayName: "Voice Pioneer", avatar: "" },
  { id: "user_demo3", username: "firefly_dev", displayName: "Firefly Developer", avatar: "" },
  { id: "user_demo4", username: "ai_researcher", displayName: "AI Researcher", avatar: "" },
  { id: "user_demo5", username: "open_hardware", displayName: "Open Hardware", avatar: "" },
];

// Demo labs data
const DEMO_LABS: { input: CreateLabInput; owner: LabOwner; stats: { stars: number; forks: number; tasks: number; papers: number } }[] = [
  // Firefly Network labs
  {
    input: {
      name: "Firefly Core",
      slug: "firefly-core",
      description: "Core MPPT algorithms and hardware integration for the Firefly mesh network",
      domainSlug: "firefly-network",
      visibility: "public",
      tags: ["mppt", "solar", "mesh", "hardware"],
      primaryColor: "#f59e0b",
    },
    owner: DEMO_OWNERS[0],
    stats: { stars: 128, forks: 34, tasks: 12, papers: 8 },
  },
  {
    input: {
      name: "Thread Protocol Implementation",
      slug: "thread-protocol",
      description: "Thread mesh networking protocol for IoT devices in the Firefly network",
      domainSlug: "firefly-network",
      visibility: "public",
      tags: ["thread", "mesh", "iot", "protocol"],
      primaryColor: "#f59e0b",
    },
    owner: DEMO_OWNERS[2],
    stats: { stars: 67, forks: 18, tasks: 8, papers: 4 },
  },

  // Voice Clone labs
  {
    input: {
      name: "Emotion TTS Research",
      slug: "emotion-tts",
      description: "Prosody and emotion control in text-to-speech synthesis using neural networks",
      domainSlug: "voice-clone",
      visibility: "public",
      tags: ["tts", "prosody", "emotion", "neural"],
      primaryColor: "#3b82f6",
    },
    owner: DEMO_OWNERS[1],
    stats: { stars: 89, forks: 27, tasks: 15, papers: 12 },
  },
  {
    input: {
      name: "Zero-Shot Voice Cloning",
      slug: "zero-shot-clone",
      description: "Speaker adaptation with minimal reference audio using VALL-E and Pocket TTS",
      domainSlug: "voice-clone",
      visibility: "public",
      tags: ["zero-shot", "vall-e", "speaker-adaptation"],
      primaryColor: "#3b82f6",
    },
    owner: DEMO_OWNERS[3],
    stats: { stars: 156, forks: 42, tasks: 9, papers: 7 },
  },

  // Quant Trading labs
  {
    input: {
      name: "Transformer Signals",
      slug: "transformer-signals",
      description: "Momentum signal generation using transformer architectures for algorithmic trading",
      domainSlug: "quant-trading",
      visibility: "public",
      tags: ["transformer", "momentum", "signals", "ml"],
      primaryColor: "#22c55e",
    },
    owner: DEMO_OWNERS[3],
    stats: { stars: 203, forks: 56, tasks: 18, papers: 15 },
  },
  {
    input: {
      name: "Portfolio Optimization",
      slug: "portfolio-opt",
      description: "Deep reinforcement learning for dynamic portfolio rebalancing",
      domainSlug: "quant-trading",
      visibility: "public",
      tags: ["rl", "portfolio", "optimization"],
      primaryColor: "#22c55e",
    },
    owner: DEMO_OWNERS[4],
    stats: { stars: 78, forks: 21, tasks: 6, papers: 9 },
  },

  // Robotics lab
  {
    input: {
      name: "Sim2Real Transfer",
      slug: "sim2real",
      description: "Domain randomization and transfer learning for robotic manipulation",
      domainSlug: "robotics",
      visibility: "public",
      tags: ["sim2real", "manipulation", "transfer-learning"],
      primaryColor: "#a855f7",
    },
    owner: DEMO_OWNERS[4],
    stats: { stars: 112, forks: 31, tasks: 11, papers: 14 },
  },

  // Biotech lab
  {
    input: {
      name: "Protein Folding Explorer",
      slug: "protein-folding",
      description: "AlphaFold-based analysis and prediction for novel protein structures",
      domainSlug: "biotech",
      visibility: "public",
      tags: ["alphafold", "protein", "structure"],
      primaryColor: "#ec4899",
    },
    owner: DEMO_OWNERS[0],
    stats: { stars: 234, forks: 67, tasks: 22, papers: 28 },
  },
];

/**
 * POST /api/labs/seed
 * Create seed labs
 */
export async function POST() {
  try {
    // Check if we already have labs
    const existing = await listLabs({ limit: 1 });
    if (existing.labs.length > 0) {
      return NextResponse.json({
        success: true,
        message: "Seed data already exists",
        seeded: false,
        count: 0,
      });
    }

    let created = 0;

    for (const { input, owner, stats } of DEMO_LABS) {
      try {
        const lab = await createLab(input, owner);

        // Update stats manually (createLab uses defaults)
        // We'll update via the store directly since updateLabStats needs the lab to exist
        const { updateLabStats } = await import("@/lib/labs/repository");
        await updateLabStats(lab.id, {
          stars: stats.stars,
          forks: stats.forks,
          tasks: stats.tasks,
          papers: stats.papers,
          experiments: Math.floor(stats.tasks * 0.7),
          viewers: Math.floor(Math.random() * 5),
        });

        created++;
      } catch (err) {
        console.error("Failed to create lab:", input.slug, err);
      }
    }

    return NextResponse.json({
      success: true,
      message: `Seeded ${created} labs`,
      seeded: true,
      count: created,
    });
  } catch (error) {
    console.error("Seed error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to seed labs" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/labs/seed
 * Check seed status
 */
export async function GET() {
  try {
    const existing = await listLabs({ limit: 1 });
    const total = existing.total;

    return NextResponse.json({
      hasData: total > 0,
      totalLabs: total,
    });
  } catch {
    return NextResponse.json({
      hasData: false,
      totalLabs: 0,
    });
  }
}
