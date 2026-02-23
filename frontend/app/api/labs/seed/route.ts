/**
 * Lab Seed API
 *
 * POST /api/labs/seed - Seed labs and sync to Workers
 * GET /api/labs/seed - Check if seed data exists
 */

import { NextResponse } from "next/server";
import { createLab, updateLab, listLabs } from "@/lib/labs/repository";
import type { LabOwner, CreateLabInput, Lab } from "@/lib/labs/types";

// Workers API URL (uses env var or defaults to production)
const WORKERS_API_URL = process.env.WORKERS_API_URL || "https://labfork-agents.workers.dev";

/**
 * Sync labs to Workers compute network for task distribution
 */
async function syncLabsToWorkers(labs: Lab[]): Promise<{ synced: number; failed: number }> {
  try {
    const response = await fetch(`${WORKERS_API_URL}/api/projects/sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        labs: labs.map(lab => ({
          id: lab.id,
          slug: lab.slug,
          name: lab.name,
          description: lab.description,
          domainSlug: lab.domainSlug,
          domainName: lab.domainName,
          tags: lab.tags,
          status: lab.status,
        })),
      }),
    });

    if (!response.ok) {
      console.error("[Seed] Workers sync failed:", response.status, await response.text());
      return { synced: 0, failed: labs.length };
    }

    const result = await response.json();
    return {
      synced: (result.created || 0) + (result.updated || 0),
      failed: result.failed || 0,
    };
  } catch (error) {
    console.error("[Seed] Workers sync error:", error);
    return { synced: 0, failed: labs.length };
  }
}

// Community owner for all seed labs
const COMMUNITY_OWNER: LabOwner = {
  id: "user_community",
  username: "labfork",
  displayName: "LabFork",
  avatar: "",
};

// Seed labs data
const SEED_LABS: { input: CreateLabInput; status: "active" | "idea" }[] = [
  // Active labs (real research)
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
    status: "active",
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
    status: "active",
  },
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
    status: "active",
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
    status: "active",
  },
  {
    input: {
      name: "Atmospheric Water Harvester",
      slug: "water-harvester",
      description: "Solar-powered water extraction from air using biomimicry, cheap sorbents, and 3D-printable components",
      domainSlug: "sustainability",
      visibility: "public",
      tags: ["water", "solar", "biomimicry", "3d-printing", "mof", "humanitarian"],
      primaryColor: "#06b6d4",
    },
    status: "active",
  },

  // Idea labs (open invitations)
  {
    input: {
      name: "Transformer Signals",
      slug: "transformer-signals",
      description: "An open research idea: momentum signal generation using transformer architectures. Fork this lab to start building.",
      domainSlug: "quant-trading",
      visibility: "public",
      tags: ["transformer", "momentum", "signals", "ml"],
      primaryColor: "#22c55e",
    },
    status: "idea",
  },
  {
    input: {
      name: "Portfolio Optimization",
      slug: "portfolio-opt",
      description: "An open research idea: deep reinforcement learning for dynamic portfolio rebalancing. Fork this lab to start building.",
      domainSlug: "quant-trading",
      visibility: "public",
      tags: ["rl", "portfolio", "optimization"],
      primaryColor: "#22c55e",
    },
    status: "idea",
  },
  {
    input: {
      name: "Sim2Real Transfer",
      slug: "sim2real",
      description: "An open research idea: domain randomization and transfer learning for robotic manipulation. Fork this lab to start building.",
      domainSlug: "robotics",
      visibility: "public",
      tags: ["sim2real", "manipulation", "transfer-learning"],
      primaryColor: "#a855f7",
    },
    status: "idea",
  },
  {
    input: {
      name: "Protein Folding Explorer",
      slug: "protein-folding",
      description: "An open research idea: AlphaFold-based analysis and prediction for novel protein structures. Fork this lab to start building.",
      domainSlug: "biotech",
      visibility: "public",
      tags: ["alphafold", "protein", "structure"],
      primaryColor: "#ec4899",
    },
    status: "idea",
  },
];

/**
 * POST /api/labs/seed
 * Create seed labs
 */
export async function POST() {
  try {
    // Get existing labs to check which ones we need to create
    const existing = await listLabs({ limit: 100 });
    const existingSlugs = new Set(existing.labs.map(lab => lab.slug));

    let created = 0;

    for (const { input, status } of SEED_LABS) {
      // Skip if this lab already exists
      if (existingSlugs.has(input.slug)) {
        continue;
      }

      try {
        const lab = await createLab(input, COMMUNITY_OWNER);

        // Set status to "idea" for idea labs (createLab defaults to "active")
        if (status === "idea") {
          await updateLab(lab.id, { status: "idea" });
        }

        created++;
      } catch (err) {
        console.error("Failed to create lab:", input.slug, err);
      }
    }

    // Now sync ALL labs to Workers (including existing ones)
    const allLabs = await listLabs({ limit: 100 });
    const workerSync = await syncLabsToWorkers(allLabs.labs);

    return NextResponse.json({
      success: true,
      message: created > 0 ? `Seeded ${created} new labs` : "All labs already exist",
      seeded: created > 0,
      count: created,
      workersSync: {
        synced: workerSync.synced,
        failed: workerSync.failed,
        total: allLabs.labs.length,
      },
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
