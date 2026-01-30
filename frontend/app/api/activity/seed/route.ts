/**
 * Activity Seed API
 *
 * POST /api/activity/seed - Seed the activity feed with demo data
 * GET /api/activity/seed - Check if seed data exists
 */

import { NextResponse } from "next/server";
import {
  createActivity,
  getActivityFeed,
} from "@/lib/social/activity";

// Demo users
const DEMO_USERS = [
  { id: "user_demo1", username: "spark_research", displayName: "Spark Research", avatar: "" },
  { id: "user_demo2", username: "voice_pioneer", displayName: "Voice Pioneer", avatar: "" },
  { id: "user_demo3", username: "firefly_dev", displayName: "Firefly Developer", avatar: "" },
  { id: "user_demo4", username: "ai_researcher", displayName: "AI Researcher", avatar: "" },
  { id: "user_demo5", username: "open_hardware", displayName: "Open Hardware", avatar: "" },
];

// Demo labs
const DEMO_LABS = [
  { id: "lab_firefly", name: "Firefly Network", slug: "firefly-network" },
  { id: "lab_voice", name: "Voice Clone Lab", slug: "voice-clone" },
  { id: "lab_quant", name: "Quant Trading Lab", slug: "quant-trading" },
];

/**
 * Generate demo activities
 */
async function generateSeedActivities() {
  const activities = [];
  const now = Date.now();

  // Result created activities
  activities.push({
    type: "result_created" as const,
    actor: DEMO_USERS[0],
    target: { type: "result" as const, id: "result_1", title: "MPPT Algorithm v2" },
    context: { metadata: { resultType: "model" } },
    labId: DEMO_LABS[0].id,
    createdAt: new Date(now - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
  });

  activities.push({
    type: "result_published" as const,
    actor: DEMO_USERS[1],
    target: { type: "result" as const, id: "result_2", title: "Voice Emotion Model" },
    labId: DEMO_LABS[1].id,
    createdAt: new Date(now - 4 * 60 * 60 * 1000).toISOString(), // 4 hours ago
  });

  // Task activities
  activities.push({
    type: "task_created" as const,
    actor: DEMO_USERS[2],
    target: { type: "task" as const, id: "task_1", title: "Implement Thread mesh protocol" },
    labId: DEMO_LABS[0].id,
    createdAt: new Date(now - 6 * 60 * 60 * 1000).toISOString(), // 6 hours ago
  });

  activities.push({
    type: "task_completed" as const,
    actor: DEMO_USERS[3],
    target: { type: "task" as const, id: "task_2", title: "Train emotion recognition model" },
    labId: DEMO_LABS[1].id,
    createdAt: new Date(now - 8 * 60 * 60 * 1000).toISOString(), // 8 hours ago
  });

  // Paper activities
  activities.push({
    type: "paper_added" as const,
    actor: DEMO_USERS[4],
    target: { type: "paper" as const, id: "paper_1", title: "Efficient MPPT for Low-Power Applications" },
    labId: DEMO_LABS[0].id,
    createdAt: new Date(now - 12 * 60 * 60 * 1000).toISOString(), // 12 hours ago
  });

  activities.push({
    type: "paper_added" as const,
    actor: DEMO_USERS[0],
    target: { type: "paper" as const, id: "paper_2", title: "Neural Codec Language Models for Speech" },
    labId: DEMO_LABS[1].id,
    createdAt: new Date(now - 18 * 60 * 60 * 1000).toISOString(), // 18 hours ago
  });

  // Social activities
  activities.push({
    type: "comment_added" as const,
    actor: DEMO_USERS[1],
    target: { type: "result" as const, id: "result_1", title: "MPPT Algorithm v2" },
    context: { snippet: "Great progress! The efficiency improvements look promising." },
    labId: DEMO_LABS[0].id,
    createdAt: new Date(now - 1 * 60 * 60 * 1000).toISOString(), // 1 hour ago
  });

  activities.push({
    type: "lab_starred" as const,
    actor: DEMO_USERS[2],
    target: { type: "lab" as const, id: DEMO_LABS[1].id, title: DEMO_LABS[1].name },
    createdAt: new Date(now - 30 * 60 * 1000).toISOString(), // 30 min ago
  });

  activities.push({
    type: "lab_forked" as const,
    actor: DEMO_USERS[3],
    target: { type: "lab" as const, id: DEMO_LABS[0].id, title: DEMO_LABS[0].name },
    createdAt: new Date(now - 45 * 60 * 1000).toISOString(), // 45 min ago
  });

  // Agent activities
  activities.push({
    type: "agent_spawned" as const,
    actor: { id: "system", username: "system", displayName: "System", avatar: "" },
    target: { type: "agent" as const, id: "agent_spark", title: "Spark (Synergy Detector)" },
    labId: DEMO_LABS[0].id,
    createdAt: new Date(now - 15 * 60 * 1000).toISOString(), // 15 min ago
  });

  activities.push({
    type: "agent_completed" as const,
    actor: { id: "system", username: "system", displayName: "System", avatar: "" },
    target: { type: "agent" as const, id: "agent_mesh", title: "Mesh (Pattern Recognizer)" },
    context: { summary: "Identified 3 optimization patterns in MPPT implementation" },
    labId: DEMO_LABS[0].id,
    createdAt: new Date(now - 10 * 60 * 1000).toISOString(), // 10 min ago
  });

  return activities;
}

/**
 * POST /api/activity/seed
 * Create seed activities
 */
export async function POST() {
  try {
    // Check if we already have activities
    const existing = await getActivityFeed({ limit: 1 });
    if (existing.activities.length > 0) {
      return NextResponse.json({
        success: true,
        message: "Seed data already exists",
        seeded: false,
        count: 0,
      });
    }

    // Generate and create seed activities
    const seedData = await generateSeedActivities();
    let created = 0;

    for (const activityData of seedData) {
      try {
        await createActivity(activityData);
        created++;
      } catch (err) {
        console.error("Failed to create activity:", err);
      }
    }

    return NextResponse.json({
      success: true,
      message: `Seeded ${created} activities`,
      seeded: true,
      count: created,
    });
  } catch (error) {
    console.error("Seed error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to seed activities" },
      { status: 500 }
    );
  }
}

/**
 * GET /api/activity/seed
 * Check seed status
 */
export async function GET() {
  try {
    const existing = await getActivityFeed({ limit: 1 });
    return NextResponse.json({
      hasData: existing.activities.length > 0,
      totalActivities: existing.total,
    });
  } catch {
    return NextResponse.json({
      hasData: false,
      totalActivities: 0,
    });
  }
}
