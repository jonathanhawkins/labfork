#!/usr/bin/env npx ts-node
/**
 * Seed Firefly Lab Script
 *
 * Run this script to initialize the Firefly Lab with all demo data:
 * - Lab configuration
 * - Research papers
 * - Tasks
 * - Agents
 * - Activities
 * - Results with comments
 * - BOM items
 *
 * Usage:
 *   npx ts-node scripts/seed-firefly.ts
 *   OR
 *   curl -X POST http://localhost:3003/api/labs/firefly
 */

async function seedFireflyLab() {
  const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3003";

  console.log("Seeding Firefly Network Lab #1...");
  console.log(`API: ${baseUrl}/api/labs/firefly`);

  try {
    const response = await fetch(`${baseUrl}/api/labs/firefly`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ force: false }),
    });

    const data = await response.json();

    if (data.success) {
      console.log("\nFirefly Lab seeded successfully!");
      console.log("-----------------------------------");

      if (data.seeded) {
        console.log("Created new lab with:");
        console.log(`  - ${data.counts.papers} research papers`);
        console.log(`  - ${data.counts.tasks} tasks`);
        console.log(`  - ${data.counts.agents} agents`);
        console.log(`  - ${data.counts.activities} activities`);
        console.log(`  - ${data.counts.results} published results`);
        console.log(`  - ${data.counts.bomItems} BOM items`);
      } else {
        console.log("Lab already exists - no changes made.");
        console.log("Use { force: true } to reseed.");
      }

      console.log("\nView the lab at:");
      console.log(`  ${baseUrl}/projects/firefly-network`);
      console.log(`  ${baseUrl}/lab?id=lab_firefly001`);
      console.log(`  ${baseUrl}/explore?domain=firefly-network`);
    } else {
      console.error("Error:", data.error);
      process.exit(1);
    }
  } catch (error) {
    console.error("Failed to seed:", error);
    console.log("\nMake sure the dev server is running:");
    console.log("  npm run dev");
    process.exit(1);
  }
}

// Run if called directly
seedFireflyLab();
