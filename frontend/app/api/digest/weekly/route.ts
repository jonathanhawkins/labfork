/**
 * Weekly Digest API
 *
 * GET /api/digest/weekly - Get the latest weekly digest
 * POST /api/digest/weekly - Generate a new digest
 */

import { NextRequest, NextResponse } from "next/server";
import {
  createDigestGenerator,
  generateDigest,
  getLatestDigest,
  publishDigest,
  DigestSourceData,
} from "@/lib/meta/community";

// Singleton digest generator
let digestGenerator = createDigestGenerator();

export async function GET() {
  try {
    const latestDigest = getLatestDigest(digestGenerator);

    if (!latestDigest) {
      return NextResponse.json(
        { error: "No digest available" },
        { status: 404 }
      );
    }

    return NextResponse.json(latestDigest);
  } catch (error) {
    console.error("Error fetching weekly digest:", error);
    return NextResponse.json(
      { error: "Failed to fetch weekly digest" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { sourceData, publish = false } = body as {
      sourceData?: DigestSourceData;
      publish?: boolean;
    };

    // Use provided source data or generate mock data
    const data: DigestSourceData = sourceData || generateMockSourceData();

    // Generate the digest
    const digest = generateDigest(digestGenerator, data);

    // Optionally publish immediately
    if (publish) {
      publishDigest(digestGenerator, digest.id);
    }

    return NextResponse.json(digest, { status: 201 });
  } catch (error) {
    console.error("Error generating weekly digest:", error);
    return NextResponse.json(
      { error: "Failed to generate weekly digest" },
      { status: 500 }
    );
  }
}

function generateMockSourceData(): DigestSourceData {
  const now = new Date();
  const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

  return {
    discoveries: [
      {
        id: "disc-1",
        type: "breakthrough",
        title: "Novel Prosody Transfer Method",
        description:
          "A new approach combining attention mechanisms with prosody features",
        significance: "high",
        labId: "lab-1",
        labName: "Voice Research Lab",
        techniqueIds: ["tech-1", "tech-2"],
        metrics: {
          improvementPercent: 35,
          affectedDomains: ["TTS", "Voice Cloning"],
          potentialApplications: 5,
        },
        timestamp: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    techniques: [
      {
        id: "tech-1",
        name: "Attention-Based Prosody",
        category: "prosody",
        usageCount: 45,
        previousUsageCount: 30,
        adoptedByLabs: ["lab-1", "lab-2", "lab-3"],
        domains: ["TTS", "Voice Cloning"],
        createdAt: weekAgo.toISOString(),
      },
      {
        id: "tech-2",
        name: "Neural Codec Language Model",
        category: "codec",
        usageCount: 60,
        previousUsageCount: 40,
        adoptedByLabs: ["lab-1", "lab-4"],
        domains: ["Speech Synthesis"],
        createdAt: weekAgo.toISOString(),
      },
    ],
    synergies: [
      {
        id: "syn-1",
        techniques: ["tech-1", "tech-2"],
        score: 0.85,
        discoveredBy: "Synergy Detector",
        labId: "lab-1",
        description: "Combining attention prosody with codec models",
        impact: "high",
        timestamp: new Date(now.getTime() - 1 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    gaps: [
      {
        id: "gap-1",
        title: "Multi-speaker prosody adaptation",
        type: "missing-technique",
        difficulty: "advanced",
        impact: "high",
        domain: "Voice Cloning",
        status: "open",
      },
    ],
    evolutions: [
      {
        id: "evo-1",
        techniqueId: "tech-1",
        techniqueName: "Attention-Based Prosody v2",
        generation: 3,
        fitnessScore: 0.92,
        previousFitness: 0.78,
        parentTechniques: ["tech-base-1", "tech-base-2"],
        newCapabilities: ["Better emotion handling", "Faster inference"],
        timestamp: new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000).toISOString(),
      },
    ],
    collaborations: [
      {
        id: "collab-1",
        title: "Joint Prosody Research Initiative",
        status: "active",
        participantCount: 4,
        recentProgress: "Completed initial experiments",
        completedObjectives: 2,
        totalObjectives: 5,
      },
    ],
    labs: [
      {
        id: "lab-1",
        name: "Voice Research Lab",
        discoveries: 5,
        contributions: 12,
        collaborations: 3,
        isActive: true,
      },
      {
        id: "lab-2",
        name: "Audio ML Team",
        discoveries: 3,
        contributions: 8,
        collaborations: 2,
        isActive: true,
      },
    ],
    papers: [
      { id: "paper-1", status: "processed", source: "arxiv" },
      { id: "paper-2", status: "processed", source: "arxiv" },
      { id: "paper-3", status: "pending", source: "semanticscholar" },
    ],
  };
}
