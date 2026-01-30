/**
 * Synergy Discovery API
 *
 * GET /api/synergy/discover - Run discovery algorithm and return proposals
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalSynergyDiscovery,
  storeProposals,
  SynergyDiscoveryConfig,
} from "@/lib/meta/synergy";

export async function GET(request: NextRequest) {
  try {
    const discovery = getGlobalSynergyDiscovery();
    const { searchParams } = new URL(request.url);

    // Build config from query params
    const configOverrides: Partial<SynergyDiscoveryConfig> = {};

    const minSimilarity = searchParams.get("minSimilarity");
    if (minSimilarity) {
      configOverrides.minSimilarity = parseFloat(minSimilarity);
    }

    const minScore = searchParams.get("minScore");
    if (minScore) {
      configOverrides.minScore = parseFloat(minScore);
    }

    const maxProposals = searchParams.get("maxProposals");
    if (maxProposals) {
      configOverrides.maxProposals = parseInt(maxProposals);
    }

    const focusDomains = searchParams.get("focusDomains");
    if (focusDomains) {
      configOverrides.focusDomains = focusDomains.split(",");
    }

    const filterTags = searchParams.get("filterTags");
    if (filterTags) {
      configOverrides.filterTags = filterTags.split(",");
    }

    const excludeExplored = searchParams.get("excludeExplored");
    if (excludeExplored !== null) {
      configOverrides.excludeExplored = excludeExplored === "true";
    }

    // Apply config overrides temporarily
    const originalConfig = discovery.getConfig();
    if (Object.keys(configOverrides).length > 0) {
      discovery.updateConfig(configOverrides);
    }

    const startTime = performance.now();

    // Run discovery
    const proposals = discovery.discover();

    // Store proposals
    storeProposals(proposals);

    // Restore original config
    if (Object.keys(configOverrides).length > 0) {
      discovery.updateConfig(originalConfig);
    }

    const executionTimeMs = performance.now() - startTime;

    return NextResponse.json({
      success: true,
      data: {
        proposals: proposals.map((p) => ({
          id: p.id,
          techniqueA: {
            id: p.techniqueA.id,
            name: p.techniqueA.name,
          },
          techniqueB: {
            id: p.techniqueB.id,
            name: p.techniqueB.name,
          },
          score: p.score,
          justification: p.justification,
          combinationAspects: p.combinationAspects,
          expectedOutcomes: p.expectedOutcomes,
          status: p.status,
          createdAt: p.createdAt,
        })),
        count: proposals.length,
      },
      meta: {
        executionTimeMs: Math.round(executionTimeMs * 100) / 100,
        config: discovery.getConfig(),
      },
    });
  } catch (error) {
    console.error("Error running synergy discovery:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to run synergy discovery",
      },
      { status: 500 }
    );
  }
}
