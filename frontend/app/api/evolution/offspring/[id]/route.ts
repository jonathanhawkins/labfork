/**
 * Evolution Offspring API
 *
 * GET /api/evolution/offspring/[id] - Get technique offspring (descendants)
 */

import { NextRequest, NextResponse } from "next/server";
import {
  createLineageTracker,
  getOffspring,
  getDescendants,
  analyzeLineage,
  toChromosomeLineage,
} from "@/lib/meta/evolution/lineage";
import { getGlobalEvolutionEngine } from "@/lib/meta/evolution/engine";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const searchParams = request.nextUrl.searchParams;
    const directOnly = searchParams.get("directOnly") === "true";
    const includeAnalysis = searchParams.get("includeAnalysis") === "true";
    const maxDepth = parseInt(searchParams.get("maxDepth") || "10", 10);

    // Create a sample population and lineage tracker for demonstration
    const engine = getGlobalEvolutionEngine();
    const population = engine.initializePopulation("Sample Population");

    // Evolve a few generations to create lineage
    let currentPop = population;
    for (let i = 0; i < 5; i++) {
      currentPop = engine.evolveGeneration(currentPop);
    }

    // Create lineage tracker
    const tracker = createLineageTracker(currentPop.id);

    // Record all chromosomes
    for (const chr of currentPop.chromosomes) {
      // Simulate parent recording
      const record = {
        chromosome: chr,
        recordedAt: new Date(),
        generation: chr.generation,
        survivalCount: 1,
        offspringCount: 0,
        contributionScore: chr.fitness,
      };
      tracker.chromosomes.set(chr.id, record);
    }

    // Try to find the chromosome (or use first one as demo)
    let targetId = id;
    let found = tracker.chromosomes.has(id);

    if (!found && currentPop.chromosomes.length > 0) {
      // Use the best chromosome for demo
      targetId = currentPop.chromosomes[0].id;
      found = true;
    }

    if (!found) {
      return NextResponse.json(
        {
          success: false,
          error: "Chromosome not found",
          suggestion: "Use a valid chromosome ID or omit to get sample data",
        },
        { status: 404 }
      );
    }

    // Get offspring
    const offspring = getOffspring(tracker, targetId, directOnly);
    const allDescendants = directOnly
      ? offspring.map((o) => o.id)
      : getDescendants(tracker, targetId, maxDepth);

    // Build response
    const response: Record<string, unknown> = {
      success: true,
      chromosomeId: targetId,
      directOnly,
      offspring: offspring.map((o) => ({
        id: o.id,
        name: o.name,
        fitness: o.fitness,
        generation: o.generation,
        fitnessComponents: o.fitnessComponents,
        parentIds: o.parentIds,
        mutationCount: o.mutations.length,
      })),
      offspringCount: offspring.length,
      totalDescendants: allDescendants.length,
    };

    // Include analysis if requested
    if (includeAnalysis) {
      const analysis = analyzeLineage(tracker, targetId, currentPop);
      const lineage = toChromosomeLineage(tracker, targetId);

      response.analysis = {
        totalDescendants: analysis.totalDescendants,
        livingDescendants: analysis.livingDescendants,
        depth: analysis.depth,
        populationContribution: analysis.populationContribution,
        keyMutations: analysis.keyMutations.slice(0, 5),
        dominantGenes: analysis.dominantGenes.slice(0, 5),
        averageFitnessOverTime: analysis.averageFitnessOverTime,
      };

      response.lineage = {
        rootId: lineage.rootId,
        ancestorCount: lineage.ancestors.length,
        depth: lineage.depth,
        fitnessTrajectory: lineage.fitnessTrajectory,
      };
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error("Offspring API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
