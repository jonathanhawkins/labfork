/**
 * Evolution Population API
 *
 * GET /api/evolution/population - Get current population
 * POST /api/evolution/population - Initialize new population
 * PATCH /api/evolution/population - Evolve population
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph, isTechniqueNode, TechniqueNode } from "@/lib/meta/knowledge-graph";
import {
  getGlobalEvolutionEngine,
  Population,
  EvolutionConfig,
  DEFAULT_EVOLUTION_CONFIG,
} from "@/lib/meta/evolution";

// Store current population in memory
let currentPopulation: Population | null = null;

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    if (!currentPopulation) {
      return NextResponse.json({
        success: true,
        data: null,
        message: "No active population. Use POST to initialize.",
        meta: {
          executionTimeMs: Date.now() - startTime,
        },
      });
    }

    // Serialize population for response
    const serialized = {
      ...currentPopulation,
      chromosomes: currentPopulation.chromosomes.map((chr) => ({
        ...chr,
        createdAt: chr.createdAt.toISOString(),
        mutations: chr.mutations.map((m) => ({
          ...m,
          timestamp: m.timestamp.toISOString(),
        })),
      })),
      generationHistory: currentPopulation.generationHistory.map((g) => ({
        ...g,
        timestamp: g.timestamp.toISOString(),
      })),
      createdAt: currentPopulation.createdAt.toISOString(),
      lastEvolved: currentPopulation.lastEvolved.toISOString(),
    };

    return NextResponse.json({
      success: true,
      data: serialized,
      meta: {
        executionTimeMs: Date.now() - startTime,
      },
    });
  } catch (error) {
    console.error("Failed to get population:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to get population",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    const body = await request.json();
    const { name = "Evolution Population", config = {} } = body;

    const engine = getGlobalEvolutionEngine();
    const graph = getGlobalGraph();

    // Get seed techniques from graph
    const techniques = graph
      .getNodesByType("technique")
      .filter(isTechniqueNode) as TechniqueNode[];

    // Initialize population
    currentPopulation = engine.initializePopulation(name, techniques);

    // Serialize for response
    const serialized = {
      ...currentPopulation,
      chromosomes: currentPopulation.chromosomes.slice(0, 10).map((chr) => ({
        ...chr,
        createdAt: chr.createdAt.toISOString(),
        mutations: chr.mutations.map((m) => ({
          ...m,
          timestamp: m.timestamp.toISOString(),
        })),
      })),
      generationHistory: currentPopulation.generationHistory.map((g) => ({
        ...g,
        timestamp: g.timestamp.toISOString(),
      })),
      createdAt: currentPopulation.createdAt.toISOString(),
      lastEvolved: currentPopulation.lastEvolved.toISOString(),
    };

    return NextResponse.json({
      success: true,
      data: {
        ...serialized,
        totalChromosomes: currentPopulation.chromosomes.length,
      },
      meta: {
        executionTimeMs: Date.now() - startTime,
        seedTechniques: techniques.length,
      },
    });
  } catch (error) {
    console.error("Failed to initialize population:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to initialize population",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

export async function PATCH(request: NextRequest) {
  const startTime = Date.now();

  try {
    if (!currentPopulation) {
      return NextResponse.json(
        {
          success: false,
          error: "No active population. Use POST to initialize first.",
        },
        { status: 400 }
      );
    }

    const body = await request.json();
    const { generations = 1, runFull = false } = body;

    const engine = getGlobalEvolutionEngine();

    if (runFull) {
      // Run full evolution
      const result = engine.runEvolution(currentPopulation);
      currentPopulation = result.population;

      return NextResponse.json({
        success: true,
        data: {
          converged: result.converged,
          totalGenerations: result.totalGenerations,
          durationMs: result.durationMs,
          bestChromosome: {
            ...result.bestChromosome,
            createdAt: result.bestChromosome.createdAt.toISOString(),
            mutations: result.bestChromosome.mutations.map((m) => ({
              ...m,
              timestamp: m.timestamp.toISOString(),
            })),
          },
          finalStats: {
            ...result.finalStats,
            timestamp: result.finalStats.timestamp.toISOString(),
          },
        },
        meta: {
          executionTimeMs: Date.now() - startTime,
        },
      });
    }

    // Evolve for specified generations
    for (let i = 0; i < generations; i++) {
      currentPopulation = engine.evolveGeneration(currentPopulation);
      if (
        currentPopulation.status === "converged" ||
        currentPopulation.status === "completed"
      ) {
        break;
      }
    }

    const latestStats =
      currentPopulation.generationHistory[
        currentPopulation.generationHistory.length - 1
      ];

    return NextResponse.json({
      success: true,
      data: {
        generation: currentPopulation.generation,
        status: currentPopulation.status,
        bestFitness: latestStats.bestFitness,
        averageFitness: latestStats.averageFitness,
        diversity: currentPopulation.fitnessDiversity,
        bestChromosomeId: currentPopulation.bestChromosomeId,
        stats: {
          ...latestStats,
          timestamp: latestStats.timestamp.toISOString(),
        },
      },
      meta: {
        executionTimeMs: Date.now() - startTime,
        generationsRun: Math.min(
          generations,
          currentPopulation.generation -
            (currentPopulation.generationHistory.length - generations - 1)
        ),
      },
    });
  } catch (error) {
    console.error("Failed to evolve population:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to evolve population",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
