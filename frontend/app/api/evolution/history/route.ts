/**
 * Evolution History API
 *
 * GET /api/evolution/history - Get full evolution history
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalEvolutionEngine } from "@/lib/meta/evolution/engine";
import {
  createLineageTracker,
  recordGeneration,
  getLineageStats,
  getEvolutionTimeline,
} from "@/lib/meta/evolution/lineage";

// In-memory storage for evolution history
const evolutionHistory = new Map<
  string,
  {
    populationId: string;
    populationName: string;
    startTime: Date;
    endTime?: Date;
    generations: Array<{
      generation: number;
      timestamp: Date;
      bestFitness: number;
      averageFitness: number;
      diversity: number;
      bestChromosomeId: string;
    }>;
    finalStats?: {
      totalGenerations: number;
      converged: boolean;
      durationMs: number;
      finalBestFitness: number;
    };
  }
>();

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const populationId = searchParams.get("populationId");
    const startGeneration = parseInt(searchParams.get("startGeneration") || "0", 10);
    const endGeneration = searchParams.get("endGeneration")
      ? parseInt(searchParams.get("endGeneration")!, 10)
      : undefined;
    const limit = parseInt(searchParams.get("limit") || "100", 10);
    const includeSample = searchParams.get("includeSample") === "true";

    // If requesting specific population history
    if (populationId) {
      const history = evolutionHistory.get(populationId);
      if (history) {
        const filteredGenerations = history.generations.filter(
          (g) =>
            g.generation >= startGeneration &&
            (endGeneration === undefined || g.generation <= endGeneration)
        );

        return NextResponse.json({
          success: true,
          populationId: history.populationId,
          populationName: history.populationName,
          startTime: history.startTime,
          endTime: history.endTime,
          generations: filteredGenerations.slice(0, limit),
          generationCount: filteredGenerations.length,
          finalStats: history.finalStats,
        });
      }
    }

    // Generate sample history for demonstration
    if (includeSample || !populationId) {
      const engine = getGlobalEvolutionEngine();
      const population = engine.initializePopulation("Sample Evolution");

      const tracker = createLineageTracker(population.id);
      const generations: Array<{
        generation: number;
        timestamp: Date;
        bestFitness: number;
        averageFitness: number;
        diversity: number;
        bestChromosomeId: string;
      }> = [];

      let currentPop = population;
      const maxGens = Math.min(limit, 20);

      for (let i = 0; i < maxGens; i++) {
        const stats = currentPop.generationHistory[currentPop.generationHistory.length - 1];

        generations.push({
          generation: currentPop.generation,
          timestamp: currentPop.lastEvolved,
          bestFitness: stats?.bestFitness || currentPop.chromosomes[0]?.fitness || 0,
          averageFitness: currentPop.averageFitness,
          diversity: currentPop.fitnessDiversity,
          bestChromosomeId: currentPop.bestChromosomeId,
        });

        if (stats) {
          recordGeneration(tracker, currentPop, stats);
        }

        if (currentPop.status === "converged") break;
        currentPop = engine.evolveGeneration(currentPop);
      }

      const lineageStats = getLineageStats(tracker);
      const timeline = getEvolutionTimeline(tracker, startGeneration, endGeneration);

      return NextResponse.json({
        success: true,
        populationId: population.id,
        populationName: population.name,
        startTime: population.createdAt,
        endTime: currentPop.lastEvolved,
        generations: generations.filter(
          (g) =>
            g.generation >= startGeneration &&
            (endGeneration === undefined || g.generation <= endGeneration)
        ),
        generationCount: generations.length,
        finalStats: {
          totalGenerations: currentPop.generation,
          converged: currentPop.status === "converged",
          durationMs: currentPop.lastEvolved.getTime() - population.createdAt.getTime(),
          finalBestFitness: currentPop.chromosomes[0]?.fitness || 0,
        },
        lineageStats: {
          totalChromosomes: lineageStats.totalChromosomes,
          totalGenerations: lineageStats.totalGenerations,
          totalBirths: lineageStats.totalBirths,
          totalDeaths: lineageStats.totalDeaths,
          averageSurvival: lineageStats.averageSurvival,
          averageOffspring: lineageStats.averageOffspring,
          longestLineage: lineageStats.longestLineage,
        },
        eventCount: timeline.length,
        note: "Sample evolution history generated for demonstration",
      });
    }

    // List all stored histories
    const histories = Array.from(evolutionHistory.entries()).map(([id, h]) => ({
      populationId: id,
      populationName: h.populationName,
      startTime: h.startTime,
      endTime: h.endTime,
      generationCount: h.generations.length,
      finalStats: h.finalStats,
    }));

    return NextResponse.json({
      success: true,
      histories,
      count: histories.length,
    });
  } catch (error) {
    console.error("Evolution history error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

// Store evolution history
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { populationId, populationName, generations, finalStats } = body;

    if (!populationId) {
      return NextResponse.json(
        { success: false, error: "populationId required" },
        { status: 400 }
      );
    }

    evolutionHistory.set(populationId, {
      populationId,
      populationName: populationName || "Unknown Population",
      startTime: new Date(),
      endTime: finalStats ? new Date() : undefined,
      generations: generations || [],
      finalStats,
    });

    return NextResponse.json({
      success: true,
      message: `Evolution history stored for ${populationId}`,
    });
  } catch (error) {
    console.error("Evolution history POST error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
