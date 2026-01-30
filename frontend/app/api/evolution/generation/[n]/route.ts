/**
 * Evolution Generation API
 *
 * GET /api/evolution/generation/[n] - Get specific generation data
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalEvolutionEngine } from "@/lib/meta/evolution/engine";

// In-memory storage for generation history (in production, use a database)
const generationHistory = new Map<number, {
  generation: number;
  timestamp: Date;
  chromosomes: Array<{
    id: string;
    name: string;
    fitness: number;
    fitnessComponents: {
      quality: number;
      efficiency: number;
      novelty: number;
      feasibility: number;
      compatibility: number;
    };
    isElite: boolean;
    parentIds: string[];
  }>;
  stats: {
    bestFitness: number;
    averageFitness: number;
    worstFitness: number;
    fitnessStdDev: number;
    mutationCount: number;
    crossoverCount: number;
    eliteCount: number;
  };
}>();

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ n: string }> }
) {
  try {
    const { n } = await params;
    const generation = parseInt(n, 10);

    if (isNaN(generation) || generation < 0) {
      return NextResponse.json(
        { success: false, error: "Invalid generation number" },
        { status: 400 }
      );
    }

    // Check if we have this generation in history
    const genData = generationHistory.get(generation);

    if (genData) {
      return NextResponse.json({
        success: true,
        generation: genData.generation,
        timestamp: genData.timestamp,
        chromosomeCount: genData.chromosomes.length,
        chromosomes: genData.chromosomes,
        stats: genData.stats,
      });
    }

    // If not in history, try to generate sample data
    // In production, this would query a database
    const engine = getGlobalEvolutionEngine();
    const samplePopulation = engine.initializePopulation(`Generation ${generation}`);

    // Simulate evolution to the requested generation
    let currentPop = samplePopulation;
    for (let i = 0; i < generation && i < 10; i++) {
      currentPop = engine.evolveGeneration(currentPop);
    }

    const stats = currentPop.generationHistory[currentPop.generationHistory.length - 1];

    return NextResponse.json({
      success: true,
      generation: currentPop.generation,
      timestamp: currentPop.lastEvolved,
      chromosomeCount: currentPop.chromosomes.length,
      chromosomes: currentPop.chromosomes.slice(0, 20).map((chr) => ({
        id: chr.id,
        name: chr.name,
        fitness: chr.fitness,
        fitnessComponents: chr.fitnessComponents,
        isElite: chr.isElite,
        parentIds: chr.parentIds,
      })),
      stats: stats
        ? {
            bestFitness: stats.bestFitness,
            averageFitness: stats.averageFitness,
            worstFitness: stats.worstFitness,
            fitnessStdDev: stats.fitnessStdDev,
            mutationCount: stats.mutationCount,
            crossoverCount: stats.crossoverCount,
            eliteCount: stats.eliteCount,
          }
        : null,
      note: "Generated sample data - in production, this would query stored history",
    });
  } catch (error) {
    console.error("Generation API error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

// Store generation data (called during evolution)
export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ n: string }> }
) {
  try {
    const { n } = await params;
    const generation = parseInt(n, 10);
    const body = await request.json();

    if (isNaN(generation) || generation < 0) {
      return NextResponse.json(
        { success: false, error: "Invalid generation number" },
        { status: 400 }
      );
    }

    generationHistory.set(generation, {
      generation,
      timestamp: new Date(),
      chromosomes: body.chromosomes || [],
      stats: body.stats || {},
    });

    return NextResponse.json({
      success: true,
      message: `Generation ${generation} stored`,
    });
  } catch (error) {
    console.error("Generation POST error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
