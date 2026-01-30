/**
 * Evolution Fitness API
 *
 * GET /api/evolution/fitness/[id] - Get fitness details for a chromosome
 */

import { NextRequest, NextResponse } from "next/server";

// Import from population route to share state
// In a real app, this would use a shared store
let currentPopulation: import("@/lib/meta/evolution").Population | null = null;

// Helper to get population (would be shared in real app)
// Note: Not exported as Next.js route files only allow HTTP method exports
function setPopulation(pop: import("@/lib/meta/evolution").Population | null) {
  currentPopulation = pop;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const startTime = Date.now();

  try {
    const { id } = await params;

    // In a real implementation, we'd use shared state
    // For now, return a mock response based on ID
    if (!id) {
      return NextResponse.json(
        {
          success: false,
          error: "Chromosome ID required",
        },
        { status: 400 }
      );
    }

    // Mock fitness data for demonstration
    const fitness = {
      chromosomeId: id,
      overall: 0.75,
      components: {
        quality: 0.8,
        efficiency: 0.7,
        novelty: 0.75,
        feasibility: 0.72,
        compatibility: 0.78,
      },
      breakdown: {
        architectureScore: 0.82,
        trainingScore: 0.68,
        hyperparameterScore: 0.75,
        dataScore: 0.71,
      },
      comparison: {
        vsPopulationAverage: 0.12,
        vsBestInGeneration: -0.05,
        percentile: 85,
      },
      history: [
        { generation: 0, fitness: 0.5 },
        { generation: 1, fitness: 0.62 },
        { generation: 2, fitness: 0.68 },
        { generation: 3, fitness: 0.75 },
      ],
    };

    return NextResponse.json({
      success: true,
      data: fitness,
      meta: {
        executionTimeMs: Date.now() - startTime,
      },
    });
  } catch (error) {
    console.error("Failed to get fitness:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to get chromosome fitness",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
