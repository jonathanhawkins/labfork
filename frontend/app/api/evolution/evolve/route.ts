/**
 * Evolution Evolve API
 *
 * POST /api/evolution/evolve - Trigger evolution cycle
 */

import { NextRequest, NextResponse } from "next/server";
import {
  EvolutionPipeline,
  createEvolutionPipeline,
  DEFAULT_PIPELINE_CONFIG,
  PipelineConfig,
} from "@/lib/meta/evolution";
import { getGlobalEvolutionEngine } from "@/lib/meta/evolution/engine";

// Store active pipelines
const activePipelines = new Map<string, EvolutionPipeline>();

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      populationId,
      populationName = "Evolution Population",
      generations = 10,
      mode = "batch",
      config = {},
    } = body;

    // Merge config
    const pipelineConfig: Partial<PipelineConfig> = {
      ...config,
      maxGenerations: generations,
      mode,
      batchSize: mode === "batch" ? Math.min(generations, 10) : 1,
    };

    // Create pipeline
    const pipeline = createEvolutionPipeline(pipelineConfig);

    // Initialize population
    const engine = getGlobalEvolutionEngine();
    const population = engine.initializePopulation(populationName);

    pipeline.initialize(population);

    // Store pipeline for later queries
    const pipelineId = `pipeline-${Date.now()}`;
    activePipelines.set(pipelineId, pipeline);

    // Run evolution
    const result = await pipeline.start();

    // Get final state
    const finalState = pipeline.getState();

    return NextResponse.json({
      success: true,
      pipelineId,
      result: {
        totalGenerations: result.totalGenerations,
        converged: result.converged,
        durationMs: result.durationMs,
        bestChromosome: {
          id: result.bestChromosome.id,
          name: result.bestChromosome.name,
          fitness: result.bestChromosome.fitness,
          fitnessComponents: result.bestChromosome.fitnessComponents,
          generation: result.bestChromosome.generation,
        },
        population: {
          id: result.population.id,
          name: result.population.name,
          generation: result.population.generation,
          size: result.population.size,
          averageFitness: result.population.averageFitness,
          status: result.population.status,
        },
        finalStats: result.finalStats,
      },
      state: {
        status: finalState.status,
        currentGeneration: finalState.currentGeneration,
        elapsedTime: finalState.elapsedTime,
        snapshotCount: finalState.snapshots.size,
        checkpointCount: finalState.checkpoints.length,
      },
    });
  } catch (error) {
    console.error("Evolution evolve error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const pipelineId = searchParams.get("pipelineId");

    if (pipelineId) {
      const pipeline = activePipelines.get(pipelineId);
      if (!pipeline) {
        return NextResponse.json(
          { success: false, error: "Pipeline not found" },
          { status: 404 }
        );
      }

      const state = pipeline.getState();
      const population = pipeline.getPopulation();

      return NextResponse.json({
        success: true,
        pipelineId,
        state: {
          status: state.status,
          currentGeneration: state.currentGeneration,
          elapsedTime: state.elapsedTime,
        },
        population: population
          ? {
              id: population.id,
              name: population.name,
              generation: population.generation,
              size: population.size,
              averageFitness: population.averageFitness,
              status: population.status,
            }
          : null,
      });
    }

    // List all active pipelines
    const pipelines = Array.from(activePipelines.entries()).map(
      ([id, pipeline]) => {
        const state = pipeline.getState();
        return {
          id,
          status: state.status,
          currentGeneration: state.currentGeneration,
        };
      }
    );

    return NextResponse.json({
      success: true,
      pipelines,
      count: pipelines.length,
    });
  } catch (error) {
    console.error("Evolution evolve GET error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
