/**
 * Evolution Lineage API
 *
 * GET /api/evolution/lineage/[id] - Get lineage for a chromosome
 */

import { NextRequest, NextResponse } from "next/server";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const startTime = Date.now();

  try {
    const { id } = await params;

    if (!id) {
      return NextResponse.json(
        {
          success: false,
          error: "Chromosome ID required",
        },
        { status: 400 }
      );
    }

    // Mock lineage data for demonstration
    const lineage = {
      chromosomeId: id,
      rootId: id,
      depth: 3,
      ancestors: [`${id}-parent-1`, `${id}-parent-2`, `${id}-grandparent`],
      tree: {
        id,
        name: `Chromosome-${id.slice(-4)}`,
        generation: 3,
        fitness: 0.75,
        children: [
          {
            id: `${id}-parent-1`,
            name: "Parent-1",
            generation: 2,
            fitness: 0.68,
            children: [
              {
                id: `${id}-grandparent`,
                name: "Grandparent",
                generation: 1,
                fitness: 0.55,
                children: [],
              },
            ],
          },
          {
            id: `${id}-parent-2`,
            name: "Parent-2",
            generation: 2,
            fitness: 0.72,
            children: [],
          },
        ],
      },
      keyMutations: [
        {
          geneId: "gene-transformer",
          originalValue: false,
          newValue: true,
          type: "point",
          timestamp: new Date(Date.now() - 86400000).toISOString(),
          fitnessImpact: 0.08,
        },
        {
          geneId: "gene-attention",
          originalValue: "basic",
          newValue: "multi-head",
          type: "point",
          timestamp: new Date(Date.now() - 43200000).toISOString(),
          fitnessImpact: 0.05,
        },
      ],
      fitnessTrajectory: [
        { generation: 0, fitness: 0.45 },
        { generation: 1, fitness: 0.55 },
        { generation: 2, fitness: 0.68 },
        { generation: 3, fitness: 0.75 },
      ],
      inheritanceBreakdown: {
        fromParent1: 0.55,
        fromParent2: 0.35,
        mutations: 0.1,
      },
    };

    return NextResponse.json({
      success: true,
      data: lineage,
      meta: {
        executionTimeMs: Date.now() - startTime,
      },
    });
  } catch (error) {
    console.error("Failed to get lineage:", error);
    return NextResponse.json(
      {
        success: false,
        error: "Failed to get chromosome lineage",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
