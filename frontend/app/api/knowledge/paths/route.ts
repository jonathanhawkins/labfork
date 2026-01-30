/**
 * Knowledge Graph Path Finding API
 *
 * GET /api/knowledge/paths - Find paths between nodes
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph, EdgeType, isEdgeType } from "@/lib/meta/knowledge-graph";

export async function GET(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const { searchParams } = new URL(request.url);

    const startId = searchParams.get("start");
    const endId = searchParams.get("end");
    const mode = searchParams.get("mode") || "shortest";
    const maxDepth = parseInt(searchParams.get("maxDepth") || "5");

    if (!startId || !endId) {
      return NextResponse.json(
        { success: false, error: "start and end parameters are required" },
        { status: 400 }
      );
    }

    // Validate nodes exist
    if (!graph.hasNode(startId)) {
      return NextResponse.json(
        { success: false, error: `Start node not found: ${startId}` },
        { status: 404 }
      );
    }

    if (!graph.hasNode(endId)) {
      return NextResponse.json(
        { success: false, error: `End node not found: ${endId}` },
        { status: 404 }
      );
    }

    // Edge type filter
    const edgeTypesParam = searchParams.get("edgeTypes");
    let edgeTypes: EdgeType[] | undefined;
    if (edgeTypesParam) {
      edgeTypes = edgeTypesParam.split(",").filter(isEdgeType) as EdgeType[];
    }

    const startTime = performance.now();

    if (mode === "shortest") {
      const path = graph.findShortestPath(startId, endId, {
        edgeTypes,
        direction: "both",
      });

      const executionTimeMs = performance.now() - startTime;

      if (!path) {
        return NextResponse.json({
          success: true,
          data: {
            found: false,
            message: "No path found between the nodes",
          },
          meta: { executionTimeMs },
        });
      }

      return NextResponse.json({
        success: true,
        data: {
          found: true,
          path: {
            nodes: path.nodes.map((n) => ({
              id: n.id,
              name: n.name,
              type: n.type,
            })),
            edges: path.edges.map((e) => ({
              id: e.id,
              type: e.type,
              sourceId: e.sourceId,
              targetId: e.targetId,
            })),
            length: path.length,
            totalWeight: path.totalWeight,
          },
        },
        meta: { executionTimeMs },
      });
    } else if (mode === "all") {
      const paths = graph.findAllPaths(startId, endId, maxDepth);

      const executionTimeMs = performance.now() - startTime;

      return NextResponse.json({
        success: true,
        data: {
          found: paths.length > 0,
          count: paths.length,
          paths: paths.map((path) => ({
            nodes: path.nodes.map((n) => ({
              id: n.id,
              name: n.name,
              type: n.type,
            })),
            edges: path.edges.map((e) => ({
              id: e.id,
              type: e.type,
            })),
            length: path.length,
            totalWeight: path.totalWeight,
          })),
        },
        meta: { executionTimeMs, maxDepth },
      });
    } else {
      return NextResponse.json(
        { success: false, error: `Invalid mode: ${mode}. Use 'shortest' or 'all'` },
        { status: 400 }
      );
    }
  } catch (error) {
    console.error("Error finding paths:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to find paths",
      },
      { status: 500 }
    );
  }
}
