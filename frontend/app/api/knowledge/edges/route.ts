/**
 * Knowledge Graph Edges API
 *
 * GET /api/knowledge/edges - List edges with filtering
 * POST /api/knowledge/edges - Create a new edge
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalGraph,
  createEdge,
  EdgeFilter,
  EdgeType,
  isEdgeType,
} from "@/lib/meta/knowledge-graph";

export async function GET(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const { searchParams } = new URL(request.url);

    // Build filter from query params
    const filter: EdgeFilter = {};

    // Filter by type(s)
    const typesParam = searchParams.get("types");
    if (typesParam) {
      const types = typesParam.split(",").filter(isEdgeType) as EdgeType[];
      if (types.length > 0) {
        filter.types = types;
      }
    }

    // Filter by source
    const sourceIds = searchParams.get("sourceIds");
    if (sourceIds) {
      filter.sourceIds = sourceIds.split(",");
    }

    // Filter by target
    const targetIds = searchParams.get("targetIds");
    if (targetIds) {
      filter.targetIds = targetIds.split(",");
    }

    // Filter by weight
    const minWeight = searchParams.get("minWeight");
    if (minWeight) {
      filter.minWeight = parseFloat(minWeight);
    }

    // Filter by confidence
    const minConfidence = searchParams.get("minConfidence");
    if (minConfidence) {
      filter.minConfidence = parseFloat(minConfidence);
    }

    // Filter by inferred
    const isInferred = searchParams.get("isInferred");
    if (isInferred !== null) {
      filter.isInferred = isInferred === "true";
    }

    // Find edges
    let edges = graph.findEdges(filter);

    // Pagination
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "100");
    const total = edges.length;
    const offset = (page - 1) * limit;

    edges = edges.slice(offset, offset + limit);

    return NextResponse.json({
      success: true,
      data: edges,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + edges.length < total,
      },
    });
  } catch (error) {
    console.error("Error fetching edges:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to fetch edges",
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const body = await request.json();

    const { type, sourceId, targetId, ...options } = body;

    if (!type || !sourceId || !targetId) {
      return NextResponse.json(
        { success: false, error: "type, sourceId, and targetId are required" },
        { status: 400 }
      );
    }

    if (!isEdgeType(type)) {
      return NextResponse.json(
        { success: false, error: `Invalid edge type: ${type}` },
        { status: 400 }
      );
    }

    // Validate source and target nodes exist
    if (!graph.hasNode(sourceId)) {
      return NextResponse.json(
        { success: false, error: `Source node not found: ${sourceId}` },
        { status: 400 }
      );
    }

    if (!graph.hasNode(targetId)) {
      return NextResponse.json(
        { success: false, error: `Target node not found: ${targetId}` },
        { status: 400 }
      );
    }

    const edge = createEdge(type, sourceId, targetId, options);
    graph.addEdge(edge);

    return NextResponse.json({
      success: true,
      data: edge,
    });
  } catch (error) {
    console.error("Error creating edge:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to create edge",
      },
      { status: 500 }
    );
  }
}
