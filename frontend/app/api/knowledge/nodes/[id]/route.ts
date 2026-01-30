/**
 * Knowledge Graph Node by ID API
 *
 * GET /api/knowledge/nodes/[id] - Get node by ID
 * PATCH /api/knowledge/nodes/[id] - Update node
 * DELETE /api/knowledge/nodes/[id] - Delete node
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";

interface RouteContext {
  params: Promise<{ id: string }>;
}

export async function GET(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const { id } = await context.params;
    const graph = getGlobalGraph();

    const node = graph.getNode(id);

    if (!node) {
      return NextResponse.json(
        { success: false, error: "Node not found" },
        { status: 404 }
      );
    }

    // Also get connected edges and neighbors
    const outgoingEdges = graph.getOutgoingEdges(id);
    const incomingEdges = graph.getIncomingEdges(id);
    const neighbors = graph.getNeighbors(id);
    const degree = graph.getNodeDegree(id);

    return NextResponse.json({
      success: true,
      data: {
        node,
        connections: {
          outgoingEdges: outgoingEdges.length,
          incomingEdges: incomingEdges.length,
          neighbors: neighbors.map((n) => ({
            id: n.id,
            name: n.name,
            type: n.type,
          })),
          degree,
        },
      },
    });
  } catch (error) {
    console.error("Error fetching node:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to fetch node",
      },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const { id } = await context.params;
    const graph = getGlobalGraph();
    const body = await request.json();

    const updated = graph.updateNode(id, body);

    if (!updated) {
      return NextResponse.json(
        { success: false, error: "Node not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: updated,
    });
  } catch (error) {
    console.error("Error updating node:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to update node",
      },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  context: RouteContext
) {
  try {
    const { id } = await context.params;
    const graph = getGlobalGraph();

    const removed = graph.removeNode(id);

    if (!removed) {
      return NextResponse.json(
        { success: false, error: "Node not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: "Node removed successfully",
    });
  } catch (error) {
    console.error("Error deleting node:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to delete node",
      },
      { status: 500 }
    );
  }
}
