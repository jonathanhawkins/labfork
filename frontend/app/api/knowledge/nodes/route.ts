/**
 * Knowledge Graph Nodes API
 *
 * GET /api/knowledge/nodes - List nodes with filtering
 * POST /api/knowledge/nodes - Create a new node
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalGraph,
  createTechniqueNode,
  createPaperNode,
  NodeType,
  NodeFilter,
  TechniqueCategory,
  isNodeType,
} from "@/lib/meta/knowledge-graph";

export async function GET(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const { searchParams } = new URL(request.url);

    // Build filter from query params
    const filter: NodeFilter = {};

    // Filter by type(s)
    const typesParam = searchParams.get("types");
    if (typesParam) {
      const types = typesParam.split(",").filter(isNodeType);
      if (types.length > 0) {
        filter.types = types;
      }
    }

    // Filter by tags
    const tagsParam = searchParams.get("tags");
    if (tagsParam) {
      filter.tags = tagsParam.split(",");
    }

    // Search query
    const search = searchParams.get("search");
    if (search) {
      filter.search = search;
    }

    // Date range
    const createdAfter = searchParams.get("createdAfter");
    if (createdAfter) {
      filter.createdAfter = createdAfter;
    }

    const createdBefore = searchParams.get("createdBefore");
    if (createdBefore) {
      filter.createdBefore = createdBefore;
    }

    // Find nodes
    let nodes = graph.findNodes(filter);

    // Pagination
    const page = parseInt(searchParams.get("page") || "1");
    const limit = parseInt(searchParams.get("limit") || "50");
    const total = nodes.length;
    const offset = (page - 1) * limit;

    nodes = nodes.slice(offset, offset + limit);

    return NextResponse.json({
      success: true,
      data: nodes,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasMore: offset + nodes.length < total,
      },
    });
  } catch (error) {
    console.error("Error fetching nodes:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to fetch nodes",
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const body = await request.json();

    const { type, name, ...options } = body;

    if (!type || !name) {
      return NextResponse.json(
        { success: false, error: "type and name are required" },
        { status: 400 }
      );
    }

    if (!isNodeType(type)) {
      return NextResponse.json(
        { success: false, error: `Invalid node type: ${type}` },
        { status: 400 }
      );
    }

    let node;

    switch (type) {
      case "technique": {
        const category = (options.category as TechniqueCategory) || "other";
        node = createTechniqueNode(name, category, options);
        break;
      }

      case "paper": {
        node = createPaperNode(name, options);
        break;
      }

      default:
        return NextResponse.json(
          { success: false, error: `Node type ${type} not yet supported for creation` },
          { status: 400 }
        );
    }

    graph.addNode(node);

    return NextResponse.json({
      success: true,
      data: node,
    });
  } catch (error) {
    console.error("Error creating node:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to create node",
      },
      { status: 500 }
    );
  }
}
