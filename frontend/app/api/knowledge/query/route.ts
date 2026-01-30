/**
 * Knowledge Graph Query API
 *
 * POST /api/knowledge/query - Execute graph queries
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalGraph,
  createQueryEngine,
  NodeFilter,
  TraversalOptions,
  SimilarityQuery,
  PatternQuery,
} from "@/lib/meta/knowledge-graph";

type QueryType = "nodes" | "edges" | "traverse" | "similar" | "pattern" | "aggregate";

interface QueryRequest {
  type: QueryType;
  filter?: NodeFilter;
  options?: Partial<TraversalOptions> | SimilarityQuery | PatternQuery;
  startId?: string;
  groupBy?: string;
  metrics?: string[];
}

export async function POST(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const engine = createQueryEngine(graph);
    const body: QueryRequest = await request.json();

    const { type } = body;

    if (!type) {
      return NextResponse.json(
        { success: false, error: "Query type is required" },
        { status: 400 }
      );
    }

    const startTime = performance.now();
    let result: unknown;

    switch (type) {
      case "nodes": {
        const queryResult = graph.findNodes(body.filter || {});
        result = {
          nodes: queryResult,
          count: queryResult.length,
        };
        break;
      }

      case "edges": {
        const edges = graph.findEdges((body.filter || {}) as unknown as Parameters<typeof graph.findEdges>[0]);
        result = {
          edges,
          count: edges.length,
        };
        break;
      }

      case "traverse": {
        if (!body.startId) {
          return NextResponse.json(
            { success: false, error: "startId is required for traverse query" },
            { status: 400 }
          );
        }

        const traversalOptions: TraversalOptions = {
          maxDepth: 2,
          direction: "both",
          includeStart: true,
          ...body.options,
        };

        const subgraph = engine.traverse(body.startId, traversalOptions);
        result = {
          nodes: subgraph.nodes,
          edges: subgraph.edges,
          rootId: subgraph.rootId,
        };
        break;
      }

      case "similar": {
        const similarityQuery = body.options as SimilarityQuery;
        if (!similarityQuery?.nodeId) {
          return NextResponse.json(
            { success: false, error: "nodeId is required for similar query" },
            { status: 400 }
          );
        }

        const similar = engine.findSimilar({
          method: "hybrid",
          limit: 10,
          ...similarityQuery,
        });

        result = {
          similar: similar.map((s) => ({
            node: s.node,
            similarity: s.similarity,
          })),
          count: similar.length,
        };
        break;
      }

      case "pattern": {
        const patternQuery = body.options as PatternQuery;
        if (!patternQuery?.pattern) {
          return NextResponse.json(
            { success: false, error: "pattern is required for pattern query" },
            { status: 400 }
          );
        }

        const matches = engine.matchPattern(patternQuery);
        result = {
          matches,
          count: matches.length,
        };
        break;
      }

      case "aggregate": {
        const aggregations = engine.aggregateNodes(
          body.filter || {},
          body.groupBy,
          body.metrics
        );
        result = {
          aggregations,
          count: aggregations.length,
        };
        break;
      }

      default:
        return NextResponse.json(
          { success: false, error: `Unknown query type: ${type}` },
          { status: 400 }
        );
    }

    const executionTimeMs = performance.now() - startTime;

    return NextResponse.json({
      success: true,
      data: result,
      meta: {
        queryType: type,
        executionTimeMs: Math.round(executionTimeMs * 100) / 100,
      },
    });
  } catch (error) {
    console.error("Error executing query:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to execute query",
      },
      { status: 500 }
    );
  }
}
