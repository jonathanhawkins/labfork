/**
 * Knowledge Graph Statistics API
 *
 * GET /api/knowledge/stats - Get graph statistics
 */

import { NextRequest, NextResponse } from "next/server";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";

export async function GET(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const stats = graph.getStats();

    // Additional computed stats
    const techniques = graph.getNodesByType("technique");
    const papers = graph.getNodesByType("paper");
    const labs = graph.getNodesByType("lab");

    // Top connected nodes
    const allNodes = graph.getAllNodes();
    const nodesByDegree = allNodes
      .map((node) => ({
        id: node.id,
        name: node.name,
        type: node.type,
        degree: graph.getNodeDegree(node.id),
      }))
      .sort((a, b) => b.degree - a.degree)
      .slice(0, 10);

    return NextResponse.json({
      success: true,
      data: {
        overview: {
          nodeCount: stats.nodeCount,
          edgeCount: stats.edgeCount,
          averageDegree: Math.round(stats.averageDegree * 100) / 100,
          density: Math.round(stats.density * 10000) / 10000,
          componentCount: stats.componentCount,
          lastUpdated: stats.lastUpdated,
        },
        nodesByType: stats.nodeCountByType,
        edgesByType: stats.edgeCountByType,
        topConnected: nodesByDegree,
        summary: {
          techniques: techniques.length,
          papers: papers.length,
          labs: labs.length,
          concepts: stats.nodeCountByType.concept || 0,
          results: stats.nodeCountByType.result || 0,
          domains: stats.nodeCountByType.domain || 0,
        },
      },
    });
  } catch (error) {
    console.error("Error getting stats:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get stats",
      },
      { status: 500 }
    );
  }
}
