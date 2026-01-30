/**
 * Tests for Pattern Report API
 */

import { describe, it, expect, beforeEach } from "vitest";
import { GET } from "@/app/api/patterns/report/route";
import { NextRequest } from "next/server";
import { resetGlobalPatternRecognition } from "@/lib/meta/patterns";
import { resetGlobalGraph, getGlobalGraph, createTechniqueNode } from "@/lib/meta/knowledge-graph";

describe("GET /api/patterns/report", () => {
  beforeEach(() => {
    resetGlobalGraph();
    resetGlobalPatternRecognition();
  });

  it("should generate a report even with no data", async () => {
    const request = new NextRequest("http://localhost:3000/api/patterns/report");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.data.id).toBeDefined();
    expect(data.data.generatedAt).toBeDefined();
    expect(data.data.summary).toBeDefined();
  });

  it("should include all report sections", async () => {
    const graph = getGlobalGraph();

    for (let i = 0; i < 5; i++) {
      const tech = createTechniqueNode(`Tech ${i}`, "architecture", {
        description: `Transformer model ${i}`,
        domains: ["speech"],
        tags: ["transformer"],
      });
      graph.addNode(tech);
    }

    const request = new NextRequest("http://localhost:3000/api/patterns/report");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.data.trends).toBeDefined();
    expect(data.data.emergingPatterns).toBeDefined();
    expect(data.data.adoptionMetrics).toBeDefined();
    expect(data.data.crossDomainTransfers).toBeDefined();
    expect(data.data.summary).toBeDefined();
  });

  it("should include summary statistics", async () => {
    const graph = getGlobalGraph();

    for (let i = 0; i < 3; i++) {
      const tech = createTechniqueNode(`Tech ${i}`, "architecture", {
        domains: ["speech"],
      });
      graph.addNode(tech);
    }

    const request = new NextRequest("http://localhost:3000/api/patterns/report");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.data.summary.techniquesAnalyzed).toBe(3);
  });

  it("should use cached report when available", async () => {
    const graph = getGlobalGraph();

    for (let i = 0; i < 3; i++) {
      const tech = createTechniqueNode(`Tech ${i}`, "architecture", {
        domains: ["speech"],
      });
      graph.addNode(tech);
    }

    // First request
    const request1 = new NextRequest("http://localhost:3000/api/patterns/report");
    const response1 = await GET(request1);
    const data1 = await response1.json();

    // Second request (should be cached)
    const request2 = new NextRequest("http://localhost:3000/api/patterns/report");
    const response2 = await GET(request2);
    const data2 = await response2.json();

    expect(data1.data.id).toBe(data2.data.id);
    expect(data2.meta.cached).toBe(true);
  });

  it("should force refresh when requested", async () => {
    const graph = getGlobalGraph();

    for (let i = 0; i < 3; i++) {
      const tech = createTechniqueNode(`Tech ${i}`, "architecture", {
        domains: ["speech"],
      });
      graph.addNode(tech);
    }

    // First request
    const request1 = new NextRequest("http://localhost:3000/api/patterns/report");
    await GET(request1);

    // Force refresh
    const request2 = new NextRequest(
      "http://localhost:3000/api/patterns/report?refresh=true"
    );
    const response2 = await GET(request2);
    const data2 = await response2.json();

    expect(data2.meta.cached).toBe(false);
  });

  it("should track execution time", async () => {
    const request = new NextRequest("http://localhost:3000/api/patterns/report");
    const response = await GET(request);
    const data = await response.json();

    expect(data.meta.executionTimeMs).toBeDefined();
    expect(typeof data.meta.executionTimeMs).toBe("number");
  });
});
