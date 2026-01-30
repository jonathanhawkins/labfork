/**
 * Tests for Pattern Trends API
 */

import { describe, it, expect, beforeEach } from "vitest";
import { GET } from "@/app/api/patterns/trends/route";
import { NextRequest } from "next/server";
import { resetGlobalPatternRecognition } from "@/lib/meta/patterns";
import { resetGlobalGraph, getGlobalGraph, createTechniqueNode } from "@/lib/meta/knowledge-graph";

describe("GET /api/patterns/trends", () => {
  beforeEach(() => {
    resetGlobalGraph();
    resetGlobalPatternRecognition();
  });

  it("should return empty trends when no data exists", async () => {
    const request = new NextRequest("http://localhost:3000/api/patterns/trends");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.data).toHaveLength(0);
  });

  it("should detect trends from techniques", async () => {
    const graph = getGlobalGraph();

    // Add techniques with common tags
    for (let i = 0; i < 5; i++) {
      const tech = createTechniqueNode(`Transformer ${i}`, "architecture", {
        description: `A transformer model ${i}`,
        domains: ["speech"],
        tags: ["transformer", "attention"],
      });
      graph.addNode(tech);
    }

    const request = new NextRequest("http://localhost:3000/api/patterns/trends");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.pagination).toBeDefined();
  });

  it("should filter by category", async () => {
    const graph = getGlobalGraph();

    for (let i = 0; i < 3; i++) {
      const tech = createTechniqueNode(`Arch Tech ${i}`, "architecture", {
        description: `Architecture model ${i}`,
        domains: ["speech"],
        tags: ["architecture", "layer"],
      });
      graph.addNode(tech);
    }

    const request = new NextRequest(
      "http://localhost:3000/api/patterns/trends?category=architecture"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    for (const trend of data.data) {
      expect(trend.category).toBe("architecture");
    }
  });

  it("should filter by minimum strength", async () => {
    const graph = getGlobalGraph();

    for (let i = 0; i < 5; i++) {
      const tech = createTechniqueNode(`Strong Tech ${i}`, "architecture", {
        domains: ["speech"],
        tags: ["strong-trend"],
      });
      graph.addNode(tech);
    }

    const request = new NextRequest(
      "http://localhost:3000/api/patterns/trends?minStrength=0.5"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    for (const trend of data.data) {
      expect(trend.strength).toBeGreaterThanOrEqual(0.5);
    }
  });

  it("should filter by domain", async () => {
    const graph = getGlobalGraph();

    for (let i = 0; i < 3; i++) {
      const tech = createTechniqueNode(`Speech Tech ${i}`, "architecture", {
        domains: ["speech"],
        tags: ["common"],
      });
      graph.addNode(tech);
    }

    for (let i = 0; i < 3; i++) {
      const tech = createTechniqueNode(`Image Tech ${i}`, "architecture", {
        domains: ["image"],
        tags: ["other"],
      });
      graph.addNode(tech);
    }

    const request = new NextRequest(
      "http://localhost:3000/api/patterns/trends?domain=speech"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    for (const trend of data.data) {
      expect(trend.domains).toContain("speech");
    }
  });

  it("should paginate results", async () => {
    const graph = getGlobalGraph();

    // Create many techniques with different tags
    const tags = ["tag-a", "tag-b", "tag-c", "tag-d", "tag-e"];
    for (const tag of tags) {
      for (let i = 0; i < 3; i++) {
        const tech = createTechniqueNode(`${tag} Tech ${i}`, "architecture", {
          domains: ["speech"],
          tags: [tag],
        });
        graph.addNode(tech);
      }
    }

    const request = new NextRequest(
      "http://localhost:3000/api/patterns/trends?page=1&limit=2"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.data.length).toBeLessThanOrEqual(2);
    expect(data.pagination.page).toBe(1);
    expect(data.pagination.limit).toBe(2);
  });
});
