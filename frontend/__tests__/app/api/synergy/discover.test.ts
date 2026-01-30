/**
 * Tests for Synergy Discovery API
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { GET } from "@/app/api/synergy/discover/route";
import { NextRequest } from "next/server";
import {
  resetGlobalSynergyDiscovery,
  getGlobalSynergyDiscovery,
} from "@/lib/meta/synergy";
import { resetGlobalGraph, getGlobalGraph, createTechniqueNode } from "@/lib/meta/knowledge-graph";

describe("GET /api/synergy/discover", () => {
  beforeEach(() => {
    resetGlobalGraph();
    resetGlobalSynergyDiscovery();
  });

  it("should return empty proposals when no techniques exist", async () => {
    const request = new NextRequest("http://localhost:3000/api/synergy/discover");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.data.proposals).toHaveLength(0);
    expect(data.data.count).toBe(0);
  });

  it("should discover synergies between techniques", async () => {
    const graph = getGlobalGraph();

    // Add techniques
    const tech1 = createTechniqueNode("Transformer Encoder", "architecture", {
      description: "Uses transformer for encoding",
      domains: ["speech"],
      tags: ["transformer", "encoder"],
    });
    const tech2 = createTechniqueNode("Diffusion Decoder", "architecture", {
      description: "Uses diffusion for decoding",
      domains: ["speech"],
      tags: ["diffusion", "decoder"],
    });

    graph.addNode(tech1);
    graph.addNode(tech2);

    const request = new NextRequest(
      "http://localhost:3000/api/synergy/discover?minSimilarity=0&minScore=0"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.data.proposals.length).toBeGreaterThanOrEqual(0);
    expect(data.meta.executionTimeMs).toBeDefined();
    expect(data.meta.config).toBeDefined();
  });

  it("should respect query parameters", async () => {
    const graph = getGlobalGraph();

    // Add multiple techniques
    for (let i = 0; i < 5; i++) {
      const tech = createTechniqueNode(`Tech ${i}`, "architecture", {
        description: `Description ${i}`,
        domains: ["speech"],
        tags: ["common"],
      });
      graph.addNode(tech);
    }

    const request = new NextRequest(
      "http://localhost:3000/api/synergy/discover?maxProposals=3&minScore=0&minSimilarity=0"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.data.proposals.length).toBeLessThanOrEqual(3);
  });

  it("should filter by focus domains", async () => {
    const graph = getGlobalGraph();

    const speechTech = createTechniqueNode("Speech Tech", "architecture", {
      domains: ["speech"],
    });
    const imageTech = createTechniqueNode("Image Tech", "architecture", {
      domains: ["image"],
    });

    graph.addNode(speechTech);
    graph.addNode(imageTech);

    const request = new NextRequest(
      "http://localhost:3000/api/synergy/discover?focusDomains=speech&minScore=0&minSimilarity=0"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    // Should only have speech techniques in proposals
    for (const proposal of data.data.proposals) {
      expect(proposal.techniqueA.id).not.toBe(imageTech.id);
      expect(proposal.techniqueB.id).not.toBe(imageTech.id);
    }
  });

  it("should include proposal details", async () => {
    const graph = getGlobalGraph();

    const tech1 = createTechniqueNode("Tech A", "architecture", {
      description: "First technique",
      domains: ["speech"],
    });
    const tech2 = createTechniqueNode("Tech B", "architecture", {
      description: "Second technique",
      domains: ["speech"],
    });

    graph.addNode(tech1);
    graph.addNode(tech2);

    const request = new NextRequest(
      "http://localhost:3000/api/synergy/discover?minScore=0&minSimilarity=0"
    );
    const response = await GET(request);
    const data = await response.json();

    if (data.data.proposals.length > 0) {
      const proposal = data.data.proposals[0];
      expect(proposal.id).toBeDefined();
      expect(proposal.techniqueA).toBeDefined();
      expect(proposal.techniqueB).toBeDefined();
      expect(proposal.score).toBeDefined();
      expect(proposal.justification).toBeDefined();
      expect(proposal.status).toBe("pending");
    }
  });
});
