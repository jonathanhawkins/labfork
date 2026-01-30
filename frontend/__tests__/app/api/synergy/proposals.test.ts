/**
 * Tests for Synergy Proposals API
 */

import { describe, it, expect, beforeEach } from "vitest";
import { GET } from "@/app/api/synergy/proposals/route";
import { GET as GET_SINGLE, PATCH } from "@/app/api/synergy/proposals/[id]/route";
import { NextRequest } from "next/server";
import {
  resetGlobalSynergyDiscovery,
  storeProposal,
  SynergyProposal,
  createProposalId,
} from "@/lib/meta/synergy";
import { resetGlobalGraph, createTechniqueNode } from "@/lib/meta/knowledge-graph";

// Helper to create mock proposals
function createMockProposal(overrides: Partial<SynergyProposal> = {}): SynergyProposal {
  const techA = createTechniqueNode("Tech A", "architecture", {
    domains: ["speech"],
  });
  const techB = createTechniqueNode("Tech B", "architecture", {
    domains: ["speech"],
  });

  return {
    id: createProposalId(),
    techniqueA: techA,
    techniqueB: techB,
    score: {
      overall: 0.75,
      components: {
        similarity: 0.6,
        complementarity: 0.8,
        novelty: 0.7,
        feasibility: 0.8,
        impact: 0.85,
      },
      confidence: 0.7,
    },
    justification: "Test justification",
    combinationAspects: [],
    expectedOutcomes: [],
    status: "pending",
    createdAt: new Date(),
    updatedAt: new Date(),
    createdBy: "system",
    ...overrides,
  };
}

describe("GET /api/synergy/proposals", () => {
  beforeEach(() => {
    resetGlobalGraph();
    resetGlobalSynergyDiscovery();
  });

  it("should return empty list when no proposals exist", async () => {
    const request = new NextRequest("http://localhost:3000/api/synergy/proposals");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.data).toHaveLength(0);
  });

  it("should return stored proposals", async () => {
    const proposal = createMockProposal();
    storeProposal(proposal);

    const request = new NextRequest("http://localhost:3000/api/synergy/proposals");
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.data.length).toBe(1);
    expect(data.data[0].id).toBe(proposal.id);
  });

  it("should filter by status", async () => {
    const pending = createMockProposal({ status: "pending" });
    const accepted = createMockProposal({ status: "accepted" });
    storeProposal(pending);
    storeProposal(accepted);

    const request = new NextRequest(
      "http://localhost:3000/api/synergy/proposals?status=pending"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(data.data.length).toBe(1);
    expect(data.data[0].status).toBe("pending");
  });

  it("should filter by minimum score", async () => {
    const low = createMockProposal({
      score: { overall: 0.3, components: {} as any, confidence: 0.5 },
    });
    const high = createMockProposal({
      score: { overall: 0.8, components: {} as any, confidence: 0.5 },
    });
    storeProposal(low);
    storeProposal(high);

    const request = new NextRequest(
      "http://localhost:3000/api/synergy/proposals?minScore=0.5"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(data.data.length).toBe(1);
    expect(data.data[0].score.overall).toBeGreaterThanOrEqual(0.5);
  });

  it("should sort by score descending by default", async () => {
    const low = createMockProposal({
      score: { overall: 0.3, components: {} as any, confidence: 0.5 },
    });
    const high = createMockProposal({
      score: { overall: 0.8, components: {} as any, confidence: 0.5 },
    });
    storeProposal(low);
    storeProposal(high);

    const request = new NextRequest("http://localhost:3000/api/synergy/proposals");
    const response = await GET(request);
    const data = await response.json();

    expect(data.data[0].score.overall).toBeGreaterThanOrEqual(data.data[1].score.overall);
  });

  it("should paginate results", async () => {
    for (let i = 0; i < 5; i++) {
      storeProposal(createMockProposal());
    }

    const request = new NextRequest(
      "http://localhost:3000/api/synergy/proposals?page=1&limit=2"
    );
    const response = await GET(request);
    const data = await response.json();

    expect(data.data.length).toBe(2);
    expect(data.pagination.page).toBe(1);
    expect(data.pagination.limit).toBe(2);
    expect(data.pagination.total).toBe(5);
    expect(data.pagination.hasMore).toBe(true);
  });
});

describe("GET /api/synergy/proposals/[id]", () => {
  beforeEach(() => {
    resetGlobalGraph();
    resetGlobalSynergyDiscovery();
  });

  it("should return 404 for non-existent proposal", async () => {
    const request = new NextRequest(
      "http://localhost:3000/api/synergy/proposals/non-existent"
    );
    const response = await GET_SINGLE(request, {
      params: Promise.resolve({ id: "non-existent" }),
    });
    const data = await response.json();

    expect(response.status).toBe(404);
    expect(data.success).toBe(false);
  });

  it("should return proposal by ID", async () => {
    const proposal = createMockProposal();
    storeProposal(proposal);

    const request = new NextRequest(
      `http://localhost:3000/api/synergy/proposals/${proposal.id}`
    );
    const response = await GET_SINGLE(request, {
      params: Promise.resolve({ id: proposal.id }),
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.data.id).toBe(proposal.id);
  });
});

describe("PATCH /api/synergy/proposals/[id]", () => {
  beforeEach(() => {
    resetGlobalGraph();
    resetGlobalSynergyDiscovery();
  });

  it("should return 404 for non-existent proposal", async () => {
    const request = new NextRequest(
      "http://localhost:3000/api/synergy/proposals/non-existent",
      {
        method: "PATCH",
        body: JSON.stringify({ status: "accepted" }),
      }
    );
    const response = await PATCH(request, {
      params: Promise.resolve({ id: "non-existent" }),
    });

    expect(response.status).toBe(404);
  });

  it("should update proposal status", async () => {
    const proposal = createMockProposal();
    storeProposal(proposal);

    const request = new NextRequest(
      `http://localhost:3000/api/synergy/proposals/${proposal.id}`,
      {
        method: "PATCH",
        body: JSON.stringify({ status: "accepted" }),
      }
    );
    const response = await PATCH(request, {
      params: Promise.resolve({ id: proposal.id }),
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.data.status).toBe("accepted");
  });

  it("should reject invalid status", async () => {
    const proposal = createMockProposal();
    storeProposal(proposal);

    const request = new NextRequest(
      `http://localhost:3000/api/synergy/proposals/${proposal.id}`,
      {
        method: "PATCH",
        body: JSON.stringify({ status: "invalid" }),
      }
    );
    const response = await PATCH(request, {
      params: Promise.resolve({ id: proposal.id }),
    });

    expect(response.status).toBe(400);
  });

  it("should add notes when updating", async () => {
    const proposal = createMockProposal();
    storeProposal(proposal);

    const request = new NextRequest(
      `http://localhost:3000/api/synergy/proposals/${proposal.id}`,
      {
        method: "PATCH",
        body: JSON.stringify({
          status: "rejected",
          notes: "Not feasible at this time",
        }),
      }
    );
    const response = await PATCH(request, {
      params: Promise.resolve({ id: proposal.id }),
    });
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.data.notes).toBe("Not feasible at this time");
  });
});
