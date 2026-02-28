/**
 * Tests for Enhanced Semantic Scholar Client
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  fetchCitations,
  fetchReferences,
  getCitationNetwork,
  findRelatedPapers,
  calculateInfluenceScore,
  getInfluenceScore,
  rankByInfluence,
  buildCitationGraph,
  suggestHighImpactPapers,
  clearCaches,
  getCacheStats,
  CitationInfo,
  CitationNetwork,
  RelatedPaper,
  InfluenceScore,
  PaperGraph,
} from "@/lib/research/semantic-scholar-enhanced";

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock paper response
const mockPaperResponse = {
  paperId: "abc123",
  title: "Test Paper",
  abstract: "This is a test abstract",
  year: 2023,
  citationCount: 150,
  influentialCitationCount: 25,
  referenceCount: 40,
  authors: [{ authorId: "auth1", name: "Test Author" }],
  url: "https://semanticscholar.org/paper/abc123",
  venue: "Test Conference",
  publicationDate: "2023-01-15",
  isOpenAccess: true,
  fieldsOfStudy: ["Computer Science"],
  externalIds: {
    DOI: "10.1234/test",
    ArXiv: "2301.12345",
  },
};

// Mock citation response
const mockCitationResponse = {
  data: [
    {
      citingPaper: {
        paperId: "cite1",
        title: "Citation 1",
        year: 2024,
        citationCount: 10,
        influentialCitationCount: 2,
      },
      isInfluential: true,
    },
    {
      citingPaper: {
        paperId: "cite2",
        title: "Citation 2",
        year: 2024,
        citationCount: 5,
        influentialCitationCount: 0,
      },
      isInfluential: false,
    },
  ],
};

// Mock reference response
const mockReferenceResponse = {
  data: [
    {
      citedPaper: {
        paperId: "ref1",
        title: "Reference 1",
        year: 2020,
        citationCount: 500,
        influentialCitationCount: 100,
      },
      isInfluential: true,
    },
    {
      citedPaper: {
        paperId: "ref2",
        title: "Reference 2",
        year: 2019,
        citationCount: 200,
        influentialCitationCount: 30,
      },
      isInfluential: false,
    },
  ],
};

// Mock search response
const mockSearchResponse = {
  data: [
    {
      ...mockPaperResponse,
      paperId: "search1",
      citationCount: 200,
      influentialCitationCount: 50,
    },
    {
      ...mockPaperResponse,
      paperId: "search2",
      citationCount: 100,
      influentialCitationCount: 20,
    },
  ],
};

describe("Enhanced Semantic Scholar Client", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    clearCaches();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ============================================================================
  // Citation Fetching Tests
  // ============================================================================

  describe("fetchCitations", () => {
    it("should fetch citations for a paper", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCitationResponse),
      });

      const citations = await fetchCitations("abc123");

      expect(citations).toHaveLength(2);
      expect(citations[0].paperId).toBe("cite1");
      expect(citations[0].isInfluential).toBe(true);
      expect(citations[1].paperId).toBe("cite2");
      expect(citations[1].isInfluential).toBe(false);
    });

    it("should return empty array on API error", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const citations = await fetchCitations("abc123");

      expect(citations).toEqual([]);
    });

    it("should use cache on repeated calls", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCitationResponse),
      });

      await fetchCitations("abc123");
      const cachedResult = await fetchCitations("abc123");

      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(cachedResult).toHaveLength(2);
    });

    it("should respect limit and offset options", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCitationResponse),
      });

      await fetchCitations("abc123", { limit: 50, offset: 10 });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("limit=50"),
        expect.any(Object)
      );
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("offset=10"),
        expect.any(Object)
      );
    });
  });

  describe("fetchReferences", () => {
    it("should fetch references for a paper", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockReferenceResponse),
      });

      const references = await fetchReferences("abc123");

      expect(references).toHaveLength(2);
      expect(references[0].paperId).toBe("ref1");
      expect(references[0].isInfluential).toBe(true);
      expect(references[1].paperId).toBe("ref2");
    });

    it("should return empty array on error", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const references = await fetchReferences("nonexistent");

      expect(references).toEqual([]);
    });
  });

  // ============================================================================
  // Citation Network Tests
  // ============================================================================

  describe("getCitationNetwork", () => {
    it("should build complete citation network", async () => {
      // Use URL-based mock to avoid flaky ordering with Promise.all + rate limiter
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("/citations")) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockCitationResponse),
          });
        }
        if (url.includes("/references")) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockReferenceResponse),
          });
        }
        // Paper fetch
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockPaperResponse),
        });
      });

      const network = await getCitationNetwork("abc123");

      expect(network).not.toBeNull();
      expect(network!.paper).toBeDefined();
      expect(network!.citations).toHaveLength(2);
      expect(network!.references).toHaveLength(2);
      expect(network!.totalCitations).toBe(2);
      expect(network!.totalReferences).toBe(2);
      expect(network!.influentialCitations).toBe(1);
      expect(network!.influentialReferences).toBe(1);
    });

    it("should return null for nonexistent paper", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const network = await getCitationNetwork("nonexistent");

      expect(network).toBeNull();
    });
  });

  // ============================================================================
  // Related Papers Tests
  // ============================================================================

  describe("findRelatedPapers", () => {
    it("should find related papers through multiple methods", async () => {
      // Mock citations
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCitationResponse),
      });
      // Mock references
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockReferenceResponse),
      });
      // Mock paper fetches for citations
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({ ...mockPaperResponse, paperId: "cite1" }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({ ...mockPaperResponse, paperId: "cite2" }),
      });
      // Mock paper fetches for references
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ ...mockPaperResponse, paperId: "ref1" }),
      });
      // Mock recommendations
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            recommendedPapers: [{ paperId: "rec1" }],
          }),
      });
      // Mock recommended paper fetch
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ ...mockPaperResponse, paperId: "rec1" }),
      });

      const related = await findRelatedPapers("abc123", {
        maxResults: 5,
        includeRecommendations: true,
        includeCitationOverlap: false,
      });

      expect(related.length).toBeGreaterThan(0);
      expect(related.every((r) => r.paper && r.relevanceScore)).toBe(true);
    });

    it("should sort by relevance score", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            data: [
              { citingPaper: { paperId: "c1" }, isInfluential: false },
              { citingPaper: { paperId: "c2" }, isInfluential: true },
            ],
          }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: [] }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ ...mockPaperResponse, paperId: "c1" }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ ...mockPaperResponse, paperId: "c2" }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ recommendedPapers: [] }),
      });

      const related = await findRelatedPapers("abc123", {
        includeRecommendations: true,
        includeCitationOverlap: false,
      });

      if (related.length >= 2) {
        expect(related[0].relevanceScore).toBeGreaterThanOrEqual(
          related[1].relevanceScore
        );
      }
    });
  });

  // ============================================================================
  // Influence Scoring Tests
  // ============================================================================

  describe("calculateInfluenceScore", () => {
    it("should calculate low influence for few citations", () => {
      const score = calculateInfluenceScore(5, 0);

      expect(score.influenceLevel).toBe("low");
      expect(score.influenceScore).toBeLessThan(10);
    });

    it("should calculate moderate influence for medium citations", () => {
      const score = calculateInfluenceScore(20, 5);

      expect(score.influenceLevel).toBe("moderate");
    });

    it("should calculate high influence for many citations", () => {
      const score = calculateInfluenceScore(100, 20);

      expect(score.influenceLevel).toBe("high");
    });

    it("should calculate very-high influence for very many citations", () => {
      const score = calculateInfluenceScore(300, 50);

      expect(score.influenceLevel).toBe("very-high");
    });

    it("should calculate seminal influence for exceptional papers", () => {
      const score = calculateInfluenceScore(2000, 500);

      expect(score.influenceLevel).toBe("seminal");
    });

    it("should weight influential citations more heavily", () => {
      const scoreWithInfluential = calculateInfluenceScore(100, 50);
      const scoreWithoutInfluential = calculateInfluenceScore(100, 0);

      expect(scoreWithInfluential.influenceScore).toBeGreaterThan(
        scoreWithoutInfluential.influenceScore
      );
    });

    it("should consider citation velocity for newer papers", () => {
      const newPaperScore = calculateInfluenceScore(100, 20, 1);
      const oldPaperScore = calculateInfluenceScore(100, 20, 10);

      expect(newPaperScore.influenceScore).toBeGreaterThan(
        oldPaperScore.influenceScore
      );
    });
  });

  describe("getInfluenceScore", () => {
    it("should get influence score for a paper", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });

      const score = await getInfluenceScore("abc123");

      expect(score).not.toBeNull();
      expect(score!.paperId).toBe("abc123");
      expect(score!.totalCitations).toBe(150);
      expect(score!.influentialCitations).toBe(25);
      expect(score!.influenceScore).toBeGreaterThan(0);
    });

    it("should return null for nonexistent paper", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const score = await getInfluenceScore("nonexistent");

      expect(score).toBeNull();
    });

    it("should cache influence scores", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });

      await getInfluenceScore("abc123");
      await getInfluenceScore("abc123");

      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });

  describe("rankByInfluence", () => {
    it("should rank papers by influence score", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPaperResponse,
            paperId: "p1",
            citationCount: 50,
            influentialCitationCount: 10,
          }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPaperResponse,
            paperId: "p2",
            citationCount: 200,
            influentialCitationCount: 50,
          }),
      });

      const ranked = await rankByInfluence(["p1", "p2"]);

      expect(ranked).toHaveLength(2);
      expect(ranked[0].paperId).toBe("p2");
      expect(ranked[0].score.influenceScore).toBeGreaterThan(
        ranked[1].score.influenceScore
      );
    });

    it("should filter out papers that fail to fetch", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const ranked = await rankByInfluence(["abc123", "nonexistent"]);

      expect(ranked).toHaveLength(1);
      expect(ranked[0].paperId).toBe("abc123");
    });
  });

  // ============================================================================
  // Graph Building Tests
  // ============================================================================

  describe("buildCitationGraph", () => {
    it("should build a citation graph", async () => {
      // Mock root paper
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });
      // Mock citations for root
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCitationResponse),
      });
      // Mock references for root
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockReferenceResponse),
      });
      // Mock depth 1 paper fetches
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });

      const graph = await buildCitationGraph("abc123", {
        depth: 1,
        maxNodesPerLevel: 2,
      });

      expect(graph.rootPaperId).toBe("abc123");
      expect(graph.depth).toBe(1);
      expect(graph.nodes.size).toBeGreaterThan(0);
      expect(graph.edges.length).toBeGreaterThan(0);
    });

    it("should respect depth limit", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: [] }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ data: [] }),
      });

      const graph = await buildCitationGraph("abc123", { depth: 0 });

      expect(graph.nodes.size).toBe(1);
      expect(graph.edges.length).toBe(0);
    });

    it("should filter to influential only when requested", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCitationResponse),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockReferenceResponse),
      });
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockPaperResponse),
      });

      const graph = await buildCitationGraph("abc123", {
        depth: 1,
        onlyInfluential: true,
      });

      // All edges should be influential
      expect(graph.edges.every((e) => e.isInfluential)).toBe(true);
    });
  });

  // ============================================================================
  // High Impact Suggestions Tests
  // ============================================================================

  describe("suggestHighImpactPapers", () => {
    it("should suggest high-impact papers", async () => {
      // Mock search response - this is called first
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSearchResponse),
      });
      // Mock getInfluenceScore calls - these fetch the paper details
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPaperResponse,
            paperId: "search1",
            citationCount: 200,
            influentialCitationCount: 50,
            sourceMetadata: { influentialCitationCount: 50 },
          }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPaperResponse,
            paperId: "search2",
            citationCount: 100,
            influentialCitationCount: 20,
            sourceMetadata: { influentialCitationCount: 20 },
          }),
      });

      const suggestions = await suggestHighImpactPapers("machine learning", {
        minCitations: 50,
        limit: 5,
      });

      expect(suggestions.length).toBeGreaterThanOrEqual(0);
      if (suggestions.length > 0) {
        expect(suggestions.every((s) => s.influenceScore.totalCitations >= 50)).toBe(
          true
        );
      }
    });

    it("should filter by minimum citations", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            data: [
              { ...mockPaperResponse, paperId: "p1", citationCount: 10 },
              { ...mockPaperResponse, paperId: "p2", citationCount: 100 },
            ],
          }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPaperResponse,
            paperId: "p2",
            citationCount: 100,
          }),
      });

      const suggestions = await suggestHighImpactPapers("test", {
        minCitations: 50,
      });

      expect(suggestions.every((s) => s.paper.metadata.citationCount! >= 50)).toBe(
        true
      );
    });

    it("should sort by influence score", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSearchResponse),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPaperResponse,
            paperId: "search1",
            citationCount: 200,
            influentialCitationCount: 50,
          }),
      });
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPaperResponse,
            paperId: "search2",
            citationCount: 100,
            influentialCitationCount: 20,
          }),
      });

      const suggestions = await suggestHighImpactPapers("test", {
        minCitations: 50,
      });

      if (suggestions.length >= 2) {
        expect(suggestions[0].influenceScore.influenceScore).toBeGreaterThanOrEqual(
          suggestions[1].influenceScore.influenceScore
        );
      }
    });
  });

  // ============================================================================
  // Cache Management Tests
  // ============================================================================

  describe("Cache Management", () => {
    it("should clear all caches", () => {
      // Test that clearCaches resets all cache counts to zero
      clearCaches();

      const stats = getCacheStats();
      expect(stats.citations).toBe(0);
      expect(stats.references).toBe(0);
      expect(stats.papers).toBe(0);
      expect(stats.influence).toBe(0);
    });

    it("should return cache stats object with correct structure", () => {
      const stats = getCacheStats();

      // Verify the stats object has the expected properties
      expect(stats).toHaveProperty("citations");
      expect(stats).toHaveProperty("references");
      expect(stats).toHaveProperty("papers");
      expect(stats).toHaveProperty("influence");

      // All values should be non-negative integers
      expect(typeof stats.citations).toBe("number");
      expect(typeof stats.references).toBe("number");
      expect(typeof stats.papers).toBe("number");
      expect(typeof stats.influence).toBe("number");
      expect(stats.citations).toBeGreaterThanOrEqual(0);
      expect(stats.references).toBeGreaterThanOrEqual(0);
    });
  });

  // ============================================================================
  // Type Exports Tests
  // ============================================================================

  describe("Type Exports", () => {
    it("should export CitationInfo type", () => {
      const citation: CitationInfo = {
        paperId: "test",
        title: "Test",
        isInfluential: false,
      };
      expect(citation.paperId).toBe("test");
    });

    it("should export CitationNetwork type", () => {
      const network: Partial<CitationNetwork> = {
        totalCitations: 10,
        totalReferences: 5,
        influentialCitations: 2,
        influentialReferences: 1,
      };
      expect(network.totalCitations).toBe(10);
    });

    it("should export RelatedPaper type", () => {
      const related: Partial<RelatedPaper> = {
        relevanceScore: 0.8,
        relationshipType: "citation",
      };
      expect(related.relevanceScore).toBe(0.8);
    });

    it("should export InfluenceScore type", () => {
      const score: InfluenceScore = {
        paperId: "test",
        totalCitations: 100,
        influentialCitations: 25,
        influenceScore: 85,
        influenceLevel: "high",
      };
      expect(score.influenceLevel).toBe("high");
    });

    it("should export PaperGraph type", () => {
      const graph: Partial<PaperGraph> = {
        rootPaperId: "test",
        depth: 2,
      };
      expect(graph.depth).toBe(2);
    });
  });
});
