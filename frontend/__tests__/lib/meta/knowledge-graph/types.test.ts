/**
 * Knowledge Graph Types Tests
 */

import { describe, it, expect } from "vitest";
import {
  // Type guards
  isNodeType,
  isEdgeType,
  isTechniqueNode,
  isDomainNode,
  isPaperNode,
  isLabNode,
  isResultNode,
  isConceptNode,
  isGraphNode,
  isGraphEdge,
  // Factory functions
  generateNodeId,
  generateEdgeId,
  createBaseNode,
  createTechniqueNode,
  createPaperNode,
  createEdge,
  // Display helpers
  getNodeTypeColor,
  getEdgeTypeColor,
  getNodeTypeLabel,
  getEdgeTypeLabel,
  // Types
  NodeType,
  EdgeType,
  TechniqueNode,
  PaperNode,
  GraphEdge,
} from "../../../../lib/meta/knowledge-graph/types";

describe("Knowledge Graph Types", () => {
  describe("Type Guards", () => {
    describe("isNodeType", () => {
      it("should return true for valid node types", () => {
        expect(isNodeType("technique")).toBe(true);
        expect(isNodeType("domain")).toBe(true);
        expect(isNodeType("paper")).toBe(true);
        expect(isNodeType("lab")).toBe(true);
        expect(isNodeType("result")).toBe(true);
        expect(isNodeType("concept")).toBe(true);
      });

      it("should return false for invalid node types", () => {
        expect(isNodeType("invalid")).toBe(false);
        expect(isNodeType("")).toBe(false);
        expect(isNodeType(null)).toBe(false);
        expect(isNodeType(undefined)).toBe(false);
        expect(isNodeType(123)).toBe(false);
      });
    });

    describe("isEdgeType", () => {
      it("should return true for valid edge types", () => {
        expect(isEdgeType("derived_from")).toBe(true);
        expect(isEdgeType("similar_to")).toBe(true);
        expect(isEdgeType("combines_with")).toBe(true);
        expect(isEdgeType("transfers_to")).toBe(true);
        expect(isEdgeType("implements")).toBe(true);
        expect(isEdgeType("cites")).toBe(true);
        expect(isEdgeType("belongs_to")).toBe(true);
        expect(isEdgeType("produces")).toBe(true);
        expect(isEdgeType("uses")).toBe(true);
        expect(isEdgeType("improves")).toBe(true);
        expect(isEdgeType("extends")).toBe(true);
        expect(isEdgeType("competes_with")).toBe(true);
        expect(isEdgeType("requires")).toBe(true);
        expect(isEdgeType("enables")).toBe(true);
        expect(isEdgeType("related_to")).toBe(true);
      });

      it("should return false for invalid edge types", () => {
        expect(isEdgeType("invalid")).toBe(false);
        expect(isEdgeType("")).toBe(false);
        expect(isEdgeType(null)).toBe(false);
      });
    });

    describe("Node Type Guards", () => {
      it("should identify technique nodes", () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        expect(isTechniqueNode(technique)).toBe(true);
        expect(isDomainNode(technique)).toBe(false);
      });

      it("should identify paper nodes", () => {
        const paper = createPaperNode("Test Paper");
        expect(isPaperNode(paper)).toBe(true);
        expect(isTechniqueNode(paper)).toBe(false);
      });
    });

    describe("isGraphNode", () => {
      it("should return true for valid nodes", () => {
        const node = createTechniqueNode("Test", "architecture");
        expect(isGraphNode(node)).toBe(true);
      });

      it("should return false for invalid nodes", () => {
        expect(isGraphNode(null)).toBe(false);
        expect(isGraphNode({})).toBe(false);
        expect(isGraphNode({ id: "test" })).toBe(false);
        expect(isGraphNode({ id: "test", type: "invalid" })).toBe(false);
      });
    });

    describe("isGraphEdge", () => {
      it("should return true for valid edges", () => {
        const edge = createEdge("cites", "source", "target");
        expect(isGraphEdge(edge)).toBe(true);
      });

      it("should return false for invalid edges", () => {
        expect(isGraphEdge(null)).toBe(false);
        expect(isGraphEdge({})).toBe(false);
        expect(isGraphEdge({ id: "test" })).toBe(false);
        expect(isGraphEdge({ id: "test", type: "invalid" })).toBe(false);
      });
    });
  });

  describe("ID Generation", () => {
    describe("generateNodeId", () => {
      it("should generate unique IDs", async () => {
        const id1 = generateNodeId("technique");
        // Small delay to ensure different timestamp
        await new Promise((r) => setTimeout(r, 1));
        const id2 = generateNodeId("technique");
        expect(id1).not.toBe(id2);
      });

      it("should include type in ID", () => {
        const id = generateNodeId("technique");
        expect(id.startsWith("technique")).toBe(true);
      });

      it("should include hint when provided", () => {
        const id = generateNodeId("technique", "Transformer");
        // The hint is used as prefix: technique-transforme- (truncated to 10 chars)
        expect(id.startsWith("technique-transforme")).toBe(true);
      });
    });

    describe("generateEdgeId", () => {
      it("should generate unique edge IDs", async () => {
        const id1 = generateEdgeId("cites", "a", "b");
        // Small delay to ensure different timestamp
        await new Promise((r) => setTimeout(r, 1));
        const id2 = generateEdgeId("cites", "a", "b");
        expect(id1).not.toBe(id2);
      });

      it("should include edge type", () => {
        const id = generateEdgeId("cites", "source", "target");
        expect(id.startsWith("cites")).toBe(true);
      });
    });
  });

  describe("Factory Functions", () => {
    describe("createBaseNode", () => {
      it("should create node with defaults", () => {
        const node = createBaseNode("technique", "Test");
        expect(node.type).toBe("technique");
        expect(node.name).toBe("Test");
        expect(node.tags).toEqual([]);
        expect(node.metadata).toEqual({});
        expect(node.createdAt).toBeDefined();
        expect(node.updatedAt).toBeDefined();
      });

      it("should accept custom options", () => {
        const node = createBaseNode("paper", "Test Paper", {
          tags: ["ml", "nlp"],
          description: "A test paper",
        });
        expect(node.tags).toEqual(["ml", "nlp"]);
        expect(node.description).toBe("A test paper");
      });
    });

    describe("createTechniqueNode", () => {
      it("should create technique node with required fields", () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        expect(technique.type).toBe("technique");
        expect(technique.name).toBe("Transformer");
        expect(technique.category).toBe("architecture");
        expect(technique.complexity).toBe("moderate");
        expect(technique.hasImplementation).toBe(false);
        expect(technique.sourcePaperIds).toEqual([]);
        expect(technique.implementingLabIds).toEqual([]);
      });

      it("should accept optional fields", () => {
        const technique = createTechniqueNode("Transformer", "architecture", {
          complexity: "complex",
          hasImplementation: true,
          architecture: "encoder-decoder",
          conditioning: "cross-attention",
          sourcePaperIds: ["paper-1"],
          metrics: { quality: 95 },
        });
        expect(technique.complexity).toBe("complex");
        expect(technique.hasImplementation).toBe(true);
        expect(technique.architecture).toBe("encoder-decoder");
        expect(technique.conditioning).toBe("cross-attention");
        expect(technique.sourcePaperIds).toEqual(["paper-1"]);
        expect(technique.metrics?.quality).toBe(95);
      });
    });

    describe("createPaperNode", () => {
      it("should create paper node with required fields", () => {
        const paper = createPaperNode("Attention Is All You Need");
        expect(paper.type).toBe("paper");
        expect(paper.name).toBe("Attention Is All You Need");
        expect(paper.title).toBe("Attention Is All You Need");
        expect(paper.authors).toEqual([]);
        expect(paper.citationCount).toBe(0);
        expect(paper.influentialCitationCount).toBe(0);
        expect(paper.abstract).toBe("");
        expect(paper.contributions).toEqual([]);
        expect(paper.domainIds).toEqual([]);
      });

      it("should accept optional fields", () => {
        const paper = createPaperNode("Test Paper", {
          authors: ["Author 1", "Author 2"],
          year: 2023,
          arxivId: "2301.12345",
          citationCount: 100,
          domainIds: ["domain-1"],
        });
        expect(paper.authors).toEqual(["Author 1", "Author 2"]);
        expect(paper.year).toBe(2023);
        expect(paper.arxivId).toBe("2301.12345");
        expect(paper.citationCount).toBe(100);
        expect(paper.domainIds).toEqual(["domain-1"]);
      });
    });

    describe("createEdge", () => {
      it("should create edge with defaults", () => {
        const edge = createEdge("cites", "paper-1", "paper-2");
        expect(edge.type).toBe("cites");
        expect(edge.sourceId).toBe("paper-1");
        expect(edge.targetId).toBe("paper-2");
        expect(edge.weight).toBe(1.0);
        expect(edge.confidence).toBe(1.0);
        expect(edge.isInferred).toBe(false);
        expect(edge.properties).toEqual({});
        expect(edge.createdAt).toBeDefined();
      });

      it("should accept optional fields", () => {
        const edge = createEdge("similar_to", "t1", "t2", {
          weight: 0.8,
          confidence: 0.9,
          isInferred: true,
          properties: { method: "embedding" },
        });
        expect(edge.weight).toBe(0.8);
        expect(edge.confidence).toBe(0.9);
        expect(edge.isInferred).toBe(true);
        expect(edge.properties).toEqual({ method: "embedding" });
      });

      it("should create edge with evidence", () => {
        const edge = createEdge("derived_from", "t1", "t2", {
          evidence: [
            {
              type: "analysis",
              description: "Based on paper analysis",
              confidence: 0.8,
              timestamp: new Date().toISOString(),
            },
          ],
        });
        expect(edge.evidence).toHaveLength(1);
        expect(edge.evidence![0].type).toBe("analysis");
      });
    });
  });

  describe("Display Helpers", () => {
    describe("getNodeTypeColor", () => {
      it("should return correct colors for each type", () => {
        expect(getNodeTypeColor("technique")).toBe("#3b82f6");
        expect(getNodeTypeColor("domain")).toBe("#8b5cf6");
        expect(getNodeTypeColor("paper")).toBe("#ef4444");
        expect(getNodeTypeColor("lab")).toBe("#22c55e");
        expect(getNodeTypeColor("result")).toBe("#f59e0b");
        expect(getNodeTypeColor("concept")).toBe("#06b6d4");
      });

      it("should return gray for unknown type", () => {
        expect(getNodeTypeColor("unknown" as NodeType)).toBe("#6b7280");
      });
    });

    describe("getEdgeTypeColor", () => {
      it("should return correct colors for common edge types", () => {
        expect(getEdgeTypeColor("derived_from")).toBe("#3b82f6");
        expect(getEdgeTypeColor("similar_to")).toBe("#8b5cf6");
        expect(getEdgeTypeColor("combines_with")).toBe("#22c55e");
        expect(getEdgeTypeColor("cites")).toBe("#ef4444");
      });
    });

    describe("getNodeTypeLabel", () => {
      it("should return correct labels", () => {
        expect(getNodeTypeLabel("technique")).toBe("Technique");
        expect(getNodeTypeLabel("domain")).toBe("Domain");
        expect(getNodeTypeLabel("paper")).toBe("Paper");
        expect(getNodeTypeLabel("lab")).toBe("Lab");
        expect(getNodeTypeLabel("result")).toBe("Result");
        expect(getNodeTypeLabel("concept")).toBe("Concept");
      });
    });

    describe("getEdgeTypeLabel", () => {
      it("should convert snake_case to Title Case", () => {
        expect(getEdgeTypeLabel("derived_from")).toBe("Derived From");
        expect(getEdgeTypeLabel("similar_to")).toBe("Similar To");
        expect(getEdgeTypeLabel("combines_with")).toBe("Combines With");
      });
    });
  });

  describe("Type Structures", () => {
    it("should enforce TechniqueNode structure", () => {
      const technique: TechniqueNode = {
        id: "tech-1",
        type: "technique",
        name: "Test Technique",
        tags: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        metadata: {},
        category: "architecture",
        complexity: "moderate",
        hasImplementation: false,
        sourcePaperIds: [],
        implementingLabIds: [],
      };

      expect(technique.type).toBe("technique");
      expect(technique.category).toBe("architecture");
    });

    it("should enforce PaperNode structure", () => {
      const paper: PaperNode = {
        id: "paper-1",
        type: "paper",
        name: "Test Paper",
        title: "Test Paper",
        tags: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        metadata: {},
        authors: [],
        url: "",
        citationCount: 0,
        influentialCitationCount: 0,
        abstract: "",
        contributions: [],
        domainIds: [],
      };

      expect(paper.type).toBe("paper");
      expect(paper.title).toBe("Test Paper");
    });

    it("should enforce GraphEdge structure", () => {
      const edge: GraphEdge = {
        id: "edge-1",
        type: "cites",
        sourceId: "paper-1",
        targetId: "paper-2",
        weight: 1.0,
        confidence: 1.0,
        isInferred: false,
        properties: {},
        createdAt: new Date().toISOString(),
      };

      expect(edge.type).toBe("cites");
      expect(edge.sourceId).toBe("paper-1");
      expect(edge.targetId).toBe("paper-2");
    });
  });
});
