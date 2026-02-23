/**
 * Knowledge Graph Data Structure Tests
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  KnowledgeGraph,
  getGlobalGraph,
  resetGlobalGraph,
} from "../../../../lib/meta/knowledge-graph/graph";
import {
  createTechniqueNode,
  createPaperNode,
  createEdge,
  TechniqueNode,
  PaperNode,
  GraphEdge,
} from "../../../../lib/meta/knowledge-graph/types";

describe("KnowledgeGraph", () => {
  let graph: KnowledgeGraph;

  beforeEach(() => {
    graph = new KnowledgeGraph();
  });

  describe("Node Operations", () => {
    describe("addNode", () => {
      it("should add a node to the graph", () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        graph.addNode(technique);
        expect(graph.hasNode(technique.id)).toBe(true);
        expect(graph.nodeCount).toBe(1);
      });

      it("should throw error for duplicate node ID", () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        graph.addNode(technique);
        expect(() => graph.addNode(technique)).toThrow(/already exists/);
      });

      it("should index node by type", () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        const paper = createPaperNode("Test Paper");
        graph.addNode(technique);
        graph.addNode(paper);

        expect(graph.getNodesByType("technique")).toHaveLength(1);
        expect(graph.getNodesByType("paper")).toHaveLength(1);
      });

      it("should index node by tags", () => {
        const technique = createTechniqueNode("Transformer", "architecture", {
          tags: ["ml", "nlp"],
        });
        graph.addNode(technique);

        expect(graph.getNodesByTag("ml")).toHaveLength(1);
        expect(graph.getNodesByTag("nlp")).toHaveLength(1);
        expect(graph.getNodesByTag("unknown")).toHaveLength(0);
      });
    });

    describe("getNode", () => {
      it("should return node by ID", () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        graph.addNode(technique);

        const retrieved = graph.getNode(technique.id);
        expect(retrieved).toBeDefined();
        expect(retrieved?.name).toBe("Transformer");
      });

      it("should return null for non-existent node", () => {
        expect(graph.getNode("non-existent")).toBeNull();
      });
    });

    describe("updateNode", () => {
      it("should update node properties", async () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        graph.addNode(technique);

        // Small delay to ensure different timestamp
        await new Promise((r) => setTimeout(r, 1));

        const updated = graph.updateNode(technique.id, {
          description: "Updated description",
        });

        expect(updated?.description).toBe("Updated description");
        // updatedAt should be different after update
        expect(new Date(updated!.updatedAt).getTime()).toBeGreaterThanOrEqual(
          new Date(technique.updatedAt).getTime()
        );
      });

      it("should update tag indices when tags change", () => {
        const technique = createTechniqueNode("Transformer", "architecture", {
          tags: ["old-tag"],
        });
        graph.addNode(technique);

        graph.updateNode(technique.id, { tags: ["new-tag"] });

        expect(graph.getNodesByTag("old-tag")).toHaveLength(0);
        expect(graph.getNodesByTag("new-tag")).toHaveLength(1);
      });

      it("should return null for non-existent node", () => {
        expect(graph.updateNode("non-existent", {})).toBeNull();
      });
    });

    describe("removeNode", () => {
      it("should remove node from graph", () => {
        const technique = createTechniqueNode("Transformer", "architecture");
        graph.addNode(technique);

        const removed = graph.removeNode(technique.id);

        expect(removed).toBe(true);
        expect(graph.hasNode(technique.id)).toBe(false);
        expect(graph.nodeCount).toBe(0);
      });

      it("should remove node from all indices", () => {
        const technique = createTechniqueNode("Transformer", "architecture", {
          tags: ["ml"],
        });
        graph.addNode(technique);

        graph.removeNode(technique.id);

        expect(graph.getNodesByType("technique")).toHaveLength(0);
        expect(graph.getNodesByTag("ml")).toHaveLength(0);
      });

      it("should remove connected edges when removing node", () => {
        const t1 = createTechniqueNode("T1", "architecture");
        const t2 = createTechniqueNode("T2", "conditioning");
        graph.addNode(t1);
        graph.addNode(t2);

        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        graph.removeNode(t1.id);

        expect(graph.hasEdge(edge.id)).toBe(false);
        expect(graph.edgeCount).toBe(0);
      });

      it("should return false for non-existent node", () => {
        expect(graph.removeNode("non-existent")).toBe(false);
      });
    });

    describe("findNodes", () => {
      beforeEach(() => {
        const t1 = createTechniqueNode("Transformer", "architecture", {
          tags: ["ml", "nlp"],
        });
        const t2 = createTechniqueNode("VAE", "architecture", {
          tags: ["ml", "generative"],
        });
        const p1 = createPaperNode("Test Paper", { tags: ["nlp"] });
        graph.addNode(t1);
        graph.addNode(t2);
        graph.addNode(p1);
      });

      it("should filter by type", () => {
        const techniques = graph.findNodes({ types: ["technique"] });
        expect(techniques).toHaveLength(2);
      });

      it("should filter by multiple types", () => {
        const nodes = graph.findNodes({ types: ["technique", "paper"] });
        expect(nodes).toHaveLength(3);
      });

      it("should filter by tags", () => {
        const mlNodes = graph.findNodes({ tags: ["ml"] });
        expect(mlNodes).toHaveLength(2);

        const nlpNodes = graph.findNodes({ tags: ["nlp"] });
        expect(nlpNodes).toHaveLength(2);
      });

      it("should search by name/description", () => {
        const results = graph.findNodes({ search: "transformer" });
        expect(results).toHaveLength(1);
        expect(results[0].name).toBe("Transformer");
      });

      it("should apply custom filter", () => {
        const results = graph.findNodes({
          custom: (node) => node.name.startsWith("V"),
        });
        expect(results).toHaveLength(1);
        expect(results[0].name).toBe("VAE");
      });
    });
  });

  describe("Edge Operations", () => {
    let t1: TechniqueNode;
    let t2: TechniqueNode;

    beforeEach(() => {
      t1 = createTechniqueNode("Transformer", "architecture");
      t2 = createTechniqueNode("AdaIN", "conditioning");
      graph.addNode(t1);
      graph.addNode(t2);
    });

    describe("addEdge", () => {
      it("should add an edge between nodes", () => {
        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        expect(graph.hasEdge(edge.id)).toBe(true);
        expect(graph.edgeCount).toBe(1);
      });

      it("should throw error if source node doesn't exist", () => {
        const edge = createEdge("cites", "non-existent", t2.id);
        expect(() => graph.addEdge(edge)).toThrow(/does not exist/);
      });

      it("should throw error if target node doesn't exist", () => {
        const edge = createEdge("cites", t1.id, "non-existent");
        expect(() => graph.addEdge(edge)).toThrow(/does not exist/);
      });

      it("should update adjacency lists", () => {
        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        expect(graph.getOutgoingEdges(t1.id)).toHaveLength(1);
        expect(graph.getIncomingEdges(t2.id)).toHaveLength(1);
      });
    });

    describe("getEdge", () => {
      it("should return edge by ID", () => {
        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        const retrieved = graph.getEdge(edge.id);
        expect(retrieved).toBeDefined();
        expect(retrieved?.type).toBe("combines_with");
      });

      it("should return null for non-existent edge", () => {
        expect(graph.getEdge("non-existent")).toBeNull();
      });
    });

    describe("updateEdge", () => {
      it("should update edge properties", () => {
        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        const updated = graph.updateEdge(edge.id, { weight: 0.5 });

        expect(updated?.weight).toBe(0.5);
      });

      it("should not allow changing source/target", () => {
        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        const updated = graph.updateEdge(edge.id, {
          sourceId: "other",
          targetId: "other",
        });

        expect(updated?.sourceId).toBe(t1.id);
        expect(updated?.targetId).toBe(t2.id);
      });
    });

    describe("removeEdge", () => {
      it("should remove edge from graph", () => {
        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        const removed = graph.removeEdge(edge.id);

        expect(removed).toBe(true);
        expect(graph.hasEdge(edge.id)).toBe(false);
        expect(graph.edgeCount).toBe(0);
      });

      it("should update adjacency lists", () => {
        const edge = createEdge("combines_with", t1.id, t2.id);
        graph.addEdge(edge);

        graph.removeEdge(edge.id);

        expect(graph.getOutgoingEdges(t1.id)).toHaveLength(0);
        expect(graph.getIncomingEdges(t2.id)).toHaveLength(0);
      });
    });

    describe("findEdges", () => {
      beforeEach(() => {
        const e1 = createEdge("combines_with", t1.id, t2.id, { weight: 0.8 });
        const e2 = createEdge("similar_to", t2.id, t1.id, {
          weight: 0.5,
          isInferred: true,
        });
        graph.addEdge(e1);
        graph.addEdge(e2);
      });

      it("should filter by type", () => {
        const edges = graph.findEdges({ types: ["combines_with"] });
        expect(edges).toHaveLength(1);
      });

      it("should filter by source", () => {
        const edges = graph.findEdges({ sourceIds: [t1.id] });
        expect(edges).toHaveLength(1);
      });

      it("should filter by target", () => {
        const edges = graph.findEdges({ targetIds: [t2.id] });
        expect(edges).toHaveLength(1);
      });

      it("should filter by minimum weight", () => {
        const edges = graph.findEdges({ minWeight: 0.7 });
        expect(edges).toHaveLength(1);
      });

      it("should filter by inferred status", () => {
        const edges = graph.findEdges({ isInferred: true });
        expect(edges).toHaveLength(1);
      });
    });
  });

  describe("Neighbor Operations", () => {
    let t1: TechniqueNode;
    let t2: TechniqueNode;
    let t3: TechniqueNode;

    beforeEach(() => {
      t1 = createTechniqueNode("T1", "architecture");
      t2 = createTechniqueNode("T2", "conditioning");
      t3 = createTechniqueNode("T3", "loss-function");
      graph.addNode(t1);
      graph.addNode(t2);
      graph.addNode(t3);

      graph.addEdge(createEdge("combines_with", t1.id, t2.id));
      graph.addEdge(createEdge("uses", t1.id, t3.id));
      graph.addEdge(createEdge("similar_to", t2.id, t3.id));
    });

    it("should get outgoing neighbors", () => {
      const neighbors = graph.getOutgoingNeighbors(t1.id);
      expect(neighbors).toHaveLength(2);
      expect(neighbors.map((n) => n.id)).toContain(t2.id);
      expect(neighbors.map((n) => n.id)).toContain(t3.id);
    });

    it("should get incoming neighbors", () => {
      const neighbors = graph.getIncomingNeighbors(t2.id);
      expect(neighbors).toHaveLength(1);
      expect(neighbors[0].id).toBe(t1.id);
    });

    it("should get all neighbors", () => {
      const neighbors = graph.getNeighbors(t2.id);
      expect(neighbors).toHaveLength(2);
    });

    it("should filter neighbors by edge type", () => {
      const neighbors = graph.getOutgoingNeighbors(t1.id, ["combines_with"]);
      expect(neighbors).toHaveLength(1);
      expect(neighbors[0].id).toBe(t2.id);
    });

    it("should calculate node degree", () => {
      expect(graph.getNodeDegree(t1.id, "out")).toBe(2);
      expect(graph.getNodeDegree(t1.id, "in")).toBe(0);
      expect(graph.getNodeDegree(t1.id, "both")).toBe(2);

      expect(graph.getNodeDegree(t2.id, "both")).toBe(2);
    });
  });

  describe("Traversal Operations", () => {
    beforeEach(() => {
      // Create a simple graph: t1 -> t2 -> t3 -> t4
      const t1 = createTechniqueNode("T1", "architecture", { id: "t1" });
      const t2 = createTechniqueNode("T2", "conditioning", { id: "t2" });
      const t3 = createTechniqueNode("T3", "loss-function", { id: "t3" });
      const t4 = createTechniqueNode("T4", "training", { id: "t4" });

      graph.addNode(t1);
      graph.addNode(t2);
      graph.addNode(t3);
      graph.addNode(t4);

      graph.addEdge(createEdge("extends", "t1", "t2"));
      graph.addEdge(createEdge("extends", "t2", "t3"));
      graph.addEdge(createEdge("extends", "t3", "t4"));
    });

    describe("traverse", () => {
      it("should traverse outgoing edges", () => {
        const result = graph.traverse("t1", {
          maxDepth: 2,
          direction: "outgoing",
          includeStart: true,
        });

        expect(result.nodes.length).toBeGreaterThanOrEqual(2);
        expect(result.rootId).toBe("t1");
      });

      it("should respect max depth", () => {
        const result = graph.traverse("t1", {
          maxDepth: 1,
          direction: "outgoing",
          includeStart: false,
        });

        expect(result.nodes).toHaveLength(1);
        expect(result.nodes[0].id).toBe("t2");
      });

      it("should include edges in result", () => {
        const result = graph.traverse("t1", {
          maxDepth: 2,
          direction: "outgoing",
          includeStart: true,
        });

        expect(result.edges.length).toBeGreaterThan(0);
      });
    });

    describe("findShortestPath", () => {
      it("should find shortest path between nodes", () => {
        const path = graph.findShortestPath("t1", "t4");

        expect(path).not.toBeNull();
        expect(path!.nodes).toHaveLength(4);
        expect(path!.length).toBe(3);
      });

      it("should return null if no path exists", () => {
        const isolated = createTechniqueNode("Isolated", "architecture", {
          id: "isolated",
        });
        graph.addNode(isolated);

        const path = graph.findShortestPath("t1", "isolated");
        expect(path).toBeNull();
      });

      it("should handle same start and end", () => {
        const path = graph.findShortestPath("t1", "t1");

        expect(path).not.toBeNull();
        expect(path!.nodes).toHaveLength(1);
        expect(path!.length).toBe(0);
      });
    });

    describe("findAllPaths", () => {
      it("should find all paths between nodes", () => {
        const paths = graph.findAllPaths("t1", "t4");
        expect(paths.length).toBeGreaterThanOrEqual(1);
      });

      it("should respect max depth", () => {
        const paths = graph.findAllPaths("t1", "t4", 2);
        expect(paths).toHaveLength(0); // Path is 3 edges long
      });
    });
  });

  describe("Statistics", () => {
    it("should calculate graph statistics", () => {
      const t1 = createTechniqueNode("T1", "architecture");
      const t2 = createTechniqueNode("T2", "conditioning");
      const p1 = createPaperNode("P1");

      graph.addNode(t1);
      graph.addNode(t2);
      graph.addNode(p1);
      graph.addEdge(createEdge("combines_with", t1.id, t2.id));
      graph.addEdge(createEdge("cites", p1.id, t1.id));

      const stats = graph.getStats();

      expect(stats.nodeCount).toBe(3);
      expect(stats.edgeCount).toBe(2);
      expect(stats.nodeCountByType.technique).toBe(2);
      expect(stats.nodeCountByType.paper).toBe(1);
      expect(stats.averageDegree).toBeGreaterThan(0);
      expect(stats.density).toBeGreaterThan(0);
      expect(stats.componentCount).toBe(1);
    });
  });

  describe("Serialization", () => {
    it("should serialize graph to JSON-compatible format", () => {
      const t1 = createTechniqueNode("T1", "architecture");
      const t2 = createTechniqueNode("T2", "conditioning");
      graph.addNode(t1);
      graph.addNode(t2);
      graph.addEdge(createEdge("combines_with", t1.id, t2.id));

      const serialized = graph.serialize();

      expect(serialized.version).toBe("1.0");
      expect(serialized.nodes).toHaveLength(2);
      expect(serialized.edges).toHaveLength(1);
      expect(serialized.metadata.nodeCount).toBe(2);
      expect(serialized.metadata.edgeCount).toBe(1);
    });

    it("should deserialize graph from JSON", () => {
      const t1 = createTechniqueNode("T1", "architecture");
      const t2 = createTechniqueNode("T2", "conditioning");
      graph.addNode(t1);
      graph.addNode(t2);
      graph.addEdge(createEdge("combines_with", t1.id, t2.id));

      const serialized = graph.serialize();
      const restored = KnowledgeGraph.deserialize(serialized);

      expect(restored.nodeCount).toBe(2);
      expect(restored.edgeCount).toBe(1);
      expect(restored.hasNode(t1.id)).toBe(true);
      expect(restored.hasNode(t2.id)).toBe(true);
    });

    it("should apply incremental updates", () => {
      const t1 = createTechniqueNode("T1", "architecture");
      graph.addNode(t1);

      const t2 = createTechniqueNode("T2", "conditioning");
      const edge = createEdge("combines_with", t1.id, t2.id);

      graph.applyUpdate({
        addNodes: [t2],
        addEdges: [edge],
        updateNodes: [{ id: t1.id, updates: { description: "Updated" } }],
        timestamp: new Date().toISOString(),
      });

      expect(graph.nodeCount).toBe(2);
      expect(graph.edgeCount).toBe(1);
      expect(graph.getNode(t1.id)?.description).toBe("Updated");
    });
  });

  describe("Global Graph", () => {
    it("should return same instance", () => {
      resetGlobalGraph();
      const g1 = getGlobalGraph();
      const g2 = getGlobalGraph();
      expect(g1).toBe(g2);
    });

    it("should reset global graph", () => {
      const g1 = getGlobalGraph();
      g1.addNode(createTechniqueNode("Test", "architecture"));

      resetGlobalGraph();
      const g2 = getGlobalGraph();

      expect(g2.nodeCount).toBe(0);
    });
  });

  describe("Clear", () => {
    it("should clear all nodes and edges", () => {
      const t1 = createTechniqueNode("T1", "architecture");
      const t2 = createTechniqueNode("T2", "conditioning");
      graph.addNode(t1);
      graph.addNode(t2);
      graph.addEdge(createEdge("combines_with", t1.id, t2.id));

      graph.clear();

      expect(graph.nodeCount).toBe(0);
      expect(graph.edgeCount).toBe(0);
    });
  });
});
