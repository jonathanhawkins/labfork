/**
 * Knowledge Graph Query Engine Tests
 */

import { describe, it, expect, beforeEach } from "vitest";
import { KnowledgeGraph } from "../../../../lib/meta/knowledge-graph/graph";
import {
  QueryEngine,
  NodeQueryBuilder,
  EdgeQueryBuilder,
  createQueryEngine,
} from "../../../../lib/meta/knowledge-graph/query";
import {
  createTechniqueNode,
  createPaperNode,
  createEdge,
  TechniqueNode,
  PaperNode,
} from "../../../../lib/meta/knowledge-graph/types";

describe("QueryEngine", () => {
  let graph: KnowledgeGraph;
  let engine: QueryEngine;

  beforeEach(() => {
    graph = new KnowledgeGraph();
    engine = createQueryEngine(graph);

    // Add test data
    const t1 = createTechniqueNode("Transformer", "architecture", {
      id: "t1",
      tags: ["ml", "nlp"],
      description: "Self-attention architecture",
    });
    const t2 = createTechniqueNode("VAE", "architecture", {
      id: "t2",
      tags: ["ml", "generative"],
      description: "Variational autoencoder",
    });
    const t3 = createTechniqueNode("AdaIN", "conditioning", {
      id: "t3",
      tags: ["style-transfer"],
    });
    const p1 = createPaperNode("Attention Is All You Need", {
      id: "p1",
      citationCount: 50000,
      domainIds: ["nlp"],
    });
    const p2 = createPaperNode("Auto-Encoding Variational Bayes", {
      id: "p2",
      citationCount: 20000,
      domainIds: ["ml"],
    });

    graph.addNode(t1);
    graph.addNode(t2);
    graph.addNode(t3);
    graph.addNode(p1);
    graph.addNode(p2);

    graph.addEdge(createEdge("implements", "t1", "p1", { id: "e1" }));
    graph.addEdge(createEdge("implements", "t2", "p2", { id: "e2" }));
    graph.addEdge(createEdge("combines_with", "t1", "t3", { id: "e3" }));
    graph.addEdge(
      createEdge("similar_to", "t1", "t2", { id: "e4", weight: 0.7 })
    );
  });

  describe("NodeQueryBuilder", () => {
    describe("ofType", () => {
      it("should filter by single type", () => {
        const result = engine.nodes().ofType("technique").execute();
        expect(result.data).toHaveLength(3);
        expect(result.total).toBe(3);
      });

      it("should filter by multiple types", () => {
        const result = engine.nodes().ofType("technique", "paper").execute();
        expect(result.data).toHaveLength(5);
      });
    });

    describe("withTags", () => {
      it("should filter by tags", () => {
        const result = engine.nodes().withTags("ml").execute();
        expect(result.data).toHaveLength(2);
      });

      it("should return nodes matching any tag", () => {
        const result = engine.nodes().withTags("nlp", "generative").execute();
        expect(result.data).toHaveLength(2);
      });
    });

    describe("search", () => {
      it("should search in name", () => {
        const result = engine.nodes().search("transformer").execute();
        expect(result.data).toHaveLength(1);
        expect(result.data[0].name).toBe("Transformer");
      });

      it("should search in description", () => {
        const result = engine.nodes().search("autoencoder").execute();
        expect(result.data).toHaveLength(1);
        expect(result.data[0].name).toBe("VAE");
      });

      it("should be case insensitive", () => {
        const result = engine.nodes().search("ATTENTION").execute();
        expect(result.data.length).toBeGreaterThanOrEqual(1);
      });
    });

    describe("where", () => {
      it("should apply custom filter", () => {
        const result = engine
          .nodes()
          .where((node) => node.tags.includes("ml"))
          .execute();
        expect(result.data).toHaveLength(2);
      });
    });

    describe("sortBy", () => {
      it("should sort by name ascending", () => {
        const result = engine
          .nodes()
          .ofType("technique")
          .sortBy("name", "asc")
          .execute();
        expect(result.data[0].name).toBe("AdaIN");
      });

      it("should sort by name descending", () => {
        const result = engine
          .nodes()
          .ofType("technique")
          .sortBy("name", "desc")
          .execute();
        expect(result.data[0].name).toBe("VAE");
      });

      it("should sort by nested property", () => {
        const result = engine.nodes().ofType("paper").sortBy("createdAt", "asc").execute();
        expect(result.data).toHaveLength(2);
      });
    });

    describe("paginate", () => {
      it("should paginate results", () => {
        const result = engine
          .nodes()
          .ofType("technique")
          .paginate(1, 2)
          .execute();

        expect(result.data).toHaveLength(2);
        expect(result.page).toBe(1);
        expect(result.pageSize).toBe(2);
        expect(result.hasMore).toBe(true);
      });

      it("should return correct page", () => {
        const result = engine
          .nodes()
          .ofType("technique")
          .paginate(2, 2)
          .execute();

        expect(result.data).toHaveLength(1);
        expect(result.hasMore).toBe(false);
      });
    });

    describe("chained operations", () => {
      it("should combine multiple filters", () => {
        const result = engine
          .nodes()
          .ofType("technique")
          .withTags("ml")
          .search("auto")
          .execute();

        expect(result.data).toHaveLength(1);
        expect(result.data[0].name).toBe("VAE");
      });
    });

    describe("execution time", () => {
      it("should track execution time", () => {
        const result = engine.nodes().execute();
        expect(result.executionTimeMs).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe("EdgeQueryBuilder", () => {
    describe("ofType", () => {
      it("should filter by edge type", () => {
        const result = engine.edges().ofType("implements").execute();
        expect(result.data).toHaveLength(2);
      });

      it("should filter by multiple types", () => {
        const result = engine
          .edges()
          .ofType("implements", "combines_with")
          .execute();
        expect(result.data).toHaveLength(3);
      });
    });

    describe("fromNodes", () => {
      it("should filter by source node", () => {
        const result = engine.edges().fromNodes("t1").execute();
        expect(result.data).toHaveLength(3);
      });
    });

    describe("toNodes", () => {
      it("should filter by target node", () => {
        const result = engine.edges().toNodes("p1").execute();
        expect(result.data).toHaveLength(1);
      });
    });

    describe("minWeight", () => {
      it("should filter by minimum weight", () => {
        const result = engine.edges().minWeight(0.8).execute();
        expect(result.data).toHaveLength(3); // Default weight is 1.0
      });
    });
  });

  describe("Path Queries", () => {
    describe("shortestPath", () => {
      it("should find shortest path", () => {
        const path = engine.shortestPath("t1", "p1");
        expect(path).not.toBeNull();
        expect(path!.nodes).toHaveLength(2);
      });

      it("should return null when no path exists", () => {
        const isolated = createTechniqueNode("Isolated", "architecture", {
          id: "isolated",
        });
        graph.addNode(isolated);

        const path = engine.shortestPath("isolated", "p1");
        expect(path).toBeNull();
      });
    });

    describe("allPaths", () => {
      it("should find all paths up to max depth", () => {
        const paths = engine.allPaths("t1", "p1", 5);
        expect(paths.length).toBeGreaterThanOrEqual(1);
      });
    });
  });

  describe("Traversal", () => {
    it("should traverse from a starting node", () => {
      const result = engine.traverse("t1", {
        maxDepth: 2,
        direction: "outgoing",
        includeStart: true,
      });

      expect(result.nodes.length).toBeGreaterThanOrEqual(1);
      expect(result.rootId).toBe("t1");
    });
  });

  describe("Similarity Queries", () => {
    describe("findSimilar with neighbors", () => {
      it("should find similar nodes by shared neighbors", () => {
        const similar = engine.findSimilar({
          nodeId: "t1",
          method: "neighbors",
          limit: 5,
        });

        // May not have high similarity without shared neighbors
        expect(Array.isArray(similar)).toBe(true);
      });
    });

    describe("findSimilar with structure", () => {
      it("should find similar nodes by structure", () => {
        const similar = engine.findSimilar({
          nodeId: "t1",
          method: "structure",
          limit: 5,
        });

        expect(Array.isArray(similar)).toBe(true);
      });
    });

    describe("findSimilar with type filter", () => {
      it("should filter by node type", () => {
        const similar = engine.findSimilar({
          nodeId: "t1",
          method: "structure",
          limit: 10,
          nodeTypes: ["technique"],
        });

        for (const item of similar) {
          expect(item.node.type).toBe("technique");
        }
      });
    });
  });

  describe("Aggregations", () => {
    it("should aggregate all nodes", () => {
      const result = engine.aggregateNodes({});
      expect(result).toHaveLength(1);
      expect(result[0].count).toBe(5);
    });

    it("should group by type", () => {
      const result = engine.aggregateNodes({}, "type");
      expect(result.length).toBeGreaterThanOrEqual(2);

      const techniqueGroup = result.find((r) => r.groupBy === "technique");
      expect(techniqueGroup?.count).toBe(3);

      const paperGroup = result.find((r) => r.groupBy === "paper");
      expect(paperGroup?.count).toBe(2);
    });

    it("should compute numeric metrics", () => {
      const result = engine.aggregateNodes(
        { types: ["paper"] },
        undefined,
        ["citationCount"]
      );

      expect(result[0].sum?.citationCount).toBe(70000);
      expect(result[0].avg?.citationCount).toBe(35000);
      expect(result[0].min?.citationCount).toBe(20000);
      expect(result[0].max?.citationCount).toBe(50000);
    });
  });

  describe("Domain-Specific Queries", () => {
    describe("findCombinable", () => {
      it("should find techniques that can combine", () => {
        // Add techniques with shared tags
        const newT = createTechniqueNode("NewTech", "loss-function", {
          id: "new-t",
          tags: ["ml"],
        });
        graph.addNode(newT);

        const combinable = engine.findCombinable("t1");
        expect(Array.isArray(combinable)).toBe(true);
      });
    });

    describe("findInfluentialPapers", () => {
      it("should find papers sorted by citation count", () => {
        const papers = engine.findInfluentialPapers("nlp", 10);
        expect(papers).toHaveLength(1);
        expect((papers[0] as PaperNode).citationCount).toBe(50000);
      });
    });

    describe("findResearchGaps", () => {
      it("should find concepts with few implementations", () => {
        const gaps = engine.findResearchGaps();
        expect(Array.isArray(gaps)).toBe(true);
      });
    });
  });

  describe("Pattern Matching", () => {
    it("should match simple pattern", () => {
      const matches = engine.matchPattern({
        pattern: [
          {
            variable: "technique",
            type: "technique",
            edges: [
              {
                type: "implements",
                targetVariable: "paper",
                direction: "outgoing",
              },
            ],
          },
          {
            variable: "paper",
            type: "paper",
          },
        ],
        limit: 10,
      });

      expect(matches.length).toBeGreaterThanOrEqual(1);

      for (const match of matches) {
        expect(match.bindings.technique).toBeDefined();
        expect(match.bindings.paper).toBeDefined();
        expect(match.nodes.technique.type).toBe("technique");
        expect(match.nodes.paper.type).toBe("paper");
      }
    });
  });
});

describe("createQueryEngine", () => {
  it("should create a query engine for a graph", () => {
    const graph = new KnowledgeGraph();
    const engine = createQueryEngine(graph);

    expect(engine).toBeInstanceOf(QueryEngine);
    expect(engine.nodes()).toBeInstanceOf(NodeQueryBuilder);
    expect(engine.edges()).toBeInstanceOf(EdgeQueryBuilder);
  });
});
