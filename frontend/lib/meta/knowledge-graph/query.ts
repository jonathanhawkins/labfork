/**
 * Knowledge Graph Query Engine
 *
 * Provides a fluent query builder API and query execution for the
 * knowledge graph. Supports:
 * - Filtering and sorting
 * - Pattern matching
 * - Aggregations
 * - Path queries
 * - Similarity queries
 */

import { KnowledgeGraph } from "./graph";
import {
  GraphNode,
  GraphEdge,
  NodeType,
  EdgeType,
  NodeFilter,
  EdgeFilter,
  TraversalOptions,
  PathResult,
  SubgraphResult,
  SimilarityQuery,
  PatternQuery,
  PatternMatch,
  TechniqueNode,
  PaperNode,
  LabNode,
  isTechniqueNode,
  isPaperNode,
  isLabNode,
} from "./types";

// ============================================================================
// Query Result Types
// ============================================================================

export interface QueryResult<T> {
  data: T[];
  total: number;
  page?: number;
  pageSize?: number;
  hasMore?: boolean;
  executionTimeMs: number;
}

export interface AggregationResult {
  groupBy?: string;
  count: number;
  sum?: Record<string, number>;
  avg?: Record<string, number>;
  min?: Record<string, number>;
  max?: Record<string, number>;
}

// ============================================================================
// Query Builder
// ============================================================================

export class NodeQueryBuilder {
  private graph: KnowledgeGraph;
  private filter: NodeFilter = {};
  private sortField?: string;
  private sortDirection: "asc" | "desc" = "desc";
  private pageNum?: number;
  private pageSizeNum?: number;

  constructor(graph: KnowledgeGraph) {
    this.graph = graph;
  }

  /**
   * Filter by node type(s)
   */
  ofType(...types: NodeType[]): this {
    this.filter.types = types;
    return this;
  }

  /**
   * Filter by tags (any match)
   */
  withTags(...tags: string[]): this {
    this.filter.tags = tags;
    return this;
  }

  /**
   * Text search in name and description
   */
  search(query: string): this {
    this.filter.search = query;
    return this;
  }

  /**
   * Filter by creation date range
   */
  createdBetween(after?: string, before?: string): this {
    if (after) this.filter.createdAfter = after;
    if (before) this.filter.createdBefore = before;
    return this;
  }

  /**
   * Custom filter function
   */
  where(predicate: (node: GraphNode) => boolean): this {
    this.filter.custom = predicate;
    return this;
  }

  /**
   * Sort results
   */
  sortBy(field: string, direction: "asc" | "desc" = "desc"): this {
    this.sortField = field;
    this.sortDirection = direction;
    return this;
  }

  /**
   * Paginate results
   */
  paginate(page: number, pageSize: number): this {
    this.pageNum = page;
    this.pageSizeNum = pageSize;
    return this;
  }

  /**
   * Execute the query
   */
  execute(): QueryResult<GraphNode> {
    const startTime = performance.now();

    let nodes = this.graph.findNodes(this.filter);

    // Sort if specified
    if (this.sortField) {
      nodes = this.sortNodes(nodes);
    }

    const total = nodes.length;

    // Paginate if specified
    if (this.pageNum !== undefined && this.pageSizeNum !== undefined) {
      const start = (this.pageNum - 1) * this.pageSizeNum;
      nodes = nodes.slice(start, start + this.pageSizeNum);
    }

    const executionTimeMs = performance.now() - startTime;

    return {
      data: nodes,
      total,
      page: this.pageNum,
      pageSize: this.pageSizeNum,
      hasMore:
        this.pageNum !== undefined &&
        this.pageSizeNum !== undefined &&
        this.pageNum * this.pageSizeNum < total,
      executionTimeMs,
    };
  }

  private sortNodes(nodes: GraphNode[]): GraphNode[] {
    return [...nodes].sort((a, b) => {
      const aVal = this.getNestedValue(a as unknown as Record<string, unknown>, this.sortField!);
      const bVal = this.getNestedValue(b as unknown as Record<string, unknown>, this.sortField!);

      let comparison = 0;

      if (typeof aVal === "string" && typeof bVal === "string") {
        comparison = aVal.localeCompare(bVal);
      } else if (typeof aVal === "number" && typeof bVal === "number") {
        comparison = aVal - bVal;
      } else if (aVal instanceof Date && bVal instanceof Date) {
        comparison = aVal.getTime() - bVal.getTime();
      } else if (typeof aVal === "string" && typeof bVal === "string") {
        // Date strings
        comparison =
          new Date(aVal).getTime() - new Date(bVal).getTime();
      }

      return this.sortDirection === "asc" ? comparison : -comparison;
    });
  }

  private getNestedValue(obj: Record<string, unknown>, path: string): unknown {
    return path.split(".").reduce((current, key) => {
      return current && typeof current === "object"
        ? (current as Record<string, unknown>)[key]
        : undefined;
    }, obj as unknown);
  }
}

export class EdgeQueryBuilder {
  private graph: KnowledgeGraph;
  private filter: EdgeFilter = {};
  private sortField?: string;
  private sortDirection: "asc" | "desc" = "desc";
  private pageNum?: number;
  private pageSizeNum?: number;

  constructor(graph: KnowledgeGraph) {
    this.graph = graph;
  }

  /**
   * Filter by edge type(s)
   */
  ofType(...types: EdgeType[]): this {
    this.filter.types = types;
    return this;
  }

  /**
   * Filter by source node(s)
   */
  fromNodes(...nodeIds: string[]): this {
    this.filter.sourceIds = nodeIds;
    return this;
  }

  /**
   * Filter by target node(s)
   */
  toNodes(...nodeIds: string[]): this {
    this.filter.targetIds = nodeIds;
    return this;
  }

  /**
   * Filter by minimum weight
   */
  minWeight(weight: number): this {
    this.filter.minWeight = weight;
    return this;
  }

  /**
   * Filter by minimum confidence
   */
  minConfidence(confidence: number): this {
    this.filter.minConfidence = confidence;
    return this;
  }

  /**
   * Filter by inferred status
   */
  inferred(isInferred: boolean): this {
    this.filter.isInferred = isInferred;
    return this;
  }

  /**
   * Sort results
   */
  sortBy(field: string, direction: "asc" | "desc" = "desc"): this {
    this.sortField = field;
    this.sortDirection = direction;
    return this;
  }

  /**
   * Paginate results
   */
  paginate(page: number, pageSize: number): this {
    this.pageNum = page;
    this.pageSizeNum = pageSize;
    return this;
  }

  /**
   * Execute the query
   */
  execute(): QueryResult<GraphEdge> {
    const startTime = performance.now();

    let edges = this.graph.findEdges(this.filter);

    // Sort if specified
    if (this.sortField) {
      edges = this.sortEdges(edges);
    }

    const total = edges.length;

    // Paginate if specified
    if (this.pageNum !== undefined && this.pageSizeNum !== undefined) {
      const start = (this.pageNum - 1) * this.pageSizeNum;
      edges = edges.slice(start, start + this.pageSizeNum);
    }

    const executionTimeMs = performance.now() - startTime;

    return {
      data: edges,
      total,
      page: this.pageNum,
      pageSize: this.pageSizeNum,
      hasMore:
        this.pageNum !== undefined &&
        this.pageSizeNum !== undefined &&
        this.pageNum * this.pageSizeNum < total,
      executionTimeMs,
    };
  }

  private sortEdges(edges: GraphEdge[]): GraphEdge[] {
    return [...edges].sort((a, b) => {
      const aVal = (a as unknown as Record<string, unknown>)[this.sortField!];
      const bVal = (b as unknown as Record<string, unknown>)[this.sortField!];

      let comparison = 0;

      if (typeof aVal === "number" && typeof bVal === "number") {
        comparison = aVal - bVal;
      } else if (typeof aVal === "string" && typeof bVal === "string") {
        comparison = aVal.localeCompare(bVal);
      }

      return this.sortDirection === "asc" ? comparison : -comparison;
    });
  }
}

// ============================================================================
// Query Engine
// ============================================================================

export class QueryEngine {
  private graph: KnowledgeGraph;

  constructor(graph: KnowledgeGraph) {
    this.graph = graph;
  }

  /**
   * Create a node query builder
   */
  nodes(): NodeQueryBuilder {
    return new NodeQueryBuilder(this.graph);
  }

  /**
   * Create an edge query builder
   */
  edges(): EdgeQueryBuilder {
    return new EdgeQueryBuilder(this.graph);
  }

  /**
   * Find shortest path between two nodes
   */
  shortestPath(
    startId: string,
    endId: string,
    options?: Partial<TraversalOptions>
  ): PathResult | null {
    return this.graph.findShortestPath(startId, endId, options);
  }

  /**
   * Find all paths between two nodes
   */
  allPaths(
    startId: string,
    endId: string,
    maxDepth: number = 5
  ): PathResult[] {
    return this.graph.findAllPaths(startId, endId, maxDepth);
  }

  /**
   * Traverse the graph from a starting node
   */
  traverse(startId: string, options: TraversalOptions): SubgraphResult {
    return this.graph.traverse(startId, options);
  }

  /**
   * Find similar nodes based on different similarity methods
   */
  findSimilar(query: SimilarityQuery): Array<{ node: GraphNode; similarity: number }> {
    const startNode = this.graph.getNode(query.nodeId);
    if (!startNode) return [];

    const candidates: Array<{ node: GraphNode; similarity: number }> = [];

    switch (query.method) {
      case "embedding":
        return this.findSimilarByEmbedding(startNode, query);

      case "neighbors":
        return this.findSimilarByNeighbors(startNode, query);

      case "structure":
        return this.findSimilarByStructure(startNode, query);

      case "hybrid":
        // Combine multiple methods
        const embeddingSimilar = this.findSimilarByEmbedding(startNode, query);
        const neighborSimilar = this.findSimilarByNeighbors(startNode, query);

        // Merge and average scores
        const scoreMap = new Map<string, number[]>();

        for (const item of embeddingSimilar) {
          scoreMap.set(item.node.id, [item.similarity]);
        }

        for (const item of neighborSimilar) {
          const scores = scoreMap.get(item.node.id) || [];
          scores.push(item.similarity);
          scoreMap.set(item.node.id, scores);
        }

        for (const [nodeId, scores] of Array.from(scoreMap)) {
          const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
          const node = this.graph.getNode(nodeId);
          if (node && avgScore >= (query.minSimilarity || 0)) {
            candidates.push({ node, similarity: avgScore });
          }
        }

        return candidates
          .sort((a, b) => b.similarity - a.similarity)
          .slice(0, query.limit);
    }

    return candidates;
  }

  private findSimilarByEmbedding(
    startNode: GraphNode,
    query: SimilarityQuery
  ): Array<{ node: GraphNode; similarity: number }> {
    if (!startNode.embedding) return [];

    const candidates: Array<{ node: GraphNode; similarity: number }> = [];
    const typeFilter = query.nodeTypes ? new Set(query.nodeTypes) : null;

    for (const node of this.graph.getAllNodes()) {
      if (node.id === startNode.id) continue;
      if (typeFilter && !typeFilter.has(node.type)) continue;
      if (!node.embedding) continue;

      const similarity = this.cosineSimilarity(startNode.embedding, node.embedding);

      if (similarity >= (query.minSimilarity || 0)) {
        candidates.push({ node, similarity });
      }
    }

    return candidates
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, query.limit);
  }

  private findSimilarByNeighbors(
    startNode: GraphNode,
    query: SimilarityQuery
  ): Array<{ node: GraphNode; similarity: number }> {
    const startNeighbors = new Set(
      this.graph.getNeighbors(startNode.id).map((n) => n.id)
    );

    const candidates: Array<{ node: GraphNode; similarity: number }> = [];
    const typeFilter = query.nodeTypes ? new Set(query.nodeTypes) : null;

    for (const node of this.graph.getAllNodes()) {
      if (node.id === startNode.id) continue;
      if (typeFilter && !typeFilter.has(node.type)) continue;

      const nodeNeighbors = new Set(
        this.graph.getNeighbors(node.id).map((n) => n.id)
      );

      // Jaccard similarity of neighbors
      const intersection = new Set(
        Array.from(startNeighbors).filter((id) => nodeNeighbors.has(id))
      );
      const union = new Set([...Array.from(startNeighbors), ...Array.from(nodeNeighbors)]);

      const similarity = union.size > 0 ? intersection.size / union.size : 0;

      if (similarity >= (query.minSimilarity || 0)) {
        candidates.push({ node, similarity });
      }
    }

    return candidates
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, query.limit);
  }

  private findSimilarByStructure(
    startNode: GraphNode,
    query: SimilarityQuery
  ): Array<{ node: GraphNode; similarity: number }> {
    // Structural similarity based on degree and edge types
    const startDegree = this.graph.getNodeDegree(startNode.id);
    const startEdgeTypes = new Set(
      this.graph.getConnectedEdges(startNode.id).map((e) => e.type)
    );

    const candidates: Array<{ node: GraphNode; similarity: number }> = [];
    const typeFilter = query.nodeTypes ? new Set(query.nodeTypes) : null;

    for (const node of this.graph.getAllNodes()) {
      if (node.id === startNode.id) continue;
      if (typeFilter && !typeFilter.has(node.type)) continue;

      const nodeDegree = this.graph.getNodeDegree(node.id);
      const nodeEdgeTypes = new Set(
        this.graph.getConnectedEdges(node.id).map((e) => e.type)
      );

      // Degree similarity (1 - normalized difference)
      const maxDegree = Math.max(startDegree, nodeDegree);
      const degreeSimilarity =
        maxDegree > 0 ? 1 - Math.abs(startDegree - nodeDegree) / maxDegree : 1;

      // Edge type similarity (Jaccard)
      const etIntersection = new Set(
        Array.from(startEdgeTypes).filter((t) => nodeEdgeTypes.has(t))
      );
      const etUnion = new Set([...Array.from(startEdgeTypes), ...Array.from(nodeEdgeTypes)]);
      const edgeTypeSimilarity =
        etUnion.size > 0 ? etIntersection.size / etUnion.size : 1;

      // Combined similarity
      const similarity = (degreeSimilarity + edgeTypeSimilarity) / 2;

      if (similarity >= (query.minSimilarity || 0)) {
        candidates.push({ node, similarity });
      }
    }

    return candidates
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, query.limit);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Execute a pattern matching query
   */
  matchPattern(query: PatternQuery): PatternMatch[] {
    const matches: PatternMatch[] = [];

    // Simple pattern matching implementation
    // For complex patterns, would need more sophisticated algorithm

    if (query.pattern.length === 0) return matches;

    // Start with first pattern node
    const firstPattern = query.pattern[0];
    let candidates = this.graph.getAllNodes();

    // Filter by type if specified
    if (firstPattern.type) {
      candidates = candidates.filter((n) => n.type === firstPattern.type);
    }

    // For each candidate, try to complete the pattern
    for (const startNode of candidates.slice(0, 1000)) {
      // Limit for performance
      const bindings: Record<string, string> = {
        [firstPattern.variable]: startNode.id,
      };
      const matchedNodes: Record<string, GraphNode> = {
        [firstPattern.variable]: startNode,
      };
      const matchedEdges: GraphEdge[] = [];

      if (
        this.matchPatternRecursive(
          query.pattern,
          1,
          bindings,
          matchedNodes,
          matchedEdges
        )
      ) {
        matches.push({
          bindings: { ...bindings },
          nodes: { ...matchedNodes },
          edges: [...matchedEdges],
        });

        if (query.limit && matches.length >= query.limit) {
          break;
        }
      }
    }

    return matches;
  }

  private matchPatternRecursive(
    pattern: PatternQuery["pattern"],
    index: number,
    bindings: Record<string, string>,
    matchedNodes: Record<string, GraphNode>,
    matchedEdges: GraphEdge[]
  ): boolean {
    if (index >= pattern.length) {
      return true;
    }

    const currentPattern = pattern[index];

    // If this variable is already bound, verify it matches
    if (bindings[currentPattern.variable]) {
      const boundNode = this.graph.getNode(bindings[currentPattern.variable]);
      if (!boundNode) return false;
      if (currentPattern.type && boundNode.type !== currentPattern.type)
        return false;

      matchedNodes[currentPattern.variable] = boundNode;
      return this.matchPatternRecursive(
        pattern,
        index + 1,
        bindings,
        matchedNodes,
        matchedEdges
      );
    }

    // Need to find a node that matches this pattern
    // Look at edges from previous nodes
    const previousPatterns = pattern.slice(0, index);
    for (const prevPattern of previousPatterns) {
      const prevNode = matchedNodes[prevPattern.variable];
      if (!prevNode || !prevPattern.edges) continue;

      for (const edgePattern of prevPattern.edges) {
        if (edgePattern.targetVariable !== currentPattern.variable) continue;

        // Find matching edges
        const edges =
          edgePattern.direction === "outgoing"
            ? this.graph.getOutgoingEdges(prevNode.id)
            : this.graph.getIncomingEdges(prevNode.id);

        for (const edge of edges) {
          if (edge.type !== edgePattern.type) continue;

          const targetId =
            edgePattern.direction === "outgoing"
              ? edge.targetId
              : edge.sourceId;
          const targetNode = this.graph.getNode(targetId);

          if (!targetNode) continue;
          if (currentPattern.type && targetNode.type !== currentPattern.type)
            continue;

          // Found a match, try to continue
          bindings[currentPattern.variable] = targetNode.id;
          matchedNodes[currentPattern.variable] = targetNode;
          matchedEdges.push(edge);

          if (
            this.matchPatternRecursive(
              pattern,
              index + 1,
              bindings,
              matchedNodes,
              matchedEdges
            )
          ) {
            return true;
          }

          // Backtrack
          delete bindings[currentPattern.variable];
          delete matchedNodes[currentPattern.variable];
          matchedEdges.pop();
        }
      }
    }

    return false;
  }

  /**
   * Aggregate nodes by a field
   */
  aggregateNodes(
    filter: NodeFilter,
    groupBy?: string,
    metrics?: string[]
  ): AggregationResult[] {
    const nodes = this.graph.findNodes(filter);

    if (!groupBy) {
      // Single aggregation
      return [this.computeAggregation(nodes, metrics)];
    }

    // Group by field
    const groups = new Map<string, GraphNode[]>();

    for (const node of nodes) {
      const value = this.getFieldValue(node as unknown as Record<string, unknown>, groupBy);
      const key = String(value || "null");

      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(node);
    }

    return Array.from(groups.entries()).map(([key, groupNodes]) => ({
      groupBy: key,
      ...this.computeAggregation(groupNodes, metrics),
    }));
  }

  private computeAggregation(
    nodes: GraphNode[],
    metrics?: string[]
  ): AggregationResult {
    const result: AggregationResult = {
      count: nodes.length,
    };

    if (!metrics || metrics.length === 0) {
      return result;
    }

    result.sum = {};
    result.avg = {};
    result.min = {};
    result.max = {};

    for (const metric of metrics) {
      const values = nodes
        .map((n) => this.getFieldValue(n as unknown as Record<string, unknown>, metric))
        .filter((v): v is number => typeof v === "number");

      if (values.length === 0) continue;

      result.sum[metric] = values.reduce((a, b) => a + b, 0);
      result.avg[metric] = result.sum[metric] / values.length;
      result.min[metric] = Math.min(...values);
      result.max[metric] = Math.max(...values);
    }

    return result;
  }

  private getFieldValue(obj: Record<string, unknown>, path: string): unknown {
    return path.split(".").reduce((current, key) => {
      return current && typeof current === "object"
        ? (current as Record<string, unknown>)[key]
        : undefined;
    }, obj as unknown);
  }

  // ==========================================================================
  // Domain-Specific Queries
  // ==========================================================================

  /**
   * Find techniques that could be combined with a given technique
   */
  findCombinable(techniqueId: string): TechniqueNode[] {
    const technique = this.graph.getNode(techniqueId);
    if (!technique || !isTechniqueNode(technique)) return [];

    // Get techniques that are different category but related domain
    const techniques = this.graph.getNodesByType("technique") as TechniqueNode[];

    return techniques.filter((t) => {
      if (t.id === techniqueId) return false;
      if (t.category === technique.category) return false;

      // Check if they share tags or are connected through papers
      const sharedTags = t.tags.filter((tag) =>
        technique.tags.includes(tag)
      );

      if (sharedTags.length > 0) return true;

      // Check for shared papers
      const sharedPapers = t.sourcePaperIds.filter((pid) =>
        technique.sourcePaperIds.includes(pid)
      );

      return sharedPapers.length > 0;
    });
  }

  /**
   * Find influential papers in a domain
   */
  findInfluentialPapers(
    domainId: string,
    limit: number = 10
  ): PaperNode[] {
    const papers = this.graph.getNodesByType("paper") as PaperNode[];

    const domainPapers = papers.filter((p) =>
      p.domainIds.includes(domainId)
    );

    return domainPapers
      .sort((a, b) => b.citationCount - a.citationCount)
      .slice(0, limit);
  }

  /**
   * Find active labs working on a technique
   */
  findLabsImplementing(techniqueId: string): LabNode[] {
    const technique = this.graph.getNode(techniqueId);
    if (!technique || !isTechniqueNode(technique)) return [];

    const labIds = technique.implementingLabIds;

    return labIds
      .map((id) => this.graph.getNode(id))
      .filter((n): n is LabNode => n !== null && isLabNode(n));
  }

  /**
   * Find research gaps (concepts with few implementations)
   */
  findResearchGaps(): Array<{ concept: GraphNode; implementationCount: number }> {
    const concepts = this.graph.getNodesByType("concept");
    const gaps: Array<{ concept: GraphNode; implementationCount: number }> = [];

    for (const concept of concepts) {
      const implementations = this.graph
        .getOutgoingEdges(concept.id)
        .filter((e) => e.type === "implements");

      gaps.push({
        concept,
        implementationCount: implementations.length,
      });
    }

    // Sort by fewest implementations
    return gaps.sort((a, b) => a.implementationCount - b.implementationCount);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a query engine for a graph
 */
export function createQueryEngine(graph: KnowledgeGraph): QueryEngine {
  return new QueryEngine(graph);
}
