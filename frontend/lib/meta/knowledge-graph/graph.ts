/**
 * Knowledge Graph Data Structure
 *
 * Efficient in-memory graph implementation for storing and traversing
 * the knowledge graph. Optimized for:
 * - Fast node/edge lookups (O(1))
 * - Efficient adjacency queries (O(degree))
 * - Memory-efficient storage for large graphs
 * - Thread-safe operations
 */

import {
  GraphNode,
  GraphEdge,
  NodeType,
  EdgeType,
  NodeFilter,
  EdgeFilter,
  TraversalOptions,
  SubgraphResult,
  PathResult,
  GraphStats,
  SerializedGraph,
  GraphUpdate,
  isGraphNode,
  isGraphEdge,
} from "./types";

// ============================================================================
// Adjacency List Types
// ============================================================================

interface AdjacencyEntry {
  /** Edge ID */
  edgeId: string;
  /** Target node ID */
  targetId: string;
  /** Edge type */
  type: EdgeType;
  /** Edge weight */
  weight: number;
}

// ============================================================================
// Knowledge Graph Class
// ============================================================================

export class KnowledgeGraph {
  /** All nodes indexed by ID */
  private nodes: Map<string, GraphNode>;

  /** All edges indexed by ID */
  private edges: Map<string, GraphEdge>;

  /** Outgoing adjacency lists (node ID -> outgoing edges) */
  private outgoing: Map<string, AdjacencyEntry[]>;

  /** Incoming adjacency lists (node ID -> incoming edges) */
  private incoming: Map<string, AdjacencyEntry[]>;

  /** Index: nodes by type */
  private nodesByType: Map<NodeType, Set<string>>;

  /** Index: edges by type */
  private edgesByType: Map<EdgeType, Set<string>>;

  /** Index: nodes by tag */
  private nodesByTag: Map<string, Set<string>>;

  /** Last modification timestamp */
  private lastModified: string;

  constructor() {
    this.nodes = new Map();
    this.edges = new Map();
    this.outgoing = new Map();
    this.incoming = new Map();
    this.nodesByType = new Map();
    this.edgesByType = new Map();
    this.nodesByTag = new Map();
    this.lastModified = new Date().toISOString();
  }

  // ==========================================================================
  // Node Operations
  // ==========================================================================

  /**
   * Add a node to the graph
   */
  addNode(node: GraphNode): void {
    if (this.nodes.has(node.id)) {
      throw new Error(`Node with ID ${node.id} already exists`);
    }

    // Store node
    this.nodes.set(node.id, node);

    // Initialize adjacency lists
    this.outgoing.set(node.id, []);
    this.incoming.set(node.id, []);

    // Update type index
    if (!this.nodesByType.has(node.type)) {
      this.nodesByType.set(node.type, new Set());
    }
    this.nodesByType.get(node.type)!.add(node.id);

    // Update tag index
    for (const tag of node.tags) {
      if (!this.nodesByTag.has(tag)) {
        this.nodesByTag.set(tag, new Set());
      }
      this.nodesByTag.get(tag)!.add(node.id);
    }

    this.lastModified = new Date().toISOString();
  }

  /**
   * Get a node by ID
   */
  getNode(id: string): GraphNode | null {
    return this.nodes.get(id) || null;
  }

  /**
   * Check if a node exists
   */
  hasNode(id: string): boolean {
    return this.nodes.has(id);
  }

  /**
   * Update a node
   */
  updateNode(id: string, updates: Partial<GraphNode>): GraphNode | null {
    const node = this.nodes.get(id);
    if (!node) return null;

    // Handle tag changes
    const oldTags = new Set(node.tags);
    const newTags = updates.tags ? new Set(updates.tags) : oldTags;

    // Remove from old tag indices
    for (const tag of Array.from(oldTags)) {
      if (!newTags.has(tag)) {
        this.nodesByTag.get(tag)?.delete(id);
      }
    }

    // Add to new tag indices
    for (const tag of Array.from(newTags)) {
      if (!oldTags.has(tag)) {
        if (!this.nodesByTag.has(tag)) {
          this.nodesByTag.set(tag, new Set());
        }
        this.nodesByTag.get(tag)!.add(id);
      }
    }

    // Update node (preserve type - can't change node type)
    const updatedNode = {
      ...node,
      ...updates,
      id: node.id, // Preserve ID
      type: node.type, // Preserve type
      updatedAt: new Date().toISOString(),
    } as GraphNode;

    this.nodes.set(id, updatedNode);
    this.lastModified = new Date().toISOString();

    return updatedNode;
  }

  /**
   * Remove a node and all its edges
   */
  removeNode(id: string): boolean {
    const node = this.nodes.get(id);
    if (!node) return false;

    // Remove all edges connected to this node
    const outEdges = this.outgoing.get(id) || [];
    const inEdges = this.incoming.get(id) || [];

    for (const entry of outEdges) {
      this.removeEdge(entry.edgeId);
    }

    for (const entry of inEdges) {
      this.removeEdge(entry.edgeId);
    }

    // Remove from indices
    this.nodesByType.get(node.type)?.delete(id);
    for (const tag of node.tags) {
      this.nodesByTag.get(tag)?.delete(id);
    }

    // Remove adjacency lists
    this.outgoing.delete(id);
    this.incoming.delete(id);

    // Remove node
    this.nodes.delete(id);
    this.lastModified = new Date().toISOString();

    return true;
  }

  /**
   * Get all nodes
   */
  getAllNodes(): GraphNode[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Get nodes by type
   */
  getNodesByType(type: NodeType): GraphNode[] {
    const ids = this.nodesByType.get(type);
    if (!ids) return [];
    return Array.from(ids)
      .map((id) => this.nodes.get(id)!)
      .filter(Boolean);
  }

  /**
   * Get nodes by tag
   */
  getNodesByTag(tag: string): GraphNode[] {
    const ids = this.nodesByTag.get(tag);
    if (!ids) return [];
    return Array.from(ids)
      .map((id) => this.nodes.get(id)!)
      .filter(Boolean);
  }

  /**
   * Find nodes matching a filter
   */
  findNodes(filter: NodeFilter): GraphNode[] {
    let candidates: GraphNode[];

    // Start with type filter if specified (most selective)
    if (filter.types && filter.types.length > 0) {
      candidates = filter.types.flatMap((type) => this.getNodesByType(type));
    } else {
      candidates = this.getAllNodes();
    }

    // Apply tag filter
    if (filter.tags && filter.tags.length > 0) {
      const tagIds = new Set<string>();
      for (const tag of filter.tags) {
        const ids = this.nodesByTag.get(tag);
        if (ids) {
          ids.forEach((id) => tagIds.add(id));
        }
      }
      candidates = candidates.filter((node) => tagIds.has(node.id));
    }

    // Apply text search
    if (filter.search) {
      const searchLower = filter.search.toLowerCase();
      candidates = candidates.filter(
        (node) =>
          node.name.toLowerCase().includes(searchLower) ||
          node.description?.toLowerCase().includes(searchLower)
      );
    }

    // Apply date filters
    if (filter.createdAfter) {
      const after = new Date(filter.createdAfter).getTime();
      candidates = candidates.filter(
        (node) => new Date(node.createdAt).getTime() >= after
      );
    }

    if (filter.createdBefore) {
      const before = new Date(filter.createdBefore).getTime();
      candidates = candidates.filter(
        (node) => new Date(node.createdAt).getTime() <= before
      );
    }

    // Apply custom filter
    if (filter.custom) {
      candidates = candidates.filter(filter.custom);
    }

    return candidates;
  }

  // ==========================================================================
  // Edge Operations
  // ==========================================================================

  /**
   * Add an edge to the graph
   */
  addEdge(edge: GraphEdge): void {
    // Validate nodes exist
    if (!this.nodes.has(edge.sourceId)) {
      throw new Error(`Source node ${edge.sourceId} does not exist`);
    }
    if (!this.nodes.has(edge.targetId)) {
      throw new Error(`Target node ${edge.targetId} does not exist`);
    }

    if (this.edges.has(edge.id)) {
      throw new Error(`Edge with ID ${edge.id} already exists`);
    }

    // Store edge
    this.edges.set(edge.id, edge);

    // Update adjacency lists
    this.outgoing.get(edge.sourceId)!.push({
      edgeId: edge.id,
      targetId: edge.targetId,
      type: edge.type,
      weight: edge.weight,
    });

    this.incoming.get(edge.targetId)!.push({
      edgeId: edge.id,
      targetId: edge.sourceId,
      type: edge.type,
      weight: edge.weight,
    });

    // Update type index
    if (!this.edgesByType.has(edge.type)) {
      this.edgesByType.set(edge.type, new Set());
    }
    this.edgesByType.get(edge.type)!.add(edge.id);

    this.lastModified = new Date().toISOString();
  }

  /**
   * Get an edge by ID
   */
  getEdge(id: string): GraphEdge | null {
    return this.edges.get(id) || null;
  }

  /**
   * Check if an edge exists
   */
  hasEdge(id: string): boolean {
    return this.edges.has(id);
  }

  /**
   * Update an edge
   */
  updateEdge(id: string, updates: Partial<GraphEdge>): GraphEdge | null {
    const edge = this.edges.get(id);
    if (!edge) return null;

    // Can't change source/target IDs
    const updatedEdge: GraphEdge = {
      ...edge,
      ...updates,
      id: edge.id,
      sourceId: edge.sourceId,
      targetId: edge.targetId,
    };

    // Update weight in adjacency lists if changed
    if (updates.weight !== undefined) {
      const outEntries = this.outgoing.get(edge.sourceId) || [];
      const inEntries = this.incoming.get(edge.targetId) || [];

      const outEntry = outEntries.find((e) => e.edgeId === id);
      const inEntry = inEntries.find((e) => e.edgeId === id);

      if (outEntry) outEntry.weight = updates.weight;
      if (inEntry) inEntry.weight = updates.weight;
    }

    this.edges.set(id, updatedEdge);
    this.lastModified = new Date().toISOString();

    return updatedEdge;
  }

  /**
   * Remove an edge
   */
  removeEdge(id: string): boolean {
    const edge = this.edges.get(id);
    if (!edge) return false;

    // Remove from adjacency lists
    const outEntries = this.outgoing.get(edge.sourceId);
    if (outEntries) {
      const idx = outEntries.findIndex((e) => e.edgeId === id);
      if (idx >= 0) outEntries.splice(idx, 1);
    }

    const inEntries = this.incoming.get(edge.targetId);
    if (inEntries) {
      const idx = inEntries.findIndex((e) => e.edgeId === id);
      if (idx >= 0) inEntries.splice(idx, 1);
    }

    // Remove from type index
    this.edgesByType.get(edge.type)?.delete(id);

    // Remove edge
    this.edges.delete(id);
    this.lastModified = new Date().toISOString();

    return true;
  }

  /**
   * Get all edges
   */
  getAllEdges(): GraphEdge[] {
    return Array.from(this.edges.values());
  }

  /**
   * Get edges by type
   */
  getEdgesByType(type: EdgeType): GraphEdge[] {
    const ids = this.edgesByType.get(type);
    if (!ids) return [];
    return Array.from(ids)
      .map((id) => this.edges.get(id)!)
      .filter(Boolean);
  }

  /**
   * Find edges matching a filter
   */
  findEdges(filter: EdgeFilter): GraphEdge[] {
    let candidates: GraphEdge[];

    // Start with type filter if specified
    if (filter.types && filter.types.length > 0) {
      candidates = filter.types.flatMap((type) => this.getEdgesByType(type));
    } else {
      candidates = this.getAllEdges();
    }

    // Apply source filter
    if (filter.sourceIds && filter.sourceIds.length > 0) {
      const sourceSet = new Set(filter.sourceIds);
      candidates = candidates.filter((edge) => sourceSet.has(edge.sourceId));
    }

    // Apply target filter
    if (filter.targetIds && filter.targetIds.length > 0) {
      const targetSet = new Set(filter.targetIds);
      candidates = candidates.filter((edge) => targetSet.has(edge.targetId));
    }

    // Apply weight filter
    if (filter.minWeight !== undefined) {
      candidates = candidates.filter((edge) => edge.weight >= filter.minWeight!);
    }

    // Apply confidence filter
    if (filter.minConfidence !== undefined) {
      candidates = candidates.filter(
        (edge) => edge.confidence >= filter.minConfidence!
      );
    }

    // Apply inferred filter
    if (filter.isInferred !== undefined) {
      candidates = candidates.filter(
        (edge) => edge.isInferred === filter.isInferred
      );
    }

    return candidates;
  }

  /**
   * Get outgoing edges from a node
   */
  getOutgoingEdges(nodeId: string): GraphEdge[] {
    const entries = this.outgoing.get(nodeId) || [];
    return entries.map((e) => this.edges.get(e.edgeId)!).filter(Boolean);
  }

  /**
   * Get incoming edges to a node
   */
  getIncomingEdges(nodeId: string): GraphEdge[] {
    const entries = this.incoming.get(nodeId) || [];
    return entries.map((e) => this.edges.get(e.edgeId)!).filter(Boolean);
  }

  /**
   * Get all edges connected to a node
   */
  getConnectedEdges(nodeId: string): GraphEdge[] {
    return [...this.getOutgoingEdges(nodeId), ...this.getIncomingEdges(nodeId)];
  }

  /**
   * Get edges between two nodes
   */
  getEdgesBetween(sourceId: string, targetId: string): GraphEdge[] {
    const outEntries = this.outgoing.get(sourceId) || [];
    return outEntries
      .filter((e) => e.targetId === targetId)
      .map((e) => this.edges.get(e.edgeId)!)
      .filter(Boolean);
  }

  // ==========================================================================
  // Neighbor Operations
  // ==========================================================================

  /**
   * Get outgoing neighbors of a node
   */
  getOutgoingNeighbors(nodeId: string, edgeTypes?: EdgeType[]): GraphNode[] {
    let entries = this.outgoing.get(nodeId) || [];

    if (edgeTypes && edgeTypes.length > 0) {
      const typeSet = new Set(edgeTypes);
      entries = entries.filter((e) => typeSet.has(e.type));
    }

    return entries.map((e) => this.nodes.get(e.targetId)!).filter(Boolean);
  }

  /**
   * Get incoming neighbors of a node
   */
  getIncomingNeighbors(nodeId: string, edgeTypes?: EdgeType[]): GraphNode[] {
    let entries = this.incoming.get(nodeId) || [];

    if (edgeTypes && edgeTypes.length > 0) {
      const typeSet = new Set(edgeTypes);
      entries = entries.filter((e) => typeSet.has(e.type));
    }

    return entries.map((e) => this.nodes.get(e.targetId)!).filter(Boolean);
  }

  /**
   * Get all neighbors of a node
   */
  getNeighbors(nodeId: string, edgeTypes?: EdgeType[]): GraphNode[] {
    const outNeighbors = this.getOutgoingNeighbors(nodeId, edgeTypes);
    const inNeighbors = this.getIncomingNeighbors(nodeId, edgeTypes);

    // Deduplicate
    const seen = new Set<string>();
    const result: GraphNode[] = [];

    for (const node of [...outNeighbors, ...inNeighbors]) {
      if (!seen.has(node.id)) {
        seen.add(node.id);
        result.push(node);
      }
    }

    return result;
  }

  /**
   * Get node degree (number of edges)
   */
  getNodeDegree(
    nodeId: string,
    direction: "in" | "out" | "both" = "both"
  ): number {
    const outDegree = this.outgoing.get(nodeId)?.length || 0;
    const inDegree = this.incoming.get(nodeId)?.length || 0;

    switch (direction) {
      case "out":
        return outDegree;
      case "in":
        return inDegree;
      case "both":
        return outDegree + inDegree;
    }
  }

  // ==========================================================================
  // Traversal Operations
  // ==========================================================================

  /**
   * Traverse the graph from a starting node using BFS
   */
  traverse(startId: string, options: TraversalOptions): SubgraphResult {
    if (!this.nodes.has(startId)) {
      return { nodes: [], edges: [], rootId: startId };
    }

    const visited = new Set<string>();
    const resultNodes: GraphNode[] = [];
    const resultEdges: GraphEdge[] = [];
    const edgeTypeSet = options.edgeTypes
      ? new Set(options.edgeTypes)
      : null;
    const nodeTypeSet = options.nodeTypes
      ? new Set(options.nodeTypes)
      : null;

    // BFS queue: [nodeId, depth]
    const queue: Array<[string, number]> = [[startId, 0]];
    visited.add(startId);

    if (options.includeStart) {
      const startNode = this.nodes.get(startId)!;
      if (!nodeTypeSet || nodeTypeSet.has(startNode.type)) {
        resultNodes.push(startNode);
      }
    }

    while (queue.length > 0 && (!options.limit || resultNodes.length < options.limit)) {
      const [currentId, depth] = queue.shift()!;

      if (depth >= options.maxDepth) continue;

      // Get edges based on direction
      let entries: AdjacencyEntry[] = [];

      if (options.direction === "outgoing" || options.direction === "both") {
        entries.push(...(this.outgoing.get(currentId) || []));
      }

      if (options.direction === "incoming" || options.direction === "both") {
        entries.push(...(this.incoming.get(currentId) || []));
      }

      for (const entry of entries) {
        // Filter by edge type
        if (edgeTypeSet && !edgeTypeSet.has(entry.type)) continue;

        // Filter by weight
        if (options.minWeight !== undefined && entry.weight < options.minWeight)
          continue;

        const neighborId = entry.targetId;

        // Get full edge
        const edge = this.edges.get(entry.edgeId);
        if (edge) {
          resultEdges.push(edge);
        }

        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          queue.push([neighborId, depth + 1]);

          const neighbor = this.nodes.get(neighborId);
          if (neighbor) {
            // Filter by node type
            if (!nodeTypeSet || nodeTypeSet.has(neighbor.type)) {
              resultNodes.push(neighbor);
            }
          }
        }
      }
    }

    return {
      nodes: resultNodes,
      edges: resultEdges,
      rootId: startId,
    };
  }

  /**
   * Find shortest path between two nodes using BFS
   */
  findShortestPath(
    startId: string,
    endId: string,
    options: Partial<TraversalOptions> = {}
  ): PathResult | null {
    if (!this.nodes.has(startId) || !this.nodes.has(endId)) {
      return null;
    }

    if (startId === endId) {
      const node = this.nodes.get(startId)!;
      return {
        nodes: [node],
        edges: [],
        length: 0,
        totalWeight: 1,
      };
    }

    const edgeTypeSet = options.edgeTypes
      ? new Set(options.edgeTypes)
      : null;
    const direction = options.direction || "both";

    // BFS with path tracking
    const visited = new Set<string>();
    const parent = new Map<string, { nodeId: string; edgeId: string }>();
    const queue: string[] = [startId];
    visited.add(startId);

    while (queue.length > 0) {
      const currentId = queue.shift()!;

      // Get edges based on direction
      let entries: AdjacencyEntry[] = [];

      if (direction === "outgoing" || direction === "both") {
        entries.push(...(this.outgoing.get(currentId) || []));
      }

      if (direction === "incoming" || direction === "both") {
        entries.push(...(this.incoming.get(currentId) || []));
      }

      for (const entry of entries) {
        // Filter by edge type
        if (edgeTypeSet && !edgeTypeSet.has(entry.type)) continue;

        const neighborId = entry.targetId;

        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          parent.set(neighborId, { nodeId: currentId, edgeId: entry.edgeId });

          if (neighborId === endId) {
            // Reconstruct path
            return this.reconstructPath(startId, endId, parent);
          }

          queue.push(neighborId);
        }
      }
    }

    return null; // No path found
  }

  private reconstructPath(
    startId: string,
    endId: string,
    parent: Map<string, { nodeId: string; edgeId: string }>
  ): PathResult {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    let currentId = endId;
    let totalWeight = 1;

    // Build path backwards
    while (currentId !== startId) {
      const node = this.nodes.get(currentId)!;
      nodes.unshift(node);

      const parentInfo = parent.get(currentId)!;
      const edge = this.edges.get(parentInfo.edgeId)!;
      edges.unshift(edge);
      totalWeight *= edge.weight;

      currentId = parentInfo.nodeId;
    }

    // Add start node
    nodes.unshift(this.nodes.get(startId)!);

    return {
      nodes,
      edges,
      length: edges.length,
      totalWeight,
    };
  }

  /**
   * Find all paths between two nodes (up to a maximum depth)
   */
  findAllPaths(
    startId: string,
    endId: string,
    maxDepth: number = 5
  ): PathResult[] {
    if (!this.nodes.has(startId) || !this.nodes.has(endId)) {
      return [];
    }

    const results: PathResult[] = [];
    const visited = new Set<string>();

    const dfs = (
      currentId: string,
      path: string[],
      edgePath: string[],
      depth: number
    ) => {
      if (depth > maxDepth) return;

      if (currentId === endId) {
        // Found a path
        const nodes = path.map((id) => this.nodes.get(id)!);
        const edges = edgePath.map((id) => this.edges.get(id)!);
        const totalWeight = edges.reduce((w, e) => w * e.weight, 1);

        results.push({
          nodes,
          edges,
          length: edges.length,
          totalWeight,
        });
        return;
      }

      visited.add(currentId);

      const entries = this.outgoing.get(currentId) || [];
      for (const entry of entries) {
        if (!visited.has(entry.targetId)) {
          dfs(
            entry.targetId,
            [...path, entry.targetId],
            [...edgePath, entry.edgeId],
            depth + 1
          );
        }
      }

      visited.delete(currentId);
    };

    dfs(startId, [startId], [], 0);
    return results;
  }

  // ==========================================================================
  // Statistics
  // ==========================================================================

  /**
   * Get graph statistics
   */
  getStats(): GraphStats {
    const nodeCount = this.nodes.size;
    const edgeCount = this.edges.size;

    // Node counts by type
    const nodeCountByType: Record<NodeType, number> = {
      technique: 0,
      domain: 0,
      paper: 0,
      lab: 0,
      result: 0,
      concept: 0,
    };

    for (const [type, ids] of Array.from(this.nodesByType)) {
      nodeCountByType[type] = ids.size;
    }

    // Edge counts by type
    const edgeCountByType: Partial<Record<EdgeType, number>> = {};
    for (const [type, ids] of Array.from(this.edgesByType)) {
      edgeCountByType[type] = ids.size;
    }

    // Average degree
    let totalDegree = 0;
    for (const [, entries] of Array.from(this.outgoing)) {
      totalDegree += entries.length;
    }
    const averageDegree = nodeCount > 0 ? (totalDegree * 2) / nodeCount : 0;

    // Graph density
    const maxEdges = nodeCount * (nodeCount - 1);
    const density = maxEdges > 0 ? edgeCount / maxEdges : 0;

    // Connected components (simple union-find)
    const componentCount = this.countConnectedComponents();

    return {
      nodeCount,
      nodeCountByType,
      edgeCount,
      edgeCountByType: edgeCountByType as Record<EdgeType, number>,
      averageDegree,
      density,
      componentCount,
      lastUpdated: this.lastModified,
    };
  }

  private countConnectedComponents(): number {
    const visited = new Set<string>();
    let count = 0;

    for (const nodeId of Array.from(this.nodes.keys())) {
      if (!visited.has(nodeId)) {
        // BFS to mark all reachable nodes
        const queue = [nodeId];
        while (queue.length > 0) {
          const current = queue.shift()!;
          if (visited.has(current)) continue;
          visited.add(current);

          const neighbors = this.getNeighbors(current);
          for (const neighbor of neighbors) {
            if (!visited.has(neighbor.id)) {
              queue.push(neighbor.id);
            }
          }
        }
        count++;
      }
    }

    return count;
  }

  // ==========================================================================
  // Serialization
  // ==========================================================================

  /**
   * Serialize the graph to a JSON-compatible format
   */
  serialize(): SerializedGraph {
    return {
      version: "1.0",
      nodes: this.getAllNodes(),
      edges: this.getAllEdges(),
      metadata: {
        createdAt: this.lastModified, // Using lastModified as approx creation time
        lastModified: this.lastModified,
        nodeCount: this.nodes.size,
        edgeCount: this.edges.size,
      },
    };
  }

  /**
   * Deserialize a graph from JSON
   */
  static deserialize(data: SerializedGraph): KnowledgeGraph {
    const graph = new KnowledgeGraph();

    // Validate and add nodes
    for (const node of data.nodes) {
      if (isGraphNode(node)) {
        graph.addNode(node);
      } else {
        console.warn("Invalid node during deserialization:", node);
      }
    }

    // Validate and add edges
    for (const edge of data.edges) {
      if (isGraphEdge(edge)) {
        try {
          graph.addEdge(edge);
        } catch (e) {
          console.warn("Failed to add edge during deserialization:", edge, e);
        }
      } else {
        console.warn("Invalid edge during deserialization:", edge);
      }
    }

    graph.lastModified = data.metadata?.lastModified || new Date().toISOString();

    return graph;
  }

  /**
   * Apply an incremental update to the graph
   */
  applyUpdate(update: GraphUpdate): void {
    // Remove edges first (before removing nodes)
    if (update.removeEdges) {
      for (const edgeId of update.removeEdges) {
        this.removeEdge(edgeId);
      }
    }

    // Remove nodes
    if (update.removeNodes) {
      for (const nodeId of update.removeNodes) {
        this.removeNode(nodeId);
      }
    }

    // Add nodes
    if (update.addNodes) {
      for (const node of update.addNodes) {
        if (isGraphNode(node)) {
          this.addNode(node);
        }
      }
    }

    // Update nodes
    if (update.updateNodes) {
      for (const { id, updates } of update.updateNodes) {
        this.updateNode(id, updates);
      }
    }

    // Add edges
    if (update.addEdges) {
      for (const edge of update.addEdges) {
        if (isGraphEdge(edge)) {
          try {
            this.addEdge(edge);
          } catch (e) {
            console.warn("Failed to add edge during update:", edge, e);
          }
        }
      }
    }

    // Update edges
    if (update.updateEdges) {
      for (const { id, updates } of update.updateEdges) {
        this.updateEdge(id, updates);
      }
    }

    this.lastModified = update.timestamp;
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  /**
   * Clear the entire graph
   */
  clear(): void {
    this.nodes.clear();
    this.edges.clear();
    this.outgoing.clear();
    this.incoming.clear();
    this.nodesByType.clear();
    this.edgesByType.clear();
    this.nodesByTag.clear();
    this.lastModified = new Date().toISOString();
  }

  /**
   * Get the number of nodes
   */
  get nodeCount(): number {
    return this.nodes.size;
  }

  /**
   * Get the number of edges
   */
  get edgeCount(): number {
    return this.edges.size;
  }

  /**
   * Get last modification timestamp
   */
  get lastModifiedAt(): string {
    return this.lastModified;
  }
}

// ============================================================================
// Singleton Instance (for global graph)
// ============================================================================

let globalGraph: KnowledgeGraph | null = null;

/**
 * Get the global knowledge graph instance
 */
export function getGlobalGraph(): KnowledgeGraph {
  if (!globalGraph) {
    globalGraph = new KnowledgeGraph();
  }
  return globalGraph;
}

/**
 * Reset the global graph (mainly for testing)
 */
export function resetGlobalGraph(): void {
  globalGraph = new KnowledgeGraph();
}
