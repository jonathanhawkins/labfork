/**
 * Knowledge Graph Type Definitions
 *
 * This module defines the schema for the knowledge graph that enables
 * meta-research intelligence by connecting techniques, papers, labs,
 * and concepts across the entire platform.
 */

// ============================================================================
// Node Types
// ============================================================================

/**
 * All possible node types in the knowledge graph
 */
export type NodeType =
  | "technique"
  | "domain"
  | "paper"
  | "lab"
  | "result"
  | "concept";

/**
 * Base properties for all nodes
 */
export interface BaseNode {
  /** Unique identifier for the node */
  id: string;
  /** Node type discriminator */
  type: NodeType;
  /** Human-readable name */
  name: string;
  /** Description of the node */
  description?: string;
  /** Tags for categorization */
  tags: string[];
  /** Embedding vector for semantic similarity (optional) */
  embedding?: number[];
  /** Creation timestamp */
  createdAt: string;
  /** Last update timestamp */
  updatedAt: string;
  /** Source metadata */
  metadata: Record<string, unknown>;
}

/**
 * Technique node - represents a research technique or method
 */
export interface TechniqueNode extends BaseNode {
  type: "technique";
  /** The primary category of the technique */
  category: TechniqueCategory;
  /** Research domains this technique applies to */
  domains: string[];
  /** Architecture type if applicable */
  architecture?: string;
  /** Conditioning method if applicable */
  conditioning?: string;
  /** Loss function if applicable */
  loss?: string;
  /** Data flow pattern */
  dataFlow?: string;
  /** Implementation complexity */
  complexity: "simple" | "moderate" | "complex" | "research";
  /** Whether there's a working implementation */
  hasImplementation: boolean;
  /** Implementation quality if implemented */
  implementationQuality?: "prototype" | "stable" | "production";
  /** Papers that introduced or describe this technique */
  sourcePaperIds: string[];
  /** Labs that have implemented this technique */
  implementingLabIds: string[];
  /** Performance metrics */
  metrics?: TechniqueMetrics;
}

export type TechniqueCategory =
  | "architecture"
  | "conditioning"
  | "loss-function"
  | "training"
  | "inference"
  | "preprocessing"
  | "postprocessing"
  | "evaluation"
  | "data-augmentation"
  | "other";

export interface TechniqueMetrics {
  /** Quality score (0-100) */
  quality?: number;
  /** Speed relative to baseline (1.0 = same) */
  speedFactor?: number;
  /** Memory usage relative to baseline */
  memoryFactor?: number;
  /** Other domain-specific metrics */
  [key: string]: number | undefined;
}

/**
 * Domain node - represents a research domain
 */
export interface DomainNode extends BaseNode {
  type: "domain";
  /** URL-safe slug */
  slug: string;
  /** Parent domain if this is a subdomain */
  parentDomainId?: string;
  /** Associated arXiv categories */
  arxivCategories: string[];
  /** Key research areas within this domain */
  researchAreas: string[];
  /** Number of active labs in this domain */
  labCount: number;
  /** Number of papers in this domain */
  paperCount: number;
}

/**
 * Paper node - represents a research paper
 */
export interface PaperNode extends BaseNode {
  type: "paper";
  /** Paper title (same as name) */
  title: string;
  /** Authors list */
  authors: string[];
  /** Publication venue */
  venue?: string;
  /** Publication year */
  year?: number;
  /** arXiv ID if available */
  arxivId?: string;
  /** DOI if available */
  doi?: string;
  /** URL to paper */
  url: string;
  /** Citation count */
  citationCount: number;
  /** Influential citation count */
  influentialCitationCount: number;
  /** Abstract text */
  abstract: string;
  /** Main contributions */
  contributions: string[];
  /** Domains this paper belongs to */
  domainIds: string[];
}

/**
 * Lab node - represents a research lab instance
 */
export interface LabNode extends BaseNode {
  type: "lab";
  /** Lab slug */
  slug: string;
  /** Owner username */
  ownerUsername: string;
  /** Domain this lab operates in */
  domainId: string;
  /** Research focus areas */
  focusAreas: string[];
  /** Number of completed experiments */
  experimentCount: number;
  /** Number of published results */
  resultCount: number;
  /** Activity level */
  activityLevel: "inactive" | "low" | "moderate" | "high" | "very-high";
  /** Lab quality score */
  qualityScore?: number;
}

/**
 * Result node - represents a research result or finding
 */
export interface ResultNode extends BaseNode {
  type: "result";
  /** Type of result */
  resultType: "model" | "benchmark" | "finding" | "dataset" | "demo";
  /** Lab that produced this result */
  labId: string;
  /** Techniques used to achieve this result */
  techniqueIds: string[];
  /** Papers associated with this result */
  paperIds: string[];
  /** Metrics achieved */
  metrics: Record<string, number>;
  /** Whether this result has been reproduced */
  isReproduced: boolean;
  /** Reproduction count */
  reproductionCount: number;
}

/**
 * Concept node - represents an abstract concept or idea
 */
export interface ConceptNode extends BaseNode {
  type: "concept";
  /** Related terms and synonyms */
  aliases: string[];
  /** Definition of the concept */
  definition: string;
  /** Domains where this concept is relevant */
  domainIds: string[];
  /** Example applications */
  examples: string[];
}

/**
 * Union type for all node types
 */
export type GraphNode =
  | TechniqueNode
  | DomainNode
  | PaperNode
  | LabNode
  | ResultNode
  | ConceptNode;

// ============================================================================
// Edge Types
// ============================================================================

/**
 * All possible edge types in the knowledge graph
 */
export type EdgeType =
  | "derived_from"
  | "similar_to"
  | "combines_with"
  | "transfers_to"
  | "implements"
  | "cites"
  | "belongs_to"
  | "produces"
  | "uses"
  | "improves"
  | "extends"
  | "competes_with"
  | "requires"
  | "enables"
  | "related_to";

/**
 * Edge in the knowledge graph
 */
export interface GraphEdge {
  /** Unique identifier for the edge */
  id: string;
  /** Edge type */
  type: EdgeType;
  /** Source node ID */
  sourceId: string;
  /** Target node ID */
  targetId: string;
  /** Edge weight (0-1) for weighted operations */
  weight: number;
  /** Confidence score (0-1) for inferred edges */
  confidence: number;
  /** Whether this edge was manually created or inferred */
  isInferred: boolean;
  /** Evidence for this edge */
  evidence?: EdgeEvidence[];
  /** Additional properties */
  properties: Record<string, unknown>;
  /** Creation timestamp */
  createdAt: string;
}

/**
 * Evidence supporting an edge
 */
export interface EdgeEvidence {
  /** Type of evidence */
  type: "citation" | "implementation" | "analysis" | "manual" | "llm";
  /** Source of evidence (paper ID, lab ID, etc.) */
  sourceId?: string;
  /** Description of evidence */
  description: string;
  /** Confidence of this specific evidence */
  confidence: number;
  /** When this evidence was recorded */
  timestamp: string;
}

// ============================================================================
// Graph Operations
// ============================================================================

/**
 * Filter for node queries
 */
export interface NodeFilter {
  /** Filter by node type(s) */
  types?: NodeType[];
  /** Filter by tags (any match) */
  tags?: string[];
  /** Text search query */
  search?: string;
  /** Filter by creation date range */
  createdAfter?: string;
  createdBefore?: string;
  /** Filter by metadata properties */
  metadata?: Record<string, unknown>;
  /** Custom filter function */
  custom?: (node: GraphNode) => boolean;
}

/**
 * Filter for edge queries
 */
export interface EdgeFilter {
  /** Filter by edge type(s) */
  types?: EdgeType[];
  /** Filter by source node ID(s) */
  sourceIds?: string[];
  /** Filter by target node ID(s) */
  targetIds?: string[];
  /** Minimum weight threshold */
  minWeight?: number;
  /** Minimum confidence threshold */
  minConfidence?: number;
  /** Filter by inferred status */
  isInferred?: boolean;
}

/**
 * Options for graph traversal
 */
export interface TraversalOptions {
  /** Maximum depth to traverse */
  maxDepth: number;
  /** Edge types to follow */
  edgeTypes?: EdgeType[];
  /** Node types to include */
  nodeTypes?: NodeType[];
  /** Direction of traversal */
  direction: "outgoing" | "incoming" | "both";
  /** Whether to include the starting node */
  includeStart: boolean;
  /** Maximum number of nodes to return */
  limit?: number;
  /** Minimum edge weight to follow */
  minWeight?: number;
}

/**
 * Result of a path query
 */
export interface PathResult {
  /** Nodes in the path (ordered) */
  nodes: GraphNode[];
  /** Edges connecting the nodes (ordered) */
  edges: GraphEdge[];
  /** Total path length */
  length: number;
  /** Total path weight (product of edge weights) */
  totalWeight: number;
}

/**
 * Result of a subgraph query
 */
export interface SubgraphResult {
  /** All nodes in the subgraph */
  nodes: GraphNode[];
  /** All edges in the subgraph */
  edges: GraphEdge[];
  /** Root node ID if applicable */
  rootId?: string;
}

/**
 * Graph statistics
 */
export interface GraphStats {
  /** Total number of nodes */
  nodeCount: number;
  /** Node count by type */
  nodeCountByType: Record<NodeType, number>;
  /** Total number of edges */
  edgeCount: number;
  /** Edge count by type */
  edgeCountByType: Record<EdgeType, number>;
  /** Average node degree */
  averageDegree: number;
  /** Graph density (edges / possible edges) */
  density: number;
  /** Number of connected components */
  componentCount: number;
  /** Last update timestamp */
  lastUpdated: string;
}

// ============================================================================
// Query Types
// ============================================================================

/**
 * Query for finding similar nodes
 */
export interface SimilarityQuery {
  /** Node to find similar nodes for */
  nodeId: string;
  /** Similarity method */
  method: "embedding" | "neighbors" | "structure" | "hybrid";
  /** Number of results to return */
  limit: number;
  /** Minimum similarity threshold (0-1) */
  minSimilarity?: number;
  /** Node types to include in results */
  nodeTypes?: NodeType[];
}

/**
 * Query for pattern matching
 */
export interface PatternQuery {
  /** Pattern to match (simplified graph pattern language) */
  pattern: PatternNode[];
  /** Variable bindings */
  bindings?: Record<string, string>;
  /** Maximum results */
  limit?: number;
}

/**
 * Node in a pattern query
 */
export interface PatternNode {
  /** Variable name for this node */
  variable: string;
  /** Required node type */
  type?: NodeType;
  /** Required properties */
  properties?: Record<string, unknown>;
  /** Edges from this node */
  edges?: PatternEdge[];
}

/**
 * Edge in a pattern query
 */
export interface PatternEdge {
  /** Edge type */
  type: EdgeType;
  /** Target node variable */
  targetVariable: string;
  /** Direction */
  direction: "outgoing" | "incoming";
}

/**
 * Result of a pattern match
 */
export interface PatternMatch {
  /** Variable bindings to actual node IDs */
  bindings: Record<string, string>;
  /** Matched nodes */
  nodes: Record<string, GraphNode>;
  /** Matched edges */
  edges: GraphEdge[];
  /** Match score if applicable */
  score?: number;
}

// ============================================================================
// Serialization Types
// ============================================================================

/**
 * Serialized graph format for persistence
 */
export interface SerializedGraph {
  /** Version of the serialization format */
  version: string;
  /** All nodes */
  nodes: GraphNode[];
  /** All edges */
  edges: GraphEdge[];
  /** Graph metadata */
  metadata: {
    createdAt: string;
    lastModified: string;
    nodeCount: number;
    edgeCount: number;
  };
}

/**
 * Incremental update to the graph
 */
export interface GraphUpdate {
  /** Nodes to add */
  addNodes?: GraphNode[];
  /** Nodes to update (by ID) */
  updateNodes?: Array<{ id: string; updates: Partial<GraphNode> }>;
  /** Node IDs to remove */
  removeNodes?: string[];
  /** Edges to add */
  addEdges?: GraphEdge[];
  /** Edges to update (by ID) */
  updateEdges?: Array<{ id: string; updates: Partial<GraphEdge> }>;
  /** Edge IDs to remove */
  removeEdges?: string[];
  /** Update timestamp */
  timestamp: string;
}

// ============================================================================
// Type Guards
// ============================================================================

export function isNodeType(value: unknown): value is NodeType {
  return (
    typeof value === "string" &&
    ["technique", "domain", "paper", "lab", "result", "concept"].includes(value)
  );
}

export function isEdgeType(value: unknown): value is EdgeType {
  return (
    typeof value === "string" &&
    [
      "derived_from",
      "similar_to",
      "combines_with",
      "transfers_to",
      "implements",
      "cites",
      "belongs_to",
      "produces",
      "uses",
      "improves",
      "extends",
      "competes_with",
      "requires",
      "enables",
      "related_to",
    ].includes(value)
  );
}

export function isTechniqueNode(node: GraphNode): node is TechniqueNode {
  return node.type === "technique";
}

export function isDomainNode(node: GraphNode): node is DomainNode {
  return node.type === "domain";
}

export function isPaperNode(node: GraphNode): node is PaperNode {
  return node.type === "paper";
}

export function isLabNode(node: GraphNode): node is LabNode {
  return node.type === "lab";
}

export function isResultNode(node: GraphNode): node is ResultNode {
  return node.type === "result";
}

export function isConceptNode(node: GraphNode): node is ConceptNode {
  return node.type === "concept";
}

export function isGraphNode(obj: unknown): obj is GraphNode {
  if (!obj || typeof obj !== "object") return false;
  const node = obj as Record<string, unknown>;
  return (
    typeof node.id === "string" &&
    isNodeType(node.type) &&
    typeof node.name === "string" &&
    Array.isArray(node.tags) &&
    typeof node.createdAt === "string" &&
    typeof node.updatedAt === "string"
  );
}

export function isGraphEdge(obj: unknown): obj is GraphEdge {
  if (!obj || typeof obj !== "object") return false;
  const edge = obj as Record<string, unknown>;
  return (
    typeof edge.id === "string" &&
    isEdgeType(edge.type) &&
    typeof edge.sourceId === "string" &&
    typeof edge.targetId === "string" &&
    typeof edge.weight === "number" &&
    typeof edge.confidence === "number"
  );
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Generate a unique ID for a node
 */
export function generateNodeId(type: NodeType, hint?: string): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  const prefix = hint
    ? `${type}-${hint.substring(0, 10).toLowerCase().replace(/\s+/g, "-")}`
    : type;
  return `${prefix}-${timestamp}-${random}`;
}

/**
 * Generate a unique ID for an edge
 */
export function generateEdgeId(
  type: EdgeType,
  sourceId: string,
  targetId: string
): string {
  const timestamp = Date.now().toString(36);
  return `${type}-${sourceId.substring(0, 8)}-${targetId.substring(0, 8)}-${timestamp}`;
}

/**
 * Create a base node with defaults
 */
export function createBaseNode(
  type: NodeType,
  name: string,
  options: Partial<BaseNode> = {}
): BaseNode {
  const now = new Date().toISOString();
  return {
    id: options.id || generateNodeId(type, name),
    type,
    name,
    description: options.description,
    tags: options.tags || [],
    embedding: options.embedding,
    createdAt: options.createdAt || now,
    updatedAt: options.updatedAt || now,
    metadata: options.metadata || {},
  };
}

/**
 * Create a technique node
 */
export function createTechniqueNode(
  name: string,
  category: TechniqueCategory,
  options: Partial<Omit<TechniqueNode, "type" | "name" | "category">> = {}
): TechniqueNode {
  const base = createBaseNode("technique", name, options);
  return {
    ...base,
    type: "technique",
    category,
    domains: options.domains || [],
    architecture: options.architecture,
    conditioning: options.conditioning,
    loss: options.loss,
    dataFlow: options.dataFlow,
    complexity: options.complexity || "moderate",
    hasImplementation: options.hasImplementation || false,
    implementationQuality: options.implementationQuality,
    sourcePaperIds: options.sourcePaperIds || [],
    implementingLabIds: options.implementingLabIds || [],
    metrics: options.metrics,
  };
}

/**
 * Create a paper node
 */
export function createPaperNode(
  title: string,
  options: Partial<Omit<PaperNode, "type" | "title" | "name">> = {}
): PaperNode {
  const base = createBaseNode("paper", title, options);
  return {
    ...base,
    type: "paper",
    title,
    authors: options.authors || [],
    venue: options.venue,
    year: options.year,
    arxivId: options.arxivId,
    doi: options.doi,
    url: options.url || "",
    citationCount: options.citationCount || 0,
    influentialCitationCount: options.influentialCitationCount || 0,
    abstract: options.abstract || "",
    contributions: options.contributions || [],
    domainIds: options.domainIds || [],
  };
}

/**
 * Create an edge
 */
export function createEdge(
  type: EdgeType,
  sourceId: string,
  targetId: string,
  options: Partial<Omit<GraphEdge, "type" | "sourceId" | "targetId">> = {}
): GraphEdge {
  return {
    id: options.id || generateEdgeId(type, sourceId, targetId),
    type,
    sourceId,
    targetId,
    weight: options.weight ?? 1.0,
    confidence: options.confidence ?? 1.0,
    isInferred: options.isInferred ?? false,
    evidence: options.evidence,
    properties: options.properties || {},
    createdAt: options.createdAt || new Date().toISOString(),
  };
}

// ============================================================================
// Display Helpers
// ============================================================================

/**
 * Get display color for a node type
 */
export function getNodeTypeColor(type: NodeType): string {
  switch (type) {
    case "technique":
      return "#3b82f6"; // blue
    case "domain":
      return "#8b5cf6"; // purple
    case "paper":
      return "#ef4444"; // red
    case "lab":
      return "#22c55e"; // green
    case "result":
      return "#f59e0b"; // amber
    case "concept":
      return "#06b6d4"; // cyan
    default:
      return "#6b7280"; // gray
  }
}

/**
 * Get display color for an edge type
 */
export function getEdgeTypeColor(type: EdgeType): string {
  switch (type) {
    case "derived_from":
      return "#3b82f6";
    case "similar_to":
      return "#8b5cf6";
    case "combines_with":
      return "#22c55e";
    case "transfers_to":
      return "#f59e0b";
    case "implements":
      return "#06b6d4";
    case "cites":
      return "#ef4444";
    default:
      return "#6b7280";
  }
}

/**
 * Get display label for a node type
 */
export function getNodeTypeLabel(type: NodeType): string {
  switch (type) {
    case "technique":
      return "Technique";
    case "domain":
      return "Domain";
    case "paper":
      return "Paper";
    case "lab":
      return "Lab";
    case "result":
      return "Result";
    case "concept":
      return "Concept";
    default:
      return "Unknown";
  }
}

/**
 * Get display label for an edge type
 */
export function getEdgeTypeLabel(type: EdgeType): string {
  return type
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
