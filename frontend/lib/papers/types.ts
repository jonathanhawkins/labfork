/**
 * Paper Ingestion Type Definitions
 *
 * This module defines types for the paper ingestion system that fetches,
 * analyzes, and converts research papers into actionable tasks.
 */

/**
 * Paper source types
 */
export type PaperSource =
  | "arxiv"
  | "semantic-scholar"
  | "github"
  | "pdf"
  | "doi"
  | "papers-with-code"
  | "manual";

/**
 * Paper processing status
 */
export type PaperStatus =
  | "pending" // Just added, not yet fetched
  | "fetching" // Currently fetching metadata
  | "fetched" // Metadata retrieved, awaiting analysis
  | "analyzing" // AI analysis in progress
  | "analyzed" // Analysis complete, awaiting decision
  | "accepted" // User accepted, tasks being generated
  | "rejected" // User rejected
  | "implemented" // Tasks generated and linked
  | "error"; // Error during processing

/**
 * Implementation complexity level
 */
export type ComplexityLevel = "simple" | "moderate" | "complex" | "research";

/**
 * Paper author information
 */
export interface PaperAuthor {
  /** Author name */
  name: string;
  /** Optional affiliation */
  affiliation?: string;
  /** Optional author ID (e.g., Semantic Scholar author ID) */
  authorId?: string;
}

/**
 * Paper metadata from external sources
 */
export interface PaperMetadata {
  /** Unique identifier (arXiv ID, DOI, etc.) */
  id: string;
  /** Paper title */
  title: string;
  /** List of authors */
  authors: PaperAuthor[];
  /** Abstract text */
  abstract: string;
  /** Publication date (ISO string) */
  publishedDate?: string;
  /** Last updated date (ISO string) */
  updatedDate?: string;
  /** Source type */
  source: PaperSource;
  /** URL to the paper */
  url: string;
  /** PDF URL if available */
  pdfUrl?: string;
  /** arXiv categories if applicable */
  categories?: string[];
  /** DOI if available */
  doi?: string;
  /** Citation count if available */
  citationCount?: number;
  /** Venue/conference/journal if available */
  venue?: string;
  /** GitHub repository URL if linked */
  githubUrl?: string;
  /** Additional source-specific metadata */
  sourceMetadata?: Record<string, unknown>;
}

/**
 * Extracted technique from paper
 */
export interface ExtractedTechnique {
  /** Technique name */
  name: string;
  /** Brief description */
  description: string;
  /** Whether this is the main contribution */
  isMainContribution: boolean;
  /** Related techniques or prior work */
  relatedTo?: string[];
}

/**
 * Resource requirement for implementation
 */
export interface ResourceRequirement {
  /** Resource type */
  type: "gpu" | "dataset" | "model" | "library" | "hardware" | "other";
  /** Resource name/description */
  name: string;
  /** Whether it's required or optional */
  required: boolean;
  /** Estimated cost/size if applicable */
  estimate?: string;
  /** Notes about the requirement */
  notes?: string;
}

/**
 * Task breakdown from paper analysis
 */
export interface TaskBreakdown {
  /** Research phase tasks */
  research: {
    title: string;
    description: string;
    estimatedHours?: number;
  };
  /** Implementation phase tasks */
  implementation: {
    title: string;
    description: string;
    estimatedHours?: number;
    dependencies?: string[];
  };
  /** Evaluation phase tasks */
  evaluation: {
    title: string;
    description: string;
    estimatedHours?: number;
    metrics?: string[];
  };
}

/**
 * AI analysis results for a paper
 */
export interface PaperAnalysis {
  /** Relevance score to the domain (0-100) */
  relevanceScore: number;
  /** Explanation of relevance scoring */
  relevanceReason: string;
  /** Key techniques extracted from paper */
  techniques: ExtractedTechnique[];
  /** Main novelty/contribution summary */
  novelty: string;
  /** Implementation complexity assessment */
  complexity: ComplexityLevel;
  /** Complexity explanation */
  complexityReason: string;
  /** Resource requirements */
  resources: ResourceRequirement[];
  /** Proposed task breakdown */
  taskBreakdown: TaskBreakdown;
  /** Potential risks or challenges */
  risks?: string[];
  /** Related papers or techniques to explore */
  relatedWork?: string[];
  /** Analysis timestamp */
  analyzedAt: string;
  /** Domain slug used for analysis */
  domainSlug?: string;
  /** Raw AI response (for debugging) */
  rawResponse?: string;
}

/**
 * Complete paper record
 */
export interface Paper {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Internal unique ID */
  id: string;
  /** Paper metadata */
  metadata: PaperMetadata;
  /** Processing status */
  status: PaperStatus;
  /** AI analysis results (if analyzed) */
  analysis?: PaperAnalysis;
  /** Error message if status is 'error' */
  error?: string;
  /** Domain this paper was added to */
  domainSlug?: string;
  /** User notes/comments */
  notes?: string;
  /** Generated task IDs (if implemented) */
  taskIds?: string[];
  /** When paper was added */
  addedAt: string;
  /** When paper was last updated */
  updatedAt: string;
}

/**
 * Paper input detection result
 */
export interface PaperInputDetection {
  /** Detected source type */
  source: PaperSource;
  /** Extracted identifier */
  identifier: string;
  /** Original input */
  originalInput: string;
  /** Confidence score (0-1) */
  confidence: number;
  /** Normalized URL if applicable */
  normalizedUrl?: string;
}

/**
 * Paper fetch result
 */
export interface PaperFetchResult {
  /** Whether fetch was successful */
  success: boolean;
  /** Paper metadata if successful */
  paper?: Paper;
  /** Error message if failed */
  error?: string;
  /** Whether result was from cache */
  fromCache?: boolean;
}

/**
 * Paper analysis request
 */
export interface PaperAnalysisRequest {
  /** Paper ID to analyze */
  paperId: string;
  /** Optional domain slug for context */
  domainSlug?: string;
  /** Whether to force re-analysis */
  force?: boolean;
}

/**
 * Paper analysis result
 */
export interface PaperAnalysisResult {
  /** Whether analysis was successful */
  success: boolean;
  /** Updated paper with analysis */
  paper?: Paper;
  /** Error message if failed */
  error?: string;
}

/**
 * Paper list filters
 */
export interface PaperListFilters {
  /** Filter by status */
  status?: PaperStatus | PaperStatus[];
  /** Filter by domain */
  domainSlug?: string;
  /** Filter by source */
  source?: PaperSource;
  /** Search query (title, authors, abstract) */
  search?: string;
  /** Minimum relevance score */
  minRelevance?: number;
  /** Sort field */
  sortBy?: "addedAt" | "relevanceScore" | "citationCount" | "publishedDate";
  /** Sort direction */
  sortOrder?: "asc" | "desc";
  /** Limit results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

/**
 * Paper list response
 */
export interface PaperListResponse {
  /** Papers matching filters */
  papers: Paper[];
  /** Total count (before pagination) */
  total: number;
  /** Applied filters */
  filters: PaperListFilters;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if value is a valid PaperSource
 */
export function isPaperSource(value: unknown): value is PaperSource {
  return (
    typeof value === "string" &&
    [
      "arxiv",
      "semantic-scholar",
      "github",
      "pdf",
      "doi",
      "papers-with-code",
      "manual",
    ].includes(value)
  );
}

/**
 * Check if value is a valid PaperStatus
 */
export function isPaperStatus(value: unknown): value is PaperStatus {
  return (
    typeof value === "string" &&
    [
      "pending",
      "fetching",
      "fetched",
      "analyzing",
      "analyzed",
      "accepted",
      "rejected",
      "implemented",
      "error",
    ].includes(value)
  );
}

/**
 * Check if value is a valid ComplexityLevel
 */
export function isComplexityLevel(value: unknown): value is ComplexityLevel {
  return (
    typeof value === "string" &&
    ["simple", "moderate", "complex", "research"].includes(value)
  );
}

/**
 * Check if object is a valid PaperMetadata
 */
export function isPaperMetadata(obj: unknown): obj is PaperMetadata {
  if (!obj || typeof obj !== "object") return false;
  const meta = obj as Record<string, unknown>;

  return (
    typeof meta.id === "string" &&
    typeof meta.title === "string" &&
    Array.isArray(meta.authors) &&
    typeof meta.abstract === "string" &&
    isPaperSource(meta.source) &&
    typeof meta.url === "string"
  );
}

/**
 * Check if object is a valid PaperAnalysis
 */
export function isPaperAnalysis(obj: unknown): obj is PaperAnalysis {
  if (!obj || typeof obj !== "object") return false;
  const analysis = obj as Record<string, unknown>;

  return (
    typeof analysis.relevanceScore === "number" &&
    analysis.relevanceScore >= 0 &&
    analysis.relevanceScore <= 100 &&
    typeof analysis.relevanceReason === "string" &&
    Array.isArray(analysis.techniques) &&
    typeof analysis.novelty === "string" &&
    isComplexityLevel(analysis.complexity) &&
    typeof analysis.complexityReason === "string" &&
    Array.isArray(analysis.resources) &&
    typeof analysis.taskBreakdown === "object" &&
    typeof analysis.analyzedAt === "string"
  );
}

/**
 * Check if object is a valid Paper
 */
export function isPaper(obj: unknown): obj is Paper {
  if (!obj || typeof obj !== "object") return false;
  const paper = obj as Record<string, unknown>;

  return (
    typeof paper.id === "string" &&
    isPaperMetadata(paper.metadata) &&
    isPaperStatus(paper.status) &&
    typeof paper.addedAt === "string" &&
    typeof paper.updatedAt === "string"
  );
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create a new paper with default values
 */
export function createPaper(
  metadata: PaperMetadata,
  domainSlug?: string
): Paper {
  const now = new Date().toISOString();
  return {
    id: generatePaperId(metadata),
    metadata,
    status: "fetched",
    domainSlug,
    addedAt: now,
    updatedAt: now,
  };
}

/**
 * Generate a unique paper ID from metadata
 */
export function generatePaperId(metadata: PaperMetadata): string {
  // Use source + identifier for uniqueness
  const base = `${metadata.source}-${metadata.id}`;
  // Add timestamp for collision avoidance
  const timestamp = Date.now().toString(36);
  return `paper-${base}-${timestamp}`;
}

/**
 * Get status display info
 */
export function getStatusDisplayInfo(status: PaperStatus): {
  label: string;
  color: string;
  bgColor: string;
} {
  switch (status) {
    case "pending":
      return {
        label: "Pending",
        color: "text-foreground-muted",
        bgColor: "bg-foreground-muted/10",
      };
    case "fetching":
      return {
        label: "Fetching",
        color: "text-blue-400",
        bgColor: "bg-blue-400/10",
      };
    case "fetched":
      return {
        label: "Ready",
        color: "text-cyan-400",
        bgColor: "bg-cyan-400/10",
      };
    case "analyzing":
      return {
        label: "Analyzing",
        color: "text-yellow-400",
        bgColor: "bg-yellow-400/10",
      };
    case "analyzed":
      return {
        label: "Analyzed",
        color: "text-purple-400",
        bgColor: "bg-purple-400/10",
      };
    case "accepted":
      return {
        label: "Accepted",
        color: "text-green-400",
        bgColor: "bg-green-400/10",
      };
    case "rejected":
      return {
        label: "Rejected",
        color: "text-red-400",
        bgColor: "bg-red-400/10",
      };
    case "implemented":
      return {
        label: "Implemented",
        color: "text-emerald-400",
        bgColor: "bg-emerald-400/10",
      };
    case "error":
      return {
        label: "Error",
        color: "text-red-500",
        bgColor: "bg-red-500/10",
      };
    default:
      return {
        label: "Unknown",
        color: "text-foreground-subtle",
        bgColor: "bg-foreground-subtle/10",
      };
  }
}

/**
 * Get complexity display info
 */
export function getComplexityDisplayInfo(complexity: ComplexityLevel): {
  label: string;
  color: string;
  bgColor: string;
  hours: string;
} {
  switch (complexity) {
    case "simple":
      return {
        label: "Simple",
        color: "text-green-400",
        bgColor: "bg-green-400/10",
        hours: "1-4 hours",
      };
    case "moderate":
      return {
        label: "Moderate",
        color: "text-yellow-400",
        bgColor: "bg-yellow-400/10",
        hours: "4-16 hours",
      };
    case "complex":
      return {
        label: "Complex",
        color: "text-orange-400",
        bgColor: "bg-orange-400/10",
        hours: "1-3 days",
      };
    case "research":
      return {
        label: "Research",
        color: "text-red-400",
        bgColor: "bg-red-400/10",
        hours: "3+ days",
      };
    default:
      return {
        label: "Unknown",
        color: "text-foreground-subtle",
        bgColor: "bg-foreground-subtle/10",
        hours: "Unknown",
      };
  }
}

/**
 * Get relevance score display info
 */
export function getRelevanceDisplayInfo(score: number): {
  label: string;
  color: string;
  bgColor: string;
} {
  if (score >= 80) {
    return {
      label: "High",
      color: "text-green-400",
      bgColor: "bg-green-400/10",
    };
  }
  if (score >= 60) {
    return {
      label: "Medium",
      color: "text-yellow-400",
      bgColor: "bg-yellow-400/10",
    };
  }
  if (score >= 40) {
    return {
      label: "Low",
      color: "text-orange-400",
      bgColor: "bg-orange-400/10",
    };
  }
  return {
    label: "Minimal",
    color: "text-red-400",
    bgColor: "bg-red-400/10",
  };
}

/**
 * Format author list for display
 */
export function formatAuthors(
  authors: PaperAuthor[],
  maxDisplay: number = 3
): string {
  if (authors.length === 0) return "Unknown authors";
  if (authors.length <= maxDisplay) {
    return authors.map((a) => a.name).join(", ");
  }
  const displayed = authors.slice(0, maxDisplay).map((a) => a.name);
  return `${displayed.join(", ")} et al.`;
}

/**
 * Get source badge info
 */
export function getSourceDisplayInfo(source: PaperSource): {
  label: string;
  color: string;
  bgColor: string;
} {
  switch (source) {
    case "arxiv":
      return {
        label: "arXiv",
        color: "text-red-400",
        bgColor: "bg-red-400/10",
      };
    case "semantic-scholar":
      return {
        label: "S2",
        color: "text-blue-400",
        bgColor: "bg-blue-400/10",
      };
    case "github":
      return {
        label: "GitHub",
        color: "text-purple-400",
        bgColor: "bg-purple-400/10",
      };
    case "pdf":
      return {
        label: "PDF",
        color: "text-orange-400",
        bgColor: "bg-orange-400/10",
      };
    case "doi":
      return {
        label: "DOI",
        color: "text-cyan-400",
        bgColor: "bg-cyan-400/10",
      };
    case "papers-with-code":
      return {
        label: "PWC",
        color: "text-yellow-400",
        bgColor: "bg-yellow-400/10",
      };
    case "manual":
      return {
        label: "Manual",
        color: "text-foreground-muted",
        bgColor: "bg-foreground-muted/10",
      };
    default:
      return {
        label: "Unknown",
        color: "text-foreground-subtle",
        bgColor: "bg-foreground-subtle/10",
      };
  }
}
