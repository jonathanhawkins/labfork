/**
 * Result Types and Schema
 *
 * Defines types for research results in the social layer.
 * Results showcase findings, models, demos, and comparisons.
 */

/**
 * Result type categories
 */
export type ResultType = "model" | "demo" | "finding" | "comparison" | "dataset" | "paper";

/**
 * Result visibility
 */
export type ResultVisibility = "public" | "private" | "unlisted";

/**
 * Result status
 */
export type ResultStatus = "draft" | "published" | "archived";

/**
 * Media type for attachments
 */
export type MediaType = "image" | "audio" | "video" | "chart" | "code" | "file";

/**
 * Media attachment
 */
export interface ResultMedia {
  /** Unique media ID */
  id: string;
  /** Media type */
  type: MediaType;
  /** URL to the media file */
  url: string;
  /** Optional thumbnail URL */
  thumbnailUrl?: string;
  /** Alt text / description */
  alt?: string;
  /** File name */
  filename?: string;
  /** File size in bytes */
  size?: number;
  /** MIME type */
  mimeType?: string;
  /** Duration in seconds (for audio/video) */
  duration?: number;
  /** Width in pixels (for images/videos) */
  width?: number;
  /** Height in pixels (for images/videos) */
  height?: number;
  /** Order index for display */
  order: number;
}

/**
 * Performance metric
 */
export interface ResultMetric {
  /** Metric name (e.g., "MOS", "WER", "RTF") */
  name: string;
  /** Metric value */
  value: number;
  /** Unit (e.g., "%", "ms", "score") */
  unit?: string;
  /** Higher is better? */
  higherIsBetter?: boolean;
  /** Baseline comparison value */
  baseline?: number;
  /** Description */
  description?: string;
}

/**
 * Result metrics collection
 */
export interface ResultMetrics {
  /** Primary metrics */
  primary: ResultMetric[];
  /** Secondary/additional metrics */
  secondary?: ResultMetric[];
  /** Comparison with baseline */
  baselineModel?: string;
  /** Benchmark dataset used */
  benchmarkDataset?: string;
}

/**
 * Model-specific metadata
 */
export interface ModelMetadata {
  /** Model name/identifier */
  modelName: string;
  /** Model version */
  version?: string;
  /** Number of parameters */
  parameters?: number;
  /** Model size in bytes */
  size?: number;
  /** Architecture type */
  architecture?: string;
  /** Training dataset */
  trainingDataset?: string;
  /** Checkpoint URL */
  checkpointUrl?: string;
  /** Config URL */
  configUrl?: string;
  /** License */
  license?: string;
}

/**
 * Demo-specific metadata
 */
export interface DemoMetadata {
  /** Demo URL */
  demoUrl: string;
  /** Is the demo live/interactive? */
  isInteractive: boolean;
  /** Technologies used */
  technologies?: string[];
  /** Source code URL */
  sourceUrl?: string;
}

/**
 * Finding-specific metadata
 */
export interface FindingMetadata {
  /** Key insight summary */
  keyInsight: string;
  /** Hypothesis tested */
  hypothesis?: string;
  /** Methodology used */
  methodology?: string;
  /** Limitations */
  limitations?: string[];
  /** Future work suggestions */
  futureWork?: string[];
}

/**
 * Comparison-specific metadata
 */
export interface ComparisonMetadata {
  /** Models being compared */
  modelsCompared: string[];
  /** Comparison criteria */
  criteria: string[];
  /** Winner (if applicable) */
  winner?: string;
  /** Summary of findings */
  summary: string;
}

/**
 * Result author information
 */
export interface ResultAuthor {
  /** User ID */
  id: string;
  /** Username */
  username: string;
  /** Display name */
  displayName: string;
  /** Avatar URL */
  avatar?: string;
}

/**
 * Result engagement stats
 */
export interface ResultStats {
  /** Number of likes */
  likes: number;
  /** Number of comments */
  comments: number;
  /** Number of shares */
  shares: number;
  /** Number of views */
  views: number;
  /** Number of saves/bookmarks */
  saves: number;
}

/**
 * Complete result record
 */
export interface Result {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Unique result ID */
  id: string;
  /** Result type */
  type: ResultType;
  /** Title */
  title: string;
  /** Short description */
  description: string;
  /** Full content (markdown) */
  content?: string;
  /** Result visibility */
  visibility: ResultVisibility;
  /** Result status */
  status: ResultStatus;
  /** Author */
  author: ResultAuthor;
  /** Lab ID this result belongs to */
  labId: string;
  /** Task ID this result is associated with (optional) */
  taskId?: string;
  /** Media attachments */
  media: ResultMedia[];
  /** Performance metrics */
  metrics?: ResultMetrics;
  /** Type-specific metadata */
  metadata?: ModelMetadata | DemoMetadata | FindingMetadata | ComparisonMetadata;
  /** Tags for discovery */
  tags: string[];
  /** Engagement stats */
  stats: ResultStats;
  /** Is featured result */
  isFeatured?: boolean;
  /** Is pinned to top */
  isPinned?: boolean;
  /** When created */
  createdAt: string;
  /** When last updated */
  updatedAt: string;
  /** When published (if status is published) */
  publishedAt?: string;
}

/**
 * Result creation input
 */
export interface CreateResultInput {
  /** Result type */
  type: ResultType;
  /** Title */
  title: string;
  /** Short description */
  description: string;
  /** Full content (markdown) */
  content?: string;
  /** Lab ID */
  labId: string;
  /** Task ID (optional) */
  taskId?: string;
  /** Visibility */
  visibility?: ResultVisibility;
  /** Tags */
  tags?: string[];
  /** Type-specific metadata */
  metadata?: ModelMetadata | DemoMetadata | FindingMetadata | ComparisonMetadata;
}

/**
 * Result update input
 */
export interface UpdateResultInput {
  /** Title */
  title?: string;
  /** Short description */
  description?: string;
  /** Full content */
  content?: string;
  /** Visibility */
  visibility?: ResultVisibility;
  /** Status */
  status?: ResultStatus;
  /** Tags */
  tags?: string[];
  /** Is pinned */
  isPinned?: boolean;
  /** Metrics */
  metrics?: ResultMetrics;
  /** Metadata */
  metadata?: ModelMetadata | DemoMetadata | FindingMetadata | ComparisonMetadata;
}

/**
 * Result list query options
 */
export interface ResultListOptions {
  /** Filter by lab ID */
  labId?: string;
  /** Filter by author ID */
  authorId?: string;
  /** Filter by type */
  type?: ResultType;
  /** Filter by visibility */
  visibility?: ResultVisibility;
  /** Filter by status */
  status?: ResultStatus;
  /** Filter by tags (any match) */
  tags?: string[];
  /** Search query */
  search?: string;
  /** Sort field */
  sortBy?: "likes" | "comments" | "views" | "created" | "updated";
  /** Sort direction */
  sortDir?: "asc" | "desc";
  /** Page number (1-based) */
  page?: number;
  /** Items per page */
  limit?: number;
}

/**
 * Paginated result list
 */
export interface ResultListResult {
  /** Results */
  results: Result[];
  /** Total count */
  total: number;
  /** Current page */
  page: number;
  /** Total pages */
  totalPages: number;
  /** Has more pages */
  hasMore: boolean;
}

/**
 * Like record
 */
export interface ResultLike {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** User ID who liked */
  userId: string;
  /** Result ID that was liked */
  resultId: string;
  /** When liked */
  createdAt: string;
}

/**
 * Save/bookmark record
 */
export interface ResultSave {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** User ID who saved */
  userId: string;
  /** Result ID that was saved */
  resultId: string;
  /** Collection name (optional) */
  collection?: string;
  /** When saved */
  createdAt: string;
}

/**
 * Default result stats
 */
export const DEFAULT_RESULT_STATS: ResultStats = {
  likes: 0,
  comments: 0,
  shares: 0,
  views: 0,
  saves: 0,
};

/**
 * Generate a unique result ID
 */
export function generateResultId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `res_${timestamp}${random}`;
}

/**
 * Generate a unique media ID
 */
export function generateMediaId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 6);
  return `med_${timestamp}${random}`;
}

/**
 * Result type display names
 */
export const RESULT_TYPE_LABELS: Record<ResultType, string> = {
  model: "Model",
  demo: "Demo",
  finding: "Finding",
  comparison: "Comparison",
  dataset: "Dataset",
  paper: "Paper",
};

/**
 * Result type icons (Lucide icon names)
 */
export const RESULT_TYPE_ICONS: Record<ResultType, string> = {
  model: "Box",
  demo: "Play",
  finding: "Lightbulb",
  comparison: "GitCompare",
  dataset: "Database",
  paper: "FileText",
};

/**
 * Media type icons (Lucide icon names)
 */
export const MEDIA_TYPE_ICONS: Record<MediaType, string> = {
  image: "Image",
  audio: "Volume2",
  video: "Video",
  chart: "BarChart3",
  code: "Code",
  file: "File",
};

/**
 * Format metric value for display
 */
export function formatMetricValue(metric: ResultMetric): string {
  const { value, unit } = metric;

  if (unit === "%") {
    return `${(value * 100).toFixed(1)}%`;
  }

  if (typeof value === "number") {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M${unit ? ` ${unit}` : ""}`;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K${unit ? ` ${unit}` : ""}`;
    }
    if (Number.isInteger(value)) {
      return `${value}${unit ? ` ${unit}` : ""}`;
    }
    return `${value.toFixed(2)}${unit ? ` ${unit}` : ""}`;
  }

  return `${value}${unit ? ` ${unit}` : ""}`;
}

/**
 * Calculate metric improvement percentage
 */
export function calculateImprovement(metric: ResultMetric): number | null {
  if (metric.baseline === undefined || metric.baseline === 0) {
    return null;
  }

  const improvement = ((metric.value - metric.baseline) / metric.baseline) * 100;
  return metric.higherIsBetter ? improvement : -improvement;
}

/**
 * Get result URL path
 */
export function getResultPath(resultId: string): string {
  return `/results/${resultId}`;
}

/**
 * Get result API path
 */
export function getResultApiPath(resultId: string): string {
  return `/api/results/${resultId}`;
}

/**
 * Check if user can edit result
 */
export function canEditResult(result: Result, userId?: string): boolean {
  if (!userId) return false;
  return result.author.id === userId;
}

/**
 * Check if result is viewable by user
 */
export function canViewResult(result: Result, userId?: string): boolean {
  if (result.visibility === "public" && result.status === "published") return true;
  if (result.visibility === "unlisted" && result.status === "published") return true;
  if (!userId) return false;
  return result.author.id === userId;
}

/**
 * Type guard for Result
 */
export function isResult(obj: unknown): obj is Result {
  if (!obj || typeof obj !== "object") return false;
  const result = obj as Record<string, unknown>;
  return (
    typeof result.id === "string" &&
    typeof result.type === "string" &&
    typeof result.title === "string" &&
    typeof result.labId === "string" &&
    result.author !== undefined &&
    result.stats !== undefined
  );
}

/**
 * Type guard for ModelMetadata
 */
export function isModelMetadata(obj: unknown): obj is ModelMetadata {
  if (!obj || typeof obj !== "object") return false;
  const meta = obj as Record<string, unknown>;
  return typeof meta.modelName === "string";
}

/**
 * Type guard for DemoMetadata
 */
export function isDemoMetadata(obj: unknown): obj is DemoMetadata {
  if (!obj || typeof obj !== "object") return false;
  const meta = obj as Record<string, unknown>;
  return typeof meta.demoUrl === "string" && typeof meta.isInteractive === "boolean";
}

/**
 * Type guard for FindingMetadata
 */
export function isFindingMetadata(obj: unknown): obj is FindingMetadata {
  if (!obj || typeof obj !== "object") return false;
  const meta = obj as Record<string, unknown>;
  return typeof meta.keyInsight === "string";
}

/**
 * Type guard for ComparisonMetadata
 */
export function isComparisonMetadata(obj: unknown): obj is ComparisonMetadata {
  if (!obj || typeof obj !== "object") return false;
  const meta = obj as Record<string, unknown>;
  return (
    Array.isArray(meta.modelsCompared) &&
    Array.isArray(meta.criteria) &&
    typeof meta.summary === "string"
  );
}
