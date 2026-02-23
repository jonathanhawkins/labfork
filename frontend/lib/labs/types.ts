/**
 * Lab Types and Interfaces
 *
 * Defines the schema for labs in the multi-lab sharing system.
 */

/**
 * Lab visibility options
 */
export type LabVisibility = "public" | "private" | "unlisted";

/**
 * Lab status
 */
export type LabStatus = "active" | "idea" | "archived" | "suspended";

/**
 * Lab statistics
 */
export interface LabStats {
  /** Number of stars */
  stars: number;
  /** Number of forks */
  forks: number;
  /** Number of tasks */
  tasks: number;
  /** Number of papers */
  papers: number;
  /** Number of experiments */
  experiments: number;
  /** Number of current viewers (live) */
  viewers: number;
  /** Total view count */
  views?: number;
}

/**
 * Lab owner information
 */
export interface LabOwner {
  /** User ID */
  id: string;
  /** Username (URL-safe) */
  username: string;
  /** Display name */
  displayName: string;
  /** Avatar URL */
  avatar?: string;
}

/**
 * Fork information
 */
export interface ForkInfo {
  /** Original lab ID */
  sourceLabId: string;
  /** Original lab slug */
  sourceSlug: string;
  /** Original owner username */
  sourceOwner: string;
  /** When the fork was created */
  forkedAt: string;
}

/**
 * Lab activity entry
 */
export interface LabActivity {
  /** Activity ID */
  id: string;
  /** Activity type */
  type: "task_completed" | "paper_added" | "agent_active" | "result_posted" | "fork" | "star";
  /** Activity description */
  description: string;
  /** When it happened */
  timestamp: string;
  /** Related entity ID (task, paper, etc.) */
  entityId?: string;
  /** User who performed the action */
  userId?: string;
}

/**
 * Lab result/output
 */
export interface LabResult {
  /** Result ID */
  id: string;
  /** Result type */
  type: "model" | "paper" | "demo" | "dataset" | "code";
  /** Title */
  title: string;
  /** Description */
  description: string;
  /** URL or path */
  url?: string;
  /** When created */
  createdAt: string;
  /** Associated task ID */
  taskId?: string;
}

/**
 * Complete lab record
 */
export interface Lab {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Unique lab ID */
  id: string;
  /** URL-safe slug (unique per user) */
  slug: string;
  /** Lab name */
  name: string;
  /** Lab description */
  description: string;
  /** Long description / README */
  readme?: string;
  /** Domain slug this lab is based on */
  domainSlug: string;
  /** Domain name for display */
  domainName: string;
  /** Lab owner */
  owner: LabOwner;
  /** Visibility setting */
  visibility: LabVisibility;
  /** Lab status */
  status: LabStatus;
  /** Lab statistics */
  stats: LabStats;
  /** Fork information (if forked) */
  forkedFrom?: ForkInfo;
  /** Tags for discovery */
  tags: string[];
  /** Is featured lab */
  isFeatured?: boolean;
  /** Primary color (hex) */
  primaryColor?: string;
  /** When created */
  createdAt: string;
  /** When last updated */
  updatedAt: string;
  /** Last activity timestamp */
  lastActivityAt: string;
}

/**
 * Lab creation input
 */
export interface CreateLabInput {
  /** Lab name */
  name: string;
  /** URL-safe slug */
  slug: string;
  /** Description */
  description: string;
  /** Domain slug */
  domainSlug: string;
  /** Visibility */
  visibility?: LabVisibility;
  /** Tags */
  tags?: string[];
  /** Primary color */
  primaryColor?: string;
  /** README content */
  readme?: string;
}

/**
 * Lab update input
 */
export interface UpdateLabInput {
  /** Lab name */
  name?: string;
  /** Description */
  description?: string;
  /** Visibility */
  visibility?: LabVisibility;
  /** Status */
  status?: LabStatus;
  /** Tags */
  tags?: string[];
  /** Primary color */
  primaryColor?: string;
  /** README content */
  readme?: string;
}

/**
 * Lab list query options
 */
export interface LabListOptions {
  /** Filter by owner username */
  owner?: string;
  /** Filter by slug */
  slug?: string;
  /** Filter by domain */
  domain?: string;
  /** Filter by visibility */
  visibility?: LabVisibility;
  /** Filter by status */
  status?: LabStatus;
  /** Filter by tags (any match) */
  tags?: string[];
  /** Search query */
  search?: string;
  /** Sort field */
  sortBy?: "stars" | "forks" | "activity" | "created" | "name";
  /** Sort direction */
  sortDir?: "asc" | "desc";
  /** Page number (1-based) */
  page?: number;
  /** Items per page */
  limit?: number;
}

/**
 * Paginated lab list result
 */
export interface LabListResult {
  /** Labs */
  labs: Lab[];
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
 * Star record
 */
export interface Star {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** User ID who starred */
  userId: string;
  /** Lab ID that was starred */
  labId: string;
  /** When starred */
  createdAt: string;
}

/**
 * Fork record
 */
export interface Fork {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Fork ID */
  id: string;
  /** Source lab ID */
  sourceLabId: string;
  /** Forked lab ID */
  forkedLabId: string;
  /** User who forked */
  userId: string;
  /** When forked */
  createdAt: string;
}

/**
 * Default lab stats
 */
export const DEFAULT_LAB_STATS: LabStats = {
  stars: 0,
  forks: 0,
  tasks: 0,
  papers: 0,
  experiments: 0,
  viewers: 0,
};

/**
 * Generate a unique lab ID
 */
export function generateLabId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `lab_${timestamp}${random}`;
}

/**
 * Validate lab slug
 */
export function isValidLabSlug(slug: string): boolean {
  // Alphanumeric, hyphens, 3-50 chars
  return /^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$/.test(slug);
}

/**
 * Slugify a string
 */
export function slugify(str: string): string {
  return str
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .substring(0, 50);
}

/**
 * Get full lab URL path
 */
export function getLabPath(owner: string, slug: string): string {
  return `/labs/${owner}/${slug}`;
}

/**
 * Get lab API URL
 */
export function getLabApiPath(labId: string): string {
  return `/api/labs/${labId}`;
}

/**
 * Format lab stats for display
 */
export function formatLabStats(stats: LabStats): string {
  const parts: string[] = [];
  if (stats.stars > 0) parts.push(`${stats.stars} stars`);
  if (stats.forks > 0) parts.push(`${stats.forks} forks`);
  if (stats.tasks > 0) parts.push(`${stats.tasks} tasks`);
  return parts.join(" | ") || "No activity";
}

/**
 * Check if user can edit lab
 */
export function canEditLab(lab: Lab, userId?: string): boolean {
  if (!userId) return false;
  return lab.owner.id === userId;
}

/**
 * Check if lab is viewable by user
 */
export function canViewLab(lab: Lab, userId?: string): boolean {
  if (lab.visibility === "public") return true;
  if (lab.visibility === "unlisted") return true;
  if (!userId) return false;
  return lab.owner.id === userId;
}

/**
 * Type guard for Lab
 */
export function isLab(obj: unknown): obj is Lab {
  if (!obj || typeof obj !== "object") return false;
  const lab = obj as Record<string, unknown>;
  return (
    typeof lab.id === "string" &&
    typeof lab.slug === "string" &&
    typeof lab.name === "string" &&
    typeof lab.domainSlug === "string" &&
    lab.owner !== undefined &&
    lab.stats !== undefined
  );
}
