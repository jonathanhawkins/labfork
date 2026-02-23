/**
 * Suggestion Types and Schema
 *
 * Defines types for research suggestions in the social layer.
 * Allows users to suggest new research directions and improvements.
 */

/**
 * Suggestion category
 */
export type SuggestionCategory =
  | "research_direction"
  | "improvement"
  | "bug_report"
  | "feature_request"
  | "question"
  | "collaboration";

/**
 * Suggestion status
 */
export type SuggestionStatus =
  | "open"
  | "under_review"
  | "planned"
  | "in_progress"
  | "completed"
  | "declined"
  | "duplicate";

/**
 * Suggestion priority
 */
export type SuggestionPriority = "low" | "medium" | "high" | "critical";

/**
 * Suggestion author information
 */
export interface SuggestionAuthor {
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
 * Suggestion stats
 */
export interface SuggestionStats {
  /** Number of upvotes */
  upvotes: number;
  /** Number of downvotes */
  downvotes: number;
  /** Number of comments */
  comments: number;
}

/**
 * Status change record
 */
export interface StatusChange {
  /** Previous status */
  from: SuggestionStatus;
  /** New status */
  to: SuggestionStatus;
  /** Who made the change */
  changedBy: string;
  /** Reason for change */
  reason?: string;
  /** When changed */
  changedAt: string;
}

/**
 * Suggestion record
 */
export interface Suggestion {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Unique suggestion ID */
  id: string;
  /** Lab ID this suggestion belongs to */
  labId: string;
  /** Related task ID (optional) */
  taskId?: string;
  /** Related result ID (optional) */
  resultId?: string;
  /** Suggestion author */
  author: SuggestionAuthor;
  /** Title */
  title: string;
  /** Description (markdown) */
  description: string;
  /** Category */
  category: SuggestionCategory;
  /** Status */
  status: SuggestionStatus;
  /** Priority */
  priority: SuggestionPriority;
  /** Tags */
  tags: string[];
  /** Stats */
  stats: SuggestionStats;
  /** Status history */
  statusHistory: StatusChange[];
  /** Is pinned */
  isPinned?: boolean;
  /** When created */
  createdAt: string;
  /** When last updated */
  updatedAt: string;
  /** When resolved/closed */
  resolvedAt?: string;
}

/**
 * Suggestion creation input
 */
export interface CreateSuggestionInput {
  /** Lab ID */
  labId: string;
  /** Task ID (optional) */
  taskId?: string;
  /** Result ID (optional) */
  resultId?: string;
  /** Title */
  title: string;
  /** Description */
  description: string;
  /** Category */
  category: SuggestionCategory;
  /** Priority */
  priority?: SuggestionPriority;
  /** Tags */
  tags?: string[];
}

/**
 * Suggestion update input
 */
export interface UpdateSuggestionInput {
  /** Title */
  title?: string;
  /** Description */
  description?: string;
  /** Category */
  category?: SuggestionCategory;
  /** Status */
  status?: SuggestionStatus;
  /** Priority */
  priority?: SuggestionPriority;
  /** Tags */
  tags?: string[];
  /** Is pinned */
  isPinned?: boolean;
  /** Status change reason */
  statusReason?: string;
}

/**
 * Suggestion list options
 */
export interface SuggestionListOptions {
  /** Lab ID */
  labId?: string;
  /** Author ID */
  authorId?: string;
  /** Category filter */
  category?: SuggestionCategory;
  /** Status filter */
  status?: SuggestionStatus | SuggestionStatus[];
  /** Priority filter */
  priority?: SuggestionPriority;
  /** Search query */
  search?: string;
  /** Sort by */
  sortBy?: "votes" | "comments" | "newest" | "oldest" | "priority";
  /** Page */
  page?: number;
  /** Limit */
  limit?: number;
}

/**
 * Paginated suggestion list result
 */
export interface SuggestionListResult {
  /** Suggestions */
  suggestions: Suggestion[];
  /** Total count */
  total: number;
  /** Current page */
  page: number;
  /** Total pages */
  totalPages: number;
  /** Has more */
  hasMore: boolean;
}

/**
 * Vote record
 */
export interface SuggestionVote {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Suggestion ID */
  suggestionId: string;
  /** User ID */
  userId: string;
  /** Vote type (1 = upvote, -1 = downvote) */
  vote: 1 | -1;
  /** When voted */
  createdAt: string;
}

/**
 * Default suggestion stats
 */
export const DEFAULT_SUGGESTION_STATS: SuggestionStats = {
  upvotes: 0,
  downvotes: 0,
  comments: 0,
};

/**
 * Category labels
 */
export const CATEGORY_LABELS: Record<SuggestionCategory, string> = {
  research_direction: "Research Direction",
  improvement: "Improvement",
  bug_report: "Bug Report",
  feature_request: "Feature Request",
  question: "Question",
  collaboration: "Collaboration",
};

/**
 * Category icons (Lucide icon names)
 */
export const CATEGORY_ICONS: Record<SuggestionCategory, string> = {
  research_direction: "Compass",
  improvement: "TrendingUp",
  bug_report: "Bug",
  feature_request: "Lightbulb",
  question: "HelpCircle",
  collaboration: "Users",
};

/**
 * Status labels
 */
export const STATUS_LABELS: Record<SuggestionStatus, string> = {
  open: "Open",
  under_review: "Under Review",
  planned: "Planned",
  in_progress: "In Progress",
  completed: "Completed",
  declined: "Declined",
  duplicate: "Duplicate",
};

/**
 * Status colors
 */
export const STATUS_COLORS: Record<SuggestionStatus, string> = {
  open: "text-blue-400 bg-blue-500/10",
  under_review: "text-yellow-400 bg-yellow-500/10",
  planned: "text-purple-400 bg-purple-500/10",
  in_progress: "text-orange-400 bg-orange-500/10",
  completed: "text-green-400 bg-green-500/10",
  declined: "text-red-400 bg-red-500/10",
  duplicate: "text-foreground-muted bg-foreground-muted/10",
};

/**
 * Priority labels
 */
export const PRIORITY_LABELS: Record<SuggestionPriority, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical",
};

/**
 * Priority colors
 */
export const PRIORITY_COLORS: Record<SuggestionPriority, string> = {
  low: "text-foreground-muted",
  medium: "text-yellow-400",
  high: "text-orange-400",
  critical: "text-red-400",
};

/**
 * Generate unique suggestion ID
 */
export function generateSuggestionId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `sug_${timestamp}${random}`;
}

/**
 * Get net vote score
 */
export function getNetVotes(stats: SuggestionStats): number {
  return stats.upvotes - stats.downvotes;
}

/**
 * Check if status is closed
 */
export function isClosedStatus(status: SuggestionStatus): boolean {
  return ["completed", "declined", "duplicate"].includes(status);
}

/**
 * Check if user can edit suggestion
 */
export function canEditSuggestion(
  suggestion: Suggestion,
  userId?: string
): boolean {
  if (!userId) return false;
  if (isClosedStatus(suggestion.status)) return false;
  return suggestion.author.id === userId;
}

/**
 * Type guard for Suggestion
 */
export function isSuggestion(obj: unknown): obj is Suggestion {
  if (!obj || typeof obj !== "object") return false;
  const sug = obj as Record<string, unknown>;
  return (
    typeof sug.id === "string" &&
    typeof sug.labId === "string" &&
    typeof sug.title === "string" &&
    typeof sug.category === "string" &&
    sug.author !== undefined
  );
}
