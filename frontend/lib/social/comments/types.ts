/**
 * Comment Types and Schema
 *
 * Defines types for comments in the social layer.
 * Supports threaded replies, reactions, and mentions.
 */

/**
 * Comment status
 */
export type CommentStatus = "active" | "deleted" | "hidden";

/**
 * Entity type that can have comments
 */
export type CommentableEntity = "result" | "lab" | "task" | "paper";

/**
 * Reaction type
 */
export type ReactionType =
  | "like"
  | "love"
  | "insightful"
  | "celebrate"
  | "curious"
  | "disagree";

/**
 * Comment author information
 */
export interface CommentAuthor {
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
 * Mention in a comment
 */
export interface CommentMention {
  /** User ID mentioned */
  userId: string;
  /** Username */
  username: string;
  /** Start position in text */
  startIndex: number;
  /** End position in text */
  endIndex: number;
}

/**
 * Reaction on a comment
 */
export interface CommentReaction {
  /** Reaction type */
  type: ReactionType;
  /** User ID who reacted */
  userId: string;
  /** When reacted */
  createdAt: string;
}

/**
 * Reaction counts for display
 */
export interface ReactionCounts {
  like: number;
  love: number;
  insightful: number;
  celebrate: number;
  curious: number;
  disagree: number;
}

/**
 * Comment record
 */
export interface Comment {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Unique comment ID */
  id: string;
  /** Entity type this comment belongs to */
  entityType: CommentableEntity;
  /** Entity ID this comment belongs to */
  entityId: string;
  /** Parent comment ID (for replies) */
  parentId?: string;
  /** Root comment ID (for deeply nested threads) */
  rootId?: string;
  /** Comment author */
  author: CommentAuthor;
  /** Comment content (markdown) */
  content: string;
  /** Plain text version for search */
  contentPlain: string;
  /** Mentions in the comment */
  mentions: CommentMention[];
  /** Comment status */
  status: CommentStatus;
  /** Reaction counts */
  reactionCounts: ReactionCounts;
  /** Number of replies */
  replyCount: number;
  /** Depth in thread (0 = top-level) */
  depth: number;
  /** When created */
  createdAt: string;
  /** When last edited */
  editedAt?: string;
  /** Is edited */
  isEdited: boolean;
}

/**
 * Comment with nested replies
 */
export interface CommentWithReplies extends Comment {
  /** Nested replies */
  replies: CommentWithReplies[];
}

/**
 * Comment creation input
 */
export interface CreateCommentInput {
  /** Entity type */
  entityType: CommentableEntity;
  /** Entity ID */
  entityId: string;
  /** Parent comment ID (for replies) */
  parentId?: string;
  /** Comment content */
  content: string;
}

/**
 * Comment update input
 */
export interface UpdateCommentInput {
  /** New content */
  content: string;
}

/**
 * Comment list options
 */
export interface CommentListOptions {
  /** Entity type */
  entityType: CommentableEntity;
  /** Entity ID */
  entityId: string;
  /** Only top-level comments */
  topLevelOnly?: boolean;
  /** Parent comment ID (for replies) */
  parentId?: string;
  /** Include nested replies */
  includeReplies?: boolean;
  /** Max depth for nested replies */
  maxDepth?: number;
  /** Sort order */
  sortBy?: "newest" | "oldest" | "popular";
  /** Page number */
  page?: number;
  /** Items per page */
  limit?: number;
}

/**
 * Paginated comment list result
 */
export interface CommentListResult {
  /** Comments */
  comments: CommentWithReplies[];
  /** Total count (top-level) */
  total: number;
  /** Current page */
  page: number;
  /** Total pages */
  totalPages: number;
  /** Has more pages */
  hasMore: boolean;
}

/**
 * Reaction record for storage
 */
export interface StoredReaction {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Comment ID */
  commentId: string;
  /** User ID */
  userId: string;
  /** Reaction type */
  type: ReactionType;
  /** When created */
  createdAt: string;
}

/**
 * Default reaction counts
 */
export const DEFAULT_REACTION_COUNTS: ReactionCounts = {
  like: 0,
  love: 0,
  insightful: 0,
  celebrate: 0,
  curious: 0,
  disagree: 0,
};

/**
 * Reaction type labels
 */
export const REACTION_LABELS: Record<ReactionType, string> = {
  like: "Like",
  love: "Love",
  insightful: "Insightful",
  celebrate: "Celebrate",
  curious: "Curious",
  disagree: "Disagree",
};

/**
 * Reaction type emojis
 */
export const REACTION_EMOJIS: Record<ReactionType, string> = {
  like: "👍",
  love: "❤️",
  insightful: "💡",
  celebrate: "🎉",
  curious: "🤔",
  disagree: "👎",
};

/**
 * Generate a unique comment ID
 */
export function generateCommentId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `cmt_${timestamp}${random}`;
}

/**
 * Extract mentions from content
 */
export function extractMentions(content: string): CommentMention[] {
  const mentions: CommentMention[] = [];
  const mentionRegex = /@([a-zA-Z0-9_-]+)/g;
  let match;

  while ((match = mentionRegex.exec(content)) !== null) {
    mentions.push({
      userId: "", // Will be resolved later
      username: match[1],
      startIndex: match.index,
      endIndex: match.index + match[0].length,
    });
  }

  return mentions;
}

/**
 * Strip markdown to plain text
 */
export function stripMarkdown(content: string): string {
  return content
    // Remove images first (before other markdown processing)
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "")
    // Convert links to their text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    // Remove markdown formatting characters
    .replace(/[#*_~`]/g, "")
    // Collapse newlines
    .replace(/\n+/g, " ")
    .trim();
}

/**
 * Get total reaction count
 */
export function getTotalReactions(counts: ReactionCounts): number {
  return Object.values(counts).reduce((sum, count) => sum + count, 0);
}

/**
 * Check if user can edit comment
 */
export function canEditComment(comment: Comment, userId?: string): boolean {
  if (!userId) return false;
  if (comment.status !== "active") return false;
  return comment.author.id === userId;
}

/**
 * Check if user can delete comment
 */
export function canDeleteComment(comment: Comment, userId?: string): boolean {
  if (!userId) return false;
  if (comment.status === "deleted") return false;
  return comment.author.id === userId;
}

/**
 * Type guard for Comment
 */
export function isComment(obj: unknown): obj is Comment {
  if (!obj || typeof obj !== "object") return false;
  const comment = obj as Record<string, unknown>;
  return (
    typeof comment.id === "string" &&
    typeof comment.entityType === "string" &&
    typeof comment.entityId === "string" &&
    typeof comment.content === "string" &&
    comment.author !== undefined
  );
}

/**
 * Max comment content length
 */
export const MAX_COMMENT_LENGTH = 10000;

/**
 * Max reply depth
 */
export const MAX_REPLY_DEPTH = 5;
