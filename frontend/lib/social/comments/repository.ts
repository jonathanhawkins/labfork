/**
 * Comment Repository Layer
 *
 * Handles CRUD operations for comments and reactions.
 * Uses JSON file-based storage for persistence.
 */

import {
  findOne,
  findMany,
  insertOne,
  updateOne,
  deleteOne,
  count,
  exists,
  findPaginated,
} from "@/lib/db/json-store";
import {
  Comment,
  CommentWithReplies,
  StoredReaction,
  CommentAuthor,
  CreateCommentInput,
  UpdateCommentInput,
  CommentListOptions,
  CommentListResult,
  ReactionType,
  ReactionCounts,
  generateCommentId,
  extractMentions,
  stripMarkdown,
  DEFAULT_REACTION_COUNTS,
  MAX_REPLY_DEPTH,
} from "./types";

/**
 * Collection names
 */
const COLLECTIONS = {
  COMMENTS: "comments",
  REACTIONS: "comment_reactions",
} as const;

/**
 * Get a comment by ID
 */
export async function getCommentById(id: string): Promise<Comment | null> {
  return findOne<Comment>(COLLECTIONS.COMMENTS, (c) => c.id === id);
}

/**
 * Get comments for an entity
 */
export async function getCommentsByEntity(
  entityType: string,
  entityId: string
): Promise<Comment[]> {
  return findMany<Comment>(
    COLLECTIONS.COMMENTS,
    (c) =>
      c.entityType === entityType &&
      c.entityId === entityId &&
      c.status === "active"
  );
}

/**
 * Get replies to a comment
 */
export async function getReplies(parentId: string): Promise<Comment[]> {
  return findMany<Comment>(
    COLLECTIONS.COMMENTS,
    (c) => c.parentId === parentId && c.status === "active"
  );
}

/**
 * Create a new comment
 */
export async function createComment(
  input: CreateCommentInput,
  author: CommentAuthor
): Promise<Comment> {
  const now = new Date().toISOString();

  // Get parent comment if this is a reply
  let depth = 0;
  let rootId: string | undefined;

  if (input.parentId) {
    const parent = await getCommentById(input.parentId);
    if (parent) {
      depth = Math.min(parent.depth + 1, MAX_REPLY_DEPTH);
      rootId = parent.rootId || parent.id;

      // Update parent's reply count
      await updateOne<Comment>(
        COLLECTIONS.COMMENTS,
        (c) => c.id === input.parentId,
        { replyCount: parent.replyCount + 1 }
      );
    }
  }

  const comment: Comment = {
    id: generateCommentId(),
    entityType: input.entityType,
    entityId: input.entityId,
    parentId: input.parentId,
    rootId,
    author,
    content: input.content,
    contentPlain: stripMarkdown(input.content),
    mentions: extractMentions(input.content),
    status: "active",
    reactionCounts: { ...DEFAULT_REACTION_COUNTS },
    replyCount: 0,
    depth,
    createdAt: now,
    isEdited: false,
  };

  return insertOne(COLLECTIONS.COMMENTS, comment);
}

/**
 * Update a comment
 */
export async function updateComment(
  id: string,
  input: UpdateCommentInput
): Promise<Comment | null> {
  const now = new Date().toISOString();

  return updateOne<Comment>(
    COLLECTIONS.COMMENTS,
    (c) => c.id === id,
    {
      content: input.content,
      contentPlain: stripMarkdown(input.content),
      mentions: extractMentions(input.content),
      editedAt: now,
      isEdited: true,
    }
  );
}

/**
 * Delete a comment (soft delete)
 */
export async function deleteComment(id: string): Promise<boolean> {
  const comment = await getCommentById(id);
  if (!comment) return false;

  // Update parent's reply count if this is a reply
  if (comment.parentId) {
    const parent = await getCommentById(comment.parentId);
    if (parent && parent.replyCount > 0) {
      await updateOne<Comment>(
        COLLECTIONS.COMMENTS,
        (c) => c.id === comment.parentId,
        { replyCount: parent.replyCount - 1 }
      );
    }
  }

  // Soft delete - mark as deleted
  const updated = await updateOne<Comment>(
    COLLECTIONS.COMMENTS,
    (c) => c.id === id,
    { status: "deleted" }
  );

  return updated !== null;
}

/**
 * Hide a comment (moderation)
 */
export async function hideComment(id: string): Promise<Comment | null> {
  return updateOne<Comment>(
    COLLECTIONS.COMMENTS,
    (c) => c.id === id,
    { status: "hidden" }
  );
}

/**
 * Restore a hidden comment
 */
export async function restoreComment(id: string): Promise<Comment | null> {
  return updateOne<Comment>(
    COLLECTIONS.COMMENTS,
    (c) => c.id === id,
    { status: "active" }
  );
}

/**
 * Add a reaction to a comment
 */
export async function addReaction(
  commentId: string,
  userId: string,
  type: ReactionType
): Promise<boolean> {
  // Check if already reacted with this type
  const existingReaction = await findOne<StoredReaction>(
    COLLECTIONS.REACTIONS,
    (r) =>
      r.commentId === commentId && r.userId === userId && r.type === type
  );

  if (existingReaction) return false;

  // Add reaction
  const reaction: StoredReaction = {
    commentId,
    userId,
    type,
    createdAt: new Date().toISOString(),
  };

  await insertOne(COLLECTIONS.REACTIONS, reaction);

  // Update comment reaction counts
  const comment = await getCommentById(commentId);
  if (comment) {
    const newCounts = { ...comment.reactionCounts };
    newCounts[type] = (newCounts[type] || 0) + 1;

    await updateOne<Comment>(
      COLLECTIONS.COMMENTS,
      (c) => c.id === commentId,
      { reactionCounts: newCounts }
    );
  }

  return true;
}

/**
 * Remove a reaction from a comment
 */
export async function removeReaction(
  commentId: string,
  userId: string,
  type: ReactionType
): Promise<boolean> {
  const deleted = await deleteOne<StoredReaction>(
    COLLECTIONS.REACTIONS,
    (r) =>
      r.commentId === commentId && r.userId === userId && r.type === type
  );

  if (!deleted) return false;

  // Update comment reaction counts
  const comment = await getCommentById(commentId);
  if (comment && comment.reactionCounts[type] > 0) {
    const newCounts = { ...comment.reactionCounts };
    newCounts[type] = newCounts[type] - 1;

    await updateOne<Comment>(
      COLLECTIONS.COMMENTS,
      (c) => c.id === commentId,
      { reactionCounts: newCounts }
    );
  }

  return true;
}

/**
 * Get user's reaction on a comment
 */
export async function getUserReaction(
  commentId: string,
  userId: string
): Promise<ReactionType | null> {
  const reaction = await findOne<StoredReaction>(
    COLLECTIONS.REACTIONS,
    (r) => r.commentId === commentId && r.userId === userId
  );

  return reaction?.type || null;
}

/**
 * Get users who reacted to a comment
 */
export async function getReactors(
  commentId: string,
  type?: ReactionType
): Promise<{ userId: string; type: ReactionType }[]> {
  const reactions = await findMany<StoredReaction>(
    COLLECTIONS.REACTIONS,
    (r) =>
      r.commentId === commentId && (type ? r.type === type : true)
  );

  return reactions.map((r) => ({
    userId: r.userId,
    type: r.type,
  }));
}

/**
 * Build comment tree with nested replies
 */
function buildCommentTree(
  comments: Comment[],
  parentId?: string,
  maxDepth = 3
): CommentWithReplies[] {
  const children = comments.filter(
    (c) =>
      (parentId ? c.parentId === parentId : !c.parentId) &&
      c.status === "active"
  );

  return children.map((comment) => ({
    ...comment,
    replies:
      comment.depth < maxDepth
        ? buildCommentTree(comments, comment.id, maxDepth)
        : [],
  }));
}

/**
 * List comments with threading and pagination
 */
export async function listComments(
  options: CommentListOptions
): Promise<CommentListResult> {
  const {
    entityType,
    entityId,
    topLevelOnly = true,
    parentId,
    includeReplies = true,
    maxDepth = 3,
    sortBy = "newest",
    page = 1,
    limit = 20,
  } = options;

  // Build filter for top-level comments
  const filter = (comment: Comment): boolean => {
    if (comment.entityType !== entityType) return false;
    if (comment.entityId !== entityId) return false;
    if (comment.status !== "active") return false;

    if (parentId) {
      return comment.parentId === parentId;
    }

    if (topLevelOnly) {
      return !comment.parentId;
    }

    return true;
  };

  // Build sort function
  const sort = (a: Comment, b: Comment): number => {
    switch (sortBy) {
      case "popular": {
        const aScore =
          Object.values(a.reactionCounts).reduce((s, c) => s + c, 0) +
          a.replyCount * 2;
        const bScore =
          Object.values(b.reactionCounts).reduce((s, c) => s + c, 0) +
          b.replyCount * 2;
        return bScore - aScore;
      }
      case "oldest":
        return (
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        );
      case "newest":
      default:
        return (
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
    }
  };

  // Get paginated top-level comments
  const { items, total, totalPages, hasMore } = await findPaginated<Comment>(
    COLLECTIONS.COMMENTS,
    { filter, sort, page, limit }
  );

  // If including replies, fetch all comments and build tree
  let commentsWithReplies: CommentWithReplies[];

  if (includeReplies && items.length > 0) {
    // Get all comments for this entity to build tree
    const allComments = await findMany<Comment>(
      COLLECTIONS.COMMENTS,
      (c) =>
        c.entityType === entityType &&
        c.entityId === entityId &&
        c.status === "active"
    );

    // Build tree for each top-level comment
    commentsWithReplies = items.map((topLevel) => ({
      ...topLevel,
      replies: buildCommentTree(allComments, topLevel.id, maxDepth),
    }));
  } else {
    commentsWithReplies = items.map((c) => ({ ...c, replies: [] }));
  }

  return {
    comments: commentsWithReplies,
    total,
    page,
    totalPages,
    hasMore,
  };
}

/**
 * Get comment count for an entity
 */
export async function getCommentCount(
  entityType: string,
  entityId: string
): Promise<number> {
  return count<Comment>(
    COLLECTIONS.COMMENTS,
    (c) =>
      c.entityType === entityType &&
      c.entityId === entityId &&
      c.status === "active"
  );
}

/**
 * Get recent comments by user
 */
export async function getCommentsByUser(
  userId: string,
  limit = 20
): Promise<Comment[]> {
  const comments = await findMany<Comment>(
    COLLECTIONS.COMMENTS,
    (c) => c.author.id === userId && c.status === "active"
  );

  return comments
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    )
    .slice(0, limit);
}

/**
 * Search comments
 */
export async function searchComments(
  query: string,
  options: {
    entityType?: string;
    entityId?: string;
    limit?: number;
  } = {}
): Promise<Comment[]> {
  const { entityType, entityId, limit = 20 } = options;
  const queryLower = query.toLowerCase();

  const comments = await findMany<Comment>(
    COLLECTIONS.COMMENTS,
    (c) => {
      if (c.status !== "active") return false;
      if (entityType && c.entityType !== entityType) return false;
      if (entityId && c.entityId !== entityId) return false;
      return c.contentPlain.toLowerCase().includes(queryLower);
    }
  );

  return comments.slice(0, limit);
}

/**
 * Get comments mentioning a user
 */
export async function getCommentsMentioningUser(
  userId: string,
  limit = 20
): Promise<Comment[]> {
  const comments = await findMany<Comment>(
    COLLECTIONS.COMMENTS,
    (c) =>
      c.status === "active" &&
      c.mentions.some((m) => m.userId === userId)
  );

  return comments
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    )
    .slice(0, limit);
}
