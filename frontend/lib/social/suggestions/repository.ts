/**
 * Suggestion Repository Layer
 *
 * Handles CRUD operations for suggestions and votes.
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
  Suggestion,
  SuggestionVote,
  SuggestionAuthor,
  CreateSuggestionInput,
  UpdateSuggestionInput,
  SuggestionListOptions,
  SuggestionListResult,
  SuggestionStatus,
  generateSuggestionId,
  DEFAULT_SUGGESTION_STATS,
  getNetVotes,
} from "./types";

/**
 * Collection names
 */
const COLLECTIONS = {
  SUGGESTIONS: "suggestions",
  VOTES: "suggestion_votes",
} as const;

/**
 * Get a suggestion by ID
 */
export async function getSuggestionById(id: string): Promise<Suggestion | null> {
  return findOne<Suggestion>(COLLECTIONS.SUGGESTIONS, (s) => s.id === id);
}

/**
 * Get suggestions by lab ID
 */
export async function getSuggestionsByLab(labId: string): Promise<Suggestion[]> {
  return findMany<Suggestion>(COLLECTIONS.SUGGESTIONS, (s) => s.labId === labId);
}

/**
 * Create a new suggestion
 */
export async function createSuggestion(
  input: CreateSuggestionInput,
  author: SuggestionAuthor
): Promise<Suggestion> {
  const now = new Date().toISOString();

  const suggestion: Suggestion = {
    id: generateSuggestionId(),
    labId: input.labId,
    taskId: input.taskId,
    resultId: input.resultId,
    author,
    title: input.title,
    description: input.description,
    category: input.category,
    status: "open",
    priority: input.priority || "medium",
    tags: input.tags || [],
    stats: { ...DEFAULT_SUGGESTION_STATS },
    statusHistory: [],
    createdAt: now,
    updatedAt: now,
  };

  return insertOne(COLLECTIONS.SUGGESTIONS, suggestion);
}

/**
 * Update a suggestion
 */
export async function updateSuggestion(
  id: string,
  input: UpdateSuggestionInput,
  changedBy?: string
): Promise<Suggestion | null> {
  const existing = await getSuggestionById(id);
  if (!existing) return null;

  const now = new Date().toISOString();
  const updates: Partial<Suggestion> = { updatedAt: now };

  if (input.title !== undefined) updates.title = input.title;
  if (input.description !== undefined) updates.description = input.description;
  if (input.category !== undefined) updates.category = input.category;
  if (input.priority !== undefined) updates.priority = input.priority;
  if (input.tags !== undefined) updates.tags = input.tags;
  if (input.isPinned !== undefined) updates.isPinned = input.isPinned;

  // Handle status change
  if (input.status !== undefined && input.status !== existing.status) {
    updates.status = input.status;
    updates.statusHistory = [
      ...existing.statusHistory,
      {
        from: existing.status,
        to: input.status,
        changedBy: changedBy || "system",
        reason: input.statusReason,
        changedAt: now,
      },
    ];

    if (["completed", "declined", "duplicate"].includes(input.status)) {
      updates.resolvedAt = now;
    }
  }

  return updateOne<Suggestion>(
    COLLECTIONS.SUGGESTIONS,
    (s) => s.id === id,
    updates
  );
}

/**
 * Delete a suggestion
 */
export async function deleteSuggestion(id: string): Promise<boolean> {
  // Delete associated votes
  const votes = await findMany<SuggestionVote>(
    COLLECTIONS.VOTES,
    (v) => v.suggestionId === id
  );
  for (const vote of votes) {
    await deleteOne<SuggestionVote>(
      COLLECTIONS.VOTES,
      (v) => v.suggestionId === id && v.userId === vote.userId
    );
  }

  return deleteOne<Suggestion>(COLLECTIONS.SUGGESTIONS, (s) => s.id === id);
}

/**
 * Vote on a suggestion
 */
export async function voteSuggestion(
  suggestionId: string,
  userId: string,
  vote: 1 | -1
): Promise<{ success: boolean; stats: Suggestion["stats"] } | null> {
  const suggestion = await getSuggestionById(suggestionId);
  if (!suggestion) return null;

  // Check for existing vote
  const existingVote = await findOne<SuggestionVote>(
    COLLECTIONS.VOTES,
    (v) => v.suggestionId === suggestionId && v.userId === userId
  );

  let newStats = { ...suggestion.stats };

  if (existingVote) {
    if (existingVote.vote === vote) {
      // Same vote, remove it
      await deleteOne<SuggestionVote>(
        COLLECTIONS.VOTES,
        (v) => v.suggestionId === suggestionId && v.userId === userId
      );

      if (vote === 1) {
        newStats.upvotes = Math.max(0, newStats.upvotes - 1);
      } else {
        newStats.downvotes = Math.max(0, newStats.downvotes - 1);
      }
    } else {
      // Different vote, update it
      await updateOne<SuggestionVote>(
        COLLECTIONS.VOTES,
        (v) => v.suggestionId === suggestionId && v.userId === userId,
        { vote, createdAt: new Date().toISOString() }
      );

      if (vote === 1) {
        newStats.upvotes += 1;
        newStats.downvotes = Math.max(0, newStats.downvotes - 1);
      } else {
        newStats.downvotes += 1;
        newStats.upvotes = Math.max(0, newStats.upvotes - 1);
      }
    }
  } else {
    // New vote
    await insertOne<SuggestionVote>(COLLECTIONS.VOTES, {
      suggestionId,
      userId,
      vote,
      createdAt: new Date().toISOString(),
    });

    if (vote === 1) {
      newStats.upvotes += 1;
    } else {
      newStats.downvotes += 1;
    }
  }

  // Update suggestion stats
  await updateOne<Suggestion>(
    COLLECTIONS.SUGGESTIONS,
    (s) => s.id === suggestionId,
    { stats: newStats }
  );

  return { success: true, stats: newStats };
}

/**
 * Get user's vote on a suggestion
 */
export async function getUserVote(
  suggestionId: string,
  userId: string
): Promise<1 | -1 | null> {
  const vote = await findOne<SuggestionVote>(
    COLLECTIONS.VOTES,
    (v) => v.suggestionId === suggestionId && v.userId === userId
  );
  return vote?.vote || null;
}

/**
 * List suggestions with filtering
 */
export async function listSuggestions(
  options: SuggestionListOptions = {}
): Promise<SuggestionListResult> {
  const {
    labId,
    authorId,
    category,
    status,
    priority,
    search,
    sortBy = "votes",
    page = 1,
    limit = 20,
  } = options;

  // Build filter
  const filter = (suggestion: Suggestion): boolean => {
    if (labId && suggestion.labId !== labId) return false;
    if (authorId && suggestion.author.id !== authorId) return false;
    if (category && suggestion.category !== category) return false;
    if (priority && suggestion.priority !== priority) return false;

    if (status) {
      const statuses = Array.isArray(status) ? status : [status];
      if (!statuses.includes(suggestion.status)) return false;
    }

    if (search) {
      const searchLower = search.toLowerCase();
      const matchesTitle = suggestion.title.toLowerCase().includes(searchLower);
      const matchesDesc = suggestion.description
        .toLowerCase()
        .includes(searchLower);
      const matchesTags = suggestion.tags.some((t) =>
        t.toLowerCase().includes(searchLower)
      );
      if (!matchesTitle && !matchesDesc && !matchesTags) return false;
    }

    return true;
  };

  // Build sort function
  const sort = (a: Suggestion, b: Suggestion): number => {
    // Pinned suggestions always come first
    if (a.isPinned && !b.isPinned) return -1;
    if (!a.isPinned && b.isPinned) return 1;

    switch (sortBy) {
      case "votes":
        return getNetVotes(b.stats) - getNetVotes(a.stats);
      case "comments":
        return b.stats.comments - a.stats.comments;
      case "oldest":
        return (
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        );
      case "priority": {
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      }
      case "newest":
      default:
        return (
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
    }
  };

  const { items, total, totalPages, hasMore } =
    await findPaginated<Suggestion>(COLLECTIONS.SUGGESTIONS, {
      filter,
      sort,
      page,
      limit,
    });

  return {
    suggestions: items,
    total,
    page,
    totalPages,
    hasMore,
  };
}

/**
 * Get suggestion count by lab
 */
export async function getSuggestionCountByLab(labId: string): Promise<number> {
  return count<Suggestion>(COLLECTIONS.SUGGESTIONS, (s) => s.labId === labId);
}

/**
 * Get open suggestion count by lab
 */
export async function getOpenSuggestionCount(labId: string): Promise<number> {
  return count<Suggestion>(
    COLLECTIONS.SUGGESTIONS,
    (s) =>
      s.labId === labId &&
      !["completed", "declined", "duplicate"].includes(s.status)
  );
}

/**
 * Pin a suggestion
 */
export async function pinSuggestion(id: string): Promise<Suggestion | null> {
  return updateOne<Suggestion>(
    COLLECTIONS.SUGGESTIONS,
    (s) => s.id === id,
    { isPinned: true, updatedAt: new Date().toISOString() }
  );
}

/**
 * Unpin a suggestion
 */
export async function unpinSuggestion(id: string): Promise<Suggestion | null> {
  return updateOne<Suggestion>(
    COLLECTIONS.SUGGESTIONS,
    (s) => s.id === id,
    { isPinned: false, updatedAt: new Date().toISOString() }
  );
}

/**
 * Increment comment count
 */
export async function incrementCommentCount(id: string): Promise<void> {
  const suggestion = await getSuggestionById(id);
  if (suggestion) {
    await updateOne<Suggestion>(
      COLLECTIONS.SUGGESTIONS,
      (s) => s.id === id,
      {
        stats: {
          ...suggestion.stats,
          comments: suggestion.stats.comments + 1,
        },
      }
    );
  }
}

/**
 * Decrement comment count
 */
export async function decrementCommentCount(id: string): Promise<void> {
  const suggestion = await getSuggestionById(id);
  if (suggestion && suggestion.stats.comments > 0) {
    await updateOne<Suggestion>(
      COLLECTIONS.SUGGESTIONS,
      (s) => s.id === id,
      {
        stats: {
          ...suggestion.stats,
          comments: suggestion.stats.comments - 1,
        },
      }
    );
  }
}
