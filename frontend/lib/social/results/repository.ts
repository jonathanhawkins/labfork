/**
 * Result Repository Layer
 *
 * Handles CRUD operations for results, likes, and saves.
 * Uses JSON file-based storage for persistence.
 */

import {
  findOne,
  findMany,
  insertOne,
  updateOne,
  deleteOne,
  deleteMany,
  count,
  exists,
  findPaginated,
} from "@/lib/db/json-store";
import {
  Result,
  ResultLike,
  ResultSave,
  ResultAuthor,
  CreateResultInput,
  UpdateResultInput,
  ResultListOptions,
  ResultListResult,
  generateResultId,
  DEFAULT_RESULT_STATS,
} from "./types";

/**
 * Collection names
 */
const COLLECTIONS = {
  RESULTS: "results",
  RESULT_LIKES: "result_likes",
  RESULT_SAVES: "result_saves",
} as const;

/**
 * Get a result by ID
 */
export async function getResultById(id: string): Promise<Result | null> {
  return findOne<Result>(COLLECTIONS.RESULTS, (r) => r.id === id);
}

/**
 * Get results by lab ID
 */
export async function getResultsByLabId(labId: string): Promise<Result[]> {
  return findMany<Result>(COLLECTIONS.RESULTS, (r) => r.labId === labId);
}

/**
 * Get results by author ID
 */
export async function getResultsByAuthorId(authorId: string): Promise<Result[]> {
  return findMany<Result>(COLLECTIONS.RESULTS, (r) => r.author.id === authorId);
}

/**
 * Create a new result
 */
export async function createResult(
  input: CreateResultInput,
  author: ResultAuthor
): Promise<Result> {
  const now = new Date().toISOString();

  const result: Result = {
    id: generateResultId(),
    type: input.type,
    title: input.title,
    description: input.description,
    content: input.content,
    visibility: input.visibility || "public",
    status: "draft",
    author,
    labId: input.labId,
    taskId: input.taskId,
    media: [],
    metrics: undefined,
    metadata: input.metadata,
    tags: input.tags || [],
    stats: { ...DEFAULT_RESULT_STATS },
    createdAt: now,
    updatedAt: now,
  };

  return insertOne(COLLECTIONS.RESULTS, result);
}

/**
 * Update a result
 */
export async function updateResult(
  id: string,
  updates: UpdateResultInput
): Promise<Result | null> {
  const now = new Date().toISOString();

  // Check if publishing
  const publishedAt =
    updates.status === "published"
      ? now
      : undefined;

  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      ...updates,
      updatedAt: now,
      ...(publishedAt && { publishedAt }),
    }
  );
}

/**
 * Delete a result
 */
export async function deleteResult(id: string): Promise<boolean> {
  // Delete associated likes and saves
  await deleteMany<ResultLike>(COLLECTIONS.RESULT_LIKES, (l) => l.resultId === id);
  await deleteMany<ResultSave>(COLLECTIONS.RESULT_SAVES, (s) => s.resultId === id);

  return deleteOne<Result>(COLLECTIONS.RESULTS, (r) => r.id === id);
}

/**
 * Publish a result
 */
export async function publishResult(id: string): Promise<Result | null> {
  const now = new Date().toISOString();

  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      status: "published",
      publishedAt: now,
      updatedAt: now,
    }
  );
}

/**
 * Archive a result
 */
export async function archiveResult(id: string): Promise<Result | null> {
  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      status: "archived",
      updatedAt: new Date().toISOString(),
    }
  );
}

/**
 * Add media to a result
 */
export async function addResultMedia(
  id: string,
  media: Result["media"][0]
): Promise<Result | null> {
  const result = await getResultById(id);
  if (!result) return null;

  const updatedMedia = [...result.media, media];

  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      media: updatedMedia,
      updatedAt: new Date().toISOString(),
    }
  );
}

/**
 * Remove media from a result
 */
export async function removeResultMedia(
  id: string,
  mediaId: string
): Promise<Result | null> {
  const result = await getResultById(id);
  if (!result) return null;

  const updatedMedia = result.media.filter((m) => m.id !== mediaId);

  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      media: updatedMedia,
      updatedAt: new Date().toISOString(),
    }
  );
}

/**
 * Update result metrics
 */
export async function updateResultMetrics(
  id: string,
  metrics: Result["metrics"]
): Promise<Result | null> {
  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      metrics,
      updatedAt: new Date().toISOString(),
    }
  );
}

/**
 * Like a result
 */
export async function likeResult(
  resultId: string,
  userId: string
): Promise<boolean> {
  // Check if already liked
  const alreadyLiked = await exists<ResultLike>(
    COLLECTIONS.RESULT_LIKES,
    (l) => l.resultId === resultId && l.userId === userId
  );

  if (alreadyLiked) return false;

  // Add like
  const like: ResultLike = {
    userId,
    resultId,
    createdAt: new Date().toISOString(),
  };

  await insertOne(COLLECTIONS.RESULT_LIKES, like);

  // Update result stats
  const result = await getResultById(resultId);
  if (result) {
    await updateOne<Result>(
      COLLECTIONS.RESULTS,
      (r) => r.id === resultId,
      {
        stats: {
          ...result.stats,
          likes: result.stats.likes + 1,
        },
      }
    );
  }

  return true;
}

/**
 * Unlike a result
 */
export async function unlikeResult(
  resultId: string,
  userId: string
): Promise<boolean> {
  const deleted = await deleteOne<ResultLike>(
    COLLECTIONS.RESULT_LIKES,
    (l) => l.resultId === resultId && l.userId === userId
  );

  if (!deleted) return false;

  // Update result stats
  const result = await getResultById(resultId);
  if (result && result.stats.likes > 0) {
    await updateOne<Result>(
      COLLECTIONS.RESULTS,
      (r) => r.id === resultId,
      {
        stats: {
          ...result.stats,
          likes: result.stats.likes - 1,
        },
      }
    );
  }

  return true;
}

/**
 * Check if user has liked a result
 */
export async function hasLikedResult(
  resultId: string,
  userId: string
): Promise<boolean> {
  return exists<ResultLike>(
    COLLECTIONS.RESULT_LIKES,
    (l) => l.resultId === resultId && l.userId === userId
  );
}

/**
 * Get users who liked a result
 */
export async function getResultLikers(resultId: string): Promise<string[]> {
  const likes = await findMany<ResultLike>(
    COLLECTIONS.RESULT_LIKES,
    (l) => l.resultId === resultId
  );
  return likes.map((l) => l.userId);
}

/**
 * Save a result
 */
export async function saveResult(
  resultId: string,
  userId: string,
  collection?: string
): Promise<boolean> {
  // Check if already saved
  const alreadySaved = await exists<ResultSave>(
    COLLECTIONS.RESULT_SAVES,
    (s) => s.resultId === resultId && s.userId === userId
  );

  if (alreadySaved) return false;

  // Add save
  const save: ResultSave = {
    userId,
    resultId,
    collection,
    createdAt: new Date().toISOString(),
  };

  await insertOne(COLLECTIONS.RESULT_SAVES, save);

  // Update result stats
  const result = await getResultById(resultId);
  if (result) {
    await updateOne<Result>(
      COLLECTIONS.RESULTS,
      (r) => r.id === resultId,
      {
        stats: {
          ...result.stats,
          saves: result.stats.saves + 1,
        },
      }
    );
  }

  return true;
}

/**
 * Unsave a result
 */
export async function unsaveResult(
  resultId: string,
  userId: string
): Promise<boolean> {
  const deleted = await deleteOne<ResultSave>(
    COLLECTIONS.RESULT_SAVES,
    (s) => s.resultId === resultId && s.userId === userId
  );

  if (!deleted) return false;

  // Update result stats
  const result = await getResultById(resultId);
  if (result && result.stats.saves > 0) {
    await updateOne<Result>(
      COLLECTIONS.RESULTS,
      (r) => r.id === resultId,
      {
        stats: {
          ...result.stats,
          saves: result.stats.saves - 1,
        },
      }
    );
  }

  return true;
}

/**
 * Check if user has saved a result
 */
export async function hasSavedResult(
  resultId: string,
  userId: string
): Promise<boolean> {
  return exists<ResultSave>(
    COLLECTIONS.RESULT_SAVES,
    (s) => s.resultId === resultId && s.userId === userId
  );
}

/**
 * Get saved results for a user
 */
export async function getSavedResults(
  userId: string,
  collection?: string
): Promise<Result[]> {
  const saves = await findMany<ResultSave>(
    COLLECTIONS.RESULT_SAVES,
    (s) =>
      s.userId === userId && (collection ? s.collection === collection : true)
  );

  const resultIds = saves.map((s) => s.resultId);
  return findMany<Result>(COLLECTIONS.RESULTS, (r) => resultIds.includes(r.id));
}

/**
 * Increment result view count
 */
export async function incrementResultViews(id: string): Promise<void> {
  const result = await getResultById(id);
  if (result) {
    await updateOne<Result>(
      COLLECTIONS.RESULTS,
      (r) => r.id === id,
      {
        stats: {
          ...result.stats,
          views: result.stats.views + 1,
        },
      }
    );
  }
}

/**
 * Increment result share count
 */
export async function incrementResultShares(id: string): Promise<void> {
  const result = await getResultById(id);
  if (result) {
    await updateOne<Result>(
      COLLECTIONS.RESULTS,
      (r) => r.id === id,
      {
        stats: {
          ...result.stats,
          shares: result.stats.shares + 1,
        },
      }
    );
  }
}

/**
 * List results with filtering and pagination
 */
export async function listResults(
  options: ResultListOptions = {}
): Promise<ResultListResult> {
  const {
    labId,
    authorId,
    type,
    visibility,
    status,
    tags,
    search,
    sortBy = "created",
    sortDir = "desc",
    page = 1,
    limit = 20,
  } = options;

  // Build filter function
  const filter = (result: Result): boolean => {
    if (labId && result.labId !== labId) return false;
    if (authorId && result.author.id !== authorId) return false;
    if (type && result.type !== type) return false;
    if (visibility && result.visibility !== visibility) return false;
    if (status && result.status !== status) return false;
    if (tags && tags.length > 0) {
      if (!tags.some((tag) => result.tags.includes(tag))) return false;
    }
    if (search) {
      const searchLower = search.toLowerCase();
      const matchesTitle = result.title.toLowerCase().includes(searchLower);
      const matchesDesc = result.description.toLowerCase().includes(searchLower);
      const matchesTags = result.tags.some((t) =>
        t.toLowerCase().includes(searchLower)
      );
      if (!matchesTitle && !matchesDesc && !matchesTags) return false;
    }
    return true;
  };

  // Build sort function
  const sort = (a: Result, b: Result): number => {
    let comparison = 0;

    switch (sortBy) {
      case "likes":
        comparison = a.stats.likes - b.stats.likes;
        break;
      case "comments":
        comparison = a.stats.comments - b.stats.comments;
        break;
      case "views":
        comparison = a.stats.views - b.stats.views;
        break;
      case "updated":
        comparison = new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime();
        break;
      case "created":
      default:
        comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        break;
    }

    return sortDir === "desc" ? -comparison : comparison;
  };

  const { items, total, totalPages, hasMore } = await findPaginated<Result>(
    COLLECTIONS.RESULTS,
    { filter, sort, page, limit }
  );

  return {
    results: items,
    total,
    page,
    totalPages,
    hasMore,
  };
}

/**
 * Get featured results
 */
export async function getFeaturedResults(limit = 10): Promise<Result[]> {
  const results = await findMany<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.isFeatured === true && r.status === "published" && r.visibility === "public"
  );

  return results
    .sort((a, b) => b.stats.likes - a.stats.likes)
    .slice(0, limit);
}

/**
 * Get trending results
 */
export async function getTrendingResults(limit = 10): Promise<Result[]> {
  const results = await findMany<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.status === "published" && r.visibility === "public"
  );

  // Simple trending algorithm: recent + engagement
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;

  return results
    .map((r) => {
      const age = (now - new Date(r.createdAt).getTime()) / dayMs;
      const engagement = r.stats.likes * 2 + r.stats.comments * 3 + r.stats.views * 0.1;
      const score = engagement / Math.pow(age + 1, 1.5);
      return { result: r, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(({ result }) => result);
}

/**
 * Get related results
 */
export async function getRelatedResults(
  result: Result,
  limit = 5
): Promise<Result[]> {
  const allResults = await findMany<Result>(
    COLLECTIONS.RESULTS,
    (r) =>
      r.id !== result.id &&
      r.status === "published" &&
      r.visibility === "public"
  );

  // Score by tag overlap and same lab/type
  return allResults
    .map((r) => {
      let score = 0;
      if (r.labId === result.labId) score += 10;
      if (r.type === result.type) score += 5;
      const tagOverlap = r.tags.filter((t) => result.tags.includes(t)).length;
      score += tagOverlap * 3;
      return { result: r, score };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map(({ result: r }) => r);
}

/**
 * Get result count by lab
 */
export async function getResultCountByLab(labId: string): Promise<number> {
  return count<Result>(COLLECTIONS.RESULTS, (r) => r.labId === labId);
}

/**
 * Get result count by author
 */
export async function getResultCountByAuthor(authorId: string): Promise<number> {
  return count<Result>(COLLECTIONS.RESULTS, (r) => r.author.id === authorId);
}

/**
 * Pin a result to top
 */
export async function pinResult(id: string): Promise<Result | null> {
  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      isPinned: true,
      updatedAt: new Date().toISOString(),
    }
  );
}

/**
 * Unpin a result
 */
export async function unpinResult(id: string): Promise<Result | null> {
  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      isPinned: false,
      updatedAt: new Date().toISOString(),
    }
  );
}

/**
 * Feature a result
 */
export async function featureResult(id: string): Promise<Result | null> {
  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      isFeatured: true,
      updatedAt: new Date().toISOString(),
    }
  );
}

/**
 * Unfeature a result
 */
export async function unfeatureResult(id: string): Promise<Result | null> {
  return updateOne<Result>(
    COLLECTIONS.RESULTS,
    (r) => r.id === id,
    {
      isFeatured: false,
      updatedAt: new Date().toISOString(),
    }
  );
}
