/**
 * Papers Repository
 *
 * CRUD operations for papers using JSON file storage.
 * This replaces the filesystem-based approach in the API route.
 */

import {
  readCollection,
  writeCollection,
  findOne,
  findMany,
  insertOne,
  updateOne,
  deleteOne,
  count,
  COLLECTIONS,
} from "@/lib/db/json-store";
import type {
  Paper,
  PaperStatus,
  PaperListFilters,
  PaperListResponse,
} from "./types";

/**
 * Papers collection name
 */
const PAPERS_COLLECTION = COLLECTIONS.PAPERS;

/**
 * Get all papers
 */
export async function getAllPapers(): Promise<Paper[]> {
  return readCollection<Paper>(PAPERS_COLLECTION);
}

/**
 * Get paper by ID
 */
export async function getPaperById(id: string): Promise<Paper | null> {
  return findOne<Paper>(PAPERS_COLLECTION, (paper) => paper.id === id);
}

/**
 * Get paper by metadata ID
 */
export async function getPaperByMetadataId(
  metadataId: string
): Promise<Paper | null> {
  return findOne<Paper>(
    PAPERS_COLLECTION,
    (paper) => paper.metadata.id === metadataId
  );
}

/**
 * Get paper by URL
 */
export async function getPaperByUrl(url: string): Promise<Paper | null> {
  return findOne<Paper>(
    PAPERS_COLLECTION,
    (paper) => paper.metadata.url === url
  );
}

/**
 * Create a new paper
 */
export async function createPaper(paper: Paper): Promise<Paper> {
  // Check for duplicates
  const existing = await getPaperByMetadataId(paper.metadata.id);
  if (existing) {
    throw new Error(`Paper with metadata ID "${paper.metadata.id}" already exists`);
  }

  return insertOne(PAPERS_COLLECTION, paper);
}

/**
 * Update a paper
 */
export async function updatePaper(
  id: string,
  updates: Partial<Paper>
): Promise<Paper | null> {
  const updated = await updateOne<Paper>(
    PAPERS_COLLECTION,
    (paper) => paper.id === id,
    {
      ...updates,
      updatedAt: new Date().toISOString(),
    }
  );
  return updated;
}

/**
 * Update paper status
 */
export async function updatePaperStatus(
  id: string,
  status: PaperStatus,
  error?: string
): Promise<Paper | null> {
  const updates: Partial<Paper> = { status };
  if (error !== undefined) {
    updates.error = error;
  }
  return updatePaper(id, updates);
}

/**
 * Delete a paper
 */
export async function deletePaper(id: string): Promise<boolean> {
  return deleteOne<Paper>(PAPERS_COLLECTION, (paper) => paper.id === id);
}

/**
 * Filter papers based on criteria
 */
function buildPaperFilter(filters: PaperListFilters) {
  return (paper: Paper): boolean => {
    // Status filter
    if (filters.status) {
      const statuses = Array.isArray(filters.status)
        ? filters.status
        : [filters.status];
      if (!statuses.includes(paper.status)) return false;
    }

    // Domain filter
    if (filters.domainSlug) {
      if (paper.domainSlug !== filters.domainSlug) return false;
    }

    // Source filter
    if (filters.source) {
      if (paper.metadata.source !== filters.source) return false;
    }

    // Search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      const titleMatch = paper.metadata.title.toLowerCase().includes(searchLower);
      const abstractMatch = paper.metadata.abstract
        .toLowerCase()
        .includes(searchLower);
      const authorMatch = paper.metadata.authors.some((a) =>
        a.name.toLowerCase().includes(searchLower)
      );
      if (!titleMatch && !abstractMatch && !authorMatch) return false;
    }

    // Minimum relevance filter
    if (filters.minRelevance !== undefined) {
      if (!paper.analysis || paper.analysis.relevanceScore < filters.minRelevance) {
        return false;
      }
    }

    return true;
  };
}

/**
 * Sort papers
 */
function buildPaperSort(filters: PaperListFilters) {
  const sortBy = filters.sortBy || "addedAt";
  const sortOrder = filters.sortOrder || "desc";
  const multiplier = sortOrder === "desc" ? -1 : 1;

  return (a: Paper, b: Paper): number => {
    switch (sortBy) {
      case "relevanceScore": {
        const scoreA = a.analysis?.relevanceScore ?? 0;
        const scoreB = b.analysis?.relevanceScore ?? 0;
        return (scoreA - scoreB) * multiplier;
      }
      case "citationCount": {
        const citesA = a.metadata.citationCount ?? 0;
        const citesB = b.metadata.citationCount ?? 0;
        return (citesA - citesB) * multiplier;
      }
      case "publishedDate": {
        const dateA = a.metadata.publishedDate || "";
        const dateB = b.metadata.publishedDate || "";
        return dateA.localeCompare(dateB) * multiplier;
      }
      case "addedAt":
      default:
        return a.addedAt.localeCompare(b.addedAt) * multiplier;
    }
  };
}

/**
 * List papers with filtering, sorting, and pagination
 */
export async function listPapers(
  filters: PaperListFilters = {}
): Promise<PaperListResponse> {
  // Get all papers
  let papers = await getAllPapers();

  // Apply filters
  const filterFn = buildPaperFilter(filters);
  papers = papers.filter(filterFn);

  const total = papers.length;

  // Apply sorting
  const sortFn = buildPaperSort(filters);
  papers.sort(sortFn);

  // Apply pagination
  if (filters.offset) {
    papers = papers.slice(filters.offset);
  }
  if (filters.limit) {
    papers = papers.slice(0, filters.limit);
  }

  return {
    papers,
    total,
    filters,
  };
}

/**
 * Get papers by domain
 */
export async function getPapersByDomain(
  domainSlug: string,
  limit?: number
): Promise<Paper[]> {
  const result = await listPapers({
    domainSlug,
    sortBy: "relevanceScore",
    sortOrder: "desc",
    limit,
  });
  return result.papers;
}

/**
 * Get papers by status
 */
export async function getPapersByStatus(
  status: PaperStatus | PaperStatus[]
): Promise<Paper[]> {
  const result = await listPapers({
    status,
    sortBy: "addedAt",
    sortOrder: "desc",
  });
  return result.papers;
}

/**
 * Search papers
 */
export async function searchPapers(
  query: string,
  options: Partial<PaperListFilters> = {}
): Promise<PaperListResponse> {
  return listPapers({
    ...options,
    search: query,
  });
}

/**
 * Get total paper count
 */
export async function getTotalPaperCount(): Promise<number> {
  return count<Paper>(PAPERS_COLLECTION);
}

/**
 * Get paper count by domain
 */
export async function getPaperCountByDomain(domainSlug: string): Promise<number> {
  return count<Paper>(
    PAPERS_COLLECTION,
    (paper) => paper.domainSlug === domainSlug
  );
}

/**
 * Get paper count by status
 */
export async function getPaperCountByStatus(status: PaperStatus): Promise<number> {
  return count<Paper>(PAPERS_COLLECTION, (paper) => paper.status === status);
}

/**
 * Check if paper exists by metadata ID
 */
export async function paperExists(metadataId: string): Promise<boolean> {
  const paper = await getPaperByMetadataId(metadataId);
  return paper !== null;
}

/**
 * Migrate papers from old filesystem storage to database
 * This is a one-time migration helper
 */
export async function migratePapersFromFile(
  oldPapers: Paper[]
): Promise<{ migrated: number; skipped: number }> {
  let migrated = 0;
  let skipped = 0;

  for (const paper of oldPapers) {
    const existing = await getPaperByMetadataId(paper.metadata.id);
    if (existing) {
      skipped++;
      continue;
    }

    try {
      await createPaper(paper);
      migrated++;
    } catch (error) {
      console.error(`Failed to migrate paper ${paper.id}:`, error);
      skipped++;
    }
  }

  return { migrated, skipped };
}
