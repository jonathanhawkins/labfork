/**
 * Lab Search and Filter Utilities
 *
 * Functions for searching, filtering, and sorting labs.
 */

import type { Lab, LabListOptions, LabListResult, LabVisibility } from "./types";

/**
 * Search labs by text query
 *
 * Searches name, description, tags, and owner name
 */
export function searchLabsInMemory(labs: Lab[], query: string): Lab[] {
  if (!query.trim()) return labs;

  const normalizedQuery = query.toLowerCase().trim();
  const queryWords = normalizedQuery.split(/\s+/);

  return labs.filter((lab) => {
    const searchableText = [
      lab.name,
      lab.description,
      lab.owner.displayName,
      lab.owner.username,
      lab.domainName,
      ...lab.tags,
    ]
      .join(" ")
      .toLowerCase();

    // All query words must match
    return queryWords.every((word) => searchableText.includes(word));
  });
}

/**
 * Filter labs by domain
 */
export function filterByDomain(labs: Lab[], domain: string): Lab[] {
  if (!domain || domain === "all") return labs;
  return labs.filter((lab) => lab.domainSlug === domain);
}

/**
 * Filter labs by visibility
 */
export function filterByVisibility(labs: Lab[], visibility: LabVisibility): Lab[] {
  return labs.filter((lab) => lab.visibility === visibility);
}

/**
 * Filter labs by featured status
 */
export function filterByFeatured(labs: Lab[], featured: boolean): Lab[] {
  return labs.filter((lab) => lab.isFeatured === featured);
}

/**
 * Filter labs by tags (any match)
 */
export function filterByTags(labs: Lab[], tags: string[]): Lab[] {
  if (!tags.length) return labs;
  const normalizedTags = tags.map((t) => t.toLowerCase());
  return labs.filter((lab) =>
    lab.tags.some((tag) => normalizedTags.includes(tag.toLowerCase()))
  );
}

/**
 * Filter labs by owner
 */
export function filterByOwner(labs: Lab[], ownerUsername: string): Lab[] {
  return labs.filter((lab) => lab.owner.username === ownerUsername);
}

/**
 * Filter labs by status
 */
export function filterByStatus(labs: Lab[], status: Lab["status"]): Lab[] {
  return labs.filter((lab) => lab.status === status);
}

/**
 * Sort options
 */
export type SortField = "stars" | "forks" | "activity" | "created" | "name" | "viewers";
export type SortDirection = "asc" | "desc";

/**
 * Sort labs by field
 */
export function sortLabs(
  labs: Lab[],
  field: SortField = "stars",
  direction: SortDirection = "desc"
): Lab[] {
  const sorted = [...labs].sort((a, b) => {
    let comparison = 0;

    switch (field) {
      case "stars":
        comparison = a.stats.stars - b.stats.stars;
        break;
      case "forks":
        comparison = a.stats.forks - b.stats.forks;
        break;
      case "activity":
        comparison = new Date(a.lastActivityAt).getTime() - new Date(b.lastActivityAt).getTime();
        break;
      case "created":
        comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        break;
      case "name":
        comparison = a.name.localeCompare(b.name);
        break;
      case "viewers":
        comparison = a.stats.viewers - b.stats.viewers;
        break;
      default:
        comparison = 0;
    }

    return direction === "desc" ? -comparison : comparison;
  });

  return sorted;
}

/**
 * Paginate labs
 */
export function paginateLabs(
  labs: Lab[],
  page: number = 1,
  limit: number = 12
): LabListResult {
  const total = labs.length;
  const totalPages = Math.ceil(total / limit);
  const startIndex = (page - 1) * limit;
  const endIndex = startIndex + limit;
  const paginatedLabs = labs.slice(startIndex, endIndex);

  return {
    labs: paginatedLabs,
    total,
    page,
    totalPages,
    hasMore: page < totalPages,
  };
}

/**
 * Apply all filters and options to labs
 */
export function applyLabFilters(labs: Lab[], options: LabListOptions): LabListResult {
  let filtered = [...labs];

  // Apply filters
  if (options.owner) {
    filtered = filterByOwner(filtered, options.owner);
  }

  if (options.domain) {
    filtered = filterByDomain(filtered, options.domain);
  }

  if (options.visibility) {
    filtered = filterByVisibility(filtered, options.visibility);
  }

  if (options.status) {
    filtered = filterByStatus(filtered, options.status);
  }

  if (options.tags && options.tags.length > 0) {
    filtered = filterByTags(filtered, options.tags);
  }

  if (options.search) {
    filtered = searchLabsInMemory(filtered, options.search);
  }

  // Apply sorting
  const sortField = options.sortBy || "stars";
  const sortDir = options.sortDir || "desc";
  filtered = sortLabs(filtered, sortField as SortField, sortDir);

  // Apply pagination
  const page = options.page || 1;
  const limit = options.limit || 12;
  return paginateLabs(filtered, page, limit);
}

/**
 * Get trending labs (most activity in last 7 days)
 */
export function getTrendingLabs(labs: Lab[], limit: number = 10): Lab[] {
  const oneWeekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;

  // Filter to recently active labs
  const recentlyActive = labs.filter(
    (lab) => new Date(lab.lastActivityAt).getTime() > oneWeekAgo
  );

  // Sort by a combination of stars and activity
  return recentlyActive
    .sort((a, b) => {
      const scoreA = a.stats.stars * 2 + a.stats.forks * 3 + a.stats.tasks;
      const scoreB = b.stats.stars * 2 + b.stats.forks * 3 + b.stats.tasks;
      return scoreB - scoreA;
    })
    .slice(0, limit);
}

/**
 * Get related labs (same domain, similar tags)
 */
export function getRelatedLabs(lab: Lab, allLabs: Lab[], limit: number = 5): Lab[] {
  // Exclude the current lab
  const otherLabs = allLabs.filter((l) => l.id !== lab.id);

  // Score each lab based on similarity
  const scored = otherLabs.map((other) => {
    let score = 0;

    // Same domain
    if (other.domainSlug === lab.domainSlug) {
      score += 10;
    }

    // Same owner
    if (other.owner.id === lab.owner.id) {
      score += 5;
    }

    // Matching tags
    const matchingTags = lab.tags.filter((tag) =>
      other.tags.map((t) => t.toLowerCase()).includes(tag.toLowerCase())
    );
    score += matchingTags.length * 3;

    // Boost for popular labs
    score += Math.log10(other.stats.stars + 1);

    return { lab: other, score };
  });

  // Sort by score and return top results
  return scored
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((s) => s.lab);
}

/**
 * Parse search query for advanced filters
 *
 * Supports: domain:voice-clone tag:prosody owner:username
 */
export function parseSearchQuery(query: string): {
  text: string;
  filters: Record<string, string>;
} {
  const filters: Record<string, string> = {};
  let text = query;

  // Extract key:value patterns
  const filterPattern = /(\w+):(\S+)/g;
  let match;

  while ((match = filterPattern.exec(query)) !== null) {
    const [fullMatch, key, value] = match;
    filters[key] = value;
    text = text.replace(fullMatch, "").trim();
  }

  return { text, filters };
}

/**
 * Highlight search matches in text
 */
export function highlightMatches(text: string, query: string): string {
  if (!query.trim()) return text;

  const words = query.toLowerCase().split(/\s+/);
  let result = text;

  words.forEach((word) => {
    const regex = new RegExp(`(${escapeRegex(word)})`, "gi");
    result = result.replace(regex, "<mark>$1</mark>");
  });

  return result;
}

/**
 * Escape special regex characters
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Get search suggestions based on partial query
 */
export function getSearchSuggestions(
  labs: Lab[],
  query: string,
  limit: number = 5
): string[] {
  if (!query.trim() || query.length < 2) return [];

  const normalizedQuery = query.toLowerCase();
  const suggestions = new Set<string>();

  // Collect matching lab names
  labs.forEach((lab) => {
    if (lab.name.toLowerCase().includes(normalizedQuery)) {
      suggestions.add(lab.name);
    }
  });

  // Collect matching tags
  labs.forEach((lab) => {
    lab.tags.forEach((tag) => {
      if (tag.toLowerCase().includes(normalizedQuery)) {
        suggestions.add(tag);
      }
    });
  });

  // Collect matching domain names
  const domains = new Set(labs.map((l) => l.domainName));
  domains.forEach((domain) => {
    if (domain.toLowerCase().includes(normalizedQuery)) {
      suggestions.add(domain);
    }
  });

  return Array.from(suggestions).slice(0, limit);
}

/**
 * Calculate lab relevance score for search ranking
 */
export function calculateRelevanceScore(lab: Lab, query: string): number {
  const normalizedQuery = query.toLowerCase();
  let score = 0;

  // Exact name match
  if (lab.name.toLowerCase() === normalizedQuery) {
    score += 100;
  }
  // Name starts with query
  else if (lab.name.toLowerCase().startsWith(normalizedQuery)) {
    score += 50;
  }
  // Name contains query
  else if (lab.name.toLowerCase().includes(normalizedQuery)) {
    score += 25;
  }

  // Tag exact match
  if (lab.tags.some((t) => t.toLowerCase() === normalizedQuery)) {
    score += 30;
  }

  // Description contains query
  if (lab.description.toLowerCase().includes(normalizedQuery)) {
    score += 10;
  }

  // Boost for popularity
  score += Math.log10(lab.stats.stars + 1) * 5;
  score += Math.log10(lab.stats.forks + 1) * 3;

  // Boost for featured
  if (lab.isFeatured) {
    score += 20;
  }

  return score;
}
