/**
 * Enhanced Semantic Scholar Client
 *
 * Extends the base client with:
 * - Citation network traversal
 * - Related paper finding
 * - Influence scoring
 * - Caching for repeated requests
 */

import {
  PaperMetadata,
  Paper,
  PaperFetchResult,
  createPaper,
} from "../papers/types";
import {
  fetchByPaperId,
  fetchByArxivId,
  fetchByDOI,
  searchSemanticScholar,
} from "../papers/semantic-scholar";

// ============================================================================
// Types
// ============================================================================

export interface CitationInfo {
  paperId: string;
  title: string;
  year?: number;
  citationCount?: number;
  influentialCitationCount?: number;
  isInfluential: boolean;
  contexts?: string[];
  intents?: string[];
}

export interface CitationNetwork {
  paper: Paper;
  citations: CitationInfo[];
  references: CitationInfo[];
  totalCitations: number;
  totalReferences: number;
  influentialCitations: number;
  influentialReferences: number;
}

export interface RelatedPaper {
  paper: Paper;
  relevanceScore: number;
  relationshipType: "citation" | "reference" | "recommended" | "similar";
  sharedAuthors?: string[];
  sharedTopics?: string[];
}

export interface InfluenceScore {
  paperId: string;
  totalCitations: number;
  influentialCitations: number;
  hIndex?: number;
  influenceScore: number;
  influenceLevel: "low" | "moderate" | "high" | "very-high" | "seminal";
  percentile?: number;
}

export interface PaperGraph {
  nodes: Map<string, Paper>;
  edges: Array<{
    from: string;
    to: string;
    type: "cites" | "cited-by";
    isInfluential: boolean;
  }>;
  rootPaperId: string;
  depth: number;
}

// ============================================================================
// Cache
// ============================================================================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class SimpleCache<T> {
  private cache = new Map<string, CacheEntry<T>>();
  private defaultTTL: number;

  constructor(defaultTTL = 5 * 60 * 1000) {
    // 5 minutes default
    this.defaultTTL = defaultTTL;
  }

  get(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  set(key: string, data: T, ttl?: number): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttl ?? this.defaultTTL,
    });
  }

  has(key: string): boolean {
    return this.get(key) !== null;
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }
}

// Global caches
const citationCache = new SimpleCache<CitationInfo[]>(10 * 60 * 1000); // 10 min
const referenceCache = new SimpleCache<CitationInfo[]>(10 * 60 * 1000);
const paperCache = new SimpleCache<Paper>(30 * 60 * 1000); // 30 min
const influenceCache = new SimpleCache<InfluenceScore>(60 * 60 * 1000); // 1 hour

// ============================================================================
// API Configuration
// ============================================================================

const S2_API_BASE = "https://api.semanticscholar.org/graph/v1";
const CITATION_FIELDS = [
  "paperId",
  "title",
  "year",
  "citationCount",
  "influentialCitationCount",
  "isInfluential",
  "contexts",
  "intents",
].join(",");

let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 100;

async function rateLimitedFetch(
  url: string,
  apiKey?: string
): Promise<Response> {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;

  if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
    await new Promise((resolve) =>
      setTimeout(resolve, MIN_REQUEST_INTERVAL - timeSinceLastRequest)
    );
  }

  lastRequestTime = Date.now();

  const headers: Record<string, string> = {
    Accept: "application/json",
  };

  if (apiKey) {
    headers["x-api-key"] = apiKey;
  }

  return fetch(url, { headers });
}

// ============================================================================
// Citation Network Functions
// ============================================================================

/**
 * Fetch citations for a paper (papers that cite this paper)
 */
export async function fetchCitations(
  paperId: string,
  options: {
    limit?: number;
    offset?: number;
    apiKey?: string;
  } = {}
): Promise<CitationInfo[]> {
  const cacheKey = `citations:${paperId}:${options.limit || 100}:${options.offset || 0}`;
  const cached = citationCache.get(cacheKey);
  if (cached) return cached;

  const { limit = 100, offset = 0, apiKey } = options;

  try {
    const url = `${S2_API_BASE}/paper/${paperId}/citations?fields=${CITATION_FIELDS}&limit=${limit}&offset=${offset}`;
    const response = await rateLimitedFetch(url, apiKey);

    if (!response.ok) {
      console.error(`Failed to fetch citations: ${response.status}`);
      return [];
    }

    const data = await response.json();
    const citations: CitationInfo[] = (data.data || []).map(
      (item: { citingPaper: CitationInfo; isInfluential?: boolean }) => ({
        paperId: item.citingPaper.paperId,
        title: item.citingPaper.title,
        year: item.citingPaper.year,
        citationCount: item.citingPaper.citationCount,
        influentialCitationCount: item.citingPaper.influentialCitationCount,
        isInfluential: item.isInfluential || false,
        contexts: item.citingPaper.contexts,
        intents: item.citingPaper.intents,
      })
    );

    citationCache.set(cacheKey, citations);
    return citations;
  } catch (error) {
    console.error("Error fetching citations:", error);
    return [];
  }
}

/**
 * Fetch references for a paper (papers this paper cites)
 */
export async function fetchReferences(
  paperId: string,
  options: {
    limit?: number;
    offset?: number;
    apiKey?: string;
  } = {}
): Promise<CitationInfo[]> {
  const cacheKey = `references:${paperId}:${options.limit || 100}:${options.offset || 0}`;
  const cached = referenceCache.get(cacheKey);
  if (cached) return cached;

  const { limit = 100, offset = 0, apiKey } = options;

  try {
    const url = `${S2_API_BASE}/paper/${paperId}/references?fields=${CITATION_FIELDS}&limit=${limit}&offset=${offset}`;
    const response = await rateLimitedFetch(url, apiKey);

    if (!response.ok) {
      console.error(`Failed to fetch references: ${response.status}`);
      return [];
    }

    const data = await response.json();
    const references: CitationInfo[] = (data.data || []).map(
      (item: { citedPaper: CitationInfo; isInfluential?: boolean }) => ({
        paperId: item.citedPaper.paperId,
        title: item.citedPaper.title,
        year: item.citedPaper.year,
        citationCount: item.citedPaper.citationCount,
        influentialCitationCount: item.citedPaper.influentialCitationCount,
        isInfluential: item.isInfluential || false,
        contexts: item.citedPaper.contexts,
        intents: item.citedPaper.intents,
      })
    );

    referenceCache.set(cacheKey, references);
    return references;
  } catch (error) {
    console.error("Error fetching references:", error);
    return [];
  }
}

/**
 * Get complete citation network for a paper
 */
export async function getCitationNetwork(
  paperId: string,
  options: {
    citationLimit?: number;
    referenceLimit?: number;
    apiKey?: string;
  } = {}
): Promise<CitationNetwork | null> {
  const { citationLimit = 100, referenceLimit = 100, apiKey } = options;

  // Fetch paper details
  const paperResult = await fetchByPaperId(paperId, apiKey);
  if (!paperResult.success || !paperResult.paper) {
    return null;
  }

  // Fetch citations and references in parallel
  const [citations, references] = await Promise.all([
    fetchCitations(paperId, { limit: citationLimit, apiKey }),
    fetchReferences(paperId, { limit: referenceLimit, apiKey }),
  ]);

  const influentialCitations = citations.filter((c) => c.isInfluential).length;
  const influentialReferences = references.filter((r) => r.isInfluential)
    .length;

  return {
    paper: paperResult.paper,
    citations,
    references,
    totalCitations: citations.length,
    totalReferences: references.length,
    influentialCitations,
    influentialReferences,
  };
}

// ============================================================================
// Related Paper Functions
// ============================================================================

/**
 * Find related papers through multiple methods
 */
export async function findRelatedPapers(
  paperId: string,
  options: {
    maxResults?: number;
    includeRecommendations?: boolean;
    includeCitationOverlap?: boolean;
    apiKey?: string;
  } = {}
): Promise<RelatedPaper[]> {
  const {
    maxResults = 20,
    includeRecommendations = true,
    includeCitationOverlap = true,
    apiKey,
  } = options;

  const related: RelatedPaper[] = [];
  const seenIds = new Set<string>();

  // 1. Get direct citations and references
  const [citations, references] = await Promise.all([
    fetchCitations(paperId, { limit: 50, apiKey }),
    fetchReferences(paperId, { limit: 50, apiKey }),
  ]);

  // Add top citations as related
  for (const citation of citations.slice(0, 10)) {
    if (seenIds.has(citation.paperId)) continue;
    seenIds.add(citation.paperId);

    const paperResult = await fetchByPaperId(citation.paperId, apiKey);
    if (paperResult.success && paperResult.paper) {
      related.push({
        paper: paperResult.paper,
        relevanceScore: citation.isInfluential ? 0.9 : 0.7,
        relationshipType: "citation",
      });
    }
  }

  // Add influential references as related
  for (const reference of references.filter((r) => r.isInfluential).slice(0, 10)) {
    if (seenIds.has(reference.paperId)) continue;
    seenIds.add(reference.paperId);

    const paperResult = await fetchByPaperId(reference.paperId, apiKey);
    if (paperResult.success && paperResult.paper) {
      related.push({
        paper: paperResult.paper,
        relevanceScore: 0.85,
        relationshipType: "reference",
      });
    }
  }

  // 2. Get recommendations from Semantic Scholar
  if (includeRecommendations) {
    try {
      const url = `${S2_API_BASE}/recommendations/v1/papers/forpaper/${paperId}?limit=10`;
      const response = await rateLimitedFetch(url, apiKey);

      if (response.ok) {
        const data = await response.json();
        for (const rec of data.recommendedPapers || []) {
          if (seenIds.has(rec.paperId)) continue;
          seenIds.add(rec.paperId);

          const paperResult = await fetchByPaperId(rec.paperId, apiKey);
          if (paperResult.success && paperResult.paper) {
            related.push({
              paper: paperResult.paper,
              relevanceScore: 0.8,
              relationshipType: "recommended",
            });
          }
        }
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
    }
  }

  // 3. Find papers with citation overlap
  if (includeCitationOverlap && references.length > 0) {
    // Find papers that share multiple references
    const referenceIds = references.slice(0, 20).map((r) => r.paperId);

    for (const refId of referenceIds.slice(0, 5)) {
      const refCitations = await fetchCitations(refId, { limit: 20, apiKey });

      for (const refCitation of refCitations.slice(0, 5)) {
        if (seenIds.has(refCitation.paperId)) continue;
        if (refCitation.paperId === paperId) continue;
        seenIds.add(refCitation.paperId);

        const paperResult = await fetchByPaperId(refCitation.paperId, apiKey);
        if (paperResult.success && paperResult.paper) {
          related.push({
            paper: paperResult.paper,
            relevanceScore: 0.6,
            relationshipType: "similar",
          });
        }
      }
    }
  }

  // Sort by relevance and limit results
  return related
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, maxResults);
}

// ============================================================================
// Influence Scoring Functions
// ============================================================================

/**
 * Calculate influence score for a paper
 */
export function calculateInfluenceScore(
  citationCount: number,
  influentialCitationCount: number,
  yearsSincePublication?: number
): InfluenceScore {
  // Base score from citation counts
  let score = 0;

  // Weight influential citations more heavily
  score += influentialCitationCount * 3;
  score += citationCount * 0.5;

  // Normalize by years if available (to favor newer impactful papers)
  if (yearsSincePublication && yearsSincePublication > 0) {
    const velocityBonus = (citationCount / yearsSincePublication) * 0.5;
    score += Math.min(velocityBonus, 50); // Cap velocity bonus
  }

  // Determine influence level
  let influenceLevel: InfluenceScore["influenceLevel"];
  if (score >= 500) {
    influenceLevel = "seminal";
  } else if (score >= 200) {
    influenceLevel = "very-high";
  } else if (score >= 50) {
    influenceLevel = "high";
  } else if (score >= 10) {
    influenceLevel = "moderate";
  } else {
    influenceLevel = "low";
  }

  return {
    paperId: "",
    totalCitations: citationCount,
    influentialCitations: influentialCitationCount,
    influenceScore: Math.round(score * 100) / 100,
    influenceLevel,
  };
}

/**
 * Get influence score for a paper
 */
export async function getInfluenceScore(
  paperId: string,
  apiKey?: string
): Promise<InfluenceScore | null> {
  const cacheKey = `influence:${paperId}`;
  const cached = influenceCache.get(cacheKey);
  if (cached) return cached;

  const paperResult = await fetchByPaperId(paperId, apiKey);
  if (!paperResult.success || !paperResult.paper) {
    return null;
  }

  const paper = paperResult.paper;
  const citationCount = paper.metadata.citationCount || 0;
  const influentialCount =
    (paper.metadata.sourceMetadata?.influentialCitationCount as number) || 0;

  // Calculate years since publication
  let yearsSince: number | undefined;
  if (paper.metadata.publishedDate) {
    const pubYear = new Date(paper.metadata.publishedDate).getFullYear();
    yearsSince = new Date().getFullYear() - pubYear;
  }

  const score = calculateInfluenceScore(
    citationCount,
    influentialCount,
    yearsSince
  );
  score.paperId = paperId;

  influenceCache.set(cacheKey, score);
  return score;
}

/**
 * Rank papers by influence
 */
export async function rankByInfluence(
  paperIds: string[],
  apiKey?: string
): Promise<Array<{ paperId: string; score: InfluenceScore }>> {
  const scores = await Promise.all(
    paperIds.map(async (id) => ({
      paperId: id,
      score: await getInfluenceScore(id, apiKey),
    }))
  );

  return scores
    .filter((s): s is { paperId: string; score: InfluenceScore } => s.score !== null)
    .sort((a, b) => b.score.influenceScore - a.score.influenceScore);
}

// ============================================================================
// Graph Building Functions
// ============================================================================

/**
 * Build a citation graph starting from a paper
 */
export async function buildCitationGraph(
  rootPaperId: string,
  options: {
    depth?: number;
    maxNodesPerLevel?: number;
    direction?: "citations" | "references" | "both";
    onlyInfluential?: boolean;
    apiKey?: string;
  } = {}
): Promise<PaperGraph> {
  const {
    depth = 2,
    maxNodesPerLevel = 10,
    direction = "both",
    onlyInfluential = false,
    apiKey,
  } = options;

  const nodes = new Map<string, Paper>();
  const edges: PaperGraph["edges"] = [];
  const visited = new Set<string>();

  // Helper to add paper to graph
  async function addPaper(paperId: string): Promise<Paper | null> {
    if (nodes.has(paperId)) {
      return nodes.get(paperId)!;
    }

    const cached = paperCache.get(paperId);
    if (cached) {
      nodes.set(paperId, cached);
      return cached;
    }

    const result = await fetchByPaperId(paperId, apiKey);
    if (result.success && result.paper) {
      nodes.set(paperId, result.paper);
      paperCache.set(paperId, result.paper);
      return result.paper;
    }
    return null;
  }

  // BFS to build graph
  async function traverse(paperId: string, currentDepth: number) {
    if (currentDepth > depth || visited.has(paperId)) return;
    visited.add(paperId);

    await addPaper(paperId);

    if (currentDepth >= depth) return;

    const fetchPromises: Promise<void>[] = [];

    // Get citations (papers that cite this one)
    if (direction === "citations" || direction === "both") {
      fetchPromises.push(
        (async () => {
          let citations = await fetchCitations(paperId, {
            limit: maxNodesPerLevel * 2,
            apiKey,
          });

          if (onlyInfluential) {
            citations = citations.filter((c) => c.isInfluential);
          }

          for (const citation of citations.slice(0, maxNodesPerLevel)) {
            edges.push({
              from: citation.paperId,
              to: paperId,
              type: "cites",
              isInfluential: citation.isInfluential,
            });
            await traverse(citation.paperId, currentDepth + 1);
          }
        })()
      );
    }

    // Get references (papers this one cites)
    if (direction === "references" || direction === "both") {
      fetchPromises.push(
        (async () => {
          let references = await fetchReferences(paperId, {
            limit: maxNodesPerLevel * 2,
            apiKey,
          });

          if (onlyInfluential) {
            references = references.filter((r) => r.isInfluential);
          }

          for (const reference of references.slice(0, maxNodesPerLevel)) {
            edges.push({
              from: paperId,
              to: reference.paperId,
              type: "cites",
              isInfluential: reference.isInfluential,
            });
            await traverse(reference.paperId, currentDepth + 1);
          }
        })()
      );
    }

    await Promise.all(fetchPromises);
  }

  await traverse(rootPaperId, 0);

  return {
    nodes,
    edges,
    rootPaperId,
    depth,
  };
}

// ============================================================================
// High-Impact Paper Suggestions
// ============================================================================

/**
 * Suggest high-impact papers in a research area
 */
export async function suggestHighImpactPapers(
  query: string,
  options: {
    limit?: number;
    minCitations?: number;
    yearRange?: string;
    apiKey?: string;
  } = {}
): Promise<
  Array<{
    paper: Paper;
    influenceScore: InfluenceScore;
  }>
> {
  const { limit = 10, minCitations = 50, yearRange, apiKey } = options;

  // Search for papers
  const searchResults = await searchSemanticScholar(query, {
    limit: limit * 3, // Get more to filter
    year: yearRange,
    apiKey,
  });

  const results: Array<{ paper: Paper; influenceScore: InfluenceScore }> = [];

  for (const result of searchResults) {
    if (!result.success || !result.paper) continue;

    const paper = result.paper;
    const citationCount = paper.metadata.citationCount || 0;

    if (citationCount < minCitations) continue;

    const score = await getInfluenceScore(paper.id, apiKey);
    if (score) {
      results.push({ paper, influenceScore: score });
    }
  }

  // Sort by influence and return top results
  return results
    .sort((a, b) => b.influenceScore.influenceScore - a.influenceScore.influenceScore)
    .slice(0, limit);
}

// ============================================================================
// Cache Management
// ============================================================================

export function clearCaches(): void {
  citationCache.clear();
  referenceCache.clear();
  paperCache.clear();
  influenceCache.clear();
}

export function getCacheStats(): {
  citations: number;
  references: number;
  papers: number;
  influence: number;
} {
  return {
    citations: citationCache.size(),
    references: referenceCache.size(),
    papers: paperCache.size(),
    influence: influenceCache.size(),
  };
}

// ============================================================================
// Convenience Exports
// ============================================================================

export {
  fetchByPaperId,
  fetchByArxivId,
  fetchByDOI,
  searchSemanticScholar,
} from "../papers/semantic-scholar";
