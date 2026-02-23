/**
 * Universal Paper Ingestion Pipeline
 *
 * Unified interface for ingesting papers from multiple sources:
 * - arXiv
 * - Semantic Scholar
 * - GitHub repositories
 * - PDF uploads
 * - Custom research goals
 * - DOI lookups
 */

import { Paper, PaperMetadata, createPaper } from "../papers/types";
import { analyzeRepository, RepoAnalysis } from "./github-analyzer";
import { analyzeGoal, GoalAnalysis } from "./goal-analyzer";
import { parsePDF, parsePDFFromText, PDFParseResult, toPaperMetadata, fromManualEntry } from "./pdf-parser";
import {
  fetchByArxivId,
  fetchByDOI,
  fetchByPaperId,
  searchSemanticScholar,
  getCitationNetwork,
  findRelatedPapers,
  getInfluenceScore,
  CitationNetwork,
  RelatedPaper,
  InfluenceScore,
} from "./semantic-scholar-enhanced";

// ============================================================================
// Types
// ============================================================================

export type SourceType =
  | "arxiv"
  | "semantic-scholar"
  | "github"
  | "pdf"
  | "doi"
  | "goal"
  | "manual"
  | "unknown";

export interface IngestionInput {
  type: SourceType;
  value: string;
  file?: File;
  metadata?: Partial<PaperMetadata>;
}

export interface IngestionResult {
  success: boolean;
  source: SourceType;
  paper?: Paper;
  papers?: Paper[];
  analysis?: RepoAnalysis | GoalAnalysis | PDFParseResult;
  enrichment?: {
    citationNetwork?: CitationNetwork;
    relatedPapers?: RelatedPaper[];
    influenceScore?: InfluenceScore;
  };
  error?: string;
  warnings?: string[];
}

export interface IngestionOptions {
  enrichWithCitations?: boolean;
  enrichWithRelated?: boolean;
  enrichWithInfluence?: boolean;
  maxRelatedPapers?: number;
  apiKey?: string;
}

export interface DeduplicationResult {
  unique: Paper[];
  duplicates: Array<{
    paper: Paper;
    duplicateOf: string;
  }>;
}

export interface BatchIngestionResult {
  successful: IngestionResult[];
  failed: IngestionResult[];
  deduplicated: DeduplicationResult;
}

// ============================================================================
// Source Detection
// ============================================================================

const ARXIV_PATTERNS = [
  /arxiv\.org\/abs\/(\d{4}\.\d{4,5})/i,
  /arxiv\.org\/pdf\/(\d{4}\.\d{4,5})/i,
  /^(\d{4}\.\d{4,5})(v\d+)?$/,
];

const DOI_PATTERNS = [
  /^10\.\d{4,}\/[^\s]+$/i,
  /doi\.org\/(10\.\d{4,}\/[^\s]+)/i,
];

const GITHUB_PATTERNS = [
  /github\.com\/([^/]+\/[^/]+)/i,
  /^([^/]+\/[^/]+)$/,
];

const S2_PATTERNS = [
  /semanticscholar\.org\/paper\/[^/]+\/([a-f0-9]{40})/i,
  /semanticscholar\.org\/paper\/([a-f0-9]{40})/i,
];

/**
 * Detect the source type from input string
 */
export function detectSourceType(input: string): SourceType {
  const trimmed = input.trim();

  // Check arXiv patterns
  if (ARXIV_PATTERNS.some((p) => p.test(trimmed))) {
    return "arxiv";
  }

  // Check DOI patterns
  if (DOI_PATTERNS.some((p) => p.test(trimmed))) {
    return "doi";
  }

  // Check Semantic Scholar patterns
  if (S2_PATTERNS.some((p) => p.test(trimmed))) {
    return "semantic-scholar";
  }

  // Check GitHub patterns
  if (GITHUB_PATTERNS.some((p) => p.test(trimmed))) {
    // Verify it looks like a valid repo path
    const match = trimmed.match(/([^/]+)\/([^/]+)/);
    if (match && match[1].length > 0 && match[2].length > 0) {
      return "github";
    }
  }

  // Check if it looks like a research goal (natural language)
  if (trimmed.length > 20 && trimmed.split(" ").length > 3) {
    return "goal";
  }

  return "unknown";
}

/**
 * Extract ID from input based on source type
 */
export function extractId(input: string, sourceType: SourceType): string | null {
  const trimmed = input.trim();

  switch (sourceType) {
    case "arxiv": {
      for (const pattern of ARXIV_PATTERNS) {
        const match = trimmed.match(pattern);
        if (match) {
          return match[1].replace(/v\d+$/, ""); // Remove version suffix
        }
      }
      return null;
    }

    case "doi": {
      if (DOI_PATTERNS[0].test(trimmed)) {
        return trimmed;
      }
      const match = trimmed.match(DOI_PATTERNS[1]);
      return match ? match[1] : null;
    }

    case "semantic-scholar": {
      for (const pattern of S2_PATTERNS) {
        const match = trimmed.match(pattern);
        if (match) {
          return match[1];
        }
      }
      return null;
    }

    case "github": {
      // Extract owner/repo from GitHub URL
      const urlMatch = trimmed.match(/github\.com\/([^/]+\/[^/]+)/i);
      if (urlMatch) {
        return urlMatch[1].replace(/\.git$/, "");
      }
      // Bare repo path
      const bareMatch = trimmed.match(/^([^/]+\/[^/]+)$/);
      return bareMatch ? bareMatch[1] : null;
    }

    default:
      return trimmed;
  }
}

// ============================================================================
// Ingestion Functions
// ============================================================================

/**
 * Ingest from arXiv
 */
async function ingestFromArxiv(
  arxivId: string,
  options: IngestionOptions = {}
): Promise<IngestionResult> {
  const result = await fetchByArxivId(arxivId, options.apiKey);

  if (!result.success || !result.paper) {
    return {
      success: false,
      source: "arxiv",
      error: result.error || "Failed to fetch from arXiv",
    };
  }

  const ingestionResult: IngestionResult = {
    success: true,
    source: "arxiv",
    paper: result.paper,
  };

  // Enrich if requested
  if (
    options.enrichWithCitations ||
    options.enrichWithRelated ||
    options.enrichWithInfluence
  ) {
    ingestionResult.enrichment = await enrichPaper(
      result.paper.id,
      options
    );
  }

  return ingestionResult;
}

/**
 * Ingest from DOI
 */
async function ingestFromDOI(
  doi: string,
  options: IngestionOptions = {}
): Promise<IngestionResult> {
  const result = await fetchByDOI(doi, options.apiKey);

  if (!result.success || !result.paper) {
    return {
      success: false,
      source: "doi",
      error: result.error || "Failed to fetch by DOI",
    };
  }

  const ingestionResult: IngestionResult = {
    success: true,
    source: "doi",
    paper: result.paper,
  };

  if (
    options.enrichWithCitations ||
    options.enrichWithRelated ||
    options.enrichWithInfluence
  ) {
    ingestionResult.enrichment = await enrichPaper(
      result.paper.id,
      options
    );
  }

  return ingestionResult;
}

/**
 * Ingest from Semantic Scholar
 */
async function ingestFromSemanticScholar(
  paperId: string,
  options: IngestionOptions = {}
): Promise<IngestionResult> {
  const result = await fetchByPaperId(paperId, options.apiKey);

  if (!result.success || !result.paper) {
    return {
      success: false,
      source: "semantic-scholar",
      error: result.error || "Failed to fetch from Semantic Scholar",
    };
  }

  const ingestionResult: IngestionResult = {
    success: true,
    source: "semantic-scholar",
    paper: result.paper,
  };

  if (
    options.enrichWithCitations ||
    options.enrichWithRelated ||
    options.enrichWithInfluence
  ) {
    ingestionResult.enrichment = await enrichPaper(
      result.paper.id,
      options
    );
  }

  return ingestionResult;
}

/**
 * Fetch GitHub repository content
 * Note: In production, this would use GitHub API or a backend service
 */
async function fetchGitHubContent(repoPath: string): Promise<{
  url: string;
  readme: string;
  codeContent: string;
  files: string[];
  requirements?: string;
}> {
  // For now, return placeholder content
  // In a real implementation, this would fetch from GitHub API
  const url = `https://github.com/${repoPath}`;
  return {
    url,
    readme: `# ${repoPath.split("/")[1] || "Repository"}\n\nPlaceholder README content.`,
    codeContent: "",
    files: [],
    requirements: undefined,
  };
}

/**
 * Ingest from GitHub repository
 */
async function ingestFromGitHub(
  repoPath: string,
  options: IngestionOptions = {}
): Promise<IngestionResult> {
  try {
    // Fetch repository content
    const content = await fetchGitHubContent(repoPath);
    const analysis = analyzeRepository(
      content.url,
      content.readme,
      content.codeContent,
      content.files,
      content.requirements
    );

    // Convert to paper-like metadata if papers are found
    const papers: Paper[] = [];
    const warnings: string[] = [];

    if (analysis.linkedPapers.length === 0) {
      warnings.push("No papers found in repository - using repository as source");
    }

    // For each linked paper, try to fetch the actual paper
    for (const paperRef of analysis.linkedPapers.slice(0, 5)) {
      if (paperRef.type === "arxiv") {
        const result = await ingestFromArxiv(paperRef.id, options);
        if (result.success && result.paper) {
          papers.push(result.paper);
        }
      } else if (paperRef.type === "doi") {
        const result = await ingestFromDOI(paperRef.id, options);
        if (result.success && result.paper) {
          papers.push(result.paper);
        }
      }
    }

    // Create a paper entry for the repository itself
    const repoMetadata: PaperMetadata = {
      id: `github:${repoPath}`,
      title: analysis.name,
      authors: [],
      abstract: analysis.readmeSummary || `GitHub repository: ${repoPath}`,
      source: "github",
      url: `https://github.com/${repoPath}`,
      categories: analysis.architectures,
      sourceMetadata: {
        repoPath,
        framework: analysis.framework,
        architectures: analysis.architectures,
        techniques: analysis.techniques.map(t => t.name),
        projectType: analysis.projectType,
        complexity: analysis.complexity,
      },
    };

    const repoPaper = createPaper(repoMetadata);

    return {
      success: true,
      source: "github",
      paper: repoPaper,
      papers: papers.length > 0 ? papers : undefined,
      analysis,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  } catch (error) {
    return {
      success: false,
      source: "github",
      error:
        error instanceof Error
          ? error.message
          : "Failed to analyze GitHub repository",
    };
  }
}

/**
 * Ingest from PDF
 */
async function ingestFromPDF(
  file: File | string,
  manualMetadata?: Partial<PaperMetadata>
): Promise<IngestionResult> {
  try {
    let parseResult: PDFParseResult;
    let pdfUrl = "";

    if (typeof file === "string") {
      // Text content provided
      parseResult = parsePDFFromText(file, "uploaded.pdf");
    } else {
      // File provided
      parseResult = await parsePDF(file);
      pdfUrl = file.name;
    }

    if (parseResult.status === "error") {
      return {
        success: false,
        source: "pdf",
        analysis: parseResult,
        error: parseResult.error || "Failed to parse PDF",
      };
    }

    // Use paperMetadata if already converted, or convert from extracted metadata
    let metadata: PaperMetadata | null = parseResult.paperMetadata || null;

    if (!metadata && parseResult.metadata) {
      metadata = toPaperMetadata(parseResult.metadata, pdfUrl);
    }

    if (!metadata) {
      return {
        success: false,
        source: "pdf",
        analysis: parseResult,
        error: "Could not extract paper metadata from PDF",
      };
    }

    // Merge with manual metadata if provided
    if (manualMetadata) {
      metadata = { ...metadata, ...manualMetadata };
    }

    const paper = createPaper(metadata);

    return {
      success: true,
      source: "pdf",
      paper,
      analysis: parseResult,
      warnings:
        parseResult.status === "needs_manual"
          ? ["Some fields may need manual verification"]
          : undefined,
    };
  } catch (error) {
    return {
      success: false,
      source: "pdf",
      error:
        error instanceof Error
          ? error.message
          : "Failed to process PDF",
    };
  }
}

/**
 * Ingest from research goal
 */
async function ingestFromGoal(
  goal: string,
  options: IngestionOptions = {}
): Promise<IngestionResult> {
  try {
    const analysis = await analyzeGoal(goal);
    const papers: Paper[] = [];
    const warnings: string[] = [];

    // Try to fetch suggested papers
    for (const suggestion of analysis.paperSuggestions.slice(0, 5)) {
      // Try to search for the paper
      const searchResults = await searchSemanticScholar(suggestion.title, {
        limit: 1,
        apiKey: options.apiKey,
      });

      if (searchResults.length > 0 && searchResults[0].success && searchResults[0].paper) {
        papers.push(searchResults[0].paper);
      }
    }

    if (papers.length === 0) {
      warnings.push("Could not find papers matching suggestions automatically");
    }

    return {
      success: true,
      source: "goal",
      papers,
      analysis,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  } catch (error) {
    return {
      success: false,
      source: "goal",
      error:
        error instanceof Error
          ? error.message
          : "Failed to analyze research goal",
    };
  }
}

/**
 * Ingest from manual entry
 */
function ingestFromManual(
  metadata: Partial<PaperMetadata>
): IngestionResult {
  if (!metadata.title) {
    return {
      success: false,
      source: "manual",
      error: "Title is required for manual entry",
    };
  }

  // fromManualEntry returns PaperMetadata directly
  const paperMetadata = fromManualEntry(
    {
      title: metadata.title,
      authors: metadata.authors?.map((a) => a.name) || [],
      abstract: metadata.abstract,
      doi: metadata.doi,
      publishedDate: metadata.publishedDate,
    },
    metadata.url || ""
  );

  // Merge with provided metadata
  const finalMetadata = { ...paperMetadata, ...metadata };
  const paper = createPaper(finalMetadata);

  return {
    success: true,
    source: "manual",
    paper,
  };
}

// ============================================================================
// Enrichment
// ============================================================================

/**
 * Enrich paper with additional data
 */
async function enrichPaper(
  paperId: string,
  options: IngestionOptions
): Promise<IngestionResult["enrichment"]> {
  const enrichment: IngestionResult["enrichment"] = {};

  if (options.enrichWithCitations) {
    const network = await getCitationNetwork(paperId, {
      apiKey: options.apiKey,
    });
    if (network) {
      enrichment.citationNetwork = network;
    }
  }

  if (options.enrichWithRelated) {
    const related = await findRelatedPapers(paperId, {
      maxResults: options.maxRelatedPapers || 10,
      apiKey: options.apiKey,
    });
    enrichment.relatedPapers = related;
  }

  if (options.enrichWithInfluence) {
    const influence = await getInfluenceScore(paperId, options.apiKey);
    if (influence) {
      enrichment.influenceScore = influence;
    }
  }

  return enrichment;
}

// ============================================================================
// Main Ingestion Function
// ============================================================================

/**
 * Universal ingestion function - auto-detects source and ingests
 */
export async function ingest(
  input: IngestionInput | string,
  options: IngestionOptions = {}
): Promise<IngestionResult> {
  // Normalize input
  let ingestionInput: IngestionInput;

  if (typeof input === "string") {
    const sourceType = detectSourceType(input);
    ingestionInput = {
      type: sourceType,
      value: input,
    };
  } else {
    ingestionInput = input;
  }

  const { type, value, file, metadata } = ingestionInput;

  switch (type) {
    case "arxiv": {
      const arxivId = extractId(value, "arxiv");
      if (!arxivId) {
        return {
          success: false,
          source: "arxiv",
          error: "Invalid arXiv ID format",
        };
      }
      return ingestFromArxiv(arxivId, options);
    }

    case "doi": {
      const doi = extractId(value, "doi");
      if (!doi) {
        return {
          success: false,
          source: "doi",
          error: "Invalid DOI format",
        };
      }
      return ingestFromDOI(doi, options);
    }

    case "semantic-scholar": {
      const paperId = extractId(value, "semantic-scholar");
      if (!paperId) {
        return {
          success: false,
          source: "semantic-scholar",
          error: "Invalid Semantic Scholar paper ID",
        };
      }
      return ingestFromSemanticScholar(paperId, options);
    }

    case "github": {
      const repoPath = extractId(value, "github");
      if (!repoPath) {
        return {
          success: false,
          source: "github",
          error: "Invalid GitHub repository path",
        };
      }
      return ingestFromGitHub(repoPath, options);
    }

    case "pdf": {
      if (file) {
        return ingestFromPDF(file, metadata);
      }
      // Assume value is text content
      return ingestFromPDF(value, metadata);
    }

    case "goal":
      return ingestFromGoal(value, options);

    case "manual":
      return ingestFromManual(metadata || {});

    case "unknown":
    default:
      return {
        success: false,
        source: "unknown",
        error: "Could not detect source type. Please specify the source type.",
      };
  }
}

// ============================================================================
// Batch Ingestion
// ============================================================================

/**
 * Ingest multiple papers at once
 */
export async function ingestBatch(
  inputs: Array<IngestionInput | string>,
  options: IngestionOptions = {}
): Promise<BatchIngestionResult> {
  const results = await Promise.all(
    inputs.map((input) => ingest(input, options))
  );

  const successful = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  // Collect all papers
  const allPapers: Paper[] = [];
  for (const result of successful) {
    if (result.paper) {
      allPapers.push(result.paper);
    }
    if (result.papers) {
      allPapers.push(...result.papers);
    }
  }

  // Deduplicate
  const deduplicated = deduplicatePapers(allPapers);

  return {
    successful,
    failed,
    deduplicated,
  };
}

// ============================================================================
// Deduplication
// ============================================================================

/**
 * Deduplicate papers based on title, DOI, or arXiv ID
 */
export function deduplicatePapers(papers: Paper[]): DeduplicationResult {
  const unique: Paper[] = [];
  const duplicates: DeduplicationResult["duplicates"] = [];
  const seen = new Map<string, string>(); // key -> original paper ID

  for (const paper of papers) {
    const keys = getPaperKeys(paper);
    let isDuplicate = false;
    let duplicateOfId = "";

    for (const key of keys) {
      if (seen.has(key)) {
        isDuplicate = true;
        duplicateOfId = seen.get(key)!;
        break;
      }
    }

    if (isDuplicate) {
      duplicates.push({
        paper,
        duplicateOf: duplicateOfId,
      });
    } else {
      unique.push(paper);
      for (const key of keys) {
        seen.set(key, paper.id);
      }
    }
  }

  return { unique, duplicates };
}

/**
 * Generate deduplication keys for a paper
 */
function getPaperKeys(paper: Paper): string[] {
  const keys: string[] = [];

  // Use ID
  keys.push(`id:${paper.id}`);

  // Use DOI if available
  if (paper.metadata.doi) {
    keys.push(`doi:${paper.metadata.doi.toLowerCase()}`);
  }

  // Use arXiv ID if available
  const arxivId = paper.metadata.sourceMetadata?.arxivId;
  if (arxivId) {
    keys.push(`arxiv:${arxivId}`);
  }

  // Use normalized title
  const normalizedTitle = normalizeTitle(paper.metadata.title);
  if (normalizedTitle.length > 10) {
    keys.push(`title:${normalizedTitle}`);
  }

  return keys;
}

/**
 * Normalize title for comparison
 */
function normalizeTitle(title: string): string {
  if (!title) return "";
  return title
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

// ============================================================================
// Search Across Sources
// ============================================================================

/**
 * Search across multiple sources
 */
export async function searchAcrossSources(
  query: string,
  options: IngestionOptions & {
    sources?: SourceType[];
    limit?: number;
  } = {}
): Promise<IngestionResult[]> {
  const { sources = ["semantic-scholar"], limit = 10 } = options;
  const results: IngestionResult[] = [];

  for (const source of sources) {
    switch (source) {
      case "semantic-scholar": {
        const searchResults = await searchSemanticScholar(query, {
          limit,
          apiKey: options.apiKey,
        });

        for (const result of searchResults) {
          if (result.success && result.paper) {
            results.push({
              success: true,
              source: "semantic-scholar",
              paper: result.paper,
            });
          }
        }
        break;
      }

      // Add more sources as needed
      default:
        break;
    }
  }

  return results;
}

// ============================================================================
// Convenience Exports
// ============================================================================

export type { RepoAnalysis } from "./github-analyzer";
export type { GoalAnalysis } from "./goal-analyzer";
export type { PDFParseResult } from "./pdf-parser";
export type { CitationNetwork, RelatedPaper, InfluenceScore } from "./semantic-scholar-enhanced";
