/**
 * Paper Input Parser
 *
 * Detects the source type from user input and routes to the appropriate client.
 */

import type { PaperSource, PaperInputDetection, PaperFetchResult } from "./types";
import { isArxivId, parseArxivInput, fetchArxivPaper } from "./arxiv";
import { isDOI, parseDOI, isSemanticScholarUrl, fetchByDOI, fetchFromS2Url } from "./semantic-scholar";
import { isGitHubUrl, fetchGitHubRepo } from "./github";

/**
 * PDF URL patterns
 */
const PDF_URL_PATTERNS = [
  /\.pdf$/i,
  /\.pdf\?/i,
  /content-type=application\/pdf/i,
];

/**
 * Papers with Code URL pattern
 */
const PAPERS_WITH_CODE_PATTERN = /paperswithcode\.com\/paper\/([^/]+)/i;

/**
 * Hugging Face Papers URL pattern
 */
const HUGGINGFACE_PAPERS_PATTERN = /huggingface\.co\/papers\/(\d{4}\.\d{4,5})/i;

/**
 * Check if input is a PDF URL
 */
export function isPdfUrl(input: string): boolean {
  try {
    const url = new URL(input);
    return PDF_URL_PATTERNS.some((pattern) => pattern.test(url.href));
  } catch {
    return false;
  }
}

/**
 * Check if input is a Papers with Code URL
 */
export function isPapersWithCodeUrl(input: string): boolean {
  return PAPERS_WITH_CODE_PATTERN.test(input);
}

/**
 * Parse Papers with Code URL
 */
export function parsePapersWithCodeUrl(input: string): string | null {
  const match = input.match(PAPERS_WITH_CODE_PATTERN);
  return match ? match[1] : null;
}

/**
 * Check if input is a Hugging Face papers URL
 */
export function isHuggingFacePapersUrl(input: string): boolean {
  return HUGGINGFACE_PAPERS_PATTERN.test(input);
}

/**
 * Parse Hugging Face papers URL to arXiv ID
 */
export function parseHuggingFacePapersUrl(input: string): string | null {
  const match = input.match(HUGGINGFACE_PAPERS_PATTERN);
  return match ? match[1] : null;
}

/**
 * Detect the source type from user input
 */
export function detectInputSource(input: string): PaperInputDetection | null {
  const trimmed = input.trim();

  if (!trimmed) {
    return null;
  }

  // Check for arXiv ID first (most common)
  if (isArxivId(trimmed)) {
    return {
      source: "arxiv",
      identifier: parseArxivInput(trimmed)!,
      originalInput: trimmed,
      confidence: 1.0,
      normalizedUrl: `https://arxiv.org/abs/${parseArxivInput(trimmed)}`,
    };
  }

  // Check for arXiv URL
  const arxivId = parseArxivInput(trimmed);
  if (arxivId) {
    return {
      source: "arxiv",
      identifier: arxivId,
      originalInput: trimmed,
      confidence: 1.0,
      normalizedUrl: `https://arxiv.org/abs/${arxivId}`,
    };
  }

  // Check for Hugging Face papers URL (redirects to arXiv)
  const hfArxivId = parseHuggingFacePapersUrl(trimmed);
  if (hfArxivId) {
    return {
      source: "arxiv",
      identifier: hfArxivId,
      originalInput: trimmed,
      confidence: 1.0,
      normalizedUrl: `https://arxiv.org/abs/${hfArxivId}`,
    };
  }

  // Check for DOI
  if (isDOI(trimmed)) {
    const doi = parseDOI(trimmed);
    return {
      source: "doi",
      identifier: doi!,
      originalInput: trimmed,
      confidence: 1.0,
      normalizedUrl: `https://doi.org/${doi}`,
    };
  }

  // Check for Semantic Scholar URL
  if (isSemanticScholarUrl(trimmed)) {
    return {
      source: "semantic-scholar",
      identifier: trimmed, // Will be parsed by the client
      originalInput: trimmed,
      confidence: 1.0,
    };
  }

  // Check for GitHub URL
  if (isGitHubUrl(trimmed)) {
    return {
      source: "github",
      identifier: trimmed, // Will be parsed by the client
      originalInput: trimmed,
      confidence: 0.9, // Slightly lower confidence as not all repos are papers
    };
  }

  // Check for Papers with Code URL
  if (isPapersWithCodeUrl(trimmed)) {
    const paperId = parsePapersWithCodeUrl(trimmed);
    return {
      source: "papers-with-code",
      identifier: paperId!,
      originalInput: trimmed,
      confidence: 0.9,
    };
  }

  // Check for PDF URL
  if (isPdfUrl(trimmed)) {
    return {
      source: "pdf",
      identifier: trimmed,
      originalInput: trimmed,
      confidence: 0.8, // Lower confidence as we can't verify content
    };
  }

  // Try to detect if it's a URL at all
  try {
    const url = new URL(trimmed);

    // Check for common academic domains
    const academicDomains = [
      "arxiv.org",
      "semanticscholar.org",
      "acm.org",
      "ieee.org",
      "openreview.net",
      "aclweb.org",
      "neurips.cc",
      "proceedings.mlr.press",
    ];

    if (academicDomains.some((d) => url.hostname.includes(d))) {
      return {
        source: "doi", // Default to DOI/S2 for academic URLs
        identifier: trimmed,
        originalInput: trimmed,
        confidence: 0.6, // Lower confidence as we don't know the exact format
      };
    }
  } catch {
    // Not a URL
  }

  // Could not detect source
  return null;
}

/**
 * Get display name for a paper source
 */
export function getSourceDisplayName(source: PaperSource): string {
  switch (source) {
    case "arxiv":
      return "arXiv";
    case "semantic-scholar":
      return "Semantic Scholar";
    case "github":
      return "GitHub";
    case "pdf":
      return "PDF";
    case "doi":
      return "DOI";
    case "papers-with-code":
      return "Papers with Code";
    case "manual":
      return "Manual Entry";
    default:
      return "Unknown";
  }
}

/**
 * Get example inputs for a source type
 */
export function getSourceExamples(source: PaperSource): string[] {
  switch (source) {
    case "arxiv":
      return [
        "2401.12345",
        "https://arxiv.org/abs/2401.12345",
        "https://arxiv.org/pdf/2401.12345.pdf",
      ];
    case "doi":
      return [
        "10.1038/nature12373",
        "https://doi.org/10.1038/nature12373",
      ];
    case "semantic-scholar":
      return [
        "https://www.semanticscholar.org/paper/...",
      ];
    case "github":
      return [
        "https://github.com/owner/repo",
      ];
    case "papers-with-code":
      return [
        "https://paperswithcode.com/paper/attention-is-all-you-need",
      ];
    case "pdf":
      return [
        "https://example.com/paper.pdf",
      ];
    default:
      return [];
  }
}

/**
 * Unified paper fetch function
 *
 * Detects the input type and fetches from the appropriate source.
 */
export async function fetchPaper(
  input: string,
  options: {
    githubToken?: string;
    s2ApiKey?: string;
  } = {}
): Promise<PaperFetchResult> {
  const detection = detectInputSource(input);

  if (!detection) {
    return {
      success: false,
      error: `Could not detect paper source from input: "${input}". Supported formats: arXiv ID/URL, DOI, Semantic Scholar URL, GitHub URL, PDF URL.`,
    };
  }

  const { source, identifier } = detection;

  switch (source) {
    case "arxiv":
      return fetchArxivPaper(identifier);

    case "doi":
      return fetchByDOI(identifier, options.s2ApiKey);

    case "semantic-scholar":
      return fetchFromS2Url(identifier, options.s2ApiKey);

    case "github":
      return fetchGitHubRepo(identifier, options.githubToken);

    case "papers-with-code":
      // Papers with Code doesn't have a public API, try to extract arXiv ID
      // Most PWC papers link to arXiv
      return {
        success: false,
        error: "Papers with Code support coming soon. Please use the arXiv ID instead.",
      };

    case "pdf":
      // For PDF URLs, we would need to download and parse
      return {
        success: false,
        error: "Direct PDF upload support coming soon. Please provide an arXiv ID or DOI instead.",
      };

    default:
      return {
        success: false,
        error: `Unsupported source type: ${source}`,
      };
  }
}

/**
 * Validate that an input can be processed
 */
export function validateInput(input: string): {
  valid: boolean;
  detection?: PaperInputDetection;
  error?: string;
} {
  const trimmed = input.trim();

  if (!trimmed) {
    return {
      valid: false,
      error: "Input is empty",
    };
  }

  if (trimmed.length < 5) {
    return {
      valid: false,
      error: "Input is too short",
    };
  }

  if (trimmed.length > 2000) {
    return {
      valid: false,
      error: "Input is too long",
    };
  }

  const detection = detectInputSource(trimmed);

  if (!detection) {
    return {
      valid: false,
      error: "Could not detect paper source. Please enter an arXiv ID, DOI, or URL.",
    };
  }

  return {
    valid: true,
    detection,
  };
}

/**
 * Get suggestions for fixing invalid input
 */
export function getInputSuggestions(input: string): string[] {
  const suggestions: string[] = [];
  const trimmed = input.trim();

  // Check if it looks like an incomplete arXiv ID
  if (/^\d{4}\.\d{1,4}$/.test(trimmed)) {
    suggestions.push(`Did you mean arXiv ID "${trimmed}" with more digits?`);
  }

  // Check if it looks like a DOI without the prefix
  if (/^\d{4,}\//.test(trimmed)) {
    suggestions.push(`Did you mean DOI "10.${trimmed}"?`);
  }

  // Check if it looks like a URL without protocol
  if (/^(www\.)?[a-z]+\.[a-z]+/i.test(trimmed)) {
    suggestions.push(`Try adding "https://" to the beginning`);
  }

  // Generic suggestions
  if (suggestions.length === 0) {
    suggestions.push("Try entering an arXiv ID like '2401.12345'");
    suggestions.push("Or paste a paper URL from arXiv, Semantic Scholar, or GitHub");
  }

  return suggestions;
}
