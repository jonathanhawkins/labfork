/**
 * Semantic Scholar API Client
 *
 * Fetches paper metadata from Semantic Scholar's public API.
 * Documentation: https://api.semanticscholar.org/api-docs/
 */

import {
  PaperMetadata,
  PaperAuthor,
  PaperFetchResult,
  createPaper,
} from "./types";

// Semantic Scholar API base URL
const S2_API_BASE = "https://api.semanticscholar.org/graph/v1";

// Rate limiting: S2 allows 100 requests per 5 minutes without API key
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 100; // ms between requests

// Fields to request from the API
const PAPER_FIELDS = [
  "paperId",
  "externalIds",
  "url",
  "title",
  "abstract",
  "venue",
  "year",
  "referenceCount",
  "citationCount",
  "influentialCitationCount",
  "isOpenAccess",
  "openAccessPdf",
  "fieldsOfStudy",
  "publicationDate",
  "authors",
  "authors.name",
  "authors.authorId",
  "authors.affiliations",
].join(",");

/**
 * Semantic Scholar URL patterns
 */
const S2_URL_PATTERNS = [
  // https://www.semanticscholar.org/paper/.../<paperid>
  /semanticscholar\.org\/paper\/[^/]+\/([a-f0-9]{40})/i,
  // https://www.semanticscholar.org/paper/<paperid>
  /semanticscholar\.org\/paper\/([a-f0-9]{40})/i,
];

/**
 * DOI patterns
 */
const DOI_PATTERNS = [
  // Standard DOI format: 10.XXXX/...
  /^10\.\d{4,}\/[^\s]+$/i,
  // DOI URL format
  /doi\.org\/(10\.\d{4,}\/[^\s]+)/i,
];

/**
 * Check if a string is a valid DOI
 */
export function isDOI(input: string): boolean {
  const trimmed = input.trim();
  return DOI_PATTERNS.some((pattern) => pattern.test(trimmed));
}

/**
 * Extract DOI from input (direct DOI or URL)
 */
export function parseDOI(input: string): string | null {
  const trimmed = input.trim();

  // Check if it's a standard DOI
  if (DOI_PATTERNS[0].test(trimmed)) {
    return trimmed;
  }

  // Try to extract from URL
  const urlMatch = trimmed.match(DOI_PATTERNS[1]);
  if (urlMatch) {
    return urlMatch[1];
  }

  return null;
}

/**
 * Check if input is a Semantic Scholar URL
 */
export function isSemanticScholarUrl(input: string): boolean {
  return S2_URL_PATTERNS.some((pattern) => pattern.test(input));
}

/**
 * Extract paper ID from Semantic Scholar URL
 */
export function parseSemanticScholarUrl(input: string): string | null {
  for (const pattern of S2_URL_PATTERNS) {
    const match = input.match(pattern);
    if (match) {
      return match[1];
    }
  }
  return null;
}

/**
 * Rate-limited fetch
 */
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

/**
 * Semantic Scholar paper response type
 */
interface S2PaperResponse {
  paperId: string;
  externalIds?: {
    DOI?: string;
    ArXiv?: string;
    CorpusId?: number;
    PubMed?: string;
    MAG?: string;
  };
  url: string;
  title: string;
  abstract?: string;
  venue?: string;
  year?: number;
  referenceCount?: number;
  citationCount?: number;
  influentialCitationCount?: number;
  isOpenAccess?: boolean;
  openAccessPdf?: {
    url: string;
    status: string;
  };
  fieldsOfStudy?: string[];
  publicationDate?: string;
  authors?: {
    authorId?: string;
    name: string;
    affiliations?: string[];
  }[];
}

/**
 * Convert S2 response to PaperMetadata
 */
function s2ResponseToMetadata(response: S2PaperResponse): PaperMetadata {
  const authors: PaperAuthor[] = (response.authors || []).map((a) => ({
    name: a.name,
    authorId: a.authorId,
    affiliation: a.affiliations?.[0],
  }));

  // Try to get arXiv categories from fields of study
  const categories = response.fieldsOfStudy || [];

  return {
    id: response.paperId,
    title: response.title,
    authors,
    abstract: response.abstract || "",
    source: "semantic-scholar",
    url: response.url,
    pdfUrl: response.openAccessPdf?.url,
    publishedDate: response.publicationDate,
    doi: response.externalIds?.DOI,
    citationCount: response.citationCount,
    venue: response.venue,
    categories,
    sourceMetadata: {
      s2PaperId: response.paperId,
      arxivId: response.externalIds?.ArXiv,
      corpusId: response.externalIds?.CorpusId,
      year: response.year,
      referenceCount: response.referenceCount,
      influentialCitationCount: response.influentialCitationCount,
      isOpenAccess: response.isOpenAccess,
      fieldsOfStudy: response.fieldsOfStudy,
    },
  };
}

/**
 * Fetch paper from Semantic Scholar by DOI
 */
export async function fetchByDOI(
  doi: string,
  apiKey?: string
): Promise<PaperFetchResult> {
  try {
    const normalizedDoi = parseDOI(doi);
    if (!normalizedDoi) {
      return {
        success: false,
        error: `Invalid DOI format: ${doi}`,
      };
    }

    const apiUrl = `${S2_API_BASE}/paper/DOI:${encodeURIComponent(normalizedDoi)}?fields=${PAPER_FIELDS}`;
    const response = await rateLimitedFetch(apiUrl, apiKey);

    if (response.status === 404) {
      return {
        success: false,
        error: `Paper not found: ${normalizedDoi}`,
      };
    }

    if (!response.ok) {
      return {
        success: false,
        error: `Semantic Scholar API error: ${response.status} ${response.statusText}`,
      };
    }

    const data: S2PaperResponse = await response.json();
    const metadata = s2ResponseToMetadata(data);
    const paper = createPaper(metadata);

    return {
      success: true,
      paper,
    };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Unknown error fetching from Semantic Scholar",
    };
  }
}

/**
 * Fetch paper from Semantic Scholar by arXiv ID
 */
export async function fetchByArxivId(
  arxivId: string,
  apiKey?: string
): Promise<PaperFetchResult> {
  try {
    // Remove version suffix if present
    const normalizedId = arxivId.replace(/v\d+$/, "");

    const apiUrl = `${S2_API_BASE}/paper/arXiv:${normalizedId}?fields=${PAPER_FIELDS}`;
    const response = await rateLimitedFetch(apiUrl, apiKey);

    if (response.status === 404) {
      return {
        success: false,
        error: `Paper not found on Semantic Scholar: arXiv:${normalizedId}`,
      };
    }

    if (!response.ok) {
      return {
        success: false,
        error: `Semantic Scholar API error: ${response.status}`,
      };
    }

    const data: S2PaperResponse = await response.json();
    const metadata = s2ResponseToMetadata(data);
    const paper = createPaper(metadata);

    return {
      success: true,
      paper,
    };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Unknown error fetching from Semantic Scholar",
    };
  }
}

/**
 * Fetch paper from Semantic Scholar by paper ID
 */
export async function fetchByPaperId(
  paperId: string,
  apiKey?: string
): Promise<PaperFetchResult> {
  try {
    const apiUrl = `${S2_API_BASE}/paper/${paperId}?fields=${PAPER_FIELDS}`;
    const response = await rateLimitedFetch(apiUrl, apiKey);

    if (response.status === 404) {
      return {
        success: false,
        error: `Paper not found: ${paperId}`,
      };
    }

    if (!response.ok) {
      return {
        success: false,
        error: `Semantic Scholar API error: ${response.status}`,
      };
    }

    const data: S2PaperResponse = await response.json();
    const metadata = s2ResponseToMetadata(data);
    const paper = createPaper(metadata);

    return {
      success: true,
      paper,
    };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Unknown error fetching from Semantic Scholar",
    };
  }
}

/**
 * Search Semantic Scholar for papers
 */
export async function searchSemanticScholar(
  query: string,
  options: {
    limit?: number;
    fields?: string[];
    year?: string; // e.g., "2020-2024" or "2024"
    openAccessOnly?: boolean;
    apiKey?: string;
  } = {}
): Promise<PaperFetchResult[]> {
  try {
    const {
      limit = 10,
      year,
      openAccessOnly,
      apiKey,
    } = options;

    let apiUrl = `${S2_API_BASE}/paper/search?query=${encodeURIComponent(query)}&limit=${limit}&fields=${PAPER_FIELDS}`;

    if (year) {
      apiUrl += `&year=${year}`;
    }

    if (openAccessOnly) {
      apiUrl += `&openAccessPdf`;
    }

    const response = await rateLimitedFetch(apiUrl, apiKey);

    if (!response.ok) {
      return [
        {
          success: false,
          error: `Semantic Scholar API error: ${response.status}`,
        },
      ];
    }

    const data = await response.json();
    const papers: PaperFetchResult[] = [];

    for (const item of data.data || []) {
      const metadata = s2ResponseToMetadata(item);
      const paper = createPaper(metadata);
      papers.push({ success: true, paper });
    }

    return papers;
  } catch (error) {
    return [
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unknown error searching Semantic Scholar",
      },
    ];
  }
}

/**
 * Get recommendations for a paper
 */
export async function getRecommendations(
  paperId: string,
  options: {
    limit?: number;
    from?: "all-cs" | "recent";
    apiKey?: string;
  } = {}
): Promise<PaperFetchResult[]> {
  try {
    const { limit = 10, from = "all-cs", apiKey } = options;

    const apiUrl = `${S2_API_BASE}/recommendations/v1/papers/forpaper/${paperId}?limit=${limit}&from=${from}&fields=${PAPER_FIELDS}`;

    const response = await rateLimitedFetch(apiUrl, apiKey);

    if (!response.ok) {
      return [
        {
          success: false,
          error: `Semantic Scholar API error: ${response.status}`,
        },
      ];
    }

    const data = await response.json();
    const papers: PaperFetchResult[] = [];

    for (const item of data.recommendedPapers || []) {
      const metadata = s2ResponseToMetadata(item);
      const paper = createPaper(metadata);
      papers.push({ success: true, paper });
    }

    return papers;
  } catch (error) {
    return [
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unknown error getting recommendations",
      },
    ];
  }
}

/**
 * Fetch paper from Semantic Scholar URL
 */
export async function fetchFromS2Url(
  url: string,
  apiKey?: string
): Promise<PaperFetchResult> {
  const paperId = parseSemanticScholarUrl(url);
  if (!paperId) {
    return {
      success: false,
      error: `Invalid Semantic Scholar URL: ${url}`,
    };
  }
  return fetchByPaperId(paperId, apiKey);
}
