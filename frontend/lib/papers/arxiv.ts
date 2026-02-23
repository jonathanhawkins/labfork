/**
 * arXiv API Client
 *
 * Fetches paper metadata from arXiv's public API.
 * Documentation: https://arxiv.org/help/api/user-manual
 */

import {
  PaperMetadata,
  PaperAuthor,
  PaperFetchResult,
  createPaper,
} from "./types";

// arXiv API base URL
const ARXIV_API_BASE = "https://export.arxiv.org/api/query";

// Rate limiting: arXiv requests max 3 requests per second
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 350; // ms

/**
 * arXiv ID patterns
 * - New format: YYMM.NNNNN (e.g., 2401.12345)
 * - Old format: archive/YYMMNNN (e.g., cs/0601001)
 */
const ARXIV_ID_PATTERNS = {
  // New format: 2401.12345 or 2401.12345v1
  new: /^(\d{4}\.\d{4,5})(v\d+)?$/,
  // Old format: cs/0601001 or cs.AI/0601001
  old: /^([a-z-]+(?:\.[A-Z]{2})?)\/(\d{7})(v\d+)?$/,
};

/**
 * arXiv URL patterns
 */
const ARXIV_URL_PATTERNS = [
  // https://arxiv.org/abs/2401.12345
  /arxiv\.org\/abs\/([a-z-]*\/?[\d.]+)(v\d+)?/i,
  // https://arxiv.org/pdf/2401.12345.pdf
  /arxiv\.org\/pdf\/([a-z-]*\/?[\d.]+)(v\d+)?\.pdf/i,
  // https://ar5iv.org/abs/2401.12345
  /ar5iv\.org\/abs\/([a-z-]*\/?[\d.]+)(v\d+)?/i,
  // https://huggingface.co/papers/2401.12345
  /huggingface\.co\/papers\/([a-z-]*\/?[\d.]+)/i,
];

/**
 * Check if a string is a valid arXiv ID
 */
export function isArxivId(input: string): boolean {
  const trimmed = input.trim();
  return (
    ARXIV_ID_PATTERNS.new.test(trimmed) || ARXIV_ID_PATTERNS.old.test(trimmed)
  );
}

/**
 * Parse arXiv ID from various input formats (ID or URL)
 */
export function parseArxivInput(input: string): string | null {
  const trimmed = input.trim();

  // Check if it's already an ID
  if (isArxivId(trimmed)) {
    // Remove version suffix for canonical ID
    return trimmed.replace(/v\d+$/, "");
  }

  // Try to extract from URL
  for (const pattern of ARXIV_URL_PATTERNS) {
    const match = trimmed.match(pattern);
    if (match) {
      return match[1].replace(/v\d+$/, "");
    }
  }

  return null;
}

/**
 * Get the canonical arXiv URL for an ID
 */
export function getArxivUrl(arxivId: string): string {
  return `https://arxiv.org/abs/${arxivId}`;
}

/**
 * Get the PDF URL for an arXiv ID
 */
export function getArxivPdfUrl(arxivId: string): string {
  return `https://arxiv.org/pdf/${arxivId}.pdf`;
}

/**
 * Rate-limited fetch to respect arXiv API limits
 */
async function rateLimitedFetch(url: string): Promise<Response> {
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;

  if (timeSinceLastRequest < MIN_REQUEST_INTERVAL) {
    await new Promise((resolve) =>
      setTimeout(resolve, MIN_REQUEST_INTERVAL - timeSinceLastRequest)
    );
  }

  lastRequestTime = Date.now();
  return fetch(url);
}

/**
 * Parse XML response from arXiv API
 * Note: We parse manually to avoid requiring xml2js in the browser
 */
function parseArxivXml(xmlText: string): ArxivEntry | null {
  try {
    // Check for error
    if (xmlText.includes("<error>")) {
      console.error("arXiv API returned error");
      return null;
    }

    // Check if entry exists
    if (!xmlText.includes("<entry>")) {
      return null;
    }

    // Extract entry section
    const entryMatch = xmlText.match(/<entry>([\s\S]*?)<\/entry>/);
    if (!entryMatch) return null;
    const entry = entryMatch[1];

    // Helper to extract tag content
    const extractTag = (tag: string, content: string = entry): string => {
      const regex = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i");
      const match = content.match(regex);
      return match ? match[1].trim() : "";
    };

    // Extract all tag contents with attributes
    const extractTagWithAttr = (
      tag: string,
      content: string = entry
    ): { content: string; attrs: Record<string, string> }[] => {
      const regex = new RegExp(`<${tag}([^>]*)>([\\s\\S]*?)<\\/${tag}>`, "gi");
      const results: { content: string; attrs: Record<string, string> }[] = [];
      let match;
      while ((match = regex.exec(content)) !== null) {
        const attrs: Record<string, string> = {};
        const attrStr = match[1];
        const attrRegex = /(\w+)="([^"]*)"/g;
        let attrMatch;
        while ((attrMatch = attrRegex.exec(attrStr)) !== null) {
          attrs[attrMatch[1]] = attrMatch[2];
        }
        results.push({ content: match[2].trim(), attrs });
      }
      return results;
    };

    // Extract ID
    const idMatch = entry.match(/<id>([^<]+)<\/id>/);
    const fullId = idMatch ? idMatch[1] : "";
    const arxivId = fullId.split("/abs/").pop()?.replace(/v\d+$/, "") || "";

    // Extract title (may have newlines)
    const title = extractTag("title").replace(/\s+/g, " ");

    // Extract abstract (summary)
    const abstract = extractTag("summary").replace(/\s+/g, " ");

    // Extract authors
    const authorMatches = entry.match(/<author>([\s\S]*?)<\/author>/g) || [];
    const authors: PaperAuthor[] = authorMatches.map((authorXml) => {
      const name = extractTag("name", authorXml);
      const affiliation = extractTag("affiliation", authorXml) || undefined;
      return { name, affiliation };
    });

    // Extract dates
    const published = extractTag("published");
    const updated = extractTag("updated");

    // Extract categories
    const categoryMatches = extractTagWithAttr("category");
    const categories = categoryMatches
      .map((c) => c.attrs.term)
      .filter(Boolean);

    // Extract links
    const links = extractTagWithAttr("link");
    const pdfLink = links.find(
      (l) => l.attrs.title === "pdf" || l.attrs.type === "application/pdf"
    );

    // Extract DOI if present
    const doiMatch = entry.match(/doi\.org\/([^<\s"]+)/i);
    const doi = doiMatch ? doiMatch[1] : undefined;

    return {
      id: arxivId,
      title,
      authors,
      abstract,
      publishedDate: published,
      updatedDate: updated,
      categories,
      pdfUrl: pdfLink?.attrs.href || getArxivPdfUrl(arxivId),
      doi,
    };
  } catch (error) {
    console.error("Failed to parse arXiv XML:", error);
    return null;
  }
}

/**
 * Internal arXiv entry type
 */
interface ArxivEntry {
  id: string;
  title: string;
  authors: PaperAuthor[];
  abstract: string;
  publishedDate?: string;
  updatedDate?: string;
  categories: string[];
  pdfUrl?: string;
  doi?: string;
}

/**
 * Convert arXiv entry to PaperMetadata
 */
function arxivEntryToMetadata(entry: ArxivEntry): PaperMetadata {
  return {
    id: entry.id,
    title: entry.title,
    authors: entry.authors,
    abstract: entry.abstract,
    source: "arxiv",
    url: getArxivUrl(entry.id),
    pdfUrl: entry.pdfUrl,
    publishedDate: entry.publishedDate,
    updatedDate: entry.updatedDate,
    categories: entry.categories,
    doi: entry.doi,
    sourceMetadata: {
      arxivId: entry.id,
      primaryCategory: entry.categories[0],
    },
  };
}

/**
 * Fetch paper from arXiv by ID
 */
export async function fetchArxivPaper(
  arxivId: string
): Promise<PaperFetchResult> {
  try {
    // Normalize ID
    const normalizedId = parseArxivInput(arxivId);
    if (!normalizedId) {
      return {
        success: false,
        error: `Invalid arXiv ID format: ${arxivId}`,
      };
    }

    // Build API URL
    const apiUrl = `${ARXIV_API_BASE}?id_list=${normalizedId}`;

    // Fetch with rate limiting
    const response = await rateLimitedFetch(apiUrl);

    if (!response.ok) {
      return {
        success: false,
        error: `arXiv API error: ${response.status} ${response.statusText}`,
      };
    }

    const xmlText = await response.text();

    // Parse response
    const entry = parseArxivXml(xmlText);

    if (!entry) {
      return {
        success: false,
        error: `Paper not found: ${normalizedId}`,
      };
    }

    // Convert to paper
    const metadata = arxivEntryToMetadata(entry);
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
          : "Unknown error fetching from arXiv",
    };
  }
}

/**
 * Search arXiv for papers matching query
 */
export async function searchArxiv(
  query: string,
  options: {
    maxResults?: number;
    categories?: string[];
    sortBy?: "relevance" | "lastUpdatedDate" | "submittedDate";
    sortOrder?: "ascending" | "descending";
  } = {}
): Promise<PaperFetchResult[]> {
  try {
    const {
      maxResults = 10,
      categories = [],
      sortBy = "relevance",
      sortOrder = "descending",
    } = options;

    // Build search query
    let searchQuery = `all:${encodeURIComponent(query)}`;

    // Add category filter
    if (categories.length > 0) {
      const catFilter = categories.map((c) => `cat:${c}`).join("+OR+");
      searchQuery = `(${searchQuery})+AND+(${catFilter})`;
    }

    // Build API URL
    const apiUrl = `${ARXIV_API_BASE}?search_query=${searchQuery}&max_results=${maxResults}&sortBy=${sortBy}&sortOrder=${sortOrder}`;

    // Fetch with rate limiting
    const response = await rateLimitedFetch(apiUrl);

    if (!response.ok) {
      return [
        {
          success: false,
          error: `arXiv API error: ${response.status}`,
        },
      ];
    }

    const xmlText = await response.text();

    // Parse all entries
    const entryMatches = xmlText.match(/<entry>([\s\S]*?)<\/entry>/g) || [];
    const results: PaperFetchResult[] = [];

    for (const entryXml of entryMatches) {
      const entry = parseArxivXml(`<feed>${entryXml}</feed>`);
      if (entry) {
        const metadata = arxivEntryToMetadata(entry);
        const paper = createPaper(metadata);
        results.push({ success: true, paper });
      }
    }

    return results;
  } catch (error) {
    return [
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unknown error searching arXiv",
      },
    ];
  }
}

/**
 * Get recent papers from specific arXiv categories
 */
export async function getRecentArxivPapers(
  categories: string[],
  maxResults: number = 10
): Promise<PaperFetchResult[]> {
  if (categories.length === 0) {
    return [];
  }

  const catQuery = categories.map((c) => `cat:${c}`).join("+OR+");
  const apiUrl = `${ARXIV_API_BASE}?search_query=${catQuery}&max_results=${maxResults}&sortBy=submittedDate&sortOrder=descending`;

  try {
    const response = await rateLimitedFetch(apiUrl);

    if (!response.ok) {
      return [
        {
          success: false,
          error: `arXiv API error: ${response.status}`,
        },
      ];
    }

    const xmlText = await response.text();
    const entryMatches = xmlText.match(/<entry>([\s\S]*?)<\/entry>/g) || [];
    const results: PaperFetchResult[] = [];

    for (const entryXml of entryMatches) {
      const entry = parseArxivXml(`<feed>${entryXml}</feed>`);
      if (entry) {
        const metadata = arxivEntryToMetadata(entry);
        const paper = createPaper(metadata);
        results.push({ success: true, paper });
      }
    }

    return results;
  } catch (error) {
    return [
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unknown error fetching recent papers",
      },
    ];
  }
}
