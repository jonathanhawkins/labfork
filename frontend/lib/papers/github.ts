/**
 * GitHub Repository Analyzer
 *
 * Fetches repository metadata and extracts paper references from ML repos.
 * Uses GitHub's public API.
 */

import {
  PaperMetadata,
  PaperAuthor,
  PaperFetchResult,
  createPaper,
} from "./types";

// GitHub API base URL
const GITHUB_API_BASE = "https://api.github.com";

// Rate limiting
let lastRequestTime = 0;
const MIN_REQUEST_INTERVAL = 100; // ms

/**
 * GitHub URL patterns
 */
const GITHUB_URL_PATTERNS = [
  // https://github.com/owner/repo
  /github\.com\/([^/]+)\/([^/]+)\/?$/i,
  // https://github.com/owner/repo/...
  /github\.com\/([^/]+)\/([^/]+)/i,
];

/**
 * Check if input is a GitHub URL
 */
export function isGitHubUrl(input: string): boolean {
  return GITHUB_URL_PATTERNS.some((pattern) => pattern.test(input));
}

/**
 * Parse GitHub URL to extract owner and repo
 */
export function parseGitHubUrl(
  input: string
): { owner: string; repo: string } | null {
  for (const pattern of GITHUB_URL_PATTERNS) {
    const match = input.match(pattern);
    if (match) {
      return {
        owner: match[1],
        repo: match[2].replace(/\.git$/, ""),
      };
    }
  }
  return null;
}

/**
 * Rate-limited fetch for GitHub API
 */
async function rateLimitedFetch(
  url: string,
  token?: string
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
    Accept: "application/vnd.github.v3+json",
  };

  if (token) {
    headers.Authorization = `token ${token}`;
  }

  return fetch(url, { headers });
}

/**
 * GitHub repository response type
 */
interface GitHubRepoResponse {
  id: number;
  name: string;
  full_name: string;
  owner: {
    login: string;
    avatar_url: string;
  };
  html_url: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  pushed_at: string;
  stargazers_count: number;
  watchers_count: number;
  forks_count: number;
  language: string | null;
  topics: string[];
  license: {
    key: string;
    name: string;
  } | null;
  default_branch: string;
}

/**
 * Patterns to find arXiv/paper references in README
 */
const PAPER_REFERENCE_PATTERNS = [
  // arXiv links
  /arxiv\.org\/abs\/(\d{4}\.\d{4,5})/gi,
  /arxiv\.org\/pdf\/(\d{4}\.\d{4,5})/gi,
  // arXiv mentions in text
  /arXiv[:\s]*(\d{4}\.\d{4,5})/gi,
  // DOI links
  /doi\.org\/(10\.\d{4,}\/[^\s\)]+)/gi,
  // Paper title patterns (between ** or in headings)
  /(?:paper|publication).*?[:\s]*["']([^"']+)["']/gi,
  // Citation patterns [1] with title
  /\[(\d+)\]\s*([^\n]+arxiv[^\n]+)/gi,
];

/**
 * ML project indicators in README or repo structure
 */
const ML_PROJECT_INDICATORS = [
  "requirements.txt",
  "setup.py",
  "pyproject.toml",
  "train.py",
  "training",
  "model.py",
  "models/",
  "dataset",
  "inference",
  "checkpoints",
  "weights",
  "pytorch",
  "tensorflow",
  "jax",
  "transformers",
  "huggingface",
];

/**
 * Extract paper references from README content
 */
export function extractPaperReferences(
  readmeContent: string
): { type: "arxiv" | "doi" | "title"; value: string }[] {
  const references: { type: "arxiv" | "doi" | "title"; value: string }[] = [];
  const seen = new Set<string>();

  // Find arXiv IDs
  const arxivPatterns = [
    /arxiv\.org\/abs\/(\d{4}\.\d{4,5})/gi,
    /arxiv\.org\/pdf\/(\d{4}\.\d{4,5})/gi,
    /arXiv[:\s]*(\d{4}\.\d{4,5})/gi,
  ];

  for (const pattern of arxivPatterns) {
    let match;
    while ((match = pattern.exec(readmeContent)) !== null) {
      const id = match[1];
      if (!seen.has(id)) {
        seen.add(id);
        references.push({ type: "arxiv", value: id });
      }
    }
  }

  // Find DOIs
  const doiPattern = /doi\.org\/(10\.\d{4,}\/[^\s\)\]]+)/gi;
  let doiMatch;
  while ((doiMatch = doiPattern.exec(readmeContent)) !== null) {
    const doi = doiMatch[1];
    if (!seen.has(doi)) {
      seen.add(doi);
      references.push({ type: "doi", value: doi });
    }
  }

  return references;
}

/**
 * Check if repository appears to be an ML research project
 */
export function isMLProject(
  repo: GitHubRepoResponse,
  readmeContent?: string
): boolean {
  // Check topics
  const mlTopics = [
    "machine-learning",
    "deep-learning",
    "pytorch",
    "tensorflow",
    "transformers",
    "nlp",
    "computer-vision",
    "neural-network",
    "artificial-intelligence",
    "ml",
    "ai",
  ];

  if (repo.topics.some((t) => mlTopics.includes(t.toLowerCase()))) {
    return true;
  }

  // Check language
  if (repo.language === "Python" || repo.language === "Jupyter Notebook") {
    // More likely to be ML if Python
    if (repo.topics.length > 0 || repo.stargazers_count > 10) {
      return true;
    }
  }

  // Check README for ML indicators
  if (readmeContent) {
    const lower = readmeContent.toLowerCase();
    const indicatorCount = ML_PROJECT_INDICATORS.filter((ind) =>
      lower.includes(ind.toLowerCase())
    ).length;
    if (indicatorCount >= 2) {
      return true;
    }
  }

  return false;
}

/**
 * Fetch repository metadata from GitHub
 */
export async function fetchRepoMetadata(
  owner: string,
  repo: string,
  token?: string
): Promise<{ repo: GitHubRepoResponse; readme: string | null } | null> {
  try {
    // Fetch repo info
    const repoUrl = `${GITHUB_API_BASE}/repos/${owner}/${repo}`;
    const repoResponse = await rateLimitedFetch(repoUrl, token);

    if (!repoResponse.ok) {
      console.error(`GitHub API error: ${repoResponse.status}`);
      return null;
    }

    const repoData: GitHubRepoResponse = await repoResponse.json();

    // Try to fetch README
    let readme: string | null = null;
    try {
      const readmeUrl = `${GITHUB_API_BASE}/repos/${owner}/${repo}/readme`;
      const readmeResponse = await rateLimitedFetch(readmeUrl, token);

      if (readmeResponse.ok) {
        const readmeData = await readmeResponse.json();
        // README content is base64 encoded
        readme = atob(readmeData.content);
      }
    } catch {
      // README not found or error decoding
    }

    return { repo: repoData, readme };
  } catch (error) {
    console.error("Failed to fetch GitHub repo:", error);
    return null;
  }
}

/**
 * Convert GitHub repo to PaperMetadata
 * Used when treating a repo as a "paper" for the research system
 */
function repoToMetadata(
  repo: GitHubRepoResponse,
  readme: string | null,
  linkedPapers: { type: "arxiv" | "doi" | "title"; value: string }[]
): PaperMetadata {
  // Try to extract title from README (often first heading)
  let title = repo.name;
  if (readme) {
    const headingMatch = readme.match(/^#\s+(.+)$/m);
    if (headingMatch) {
      title = headingMatch[1].replace(/[*_`]/g, "").trim();
    }
  }

  // Extract abstract from description or README intro
  let abstract = repo.description || "";
  if (readme && !abstract) {
    // Get first paragraph after title
    const paragraphMatch = readme.match(/^#.+\n\n([^#\n][^\n]+)/m);
    if (paragraphMatch) {
      abstract = paragraphMatch[1].replace(/[*_`\[\]]/g, "").trim();
    }
  }

  // Create author from owner
  const authors: PaperAuthor[] = [
    {
      name: repo.owner.login,
      authorId: `github:${repo.owner.login}`,
    },
  ];

  return {
    id: `github:${repo.full_name}`,
    title,
    authors,
    abstract: abstract || `GitHub repository: ${repo.full_name}`,
    source: "github",
    url: repo.html_url,
    publishedDate: repo.created_at,
    updatedDate: repo.pushed_at,
    categories: repo.topics,
    githubUrl: repo.html_url,
    sourceMetadata: {
      repoId: repo.id,
      fullName: repo.full_name,
      stars: repo.stargazers_count,
      forks: repo.forks_count,
      language: repo.language,
      topics: repo.topics,
      license: repo.license?.name,
      linkedPapers,
    },
  };
}

/**
 * Fetch and analyze GitHub repository as a research source
 */
export async function fetchGitHubRepo(
  url: string,
  token?: string
): Promise<PaperFetchResult> {
  try {
    const parsed = parseGitHubUrl(url);
    if (!parsed) {
      return {
        success: false,
        error: `Invalid GitHub URL: ${url}`,
      };
    }

    const data = await fetchRepoMetadata(parsed.owner, parsed.repo, token);
    if (!data) {
      return {
        success: false,
        error: `Repository not found: ${parsed.owner}/${parsed.repo}`,
      };
    }

    // Extract paper references from README
    const linkedPapers = data.readme
      ? extractPaperReferences(data.readme)
      : [];

    // Check if it's an ML project
    const isMl = isMLProject(data.repo, data.readme || undefined);

    if (!isMl && linkedPapers.length === 0) {
      return {
        success: false,
        error: `Repository does not appear to be an ML research project: ${data.repo.full_name}`,
      };
    }

    const metadata = repoToMetadata(data.repo, data.readme, linkedPapers);
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
          : "Unknown error fetching GitHub repository",
    };
  }
}

/**
 * Search GitHub for ML research repositories
 */
export async function searchGitHubRepos(
  query: string,
  options: {
    limit?: number;
    language?: string;
    minStars?: number;
    sort?: "stars" | "updated" | "help-wanted-issues";
    token?: string;
  } = {}
): Promise<PaperFetchResult[]> {
  try {
    const {
      limit = 10,
      language = "python",
      minStars = 10,
      sort = "stars",
      token,
    } = options;

    // Build search query
    let searchQuery = `${query} language:${language} stars:>=${minStars}`;

    // Add ML-related topics
    searchQuery += " (topic:machine-learning OR topic:deep-learning OR topic:pytorch OR topic:tensorflow)";

    const apiUrl = `${GITHUB_API_BASE}/search/repositories?q=${encodeURIComponent(searchQuery)}&sort=${sort}&per_page=${limit}`;

    const response = await rateLimitedFetch(apiUrl, token);

    if (!response.ok) {
      return [
        {
          success: false,
          error: `GitHub API error: ${response.status}`,
        },
      ];
    }

    const data = await response.json();
    const results: PaperFetchResult[] = [];

    for (const repo of data.items || []) {
      // Create basic metadata without fetching README for search results
      const metadata = repoToMetadata(repo, null, []);
      const paper = createPaper(metadata);
      results.push({ success: true, paper });
    }

    return results;
  } catch (error) {
    return [
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Unknown error searching GitHub",
      },
    ];
  }
}

/**
 * Get linked papers from a GitHub repository
 * Returns arXiv IDs and DOIs found in the README
 */
export async function getLinkedPapers(
  url: string,
  token?: string
): Promise<{ type: "arxiv" | "doi" | "title"; value: string }[]> {
  const parsed = parseGitHubUrl(url);
  if (!parsed) {
    return [];
  }

  const data = await fetchRepoMetadata(parsed.owner, parsed.repo, token);
  if (!data || !data.readme) {
    return [];
  }

  return extractPaperReferences(data.readme);
}
