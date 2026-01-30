/**
 * GitHub Repository Analysis API
 *
 * POST - Analyze a GitHub repository for ML patterns
 */

import { NextRequest, NextResponse } from "next/server";
import { analyzeRepository, RepoAnalysis } from "@/lib/research/github-analyzer";

export const dynamic = "force-dynamic";

/**
 * Fetch repository content from GitHub API
 * Uses raw.githubusercontent.com for README and tree API for file listing
 */
async function fetchGitHubContent(repoUrl: string): Promise<{
  url: string;
  readme: string;
  codeContent: string;
  files: string[];
  requirements?: string;
}> {
  // Extract owner/repo from URL or path
  let owner: string;
  let repo: string;

  const urlMatch = repoUrl.match(/github\.com\/([^/]+)\/([^/]+)/);
  if (urlMatch) {
    owner = urlMatch[1];
    repo = urlMatch[2].replace(/\.git$/, "");
  } else if (repoUrl.includes("/")) {
    [owner, repo] = repoUrl.split("/");
  } else {
    throw new Error("Invalid repository format. Use owner/repo or GitHub URL.");
  }

  const baseUrl = `https://github.com/${owner}/${repo}`;
  const rawUrl = `https://raw.githubusercontent.com/${owner}/${repo}/main`;

  // Fetch README
  let readme = "";
  try {
    const readmeResp = await fetch(`${rawUrl}/README.md`);
    if (readmeResp.ok) {
      readme = await readmeResp.text();
    }
  } catch {
    // Try master branch
    try {
      const readmeResp = await fetch(
        `https://raw.githubusercontent.com/${owner}/${repo}/master/README.md`
      );
      if (readmeResp.ok) {
        readme = await readmeResp.text();
      }
    } catch {
      readme = `# ${repo}\n\nNo README found.`;
    }
  }

  // Fetch file list from API
  let files: string[] = [];
  try {
    const treeResp = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/git/trees/main?recursive=1`,
      {
        headers: {
          Accept: "application/vnd.github.v3+json",
          "User-Agent": "Voice-Clone-Pipeline",
        },
      }
    );
    if (treeResp.ok) {
      const treeData = await treeResp.json();
      files = treeData.tree
        ?.filter((item: { type: string }) => item.type === "blob")
        ?.map((item: { path: string }) => item.path) || [];
    }
  } catch {
    // Ignore file listing errors
  }

  // Fetch requirements.txt if exists
  let requirements: string | undefined;
  try {
    const reqResp = await fetch(`${rawUrl}/requirements.txt`);
    if (reqResp.ok) {
      requirements = await reqResp.text();
    }
  } catch {
    // Ignore
  }

  // Fetch a sample of Python files for code analysis
  let codeContent = "";
  const pyFiles = files.filter(
    (f) =>
      f.endsWith(".py") &&
      !f.includes("test") &&
      !f.startsWith(".")
  ).slice(0, 5);

  for (const pyFile of pyFiles) {
    try {
      const resp = await fetch(`${rawUrl}/${pyFile}`);
      if (resp.ok) {
        const content = await resp.text();
        codeContent += `\n\n# File: ${pyFile}\n${content.slice(0, 2000)}`;
      }
    } catch {
      // Ignore individual file errors
    }
  }

  return {
    url: baseUrl,
    readme,
    codeContent,
    files,
    requirements,
  };
}

/**
 * POST /api/research/github - Analyze GitHub repository
 *
 * Body: { repoUrl: string }
 * Returns: { success: boolean, analysis?: RepoAnalysis, paper?: Paper, error?: string }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { repoUrl } = body;

    if (!repoUrl) {
      return NextResponse.json(
        { success: false, error: "Repository URL is required" },
        { status: 400 }
      );
    }

    // Validate format
    const trimmed = repoUrl.trim();
    const isValidUrl = /github\.com\/[^/]+\/[^/]+/i.test(trimmed);
    const isValidPath = /^[^/]+\/[^/]+$/.test(trimmed);

    if (!isValidUrl && !isValidPath) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid repository format. Use owner/repo or GitHub URL.",
        },
        { status: 400 }
      );
    }

    // Fetch repository content
    const content = await fetchGitHubContent(trimmed);

    // Analyze repository
    const analysis: RepoAnalysis = analyzeRepository(
      content.url,
      content.readme,
      content.codeContent,
      content.files,
      content.requirements
    );

    // Create a paper-like response for the repository
    const paper = {
      id: `github:${analysis.owner}/${analysis.name}`,
      metadata: {
        id: `github:${analysis.owner}/${analysis.name}`,
        title: analysis.name,
        authors: [],
        abstract: analysis.readmeSummary || `GitHub repository: ${analysis.owner}/${analysis.name}`,
        source: "github" as const,
        url: content.url,
        categories: analysis.architectures,
        sourceMetadata: {
          repoPath: `${analysis.owner}/${analysis.name}`,
          framework: analysis.framework,
          architectures: analysis.architectures,
          techniques: analysis.techniques.map((t) => t.name),
          projectType: analysis.projectType,
          complexity: analysis.complexity,
        },
      },
      status: "unread" as const,
      addedAt: new Date().toISOString(),
    };

    return NextResponse.json({
      success: true,
      analysis,
      paper,
    });
  } catch (error) {
    console.error("Error analyzing GitHub repository:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to analyze repository",
      },
      { status: 500 }
    );
  }
}
