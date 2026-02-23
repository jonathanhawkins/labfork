import { describe, it, expect } from "vitest";
import {
  isGitHubUrl,
  parseGitHubUrl,
  extractPaperReferences,
  isMLProject,
} from "@/lib/papers/github";

describe("GitHub Repository Analyzer", () => {
  describe("isGitHubUrl", () => {
    it("should match GitHub repository URLs", () => {
      expect(isGitHubUrl("https://github.com/user/repo")).toBe(true);
      expect(isGitHubUrl("https://github.com/org/project")).toBe(true);
      expect(isGitHubUrl("http://github.com/user/repo")).toBe(true);
      expect(isGitHubUrl("https://www.github.com/user/repo")).toBe(true);
    });

    it("should match URLs with paths", () => {
      expect(isGitHubUrl("https://github.com/user/repo/tree/main")).toBe(true);
      expect(isGitHubUrl("https://github.com/user/repo/blob/main/README.md")).toBe(true);
    });

    it("should reject non-GitHub URLs", () => {
      expect(isGitHubUrl("https://gitlab.com/user/repo")).toBe(false);
      expect(isGitHubUrl("https://arxiv.org/abs/2401.12345")).toBe(false);
      expect(isGitHubUrl("not-a-url")).toBe(false);
      expect(isGitHubUrl("")).toBe(false);
    });
  });

  describe("parseGitHubUrl", () => {
    it("should extract owner and repo from GitHub URLs", () => {
      expect(parseGitHubUrl("https://github.com/facebook/react")).toEqual({
        owner: "facebook",
        repo: "react",
      });
      expect(parseGitHubUrl("https://github.com/pytorch/pytorch")).toEqual({
        owner: "pytorch",
        repo: "pytorch",
      });
    });

    it("should handle URLs with additional paths", () => {
      expect(
        parseGitHubUrl("https://github.com/user/repo/tree/main/src")
      ).toEqual({
        owner: "user",
        repo: "repo",
      });
    });

    it("should return null for invalid URLs", () => {
      expect(parseGitHubUrl("https://github.com")).toBeNull();
      expect(parseGitHubUrl("https://github.com/user")).toBeNull();
      expect(parseGitHubUrl("not-a-url")).toBeNull();
    });
  });

  describe("extractPaperReferences", () => {
    it("should extract arXiv IDs from README", () => {
      const readme = `
        # My Project
        Based on the paper: https://arxiv.org/abs/2401.12345
        Also see arXiv:2312.00001 for more details.
      `;
      const refs = extractPaperReferences(readme);
      expect(refs).toContainEqual({ type: "arxiv", value: "2401.12345" });
      expect(refs).toContainEqual({ type: "arxiv", value: "2312.00001" });
    });

    it("should extract DOIs from README", () => {
      // DOI extraction only works from doi.org URLs, not bare DOIs
      const readme = `
        # Citation
        Please cite: https://doi.org/10.1234/example
        Or see https://doi.org/10.5678/another
      `;
      const refs = extractPaperReferences(readme);
      expect(refs).toContainEqual({ type: "doi", value: "10.1234/example" });
      expect(refs).toContainEqual({ type: "doi", value: "10.5678/another" });
    });

    it("should return empty array when no references found", () => {
      const readme = `
        # My Project
        This is a simple project with no paper references.
      `;
      expect(extractPaperReferences(readme)).toEqual([]);
    });
  });

  describe("isMLProject", () => {
    // Create mock GitHubRepoResponse objects
    const createMockRepo = (overrides: Partial<{
      name: string;
      description: string | null;
      topics: string[];
      language: string | null;
      stargazers_count: number;
    }>) => ({
      id: 1,
      name: overrides.name || "repo",
      full_name: "owner/repo",
      owner: { login: "owner", avatar_url: "" },
      html_url: "https://github.com/owner/repo",
      description: overrides.description ?? null,
      created_at: "2024-01-01T00:00:00Z",
      updated_at: "2024-01-01T00:00:00Z",
      pushed_at: "2024-01-01T00:00:00Z",
      stargazers_count: overrides.stargazers_count ?? 0,
      watchers_count: 0,
      forks_count: 0,
      language: overrides.language ?? null,
      topics: overrides.topics ?? [],
      license: null,
      default_branch: "main",
    });

    it("should identify ML projects by topics", () => {
      expect(
        isMLProject(createMockRepo({
          topics: ["machine-learning"],
        }))
      ).toBe(true);

      expect(
        isMLProject(createMockRepo({
          topics: ["deep-learning", "neural-network"],
        }))
      ).toBe(true);

      expect(
        isMLProject(createMockRepo({
          topics: ["tensorflow", "nlp"],
        }))
      ).toBe(true);
    });

    it("should identify Python ML projects with stars", () => {
      expect(
        isMLProject(createMockRepo({
          language: "Python",
          stargazers_count: 100,
        }))
      ).toBe(true);
    });

    it("should reject non-ML projects", () => {
      expect(
        isMLProject(createMockRepo({
          name: "my-website",
          description: "A personal website",
          topics: ["web", "html"],
          language: "JavaScript",
        }))
      ).toBe(false);

      expect(
        isMLProject(createMockRepo({
          name: "utils",
          description: "Utility functions",
          topics: [],
          language: "Go",
        }))
      ).toBe(false);
    });
  });
});
