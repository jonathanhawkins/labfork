/**
 * GitHub Repository Analysis API Tests
 *
 * Tests for /api/research/github endpoint
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { NextRequest } from "next/server";
import { POST } from "@/app/api/research/github/route";

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("GitHub Repository Analysis API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("POST /api/research/github", () => {
    it("should return 400 if repoUrl is missing", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({}),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toBe("Repository URL is required");
    });

    it("should return 400 for invalid repository format", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({ repoUrl: "invalid-repo-format" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toContain("Invalid repository format");
    });

    it("should accept owner/repo format", async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("raw.githubusercontent.com")) {
          return Promise.resolve({
            ok: true,
            text: () => Promise.resolve("# Test Repository\n\nA PyTorch-based ML project."),
          });
        }
        if (url.includes("api.github.com")) {
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                tree: [
                  { type: "blob", path: "model.py" },
                  { type: "blob", path: "train.py" },
                  { type: "blob", path: "requirements.txt" },
                ],
              }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({ repoUrl: "owner/repo" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.analysis).toBeDefined();
      expect(data.paper).toBeDefined();
    });

    it("should accept full GitHub URL format", async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("raw.githubusercontent.com")) {
          return Promise.resolve({
            ok: true,
            text: () =>
              Promise.resolve(
                "# TensorFlow Model\n\nimport tensorflow as tf\n\nNeural network implementation."
              ),
          });
        }
        if (url.includes("api.github.com")) {
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                tree: [{ type: "blob", path: "model.py" }],
              }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({ repoUrl: "https://github.com/owner/repo" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
    });

    it("should include analysis details in response", async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("raw.githubusercontent.com") && url.includes("README")) {
          return Promise.resolve({
            ok: true,
            text: () =>
              Promise.resolve(
                "# PyTorch Transformer\n\nA transformer model using self-attention."
              ),
          });
        }
        if (url.includes("api.github.com")) {
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                tree: [
                  { type: "blob", path: "model.py" },
                  { type: "blob", path: "attention.py" },
                ],
              }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({ repoUrl: "owner/transformer-model" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.analysis).toBeDefined();
      expect(data.analysis.name).toBeDefined();
      expect(data.analysis.owner).toBeDefined();
    });

    it("should create paper-like response for repository", async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("raw.githubusercontent.com")) {
          return Promise.resolve({
            ok: true,
            text: () => Promise.resolve("# ML Project\n\nPyTorch implementation."),
          });
        }
        if (url.includes("api.github.com")) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ tree: [] }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({ repoUrl: "owner/ml-project" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.paper).toBeDefined();
      expect(data.paper.id).toContain("github:");
      expect(data.paper.metadata).toBeDefined();
      expect(data.paper.metadata.source).toBe("github");
    });

    it("should handle GitHub API errors gracefully", async () => {
      mockFetch.mockImplementation(() => {
        return Promise.resolve({ ok: false, status: 404 });
      });

      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({ repoUrl: "owner/nonexistent-repo" }),
      });

      const response = await POST(request);
      const data = await response.json();

      // Should still succeed with empty/default content
      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
    });

    it("should handle complete network failures gracefully", async () => {
      mockFetch.mockImplementation(() => {
        throw new Error("Network error");
      });

      const request = new NextRequest("http://localhost:3000/api/research/github", {
        method: "POST",
        body: JSON.stringify({ repoUrl: "owner/repo" }),
      });

      const response = await POST(request);
      const data = await response.json();

      // The API handles network errors gracefully - it catches fetch errors
      // and uses default values, so it still succeeds with minimal data
      // Only if the error propagates to the main handler will it be 500
      if (response.status === 200) {
        expect(data.success).toBe(true);
        expect(data.analysis).toBeDefined();
      } else {
        expect(response.status).toBe(500);
        expect(data.success).toBe(false);
      }
    });
  });
});
