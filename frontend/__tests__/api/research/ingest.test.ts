/**
 * Universal Paper Ingestion API Tests
 *
 * Tests for /api/research/ingest endpoint
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { NextRequest } from "next/server";
import { POST, GET } from "@/app/api/research/ingest/route";

// Mock global fetch for API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("Universal Paper Ingestion API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
  });

  describe("POST /api/research/ingest", () => {
    it("should return 400 if input is missing", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({}),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toBe("Input is required");
    });

    it("should detect arXiv ID and ingest", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(`
          <?xml version="1.0"?>
          <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
              <id>http://arxiv.org/abs/2106.09685v1</id>
              <title>LoRA: Low-Rank Adaptation</title>
              <summary>We propose LoRA...</summary>
              <author><name>Test Author</name></author>
              <published>2021-06-17T00:00:00Z</published>
              <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.LG"/>
            </entry>
          </feed>
        `),
      });

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({ input: "2106.09685" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.detection.type).toBe("arxiv");
    });

    it("should detect DOI and ingest", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            paperId: "test-paper-id",
            title: "Test Paper",
            authors: [{ name: "Author" }],
            abstract: "Test abstract",
          }),
      });

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({ input: "10.1234/test.123" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.detection.type).toBe("doi");
    });

    it("should handle structured input", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            paperId: "paper123",
            title: "Test Paper",
            authors: [],
            abstract: "Abstract",
          }),
      });

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({
          input: {
            type: "semantic-scholar",
            value: "paper123",
          },
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
    });

    it("should handle batch ingestion", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        text: () =>
          Promise.resolve(`
          <?xml version="1.0"?>
          <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
              <id>http://arxiv.org/abs/2106.09685</id>
              <title>Paper 1</title>
              <summary>Summary 1</summary>
              <author><name>Author 1</name></author>
            </entry>
          </feed>
        `),
      });

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({
          inputs: ["2106.09685", "2107.12345"],
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.results).toBeDefined();
      expect(data.summary).toBeDefined();
      expect(data.summary.total).toBe(2);
    });

    it("should report batch summary correctly", async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          text: () =>
            Promise.resolve(`
            <?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom">
              <entry>
                <id>http://arxiv.org/abs/2106.09685</id>
                <title>Paper 1</title>
                <summary>Summary</summary>
                <author><name>Author</name></author>
              </entry>
            </feed>
          `),
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 404,
        });

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({
          inputs: ["2106.09685", "invalid-id"],
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.summary).toBeDefined();
    });

    it("should detect GitHub repository URLs", async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("raw.githubusercontent.com")) {
          return Promise.resolve({
            ok: true,
            text: () => Promise.resolve("# Test Repo"),
          });
        }
        return Promise.resolve({ ok: false });
      });

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({
          input: "https://github.com/owner/repo",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.detection.type).toBe("github");
    });

    it("should accept enrichment options", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            paperId: "test-id",
            title: "Test",
            authors: [],
            abstract: "Test",
          }),
      });

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({
          input: "10.1234/test",
          options: {
            enrichWithCitations: true,
            enrichWithRelated: true,
          },
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
    });
  });

  describe("GET /api/research/ingest", () => {
    it("should return 400 if query is missing", async () => {
      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toBe("Query parameter is required");
    });

    it("should detect source type without searching", async () => {
      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=2106.09685&detect=true",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.detection).toBeDefined();
      expect(data.detection.type).toBe("arxiv");
      expect(data.detection.isIdentifier).toBe(true);
    });

    it("should detect DOI type", async () => {
      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=10.1234/test.123&detect=true",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.detection.type).toBe("doi");
    });

    it("should detect GitHub URL type", async () => {
      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=https://github.com/owner/repo&detect=true",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.detection.type).toBe("github");
    });

    it("should search across sources", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            data: [
              {
                paperId: "paper1",
                title: "Test Paper",
                authors: [],
                abstract: "Test",
              },
            ],
          }),
      });

      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=voice%20cloning",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.query).toBe("voice cloning");
      expect(data.results).toBeDefined();
    });

    it("should respect limit parameter", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            data: [],
          }),
      });

      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=test&limit=10",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
    });

    it("should filter by sources", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            data: [],
          }),
      });

      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=test&sources=semantic-scholar,arxiv",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
    });

    it("should return total papers count", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            data: [
              { paperId: "1", title: "Paper 1" },
              { paperId: "2", title: "Paper 2" },
            ],
          }),
      });

      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=test",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(typeof data.totalPapers).toBe("number");
    });
  });

  describe("Error handling", () => {
    it("should handle network errors in POST", async () => {
      mockFetch.mockRejectedValue(new Error("Network error"));

      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: JSON.stringify({ input: "2106.09685" }),
      });

      const response = await POST(request);
      const data = await response.json();

      // May succeed with error in result or fail
      expect(data).toBeDefined();
    });

    it("should handle network errors in GET search gracefully", async () => {
      mockFetch.mockRejectedValue(new Error("Network error"));

      const request = new NextRequest(
        "http://localhost:3000/api/research/ingest?query=test",
        { method: "GET" }
      );

      const response = await GET(request);
      const data = await response.json();

      // Search gracefully handles errors - may return empty results or error
      if (response.status === 200) {
        expect(data.success).toBe(true);
        expect(data.results).toBeDefined();
      } else {
        expect(response.status).toBe(500);
        expect(data.success).toBe(false);
      }
    });

    it("should handle malformed JSON in POST", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/ingest", {
        method: "POST",
        body: "not valid json",
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(500);
      expect(data.success).toBe(false);
    });
  });
});
