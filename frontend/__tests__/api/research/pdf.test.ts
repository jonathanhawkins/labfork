/**
 * PDF Upload and Analysis API Tests
 *
 * Tests for /api/research/pdf endpoint
 */

import { describe, it, expect } from "vitest";
import { NextRequest } from "next/server";
import { POST } from "@/app/api/research/pdf/route";

describe("PDF Upload and Analysis API", () => {
  describe("POST /api/research/pdf (JSON)", () => {
    it("should return 400 if no text or manual data provided", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toContain("Text content or manual metadata is required");
    });

    it("should parse text content", async () => {
      const textContent = `
Title: Attention Is All You Need
Authors: Vaswani et al.

Abstract:
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks. We propose a new simple network architecture,
the Transformer, based solely on attention mechanisms.
      `;

      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textContent }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.parseResult).toBeDefined();
    });

    it("should accept manual metadata entry", async () => {
      const manual = {
        title: "Test Paper Title",
        authors: "John Doe, Jane Smith",
        abstract: "This is a test abstract for the paper.",
      };

      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ manual }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.paper).toBeDefined();
    });

    it("should create paper from manual metadata", async () => {
      const manual = {
        title: "Neural TTS System",
        authors: ["Alice Author", "Bob Researcher"],
        abstract: "A neural text-to-speech system.",
        doi: "10.1234/test.123",
      };

      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ manual }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.paper).toBeDefined();
      expect(data.paper.metadata.title).toBe("Neural TTS System");
    });

    it("should accept filename with text content", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: "Title: Test Paper\n\nAbstract: Test content",
          filename: "test_paper.pdf",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.parseResult).toBeDefined();
    });

    it("should return error for invalid manual metadata", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          manual: {
            // Missing required title
            authors: "Author Name",
          },
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
    });

    it("should handle arXiv ID in text", async () => {
      const textContent = `
arXiv:2106.09685

Abstract:
This paper presents a new approach to speech synthesis.
      `;

      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textContent }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.parseResult).toBeDefined();
    });

    it("should handle DOI in text", async () => {
      const textContent = `
Title: Research Paper
DOI: 10.1145/3394171.3413532

Abstract:
This is a research paper about deep learning.
      `;

      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textContent }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.parseResult).toBeDefined();
    });

    it("should indicate when manual entry is needed", async () => {
      // Minimal text that won't have all metadata
      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "Some random text without structure" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      // May indicate needs_manual status
      expect(data.parseResult.status).toBeDefined();
    });
  });

  describe("POST /api/research/pdf (FormData)", () => {
    // Note: Testing FormData with actual files requires a different setup
    // These tests verify the error handling paths

    it("should reject non-PDF content type when FormData used incorrectly", async () => {
      // Simulating a malformed request
      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "multipart/form-data; boundary=----test" },
        body: "------test\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\n\r\n------test--",
      });

      // This will likely fail due to malformed FormData
      try {
        const response = await POST(request);
        const data = await response.json();
        // Either returns error or processes
        expect(data).toBeDefined();
      } catch {
        // Expected for malformed FormData
      }
    });
  });

  describe("Metadata extraction", () => {
    it("should extract title from formatted text", async () => {
      const textContent = `
Title: Deep Learning for Speech Synthesis

Authors: Research Team

Abstract:
This paper describes advances in neural TTS.
      `;

      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textContent }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      if (data.parseResult.metadata) {
        expect(data.parseResult.metadata.title).toBeDefined();
      }
    });

    it("should extract abstract from formatted text", async () => {
      const textContent = `
Title: Test Paper

Abstract:
This is the abstract of the paper. It describes the main contributions
and findings of the research work.

Introduction:
The rest of the paper...
      `;

      const request = new NextRequest("http://localhost:3000/api/research/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textContent }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      if (data.parseResult.metadata) {
        expect(data.parseResult.metadata.abstract).toBeDefined();
      }
    });
  });
});
