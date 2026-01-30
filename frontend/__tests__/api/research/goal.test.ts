/**
 * Research Goal Analysis API Tests
 *
 * Tests for /api/research/goal endpoint
 */

import { describe, it, expect } from "vitest";
import { NextRequest } from "next/server";
import { POST } from "@/app/api/research/goal/route";

describe("Research Goal Analysis API", () => {
  describe("POST /api/research/goal", () => {
    it("should return 400 if goal is missing", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({}),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toBe("Research goal is required");
    });

    it("should return 400 if goal is too short", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({ goal: "Short goal" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toContain("at least 20 characters");
    });

    it("should return 400 if goal has too few words", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({ goal: "This-is-a-very-long-single-word-goal" }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
      expect(data.error).toContain("at least 4 words");
    });

    it("should analyze valid research goal", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "I want to improve prosody control in voice cloning by using emotional embeddings and style transfer techniques",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.success).toBe(true);
      expect(data.analysis).toBeDefined();
    });

    it("should return analysis with all required fields", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Develop a transformer-based speech synthesis model with improved naturalness",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.analysis.originalGoal).toBeDefined();
      expect(data.analysis.concepts).toBeDefined();
      expect(data.analysis.recommendedDomain).toBeDefined();
      expect(data.analysis.alternativeDomains).toBeDefined();
      expect(data.analysis.paperSuggestions).toBeDefined();
      expect(data.analysis.plan).toBeDefined();
    });

    it("should return paper suggestions", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Build a real-time voice cloning system with low latency and high quality",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.papers).toBeDefined();
      expect(Array.isArray(data.papers)).toBe(true);
    });

    it("should generate paper objects with proper structure", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Research deep learning techniques for speech emotion recognition",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      if (data.papers.length > 0) {
        const paper = data.papers[0];
        expect(paper.id).toBeDefined();
        expect(paper.metadata).toBeDefined();
        expect(paper.metadata.source).toBe("goal");
        expect(paper.status).toBe("suggested");
      }
    });

    it("should return research plan with milestones", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Create an end-to-end neural TTS system with prosody modeling and speaker adaptation",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.analysis.plan).toBeDefined();
      expect(data.analysis.plan.milestones).toBeDefined();
      expect(Array.isArray(data.analysis.plan.milestones)).toBe(true);
    });

    it("should identify techniques from goal", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Implement attention mechanisms and transformer architecture for speech synthesis with improved prosody",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.analysis.techniques).toBeDefined();
      expect(Array.isArray(data.analysis.techniques)).toBe(true);
    });

    it("should extract key concepts from goal", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Study neural audio codecs for efficient speech representation learning",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.analysis.concepts).toBeDefined();
      expect(Array.isArray(data.analysis.concepts)).toBe(true);
      expect(data.analysis.concepts.length).toBeGreaterThan(0);
    });

    it("should recommend appropriate research domain", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Develop a voice cloning model with emotional expression and prosody control",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.analysis.recommendedDomain).toBeDefined();
      expect(data.analysis.recommendedDomain.name).toBeDefined();
      expect(data.analysis.recommendedDomain.slug).toBeDefined();
    });

    it("should provide alternative domains", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({
          goal: "Research multimodal learning for audio-visual speech recognition",
        }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.analysis.alternativeDomains).toBeDefined();
      expect(Array.isArray(data.analysis.alternativeDomains)).toBe(true);
    });

    it("should handle empty goal after trimming", async () => {
      const request = new NextRequest("http://localhost:3000/api/research/goal", {
        method: "POST",
        body: JSON.stringify({ goal: "   " }),
      });

      const response = await POST(request);
      const data = await response.json();

      expect(response.status).toBe(400);
      expect(data.success).toBe(false);
    });
  });
});
