import { describe, it, expect } from "vitest";
import {
  generateTasksFromPaper,
  estimateTotalEffort,
  getTaskPrefix,
  generateTaskSummary,
} from "@/lib/papers/task-generator";
import type { Paper, PaperAnalysis } from "@/lib/papers/types";

describe("Task Generator", () => {
  const mockAnalysis: PaperAnalysis = {
    relevanceScore: 85,
    relevanceReason: "Highly relevant to voice cloning research",
    complexity: "moderate",
    complexityReason: "Requires understanding of existing codebase",
    novelty: "Novel approach to prosody control",
    techniques: [
      {
        name: "Prosody Embedding",
        description: "Encodes prosody features into embeddings",
        isMainContribution: true,
      },
      {
        name: "Cross-attention",
        description: "Uses cross-attention for conditioning",
        isMainContribution: false,
      },
    ],
    resources: [
      {
        type: "dataset",
        name: "LibriTTS",
        required: true,
      },
      {
        type: "model",
        name: "Pretrained encoder",
        required: false,
      },
    ],
    taskBreakdown: {
      research: {
        title: "Study prosody embedding technique",
        description: "Read and understand the paper methodology",
        estimatedHours: 4,
      },
      implementation: {
        title: "Implement prosody conditioning",
        description: "Add prosody conditioning to the model",
        estimatedHours: 16,
        codeAreas: ["training/", "models/"],
      },
      evaluation: {
        title: "Evaluate prosody control",
        description: "Test the implementation",
        estimatedHours: 8,
        metrics: ["F0 correlation", "MOS"],
      },
    },
    analyzedAt: new Date().toISOString(),
  };

  const mockPaper: Paper = {
    id: "test-paper-123",
    metadata: {
      title: "Test Paper on Prosody Control",
      abstract: "This paper presents a novel approach...",
      authors: [{ name: "John Doe" }, { name: "Jane Smith" }],
      source: "arxiv",
      sourceId: "2401.12345",
      url: "https://arxiv.org/abs/2401.12345",
      pdfUrl: "https://arxiv.org/pdf/2401.12345.pdf",
    },
    status: "analyzed",
    analysis: mockAnalysis,
    addedAt: new Date().toISOString(),
  };

  describe("generateTasksFromPaper", () => {
    it("should generate 3 tasks from analyzed paper", () => {
      const tasks = generateTasksFromPaper(mockPaper);
      expect(tasks).not.toBeNull();
      expect(tasks?.research).toBeDefined();
      expect(tasks?.implementation).toBeDefined();
      expect(tasks?.evaluation).toBeDefined();
    });

    it("should include paper metadata in task metadata", () => {
      const tasks = generateTasksFromPaper(mockPaper);
      expect(tasks?.research.metadata?.paperId).toBe(mockPaper.id);
      expect(tasks?.research.metadata?.paperTitle).toBe(mockPaper.metadata.title);
      expect(tasks?.research.metadata?.paperUrl).toBe(mockPaper.metadata.url);
    });

    it("should set correct task phases", () => {
      const tasks = generateTasksFromPaper(mockPaper);
      expect(tasks?.research.metadata?.phase).toBe("research");
      expect(tasks?.implementation.metadata?.phase).toBe("implementation");
      expect(tasks?.evaluation.metadata?.phase).toBe("evaluation");
    });

    it("should include domain slug when provided", () => {
      const tasks = generateTasksFromPaper(mockPaper, "voice-cloning");
      expect(tasks?.research.metadata?.domainSlug).toBe("voice-cloning");
    });

    it("should return null for paper without analysis", () => {
      const paperWithoutAnalysis: Paper = {
        ...mockPaper,
        analysis: undefined,
      };
      expect(generateTasksFromPaper(paperWithoutAnalysis)).toBeNull();
    });

    it("should prefix task subjects with phase markers", () => {
      const tasks = generateTasksFromPaper(mockPaper);
      expect(tasks?.research.subject).toContain("[Research]");
      expect(tasks?.implementation.subject).toContain("[Implement]");
      expect(tasks?.evaluation.subject).toContain("[Evaluate]");
    });
  });

  describe("estimateTotalEffort", () => {
    it("should calculate total hours correctly", () => {
      const effort = estimateTotalEffort(mockAnalysis);
      expect(effort.hours).toBe(4 + 16 + 8); // 28 hours
    });

    it("should calculate days correctly", () => {
      const effort = estimateTotalEffort(mockAnalysis);
      expect(effort.days).toBe(4); // 28 hours / 8 = 3.5, rounded up to 4
    });

    it("should provide appropriate description", () => {
      const effort = estimateTotalEffort(mockAnalysis);
      expect(effort.description).toBeDefined();
      expect(effort.description.length).toBeGreaterThan(0);
    });

    it("should handle small effort", () => {
      const smallAnalysis = {
        ...mockAnalysis,
        taskBreakdown: {
          ...mockAnalysis.taskBreakdown,
          research: { ...mockAnalysis.taskBreakdown.research, estimatedHours: 2 },
          implementation: { ...mockAnalysis.taskBreakdown.implementation, estimatedHours: 4 },
          evaluation: { ...mockAnalysis.taskBreakdown.evaluation, estimatedHours: 2 },
        },
      };
      const effort = estimateTotalEffort(smallAnalysis);
      expect(effort.hours).toBe(8);
      expect(effort.description).toContain("day");
    });
  });

  describe("getTaskPrefix", () => {
    it("should return correct prefix for complexity levels", () => {
      expect(getTaskPrefix("simple")).toBe("[Quick]");
      expect(getTaskPrefix("moderate")).toBe("");
      expect(getTaskPrefix("complex")).toBe("[Deep]");
      expect(getTaskPrefix("research")).toBe("[Exp]");
    });

    it("should return empty string for unknown complexity", () => {
      expect(getTaskPrefix("unknown")).toBe("");
    });
  });

  describe("generateTaskSummary", () => {
    it("should generate markdown summary", () => {
      const tasks = generateTasksFromPaper(mockPaper)!;
      const summary = generateTaskSummary(mockPaper, tasks);

      expect(summary).toContain(mockPaper.metadata.title);
      expect(summary).toContain("Relevance");
      expect(summary).toContain("85/100");
      expect(summary).toContain("Complexity");
      expect(summary).toContain("moderate");
      expect(summary).toContain("Tasks to Create");
      expect(summary).toContain("Dependencies");
    });

    it("should return error message for paper without analysis", () => {
      const paperWithoutAnalysis: Paper = {
        ...mockPaper,
        analysis: undefined,
      };
      const tasks = generateTasksFromPaper(mockPaper)!;
      const summary = generateTaskSummary(paperWithoutAnalysis, tasks);
      expect(summary).toContain("No analysis available");
    });
  });
});
