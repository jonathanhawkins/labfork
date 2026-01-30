import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { detectInputSource, validateInput } from "@/lib/papers/parser";
import { generateTasksFromPaper, estimateTotalEffort } from "@/lib/papers/task-generator";
import type { Paper, PaperAnalysis } from "@/lib/papers/types";

/**
 * Integration tests for the paper ingestion flow
 *
 * Tests the end-to-end flow from input detection to task generation
 */
describe("Paper Ingestion Flow", () => {
  describe("Input Detection -> Validation -> Fetch", () => {
    it("should detect and validate arXiv ID", () => {
      const input = "2401.12345";

      // Step 1: Detect input source
      const detection = detectInputSource(input);
      expect(detection).not.toBeNull();
      expect(detection?.source).toBe("arxiv");
      expect(detection?.identifier).toBe("2401.12345");
      expect(detection?.confidence).toBe(1);

      // Step 2: Validate input
      const validation = validateInput(input);
      expect(validation.valid).toBe(true);
    });

    it("should detect and validate arXiv URL", () => {
      const input = "https://arxiv.org/abs/2401.12345";

      const detection = detectInputSource(input);
      expect(detection).not.toBeNull();
      expect(detection?.source).toBe("arxiv");
      expect(detection?.identifier).toBe("2401.12345");

      const validation = validateInput(input);
      expect(validation.valid).toBe(true);
    });

    it("should detect and validate DOI", () => {
      const input = "10.1234/test.paper.123";

      const detection = detectInputSource(input);
      expect(detection).not.toBeNull();
      expect(detection?.source).toBe("doi");

      const validation = validateInput(input);
      expect(validation.valid).toBe(true);
    });

    it("should detect and validate GitHub URL", () => {
      const input = "https://github.com/openai/whisper";

      const detection = detectInputSource(input);
      expect(detection).not.toBeNull();
      expect(detection?.source).toBe("github");

      const validation = validateInput(input);
      expect(validation.valid).toBe(true);
    });

    it("should reject invalid input", () => {
      const input = "random text that is not a paper reference";

      const detection = detectInputSource(input);
      expect(detection).toBeNull();

      const validation = validateInput(input);
      expect(validation.valid).toBe(false);
      expect(validation.error).toBeDefined();
    });
  });

  describe("Analysis -> Task Generation", () => {
    const mockAnalysis: PaperAnalysis = {
      relevanceScore: 85,
      relevanceReason: "Highly relevant to prosody control in voice cloning",
      complexity: "moderate",
      complexityReason: "Requires integration with existing training pipeline",
      novelty: "Novel approach to disentangling prosody from speaker identity",
      techniques: [
        {
          name: "Prosody Disentanglement",
          description: "Separates prosody features from speaker embeddings",
          isMainContribution: true,
        },
        {
          name: "Cross-attention Conditioning",
          description: "Uses cross-attention for conditioning on prosody",
          isMainContribution: false,
        },
      ],
      resources: [
        { type: "dataset", name: "LibriTTS", required: true },
        { type: "compute", name: "GPU (24GB)", required: true },
        { type: "model", name: "Pretrained encoder", required: false },
      ],
      taskBreakdown: {
        research: {
          title: "Study prosody disentanglement approach",
          description: "Read and understand the paper methodology",
          estimatedHours: 4,
        },
        implementation: {
          title: "Implement prosody conditioning module",
          description: "Add prosody conditioning to training pipeline",
          estimatedHours: 16,
          codeAreas: ["training/prosody.py", "models/encoder.py"],
        },
        evaluation: {
          title: "Evaluate prosody control quality",
          description: "Test prosody transfer and control",
          estimatedHours: 8,
          metrics: ["F0 correlation", "MOS", "Speaker similarity"],
        },
      },
      analyzedAt: new Date().toISOString(),
    };

    const mockPaper: Paper = {
      id: "test-paper-123",
      metadata: {
        title: "Disentangled Prosody Control for Voice Cloning",
        abstract: "We present a novel approach to prosody control...",
        authors: [
          { name: "Alice Chen" },
          { name: "Bob Zhang" },
        ],
        source: "arxiv",
        sourceId: "2401.12345",
        url: "https://arxiv.org/abs/2401.12345",
        pdfUrl: "https://arxiv.org/pdf/2401.12345.pdf",
        publishedDate: "2024-01-15",
      },
      status: "analyzed",
      analysis: mockAnalysis,
      addedAt: new Date().toISOString(),
    };

    it("should generate 3 tasks from analyzed paper", () => {
      const tasks = generateTasksFromPaper(mockPaper, "voice-cloning");

      expect(tasks).not.toBeNull();
      expect(tasks?.research).toBeDefined();
      expect(tasks?.implementation).toBeDefined();
      expect(tasks?.evaluation).toBeDefined();
    });

    it("should set correct task subjects with phase prefixes", () => {
      const tasks = generateTasksFromPaper(mockPaper, "voice-cloning");

      expect(tasks?.research.subject).toContain("[Research]");
      expect(tasks?.implementation.subject).toContain("[Implement]");
      expect(tasks?.evaluation.subject).toContain("[Evaluate]");
    });

    it("should include paper metadata in task metadata", () => {
      const tasks = generateTasksFromPaper(mockPaper, "voice-cloning");

      // Check research task metadata
      expect(tasks?.research.metadata?.paperId).toBe(mockPaper.id);
      expect(tasks?.research.metadata?.paperTitle).toBe(mockPaper.metadata.title);
      expect(tasks?.research.metadata?.paperUrl).toBe(mockPaper.metadata.url);
      expect(tasks?.research.metadata?.domainSlug).toBe("voice-cloning");
    });

    it("should set correct phases in metadata", () => {
      const tasks = generateTasksFromPaper(mockPaper, "voice-cloning");

      expect(tasks?.research.metadata?.phase).toBe("research");
      expect(tasks?.implementation.metadata?.phase).toBe("implementation");
      expect(tasks?.evaluation.metadata?.phase).toBe("evaluation");
    });

    it("should estimate total effort correctly", () => {
      const effort = estimateTotalEffort(mockAnalysis);

      // 4 + 16 + 8 = 28 hours
      expect(effort.hours).toBe(28);
      expect(effort.days).toBe(4); // 28 / 8 = 3.5, rounded up to 4
      expect(effort.description).toBeDefined();
    });

    it("should not generate tasks for paper without analysis", () => {
      const paperWithoutAnalysis: Paper = {
        ...mockPaper,
        status: "fetched",
        analysis: undefined,
      };

      const tasks = generateTasksFromPaper(paperWithoutAnalysis);
      expect(tasks).toBeNull();
    });
  });

  describe("Full Pipeline Simulation", () => {
    it("should handle complete paper ingestion flow", () => {
      // Step 1: User inputs arXiv ID
      const userInput = "2401.12345";

      // Step 2: Detect and validate
      const detection = detectInputSource(userInput);
      expect(detection?.source).toBe("arxiv");

      const validation = validateInput(userInput);
      expect(validation.valid).toBe(true);

      // Step 3: Simulate fetched paper
      const fetchedPaper: Paper = {
        id: "paper-" + Date.now(),
        metadata: {
          title: "Test Paper",
          abstract: "Test abstract",
          authors: [{ name: "Test Author" }],
          source: "arxiv",
          sourceId: "2401.12345",
          url: `https://arxiv.org/abs/${detection?.identifier}`,
        },
        status: "fetched",
        addedAt: new Date().toISOString(),
      };

      expect(fetchedPaper.status).toBe("fetched");

      // Step 4: Simulate analysis
      const analyzedPaper: Paper = {
        ...fetchedPaper,
        status: "analyzed",
        analysis: {
          relevanceScore: 75,
          relevanceReason: "Relevant to our research",
          complexity: "simple",
          complexityReason: "Straightforward implementation",
          novelty: "Incremental improvement",
          techniques: [{ name: "Test technique", description: "...", isMainContribution: true }],
          resources: [],
          taskBreakdown: {
            research: { title: "Research", description: "...", estimatedHours: 2 },
            implementation: { title: "Implement", description: "...", estimatedHours: 8 },
            evaluation: { title: "Evaluate", description: "...", estimatedHours: 4, metrics: [] },
          },
          analyzedAt: new Date().toISOString(),
        },
      };

      expect(analyzedPaper.status).toBe("analyzed");
      expect(analyzedPaper.analysis?.relevanceScore).toBe(75);

      // Step 5: Generate tasks
      const tasks = generateTasksFromPaper(analyzedPaper, "voice-cloning");
      expect(tasks).not.toBeNull();
      expect(tasks?.research).toBeDefined();
      expect(tasks?.implementation).toBeDefined();
      expect(tasks?.evaluation).toBeDefined();

      // Step 6: Estimate effort
      const effort = estimateTotalEffort(analyzedPaper.analysis!);
      expect(effort.hours).toBe(14); // 2 + 8 + 4
    });

    it("should handle multiple input formats consistently", () => {
      const inputs = [
        "2401.12345",
        "https://arxiv.org/abs/2401.12345",
        "https://arxiv.org/pdf/2401.12345.pdf",
        "https://huggingface.co/papers/2401.12345",
      ];

      for (const input of inputs) {
        const detection = detectInputSource(input);
        expect(detection?.source).toBe("arxiv");
        expect(detection?.identifier).toBe("2401.12345");
      }
    });
  });

  describe("Edge Cases", () => {
    it("should handle paper with empty techniques list", () => {
      const paper: Paper = {
        id: "test-1",
        metadata: {
          title: "Test",
          abstract: "Test",
          authors: [],
          source: "arxiv",
          url: "https://arxiv.org/abs/2401.12345",
        },
        status: "analyzed",
        analysis: {
          relevanceScore: 50,
          relevanceReason: "...",
          complexity: "simple",
          complexityReason: "...",
          novelty: "...",
          techniques: [], // Empty
          resources: [],
          taskBreakdown: {
            research: { title: "R", description: "...", estimatedHours: 1 },
            implementation: { title: "I", description: "...", estimatedHours: 1 },
            evaluation: { title: "E", description: "...", estimatedHours: 1, metrics: [] },
          },
          analyzedAt: new Date().toISOString(),
        },
        addedAt: new Date().toISOString(),
      };

      const tasks = generateTasksFromPaper(paper);
      expect(tasks).not.toBeNull();
    });

    it("should handle paper with very high complexity", () => {
      const paper: Paper = {
        id: "test-2",
        metadata: {
          title: "Complex Paper",
          abstract: "Very complex...",
          authors: [{ name: "Expert" }],
          source: "arxiv",
          url: "https://arxiv.org/abs/2401.12345",
        },
        status: "analyzed",
        analysis: {
          relevanceScore: 95,
          relevanceReason: "Critical to research",
          complexity: "research", // Highest complexity
          complexityReason: "Requires PhD-level expertise",
          novelty: "Groundbreaking",
          techniques: [],
          resources: [
            { type: "compute", name: "Multi-GPU cluster", required: true },
          ],
          taskBreakdown: {
            research: { title: "Deep study", description: "...", estimatedHours: 40 },
            implementation: { title: "Complex impl", description: "...", estimatedHours: 120 },
            evaluation: { title: "Extensive eval", description: "...", estimatedHours: 40, metrics: [] },
          },
          analyzedAt: new Date().toISOString(),
        },
        addedAt: new Date().toISOString(),
      };

      const tasks = generateTasksFromPaper(paper);
      expect(tasks).not.toBeNull();

      const effort = estimateTotalEffort(paper.analysis!);
      expect(effort.hours).toBe(200); // 40 + 120 + 40
      expect(effort.days).toBe(25); // 200 / 8 = 25
      expect(effort.description).toContain("Multi-week");
    });

    it("should handle whitespace in input", () => {
      const inputs = [
        "  2401.12345  ",
        "\n2401.12345\n",
        "\t2401.12345\t",
      ];

      for (const input of inputs) {
        const detection = detectInputSource(input);
        expect(detection?.identifier).toBe("2401.12345");
      }
    });
  });
});
