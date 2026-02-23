/**
 * Goal Analyzer Utility Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  analyzeGoal,
  generateInitialTasks,
  GOAL_ANALYSIS_SYSTEM_PROMPT,
  generateGoalPrompt,
  parseGoalAnalysisResponse,
  estimateTimeline,
  getTaskTypeInfo,
  getPriorityInfo,
  applyAnalysisToGoal,
  MOCK_GOAL_ANALYSIS,
} from "@/lib/lab-wizard/goal-analyzer";
import type { InitialTask, ResearchGoal } from "@/lib/lab-wizard/types";

describe("goal-analyzer", () => {
  describe("analyzeGoal", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should return analysis for valid goal", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({ success: true, analysis: MOCK_GOAL_ANALYSIS }),
      });

      const result = await analyzeGoal("Build a voice cloning system");

      expect(result.success).toBe(true);
      expect(result.analysis).toBeDefined();
      expect(result.analysis?.suggestedDomain).toBe("voice-clone");
    });

    it("should handle API errors gracefully", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: false, error: "API error" }),
      });

      const result = await analyzeGoal("Test goal");

      expect(result.success).toBe(false);
      expect(result.error).toBe("API error");
    });

    it("should handle network errors", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("Network error")
      );

      const result = await analyzeGoal("Test goal");

      expect(result.success).toBe(false);
      expect(result.error).toContain("Network error");
    });

    it("should include preferred domain in request", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({ success: true, analysis: MOCK_GOAL_ANALYSIS }),
      });

      await analyzeGoal("Test goal", { preferredDomain: "voice-clone" });

      const fetchCall = (global.fetch as ReturnType<typeof vi.fn>).mock
        .calls[0];
      const body = JSON.parse(fetchCall[1].body);

      expect(body.preferredDomain).toBe("voice-clone");
    });

    it("should include hardware VRAM in request", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({ success: true, analysis: MOCK_GOAL_ANALYSIS }),
      });

      await analyzeGoal("Test goal", { hardwareVram: 24 });

      const fetchCall = (global.fetch as ReturnType<typeof vi.fn>).mock
        .calls[0];
      const body = JSON.parse(fetchCall[1].body);

      expect(body.hardwareVram).toBe(24);
    });
  });

  describe("generateInitialTasks", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should return tasks for valid goal", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            tasks: MOCK_GOAL_ANALYSIS.initialTasks,
          }),
      });

      const result = await generateInitialTasks("Build TTS", "voice-clone");

      expect(result.success).toBe(true);
      expect(result.tasks).toBeDefined();
    });

    it("should include generateTasks flag in request", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, tasks: [] }),
      });

      await generateInitialTasks("Test goal", "voice-clone");

      const fetchCall = (global.fetch as ReturnType<typeof vi.fn>).mock
        .calls[0];
      const body = JSON.parse(fetchCall[1].body);

      expect(body.generateTasks).toBe(true);
    });
  });

  describe("GOAL_ANALYSIS_SYSTEM_PROMPT", () => {
    it("should contain instructions for AI", () => {
      expect(GOAL_ANALYSIS_SYSTEM_PROMPT).toContain("research");
      expect(GOAL_ANALYSIS_SYSTEM_PROMPT).toContain("domain");
      expect(GOAL_ANALYSIS_SYSTEM_PROMPT).toContain("arXiv");
      expect(GOAL_ANALYSIS_SYSTEM_PROMPT).toContain("tasks");
    });

    it("should contain JSON format instructions", () => {
      expect(GOAL_ANALYSIS_SYSTEM_PROMPT).toContain("JSON");
      expect(GOAL_ANALYSIS_SYSTEM_PROMPT).toContain("suggestedDomain");
    });
  });

  describe("generateGoalPrompt", () => {
    it("should include goal text in prompt", () => {
      const result = generateGoalPrompt("Build a voice cloning system");

      expect(result).toContain("Build a voice cloning system");
    });

    it("should include preferred domain when provided", () => {
      const result = generateGoalPrompt("Test goal", {
        preferredDomain: "voice-clone",
      });

      expect(result).toContain("voice-clone");
    });

    it("should include hardware VRAM when provided", () => {
      const result = generateGoalPrompt("Test goal", { hardwareVram: 24 });

      expect(result).toContain("24GB");
    });

    it("should mention hardware constraints when VRAM provided", () => {
      const result = generateGoalPrompt("Test goal", { hardwareVram: 8 });

      expect(result).toContain("hardware constraints");
    });
  });

  describe("parseGoalAnalysisResponse", () => {
    it("should parse valid JSON response", () => {
      const response = JSON.stringify({
        suggestedDomain: "voice-clone",
        domainName: "Voice Cloning",
        domainReason: "Because of TTS focus",
        arxivCategories: ["cs.SD"],
        keywords: ["TTS"],
        recommendedPapers: [],
        initialTasks: [],
        estimatedHours: 40,
        confidence: 85,
      });

      const result = parseGoalAnalysisResponse(response);

      expect(result.success).toBe(true);
      expect(result.analysis?.suggestedDomain).toBe("voice-clone");
    });

    it("should handle response with extra text", () => {
      const response = `Here is my analysis:\n${JSON.stringify({
        suggestedDomain: "robotics",
        domainName: "Robotics",
        arxivCategories: ["cs.RO"],
      })}\n\nLet me know if you have questions.`;

      const result = parseGoalAnalysisResponse(response);

      expect(result.success).toBe(true);
      expect(result.analysis?.suggestedDomain).toBe("robotics");
    });

    it("should return error for invalid JSON", () => {
      const response = "This is not valid JSON";

      const result = parseGoalAnalysisResponse(response);

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it("should return error for missing required fields", () => {
      const response = JSON.stringify({
        keywords: ["test"],
      });

      const result = parseGoalAnalysisResponse(response);

      expect(result.success).toBe(false);
      expect(result.error).toContain("Invalid");
    });

    it("should provide default values for optional fields", () => {
      const response = JSON.stringify({
        suggestedDomain: "voice-clone",
        domainName: "Voice Cloning",
      });

      const result = parseGoalAnalysisResponse(response);

      expect(result.success).toBe(true);
      expect(result.analysis?.arxivCategories).toEqual([]);
      expect(result.analysis?.keywords).toEqual([]);
      expect(result.analysis?.recommendedPapers).toEqual([]);
      expect(result.analysis?.initialTasks).toEqual([]);
    });
  });

  describe("estimateTimeline", () => {
    it("should calculate timeline for single task", () => {
      const tasks: InitialTask[] = [
        {
          subject: "Task 1",
          description: "Test",
          type: "research",
          estimatedHours: 8,
        },
      ];

      const result = estimateTimeline(tasks);

      expect(result.totalHours).toBe(8);
      expect(result.description).toContain("week");
    });

    it("should calculate timeline for multiple tasks", () => {
      const tasks: InitialTask[] = [
        { subject: "Task 1", description: "Test", type: "research", estimatedHours: 20 },
        { subject: "Task 2", description: "Test", type: "implementation", estimatedHours: 20 },
      ];

      const result = estimateTimeline(tasks);

      expect(result.totalHours).toBe(40);
      expect(result.weeks).toBe(2);
    });

    it("should handle tasks without estimated hours", () => {
      const tasks: InitialTask[] = [
        { subject: "Task 1", description: "Test", type: "research" },
        { subject: "Task 2", description: "Test", type: "implementation" },
      ];

      const result = estimateTimeline(tasks);

      expect(result.totalHours).toBe(8); // Default 4 hours each
    });

    it("should describe short timelines appropriately", () => {
      const tasks: InitialTask[] = [
        { subject: "Task 1", description: "Test", type: "research", estimatedHours: 4 },
      ];

      const result = estimateTimeline(tasks);

      expect(result.description).toContain("week");
    });

    it("should describe long timelines appropriately", () => {
      const tasks: InitialTask[] = [
        { subject: "Task 1", description: "Test", type: "research", estimatedHours: 160 },
      ];

      const result = estimateTimeline(tasks);

      expect(result.description).toContain("Multi-month");
    });
  });

  describe("getTaskTypeInfo", () => {
    it("should return info for research type", () => {
      const result = getTaskTypeInfo("research");

      expect(result.label).toBe("Research");
      expect(result.color).toContain("blue");
    });

    it("should return info for implementation type", () => {
      const result = getTaskTypeInfo("implementation");

      expect(result.label).toBe("Implementation");
      expect(result.color).toContain("purple");
    });

    it("should return info for evaluation type", () => {
      const result = getTaskTypeInfo("evaluation");

      expect(result.label).toBe("Evaluation");
      expect(result.color).toContain("green");
    });

    it("should return info for setup type", () => {
      const result = getTaskTypeInfo("setup");

      expect(result.label).toBe("Setup");
      expect(result.color).toContain("yellow");
    });
  });

  describe("getPriorityInfo", () => {
    it("should return info for high priority", () => {
      const result = getPriorityInfo("high");

      expect(result.label).toBe("High");
      expect(result.color).toContain("red");
    });

    it("should return info for medium priority", () => {
      const result = getPriorityInfo("medium");

      expect(result.label).toBe("Medium");
      expect(result.color).toContain("yellow");
    });

    it("should return info for low priority", () => {
      const result = getPriorityInfo("low");

      expect(result.label).toBe("Low");
    });
  });

  describe("applyAnalysisToGoal", () => {
    it("should apply analysis to research goal", () => {
      const goal: ResearchGoal = {
        description: "Build voice cloning",
        keywords: [],
      };

      const result = applyAnalysisToGoal(goal, MOCK_GOAL_ANALYSIS);

      expect(result.suggestedDomain).toBe(MOCK_GOAL_ANALYSIS.suggestedDomain);
      expect(result.suggestedCategories).toEqual(
        MOCK_GOAL_ANALYSIS.arxivCategories
      );
      expect(result.suggestedKeywords).toEqual(MOCK_GOAL_ANALYSIS.keywords);
      expect(result.analyzed).toBe(true);
    });

    it("should preserve original goal properties", () => {
      const goal: ResearchGoal = {
        description: "Original goal",
        keywords: ["existing"],
      };

      const result = applyAnalysisToGoal(goal, MOCK_GOAL_ANALYSIS);

      expect(result.description).toBe("Original goal");
    });

    it("should handle null analysis", () => {
      const goal: ResearchGoal = {
        description: "Test goal",
        keywords: [],
      };

      const result = applyAnalysisToGoal(goal, undefined);

      expect(result).toEqual(goal);
    });
  });

  describe("MOCK_GOAL_ANALYSIS", () => {
    it("should have valid structure", () => {
      expect(MOCK_GOAL_ANALYSIS.suggestedDomain).toBeDefined();
      expect(MOCK_GOAL_ANALYSIS.domainName).toBeDefined();
      expect(MOCK_GOAL_ANALYSIS.arxivCategories).toBeInstanceOf(Array);
      expect(MOCK_GOAL_ANALYSIS.keywords).toBeInstanceOf(Array);
      expect(MOCK_GOAL_ANALYSIS.recommendedPapers).toBeInstanceOf(Array);
      expect(MOCK_GOAL_ANALYSIS.initialTasks).toBeInstanceOf(Array);
    });

    it("should have valid initial tasks", () => {
      MOCK_GOAL_ANALYSIS.initialTasks.forEach((task) => {
        expect(task.subject).toBeDefined();
        expect(task.description).toBeDefined();
        expect(task.type).toBeDefined();
      });
    });
  });
});
