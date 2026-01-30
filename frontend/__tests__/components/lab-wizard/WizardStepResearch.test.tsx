/**
 * WizardStepResearch Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { WizardStepResearch, GoalAnalysisResult } from "@/components/lab-wizard/WizardStepResearch";
import type { ResearchGoal } from "@/lib/lab-wizard/types";

describe("WizardStepResearch", () => {
  const defaultGoal: ResearchGoal = {
    description: "",
    keywords: [],
  };

  const defaultProps = {
    goal: defaultGoal,
    onGoalChange: vi.fn(),
  };

  const mockAnalysis: GoalAnalysisResult = {
    suggestedDomain: "voice-clone",
    domainName: "Voice Cloning",
    domainReason: "Based on your goal, voice cloning is the best fit.",
    arxivCategories: ["cs.SD", "eess.AS"],
    keywords: ["TTS", "prosody", "emotion"],
    reasoning: "Your research goal focuses on speech synthesis.",
    suggestedTasks: [
      {
        subject: "Research TTS architectures",
        description: "Survey recent TTS papers",
        type: "research",
        priority: "high",
        estimatedHours: 8,
      },
      {
        subject: "Implement baseline",
        description: "Create baseline TTS model",
        type: "implementation",
        priority: "medium",
        estimatedHours: 16,
      },
    ],
    suggestedPapers: [
      {
        title: "FastSpeech 2",
        authors: "Ren et al.",
        reason: "State-of-the-art TTS architecture",
      },
    ],
    estimatedTimeline: "About 2 weeks",
    confidenceScore: 0.85,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("rendering", () => {
    it("renders info banner", () => {
      render(<WizardStepResearch {...defaultProps} />);
      expect(screen.getByText(/Describe your research goal/)).toBeDefined();
    });

    it("renders goal input label", () => {
      render(<WizardStepResearch {...defaultProps} />);
      expect(screen.getByText("What do you want to research?")).toBeDefined();
    });

    it("renders goal textarea", () => {
      render(<WizardStepResearch {...defaultProps} />);
      expect(screen.getByPlaceholderText(/Example: I want to create/)).toBeDefined();
    });

    it("renders character count", () => {
      render(<WizardStepResearch {...defaultProps} />);
      expect(screen.getByText("0 / 500")).toBeDefined();
    });

    it("renders keywords input", () => {
      render(<WizardStepResearch {...defaultProps} />);
      expect(screen.getByPlaceholderText(/TTS, prosody, emotion/)).toBeDefined();
    });

    it("renders Analyze with AI button", () => {
      render(<WizardStepResearch {...defaultProps} />);
      expect(screen.getByText("Analyze with AI")).toBeDefined();
    });

    it("renders skip option when no analysis", () => {
      render(<WizardStepResearch {...defaultProps} />);
      expect(screen.getByText(/skip AI analysis/)).toBeDefined();
    });
  });

  describe("goal input", () => {
    it("updates character count when typing", () => {
      const goalWithText: ResearchGoal = {
        description: "Test goal",
        keywords: [],
      };
      render(<WizardStepResearch {...defaultProps} goal={goalWithText} />);
      expect(screen.getByText("9 / 500")).toBeDefined();
    });

    it("calls onGoalChange when description changes", () => {
      const onGoalChange = vi.fn();
      render(<WizardStepResearch {...defaultProps} onGoalChange={onGoalChange} />);

      const textarea = screen.getByPlaceholderText(/Example: I want to create/);
      fireEvent.change(textarea, { target: { value: "New research goal" } });

      expect(onGoalChange).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "New research goal",
        })
      );
    });

    it("truncates description at 500 characters", () => {
      const onGoalChange = vi.fn();
      render(<WizardStepResearch {...defaultProps} onGoalChange={onGoalChange} />);

      const longText = "a".repeat(600);
      const textarea = screen.getByPlaceholderText(/Example: I want to create/);
      fireEvent.change(textarea, { target: { value: longText } });

      expect(onGoalChange).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "a".repeat(500),
        })
      );
    });
  });

  describe("keywords input", () => {
    it("displays existing keywords", () => {
      const goalWithKeywords: ResearchGoal = {
        description: "",
        keywords: ["TTS", "prosody"],
      };
      render(<WizardStepResearch {...defaultProps} goal={goalWithKeywords} />);

      const input = screen.getByPlaceholderText(/TTS, prosody, emotion/) as HTMLInputElement;
      expect(input.value).toBe("TTS, prosody");
    });

    it("calls onGoalChange when keywords change", () => {
      const onGoalChange = vi.fn();
      render(<WizardStepResearch {...defaultProps} onGoalChange={onGoalChange} />);

      const input = screen.getByPlaceholderText(/TTS, prosody, emotion/);
      fireEvent.change(input, { target: { value: "ML, AI, speech" } });

      expect(onGoalChange).toHaveBeenCalledWith(
        expect.objectContaining({
          keywords: ["ML", "AI", "speech"],
        })
      );
    });
  });

  describe("AI analysis", () => {
    it("disables analyze button when description is empty", () => {
      render(<WizardStepResearch {...defaultProps} />);
      const button = screen.getByText("Analyze with AI").closest("button");
      expect(button?.disabled).toBe(true);
    });

    it("enables analyze button when description is provided", () => {
      const goalWithText: ResearchGoal = {
        description: "I want to build a TTS system",
        keywords: [],
      };
      render(<WizardStepResearch {...defaultProps} goal={goalWithText} />);
      const button = screen.getByText("Analyze with AI").closest("button");
      expect(button?.disabled).toBe(false);
    });

    it("button is disabled when description is empty", () => {
      render(<WizardStepResearch {...defaultProps} />);
      const button = screen.getByText("Analyze with AI").closest("button");
      expect(button?.disabled).toBe(true);
    });

    it("calls API when analyze button is clicked", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, analysis: mockAnalysis }),
      });

      const goalWithText: ResearchGoal = {
        description: "I want to build a TTS system",
        keywords: [],
      };
      render(<WizardStepResearch {...defaultProps} goal={goalWithText} />);

      const button = screen.getByText("Analyze with AI");
      fireEvent.click(button);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith(
          "/api/lab/analyze-goal",
          expect.objectContaining({
            method: "POST",
            body: expect.stringContaining("I want to build a TTS system"),
          })
        );
      });
    });

    it("shows loading state during analysis", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockImplementationOnce(
        () =>
          new Promise((resolve) =>
            setTimeout(
              () =>
                resolve({
                  ok: true,
                  json: () => Promise.resolve({ success: true, analysis: mockAnalysis }),
                }),
              100
            )
          )
      );

      const goalWithText: ResearchGoal = {
        description: "I want to build a TTS system",
        keywords: [],
      };
      render(<WizardStepResearch {...defaultProps} goal={goalWithText} />);

      const button = screen.getByText("Analyze with AI");
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText("Analyzing...")).toBeDefined();
      });
    });

    it("shows error when analysis fails", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: false, error: "Analysis failed" }),
      });

      const goalWithText: ResearchGoal = {
        description: "I want to build a TTS system",
        keywords: [],
      };
      render(<WizardStepResearch {...defaultProps} goal={goalWithText} />);

      const button = screen.getByText("Analyze with AI");
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText("Analysis failed")).toBeDefined();
      });
    });
  });

  describe("analysis results display", () => {
    it("shows AI Analysis section when analysis is provided", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("AI Analysis")).toBeDefined();
    });

    it("shows reasoning from analysis", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText(/Your research goal focuses on speech synthesis/)).toBeDefined();
    });

    it("shows arxiv categories from analysis", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("cs.SD")).toBeDefined();
      expect(screen.getByText("eess.AS")).toBeDefined();
    });

    it("shows estimated timeline", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("Estimated Timeline")).toBeDefined();
      expect(screen.getByText("About 2 weeks")).toBeDefined();
    });

    it("shows suggested tasks count", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("Suggested Tasks (2)")).toBeDefined();
    });

    it("shows task subjects", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("Research TTS architectures")).toBeDefined();
      expect(screen.getByText("Implement baseline")).toBeDefined();
    });

    it("shows task priorities", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("high")).toBeDefined();
      expect(screen.getByText("medium")).toBeDefined();
    });

    it("shows task types", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("research")).toBeDefined();
      expect(screen.getByText("implementation")).toBeDefined();
    });

    it("shows estimated hours for tasks", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("Est. 8h")).toBeDefined();
      expect(screen.getByText("Est. 16h")).toBeDefined();
    });

    it("shows suggested papers section", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("Suggested Papers (1)")).toBeDefined();
    });

    it("hides skip option when analysis is provided", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.queryByText(/skip AI analysis/)).toBeNull();
    });
  });

  describe("task selection", () => {
    it("tasks are selected by default after analysis", async () => {
      const onAnalysisChange = vi.fn();
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, analysis: mockAnalysis }),
      });

      const goalWithText: ResearchGoal = {
        description: "I want to build a TTS system",
        keywords: [],
      };
      render(
        <WizardStepResearch
          {...defaultProps}
          goal={goalWithText}
          onAnalysisChange={onAnalysisChange}
        />
      );

      const button = screen.getByText("Analyze with AI");
      fireEvent.click(button);

      await waitFor(() => {
        expect(onAnalysisChange).toHaveBeenCalledWith(mockAnalysis);
      });
    });

    it("shows selected task count", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      // Initially all tasks should be selected
      expect(screen.getByText(/of 2 tasks selected/)).toBeDefined();
    });

    it("calls onGoalChange when task is toggled", () => {
      const onGoalChange = vi.fn();
      render(
        <WizardStepResearch
          {...defaultProps}
          analysis={mockAnalysis}
          onGoalChange={onGoalChange}
        />
      );

      // Click on first task to toggle it
      const taskElement = screen.getByText("Research TTS architectures").closest("div[class*='cursor-pointer']");
      fireEvent.click(taskElement!);

      expect(onGoalChange).toHaveBeenCalled();
    });
  });

  describe("task type colors", () => {
    it("applies blue color for research tasks", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      const researchType = screen.getByText("research");
      expect(researchType.className).toContain("blue");
    });

    it("applies purple color for implementation tasks", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      const implementationType = screen.getByText("implementation");
      expect(implementationType.className).toContain("purple");
    });
  });

  describe("priority colors", () => {
    it("applies red color for high priority", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      const highPriority = screen.getByText("high");
      expect(highPriority.className).toContain("red");
    });

    it("applies yellow color for medium priority", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      const mediumPriority = screen.getByText("medium");
      expect(mediumPriority.className).toContain("yellow");
    });
  });

  describe("collapsible sections", () => {
    it("tasks section is expanded by default", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText("Research TTS architectures")).toBeDefined();
    });

    it("papers section is collapsed by default", () => {
      render(<WizardStepResearch {...defaultProps} analysis={mockAnalysis} />);
      // Papers section header should be visible
      expect(screen.getByText("Suggested Papers (1)")).toBeDefined();
      // But paper content should be hidden initially
      // Clicking header should expand it
      const papersHeader = screen.getByText("Suggested Papers (1)").closest("button");
      fireEvent.click(papersHeader!);
      expect(screen.getByText("FastSpeech 2")).toBeDefined();
    });
  });

  describe("styling", () => {
    it("applies custom className", () => {
      const { container } = render(
        <WizardStepResearch {...defaultProps} className="custom-class" />
      );
      expect(container.firstChild?.className).toContain("custom-class");
    });
  });
});
