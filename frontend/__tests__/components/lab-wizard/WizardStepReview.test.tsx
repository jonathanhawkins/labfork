/**
 * WizardStepReview Component Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { WizardStepReview } from "@/components/lab-wizard/WizardStepReview";
import type { LabConfig } from "@/lib/lab-wizard/types";
import type { GoalAnalysisResult } from "@/components/lab-wizard/WizardStepResearch";

describe("WizardStepReview", () => {
  const defaultConfig: LabConfig = {
    createNewDomain: true,
    domain: {
      name: "Voice Clone",
      slug: "voice-clone",
      description: "Speech synthesis research",
      research: {
        arxivCategories: ["cs.SD"],
        keywords: ["TTS"],
      },
      branding: {
        primaryColor: "#3b82f6",
        accentColor: "#22c55e",
        backgroundStyle: "sky",
      },
    },
    hardware: {
      type: "local",
      local: {
        gpu: {
          name: "RTX 4090",
          vram: 24,
          available: true,
        },
      },
    },
    research: {
      path: "goal",
      goal: {
        description: "Build a TTS system with emotion control",
        keywords: ["TTS", "emotion"],
        initialTasks: [
          {
            subject: "Research TTS",
            description: "Survey papers",
            type: "research",
            estimatedHours: 8,
          },
        ],
      },
    },
  };

  const mockAnalysis: GoalAnalysisResult = {
    suggestedDomain: "voice-clone",
    domainName: "Voice Cloning",
    arxivCategories: ["cs.SD"],
    keywords: ["TTS"],
    estimatedTimeline: "About 2 weeks",
  };

  const defaultProps = {
    config: defaultConfig,
    onEditStep: vi.fn(),
    onCreateLab: vi.fn().mockResolvedValue(undefined),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders summary header", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Review Your Lab Configuration")).toBeDefined();
    });

    it("renders subheader", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText(/Verify everything looks correct/)).toBeDefined();
    });

    it("renders Launch Lab button", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Launch Lab")).toBeDefined();
    });

    it("renders help text", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText(/create your domain configuration/)).toBeDefined();
    });
  });

  describe("domain section", () => {
    it("renders domain name", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Voice Clone")).toBeDefined();
    });

    it("indicates new domain", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText(/New domain/)).toBeDefined();
    });

    it("renders domain slug", () => {
      render(<WizardStepReview {...defaultProps} />);
      // Multiple elements contain voice-clone, so use getAllByText
      const elements = screen.getAllByText(/voice-clone/);
      expect(elements.length).toBeGreaterThan(0);
    });

    it("renders domain description when provided", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Speech synthesis research")).toBeDefined();
    });

    it("renders domain arxiv categories", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("cs.SD")).toBeDefined();
    });

    it("renders Edit button for domain", () => {
      render(<WizardStepReview {...defaultProps} />);
      const editButtons = screen.getAllByText("Edit");
      expect(editButtons.length).toBeGreaterThan(0);
    });

    it("calls onEditStep with 'domain' when Edit is clicked", () => {
      const onEditStep = vi.fn();
      render(<WizardStepReview {...defaultProps} onEditStep={onEditStep} />);

      const editButtons = screen.getAllByText("Edit");
      fireEvent.click(editButtons[0]); // First Edit button is for domain

      expect(onEditStep).toHaveBeenCalledWith("domain");
    });

    it("shows existing domain label for non-new domain", () => {
      const existingConfig: LabConfig = {
        ...defaultConfig,
        createNewDomain: false,
        existingDomainSlug: "robotics",
        domain: undefined,
      };
      render(<WizardStepReview {...defaultProps} config={existingConfig} />);
      expect(screen.getByText(/Existing domain/)).toBeDefined();
    });
  });

  describe("hardware section", () => {
    it("renders local machine label", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Local Machine")).toBeDefined();
    });

    it("renders GPU info", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("RTX 4090 (24GB)")).toBeDefined();
    });

    it("renders Edit button for hardware", () => {
      const onEditStep = vi.fn();
      render(<WizardStepReview {...defaultProps} onEditStep={onEditStep} />);

      const editButtons = screen.getAllByText("Edit");
      fireEvent.click(editButtons[1]); // Second Edit button is for hardware

      expect(onEditStep).toHaveBeenCalledWith("hardware");
    });

    it("shows SSH info for remote-ssh type", () => {
      const sshConfig: LabConfig = {
        ...defaultConfig,
        hardware: {
          type: "remote-ssh",
          ssh: {
            host: "192.168.1.100",
            port: 22,
            user: "admin",
            verified: true,
          },
        },
      };
      render(<WizardStepReview {...defaultProps} config={sshConfig} />);
      expect(screen.getByText("Remote SSH")).toBeDefined();
      expect(screen.getByText("admin@192.168.1.100:22")).toBeDefined();
    });

    it("shows connection verified for verified SSH", () => {
      const sshConfig: LabConfig = {
        ...defaultConfig,
        hardware: {
          type: "remote-ssh",
          ssh: {
            host: "192.168.1.100",
            port: 22,
            user: "admin",
            verified: true,
          },
        },
      };
      render(<WizardStepReview {...defaultProps} config={sshConfig} />);
      expect(screen.getByText("Connection verified")).toBeDefined();
    });

    it("shows cloud provider info for cloud type", () => {
      const cloudConfig: LabConfig = {
        ...defaultConfig,
        hardware: {
          type: "cloud",
          cloud: {
            provider: "runpod",
            apiKey: "test-key",
          },
        },
      };
      render(<WizardStepReview {...defaultProps} config={cloudConfig} />);
      expect(screen.getByText("Cloud (runpod)")).toBeDefined();
      expect(screen.getByText("API key configured")).toBeDefined();
    });

    it("shows no API key for cloud without key", () => {
      const cloudConfig: LabConfig = {
        ...defaultConfig,
        hardware: {
          type: "cloud",
          cloud: {
            provider: "aws",
          },
        },
      };
      render(<WizardStepReview {...defaultProps} config={cloudConfig} />);
      expect(screen.getByText("No API key")).toBeDefined();
    });
  });

  describe("research goal section", () => {
    it("renders Research Goal label", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Research Goal")).toBeDefined();
    });

    it("renders truncated goal description", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText(/Build a TTS system/)).toBeDefined();
    });

    it("renders goal keywords", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("TTS")).toBeDefined();
      expect(screen.getByText("emotion")).toBeDefined();
    });

    it("calls onEditStep with 'research' when Edit is clicked", () => {
      const onEditStep = vi.fn();
      render(<WizardStepReview {...defaultProps} onEditStep={onEditStep} />);

      const editButtons = screen.getAllByText("Edit");
      fireEvent.click(editButtons[2]); // Third Edit button is for research

      expect(onEditStep).toHaveBeenCalledWith("research");
    });

    it("shows 'No goal defined' when no goal", () => {
      const noGoalConfig: LabConfig = {
        ...defaultConfig,
        research: {
          path: "goal",
        },
      };
      render(<WizardStepReview {...defaultProps} config={noGoalConfig} />);
      expect(screen.getByText("No goal defined")).toBeDefined();
    });
  });

  describe("tasks section", () => {
    it("renders Initial Tasks label", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Initial Tasks")).toBeDefined();
    });

    it("renders task count", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText(/1 tasks will be created/)).toBeDefined();
    });

    it("renders task subjects", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Research TTS")).toBeDefined();
    });

    it("renders task estimated hours", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("8h")).toBeDefined();
    });

    it("tasks section is expandable", () => {
      render(<WizardStepReview {...defaultProps} />);
      const tasksButton = screen.getByText("Initial Tasks").closest("button");
      fireEvent.click(tasksButton!);
      // Should toggle visibility - tasks still visible after click means it collapsed then needs another click
    });

    it("does not render tasks section when no tasks", () => {
      const noTasksConfig: LabConfig = {
        ...defaultConfig,
        research: {
          path: "goal",
          goal: {
            description: "Test",
            keywords: [],
          },
        },
      };
      render(<WizardStepReview {...defaultProps} config={noTasksConfig} analysis={undefined} />);
      expect(screen.queryByText("Initial Tasks")).toBeNull();
    });
  });

  describe("files section", () => {
    it("renders Files to Create section for new domain", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText("Files to Create")).toBeDefined();
    });

    it("renders file count", () => {
      render(<WizardStepReview {...defaultProps} />);
      expect(screen.getByText(/4 files in .domains/)).toBeDefined();
    });

    it("files section is expandable", () => {
      render(<WizardStepReview {...defaultProps} />);
      const filesButton = screen.getByText("Files to Create").closest("button");
      fireEvent.click(filesButton!);
      // After clicking, files should be visible
      expect(screen.getByText(/.domains\/voice-clone\/domain.yaml/)).toBeDefined();
    });

    it("does not render files section for existing domain", () => {
      const existingConfig: LabConfig = {
        ...defaultConfig,
        createNewDomain: false,
        existingDomainSlug: "robotics",
        domain: undefined,
      };
      render(<WizardStepReview {...defaultProps} config={existingConfig} />);
      expect(screen.queryByText("Files to Create")).toBeNull();
    });
  });

  describe("estimated timeline", () => {
    it("shows estimated timeline from analysis", () => {
      render(<WizardStepReview {...defaultProps} analysis={mockAnalysis} />);
      expect(screen.getByText(/Estimated research timeline/)).toBeDefined();
      expect(screen.getByText(/About 2 weeks/)).toBeDefined();
    });

    it("does not show timeline when no analysis", () => {
      render(<WizardStepReview {...defaultProps} analysis={undefined} />);
      expect(screen.queryByText(/Estimated research timeline/)).toBeNull();
    });
  });

  describe("validation errors", () => {
    it("shows validation errors when provided", () => {
      render(
        <WizardStepReview
          {...defaultProps}
          validationErrors={["Domain name is required", "SSH host is required"]}
        />
      );
      expect(screen.getByText("Domain name is required")).toBeDefined();
      expect(screen.getByText("SSH host is required")).toBeDefined();
    });

    it("shows error header when validation errors exist", () => {
      render(
        <WizardStepReview
          {...defaultProps}
          validationErrors={["Error 1"]}
        />
      );
      expect(screen.getByText(/Please fix the following issues/)).toBeDefined();
    });

    it("disables Launch button when validation errors exist", () => {
      render(
        <WizardStepReview
          {...defaultProps}
          validationErrors={["Error 1"]}
        />
      );
      const button = screen.getByText("Launch Lab").closest("button");
      expect(button?.disabled).toBe(true);
    });

    it("enables Launch button when no validation errors", () => {
      render(<WizardStepReview {...defaultProps} validationErrors={[]} />);
      const button = screen.getByText("Launch Lab").closest("button");
      expect(button?.disabled).toBe(false);
    });
  });

  describe("creation error", () => {
    it("shows creation error when provided", () => {
      render(
        <WizardStepReview
          {...defaultProps}
          createError="Failed to create lab"
        />
      );
      expect(screen.getByText("Failed to create lab")).toBeDefined();
    });
  });

  describe("lab creation", () => {
    it("calls onCreateLab when Launch button is clicked", () => {
      const onCreateLab = vi.fn().mockResolvedValue(undefined);
      render(<WizardStepReview {...defaultProps} onCreateLab={onCreateLab} />);

      const button = screen.getByText("Launch Lab");
      fireEvent.click(button);

      expect(onCreateLab).toHaveBeenCalledTimes(1);
    });

    it("shows loading state when creating", () => {
      render(<WizardStepReview {...defaultProps} isCreating={true} />);
      expect(screen.getByText("Creating Lab...")).toBeDefined();
    });

    it("disables button when creating", () => {
      render(<WizardStepReview {...defaultProps} isCreating={true} />);
      const button = screen.getByText("Creating Lab...").closest("button");
      expect(button?.disabled).toBe(true);
    });
  });

  describe("styling", () => {
    it("applies custom className", () => {
      const { container } = render(
        <WizardStepReview {...defaultProps} className="custom-class" />
      );
      expect(container.firstChild?.className).toContain("custom-class");
    });
  });
});
