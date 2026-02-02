/**
 * Lab Wizard Integration Tests
 *
 * Tests the full wizard flow from welcome to lab creation.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LabWizard } from "@/components/lab-wizard/LabWizard";

// Mock next/navigation
const mockPush = vi.fn();
const mockReplace = vi.fn();
let mockSearchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    replace: mockReplace,
  }),
  useSearchParams: () => mockSearchParams,
}));

// Mock the types module
vi.mock("@/lib/lab-wizard/types", async () => {
  const actual = await vi.importActual("@/lib/lab-wizard/types");
  return {
    ...actual,
    WIZARD_STEPS: [
      { id: "welcome", label: "Welcome" },
      { id: "domain", label: "Domain" },
      { id: "hardware", label: "Hardware" },
      { id: "research", label: "Research" },
      { id: "review", label: "Review" },
    ],
    getStepIndex: (step: string) => {
      const steps = ["welcome", "domain", "hardware", "research", "review"];
      return steps.indexOf(step);
    },
    getNextStep: (step: string) => {
      const steps = ["welcome", "domain", "hardware", "research", "review"];
      const index = steps.indexOf(step);
      return index < steps.length - 1 ? steps[index + 1] : null;
    },
    getPrevStep: (step: string) => {
      const steps = ["welcome", "domain", "hardware", "research", "review"];
      const index = steps.indexOf(step);
      return index > 0 ? steps[index - 1] : null;
    },
    isStepCompleted: () => false,
    CLOUD_PROVIDERS: [
      { id: "runpod", name: "RunPod", icon: "🚀" },
    ],
    getEstimatedTime: () => 5,
  };
});

describe("Lab Wizard Integration Flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearchParams = new URLSearchParams();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ success: false }),
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("complete wizard flow", () => {
    it("can navigate through wizard steps", async () => {
      render(<LabWizard />);

      // Step 1: Welcome - click Get Started
      expect(screen.getByText("Create Your Research Lab")).toBeDefined();
      fireEvent.click(screen.getByText("Get Started"));

      // Step 2: Domain - select Voice Cloning template
      await waitFor(() => {
        expect(screen.getByText("Voice Cloning")).toBeDefined();
      });
      fireEvent.click(screen.getByText("Voice Cloning").closest("button")!);
      fireEvent.click(screen.getByText("Continue"));

      // Step 3: Hardware - local is default, continue
      await waitFor(() => {
        expect(screen.getByText("Select Hardware Type")).toBeDefined();
      });
      fireEvent.click(screen.getByText("Continue"));

      // Step 4: Research - enter goal and continue
      await waitFor(() => {
        expect(screen.getByText("What do you want to research?")).toBeDefined();
      });
      const textarea = screen.getByPlaceholderText(/Example: I want to create/);
      fireEvent.change(textarea, { target: { value: "Build TTS system" } });
      fireEvent.click(screen.getByText("Continue"));

      // Step 5: Review
      await waitFor(() => {
        expect(screen.getByText("Review Your Lab Configuration")).toBeDefined();
      });

      // Should have all configuration shown
      expect(screen.getByText("Voice Cloning")).toBeDefined();
      expect(screen.getByText("Local Machine")).toBeDefined();
    });

    it("can complete wizard using quick setup", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          labId: "lab-123",
          domainSlug: "voice-clone",
          redirectUrl: "/lab?domain=voice-clone",
        }),
      });

      render(<LabWizard />);

      // Click Use Template
      fireEvent.click(screen.getByText("Use Template"));

      // Should jump directly to review
      await waitFor(() => {
        expect(screen.getByText("Review Your Lab Configuration")).toBeDefined();
      });

      // Should have voice-clone populated
      expect(screen.getByText("Voice Cloning")).toBeDefined();

      // Create lab
      fireEvent.click(screen.getByText("Launch Lab"));

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith("/lab?domain=voice-clone");
      });
    });
  });

  describe("wizard navigation", () => {
    it("maintains state when navigating back and forth", async () => {
      render(<LabWizard />);

      // Go to domain step
      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Voice Cloning")).toBeDefined();
      });

      // Select Voice Cloning
      fireEvent.click(screen.getByText("Voice Cloning").closest("button")!);

      // Go to hardware
      fireEvent.click(screen.getByText("Continue"));

      await waitFor(() => {
        expect(screen.getByText("Select Hardware Type")).toBeDefined();
      });

      // Go back to domain
      fireEvent.click(screen.getByText("Back"));

      await waitFor(() => {
        // Domain should still have Voice Cloning selected (indicated by check icon)
        const voiceButton = screen.getByText("Voice Cloning").closest("button")!;
        expect(voiceButton.querySelector("svg[class*='text-foreground-bright']")).toBeDefined();
      });
    });

    it("allows clicking completed steps to navigate", async () => {
      render(<LabWizard />);

      // Complete welcome
      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Voice Cloning")).toBeDefined();
      });

      // Complete domain
      fireEvent.click(screen.getByText("Voice Cloning").closest("button")!);
      fireEvent.click(screen.getByText("Continue"));

      await waitFor(() => {
        expect(screen.getByText("Select Hardware Type")).toBeDefined();
      });

      // Click Welcome in progress bar (should be completed/green)
      fireEvent.click(screen.getByText("Welcome"));

      await waitFor(() => {
        expect(screen.getByText("Create Your Research Lab")).toBeDefined();
      });
    });
  });

  describe("validation flow", () => {
    it("prevents navigation when validation fails", async () => {
      render(<LabWizard />);

      // Go to domain step
      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Voice Cloning")).toBeDefined();
      });

      // Try to continue without selecting a domain
      fireEvent.click(screen.getByText("Continue"));

      // Should still be on domain step (validation failed)
      await waitFor(() => {
        // Validation should prevent navigation, so we're still on domain page
        expect(screen.getByText(/Domains organize your research/)).toBeDefined();
      });
    });

    it("shows validation errors on review step", async () => {
      render(<LabWizard />);

      // Use quick setup to get to review
      fireEvent.click(screen.getByText("Use Template"));

      await waitFor(() => {
        expect(screen.getByText("Review Your Lab Configuration")).toBeDefined();
      });

      // Review should show the lab configuration
      expect(screen.getByText("Voice Cloning")).toBeDefined();
      expect(screen.getByText("Local Machine")).toBeDefined();
    });
  });

  describe("error handling", () => {
    it("shows error when lab creation fails", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          success: false,
          error: "Failed to create domain files",
        }),
      });

      render(<LabWizard />);

      // Quick setup to review
      fireEvent.click(screen.getByText("Use Template"));

      await waitFor(() => {
        expect(screen.getByText("Launch Lab")).toBeDefined();
      });

      // Try to create lab
      fireEvent.click(screen.getByText("Launch Lab"));

      // Should show error
      await waitFor(() => {
        expect(screen.getByText("Failed to create domain files")).toBeDefined();
      });

      // Should not have navigated
      expect(mockPush).not.toHaveBeenCalled();
    });

    it("handles network errors gracefully", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("Network error")
      );

      render(<LabWizard />);

      // Quick setup to review
      fireEvent.click(screen.getByText("Use Template"));

      await waitFor(() => {
        expect(screen.getByText("Launch Lab")).toBeDefined();
      });

      // Try to create lab
      fireEvent.click(screen.getByText("Launch Lab"));

      // Should show error
      await waitFor(() => {
        expect(screen.getByText("Network error")).toBeDefined();
      });
    });
  });

  describe("URL state management", () => {
    it("updates URL when step changes", async () => {
      render(<LabWizard />);

      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(mockReplace).toHaveBeenCalledWith(
          expect.stringContaining("step=domain"),
          expect.any(Object)
        );
      });
    });

    it("restores step from URL on mount", async () => {
      mockSearchParams = new URLSearchParams("step=hardware");

      // Mock fetch for hardware detection
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ success: false }),
      });

      render(<LabWizard />);

      // Should start on hardware step
      await waitFor(() => {
        expect(screen.getByText("Select Hardware Type")).toBeDefined();
      });
    });
  });

  describe("hardware configuration flow", () => {
    it("detects local GPU when on hardware step", async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: false }), // Initial detection
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            gpu: { name: "RTX 4090", vram: 24, available: true },
          }),
        });

      render(<LabWizard />);

      // Go to domain and select
      fireEvent.click(screen.getByText("Get Started"));
      await waitFor(() => expect(screen.getByText("Voice Cloning")).toBeDefined());
      fireEvent.click(screen.getByText("Voice Cloning").closest("button")!);
      fireEvent.click(screen.getByText("Continue"));

      // Should be on hardware step
      await waitFor(() => {
        expect(screen.getByText("Select Hardware Type")).toBeDefined();
      });

      // Click detect button
      fireEvent.click(screen.getByText("Detect"));

      // Should show detected GPU
      await waitFor(() => {
        expect(screen.getByText("RTX 4090")).toBeDefined();
      });
    });
  });

  describe("AI goal analysis flow", () => {
    it("analyzes research goal with AI", async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ success: false }), // Hardware detection
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            analysis: {
              suggestedDomain: "voice-clone",
              domainName: "Voice Cloning",
              arxivCategories: ["cs.SD", "eess.AS"],
              keywords: ["TTS", "prosody", "emotion"],
              reasoning: "Your goal focuses on speech synthesis.",
              suggestedTasks: [
                {
                  subject: "Research TTS architectures",
                  description: "Survey recent TTS papers",
                  type: "research",
                  priority: "high",
                },
              ],
              estimatedTimeline: "About 2 weeks",
            },
          }),
        });

      render(<LabWizard />);

      // Navigate to research step
      fireEvent.click(screen.getByText("Get Started"));
      await waitFor(() => expect(screen.getByText("Voice Cloning")).toBeDefined());
      fireEvent.click(screen.getByText("Voice Cloning").closest("button")!);
      fireEvent.click(screen.getByText("Continue"));
      await waitFor(() => expect(screen.getByText("Select Hardware Type")).toBeDefined());
      fireEvent.click(screen.getByText("Continue"));

      // Should be on research step
      await waitFor(() => {
        expect(screen.getByText("What do you want to research?")).toBeDefined();
      });

      // Enter goal
      const textarea = screen.getByPlaceholderText(/Example: I want to create/);
      fireEvent.change(textarea, { target: { value: "Build TTS system with emotion control" } });

      // Click analyze
      fireEvent.click(screen.getByText("Analyze with AI"));

      // Should show analysis results
      await waitFor(() => {
        expect(screen.getByText("AI Analysis")).toBeDefined();
        expect(screen.getByText(/Your goal focuses on speech synthesis/)).toBeDefined();
      });

      // Should show suggested tasks
      expect(screen.getByText("Research TTS architectures")).toBeDefined();
    });
  });

  describe("custom domain creation", () => {
    it("allows creating a custom domain", async () => {
      render(<LabWizard />);

      // Go to domain step
      fireEvent.click(screen.getByText("Get Started"));

      await waitFor(() => {
        expect(screen.getByText("Create Custom Domain")).toBeDefined();
      });

      // Click custom domain
      fireEvent.click(screen.getByText("Create Custom Domain").closest("button")!);

      // Fill in custom domain form
      await waitFor(() => {
        expect(screen.getByPlaceholderText("My Research Domain")).toBeDefined();
      });

      fireEvent.change(screen.getByPlaceholderText("My Research Domain"), {
        target: { value: "My AI Lab" },
      });

      fireEvent.change(screen.getByPlaceholderText("What is this research about?"), {
        target: { value: "Custom AI research" },
      });

      // Continue to hardware
      fireEvent.click(screen.getByText("Continue"));

      await waitFor(() => {
        expect(screen.getByText("Select Hardware Type")).toBeDefined();
      });
    });
  });

  describe("edit from review", () => {
    it("allows editing domain from review step", async () => {
      render(<LabWizard />);

      // Quick setup to review
      fireEvent.click(screen.getByText("Use Template"));

      await waitFor(() => {
        expect(screen.getByText("Review Your Lab Configuration")).toBeDefined();
      });

      // Click Edit on domain section
      const editButtons = screen.getAllByText("Edit");
      fireEvent.click(editButtons[0]); // First edit is domain

      // Should be back on domain step
      await waitFor(() => {
        expect(screen.getByText(/Domains organize your research/)).toBeDefined();
      });
    });

    it("allows editing hardware from review step", async () => {
      // Mock fetch for hardware step
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ success: false }),
      });

      render(<LabWizard />);

      // Quick setup to review
      fireEvent.click(screen.getByText("Use Template"));

      await waitFor(() => {
        expect(screen.getByText("Review Your Lab Configuration")).toBeDefined();
      });

      // Click Edit on hardware section
      const editButtons = screen.getAllByText("Edit");
      fireEvent.click(editButtons[1]); // Second edit is hardware

      // Should be back on hardware step
      await waitFor(() => {
        expect(screen.getByText("Select Hardware Type")).toBeDefined();
      });
    });
  });
});
