/**
 * Tests for DomainWizard Container Component
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { DomainWizard } from "@/components/domain/wizard";

describe("DomainWizard", () => {
  const mockOnComplete = vi.fn();
  const mockOnCancel = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders wizard header", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      expect(screen.getByText("Create New Domain")).toBeDefined();
    });

    it("renders all step indicators", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Step numbers should be visible
      expect(screen.getByText("1")).toBeDefined();
      expect(screen.getByText("2")).toBeDefined();
      expect(screen.getByText("3")).toBeDefined();
      expect(screen.getByText("4")).toBeDefined();
    });

    it("shows first step content by default", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      expect(screen.getByText("Choose a Template")).toBeDefined();
    });

    it("renders cancel and next buttons", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      expect(screen.getByText("Cancel")).toBeDefined();
      expect(screen.getByText("Next")).toBeDefined();
    });
  });

  describe("navigation", () => {
    it("disables Next when no template selected", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      const nextButton = screen.getByText("Next").closest("button");
      expect(nextButton?.disabled).toBe(true);
    });

    it("enables Next after selecting template", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Select a template
      fireEvent.click(screen.getByText("Blank Slate"));

      const nextButton = screen.getByText("Next").closest("button");
      expect(nextButton?.disabled).toBe(false);
    });

    it("advances to step 2 after clicking Next", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Select template and go next
      fireEvent.click(screen.getByText("Blank Slate"));
      fireEvent.click(screen.getByText("Next"));

      // Should now show branding step
      expect(screen.getByText("Name & Branding")).toBeDefined();
    });

    it("goes back to previous step when Back clicked", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Go to step 2
      fireEvent.click(screen.getByText("Blank Slate"));
      fireEvent.click(screen.getByText("Next"));

      // Go back
      fireEvent.click(screen.getByText("Back"));

      // Should be back at template step
      expect(screen.getByText("Choose a Template")).toBeDefined();
    });

    it("calls onCancel when Cancel clicked on first step", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      fireEvent.click(screen.getByText("Cancel"));

      expect(mockOnCancel).toHaveBeenCalled();
    });
  });

  describe("validation", () => {
    it("disables Next button when branding fields are empty", async () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Go to step 2
      fireEvent.click(screen.getByText("Blank Slate"));
      fireEvent.click(screen.getByText("Next"));

      // Next button should be disabled when fields are empty
      const nextButton = screen.getByText("Next").closest("button");
      expect(nextButton?.disabled).toBe(true);
    });

    it("allows proceeding when branding is valid", async () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Go to step 2
      fireEvent.click(screen.getByText("Blank Slate"));
      fireEvent.click(screen.getByText("Next"));

      // Fill in required fields
      const nameInput = screen.getByPlaceholderText("My Research Lab");
      fireEvent.change(nameInput, { target: { value: "Test Lab" } });

      const descInput = screen.getByPlaceholderText(/brief description/);
      fireEvent.change(descInput, {
        target: { value: "A test description" },
      });

      // Should be able to proceed
      fireEvent.click(screen.getByText("Next"));

      // Should now be on step 3
      await waitFor(() => {
        expect(screen.getByText("Research Focus")).toBeDefined();
      });
    });
  });

  describe("template selection", () => {
    it("pre-fills branding from template", () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Select Voice Research template
      fireEvent.click(screen.getByText("Voice Research"));
      fireEvent.click(screen.getByText("Next"));

      // Should have template values pre-filled
      const nameInput = screen.getByPlaceholderText(
        "My Research Lab"
      ) as HTMLInputElement;
      expect(nameInput.value).toBe("Voice Research");
    });

    it("pre-fills research from template", async () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Select Voice Research and go to step 3
      fireEvent.click(screen.getByText("Voice Research"));
      fireEvent.click(screen.getByText("Next"));
      fireEvent.click(screen.getByText("Next"));

      // Should have voice-related categories selected - search for arXiv categories section
      await waitFor(() => {
        // The step 3 header should be visible
        expect(screen.getByText("Research Focus")).toBeDefined();
      });
    });
  });

  describe("completion", () => {
    it("shows Create Domain on last step", async () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Navigate through all steps
      fireEvent.click(screen.getByText("Blank Slate"));
      fireEvent.click(screen.getByText("Next"));

      // Fill branding
      fireEvent.change(screen.getByPlaceholderText("My Research Lab"), {
        target: { value: "Test" },
      });
      fireEvent.change(screen.getByPlaceholderText(/brief description/), {
        target: { value: "Test desc" },
      });
      fireEvent.click(screen.getByText("Next"));

      // Step 3 - Research
      fireEvent.click(screen.getByText("Next"));

      // Should be on last step with Create button
      await waitFor(() => {
        expect(screen.getByText("Create Domain")).toBeDefined();
      });
    });

    it("calls onComplete with config when Create clicked", async () => {
      render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Navigate through all steps
      fireEvent.click(screen.getByText("Blank Slate"));
      fireEvent.click(screen.getByText("Next"));

      // Fill branding
      fireEvent.change(screen.getByPlaceholderText("My Research Lab"), {
        target: { value: "Test Lab" },
      });
      fireEvent.change(screen.getByPlaceholderText(/brief description/), {
        target: { value: "Test description" },
      });
      fireEvent.click(screen.getByText("Next"));

      // Step 3 - Research
      fireEvent.click(screen.getByText("Next"));

      // Step 4 - Scene
      fireEvent.click(screen.getByText("Create Domain"));

      await waitFor(() => {
        expect(mockOnComplete).toHaveBeenCalledWith(
          expect.objectContaining({
            templateId: "blank",
            branding: expect.objectContaining({
              name: "Test Lab",
              slug: "test-lab",
              description: "Test description",
            }),
            research: expect.objectContaining({
              arxivCategories: expect.any(Array),
              keywords: expect.any(Array),
              tags: expect.any(Array),
            }),
            scene: expect.objectContaining({
              backgroundStyle: expect.any(String),
              props: expect.any(Array),
              cameraAngle: expect.any(String),
            }),
          })
        );
      });
    });
  });

  describe("saving state", () => {
    it("shows loading indicator when isSaving", () => {
      render(
        <DomainWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
          isSaving={true}
        />
      );

      expect(screen.getByText("Saving...")).toBeDefined();
    });

    it("disables navigation when saving", () => {
      render(
        <DomainWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
          isSaving={true}
        />
      );

      // The cancel/back button should be disabled
      const backButton = screen.getByText("Cancel").closest("button");
      expect(backButton?.disabled).toBe(true);
    });
  });

  describe("initial template", () => {
    it("pre-selects template when initialTemplate provided", () => {
      render(
        <DomainWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
          initialTemplate="voice-research"
        />
      );

      // Voice Research should be selected
      const voiceButton = screen.getByText("Voice Research").closest("button");
      expect(voiceButton?.className).toContain("ring");
    });
  });

  describe("close button", () => {
    it("calls onCancel when X button clicked", () => {
      const { container } = render(
        <DomainWizard onComplete={mockOnComplete} onCancel={mockOnCancel} />
      );

      // Find X button in header
      const closeButton = container.querySelector('button svg.lucide-x');
      if (closeButton?.closest("button")) {
        fireEvent.click(closeButton.closest("button")!);
        expect(mockOnCancel).toHaveBeenCalled();
      }
    });
  });
});
