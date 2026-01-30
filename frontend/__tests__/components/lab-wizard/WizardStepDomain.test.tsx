/**
 * WizardStepDomain Component Tests
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { WizardStepDomain } from "@/components/lab-wizard/WizardStepDomain";
import type { DomainConfig } from "@/lib/domain/types";

describe("WizardStepDomain", () => {
  const defaultProps = {
    createNew: false,
    onSelectDomain: vi.fn(),
    onToggleCreateNew: vi.fn(),
  };

  const mockExistingDomains: DomainConfig[] = [
    {
      name: "Voice Clone",
      slug: "voice-clone",
      description: "Speech synthesis research",
      difficulty: "advanced",
      branding: {
        primaryColor: "#3b82f6",
        accentColor: "#22c55e",
        backgroundStyle: "sky",
      },
      scene: { props: [], decorations: { plants: true, floatingCubes: true, particles: true } },
      research: { arxivCategories: ["cs.SD"], keywords: ["TTS"] },
      version: "1.0",
    },
    {
      name: "Robotics",
      slug: "robotics",
      description: "Robot learning research",
      difficulty: "intermediate",
      branding: {
        primaryColor: "#a855f7",
        accentColor: "#22c55e",
        backgroundStyle: "night",
      },
      scene: { props: [], decorations: { plants: true, floatingCubes: true, particles: true } },
      research: { arxivCategories: ["cs.RO"], keywords: ["control"] },
      version: "1.0",
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders info banner", () => {
      render(<WizardStepDomain {...defaultProps} />);
      expect(screen.getByText(/Domains organize your research/)).toBeDefined();
    });

    it("renders domain templates", () => {
      render(<WizardStepDomain {...defaultProps} />);
      expect(screen.getByText("Voice Cloning")).toBeDefined();
      expect(screen.getByText("Quantitative Trading")).toBeDefined();
      expect(screen.getByText("Robotics")).toBeDefined();
      expect(screen.getByText("Biotech & Drug Discovery")).toBeDefined();
    });

    it("renders template descriptions", () => {
      render(<WizardStepDomain {...defaultProps} />);
      expect(screen.getByText(/Speech synthesis, prosody control/)).toBeDefined();
      expect(screen.getByText(/Algorithmic trading/)).toBeDefined();
    });

    it("renders template tags", () => {
      render(<WizardStepDomain {...defaultProps} />);
      expect(screen.getByText("TTS")).toBeDefined();
      expect(screen.getByText("prosody")).toBeDefined();
    });

    it("renders custom domain option", () => {
      render(<WizardStepDomain {...defaultProps} />);
      expect(screen.getByText("Create Custom Domain")).toBeDefined();
    });
  });

  describe("existing domains", () => {
    it("renders existing domains section when provided", () => {
      render(
        <WizardStepDomain
          {...defaultProps}
          existingDomains={mockExistingDomains}
        />
      );
      expect(screen.getByText("Your Existing Domains")).toBeDefined();
    });

    it("renders each existing domain", () => {
      render(
        <WizardStepDomain
          {...defaultProps}
          existingDomains={mockExistingDomains}
        />
      );
      // Voice Clone appears in both existing domains and templates
      const voiceCloneElements = screen.getAllByText("Voice Clone");
      expect(voiceCloneElements.length).toBeGreaterThanOrEqual(1);
    });

    it("shows different header when existing domains present", () => {
      render(
        <WizardStepDomain
          {...defaultProps}
          existingDomains={mockExistingDomains}
        />
      );
      expect(screen.getByText("Or Start with a Template")).toBeDefined();
    });

    it("shows default header when no existing domains", () => {
      render(<WizardStepDomain {...defaultProps} />);
      expect(screen.getByText("Choose a Domain Template")).toBeDefined();
    });
  });

  describe("template selection", () => {
    it("calls onSelectDomain when template is clicked", () => {
      const onSelectDomain = vi.fn();
      render(
        <WizardStepDomain {...defaultProps} onSelectDomain={onSelectDomain} />
      );

      const voiceTemplate = screen.getByText("Voice Cloning").closest("button");
      fireEvent.click(voiceTemplate!);

      expect(onSelectDomain).toHaveBeenCalledWith("voice-clone");
    });

    it("calls onToggleCreateNew(true) when template is clicked", () => {
      const onToggleCreateNew = vi.fn();
      render(
        <WizardStepDomain {...defaultProps} onToggleCreateNew={onToggleCreateNew} />
      );

      const template = screen.getByText("Voice Cloning").closest("button");
      fireEvent.click(template!);

      expect(onToggleCreateNew).toHaveBeenCalledWith(true);
    });

    it("calls onNewDomainChange with template config", () => {
      const onNewDomainChange = vi.fn();
      render(
        <WizardStepDomain
          {...defaultProps}
          onNewDomainChange={onNewDomainChange}
        />
      );

      const template = screen.getByText("Voice Cloning").closest("button");
      fireEvent.click(template!);

      expect(onNewDomainChange).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Voice Cloning",
          slug: "voice-clone",
        })
      );
    });

    it("highlights selected template", () => {
      const { container } = render(
        <WizardStepDomain
          {...defaultProps}
          selectedDomain="voice-clone"
          createNew={true}
        />
      );

      // Should have a check mark for selected template
      const checkmarks = container.querySelectorAll('[class*="text-foreground-bright"]');
      expect(checkmarks.length).toBeGreaterThan(0);
    });
  });

  describe("existing domain selection", () => {
    it("calls onSelectDomain when existing domain is clicked", () => {
      const onSelectDomain = vi.fn();
      render(
        <WizardStepDomain
          {...defaultProps}
          existingDomains={mockExistingDomains}
          onSelectDomain={onSelectDomain}
        />
      );

      // Find the existing domain button (not the template)
      const buttons = screen.getAllByRole("button");
      const existingDomainButton = buttons.find((btn) =>
        btn.textContent?.includes("Speech synthesis research")
      );
      fireEvent.click(existingDomainButton!);

      expect(onSelectDomain).toHaveBeenCalledWith("voice-clone");
    });

    it("calls onToggleCreateNew(false) when existing domain is clicked", () => {
      const onToggleCreateNew = vi.fn();
      render(
        <WizardStepDomain
          {...defaultProps}
          existingDomains={mockExistingDomains}
          onToggleCreateNew={onToggleCreateNew}
        />
      );

      const buttons = screen.getAllByRole("button");
      const existingDomainButton = buttons.find((btn) =>
        btn.textContent?.includes("Speech synthesis research")
      );
      fireEvent.click(existingDomainButton!);

      expect(onToggleCreateNew).toHaveBeenCalledWith(false);
    });
  });

  describe("custom domain form", () => {
    it("shows custom form when Create Custom Domain is clicked", () => {
      render(<WizardStepDomain {...defaultProps} />);

      const customButton = screen.getByText("Create Custom Domain").closest("button");
      fireEvent.click(customButton!);

      expect(screen.getByPlaceholderText("My Research Domain")).toBeDefined();
    });

    it("shows description input in custom form", () => {
      render(<WizardStepDomain {...defaultProps} />);

      const customButton = screen.getByText("Create Custom Domain").closest("button");
      fireEvent.click(customButton!);

      expect(screen.getByPlaceholderText("What is this research about?")).toBeDefined();
    });

    it("shows arxiv categories input in custom form", () => {
      render(<WizardStepDomain {...defaultProps} />);

      const customButton = screen.getByText("Create Custom Domain").closest("button");
      fireEvent.click(customButton!);

      expect(screen.getByPlaceholderText("cs.LG, cs.AI")).toBeDefined();
    });

    it("calls onNewDomainChange when name input changes", () => {
      const onNewDomainChange = vi.fn();
      render(
        <WizardStepDomain
          {...defaultProps}
          onNewDomainChange={onNewDomainChange}
        />
      );

      const customButton = screen.getByText("Create Custom Domain").closest("button");
      fireEvent.click(customButton!);

      const nameInput = screen.getByPlaceholderText("My Research Domain");
      fireEvent.change(nameInput, { target: { value: "Test Domain" } });

      expect(onNewDomainChange).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Test Domain",
          slug: "test-domain",
        })
      );
    });

    it("generates slug from name automatically", () => {
      const onNewDomainChange = vi.fn();
      render(
        <WizardStepDomain
          {...defaultProps}
          onNewDomainChange={onNewDomainChange}
        />
      );

      const customButton = screen.getByText("Create Custom Domain").closest("button");
      fireEvent.click(customButton!);

      const nameInput = screen.getByPlaceholderText("My Research Domain");
      fireEvent.change(nameInput, { target: { value: "My Cool Research" } });

      expect(onNewDomainChange).toHaveBeenCalledWith(
        expect.objectContaining({
          slug: "my-cool-research",
        })
      );
    });
  });

  describe("styling", () => {
    it("applies custom className", () => {
      const { container } = render(
        <WizardStepDomain {...defaultProps} className="custom-class" />
      );
      expect(container.firstChild?.className).toContain("custom-class");
    });
  });
});
