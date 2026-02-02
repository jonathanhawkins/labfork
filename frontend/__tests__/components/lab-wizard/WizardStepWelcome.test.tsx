/**
 * WizardStepWelcome Component Tests
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { WizardStepWelcome } from "@/components/lab-wizard/WizardStepWelcome";

// Mock the types module
vi.mock("@/lib/lab-wizard/types", () => ({
  getEstimatedTime: () => 5,
}));

describe("WizardStepWelcome", () => {
  const defaultProps = {
    onGetStarted: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders the hero section", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.getByText("Create Your Research Lab")).toBeDefined();
    });

    it("renders the estimated time", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.getByText(/About 5 minutes/)).toBeDefined();
    });

    it("renders all feature cards", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.getByText("Configure Hardware")).toBeDefined();
      expect(screen.getByText("Choose Your Domain")).toBeDefined();
      expect(screen.getByText("AI-Assisted Setup")).toBeDefined();
      expect(screen.getByText("Launch Your Lab")).toBeDefined();
    });

    it("renders feature descriptions", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.getByText(/Local GPU, remote SSH, or cloud/)).toBeDefined();
      expect(screen.getByText(/Voice cloning, trading, robotics/)).toBeDefined();
    });

    it("renders what will be created section", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.getByText("Domain configuration file")).toBeDefined();
      expect(screen.getByText("Custom prompt templates")).toBeDefined();
      expect(screen.getByText("Initial research tasks")).toBeDefined();
      expect(screen.getByText("Hardware connection")).toBeDefined();
    });

    it("renders Get Started button", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.getByText("Get Started")).toBeDefined();
    });

    it("renders help text", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.getByText(/save your progress/)).toBeDefined();
    });
  });

  describe("interactions", () => {
    it("calls onGetStarted when Get Started button is clicked", () => {
      const onGetStarted = vi.fn();
      render(<WizardStepWelcome onGetStarted={onGetStarted} />);

      const button = screen.getByText("Get Started");
      fireEvent.click(button);

      expect(onGetStarted).toHaveBeenCalledTimes(1);
    });

    it("does not render Use Template button when onQuickSetup is not provided", () => {
      render(<WizardStepWelcome {...defaultProps} />);
      expect(screen.queryByText("Use Template")).toBeNull();
    });

    it("renders Use Template button when onQuickSetup is provided", () => {
      render(<WizardStepWelcome {...defaultProps} onQuickSetup={vi.fn()} />);
      expect(screen.getByText("Use Template")).toBeDefined();
    });

    it("calls onQuickSetup when Use Template button is clicked", () => {
      const onQuickSetup = vi.fn();
      render(<WizardStepWelcome {...defaultProps} onQuickSetup={onQuickSetup} />);

      const button = screen.getByText("Use Template");
      fireEvent.click(button);

      expect(onQuickSetup).toHaveBeenCalledTimes(1);
    });
  });

  describe("styling", () => {
    it("applies custom className", () => {
      const { container } = render(
        <WizardStepWelcome {...defaultProps} className="custom-class" />
      );
      expect(container.firstChild?.className).toContain("custom-class");
    });
  });
});
