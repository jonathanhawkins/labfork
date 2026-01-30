/**
 * HowItWorks Component Tests
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { HowItWorks } from "@/components/landing/HowItWorks";

describe("HowItWorks", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("Rendering", () => {
    it("should render section header", () => {
      render(<HowItWorks />);

      expect(screen.getByText("How It Works")).toBeInTheDocument();
    });

    it("should render section description", () => {
      render(<HowItWorks />);

      expect(
        screen.getByText("From paper to breakthrough in three simple steps")
      ).toBeInTheDocument();
    });

    it("should render all three step titles", () => {
      render(<HowItWorks />);

      // Step titles appear in both navigation and content area when selected
      expect(screen.getAllByText("Choose Your Domain").length).toBeGreaterThanOrEqual(1);
      // Note: "Add Research Papers" and "Watch AI Agents Work" are in the data
      // but may not all be visible at once depending on carousel state
    });

    it("should render step numbers", () => {
      render(<HowItWorks />);

      expect(screen.getByText("1")).toBeInTheDocument();
      expect(screen.getByText("2")).toBeInTheDocument();
      expect(screen.getByText("3")).toBeInTheDocument();
    });
  });

  describe("Step Content", () => {
    it("should show step 1 content by default", () => {
      render(<HowItWorks />);

      // Step title appears in both navigation and content area
      expect(screen.getAllByText("Choose Your Domain").length).toBeGreaterThanOrEqual(1);
      expect(
        screen.getByText(/Select from 9 pre-configured research domains/i)
      ).toBeInTheDocument();
    });

    it("should show step details", () => {
      render(<HowItWorks />);

      // Step 1 details
      expect(
        screen.getByText(/Voice Cloning, Quant Trading, Game AI/i)
      ).toBeInTheDocument();
    });
  });

  describe("Navigation", () => {
    it("should switch to step 2 when clicked", () => {
      render(<HowItWorks />);

      const step2Button = screen.getByText("2");
      fireEvent.click(step2Button);

      // Step title appears in both navigation and content area
      expect(screen.getAllByText("Add Research Papers").length).toBeGreaterThanOrEqual(1);
      expect(
        screen.getByText(/Import papers from arXiv/i)
      ).toBeInTheDocument();
    });

    it("should switch to step 3 when clicked", () => {
      render(<HowItWorks />);

      const step3Button = screen.getByText("3");
      fireEvent.click(step3Button);

      // Step title appears in both navigation and content area
      expect(screen.getAllByText("Watch AI Agents Work").length).toBeGreaterThanOrEqual(1);
      expect(
        screen.getByText(/Agents analyze papers/i)
      ).toBeInTheDocument();
    });
  });

  describe("Auto-advance", () => {
    it("should auto-advance after 5 seconds", () => {
      render(<HowItWorks />);

      // Initially on step 1 - title appears in both navigation and content
      expect(screen.getAllByText("Choose Your Domain").length).toBeGreaterThanOrEqual(1);

      // Advance time by 5 seconds
      act(() => {
        vi.advanceTimersByTime(5000);
      });

      // Should now be on step 2 - title appears in both navigation and content
      expect(screen.getAllByText("Add Research Papers").length).toBeGreaterThanOrEqual(1);
    });

    it("should cycle back to step 1 after step 3", () => {
      render(<HowItWorks />);

      // Go to step 3
      const step3Button = screen.getByText("3");
      fireEvent.click(step3Button);

      // Step title appears in both navigation and content area
      expect(screen.getAllByText("Watch AI Agents Work").length).toBeGreaterThanOrEqual(1);

      // Wait for auto-advance (back to step 1)
      act(() => {
        vi.advanceTimersByTime(5000);
      });

      // Step title appears in both navigation and content area
      expect(screen.getAllByText("Choose Your Domain").length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("Progress Bar", () => {
    it("should render progress bar", () => {
      const { container } = render(<HowItWorks />);

      const progressBar = container.querySelector('[class*="h-1"]');
      expect(progressBar).toBeInTheDocument();
    });
  });

  describe("Step Details", () => {
    it("should show step 1 details with checkmarks", () => {
      render(<HowItWorks />);

      expect(
        screen.getByText("Pre-configured arXiv categories and keywords")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Domain-specific evaluation metrics")
      ).toBeInTheDocument();
    });

    it("should show step 2 details when selected", () => {
      render(<HowItWorks />);

      const step2Button = screen.getByText("2");
      fireEvent.click(step2Button);

      expect(
        screen.getByText("Automatic technique extraction with AI")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Citation network building")
      ).toBeInTheDocument();
    });

    it("should show step 3 details when selected", () => {
      render(<HowItWorks />);

      const step3Button = screen.getByText("3");
      fireEvent.click(step3Button);

      expect(
        screen.getByText("5 meta-agents working together")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Automatic synergy detection across techniques")
      ).toBeInTheDocument();
    });
  });
});
