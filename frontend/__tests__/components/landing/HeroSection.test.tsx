/**
 * HeroSection Component Tests
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { HeroSection } from "@/components/landing/HeroSection";

describe("HeroSection", () => {
  describe("Rendering", () => {
    it("should render the main headline", () => {
      render(<HeroSection />);

      expect(screen.getByText("LabFork")).toBeInTheDocument();
      expect(screen.getByText("Fork. Watch. Discover.")).toBeInTheDocument();
    });

    it("should render the subheadline", () => {
      render(<HeroSection />);

      expect(
        screen.getByText(/Fork research labs\. Watch AI agents implement papers/i)
      ).toBeInTheDocument();
    });

    it("should render CTA buttons", () => {
      render(<HeroSection />);

      expect(screen.getByText("Create Your Lab")).toBeInTheDocument();
      expect(screen.getByText("Explore Public Labs")).toBeInTheDocument();
    });

    it("should render the canvas for particle animation", () => {
      const { container } = render(<HeroSection />);

      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
    });
  });

  describe("Stats Display", () => {
    it("should show default lab count", () => {
      render(<HeroSection />);

      expect(screen.getByText("1,247")).toBeInTheDocument();
      expect(screen.getByText("Active Labs")).toBeInTheDocument();
    });

    it("should show custom lab count when provided", () => {
      render(<HeroSection labCount={5000} />);

      expect(screen.getByText("5,000")).toBeInTheDocument();
    });

    it("should show discovery count", () => {
      render(<HeroSection discoveryCount={150} />);

      expect(screen.getByText("150")).toBeInTheDocument();
      expect(screen.getByText("Synergies Found")).toBeInTheDocument();
    });

    it("should show domain count", () => {
      render(<HeroSection />);

      expect(screen.getByText("9")).toBeInTheDocument();
      expect(screen.getByText("Domains")).toBeInTheDocument();
    });
  });

  describe("Links", () => {
    it("should have correct href for Create Your Lab", () => {
      render(<HeroSection />);

      const createLabLink = screen.getByText("Create Your Lab").closest("a");
      expect(createLabLink).toHaveAttribute("href", "/lab/new");
    });

    it("should have correct href for Explore Public Labs", () => {
      render(<HeroSection />);

      const exploreLink = screen.getByText("Explore Public Labs").closest("a");
      expect(exploreLink).toHaveAttribute("href", "/explore");
    });
  });

  describe("Activity Badge", () => {
    it("should show labs actively researching message", () => {
      render(<HeroSection labCount={1000} />);

      expect(screen.getByText(/1,000 labs actively researching/i)).toBeInTheDocument();
    });
  });
});
