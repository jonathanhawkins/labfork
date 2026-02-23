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

      expect(screen.getByText("Fork a Lab")).toBeInTheDocument();
      expect(screen.getByText("Explore Public Labs")).toBeInTheDocument();
    });

    it("should render the canvas for particle animation", () => {
      const { container } = render(<HeroSection />);

      const canvas = container.querySelector("canvas");
      expect(canvas).toBeInTheDocument();
    });
  });

  describe("Platform Badge", () => {
    it("should show open research platform text", () => {
      render(<HeroSection />);

      expect(screen.getByText("Open Research Platform")).toBeInTheDocument();
    });
  });

  describe("Links", () => {
    it("should have correct href for Fork a Lab", () => {
      render(<HeroSection />);

      const createLabLink = screen.getByText("Fork a Lab").closest("a");
      expect(createLabLink).toHaveAttribute("href", "/lab/new");
    });

    it("should have correct href for Explore Public Labs", () => {
      render(<HeroSection />);

      const exploreLink = screen.getByText("Explore Public Labs").closest("a");
      expect(exploreLink).toHaveAttribute("href", "/explore");
    });
  });

});
