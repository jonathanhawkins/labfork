/**
 * LandingFooter Component Tests
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { LandingFooter } from "@/components/landing/LandingFooter";

describe("LandingFooter", () => {
  describe("Final CTA", () => {
    it("should render final CTA headline", () => {
      render(<LandingFooter />);

      expect(
        screen.getByText("Ready to Fork Your First Lab?")
      ).toBeInTheDocument();
    });

    it("should render final CTA description", () => {
      render(<LandingFooter />);

      expect(
        screen.getByText(/Join 1,000\+ researchers using LabFork/i)
      ).toBeInTheDocument();
    });

    it("should render Create Your Lab Free button", () => {
      render(<LandingFooter />);

      expect(screen.getByText("Create Your Lab Free")).toBeInTheDocument();
    });

    it("should render View Example Labs link", () => {
      render(<LandingFooter />);

      expect(screen.getByText(/View Example Labs/i)).toBeInTheDocument();
    });
  });

  describe("Domain Quick Links", () => {
    it("should render all 9 domain links", () => {
      render(<LandingFooter />);

      expect(screen.getByText("Voice Clone")).toBeInTheDocument();
      expect(screen.getByText("Quant Trading")).toBeInTheDocument();
      expect(screen.getByText("Game AI")).toBeInTheDocument();
      expect(screen.getByText("Robotics")).toBeInTheDocument();
      expect(screen.getByText("Drug Discovery")).toBeInTheDocument();
      expect(screen.getByText("Climate")).toBeInTheDocument();
      expect(screen.getByText("NLP")).toBeInTheDocument();
      expect(screen.getByText("Computer Vision")).toBeInTheDocument();
      expect(screen.getByText("Biotech NLP")).toBeInTheDocument();
    });

    it("should have correct href for domain links", () => {
      render(<LandingFooter />);

      const voiceCloneLink = screen.getByText("Voice Clone").closest("a");
      expect(voiceCloneLink).toHaveAttribute("href", "/lab?domain=voice-clone");

      const quantTradingLink = screen.getByText("Quant Trading").closest("a");
      expect(quantTradingLink).toHaveAttribute("href", "/lab?domain=quant-trading");
    });
  });

  describe("Footer Sections", () => {
    it("should render Product section", () => {
      render(<LandingFooter />);

      expect(screen.getByText("Product")).toBeInTheDocument();
      expect(screen.getByText("Features")).toBeInTheDocument();
      expect(screen.getByText("Domains")).toBeInTheDocument();
      expect(screen.getByText("Pricing")).toBeInTheDocument();
      expect(screen.getByText("Changelog")).toBeInTheDocument();
    });

    it("should render Resources section", () => {
      render(<LandingFooter />);

      expect(screen.getByText("Resources")).toBeInTheDocument();
      expect(screen.getByText("Documentation")).toBeInTheDocument();
      expect(screen.getByText("API Reference")).toBeInTheDocument();
      expect(screen.getByText("Tutorials")).toBeInTheDocument();
      expect(screen.getByText("Blog")).toBeInTheDocument();
    });

    it("should render Community section", () => {
      render(<LandingFooter />);

      expect(screen.getByText("Community")).toBeInTheDocument();
      expect(screen.getByText("Discord")).toBeInTheDocument();
      expect(screen.getByText("GitHub")).toBeInTheDocument();
      expect(screen.getByText("Twitter")).toBeInTheDocument();
      expect(screen.getByText("Newsletter")).toBeInTheDocument();
    });

    it("should render Company section", () => {
      render(<LandingFooter />);

      expect(screen.getByText("Company")).toBeInTheDocument();
      expect(screen.getByText("About")).toBeInTheDocument();
      expect(screen.getByText("Contact")).toBeInTheDocument();
      expect(screen.getByText("Privacy")).toBeInTheDocument();
      expect(screen.getByText("Terms")).toBeInTheDocument();
    });
  });

  describe("Brand Section", () => {
    it("should render brand name", () => {
      render(<LandingFooter />);

      expect(screen.getByText("LabFork")).toBeInTheDocument();
    });

    it("should render brand tagline", () => {
      render(<LandingFooter />);

      expect(
        screen.getByText("Fork research labs. Watch AI agents work. Collaborate globally.")
      ).toBeInTheDocument();
    });
  });

  describe("Social Links", () => {
    it("should have GitHub social link", () => {
      render(<LandingFooter />);

      const githubLinks = screen.getAllByRole("link").filter((link) =>
        link.getAttribute("href")?.includes("github.com")
      );
      expect(githubLinks.length).toBeGreaterThan(0);
    });

    it("should have Discord social link", () => {
      render(<LandingFooter />);

      const discordLinks = screen.getAllByRole("link").filter((link) =>
        link.getAttribute("href")?.includes("discord")
      );
      expect(discordLinks.length).toBeGreaterThan(0);
    });

    it("should have Twitter social link", () => {
      render(<LandingFooter />);

      const twitterLinks = screen.getAllByRole("link").filter((link) =>
        link.getAttribute("href")?.includes("twitter")
      );
      expect(twitterLinks.length).toBeGreaterThan(0);
    });
  });

  describe("Bottom Bar", () => {
    it("should render copyright", () => {
      render(<LandingFooter />);

      expect(screen.getByText(/2026 LabFork/i)).toBeInTheDocument();
    });

    it("should render tech stack mention", () => {
      render(<LandingFooter />);

      expect(screen.getByText(/Built with Next\.js, Three\.js, Claude/i)).toBeInTheDocument();
    });

    it("should render Star on GitHub link", () => {
      render(<LandingFooter />);

      expect(screen.getByText("Star on GitHub")).toBeInTheDocument();
    });
  });
});
