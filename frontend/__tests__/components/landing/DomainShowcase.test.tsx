/**
 * DomainShowcase Component Tests
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DomainShowcase, defaultDomains } from "@/components/landing/DomainShowcase";

describe("DomainShowcase", () => {
  describe("Rendering", () => {
    it("should render section header", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      expect(screen.getByText("Choose Your Research Domain")).toBeInTheDocument();
    });

    it("should render section description", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      expect(
        screen.getByText(/9 specialized labs with pre-configured tools/i)
      ).toBeInTheDocument();
    });

    it("should render all 9 default domains", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      expect(screen.getByText("Voice Clone Lab")).toBeInTheDocument();
      expect(screen.getByText("Quant Trading Lab")).toBeInTheDocument();
      expect(screen.getByText("Game AI Lab")).toBeInTheDocument();
      expect(screen.getByText("Robotics ML Lab")).toBeInTheDocument();
      expect(screen.getByText("Drug Discovery Lab")).toBeInTheDocument();
      expect(screen.getByText("Climate Modeling Lab")).toBeInTheDocument();
      expect(screen.getByText("NLP Research Lab")).toBeInTheDocument();
      expect(screen.getByText("Computer Vision Lab")).toBeInTheDocument();
      expect(screen.getByText("Biotech NLP Lab")).toBeInTheDocument();
    });

    it("should render custom domain CTA", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      expect(screen.getByText("Create Custom Domain")).toBeInTheDocument();
    });
  });

  describe("Domain Cards", () => {
    it("should show agent count for each domain", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      // Multiple domains have 3 agents (Voice Clone, Robotics ML)
      expect(screen.getAllByText("3 agents").length).toBeGreaterThanOrEqual(1);
      // Quant Trading and NLP have 5 agents
      expect(screen.getAllByText("5 agents").length).toBeGreaterThanOrEqual(1);
    });

    it("should show paper count for each domain", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      // Voice Clone has 127 papers
      expect(screen.getByText("127 papers")).toBeInTheDocument();
    });

    it("should show domain descriptions", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      expect(
        screen.getByText("Prosody and emotion conditioning in TTS systems")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Algorithmic trading and portfolio optimization")
      ).toBeInTheDocument();
    });

    it("should show breakthrough text when available", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      expect(
        screen.getByText("Emotion transfer without paired data")
      ).toBeInTheDocument();
    });

    it("should show domain tags", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      expect(screen.getByText("TTS")).toBeInTheDocument();
      expect(screen.getByText("Prosody")).toBeInTheDocument();
      expect(screen.getByText("Finance")).toBeInTheDocument();
    });
  });

  describe("Links", () => {
    it("should link to correct domain lab page", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      const voiceCloneCard = screen.getByText("Voice Clone Lab").closest("a");
      expect(voiceCloneCard).toHaveAttribute("href", "/lab?domain=voice-clone");
    });

    it("should link to custom domain creation", () => {
      render(<DomainShowcase domains={defaultDomains} />);

      const customDomainLink = screen.getByText("Create Custom Domain").closest("a");
      expect(customDomainLink).toHaveAttribute("href", "/domains/new");
    });
  });

  describe("Custom Domains", () => {
    it("should render custom domains when provided", () => {
      // Note: DomainInfo interface uses nameKey and descKey for translation lookup
      // If the key doesn't exist in translations, it returns the key itself
      const customDomains = [
        {
          slug: "custom-domain",
          nameKey: "customDomainName",
          descKey: "customDomainDesc",
          primaryColor: "#ff0000",
          accentColor: "#00ff00",
          icon: null,
          activeAgents: 2,
          papersProcessed: 50,
          tags: ["custom", "test"],
        },
      ];

      render(<DomainShowcase domains={customDomains} />);

      // When translation key doesn't exist, it falls back to the key name
      expect(screen.getByText("customDomainName")).toBeInTheDocument();
      expect(screen.getByText("customDomainDesc")).toBeInTheDocument();
      expect(screen.getByText("2 agents")).toBeInTheDocument();
      expect(screen.getByText("50 papers")).toBeInTheDocument();
    });

    it("should handle empty domains array", () => {
      render(<DomainShowcase domains={[]} />);

      expect(screen.getByText("Choose Your Research Domain")).toBeInTheDocument();
      expect(screen.queryByText("Voice Clone Lab")).not.toBeInTheDocument();
    });
  });
});
