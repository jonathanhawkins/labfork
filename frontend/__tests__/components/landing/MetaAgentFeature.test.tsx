/**
 * MetaAgentFeature Component Tests
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MetaAgentFeature } from "@/components/landing/MetaAgentFeature";

describe("MetaAgentFeature", () => {
  describe("Rendering", () => {
    it("should render section header", () => {
      render(<MetaAgentFeature />);

      expect(screen.getByText("5 AI Agents Working Together")).toBeInTheDocument();
    });

    it("should render section badge", () => {
      render(<MetaAgentFeature />);

      expect(screen.getByText("Meta-Research Intelligence")).toBeInTheDocument();
    });

    it("should render section description", () => {
      render(<MetaAgentFeature />);

      expect(
        screen.getByText(/A team of specialized agents that analyze/i)
      ).toBeInTheDocument();
    });

    it("should render all 5 agent names in selector", () => {
      render(<MetaAgentFeature />);

      // Synergy Detector appears twice (selector + detail panel when selected)
      expect(screen.getAllByText("Synergy Detector").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Pattern Recognizer")).toBeInTheDocument();
      expect(screen.getByText("Gap Analyzer")).toBeInTheDocument();
      expect(screen.getByText("Evolution Engine")).toBeInTheDocument();
      expect(screen.getByText("Transfer Agent")).toBeInTheDocument();
    });
  });

  describe("Agent Selection", () => {
    it("should show Synergy Detector details by default", () => {
      render(<MetaAgentFeature />);

      // Description appears in both selector and detail panel
      expect(
        screen.getAllByText("Finds powerful combinations of techniques from different papers").length
      ).toBeGreaterThanOrEqual(1);
    });

    it("should show Synergy Detector capabilities", () => {
      render(<MetaAgentFeature />);

      expect(screen.getByText("Analyzes technique compatibility")).toBeInTheDocument();
      expect(screen.getByText("Scores potential synergies")).toBeInTheDocument();
    });

    it("should show Synergy Detector example discovery", () => {
      render(<MetaAgentFeature />);

      expect(screen.getByText("Voice + Sentiment Synergy")).toBeInTheDocument();
    });

    it("should switch to Pattern Recognizer when clicked", () => {
      render(<MetaAgentFeature />);

      // Find and click Pattern Recognizer
      const patternButton = screen.getByText("Pattern Recognizer").closest("button");
      fireEvent.click(patternButton!);

      // Check for Pattern Recognizer content (appears in both selector and detail panel)
      expect(
        screen.getAllByText("Identifies emerging research trends and paradigm shifts").length
      ).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Tracks citation velocity")).toBeInTheDocument();
    });

    it("should switch to Gap Analyzer when clicked", () => {
      render(<MetaAgentFeature />);

      const gapButton = screen.getByText("Gap Analyzer").closest("button");
      fireEvent.click(gapButton!);

      // Description appears in both selector and detail panel
      expect(
        screen.getAllByText("Discovers unexplored research opportunities").length
      ).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Maps research landscape")).toBeInTheDocument();
    });

    it("should switch to Evolution Engine when clicked", () => {
      render(<MetaAgentFeature />);

      const evolutionButton = screen.getByText("Evolution Engine").closest("button");
      fireEvent.click(evolutionButton!);

      // Description appears in both selector and detail panel
      expect(
        screen.getAllByText("Evolves techniques through genetic algorithms").length
      ).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Crossover of successful techniques")).toBeInTheDocument();
    });

    it("should switch to Transfer Agent when clicked", () => {
      render(<MetaAgentFeature />);

      const transferButton = screen.getByText("Transfer Agent").closest("button");
      fireEvent.click(transferButton!);

      // Description appears in both selector and detail panel
      expect(
        screen.getAllByText("Bridges knowledge across different research domains").length
      ).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Domain similarity analysis")).toBeInTheDocument();
    });
  });

  describe("Stats Display", () => {
    it("should show synergies found count", () => {
      render(<MetaAgentFeature />);

      expect(screen.getByText("1,247")).toBeInTheDocument();
      expect(screen.getByText("Synergies Found")).toBeInTheDocument();
    });

    it("should show techniques evolved count", () => {
      render(<MetaAgentFeature />);

      expect(screen.getByText("89")).toBeInTheDocument();
      expect(screen.getByText("Techniques Evolved")).toBeInTheDocument();
    });

    it("should show cross-domain transfers count", () => {
      render(<MetaAgentFeature />);

      expect(screen.getByText("23")).toBeInTheDocument();
      expect(screen.getByText("Cross-Domain Transfers")).toBeInTheDocument();
    });
  });

  describe("Example Discoveries", () => {
    it("should show example discovery for each agent", () => {
      render(<MetaAgentFeature />);

      // Synergy Detector (default)
      expect(screen.getByText("Voice + Sentiment Synergy")).toBeInTheDocument();

      // Switch to Pattern Recognizer
      const patternButton = screen.getByText("Pattern Recognizer").closest("button");
      fireEvent.click(patternButton!);
      expect(screen.getByText("Diffusion Trend")).toBeInTheDocument();

      // Switch to Gap Analyzer
      const gapButton = screen.getByText("Gap Analyzer").closest("button");
      fireEvent.click(gapButton!);
      expect(screen.getByText("Prosody Gap")).toBeInTheDocument();

      // Switch to Evolution Engine
      const evolutionButton = screen.getByText("Evolution Engine").closest("button");
      fireEvent.click(evolutionButton!);
      expect(screen.getByText("Evolved Architecture")).toBeInTheDocument();

      // Switch to Transfer Agent
      const transferButton = screen.getByText("Transfer Agent").closest("button");
      fireEvent.click(transferButton!);
      expect(screen.getByText("Game to Finance Transfer")).toBeInTheDocument();
    });
  });
});
