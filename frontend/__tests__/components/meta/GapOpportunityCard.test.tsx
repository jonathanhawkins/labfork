/**
 * Tests for GapOpportunityCard Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GapOpportunityCard, GapCard } from "@/components/meta/GapOpportunityCard";
import { GapOpportunity, ResearchGap, createDefaultEffort } from "@/lib/meta/gaps";

const mockOpportunity: GapOpportunity = {
  id: "opp-123",
  gapId: "gap-456",
  title: "Explore Transformer-Diffusion Combination",
  description: "Combine transformer attention with diffusion models for improved quality.",
  type: "technique_combination",
  impactScore: 0.85,
  effort: {
    level: "medium",
    personWeeks: 3,
    skills: ["deep learning", "python"],
    resources: ["GPU"],
    risks: ["complexity"],
  },
  prerequisites: ["Understanding of transformers", "Diffusion model knowledge"],
  potentialOutcomes: ["Higher quality output", "Novel architecture"],
  suggestedTechniques: ["tech-1", "tech-2"],
  priorityScore: 0.72,
  identifiedAt: new Date(),
  confidence: 0.8,
};

const mockGap: ResearchGap = {
  id: "gap-456",
  type: "unexplored_combination",
  title: "Transformer-Diffusion Gap",
  description: "No existing work combines these approaches.",
  severity: "high",
  domains: ["speech", "audio"],
  relatedTechniques: ["tech-1", "tech-2"],
  evidence: [],
  detectedAt: new Date(),
  confidence: 0.75,
};

describe("GapOpportunityCard", () => {
  it("should render opportunity title", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    expect(screen.getByText("Explore Transformer-Diffusion Combination")).toBeInTheDocument();
  });

  it("should display priority score", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    expect(screen.getByText("72")).toBeInTheDocument();
    expect(screen.getByText("Priority")).toBeInTheDocument();
  });

  it("should display impact score", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    expect(screen.getByText("85%")).toBeInTheDocument();
    expect(screen.getByText("Impact")).toBeInTheDocument();
  });

  it("should display effort level", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    expect(screen.getByText(/Medium/)).toBeInTheDocument();
  });

  it("should show opportunity type badge", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    expect(screen.getByText("technique combination")).toBeInTheDocument();
  });

  it("should show description", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    expect(screen.getByText(/Combine transformer attention/)).toBeInTheDocument();
  });

  it("should toggle details on click", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    const toggleButton = screen.getByText("Show Details");
    fireEvent.click(toggleButton);

    expect(screen.getByText("Prerequisites")).toBeInTheDocument();
    expect(screen.getByText("Potential Outcomes")).toBeInTheDocument();
    expect(screen.getByText("Hide Details")).toBeInTheDocument();
  });

  it("should display prerequisites when expanded", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    fireEvent.click(screen.getByText("Show Details"));

    expect(screen.getByText("Understanding of transformers")).toBeInTheDocument();
    expect(screen.getByText("Diffusion model knowledge")).toBeInTheDocument();
  });

  it("should display skills when expanded", () => {
    render(<GapOpportunityCard opportunity={mockOpportunity} />);

    fireEvent.click(screen.getByText("Show Details"));

    expect(screen.getByText("deep learning")).toBeInTheDocument();
    expect(screen.getByText("python")).toBeInTheDocument();
  });

  it("should call onPursue when button clicked", () => {
    const onPursue = vi.fn();
    render(<GapOpportunityCard opportunity={mockOpportunity} onPursue={onPursue} />);

    fireEvent.click(screen.getByText("Pursue Opportunity"));

    expect(onPursue).toHaveBeenCalledWith(mockOpportunity.id);
  });

  it("should call onDismiss when button clicked", () => {
    const onDismiss = vi.fn();
    render(<GapOpportunityCard opportunity={mockOpportunity} onDismiss={onDismiss} />);

    fireEvent.click(screen.getByText("Dismiss"));

    expect(onDismiss).toHaveBeenCalledWith(mockOpportunity.id);
  });

  it("should show gap info when provided", () => {
    render(
      <GapOpportunityCard
        opportunity={mockOpportunity}
        gap={mockGap}
        showGapInfo={true}
      />
    );

    fireEvent.click(screen.getByText("Show Details"));

    expect(screen.getByText("Related Gap")).toBeInTheDocument();
    expect(screen.getByText("Transformer-Diffusion Gap")).toBeInTheDocument();
  });

  it("should render compact version", () => {
    const onViewDetails = vi.fn();
    render(
      <GapOpportunityCard
        opportunity={mockOpportunity}
        compact={true}
        onViewDetails={onViewDetails}
      />
    );

    // Compact version should not show full details
    expect(screen.queryByText("Prerequisites")).not.toBeInTheDocument();
    expect(screen.queryByText("Potential Outcomes")).not.toBeInTheDocument();

    // But should show title and scores
    expect(screen.getByText("Explore Transformer-Diffusion Combination")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    const { container } = render(
      <GapOpportunityCard opportunity={mockOpportunity} className="custom-class" />
    );

    expect(container.firstChild).toHaveClass("custom-class");
  });
});

describe("GapCard", () => {
  it("should render gap title", () => {
    render(<GapCard gap={mockGap} />);

    expect(screen.getByText("Transformer-Diffusion Gap")).toBeInTheDocument();
  });

  it("should display severity badge", () => {
    render(<GapCard gap={mockGap} />);

    expect(screen.getByText("high")).toBeInTheDocument();
  });

  it("should display gap type", () => {
    render(<GapCard gap={mockGap} />);

    expect(screen.getByText("Unexplored Combination")).toBeInTheDocument();
  });

  it("should show domains", () => {
    render(<GapCard gap={mockGap} />);

    expect(screen.getByText("speech")).toBeInTheDocument();
    expect(screen.getByText("audio")).toBeInTheDocument();
  });

  it("should display confidence score", () => {
    render(<GapCard gap={mockGap} />);

    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("Confidence")).toBeInTheDocument();
  });

  it("should show opportunities count when provided", () => {
    const opportunities = [mockOpportunity];
    const onViewOpportunities = vi.fn();

    render(
      <GapCard
        gap={mockGap}
        opportunities={opportunities}
        onViewOpportunities={onViewOpportunities}
      />
    );

    expect(screen.getByText("1 opportunity available")).toBeInTheDocument();
  });

  it("should call onViewOpportunities when clicked", () => {
    const opportunities = [mockOpportunity];
    const onViewOpportunities = vi.fn();

    render(
      <GapCard
        gap={mockGap}
        opportunities={opportunities}
        onViewOpportunities={onViewOpportunities}
      />
    );

    fireEvent.click(screen.getByText("1 opportunity available"));

    expect(onViewOpportunities).toHaveBeenCalledWith(mockGap.id);
  });
});
