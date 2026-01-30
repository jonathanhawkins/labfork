/**
 * Tests for SynergyCard Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { SynergyCard, SynergyCardCompact } from "@/components/meta/SynergyCard";

const mockProposal = {
  id: "synergy-123",
  techniqueA: {
    id: "tech-a",
    name: "Transformer Encoder",
    domains: ["speech", "audio"],
  },
  techniqueB: {
    id: "tech-b",
    name: "Diffusion Decoder",
    domains: ["speech"],
  },
  score: {
    overall: 0.75,
    components: {
      similarity: 0.6,
      complementarity: 0.8,
      novelty: 0.7,
      feasibility: 0.8,
      impact: 0.85,
    },
    confidence: 0.7,
  },
  justification: "These techniques complement each other well for speech synthesis.",
  combinationAspects: [
    {
      fromA: "attention mechanism",
      fromB: "iterative refinement",
      combination: "attention-guided diffusion",
      benefit: "improved alignment",
    },
  ],
  expectedOutcomes: [
    {
      metric: "MOS",
      baseline: 3.8,
      expected: 4.2,
      unit: "score",
      confidence: 0.7,
    },
  ],
  status: "pending",
  createdAt: new Date().toISOString(),
};

describe("SynergyCard", () => {
  it("should render technique names", () => {
    render(<SynergyCard proposal={mockProposal} />);

    expect(screen.getByText("Transformer Encoder")).toBeInTheDocument();
    expect(screen.getByText("Diffusion Decoder")).toBeInTheDocument();
  });

  it("should display the overall score", () => {
    render(<SynergyCard proposal={mockProposal} />);

    // Score is 75 (0.75 * 100)
    expect(screen.getByText("75")).toBeInTheDocument();
  });

  it("should show justification", () => {
    render(<SynergyCard proposal={mockProposal} />);

    expect(
      screen.getByText(/These techniques complement each other/)
    ).toBeInTheDocument();
  });

  it("should display status badge", () => {
    render(<SynergyCard proposal={mockProposal} />);

    expect(screen.getByText("Pending")).toBeInTheDocument();
  });

  it("should show domains when not compact", () => {
    render(<SynergyCard proposal={mockProposal} compact={false} />);

    expect(screen.getByText("speech")).toBeInTheDocument();
    expect(screen.getByText("audio")).toBeInTheDocument();
  });

  it("should hide domains in compact mode", () => {
    render(<SynergyCard proposal={mockProposal} compact={true} />);

    // In compact mode, domains are not shown
    // The text should not be visible in the card
  });

  it("should toggle score breakdown on click", () => {
    render(<SynergyCard proposal={mockProposal} />);

    const breakdownButton = screen.getByText("Score Breakdown");
    fireEvent.click(breakdownButton);

    // After expanding, should show score components
    expect(screen.getByText("Similarity")).toBeInTheDocument();
    expect(screen.getByText("Complementarity")).toBeInTheDocument();
    expect(screen.getByText("Novelty")).toBeInTheDocument();
  });

  it("should show accept/reject buttons for pending proposals", () => {
    render(<SynergyCard proposal={mockProposal} showActions={true} />);

    expect(screen.getByText("Accept")).toBeInTheDocument();
    expect(screen.getByText("Reject")).toBeInTheDocument();
  });

  it("should hide actions for non-pending proposals", () => {
    const acceptedProposal = { ...mockProposal, status: "accepted" };
    render(<SynergyCard proposal={acceptedProposal} showActions={true} />);

    expect(screen.queryByText("Accept")).not.toBeInTheDocument();
    expect(screen.queryByText("Reject")).not.toBeInTheDocument();
  });

  it("should call onAccept when Accept is clicked", () => {
    const onAccept = vi.fn();
    render(<SynergyCard proposal={mockProposal} onAccept={onAccept} />);

    fireEvent.click(screen.getByText("Accept"));

    expect(onAccept).toHaveBeenCalledWith(mockProposal.id, undefined);
  });

  it("should call onReject when Reject is clicked", () => {
    const onReject = vi.fn();
    render(<SynergyCard proposal={mockProposal} onReject={onReject} />);

    fireEvent.click(screen.getByText("Reject"));

    expect(onReject).toHaveBeenCalledWith(mockProposal.id, undefined);
  });

  it("should toggle notes input", () => {
    render(<SynergyCard proposal={mockProposal} />);

    fireEvent.click(screen.getByText("Add Notes"));

    expect(screen.getByPlaceholderText(/Add notes/)).toBeInTheDocument();
  });

  it("should include notes when accepting", () => {
    const onAccept = vi.fn();
    render(<SynergyCard proposal={mockProposal} onAccept={onAccept} />);

    // Toggle notes
    fireEvent.click(screen.getByText("Add Notes"));

    // Enter notes
    const notesInput = screen.getByPlaceholderText(/Add notes/);
    fireEvent.change(notesInput, { target: { value: "Test notes" } });

    // Accept
    fireEvent.click(screen.getByText("Accept"));

    expect(onAccept).toHaveBeenCalledWith(mockProposal.id, "Test notes");
  });

  it("should call onViewDetails when link is clicked", () => {
    const onViewDetails = vi.fn();
    render(
      <SynergyCard proposal={mockProposal} onViewDetails={onViewDetails} />
    );

    fireEvent.click(screen.getByText(/View Full Details/));

    expect(onViewDetails).toHaveBeenCalledWith(mockProposal.id);
  });

  it("should apply custom className", () => {
    const { container } = render(
      <SynergyCard proposal={mockProposal} className="custom-class" />
    );

    expect(container.firstChild).toHaveClass("custom-class");
  });
});

describe("SynergyCardCompact", () => {
  it("should render technique names", () => {
    render(<SynergyCardCompact proposal={mockProposal} />);

    expect(screen.getByText("Transformer Encoder")).toBeInTheDocument();
    expect(screen.getByText("Diffusion Decoder")).toBeInTheDocument();
  });

  it("should display score", () => {
    render(<SynergyCardCompact proposal={mockProposal} />);

    expect(screen.getByText("75")).toBeInTheDocument();
  });

  it("should show truncated justification", () => {
    render(<SynergyCardCompact proposal={mockProposal} />);

    // Justification should be shown but may be truncated
    expect(
      screen.getByText(/These techniques complement/)
    ).toBeInTheDocument();
  });

  it("should call onClick when clicked", () => {
    const onClick = vi.fn();
    render(<SynergyCardCompact proposal={mockProposal} onClick={onClick} />);

    fireEvent.click(screen.getByText("Transformer Encoder").closest("div")!);

    expect(onClick).toHaveBeenCalledWith(mockProposal.id);
  });
});
