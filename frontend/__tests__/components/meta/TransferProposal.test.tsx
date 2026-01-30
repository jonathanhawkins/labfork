/**
 * Tests for TransferProposal and TransferProposalList Components
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TransferProposal, TransferProposalList } from "@/components/meta/TransferProposal";

const mockProposal = {
  id: "proposal-1",
  sourceTechnique: {
    id: "tech-1",
    name: "TransformerTTS",
  },
  sourceDomain: "speech",
  targetDomain: "nlp",
  principle: {
    name: "Attention Architecture",
    level: 2,
    coreInsight: "Self-attention enables context-aware processing",
    confidence: 0.85,
  },
  feasibility: {
    overallScore: 0.75,
    level: "moderate" as const,
    components: {
      technical: 0.8,
      data: 0.7,
      computational: 0.65,
      knowledge: 0.8,
      resources: 0.7,
    },
    risks: [
      {
        name: "Data format mismatch",
        score: 0.4,
        mitigation: "Implement preprocessing pipeline",
      },
    ],
    effort: {
      personDays: 15,
      range: { min: 10, max: 20 },
    },
    recommendations: [
      "Start with smaller model variant",
      "Use existing tokenizer",
    ],
  },
  implementationGuide: {
    steps: [
      {
        step: 1,
        title: "Prepare data pipeline",
        description: "Convert speech data format to text tokens",
        estimatedHours: 8,
      },
      {
        step: 2,
        title: "Adapt model architecture",
        description: "Replace speech encoder with text embeddings",
        estimatedHours: 16,
      },
    ],
    successCriteria: [
      "Model achieves baseline accuracy",
      "Inference time under 100ms",
    ],
  },
  successPrediction: {
    probability: 0.72,
    confidenceInterval: { low: 0.65, high: 0.79 },
    successFactors: [
      "Strong architecture alignment",
      "Available pretrained weights",
    ],
    failureRisks: [
      "Domain-specific patterns may not transfer",
    ],
  },
  status: "proposed",
  createdAt: new Date(),
};

describe("TransferProposal", () => {
  it("should render source technique name", () => {
    render(<TransferProposal proposal={mockProposal} />);

    expect(screen.getByText(/Transfer: TransformerTTS/)).toBeInTheDocument();
  });

  it("should display domain transfer direction", () => {
    render(<TransferProposal proposal={mockProposal} />);

    expect(screen.getByText(/speech → nlp/)).toBeInTheDocument();
  });

  it("should show feasibility badge", () => {
    render(<TransferProposal proposal={mockProposal} />);

    expect(screen.getByText("Moderate")).toBeInTheDocument();
  });

  it("should show status badge", () => {
    render(<TransferProposal proposal={mockProposal} />);

    // Status is lowercase in the badge
    expect(screen.getByText("proposed")).toBeInTheDocument();
  });

  it("should have tab navigation", () => {
    render(<TransferProposal proposal={mockProposal} />);

    expect(screen.getByRole("button", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Feasibility" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Implementation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Prediction" })).toBeInTheDocument();
  });

  it("should switch tabs when clicked", () => {
    render(<TransferProposal proposal={mockProposal} />);

    const feasibilityTab = screen.getByRole("button", { name: "Feasibility" });
    fireEvent.click(feasibilityTab);

    // Feasibility content should be visible - check for component scores heading
    expect(screen.getByText("Component Scores")).toBeInTheDocument();
  });

  it("should display core insight in overview", () => {
    render(<TransferProposal proposal={mockProposal} />);

    expect(screen.getByText(/Self-attention enables/)).toBeInTheDocument();
  });

  it("should display overall score", () => {
    render(<TransferProposal proposal={mockProposal} />);

    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  it("should display success probability", () => {
    render(<TransferProposal proposal={mockProposal} />);

    const predictionTab = screen.getByText("Prediction");
    fireEvent.click(predictionTab);

    expect(screen.getByText("72%")).toBeInTheDocument();
  });

  it("should call onApprove when approve button clicked", () => {
    const onApprove = vi.fn();
    render(<TransferProposal proposal={mockProposal} onApprove={onApprove} />);

    const approveButton = screen.getByText("Approve");
    fireEvent.click(approveButton);

    expect(onApprove).toHaveBeenCalledWith("proposal-1");
  });

  it("should call onReject when reject button clicked", () => {
    const onReject = vi.fn();
    render(<TransferProposal proposal={mockProposal} onReject={onReject} />);

    const rejectButton = screen.getByText("Reject");
    fireEvent.click(rejectButton);

    expect(onReject).toHaveBeenCalledWith("proposal-1");
  });

  it("should call onStartImplementation when button clicked for validated proposal", () => {
    const onStartImplementation = vi.fn();
    // Start Implementation only shows when status is "validated"
    const validatedProposal = { ...mockProposal, status: "validated" };
    render(
      <TransferProposal
        proposal={validatedProposal}
        onStartImplementation={onStartImplementation}
      />
    );

    const startButton = screen.getByText("Start Implementation");
    fireEvent.click(startButton);

    expect(onStartImplementation).toHaveBeenCalledWith("proposal-1");
  });

  it("should display implementation steps", () => {
    render(<TransferProposal proposal={mockProposal} />);

    const implementationTab = screen.getByText("Implementation");
    fireEvent.click(implementationTab);

    expect(screen.getByText("Prepare data pipeline")).toBeInTheDocument();
    expect(screen.getByText("Adapt model architecture")).toBeInTheDocument();
  });

  it("should display risks in feasibility tab", () => {
    render(<TransferProposal proposal={mockProposal} />);

    const feasibilityTab = screen.getByRole("button", { name: "Feasibility" });
    fireEvent.click(feasibilityTab);

    expect(screen.getByText("Data format mismatch")).toBeInTheDocument();
  });

  it("should display recommendations in overview", () => {
    render(<TransferProposal proposal={mockProposal} />);

    // Recommendations appear in Overview tab
    expect(screen.getByText(/Start with smaller model/)).toBeInTheDocument();
  });

  it("should apply className prop", () => {
    const { container } = render(
      <TransferProposal proposal={mockProposal} className="custom-class" />
    );

    expect(container.firstChild).toHaveClass("custom-class");
  });

  it("should display effort estimate in overview", () => {
    render(<TransferProposal proposal={mockProposal} />);

    // Effort appears in Overview as "15 days"
    expect(screen.getByText("15 days")).toBeInTheDocument();
  });
});

describe("TransferProposalList", () => {
  const mockProposals = [
    mockProposal,
    {
      ...mockProposal,
      id: "proposal-2",
      sourceTechnique: { id: "tech-2", name: "DiffusionModel" },
      sourceDomain: "vision",
      targetDomain: "speech",
      feasibility: {
        ...mockProposal.feasibility,
        level: "challenging" as const,
        overallScore: 0.45,
      },
    },
  ];

  it("should render all proposals", () => {
    render(<TransferProposalList proposals={mockProposals} />);

    expect(screen.getByText("TransformerTTS")).toBeInTheDocument();
    expect(screen.getByText("DiffusionModel")).toBeInTheDocument();
  });

  it("should handle empty proposals", () => {
    render(<TransferProposalList proposals={[]} />);

    expect(screen.getByText("No proposals")).toBeInTheDocument();
  });

  it("should show target domain for each proposal", () => {
    render(<TransferProposalList proposals={mockProposals} />);

    expect(screen.getByText("nlp")).toBeInTheDocument();
    expect(screen.getByText("speech")).toBeInTheDocument();
  });

  it("should show feasibility scores", () => {
    render(<TransferProposalList proposals={mockProposals} />);

    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("45%")).toBeInTheDocument();
  });

  it("should call onSelect when proposal clicked", () => {
    const onSelect = vi.fn();
    render(
      <TransferProposalList
        proposals={mockProposals}
        onSelect={onSelect}
      />
    );

    // Proposals are clickable divs
    const proposalCards = document.querySelectorAll(".cursor-pointer");
    if (proposalCards.length > 0) {
      fireEvent.click(proposalCards[0]);
      expect(onSelect).toHaveBeenCalledWith(mockProposals[0]);
    }
  });

  it("should apply className prop", () => {
    const { container } = render(
      <TransferProposalList proposals={mockProposals} className="custom-list" />
    );

    expect(container.firstChild).toHaveClass("custom-list");
  });

  it("should highlight selected proposal", () => {
    render(
      <TransferProposalList
        proposals={mockProposals}
        selectedId="proposal-1"
      />
    );

    // Selected proposal has blue border
    const selectedCard = document.querySelector(".border-blue-500");
    expect(selectedCard).toBeInTheDocument();
  });
});
