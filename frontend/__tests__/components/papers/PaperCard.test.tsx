import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PaperCard } from "@/components/papers/PaperCard";
import type { Paper } from "@/lib/papers/types";

describe("PaperCard", () => {
  const mockPaper: Paper = {
    id: "test-paper-1",
    metadata: {
      title: "Test Paper Title for Voice Cloning",
      abstract: "This is a test abstract about voice cloning techniques.",
      authors: [
        { name: "John Doe" },
        { name: "Jane Smith" },
        { name: "Bob Johnson" },
      ],
      source: "arxiv",
      sourceId: "2401.12345",
      url: "https://arxiv.org/abs/2401.12345",
      pdfUrl: "https://arxiv.org/pdf/2401.12345.pdf",
      publishedDate: "2024-01-15",
    },
    status: "fetched",
    addedAt: new Date().toISOString(),
  };

  const mockAnalyzedPaper: Paper = {
    ...mockPaper,
    id: "test-paper-2",
    status: "analyzed",
    analysis: {
      relevanceScore: 85,
      relevanceReason: "Highly relevant to prosody control",
      complexity: "moderate",
      complexityReason: "Requires moderate implementation effort",
      novelty: "Novel approach to emotion control",
      techniques: [
        {
          name: "Prosody Embedding",
          description: "Encodes prosody features",
          isMainContribution: true,
        },
      ],
      resources: [
        { type: "dataset", name: "LibriTTS", required: true },
      ],
      taskBreakdown: {
        research: {
          title: "Study technique",
          description: "Read the paper",
          estimatedHours: 4,
        },
        implementation: {
          title: "Implement technique",
          description: "Code implementation",
          estimatedHours: 16,
          codeAreas: ["training/"],
        },
        evaluation: {
          title: "Evaluate implementation",
          description: "Test the code",
          estimatedHours: 8,
          metrics: ["F0 correlation"],
        },
      },
      analyzedAt: new Date().toISOString(),
    },
  };

  it("renders paper title", () => {
    render(<PaperCard paper={mockPaper} />);
    expect(
      screen.getByText("Test Paper Title for Voice Cloning")
    ).toBeInTheDocument();
  });

  it("renders paper source badge", () => {
    render(<PaperCard paper={mockPaper} />);
    expect(screen.getByText("arXiv")).toBeInTheDocument();
  });

  it("renders paper status badge", () => {
    render(<PaperCard paper={mockPaper} />);
    expect(screen.getByText("Ready")).toBeInTheDocument();
  });

  it("renders authors", () => {
    render(<PaperCard paper={mockPaper} />);
    expect(screen.getByText(/John Doe/)).toBeInTheDocument();
  });

  it("renders external link", () => {
    render(<PaperCard paper={mockPaper} />);
    const link = screen.getByTitle("Open paper");
    expect(link).toHaveAttribute("href", "https://arxiv.org/abs/2401.12345");
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("calls onAnalyze when analyze button clicked for fetched paper", () => {
    const onAnalyze = vi.fn();
    render(<PaperCard paper={mockPaper} onAnalyze={onAnalyze} />);

    const analyzeButton = screen.getByRole("button", { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    expect(onAnalyze).toHaveBeenCalledWith("test-paper-1");
  });

  it("calls onAccept when accept button clicked for analyzed paper", () => {
    const onAccept = vi.fn();
    render(<PaperCard paper={mockAnalyzedPaper} onAccept={onAccept} />);

    const acceptButton = screen.getByRole("button", { name: /Accept/i });
    fireEvent.click(acceptButton);

    expect(onAccept).toHaveBeenCalledWith("test-paper-2");
  });

  it("calls onReject when reject button clicked for analyzed paper", () => {
    const onReject = vi.fn();
    render(<PaperCard paper={mockAnalyzedPaper} onReject={onReject} />);

    const rejectButton = screen.getByRole("button", { name: /Reject/i });
    fireEvent.click(rejectButton);

    expect(onReject).toHaveBeenCalledWith("test-paper-2");
  });

  it("shows relevance score for analyzed paper", () => {
    render(<PaperCard paper={mockAnalyzedPaper} />);
    expect(screen.getByText("85")).toBeInTheDocument();
  });

  it("shows complexity badge for analyzed paper", () => {
    render(<PaperCard paper={mockAnalyzedPaper} />);
    expect(screen.getByText("Moderate")).toBeInTheDocument();
  });

  it("disables buttons when loading", () => {
    const onAnalyze = vi.fn();
    render(
      <PaperCard paper={mockPaper} onAnalyze={onAnalyze} isLoading={true} />
    );

    const analyzeButton = screen.getByRole("button", { name: /Analyze/i });
    expect(analyzeButton).toBeDisabled();
  });

  it("expands to show abstract when clicked", () => {
    render(<PaperCard paper={mockPaper} />);

    // Click the expand button
    const expandButton = screen.getByRole("button", { name: "" });
    fireEvent.click(expandButton);

    // Abstract should now be visible
    expect(
      screen.getByText("This is a test abstract about voice cloning techniques.")
    ).toBeInTheDocument();
  });

  it("shows error message for error status", () => {
    const errorPaper: Paper = {
      ...mockPaper,
      status: "error",
      error: "Failed to fetch paper metadata",
    };
    render(<PaperCard paper={errorPaper} />);
    expect(screen.getByText("Failed to fetch paper metadata")).toBeInTheDocument();
  });

  it("hides actions in compact mode", () => {
    const onAnalyze = vi.fn();
    render(
      <PaperCard paper={mockPaper} onAnalyze={onAnalyze} compact={true} />
    );

    expect(screen.queryByRole("button", { name: /Analyze/i })).not.toBeInTheDocument();
  });
});
