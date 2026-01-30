import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { PaperQueue } from "@/components/papers/PaperQueue";
import type { Paper } from "@/lib/papers/types";

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("PaperQueue", () => {
  const mockPapers: Paper[] = [
    {
      id: "paper-1",
      metadata: {
        title: "First Test Paper",
        abstract: "Abstract for first paper",
        authors: [{ name: "Author One" }],
        source: "arxiv",
        sourceId: "2401.11111",
        url: "https://arxiv.org/abs/2401.11111",
      },
      status: "fetched",
      addedAt: new Date("2024-01-15").toISOString(),
    },
    {
      id: "paper-2",
      metadata: {
        title: "Second Test Paper",
        abstract: "Abstract for second paper",
        authors: [{ name: "Author Two" }],
        source: "arxiv",
        sourceId: "2401.22222",
        url: "https://arxiv.org/abs/2401.22222",
      },
      status: "analyzed",
      analysis: {
        relevanceScore: 90,
        relevanceReason: "Highly relevant",
        complexity: "moderate",
        complexityReason: "Medium effort",
        novelty: "Novel approach",
        techniques: [],
        resources: [],
        taskBreakdown: {
          research: { title: "Research", description: "...", estimatedHours: 4 },
          implementation: { title: "Impl", description: "...", estimatedHours: 16 },
          evaluation: { title: "Eval", description: "...", estimatedHours: 8, metrics: [] },
        },
        analyzedAt: new Date().toISOString(),
      },
      addedAt: new Date("2024-01-14").toISOString(),
    },
  ];

  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("renders papers from initialPapers prop", () => {
    render(<PaperQueue initialPapers={mockPapers} />);

    expect(screen.getByText("First Test Paper")).toBeInTheDocument();
    expect(screen.getByText("Second Test Paper")).toBeInTheDocument();
  });

  it("renders loading state when fetching", () => {
    mockFetch.mockImplementation(
      () =>
        new Promise(() => {
          // Never resolves to keep loading state
        })
    );

    render(<PaperQueue autoRefresh={false} />);

    // Look for the loading spinner (Loader2 icon creates a spinner)
    const loadingSpinner = document.querySelector(".animate-spin");
    expect(loadingSpinner).toBeInTheDocument();
  });

  it("renders empty state when no papers", () => {
    render(<PaperQueue initialPapers={[]} autoRefresh={false} />);

    expect(screen.getByText("No papers found")).toBeInTheDocument();
  });

  it("renders stats bar with correct counts", () => {
    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    expect(screen.getByText("2 papers")).toBeInTheDocument();
  });

  it("shows search input", () => {
    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    expect(screen.getByPlaceholderText("Search papers...")).toBeInTheDocument();
  });

  it("filters papers by search term", () => {
    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    const searchInput = screen.getByPlaceholderText("Search papers...");
    fireEvent.change(searchInput, { target: { value: "First" } });

    expect(screen.getByText("First Test Paper")).toBeInTheDocument();
    expect(screen.queryByText("Second Test Paper")).not.toBeInTheDocument();
  });

  it("clears search when X button clicked", () => {
    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    const searchInput = screen.getByPlaceholderText("Search papers...");
    fireEvent.change(searchInput, { target: { value: "First" } });

    // Find the clear button by looking for the X icon near the search input
    const clearButton = document.querySelector("input + button");
    if (clearButton) {
      fireEvent.click(clearButton);
    }

    expect(searchInput).toHaveValue("");
    expect(screen.getByText("Second Test Paper")).toBeInTheDocument();
  });

  it("shows filter panel when Filters button clicked", () => {
    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    const filtersButton = screen.getByRole("button", { name: /Filters/i });
    fireEvent.click(filtersButton);

    expect(screen.getByText("Status:")).toBeInTheDocument();
    expect(screen.getByText("Sort:")).toBeInTheDocument();
  });

  it("calls onPaperClick when paper is clicked", () => {
    const onPaperClick = vi.fn();
    render(
      <PaperQueue
        initialPapers={mockPapers}
        autoRefresh={false}
        onPaperClick={onPaperClick}
      />
    );

    // Click on the first paper's title area
    fireEvent.click(screen.getByText("First Test Paper"));

    expect(onPaperClick).toHaveBeenCalledWith(mockPapers[0]);
  });

  it("refreshes when refresh button clicked", async () => {
    mockFetch.mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          papers: mockPapers,
        }),
    });

    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    // Find the refresh button by looking for RefreshCw icon
    const refreshButton = document.querySelector("button svg.lucide-refresh-cw")?.parentElement;
    if (refreshButton) {
      fireEvent.click(refreshButton);
    }

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining("/api/papers"));
    });
  });

  it("shows pending count in stats", () => {
    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    expect(screen.getByText(/1 pending/)).toBeInTheDocument();
  });

  it("shows analyzed count in stats", () => {
    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    expect(screen.getByText("1 analyzed")).toBeInTheDocument();
  });

  it("handles analyze action", async () => {
    mockFetch.mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          success: true,
          paper: {
            ...mockPapers[0],
            status: "analyzed",
            analysis: {
              relevanceScore: 85,
              relevanceReason: "Relevant",
              complexity: "simple",
              complexityReason: "Easy",
              novelty: "Novel",
              techniques: [],
              resources: [],
              taskBreakdown: {
                research: { title: "R", description: "...", estimatedHours: 4 },
                implementation: { title: "I", description: "...", estimatedHours: 8 },
                evaluation: { title: "E", description: "...", estimatedHours: 4, metrics: [] },
              },
              analyzedAt: new Date().toISOString(),
            },
          },
        }),
    });

    render(<PaperQueue initialPapers={mockPapers} autoRefresh={false} />);

    // Find and click analyze button for the first paper
    const analyzeButton = screen.getByRole("button", { name: /Analyze/i });
    fireEvent.click(analyzeButton);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        "/api/papers/analyze",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });
});
