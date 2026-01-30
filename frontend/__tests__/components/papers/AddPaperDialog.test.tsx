import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AddPaperDialog } from "@/components/papers/AddPaperDialog";

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("AddPaperDialog", () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it("does not render when closed", () => {
    render(<AddPaperDialog isOpen={false} onClose={() => {}} />);
    expect(screen.queryByText("Add Paper")).not.toBeInTheDocument();
  });

  it("renders when open", () => {
    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);
    // Use heading role to find the title specifically
    expect(screen.getByRole("heading", { name: "Add Paper" })).toBeInTheDocument();
  });

  it("shows input step by default", () => {
    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);
    expect(screen.getByText(/Enter arXiv ID, DOI, or URL/i)).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/e.g., 2401.12345/)
    ).toBeInTheDocument();
  });

  it("calls onClose when X button clicked", () => {
    const onClose = vi.fn();
    render(<AddPaperDialog isOpen={true} onClose={onClose} />);

    // Find and click the close button (X icon)
    const closeButtons = screen.getAllByRole("button");
    const closeButton = closeButtons.find((btn) =>
      btn.querySelector("svg.lucide-x")
    );
    if (closeButton) {
      fireEvent.click(closeButton);
    }

    expect(onClose).toHaveBeenCalled();
  });

  it("calls onClose when Cancel button clicked", () => {
    const onClose = vi.fn();
    render(<AddPaperDialog isOpen={true} onClose={onClose} />);

    const cancelButton = screen.getByRole("button", { name: /Cancel/i });
    fireEvent.click(cancelButton);

    expect(onClose).toHaveBeenCalled();
  });

  it("updates input value when typing", () => {
    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);

    const input = screen.getByPlaceholderText(/e.g., 2401.12345/);
    fireEvent.change(input, { target: { value: "2401.12345" } });

    expect(input).toHaveValue("2401.12345");
  });

  it("detects arXiv input type", async () => {
    mockFetch.mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          detection: {
            source: "arxiv",
            identifier: "2401.12345",
            confidence: 1.0,
          },
        }),
    });

    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);

    const input = screen.getByPlaceholderText(/e.g., 2401.12345/);
    fireEvent.change(input, { target: { value: "2401.12345" } });

    await waitFor(() => {
      expect(screen.getByText("arXiv")).toBeInTheDocument();
    });
  });

  it("calls Add Paper when button clicked with valid input", async () => {
    mockFetch.mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          success: true,
          paper: {
            id: "paper-1",
            metadata: {
              title: "Test Paper",
              abstract: "Test abstract",
              authors: [{ name: "Author" }],
              source: "arxiv",
              url: "https://arxiv.org/abs/2401.12345",
            },
            status: "fetched",
          },
        }),
    });

    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);

    const input = screen.getByPlaceholderText(/e.g., 2401.12345/);
    fireEvent.change(input, { target: { value: "2401.12345" } });

    const addButton = screen.getByRole("button", { name: /Add Paper/i });
    fireEvent.click(addButton);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        "/api/papers",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });

  it("shows preview step after successful fetch", async () => {
    mockFetch.mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          success: true,
          paper: {
            id: "paper-1",
            metadata: {
              title: "Test Paper Title",
              abstract: "This is the paper abstract.",
              authors: [{ name: "John Doe" }],
              source: "arxiv",
              url: "https://arxiv.org/abs/2401.12345",
            },
            status: "fetched",
          },
        }),
    });

    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);

    const input = screen.getByPlaceholderText(/e.g., 2401.12345/);
    fireEvent.change(input, { target: { value: "2401.12345" } });

    const addButton = screen.getByRole("button", { name: /Add Paper/i });
    fireEvent.click(addButton);

    await waitFor(() => {
      expect(screen.getByText("Test Paper Title")).toBeInTheDocument();
    });
  });

  it("shows error message on fetch failure", async () => {
    mockFetch.mockResolvedValueOnce({
      json: () =>
        Promise.resolve({
          success: false,
          error: "Paper not found",
        }),
    });

    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);

    const input = screen.getByPlaceholderText(/e.g., 2401.12345/);
    fireEvent.change(input, { target: { value: "invalid-id" } });

    const addButton = screen.getByRole("button", { name: /Add Paper/i });
    fireEvent.click(addButton);

    await waitFor(() => {
      expect(screen.getByText("Paper not found")).toBeInTheDocument();
    });
  });

  it("shows supported formats list", () => {
    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);

    expect(screen.getByText("Supported formats:")).toBeInTheDocument();
    expect(screen.getByText(/arXiv ID: 2401.12345/)).toBeInTheDocument();
  });

  it("disables Add Paper button when input is empty", () => {
    render(<AddPaperDialog isOpen={true} onClose={() => {}} />);

    const addButton = screen.getByRole("button", { name: /Add Paper/i });
    expect(addButton).toBeDisabled();
  });

  it("calls onClose when clicking outside the dialog", () => {
    const onClose = vi.fn();
    render(<AddPaperDialog isOpen={true} onClose={onClose} />);

    // The backdrop is the fixed div with the bg-black/60 class
    const backdrop = document.querySelector(".fixed.inset-0");
    if (backdrop) {
      // Simulate click on the backdrop itself (not a child)
      fireEvent.click(backdrop, {
        target: backdrop,
        currentTarget: backdrop,
      });
    }

    expect(onClose).toHaveBeenCalled();
  });
});
