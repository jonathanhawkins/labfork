import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { StarButton, StarIconButton } from "@/components/labs/StarButton";

describe("StarButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders with initial count", () => {
    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
      />
    );

    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("shows Star text when count is 0", () => {
    render(
      <StarButton
        labId="lab-1"
        initialCount={0}
        initialStarred={false}
      />
    );

    expect(screen.getByText("Star")).toBeInTheDocument();
  });

  it("hides count when showCount is false", () => {
    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
        showCount={false}
      />
    );

    expect(screen.queryByText("42")).not.toBeInTheDocument();
  });

  it("shows starred state when initially starred", () => {
    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={true}
      />
    );

    // Should have yellow styling and filled star
    const button = screen.getByRole("button");
    expect(button).toHaveClass("text-yellow-400");
  });

  it("shows correct title for starred state", () => {
    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={true}
      />
    );

    expect(screen.getByTitle("Unstar this lab")).toBeInTheDocument();
  });

  it("shows correct title for unstarred state", () => {
    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
      />
    );

    expect(screen.getByTitle("Star this lab")).toBeInTheDocument();
  });

  it("performs optimistic update on click", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: true, starred: true, count: 43 }),
    });

    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
      />
    );

    const button = screen.getByRole("button");
    fireEvent.click(button);

    // Should immediately update (optimistic)
    expect(screen.getByText("43")).toBeInTheDocument();
  });

  it("reverts on API error", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: false, error: "Failed" }),
    });

    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
      />
    );

    const button = screen.getByRole("button");
    fireEvent.click(button);

    // Wait for API response and revert
    await waitFor(() => {
      expect(screen.getByText("42")).toBeInTheDocument();
    });
  });

  it("calls onToggle callback with correct values", async () => {
    const onToggle = vi.fn();
    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: true, starred: true, count: 43 }),
    });

    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
        onToggle={onToggle}
      />
    );

    const button = screen.getByRole("button");
    fireEvent.click(button);

    await waitFor(() => {
      expect(onToggle).toHaveBeenCalledWith(true, 43);
    });
  });

  it("disables button while loading", async () => {
    global.fetch = vi.fn().mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 100))
    );

    render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
      />
    );

    const button = screen.getByRole("button");
    fireEvent.click(button);

    expect(button).toBeDisabled();
  });

  it("applies size classes with 44px minimum touch targets", () => {
    const { rerender } = render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
        size="sm"
      />
    );

    let button = screen.getByRole("button");
    // Small size now has 44px minimum height for accessibility
    expect(button).toHaveClass("px-3");
    expect(button).toHaveClass("min-h-[44px]");

    rerender(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
        size="lg"
      />
    );

    button = screen.getByRole("button");
    // Large size has 48px minimum height
    expect(button).toHaveClass("px-5");
    expect(button).toHaveClass("min-h-[48px]");
  });
});

describe("StarIconButton", () => {
  it("renders without count", () => {
    render(
      <StarIconButton
        labId="lab-1"
        initialStarred={false}
      />
    );

    // Should only have the button, no count text
    expect(screen.getByRole("button")).toBeInTheDocument();
    expect(screen.queryByText(/\d+/)).not.toBeInTheDocument();
  });

  it("toggles star state", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: true, starred: true, count: 1 }),
    });

    render(
      <StarIconButton
        labId="lab-1"
        initialStarred={false}
      />
    );

    const button = screen.getByRole("button");
    fireEvent.click(button);

    await waitFor(() => {
      expect(button).toHaveClass("text-yellow-400");
    });
  });

  it("applies correct size classes with 44px minimum touch targets", () => {
    const { rerender } = render(
      <StarIconButton
        labId="lab-1"
        initialStarred={false}
        size="sm"
      />
    );

    let button = screen.getByRole("button");
    // All sizes now have 44px minimum for accessibility
    expect(button).toHaveClass("min-w-[44px]");
    expect(button).toHaveClass("min-h-[44px]");

    rerender(
      <StarIconButton
        labId="lab-1"
        initialStarred={false}
        size="lg"
      />
    );

    button = screen.getByRole("button");
    // Large size has 48px minimum
    expect(button).toHaveClass("min-w-[48px]");
    expect(button).toHaveClass("min-h-[48px]");
  });
});
