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

  it("applies size classes correctly", () => {
    const { rerender } = render(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
        size="sm"
      />
    );

    let button = screen.getByRole("button");
    expect(button).toHaveClass("px-2");
    expect(button).toHaveClass("py-1");

    rerender(
      <StarButton
        labId="lab-1"
        initialCount={42}
        initialStarred={false}
        size="lg"
      />
    );

    button = screen.getByRole("button");
    expect(button).toHaveClass("px-4");
    expect(button).toHaveClass("py-2");
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

  it("applies correct size classes", () => {
    const { rerender } = render(
      <StarIconButton
        labId="lab-1"
        initialStarred={false}
        size="sm"
      />
    );

    let button = screen.getByRole("button");
    expect(button).toHaveClass("w-7");
    expect(button).toHaveClass("h-7");

    rerender(
      <StarIconButton
        labId="lab-1"
        initialStarred={false}
        size="lg"
      />
    );

    button = screen.getByRole("button");
    expect(button).toHaveClass("w-11");
    expect(button).toHaveClass("h-11");
  });
});
