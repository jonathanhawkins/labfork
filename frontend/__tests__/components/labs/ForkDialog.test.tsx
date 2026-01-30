import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ForkDialog } from "@/components/labs/ForkDialog";
import type { Lab } from "@/lib/labs/types";

const mockLab: Lab = {
  id: "lab-1",
  name: "Original Lab",
  slug: "original-lab",
  description: "Original description",
  visibility: "public",
  status: "active",
  domainSlug: "voice-clone",
  domainName: "Voice Clone",
  owner: {
    id: "user-1",
    username: "originaluser",
    displayName: "Original User",
    avatarUrl: null,
  },
  stats: {
    stars: 10,
    forks: 5,
    tasks: 3,
    papers: 1,
    experiments: 2,
    viewers: 0,
  },
  tags: ["test"],
  createdAt: "2024-01-01T00:00:00Z",
  updatedAt: "2024-01-01T00:00:00Z",
  lastActivityAt: "2024-01-01T00:00:00Z",
  isFeatured: false,
  forkedFrom: null,
};

describe("ForkDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does not render when closed", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={false}
        onClose={vi.fn()}
      />
    );

    expect(screen.queryByText("Fork Lab")).not.toBeInTheDocument();
  });

  it("renders when open", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    // Use role to get the heading specifically
    expect(screen.getByRole("heading", { name: "Fork Lab" })).toBeInTheDocument();
  });

  it("shows source lab info", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByText("Forking from")).toBeInTheDocument();
    expect(screen.getByText("originaluser/original-lab")).toBeInTheDocument();
  });

  it("shows default fork slug from name", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    // Slug is auto-generated from name by slugify()
    const slugInput = screen.getByPlaceholderText("my-lab");
    expect(slugInput).toHaveValue("original-lab");
  });

  it("auto-generates slug from name", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    const nameInput = screen.getByPlaceholderText("My Lab");
    fireEvent.change(nameInput, { target: { value: "My Custom Fork" } });

    const slugInput = screen.getByPlaceholderText("my-lab");
    expect(slugInput).toHaveValue("my-custom-fork");
  });

  it("allows manual slug editing", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    const slugInput = screen.getByPlaceholderText("my-lab");
    fireEvent.change(slugInput, { target: { value: "custom-slug" } });

    expect(slugInput).toHaveValue("custom-slug");

    // Name change should not update slug after manual edit
    const nameInput = screen.getByPlaceholderText("My Lab");
    fireEvent.change(nameInput, { target: { value: "New Name" } });

    expect(slugInput).toHaveValue("custom-slug");
  });

  it("shows what gets copied", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByText("What gets copied:")).toBeInTheDocument();
    expect(screen.getByText("Config")).toBeInTheDocument();
    expect(screen.getByText("Prompts")).toBeInTheDocument();
    expect(screen.getByText("Tasks")).toBeInTheDocument();
  });

  it("calls onClose when cancel clicked", () => {
    const onClose = vi.fn();
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={onClose}
      />
    );

    fireEvent.click(screen.getByText("Cancel"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("calls onClose when backdrop clicked", () => {
    const onClose = vi.fn();
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={onClose}
      />
    );

    // Click the backdrop (the blurred overlay)
    const backdrop = document.querySelector(".backdrop-blur-sm");
    if (backdrop) {
      fireEvent.click(backdrop);
    }
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("disables fork button when name is empty", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    const nameInput = screen.getByPlaceholderText("My Lab");
    fireEvent.change(nameInput, { target: { value: "" } });

    const forkButton = screen.getByRole("button", { name: /fork lab/i });
    expect(forkButton).toBeDisabled();
  });

  it("disables fork button when slug is empty", () => {
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    const slugInput = screen.getByPlaceholderText("my-lab");
    fireEvent.change(slugInput, { target: { value: "" } });

    const forkButton = screen.getByRole("button", { name: /fork lab/i });
    expect(forkButton).toBeDisabled();
  });

  it("shows loading state while forking", async () => {
    global.fetch = vi.fn().mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 100))
    );

    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    const forkButton = screen.getByRole("button", { name: /fork lab/i });
    fireEvent.click(forkButton);

    expect(screen.getByText("Forking...")).toBeInTheDocument();
    expect(forkButton).toBeDisabled();
  });

  it("shows success state after forking", async () => {
    const forkedLab: Lab = {
      ...mockLab,
      id: "forked-1",
      slug: "original-lab-fork",
      owner: {
        id: "user-2",
        username: "newuser",
        displayName: "New User",
        avatarUrl: null,
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: true, lab: forkedLab }),
    });

    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    const forkButton = screen.getByRole("button", { name: /fork lab/i });
    fireEvent.click(forkButton);

    await waitFor(() => {
      expect(screen.getByText("Lab forked successfully!")).toBeInTheDocument();
    });

    expect(screen.getByText("Go to Fork")).toBeInTheDocument();
  });

  it("calls onSuccess callback after successful fork", async () => {
    const forkedLab: Lab = {
      ...mockLab,
      id: "forked-1",
    };

    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: true, lab: forkedLab }),
    });

    const onSuccess = vi.fn();
    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
        onSuccess={onSuccess}
      />
    );

    const forkButton = screen.getByRole("button", { name: /fork lab/i });
    fireEvent.click(forkButton);

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith(forkedLab);
    });
  });

  it("shows error message on fork failure", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ success: false, error: "Slug already exists" }),
    });

    render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    const forkButton = screen.getByRole("button", { name: /fork lab/i });
    fireEvent.click(forkButton);

    await waitFor(() => {
      expect(screen.getByText("Slug already exists")).toBeInTheDocument();
    });
  });

  it("resets state when dialog reopens", async () => {
    const { rerender } = render(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    // Change the name
    const nameInput = screen.getByPlaceholderText("My Lab");
    fireEvent.change(nameInput, { target: { value: "Custom Name" } });

    // Close dialog
    rerender(
      <ForkDialog
        lab={mockLab}
        isOpen={false}
        onClose={vi.fn()}
      />
    );

    // Reopen dialog
    rerender(
      <ForkDialog
        lab={mockLab}
        isOpen={true}
        onClose={vi.fn()}
      />
    );

    // Name should be reset
    await waitFor(() => {
      const newNameInput = screen.getByPlaceholderText("My Lab");
      expect(newNameInput).toHaveValue("Original Lab");
    });
  });
});
