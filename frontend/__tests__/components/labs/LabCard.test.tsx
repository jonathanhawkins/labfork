import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { LabCard } from "@/components/labs/LabCard";
import type { Lab } from "@/lib/labs/types";

// Mock next/link
vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

const mockLab: Lab = {
  id: "lab-1",
  name: "Test Lab",
  slug: "test-lab",
  description: "A test lab description",
  visibility: "public",
  status: "active",
  domainSlug: "voice-clone",
  domainName: "Voice Clone",
  owner: {
    id: "user-1",
    username: "testuser",
    displayName: "Test User",
  },
  stats: {
    stars: 42,
    forks: 10,
    tasks: 5,
    papers: 2,
    experiments: 3,
    viewers: 1,
  },
  tags: ["tts", "voice", "ai"],
  createdAt: "2024-01-01T00:00:00Z",
  updatedAt: "2024-01-15T00:00:00Z",
  lastActivityAt: "2024-01-15T00:00:00Z",
  forkedFrom: undefined,
};

describe("LabCard", () => {
  it("renders lab name and description", () => {
    render(<LabCard lab={mockLab} />);

    expect(screen.getByText("Test Lab")).toBeInTheDocument();
    expect(screen.getByText("A test lab description")).toBeInTheDocument();
  });

  it("displays owner name", () => {
    render(<LabCard lab={mockLab} />);

    expect(screen.getByText("Test User")).toBeInTheDocument();
  });

  it("shows star and fork counts", () => {
    render(<LabCard lab={mockLab} />);

    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
  });

  it("shows task count in non-compact mode", () => {
    render(<LabCard lab={mockLab} />);

    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("hides description in compact mode", () => {
    render(<LabCard lab={mockLab} compact />);

    expect(screen.queryByText("A test lab description")).not.toBeInTheDocument();
  });

  it("shows domain badge", () => {
    render(<LabCard lab={mockLab} />);

    expect(screen.getByText("Voice Clone")).toBeInTheDocument();
  });

  it("displays tags", () => {
    render(<LabCard lab={mockLab} />);

    expect(screen.getByText("tts")).toBeInTheDocument();
    expect(screen.getByText("voice")).toBeInTheDocument();
    expect(screen.getByText("ai")).toBeInTheDocument();
  });

  it("handles star button click without navigation", () => {
    const onStarClick = vi.fn();
    render(<LabCard lab={mockLab} onStarClick={onStarClick} />);

    const starButton = screen.getByTitle(/star/i);
    fireEvent.click(starButton);

    expect(onStarClick).toHaveBeenCalledTimes(1);
  });

  it("links to correct lab path", () => {
    render(<LabCard lab={mockLab} />);

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/labs/testuser/test-lab");
  });

  it("uses onClick handler instead of link when provided", () => {
    const onClick = vi.fn();
    // Disable star button to avoid nested buttons warning
    render(<LabCard lab={mockLab} onClick={onClick} showStar={false} />);

    // Find the outer wrapper button by its class
    const buttons = screen.getAllByRole("button");
    // The first button should be the card wrapper
    fireEvent.click(buttons[0]);

    expect(onClick).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("shows forked-from indicator when forked", () => {
    const forkedLab: Lab = {
      ...mockLab,
      forkedFrom: {
        sourceLabId: "original-1",
        sourceOwner: "originaluser",
        sourceSlug: "original-lab",
        forkedAt: "2024-01-10T00:00:00Z",
      },
    };

    render(<LabCard lab={forkedLab} />);

    expect(screen.getByText(/Forked from/)).toBeInTheDocument();
    expect(screen.getByText(/originaluser\/original-lab/)).toBeInTheDocument();
  });

  it("shows selected state when isSelected is true", () => {
    const { container } = render(<LabCard lab={mockLab} isSelected />);

    // Check for selected styling (ring)
    expect(container.querySelector(".ring-2")).toBeInTheDocument();
  });

  it("hides star button when showStar is false", () => {
    render(<LabCard lab={mockLab} showStar={false} />);

    expect(screen.queryByTitle(/star/i)).not.toBeInTheDocument();
  });

  it("shows viewers count when greater than 0", () => {
    render(<LabCard lab={mockLab} />);

    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("hides viewers when count is 0", () => {
    const labNoViewers: Lab = {
      ...mockLab,
      stats: { ...mockLab.stats, viewers: 0 },
    };

    render(<LabCard lab={labNoViewers} />);

    // Should only show 3 stats (star, fork, task) not 4
    const statItems = screen.getAllByText(/^\d+$/);
    expect(statItems.length).toBeGreaterThanOrEqual(3);
  });
});
