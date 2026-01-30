/**
 * Tests for LineageTree and LineageTimeline Components
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { LineageTree, LineageTimeline } from "@/components/meta/LineageTree";

const mockRoot = {
  id: "chr-root",
  name: "Root Chromosome",
  generation: 0,
  fitness: 0.5,
  isAlive: false,
  mutations: [],
  children: [
    {
      id: "chr-child-1",
      name: "Child 1",
      generation: 1,
      fitness: 0.6,
      isAlive: true,
      mutations: [{ geneId: "gene-1", type: "gaussian", fitnessImpact: 0.1 }],
      children: [
        {
          id: "chr-grandchild-1",
          name: "Grandchild 1",
          generation: 2,
          fitness: 0.7,
          isAlive: true,
          mutations: [],
          children: [],
        },
      ],
    },
    {
      id: "chr-child-2",
      name: "Child 2",
      generation: 1,
      fitness: 0.55,
      isAlive: false,
      mutations: [{ geneId: "gene-2", type: "point", fitnessImpact: 0.05 }],
      children: [],
    },
  ],
};

const mockFitnessTrajectory = [
  { generation: 0, fitness: 0.5 },
  { generation: 1, fitness: 0.6 },
  { generation: 2, fitness: 0.65 },
  { generation: 3, fitness: 0.7 },
  { generation: 4, fitness: 0.72 },
];

const mockKeyMutations = [
  { generation: 1, fitnessImprovement: 0.1, description: "Increased attention heads" },
  { generation: 3, fitnessImprovement: 0.05, description: "Added layer norm" },
];

describe("LineageTree", () => {
  it("should render tree header", () => {
    render(<LineageTree root={mockRoot} />);

    expect(screen.getByText("Lineage Tree")).toBeInTheDocument();
  });

  it("should show root name in header", () => {
    render(<LineageTree root={mockRoot} />);

    expect(screen.getByText(/Root: Root Chromosome/)).toBeInTheDocument();
  });

  it("should display tree depth", () => {
    render(<LineageTree root={mockRoot} />);

    // Root -> Child -> Grandchild = depth 2
    expect(screen.getByText(/Depth: 2/)).toBeInTheDocument();
  });

  it("should display total nodes", () => {
    render(<LineageTree root={mockRoot} />);

    // 1 root + 2 children + 1 grandchild = 4 nodes
    expect(screen.getByText(/Nodes: 4/)).toBeInTheDocument();
  });

  it("should have zoom controls", () => {
    render(<LineageTree root={mockRoot} />);

    expect(screen.getByText("-")).toBeInTheDocument();
    expect(screen.getByText("+")).toBeInTheDocument();
    expect(screen.getByText("Reset")).toBeInTheDocument();
  });

  it("should show initial zoom level", () => {
    render(<LineageTree root={mockRoot} />);

    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("should increase zoom when + clicked", () => {
    render(<LineageTree root={mockRoot} />);

    const zoomInButton = screen.getByText("+");
    fireEvent.click(zoomInButton);

    expect(screen.getByText("125%")).toBeInTheDocument();
  });

  it("should decrease zoom when - clicked", () => {
    render(<LineageTree root={mockRoot} />);

    const zoomOutButton = screen.getByText("-");
    fireEvent.click(zoomOutButton);

    expect(screen.getByText("75%")).toBeInTheDocument();
  });

  it("should reset zoom when Reset clicked", () => {
    render(<LineageTree root={mockRoot} />);

    // First zoom in
    fireEvent.click(screen.getByText("+"));
    expect(screen.getByText("125%")).toBeInTheDocument();

    // Then reset
    fireEvent.click(screen.getByText("Reset"));
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("should call onNodeClick when node clicked", () => {
    const onNodeClick = vi.fn();
    render(<LineageTree root={mockRoot} onNodeClick={onNodeClick} />);

    // Click on the root node (first node in the SVG)
    const nodeElements = document.querySelectorAll("g.cursor-pointer");
    if (nodeElements.length > 0) {
      fireEvent.click(nodeElements[0]);
      expect(onNodeClick).toHaveBeenCalled();
    }
  });

  it("should render SVG element", () => {
    render(<LineageTree root={mockRoot} />);

    const svg = document.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("should render edges between nodes", () => {
    render(<LineageTree root={mockRoot} />);

    const paths = document.querySelectorAll("path");
    expect(paths.length).toBeGreaterThan(0);
  });

  it("should apply className prop", () => {
    const { container } = render(<LineageTree root={mockRoot} className="test-class" />);

    expect(container.firstChild).toHaveClass("test-class");
  });

  it("should handle node hover", () => {
    const onNodeHover = vi.fn();
    render(<LineageTree root={mockRoot} onNodeHover={onNodeHover} />);

    const nodeElements = document.querySelectorAll("g.cursor-pointer");
    if (nodeElements.length > 0) {
      fireEvent.mouseEnter(nodeElements[0]);
      expect(onNodeHover).toHaveBeenCalled();
    }
  });

  it("should display node details when selected", () => {
    render(<LineageTree root={mockRoot} />);

    const nodeElements = document.querySelectorAll("g.cursor-pointer");
    if (nodeElements.length > 0) {
      fireEvent.click(nodeElements[0]);
      // Node details panel should appear with the node name
      expect(screen.getByText("Root Chromosome")).toBeInTheDocument();
    }
  });
});

describe("LineageTimeline", () => {
  it("should render timeline header", () => {
    render(<LineageTimeline fitnessTrajectory={mockFitnessTrajectory} />);

    expect(screen.getByText("Fitness Trajectory")).toBeInTheDocument();
  });

  it("should render SVG element", () => {
    render(<LineageTimeline fitnessTrajectory={mockFitnessTrajectory} />);

    const svg = document.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("should handle empty trajectory", () => {
    render(<LineageTimeline fitnessTrajectory={[]} />);

    expect(screen.getByText("No trajectory data")).toBeInTheDocument();
  });

  it("should render path for fitness line", () => {
    render(<LineageTimeline fitnessTrajectory={mockFitnessTrajectory} />);

    // Component uses <path> for the fitness line
    const path = document.querySelector("path");
    expect(path).toBeInTheDocument();
  });

  it("should render key mutations as markers", () => {
    render(
      <LineageTimeline
        fitnessTrajectory={mockFitnessTrajectory}
        keyMutations={mockKeyMutations}
      />
    );

    // Key mutations should be rendered as circles
    const circles = document.querySelectorAll("circle");
    expect(circles.length).toBeGreaterThan(0);
  });

  it("should apply className prop", () => {
    const { container } = render(
      <LineageTimeline
        fitnessTrajectory={mockFitnessTrajectory}
        className="custom-timeline"
      />
    );

    expect(container.firstChild).toHaveClass("custom-timeline");
  });

  it("should display generation axis label", () => {
    render(<LineageTimeline fitnessTrajectory={mockFitnessTrajectory} />);

    // Should have "Generation" axis label
    expect(screen.getByText("Generation")).toBeInTheDocument();
  });
});
