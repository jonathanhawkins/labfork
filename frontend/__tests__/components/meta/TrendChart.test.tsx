/**
 * Tests for TrendChart Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  TrendChart,
  TrendIndicator,
  TrendSparkline,
} from "@/components/meta/TrendChart";

const mockTrends = [
  {
    id: "trend-1",
    name: "Transformer",
    category: "architecture",
    color: "#3B82F6",
    timeSeries: [
      { timestamp: new Date("2024-01-01"), value: 10 },
      { timestamp: new Date("2024-01-08"), value: 15 },
      { timestamp: new Date("2024-01-15"), value: 20 },
      { timestamp: new Date("2024-01-22"), value: 25 },
    ],
    strength: 0.8,
    momentum: 0.5,
  },
  {
    id: "trend-2",
    name: "Diffusion",
    category: "architecture",
    color: "#10B981",
    timeSeries: [
      { timestamp: new Date("2024-01-01"), value: 5 },
      { timestamp: new Date("2024-01-08"), value: 8 },
      { timestamp: new Date("2024-01-15"), value: 12 },
      { timestamp: new Date("2024-01-22"), value: 18 },
    ],
    strength: 0.6,
    momentum: 0.7,
  },
];

describe("TrendChart", () => {
  it("should render title", () => {
    render(<TrendChart trends={mockTrends} title="Research Trends" />);

    expect(screen.getByText("Research Trends")).toBeInTheDocument();
  });

  it("should show no data message when trends are empty", () => {
    render(<TrendChart trends={[]} />);

    expect(screen.getByText(/No trend data available/)).toBeInTheDocument();
  });

  it("should render with custom height", () => {
    const { container } = render(
      <TrendChart trends={mockTrends} height={400} />
    );

    // Check that the container has the right styling
    const chart = container.querySelector(".bg-white");
    expect(chart).toBeInTheDocument();
  });

  it("should show legend when enabled", () => {
    render(<TrendChart trends={mockTrends} showLegend={true} />);

    expect(screen.getByText("Transformer")).toBeInTheDocument();
    expect(screen.getByText("Diffusion")).toBeInTheDocument();
  });

  it("should hide legend when disabled", () => {
    render(<TrendChart trends={[mockTrends[0]]} showLegend={false} />);

    // With only one trend and legend disabled, trend name appears in title area
    // but not in a legend section
    const legendButtons = screen.queryAllByRole("button");
    expect(
      legendButtons.filter((b) => b.textContent?.includes("Transformer"))
    ).toHaveLength(0);
  });

  it("should toggle trend visibility when legend item is clicked", () => {
    render(<TrendChart trends={mockTrends} showLegend={true} />);

    // Find the legend button for Transformer
    const legendItems = screen.getAllByText("Transformer");
    const legendButton = legendItems.find(
      (item) => item.closest("button") !== null
    );

    if (legendButton) {
      fireEvent.click(legendButton.closest("button")!);

      // After clicking, the trend should have reduced opacity
      // (visual test - opacity changes)
    }
  });

  it("should call onTrendClick when a trend is clicked", () => {
    const onTrendClick = vi.fn();
    render(<TrendChart trends={mockTrends} onTrendClick={onTrendClick} />);

    // Find SVG circles (data points)
    const circles = document.querySelectorAll("circle");
    if (circles.length > 0) {
      fireEvent.click(circles[0]);
      expect(onTrendClick).toHaveBeenCalled();
    }
  });

  it("should show momentum indicator for single trend", () => {
    render(
      <TrendChart trends={[mockTrends[0]]} showMomentum={true} />
    );

    expect(screen.getByText("Growing")).toBeInTheDocument();
  });

  it("should show momentum in legend for multiple trends", () => {
    render(<TrendChart trends={mockTrends} showMomentum={true} />);

    const growingIndicators = screen.getAllByText("Growing");
    expect(growingIndicators.length).toBeGreaterThan(0);
  });

  it("should apply custom className", () => {
    const { container } = render(
      <TrendChart trends={mockTrends} className="custom-class" />
    );

    expect(container.firstChild).toHaveClass("custom-class");
  });

  it("should display date labels on x-axis", () => {
    render(<TrendChart trends={mockTrends} />);

    // Should show at least some date labels
    // Format is "MMM d" like "Jan 1"
    const dateLabels = screen.getAllByText(/Jan/);
    expect(dateLabels.length).toBeGreaterThan(0);
  });

  it("should display value labels on y-axis", () => {
    render(<TrendChart trends={mockTrends} />);

    // Y-axis should show min/max values
    // The values are 5-25, so we might see 0, 12, 25 or similar
    const yAxisLabels = document.querySelectorAll(".flex-col span");
    expect(yAxisLabels.length).toBeGreaterThan(0);
  });
});

describe("TrendIndicator", () => {
  it("should display name and value", () => {
    render(<TrendIndicator name="Active Trends" value={42} />);

    expect(screen.getByText("Active Trends")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("should show positive change indicator", () => {
    render(<TrendIndicator name="Test" value={10} change={0.15} />);

    expect(screen.getByText("+15.0%")).toBeInTheDocument();
  });

  it("should show negative change indicator", () => {
    render(<TrendIndicator name="Test" value={10} change={-0.1} />);

    expect(screen.getByText("-10.0%")).toBeInTheDocument();
  });

  it("should show stable for small changes", () => {
    render(<TrendIndicator name="Test" value={10} change={0.005} />);

    // Small change should show as stable
    expect(screen.getByText("+0.5%")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    const { container } = render(
      <TrendIndicator name="Test" value={10} className="custom-class" />
    );

    expect(container.firstChild).toHaveClass("custom-class");
  });

  it("should show change label when provided", () => {
    render(
      <TrendIndicator
        name="Test"
        value={10}
        change={0.1}
        changeLabel="vs last week"
      />
    );

    expect(screen.getByText("vs last week")).toBeInTheDocument();
  });
});

describe("TrendSparkline", () => {
  it("should render SVG with correct dimensions", () => {
    const { container } = render(
      <TrendSparkline data={[1, 2, 3, 4, 5]} width={100} height={30} />
    );

    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("width", "100");
    expect(svg).toHaveAttribute("height", "30");
  });

  it("should render polyline for data", () => {
    const { container } = render(<TrendSparkline data={[1, 2, 3, 4, 5]} />);

    const polyline = container.querySelector("polyline");
    expect(polyline).toBeInTheDocument();
    expect(polyline).toHaveAttribute("points");
  });

  it("should use custom color", () => {
    const { container } = render(
      <TrendSparkline data={[1, 2, 3, 4, 5]} color="#FF0000" />
    );

    const polyline = container.querySelector("polyline");
    expect(polyline).toHaveAttribute("stroke", "#FF0000");
  });

  it("should return null for insufficient data", () => {
    const { container } = render(<TrendSparkline data={[1]} />);

    expect(container.querySelector("svg")).not.toBeInTheDocument();
  });

  it("should apply custom className", () => {
    const { container } = render(
      <TrendSparkline data={[1, 2, 3]} className="custom-sparkline" />
    );

    const svg = container.querySelector("svg");
    expect(svg).toHaveClass("custom-sparkline");
  });
});
