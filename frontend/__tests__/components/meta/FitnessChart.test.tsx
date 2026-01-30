/**
 * Tests for FitnessChart Component
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  FitnessChart,
  FitnessRadar,
  PopulationStats,
  FitnessSparkline,
} from "@/components/meta/FitnessChart";
import { GenerationStats, FitnessComponents, Population } from "@/lib/meta/evolution";

const mockGenerationHistory: GenerationStats[] = [
  {
    generation: 0,
    bestFitness: 0.5,
    averageFitness: 0.4,
    worstFitness: 0.2,
    fitnessStdDev: 0.1,
    mutationCount: 0,
    crossoverCount: 0,
    eliteCount: 5,
    timestamp: new Date(),
  },
  {
    generation: 1,
    bestFitness: 0.6,
    averageFitness: 0.5,
    worstFitness: 0.3,
    fitnessStdDev: 0.08,
    mutationCount: 10,
    crossoverCount: 20,
    eliteCount: 5,
    timestamp: new Date(),
  },
  {
    generation: 2,
    bestFitness: 0.7,
    averageFitness: 0.55,
    worstFitness: 0.35,
    fitnessStdDev: 0.06,
    mutationCount: 8,
    crossoverCount: 22,
    eliteCount: 5,
    timestamp: new Date(),
  },
];

const mockFitnessComponents: FitnessComponents = {
  quality: 0.8,
  efficiency: 0.7,
  novelty: 0.6,
  feasibility: 0.75,
  compatibility: 0.65,
};

const mockPopulation: Population = {
  id: "pop-1",
  name: "Test Population",
  generation: 2,
  chromosomes: [],
  size: 50,
  bestChromosomeId: "chr-1",
  averageFitness: 0.55,
  fitnessDiversity: 0.06,
  generationHistory: mockGenerationHistory,
  config: {} as any,
  createdAt: new Date(),
  lastEvolved: new Date(),
  status: "evolving",
};

describe("FitnessChart", () => {
  it("should render title", () => {
    render(
      <FitnessChart
        generationHistory={mockGenerationHistory}
        title="Evolution Progress"
      />
    );

    expect(screen.getByText("Evolution Progress")).toBeInTheDocument();
  });

  it("should show no data message when history is empty", () => {
    render(<FitnessChart generationHistory={[]} />);

    expect(screen.getByText("No generation data available")).toBeInTheDocument();
  });

  it("should display best fitness", () => {
    render(<FitnessChart generationHistory={mockGenerationHistory} />);

    expect(screen.getByText(/Best:/)).toBeInTheDocument();
    // Multiple elements may contain 70.0% (legend and chart tooltips)
    const bestFitnessElements = screen.getAllByText(/70\.0%/);
    expect(bestFitnessElements.length).toBeGreaterThan(0);
  });

  it("should display average fitness when enabled", () => {
    render(
      <FitnessChart
        generationHistory={mockGenerationHistory}
        showAverage={true}
      />
    );

    expect(screen.getByText(/Avg:/)).toBeInTheDocument();
    // The average fitness from the last generation is displayed
    expect(screen.getByText(/55\.0%/)).toBeInTheDocument();
  });

  it("should display diversity when enabled", () => {
    render(
      <FitnessChart
        generationHistory={mockGenerationHistory}
        showDiversity={true}
      />
    );

    expect(screen.getByText(/Diversity:/)).toBeInTheDocument();
  });

  it("should show generation count", () => {
    render(<FitnessChart generationHistory={mockGenerationHistory} />);

    expect(screen.getByText("Generations")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("should show total mutations", () => {
    render(<FitnessChart generationHistory={mockGenerationHistory} />);

    expect(screen.getByText("Mutations")).toBeInTheDocument();
    expect(screen.getByText("18")).toBeInTheDocument();
  });

  it("should show total crossovers", () => {
    render(<FitnessChart generationHistory={mockGenerationHistory} />);

    expect(screen.getByText("Crossovers")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("should calculate improvement", () => {
    render(<FitnessChart generationHistory={mockGenerationHistory} />);

    expect(screen.getByText("Improvement")).toBeInTheDocument();
    expect(screen.getByText("+20.0%")).toBeInTheDocument();
  });

  it("should render SVG chart", () => {
    const { container } = render(
      <FitnessChart generationHistory={mockGenerationHistory} />
    );

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    const { container } = render(
      <FitnessChart
        generationHistory={mockGenerationHistory}
        className="custom-class"
      />
    );

    expect(container.firstChild).toHaveClass("custom-class");
  });
});

describe("FitnessRadar", () => {
  it("should render radar chart", () => {
    const { container } = render(<FitnessRadar components={mockFitnessComponents} />);

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("should show all component labels", () => {
    render(<FitnessRadar components={mockFitnessComponents} />);

    expect(screen.getByText("Quality")).toBeInTheDocument();
    expect(screen.getByText("Efficiency")).toBeInTheDocument();
    expect(screen.getByText("Novelty")).toBeInTheDocument();
    expect(screen.getByText("Feasibility")).toBeInTheDocument();
    expect(screen.getByText("Compatibility")).toBeInTheDocument();
  });

  it("should display component values", () => {
    render(<FitnessRadar components={mockFitnessComponents} />);

    expect(screen.getByText("80%")).toBeInTheDocument();
    expect(screen.getByText("70%")).toBeInTheDocument();
    expect(screen.getByText("60%")).toBeInTheDocument();
  });

  it("should render comparison when provided", () => {
    const compareWith: FitnessComponents = {
      quality: 0.6,
      efficiency: 0.5,
      novelty: 0.4,
      feasibility: 0.55,
      compatibility: 0.45,
    };

    const { container } = render(
      <FitnessRadar components={mockFitnessComponents} compareWith={compareWith} />
    );

    // Should have multiple paths for comparison
    const paths = container.querySelectorAll("path");
    expect(paths.length).toBeGreaterThan(1);
  });
});

describe("PopulationStats", () => {
  it("should render population name", () => {
    render(<PopulationStats population={mockPopulation} />);

    expect(screen.getByText("Test Population")).toBeInTheDocument();
  });

  it("should show status badge", () => {
    render(<PopulationStats population={mockPopulation} />);

    expect(screen.getByText("evolving")).toBeInTheDocument();
  });

  it("should display generation number", () => {
    render(<PopulationStats population={mockPopulation} />);

    expect(screen.getByText("Generation")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("should display best fitness", () => {
    render(<PopulationStats population={mockPopulation} />);

    expect(screen.getByText("Best Fitness")).toBeInTheDocument();
    expect(screen.getByText("70.0%")).toBeInTheDocument();
  });

  it("should display average fitness", () => {
    render(<PopulationStats population={mockPopulation} />);

    expect(screen.getByText("Avg Fitness")).toBeInTheDocument();
    expect(screen.getByText("55.0%")).toBeInTheDocument();
  });

  it("should display population size", () => {
    render(<PopulationStats population={mockPopulation} />);

    expect(screen.getByText("Population")).toBeInTheDocument();
    expect(screen.getByText("50")).toBeInTheDocument();
  });

  it("should show created and evolved dates", () => {
    render(<PopulationStats population={mockPopulation} />);

    expect(screen.getByText(/Created:/)).toBeInTheDocument();
    expect(screen.getByText(/Last evolved:/)).toBeInTheDocument();
  });
});

describe("FitnessSparkline", () => {
  it("should render SVG", () => {
    const { container } = render(
      <FitnessSparkline data={[0.3, 0.5, 0.6, 0.7, 0.8]} />
    );

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("should return null for insufficient data", () => {
    const { container } = render(<FitnessSparkline data={[0.5]} />);

    expect(container.querySelector("svg")).not.toBeInTheDocument();
  });

  it("should use custom dimensions", () => {
    const { container } = render(
      <FitnessSparkline data={[0.3, 0.5, 0.7]} width={150} height={40} />
    );

    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("width", "150");
    expect(svg).toHaveAttribute("height", "40");
  });

  it("should use custom color", () => {
    const { container } = render(
      <FitnessSparkline data={[0.3, 0.5, 0.7]} color="#ff0000" />
    );

    const polyline = container.querySelector("polyline");
    expect(polyline).toHaveAttribute("stroke", "#ff0000");
  });

  it("should render polyline with points", () => {
    const { container } = render(
      <FitnessSparkline data={[0.3, 0.5, 0.7]} />
    );

    const polyline = container.querySelector("polyline");
    expect(polyline).toBeInTheDocument();
    expect(polyline).toHaveAttribute("points");
  });
});
