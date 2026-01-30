/**
 * Tests for EvolutionViewer and GenerationCompare Components
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { EvolutionViewer, GenerationCompare } from "@/components/meta/EvolutionViewer";

const mockGenerationHistory = [
  {
    generation: 0,
    bestFitness: 0.5,
    averageFitness: 0.4,
    worstFitness: 0.2,
    diversity: 0.1,
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
    diversity: 0.08,
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
    diversity: 0.06,
    mutationCount: 8,
    crossoverCount: 22,
    eliteCount: 5,
    timestamp: new Date(),
  },
];

const mockTopChromosomes = [
  { id: "chr-1", name: "Elite-1", fitness: 0.7, isElite: true },
  { id: "chr-2", name: "Elite-2", fitness: 0.68, isElite: true },
  { id: "chr-3", name: "Chromosome-3", fitness: 0.65, isElite: false },
];

describe("EvolutionViewer", () => {
  it("should render population name", () => {
    render(
      <EvolutionViewer
        populationName="Test Population"
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
      />
    );

    expect(screen.getByText("Test Population")).toBeInTheDocument();
  });

  it("should display current generation", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
      />
    );

    expect(screen.getByText(/Generation 2/)).toBeInTheDocument();
  });

  it("should show status badge", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
      />
    );

    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("should show pause button when running", () => {
    const onPause = vi.fn();
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
        onPause={onPause}
      />
    );

    const pauseButton = screen.getByText("Pause");
    expect(pauseButton).toBeInTheDocument();
  });

  it("should call onPause when pause clicked", () => {
    const onPause = vi.fn();
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
        onPause={onPause}
      />
    );

    fireEvent.click(screen.getByText("Pause"));
    expect(onPause).toHaveBeenCalled();
  });

  it("should show resume button when paused", () => {
    const onResume = vi.fn();
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="paused"
        onResume={onResume}
      />
    );

    expect(screen.getByText("Resume")).toBeInTheDocument();
  });

  it("should call onResume when resume clicked", () => {
    const onResume = vi.fn();
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="paused"
        onResume={onResume}
      />
    );

    fireEvent.click(screen.getByText("Resume"));
    expect(onResume).toHaveBeenCalled();
  });

  it("should show stop button when running or paused", () => {
    const onStop = vi.fn();
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
        onStop={onStop}
      />
    );

    expect(screen.getByText("Stop")).toBeInTheDocument();
  });

  it("should display best fitness", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
      />
    );

    expect(screen.getByText("Best Fitness")).toBeInTheDocument();
    expect(screen.getByText("70.0%")).toBeInTheDocument();
  });

  it("should display top chromosomes", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
        topChromosomes={mockTopChromosomes}
      />
    );

    expect(screen.getByText("Elite-1")).toBeInTheDocument();
    expect(screen.getByText("Elite-2")).toBeInTheDocument();
  });

  it("should have metric selector buttons", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
      />
    );

    expect(screen.getByRole("button", { name: "Fitness" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Diversity" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Operators" })).toBeInTheDocument();
  });

  it("should switch metrics when button clicked", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="running"
      />
    );

    const diversityButton = screen.getByRole("button", { name: "Diversity" });
    fireEvent.click(diversityButton);
    // Verify button becomes selected (has blue background class)
    expect(diversityButton).toHaveClass("bg-blue-100");
  });

  it("should show completed status", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="completed"
      />
    );

    expect(screen.getByText("Completed")).toBeInTheDocument();
  });

  it("should show converged status", () => {
    render(
      <EvolutionViewer
        generationHistory={mockGenerationHistory}
        currentGeneration={2}
        status="converged"
      />
    );

    expect(screen.getByText("Converged")).toBeInTheDocument();
  });

  it("should handle empty generation history", () => {
    render(
      <EvolutionViewer
        generationHistory={[]}
        currentGeneration={0}
        status="idle"
      />
    );

    expect(screen.getByText("No data yet")).toBeInTheDocument();
  });
});

describe("GenerationCompare", () => {
  it("should render both generations", () => {
    render(
      <GenerationCompare
        generation1={mockGenerationHistory[0]}
        generation2={mockGenerationHistory[2]}
      />
    );

    expect(screen.getByText(/Generation 0/)).toBeInTheDocument();
    expect(screen.getByText(/Generation 2/)).toBeInTheDocument();
  });

  it("should show fitness comparison", () => {
    render(
      <GenerationCompare
        generation1={mockGenerationHistory[0]}
        generation2={mockGenerationHistory[2]}
      />
    );

    // Both generations' best fitness
    expect(screen.getByText("50.0%")).toBeInTheDocument();
    expect(screen.getByText("70.0%")).toBeInTheDocument();
  });

  it("should show improvement summary", () => {
    render(
      <GenerationCompare
        generation1={mockGenerationHistory[0]}
        generation2={mockGenerationHistory[2]}
      />
    );

    expect(screen.getByText("Improvement:")).toBeInTheDocument();
    // 0.7 - 0.5 = 0.2 = 20%
    expect(screen.getByText("+20.0%")).toBeInTheDocument();
  });

  it("should display diversity metrics", () => {
    render(
      <GenerationCompare
        generation1={mockGenerationHistory[0]}
        generation2={mockGenerationHistory[2]}
      />
    );

    expect(screen.getAllByText("Diversity").length).toBeGreaterThanOrEqual(1);
  });

  it("should display mutation counts", () => {
    render(
      <GenerationCompare
        generation1={mockGenerationHistory[0]}
        generation2={mockGenerationHistory[2]}
      />
    );

    expect(screen.getAllByText("Mutations").length).toBeGreaterThanOrEqual(1);
  });

  it("should display crossover counts", () => {
    render(
      <GenerationCompare
        generation1={mockGenerationHistory[0]}
        generation2={mockGenerationHistory[2]}
      />
    );

    expect(screen.getAllByText("Crossovers").length).toBeGreaterThanOrEqual(1);
  });
});
