/**
 * Tests for PlatformMetrics Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PlatformMetrics } from "@/components/community/PlatformMetrics";
import type {
  PlatformMetrics as MetricsType,
  MetricsSummary,
} from "@/lib/meta/community/types";

describe("PlatformMetrics", () => {
  const mockMetrics: MetricsType = {
    id: "metrics-1",
    timestamp: "2025-01-27T10:00:00Z",
    period: "daily",
    labs: {
      total: 150,
      active: 120,
      new: 8,
      inactive: 30,
      avgTechniquesPerLab: 12.5,
      avgCollaborationsPerLab: 2.3,
      topDomains: [
        { domain: "TTS", count: 80, percentage: 53.3 },
        { domain: "Voice Cloning", count: 45, percentage: 30 },
        { domain: "Speech Recognition", count: 25, percentage: 16.7 },
      ],
    },
    papers: {
      total: 500,
      processed: 450,
      pending: 40,
      failed: 10,
      avgProcessingTime: 5000,
      topSources: [
        { source: "arxiv", count: 350, percentage: 70 },
        { source: "semanticscholar", count: 100, percentage: 20 },
        { source: "github", count: 50, percentage: 10 },
      ],
    },
    techniques: {
      total: 1200,
      active: 1000,
      deprecated: 100,
      experimental: 100,
      avgUsageCount: 25,
      topCategories: [
        { category: "prosody", count: 300, percentage: 25 },
        { category: "codec", count: 250, percentage: 20.8 },
        { category: "attention", count: 200, percentage: 16.7 },
      ],
      evolutionGenerations: 5,
    },
    tasks: {
      total: 800,
      completed: 650,
      inProgress: 80,
      pending: 50,
      failed: 20,
      avgCompletionTime: 3600000,
      completionRate: 97,
    },
    collaborations: {
      total: 120,
      active: 40,
      completed: 70,
      avgParticipants: 4.2,
      avgDuration: 25,
      successRate: 87.5,
      topCollaborators: [
        { labId: "lab-1", labName: "Voice Lab", count: 15 },
        { labId: "lab-2", labName: "Audio Team", count: 12 },
      ],
    },
    adoptionRates: {
      newTechniques: 15,
      synergyDiscoveries: 8,
      evolutionExperiments: 5,
      collaborationJoins: 25,
    },
    successRates: {
      taskCompletion: 97,
      collaborationSuccess: 87.5,
      gapResolution: 75,
      evolutionImprovement: 60,
    },
    growth: {
      labsGrowth: 8.5,
      papersGrowth: 12.3,
      techniquesGrowth: 5.2,
      collaborationsGrowth: 15.7,
      weekOverWeek: 10,
      monthOverMonth: 35,
    },
    costs: {
      totalApiCalls: 50000,
      apiCostEstimate: 50,
      computeHours: 120,
      storageGB: 45,
      avgCostPerPaper: 0.1,
      avgCostPerDiscovery: 0.5,
    },
  };

  const mockSummary: MetricsSummary = {
    highlights: [
      "Strong lab growth: 8.5% increase",
      "High task completion rate: 97%",
    ],
    concerns: ["Paper failure rate slightly elevated"],
    recommendations: ["Consider validating experimental techniques"],
  };

  describe("Header", () => {
    it("should render title", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Platform Metrics")).toBeInTheDocument();
    });

    it("should render period buttons", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Daily")).toBeInTheDocument();
      expect(screen.getByText("Weekly")).toBeInTheDocument();
      expect(screen.getByText("Monthly")).toBeInTheDocument();
      expect(screen.getByText("All time")).toBeInTheDocument();
    });

    it("should call onPeriodChange when period clicked", () => {
      const onPeriodChange = vi.fn();
      render(<PlatformMetrics metrics={mockMetrics} onPeriodChange={onPeriodChange} />);

      fireEvent.click(screen.getByText("Weekly"));

      expect(onPeriodChange).toHaveBeenCalledWith("weekly");
    });
  });

  describe("Summary Section", () => {
    it("should render highlights", () => {
      render(<PlatformMetrics metrics={mockMetrics} summary={mockSummary} />);

      expect(screen.getByText("Highlights")).toBeInTheDocument();
      expect(screen.getByText(/Strong lab growth/)).toBeInTheDocument();
    });

    it("should render concerns", () => {
      render(<PlatformMetrics metrics={mockMetrics} summary={mockSummary} />);

      expect(screen.getByText("Concerns")).toBeInTheDocument();
      expect(screen.getByText(/Paper failure rate/)).toBeInTheDocument();
    });

    it("should render recommendations", () => {
      render(<PlatformMetrics metrics={mockMetrics} summary={mockSummary} />);

      expect(screen.getByText("Recommendations")).toBeInTheDocument();
      expect(screen.getByText(/validating experimental/)).toBeInTheDocument();
    });
  });

  describe("Main Stats Grid", () => {
    it("should render labs metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      // Labs appears in both main stats and growth sections
      expect(screen.getAllByText("Labs").length).toBeGreaterThan(0);
      // Total labs value (150) appears as the main value
      expect(screen.getAllByText("150").length).toBeGreaterThan(0);
      // Active label appears in the sub-values (multiple instances)
      const activeLabels = screen.getAllByText("Active");
      expect(activeLabels.length).toBeGreaterThan(0);
      // Active labs count (120) appears as sub-value
      expect(screen.getAllByText("120").length).toBeGreaterThan(0);
    });

    it("should render papers metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      // Papers appears in both main stats and growth sections
      expect(screen.getAllByText("Papers").length).toBeGreaterThan(0);
      // Total papers (500) appears as the main value
      expect(screen.getAllByText("500").length).toBeGreaterThan(0);
      expect(screen.getByText("Processed")).toBeInTheDocument();
      // Processed papers count (450) appears as sub-value
      expect(screen.getAllByText("450").length).toBeGreaterThan(0);
    });

    it("should render techniques metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      // Techniques appears in both main stats and growth sections
      expect(screen.getAllByText("Techniques").length).toBeGreaterThan(0);
      // Total techniques (1200) formatted as 1,200
      const formattedValue = screen.getAllByText("1,200");
      expect(formattedValue.length).toBeGreaterThan(0);
    });

    it("should render collaborations metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      // Collaborations appears in both main stats and growth sections
      expect(screen.getAllByText("Collaborations").length).toBeGreaterThan(0);
      // Success rate 87.5% appears in sub-values (formatted as 88% due to Math.round)
      expect(screen.getAllByText("88%").length).toBeGreaterThan(0);
    });
  });

  describe("Growth Section", () => {
    it("should render growth metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Growth")).toBeInTheDocument();
      expect(screen.getByText("+8.5%")).toBeInTheDocument();
      expect(screen.getByText("+12.3%")).toBeInTheDocument();
    });

    it("should show positive growth in green", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      const growthValues = screen.getAllByText(/\+\d+\.\d+%/);
      growthValues.forEach((el) => {
        expect(el.className).toContain("text-green-600");
      });
    });
  });

  describe("Success Rates Section", () => {
    it("should render success rate metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Success Rates")).toBeInTheDocument();
      expect(screen.getByText("Task Completion")).toBeInTheDocument();
      expect(screen.getByText("97%")).toBeInTheDocument();
    });

    it("should render progress bars", () => {
      const { container } = render(<PlatformMetrics metrics={mockMetrics} />);

      // Progress bars use h-2 or h-full class inside rounded-full containers
      const progressBars = container.querySelectorAll(".h-2.rounded-full, .h-full.rounded-full");
      expect(progressBars.length).toBeGreaterThan(0);
    });
  });

  describe("Adoption Rates Section", () => {
    it("should render adoption metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Adoption This Period")).toBeInTheDocument();
      expect(screen.getByText("New Techniques")).toBeInTheDocument();
      // New techniques count (15) appears as main value
      expect(screen.getAllByText("15").length).toBeGreaterThan(0);
      expect(screen.getByText("Synergy Discoveries")).toBeInTheDocument();
      // Synergy discoveries count (8) may appear multiple times
      expect(screen.getAllByText("8").length).toBeGreaterThan(0);
    });
  });

  describe("Top Domains Section", () => {
    it("should render top domains", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Top Domains")).toBeInTheDocument();
      expect(screen.getByText("TTS")).toBeInTheDocument();
      expect(screen.getByText("Voice Cloning")).toBeInTheDocument();
    });

    it("should render domain progress bars", () => {
      const { container } = render(<PlatformMetrics metrics={mockMetrics} />);

      const indigo = container.querySelectorAll(".bg-indigo-500");
      expect(indigo.length).toBeGreaterThan(0);
    });
  });

  describe("Top Categories Section", () => {
    it("should render top categories", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Top Categories")).toBeInTheDocument();
      expect(screen.getByText("prosody")).toBeInTheDocument();
      expect(screen.getByText("codec")).toBeInTheDocument();
    });
  });

  describe("Cost Metrics Section", () => {
    it("should render cost metrics", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Cost Metrics")).toBeInTheDocument();
      expect(screen.getByText("API Calls")).toBeInTheDocument();
      expect(screen.getByText("50,000")).toBeInTheDocument();
      expect(screen.getByText("$50.00")).toBeInTheDocument();
    });

    it("should render compute hours", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Compute Hours")).toBeInTheDocument();
      expect(screen.getByText("120.0")).toBeInTheDocument();
    });

    it("should render cost per paper", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      expect(screen.getByText("Cost/Paper")).toBeInTheDocument();
      expect(screen.getByText("$0.100")).toBeInTheDocument();
    });
  });

  describe("Period Selection", () => {
    it("should highlight selected period", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      const dailyBtn = screen.getByText("Daily");
      expect(dailyBtn.className).toContain("bg-indigo-600");
    });

    it("should update selected period on click", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      fireEvent.click(screen.getByText("Monthly"));

      const monthlyBtn = screen.getByText("Monthly");
      expect(monthlyBtn.className).toContain("bg-indigo-600");
    });
  });

  describe("New Items Highlight", () => {
    it("should highlight new labs count", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      // New labs count (8) is highlighted with green color
      const newValues = screen.getAllByText("8");
      const highlightedValue = newValues.find((el) =>
        el.className.includes("text-green-600")
      );
      expect(highlightedValue).toBeTruthy();
    });
  });

  describe("Failed Items Warning", () => {
    it("should show failed paper count", () => {
      render(<PlatformMetrics metrics={mockMetrics} />);

      // Failed count is 10, should be displayed
      expect(screen.getAllByText("10").length).toBeGreaterThan(0);
    });
  });
});
