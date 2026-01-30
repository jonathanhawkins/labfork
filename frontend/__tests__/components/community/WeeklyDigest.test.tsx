/**
 * Tests for WeeklyDigest Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { WeeklyDigest } from "@/components/community/WeeklyDigest";
import type { WeeklyDigest as DigestType } from "@/lib/meta/community/types";

describe("WeeklyDigest", () => {
  const mockDigest: DigestType = {
    id: "digest-2025-W04",
    weekNumber: 4,
    year: 2025,
    startDate: "2025-01-20T00:00:00Z",
    endDate: "2025-01-26T23:59:59Z",
    generatedAt: "2025-01-27T10:00:00Z",
    publishedAt: "2025-01-27T12:00:00Z",
    status: "published",
    summary: {
      headline: "Week of Breakthroughs: Novel Prosody Transfer",
      keyHighlights: [
        "3 breakthroughs discovered",
        "Attention Prosody trending with 67% growth",
        "5 new synergies identified",
      ],
      wordCount: 1200,
      readTimeMinutes: 6,
    },
    breakthroughs: [
      {
        id: "break-1",
        title: "Novel Prosody Transfer Method",
        description: "A breakthrough in prosody transfer using attention",
        significance: "high",
        discoveredBy: "Voice Lab",
        labId: "lab-1",
        labName: "Voice Lab",
        techniqueIds: ["tech-1", "tech-2"],
        metrics: {
          improvementPercent: 35,
          affectedDomains: ["TTS", "Voice Cloning"],
          potentialApplications: 5,
          communityInterest: 120,
        },
        timestamp: "2025-01-25T10:00:00Z",
      },
    ],
    trendingTechniques: [
      {
        id: "tech-1",
        name: "Attention Prosody",
        category: "prosody",
        trendScore: 85,
        adoptionRate: 0.6,
        weeklyGrowth: 67,
        usageCount: 150,
        topLabs: ["lab-1", "lab-2"],
        domains: ["TTS"],
      },
    ],
    newSynergies: [
      {
        id: "syn-1",
        techniques: ["Attention Prosody", "Neural Codec"],
        synergyScore: 0.85,
        discoveredBy: "Synergy Detector",
        labId: "lab-1",
        description: "Excellent synergy between attention and codec",
        potentialImpact: "high",
        timestamp: "2025-01-24T10:00:00Z",
      },
    ],
    researchOpportunities: [
      {
        id: "opp-1",
        title: "Multi-speaker adaptation",
        gapType: "missing-technique",
        difficulty: "advanced",
        impact: "high",
        domain: "Voice Cloning",
      },
    ],
    evolutionHighlights: [
      {
        id: "evo-1",
        techniqueId: "tech-1",
        techniqueName: "Attention Prosody v2",
        generation: 3,
        fitnessImprovement: 18.5,
        parentTechniques: ["tech-base"],
        newCapabilities: ["Better emotion", "Faster inference"],
        timestamp: "2025-01-23T10:00:00Z",
      },
    ],
    collaborationUpdates: [
      {
        id: "collab-1",
        metaTaskId: "meta-1",
        title: "Joint Prosody Research",
        status: "active",
        participantCount: 4,
        recentProgress: "Completed phase 1",
        completedObjectives: 2,
        totalObjectives: 5,
      },
    ],
    topContributors: [
      {
        labId: "lab-1",
        labName: "Voice Research Lab",
        contributions: 15,
        discoveries: 5,
        collaborations: 3,
        weeklyScore: 85,
        rank: 1,
      },
      {
        labId: "lab-2",
        labName: "Audio ML Team",
        contributions: 10,
        discoveries: 3,
        collaborations: 2,
        weeklyScore: 55,
        rank: 2,
      },
    ],
    stats: {
      totalPapers: 500,
      newPapers: 25,
      totalTechniques: 1200,
      newTechniques: 15,
      totalLabs: 150,
      activeLabs: 120,
      synergiesDiscovered: 8,
      gapsFilled: 3,
      collaborationsStarted: 2,
      collaborationsCompleted: 1,
      evolutionGenerations: 5,
      topDomain: "TTS",
    },
  };

  describe("Rendering", () => {
    it("should render digest header", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("Week 4, 2025")).toBeInTheDocument();
      expect(screen.getByText(mockDigest.summary.headline)).toBeInTheDocument();
    });

    it("should render key highlights", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("Key Highlights")).toBeInTheDocument();
      mockDigest.summary.keyHighlights.forEach((highlight) => {
        expect(screen.getByText(highlight)).toBeInTheDocument();
      });
    });

    it("should render stats bar", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("Active Labs")).toBeInTheDocument();
      expect(screen.getByText("New Techniques")).toBeInTheDocument();
      expect(screen.getByText("Synergies")).toBeInTheDocument();
      expect(screen.getByText(String(mockDigest.stats.activeLabs))).toBeInTheDocument();
    });

    it("should render breakthroughs section", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("Breakthroughs")).toBeInTheDocument();
      expect(screen.getByText("Novel Prosody Transfer Method")).toBeInTheDocument();
    });

    it("should render trending techniques section", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("Trending Techniques")).toBeInTheDocument();
      // Click to expand trending section
      const trendingBtn = screen.getByText("Trending Techniques").closest("button");
      fireEvent.click(trendingBtn!);

      expect(screen.getByText("Attention Prosody")).toBeInTheDocument();
      expect(screen.getByText("+67%")).toBeInTheDocument();
    });

    it("should render synergies section", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("New Synergies")).toBeInTheDocument();
    });

    it("should render top contributors section", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("Top Contributors")).toBeInTheDocument();
      // Click to expand contributors section
      const contributorsBtn = screen.getByText("Top Contributors").closest("button");
      fireEvent.click(contributorsBtn!);

      expect(screen.getByText("Voice Research Lab")).toBeInTheDocument();
    });

    it("should render read time", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("6 min read")).toBeInTheDocument();
    });
  });

  describe("Sections Expansion", () => {
    it("should expand/collapse sections", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      // Breakthroughs section starts expanded by default
      expect(screen.getByText("Novel Prosody Transfer Method")).toBeInTheDocument();

      // Click on Breakthroughs to collapse
      const breakthroughsBtn = screen.getByText("Breakthroughs").closest("button");
      fireEvent.click(breakthroughsBtn!);

      // Click on Trending Techniques to expand
      const trendingBtn = screen.getByText("Trending Techniques").closest("button");
      fireEvent.click(trendingBtn!);
    });
  });

  describe("Actions", () => {
    it("should call onShare when share button clicked", () => {
      const onShare = vi.fn();
      render(<WeeklyDigest digest={mockDigest} onShare={onShare} />);

      const shareBtn = screen.getByText("Share");
      fireEvent.click(shareBtn);

      expect(onShare).toHaveBeenCalledTimes(1);
    });

    it("should call onSubscribe when subscribe button clicked", () => {
      const onSubscribe = vi.fn();
      render(<WeeklyDigest digest={mockDigest} onSubscribe={onSubscribe} />);

      const subscribeBtn = screen.getByText("Subscribe");
      fireEvent.click(subscribeBtn);

      expect(onSubscribe).toHaveBeenCalledTimes(1);
    });

    it("should not render action buttons if handlers not provided", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.queryByText("Share")).not.toBeInTheDocument();
      expect(screen.queryByText("Subscribe")).not.toBeInTheDocument();
    });
  });

  describe("Empty States", () => {
    it("should handle empty breakthroughs", () => {
      const emptyDigest = { ...mockDigest, breakthroughs: [] };
      render(<WeeklyDigest digest={emptyDigest} />);

      expect(screen.queryByText("Breakthroughs")).not.toBeInTheDocument();
    });

    it("should handle empty trending techniques", () => {
      const emptyDigest = { ...mockDigest, trendingTechniques: [] };
      render(<WeeklyDigest digest={emptyDigest} />);

      expect(screen.queryByText("Trending Techniques")).not.toBeInTheDocument();
    });
  });

  describe("Significance Display", () => {
    it("should show significance badge for breakthroughs", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("high")).toBeInTheDocument();
    });

    it("should show improvement percentage", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      expect(screen.getByText("+35% improvement")).toBeInTheDocument();
    });
  });

  describe("Contributors Display", () => {
    it("should show rank badges", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      // Click to expand contributors section
      const contributorsBtn = screen.getByText("Top Contributors").closest("button");
      fireEvent.click(contributorsBtn!);

      // Check for rank badges - look for elements with rank styling
      const rankElements = screen.getAllByText((content, element) => {
        return (content === "1" || content === "2") &&
               element?.className?.includes("rounded-full") === true;
      });
      expect(rankElements.length).toBeGreaterThanOrEqual(2);
    });

    it("should show contributor stats", () => {
      render(<WeeklyDigest digest={mockDigest} />);

      const contributorsBtn = screen.getByText("Top Contributors").closest("button");
      fireEvent.click(contributorsBtn!);

      expect(screen.getByText(/5 discoveries/)).toBeInTheDocument();
      expect(screen.getByText(/15 contributions/)).toBeInTheDocument();
    });
  });
});
