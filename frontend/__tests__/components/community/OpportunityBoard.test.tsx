/**
 * Tests for OpportunityBoard Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { OpportunityBoard } from "@/components/community/OpportunityBoard";
import type {
  ResearchOpportunity,
  OpportunityLeaderboard,
} from "@/lib/meta/community/types";

describe("OpportunityBoard", () => {
  const mockOpportunities: ResearchOpportunity[] = [
    {
      id: "opp-1",
      title: "Multi-speaker prosody adaptation",
      description: "Develop technique for adapting prosody across speakers",
      gapType: "missing-technique",
      difficulty: "advanced",
      estimatedEffort: { minHours: 40, maxHours: 80, recommendedTeamSize: 2 },
      impact: "high",
      domain: "Voice Cloning",
      tags: ["prosody", "multi-speaker"],
      status: "open",
      progress: 0,
      hasBounty: true,
      bounty: { amount: 500, currency: "USD", sponsor: "Voice AI", conditions: [] },
      relatedTechniques: ["tech-1"],
      requiredExpertise: ["Deep Learning"],
      suggestedApproaches: ["Transfer learning"],
      createdAt: "2025-01-20T10:00:00Z",
      updatedAt: "2025-01-20T10:00:00Z",
    },
    {
      id: "opp-2",
      title: "Real-time emotion detection",
      description: "Create fast emotion detection module",
      gapType: "performance-optimization",
      difficulty: "intermediate",
      estimatedEffort: { minHours: 20, maxHours: 40, recommendedTeamSize: 1 },
      impact: "medium",
      domain: "TTS",
      tags: ["emotion", "real-time"],
      status: "claimed",
      progress: 30,
      hasBounty: false,
      claimedBy: { labId: "lab-1", labName: "Test Lab", claimedAt: "2025-01-22T10:00:00Z" },
      relatedTechniques: [],
      requiredExpertise: [],
      suggestedApproaches: [],
      createdAt: "2025-01-21T10:00:00Z",
      updatedAt: "2025-01-22T10:00:00Z",
    },
    {
      id: "opp-3",
      title: "Documentation for voice cloning",
      description: "Write comprehensive documentation",
      gapType: "documentation",
      difficulty: "beginner",
      estimatedEffort: { minHours: 5, maxHours: 10, recommendedTeamSize: 1 },
      impact: "low",
      domain: "Documentation",
      tags: ["docs"],
      status: "open",
      progress: 0,
      hasBounty: false,
      relatedTechniques: [],
      requiredExpertise: [],
      suggestedApproaches: [],
      createdAt: "2025-01-19T10:00:00Z",
      updatedAt: "2025-01-19T10:00:00Z",
    },
  ];

  const mockLeaderboard: OpportunityLeaderboard = {
    entries: [
      {
        rank: 1,
        labId: "lab-1",
        labName: "Voice Lab",
        opportunitiesCompleted: 10,
        totalBountyEarned: 2500,
        impactScore: 850,
        streak: 5,
        badges: ["veteran", "top-contributor"],
      },
      {
        rank: 2,
        labId: "lab-2",
        labName: "Audio Team",
        opportunitiesCompleted: 7,
        totalBountyEarned: 1500,
        impactScore: 600,
        streak: 3,
        badges: ["high-impact"],
      },
    ],
    period: "all-time",
    updatedAt: "2025-01-27T10:00:00Z",
  };

  describe("Rendering", () => {
    it("should render board header", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText("Research Opportunities")).toBeInTheDocument();
      expect(screen.getByText("3 opportunities available")).toBeInTheDocument();
    });

    it("should render status summary", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText("Open")).toBeInTheDocument();
      expect(screen.getByText("Claimed")).toBeInTheDocument();
    });

    it("should render all opportunities", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText("Multi-speaker prosody adaptation")).toBeInTheDocument();
      expect(screen.getByText("Real-time emotion detection")).toBeInTheDocument();
      expect(screen.getByText("Documentation for voice cloning")).toBeInTheDocument();
    });

    it("should render opportunity details", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      // Voice Cloning appears in both dropdown and card
      expect(screen.getAllByText("Voice Cloning").length).toBeGreaterThan(0);
      expect(screen.getByText("Missing Technique")).toBeInTheDocument();
      // Difficulty badge may appear multiple times
      expect(screen.getAllByText("advanced").length).toBeGreaterThan(0);
      // Impact badge
      expect(screen.getAllByText("high impact").length).toBeGreaterThan(0);
    });

    it("should render bounty badge", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText("$500 bounty")).toBeInTheDocument();
    });

    it("should render claimed info", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText(/Claimed by/)).toBeInTheDocument();
      expect(screen.getByText("Test Lab")).toBeInTheDocument();
    });

    it("should render progress bar for claimed opportunities", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText("Progress")).toBeInTheDocument();
      expect(screen.getByText("30%")).toBeInTheDocument();
    });
  });

  describe("Filtering", () => {
    it("should filter by search query", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      const searchInput = screen.getByPlaceholderText("Search opportunities...");
      fireEvent.change(searchInput, { target: { value: "emotion" } });

      expect(screen.queryByText("Multi-speaker prosody adaptation")).not.toBeInTheDocument();
      expect(screen.getByText("Real-time emotion detection")).toBeInTheDocument();
    });

    it("should filter by domain", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      const domainSelect = screen.getByDisplayValue("All Domains");
      fireEvent.change(domainSelect, { target: { value: "TTS" } });

      expect(screen.queryByText("Multi-speaker prosody adaptation")).not.toBeInTheDocument();
      expect(screen.getByText("Real-time emotion detection")).toBeInTheDocument();
    });

    it("should filter by difficulty", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      const difficultySelect = screen.getByDisplayValue("All Difficulties");
      fireEvent.change(difficultySelect, { target: { value: "beginner" } });

      expect(screen.queryByText("Multi-speaker prosody adaptation")).not.toBeInTheDocument();
      expect(screen.getByText("Documentation for voice cloning")).toBeInTheDocument();
    });

    it("should filter by bounty only", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      const bountyCheckbox = screen.getByLabelText("Bounty Only");
      fireEvent.click(bountyCheckbox);

      expect(screen.getByText("Multi-speaker prosody adaptation")).toBeInTheDocument();
      expect(screen.queryByText("Real-time emotion detection")).not.toBeInTheDocument();
    });
  });

  describe("Leaderboard", () => {
    it("should show leaderboard button when leaderboard provided", () => {
      render(
        <OpportunityBoard
          opportunities={mockOpportunities}
          leaderboard={mockLeaderboard}
        />
      );

      expect(screen.getByText("Leaderboard")).toBeInTheDocument();
    });

    it("should toggle leaderboard visibility", () => {
      render(
        <OpportunityBoard
          opportunities={mockOpportunities}
          leaderboard={mockLeaderboard}
        />
      );

      const leaderboardBtn = screen.getByText("Leaderboard");
      fireEvent.click(leaderboardBtn);

      expect(screen.getByText("Top Contributors")).toBeInTheDocument();
      expect(screen.getByText("Voice Lab")).toBeInTheDocument();
      expect(screen.getByText("Audio Team")).toBeInTheDocument();
    });

    it("should highlight current user in leaderboard", () => {
      render(
        <OpportunityBoard
          opportunities={mockOpportunities}
          leaderboard={mockLeaderboard}
          currentLabId="lab-1"
        />
      );

      const leaderboardBtn = screen.getByText("Leaderboard");
      fireEvent.click(leaderboardBtn);

      expect(screen.getByText("(You)")).toBeInTheDocument();
    });
  });

  describe("Actions", () => {
    it("should call onClaim when claim button clicked", () => {
      const onClaim = vi.fn();
      render(
        <OpportunityBoard
          opportunities={mockOpportunities}
          onClaim={onClaim}
        />
      );

      const claimButtons = screen.getAllByText("Claim Opportunity");
      fireEvent.click(claimButtons[0]);

      expect(onClaim).toHaveBeenCalledWith("opp-1");
    });

    it("should call onUnclaim when release claim clicked", () => {
      const onUnclaim = vi.fn();
      render(
        <OpportunityBoard
          opportunities={mockOpportunities}
          onUnclaim={onUnclaim}
          currentLabId="lab-1"
        />
      );

      const releaseBtn = screen.getByText("Release Claim");
      fireEvent.click(releaseBtn);

      expect(onUnclaim).toHaveBeenCalledWith("opp-2");
    });

    it("should call onViewDetails when view details clicked", () => {
      const onViewDetails = vi.fn();
      render(
        <OpportunityBoard
          opportunities={mockOpportunities}
          onViewDetails={onViewDetails}
        />
      );

      const viewButtons = screen.getAllByText("View Details");
      fireEvent.click(viewButtons[0]);

      expect(onViewDetails).toHaveBeenCalledWith(mockOpportunities[0]);
    });
  });

  describe("Empty State", () => {
    it("should show empty state when no opportunities", () => {
      render(<OpportunityBoard opportunities={[]} />);

      expect(screen.getByText("No opportunities found")).toBeInTheDocument();
      expect(screen.getByText("Try adjusting your filters")).toBeInTheDocument();
    });

    it("should show empty state when filters match nothing", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      const searchInput = screen.getByPlaceholderText("Search opportunities...");
      fireEvent.change(searchInput, { target: { value: "nonexistent" } });

      expect(screen.getByText("No opportunities found")).toBeInTheDocument();
    });
  });

  describe("Tags and Badges", () => {
    it("should render tags", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText("prosody")).toBeInTheDocument();
      expect(screen.getByText("multi-speaker")).toBeInTheDocument();
    });

    it("should render effort estimate", () => {
      render(<OpportunityBoard opportunities={mockOpportunities} />);

      expect(screen.getByText("40-80h")).toBeInTheDocument();
    });
  });
});
