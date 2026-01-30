/**
 * MetaTaskCard Component Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MetaTaskCard, MetaTaskSummary } from "@/components/collaboration/MetaTaskCard";
import { MetaTask, Participant, createParticipantId, createMetaTaskId } from "@/lib/meta/collaboration/types";

describe("MetaTaskCard", () => {
  const createMockParticipant = (overrides: Partial<Participant> = {}): Participant => ({
    id: createParticipantId(),
    labId: "lab-1",
    labName: "Test Lab",
    role: "contributor",
    status: "active",
    expertise: ["ml"],
    commitment: { hoursPerWeek: 10, durationWeeks: 4, resources: [], responsibilities: [] },
    joinedAt: new Date().toISOString(),
    lastActiveAt: new Date().toISOString(),
    contributions: [],
    assignedObjectives: [],
    ...overrides,
  });

  const createMockTask = (overrides: Partial<MetaTask> = {}): MetaTask => {
    const leadParticipant = createMockParticipant({ role: "lead", labName: "Lead Lab" });
    return {
      id: createMetaTaskId(),
      title: "Test Meta-Task",
      description: "A test task for collaboration",
      category: "exploration",
      status: "proposed",
      lead: leadParticipant,
      participants: [leadParticipant],
      objectives: [],
      results: [],
      requirements: {
        minParticipants: 2,
        maxParticipants: 10,
        requiredExpertise: [],
        preferredExpertise: [],
        resourceRequirements: [],
      },
      timeline: { phases: [] },
      completionCriteria: [],
      tags: ["test", "collaboration"],
      domains: ["research"],
      visibility: "public",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      ...overrides,
    };
  };

  const createMockSummary = (overrides: Partial<MetaTaskSummary> = {}): MetaTaskSummary => ({
    id: "meta-task-123",
    title: "Summary Task",
    description: "A summary for listing",
    category: "integration",
    status: "recruiting",
    leadLab: "Lead Lab",
    participantCount: 3,
    objectiveCount: 5,
    completedObjectives: 2,
    visibility: "public",
    tags: ["summary"],
    domains: ["nlp"],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    ...overrides,
  });

  describe("Rendering", () => {
    it("should render task title", () => {
      render(<MetaTaskCard task={createMockTask()} />);
      expect(screen.getByText("Test Meta-Task")).toBeInTheDocument();
    });

    it("should render task description", () => {
      render(<MetaTaskCard task={createMockTask()} />);
      expect(screen.getByText("A test task for collaboration")).toBeInTheDocument();
    });

    it("should render status badge", () => {
      render(<MetaTaskCard task={createMockTask({ status: "active" })} />);
      expect(screen.getByText("active")).toBeInTheDocument();
    });

    it("should render category label", () => {
      render(<MetaTaskCard task={createMockTask({ category: "benchmark" })} />);
      expect(screen.getByText("Benchmark")).toBeInTheDocument();
    });

    it("should render lead lab name", () => {
      render(<MetaTaskCard task={createMockTask()} />);
      expect(screen.getByText("Lead Lab")).toBeInTheDocument();
    });

    it("should render participant count", () => {
      render(<MetaTaskCard task={createMockTask()} />);
      expect(screen.getByText(/1 participant/)).toBeInTheDocument();
    });

    it("should render tags", () => {
      render(<MetaTaskCard task={createMockTask()} />);
      expect(screen.getByText("test")).toBeInTheDocument();
      expect(screen.getByText("collaboration")).toBeInTheDocument();
    });

    it("should render domains", () => {
      render(<MetaTaskCard task={createMockTask()} />);
      expect(screen.getByText("research")).toBeInTheDocument();
    });
  });

  describe("Progress Display", () => {
    it("should show progress bar when objectives exist", () => {
      const task = createMockTask({
        objectives: [
          {
            id: "obj-1",
            title: "Objective 1",
            description: "First objective",
            priority: "high",
            status: "completed",
            assignedTo: [],
            dependencies: [],
            milestones: [],
            deliverables: [],
            progress: 100,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
          {
            id: "obj-2",
            title: "Objective 2",
            description: "Second objective",
            priority: "medium",
            status: "in_progress",
            assignedTo: [],
            dependencies: [],
            milestones: [],
            deliverables: [],
            progress: 50,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
        ],
      });

      render(<MetaTaskCard task={task} />);
      expect(screen.getByText(/1\/2 objectives/)).toBeInTheDocument();
    });

    it("should show correct progress percentage", () => {
      const summary = createMockSummary({
        objectiveCount: 4,
        completedObjectives: 2,
      });

      render(<MetaTaskCard task={summary} />);
      expect(screen.getByText(/50%/)).toBeInTheDocument();
    });
  });

  describe("Summary Mode", () => {
    it("should render from summary data", () => {
      render(<MetaTaskCard task={createMockSummary()} />);
      expect(screen.getByText("Summary Task")).toBeInTheDocument();
      expect(screen.getByText("recruiting")).toBeInTheDocument();
    });

    it("should show participant count from summary", () => {
      render(<MetaTaskCard task={createMockSummary({ participantCount: 5 })} />);
      expect(screen.getByText(/5 participant/)).toBeInTheDocument();
    });
  });

  describe("Compact Mode", () => {
    it("should render in compact mode", () => {
      render(<MetaTaskCard task={createMockTask()} compact />);
      // Should have title but less detail
      expect(screen.getByText("Test Meta-Task")).toBeInTheDocument();
    });

    it("should be clickable in compact mode", () => {
      const onView = vi.fn();
      render(<MetaTaskCard task={createMockTask()} compact onView={onView} />);

      fireEvent.click(screen.getByText("Test Meta-Task"));
      expect(onView).toHaveBeenCalled();
    });
  });

  describe("Actions", () => {
    it("should show View Details button when onView provided", () => {
      const onView = vi.fn();
      render(<MetaTaskCard task={createMockTask()} onView={onView} />);
      expect(screen.getByText("View Details")).toBeInTheDocument();
    });

    it("should call onView when View Details clicked", () => {
      const onView = vi.fn();
      render(<MetaTaskCard task={createMockTask()} onView={onView} />);

      fireEvent.click(screen.getByText("View Details"));
      expect(onView).toHaveBeenCalled();
    });

    it("should show Join Task button for open tasks", () => {
      const onJoin = vi.fn();
      render(<MetaTaskCard task={createMockTask({ status: "recruiting" })} onJoin={onJoin} />);
      expect(screen.getByText("Join Task")).toBeInTheDocument();
    });

    it("should call onJoin when Join Task clicked", () => {
      const onJoin = vi.fn();
      render(<MetaTaskCard task={createMockTask({ status: "recruiting" })} onJoin={onJoin} />);

      fireEvent.click(screen.getByText("Join Task"));
      expect(onJoin).toHaveBeenCalled();
    });

    it("should not show Join Task for active tasks", () => {
      const onJoin = vi.fn();
      render(<MetaTaskCard task={createMockTask({ status: "active" })} onJoin={onJoin} />);
      expect(screen.queryByText("Join Task")).not.toBeInTheDocument();
    });

    it("should hide actions when showActions is false", () => {
      const onView = vi.fn();
      const onJoin = vi.fn();
      render(
        <MetaTaskCard
          task={createMockTask({ status: "proposed" })}
          onView={onView}
          onJoin={onJoin}
          showActions={false}
        />
      );
      expect(screen.queryByText("View Details")).not.toBeInTheDocument();
      expect(screen.queryByText("Join Task")).not.toBeInTheDocument();
    });
  });

  describe("Visibility Indicators", () => {
    it("should show private indicator", () => {
      render(<MetaTaskCard task={createMockTask({ visibility: "private" })} />);
      expect(screen.getByText("Private")).toBeInTheDocument();
    });

    it("should show invite only indicator", () => {
      render(<MetaTaskCard task={createMockTask({ visibility: "invite_only" })} />);
      expect(screen.getByText("Invite Only")).toBeInTheDocument();
    });

    it("should not show indicator for public tasks", () => {
      render(<MetaTaskCard task={createMockTask({ visibility: "public" })} />);
      expect(screen.queryByText("Private")).not.toBeInTheDocument();
      expect(screen.queryByText("Invite Only")).not.toBeInTheDocument();
    });
  });

  describe("Status Colors", () => {
    it("should have appropriate styling for different statuses", () => {
      const statuses = [
        "proposed",
        "recruiting",
        "accepted",
        "active",
        "reviewing",
        "completed",
        "cancelled",
        "failed",
      ] as const;

      statuses.forEach((status) => {
        const { unmount } = render(<MetaTaskCard task={createMockTask({ status })} />);
        expect(screen.getByText(status)).toBeInTheDocument();
        unmount();
      });
    });
  });
});
