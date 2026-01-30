/**
 * CollaborationBoard Component Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { CollaborationBoard } from "@/components/collaboration/CollaborationBoard";
import { MetaTaskSummary } from "@/components/collaboration/MetaTaskCard";

describe("CollaborationBoard", () => {
  const createMockTask = (
    id: string,
    overrides: Partial<MetaTaskSummary> = {}
  ): MetaTaskSummary => ({
    id,
    title: `Task ${id}`,
    description: `Description for ${id}`,
    category: "exploration",
    status: "proposed",
    leadLab: "Lead Lab",
    participantCount: 1,
    objectiveCount: 2,
    completedObjectives: 0,
    visibility: "public",
    tags: [],
    domains: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    ...overrides,
  });

  const mockTasks: MetaTaskSummary[] = [
    createMockTask("task-1", { status: "proposed", category: "exploration" }),
    createMockTask("task-2", { status: "recruiting", category: "integration" }),
    createMockTask("task-3", { status: "active", category: "benchmark" }),
    createMockTask("task-4", { status: "reviewing", category: "dataset" }),
    createMockTask("task-5", { status: "completed", category: "exploration" }),
  ];

  describe("Rendering", () => {
    it("should render board header", () => {
      render(<CollaborationBoard tasks={mockTasks} />);
      expect(screen.getByText("Collaboration Board")).toBeInTheDocument();
    });

    it("should show task statistics", () => {
      render(<CollaborationBoard tasks={mockTasks} />);
      expect(screen.getByText(/5 tasks/)).toBeInTheDocument();
    });

    it("should show loading state", () => {
      render(<CollaborationBoard tasks={[]} loading />);
      // Should have loading spinner
      expect(document.querySelector(".animate-spin")).toBeInTheDocument();
    });

    it("should show New Task button when onCreateTask provided", () => {
      const onCreateTask = vi.fn();
      render(<CollaborationBoard tasks={mockTasks} onCreateTask={onCreateTask} />);
      expect(screen.getByText("New Task")).toBeInTheDocument();
    });

    it("should call onCreateTask when clicked", () => {
      const onCreateTask = vi.fn();
      render(<CollaborationBoard tasks={mockTasks} onCreateTask={onCreateTask} />);

      fireEvent.click(screen.getByText("New Task"));
      expect(onCreateTask).toHaveBeenCalled();
    });
  });

  describe("Board View", () => {
    it("should render status columns", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      expect(screen.getByText("Proposed")).toBeInTheDocument();
      expect(screen.getByText("Recruiting")).toBeInTheDocument();
      expect(screen.getByText("Active")).toBeInTheDocument();
      expect(screen.getByText("Reviewing")).toBeInTheDocument();
    });

    it("should group tasks by status", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      // Each column should have its task
      expect(screen.getByText("Task task-1")).toBeInTheDocument();
      expect(screen.getByText("Task task-2")).toBeInTheDocument();
      expect(screen.getByText("Task task-3")).toBeInTheDocument();
    });

    it("should not show completed column by default", () => {
      render(<CollaborationBoard tasks={mockTasks} />);
      // Completed task should not be visible by default
      expect(screen.queryByText("Task task-5")).not.toBeInTheDocument();
    });
  });

  describe("List View", () => {
    it("should switch to list view", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      // Find and click list view button
      const listButton = screen.getByRole("button", { name: /list view/i });
      fireEvent.click(listButton);

      // In list view, tasks should be shown differently
      expect(screen.getByText("Task task-1")).toBeInTheDocument();
    });
  });

  describe("Filtering", () => {
    it("should filter by search query", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      const searchInput = screen.getByPlaceholderText("Search tasks...");
      fireEvent.change(searchInput, { target: { value: "task-1" } });

      expect(screen.getByText("Task task-1")).toBeInTheDocument();
      expect(screen.queryByText("Task task-2")).not.toBeInTheDocument();
    });

    it("should filter by category", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      const categorySelect = screen.getByRole("combobox");
      fireEvent.change(categorySelect, { target: { value: "integration" } });

      expect(screen.getByText("Task task-2")).toBeInTheDocument();
      expect(screen.queryByText("Task task-1")).not.toBeInTheDocument();
    });

    it("should show completed tasks when toggled", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      const showCompletedCheckbox = screen.getByRole("checkbox");
      fireEvent.click(showCompletedCheckbox);

      // Completed column should now show
      expect(screen.getByText("Completed")).toBeInTheDocument();
    });

    it("should combine multiple filters", () => {
      const tasks = [
        createMockTask("match", {
          title: "Matching Task",
          category: "exploration",
          status: "proposed",
        }),
        createMockTask("no-match-1", {
          title: "Other Task",
          category: "integration",
          status: "proposed",
        }),
        createMockTask("no-match-2", {
          title: "Matching Name",
          category: "exploration",
          status: "active",
        }),
      ];

      render(<CollaborationBoard tasks={tasks} />);

      // Apply search filter
      const searchInput = screen.getByPlaceholderText("Search tasks...");
      fireEvent.change(searchInput, { target: { value: "Matching" } });

      // Apply category filter
      const categorySelect = screen.getByRole("combobox");
      fireEvent.change(categorySelect, { target: { value: "exploration" } });

      // Both matching tasks should be visible
      expect(screen.getByText("Matching Task")).toBeInTheDocument();
      expect(screen.getByText("Matching Name")).toBeInTheDocument();
      expect(screen.queryByText("Other Task")).not.toBeInTheDocument();
    });

    it("should search in tags", () => {
      const tasks = [
        createMockTask("tagged", {
          title: "Tagged Task",
          tags: ["important", "research"],
        }),
        createMockTask("untagged", {
          title: "Untagged Task",
          tags: [],
        }),
      ];

      render(<CollaborationBoard tasks={tasks} />);

      const searchInput = screen.getByPlaceholderText("Search tasks...");
      fireEvent.change(searchInput, { target: { value: "important" } });

      expect(screen.getByText("Tagged Task")).toBeInTheDocument();
      expect(screen.queryByText("Untagged Task")).not.toBeInTheDocument();
    });
  });

  describe("Task Actions", () => {
    it("should call onViewTask when task is clicked", () => {
      const onViewTask = vi.fn();
      render(<CollaborationBoard tasks={mockTasks} onViewTask={onViewTask} />);

      // Click on a task
      fireEvent.click(screen.getByText("Task task-1"));
      expect(onViewTask).toHaveBeenCalledWith("task-1");
    });

    it("should call onJoinTask when join is triggered", () => {
      const onJoinTask = vi.fn();
      const tasks = [
        createMockTask("joinable", {
          title: "Joinable Task",
          status: "recruiting",
        }),
      ];

      render(<CollaborationBoard tasks={tasks} onJoinTask={onJoinTask} />);

      // In list view for easier access to Join button
      const listButton = screen.getByRole("button", { name: /list view/i });
      fireEvent.click(listButton);

      const joinButton = screen.getByText("Join Task");
      fireEvent.click(joinButton);

      expect(onJoinTask).toHaveBeenCalledWith("joinable");
    });
  });

  describe("Empty States", () => {
    it("should show empty state when no tasks", () => {
      render(<CollaborationBoard tasks={[]} />);
      expect(screen.getByText(/0 tasks/)).toBeInTheDocument();
    });

    it("should show empty message when no tasks match filter", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      const searchInput = screen.getByPlaceholderText("Search tasks...");
      fireEvent.change(searchInput, { target: { value: "nonexistent" } });

      // Switch to list view to see empty message
      const listButton = screen.getByRole("button", { name: /list view/i });
      fireEvent.click(listButton);

      expect(screen.getByText(/No tasks match/)).toBeInTheDocument();
    });

    it("should show 'No tasks' in empty columns", () => {
      const tasks = [createMockTask("only-one", { status: "active" })];

      render(<CollaborationBoard tasks={tasks} />);

      // Proposed column should show "No tasks"
      expect(screen.getAllByText("No tasks").length).toBeGreaterThan(0);
    });
  });

  describe("View Mode Toggle", () => {
    it("should default to board view", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      // Board view button should be active (styled differently)
      const boardButton = screen.getByRole("button", { name: /board view/i });
      expect(boardButton.className).toContain("bg-blue");
    });

    it("should toggle between views", () => {
      render(<CollaborationBoard tasks={mockTasks} />);

      const listButton = screen.getByRole("button", { name: /list view/i });
      fireEvent.click(listButton);
      expect(listButton.className).toContain("bg-blue");

      const boardButton = screen.getByRole("button", { name: /board view/i });
      fireEvent.click(boardButton);
      expect(boardButton.className).toContain("bg-blue");
    });
  });
});
