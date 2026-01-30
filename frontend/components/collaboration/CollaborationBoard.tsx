/**
 * CollaborationBoard
 *
 * A Kanban-style board displaying meta-tasks organized by status.
 * Supports filtering, search, and actions on tasks.
 */

"use client";

import React, { useState, useMemo } from "react";
import { MetaTaskCard, MetaTaskSummary } from "./MetaTaskCard";
import {
  MetaTask,
  MetaTaskStatus,
  MetaTaskCategory,
} from "@/lib/meta/collaboration/types";

interface CollaborationBoardProps {
  tasks: (MetaTask | MetaTaskSummary)[];
  onJoinTask?: (taskId: string) => void;
  onViewTask?: (taskId: string) => void;
  onCreateTask?: () => void;
  loading?: boolean;
  currentLabId?: string;
}

type ViewMode = "board" | "list";

const statusColumns: { status: MetaTaskStatus; label: string; color: string }[] = [
  { status: "proposed", label: "Proposed", color: "border-gray-300" },
  { status: "recruiting", label: "Recruiting", color: "border-blue-300" },
  { status: "active", label: "Active", color: "border-green-300" },
  { status: "reviewing", label: "Reviewing", color: "border-yellow-300" },
  { status: "completed", label: "Completed", color: "border-emerald-300" },
];

const categoryOptions: { value: MetaTaskCategory | "all"; label: string }[] = [
  { value: "all", label: "All Categories" },
  { value: "exploration", label: "Exploration" },
  { value: "integration", label: "Integration" },
  { value: "benchmark", label: "Benchmark" },
  { value: "dataset", label: "Dataset" },
  { value: "replication", label: "Replication" },
  { value: "extension", label: "Extension" },
  { value: "application", label: "Application" },
];

export function CollaborationBoard({
  tasks,
  onJoinTask,
  onViewTask,
  onCreateTask,
  loading = false,
  currentLabId,
}: CollaborationBoardProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("board");
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<MetaTaskCategory | "all">("all");
  const [showCompleted, setShowCompleted] = useState(false);

  // Filter tasks
  const filteredTasks = useMemo(() => {
    return tasks.filter((task) => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesTitle = task.title.toLowerCase().includes(query);
        const matchesDescription = task.description.toLowerCase().includes(query);
        const matchesTags = task.tags.some((tag) =>
          tag.toLowerCase().includes(query)
        );
        if (!matchesTitle && !matchesDescription && !matchesTags) {
          return false;
        }
      }

      // Category filter
      if (categoryFilter !== "all" && task.category !== categoryFilter) {
        return false;
      }

      // Completed filter
      if (!showCompleted && (task.status === "completed" || task.status === "cancelled" || task.status === "failed")) {
        return false;
      }

      return true;
    });
  }, [tasks, searchQuery, categoryFilter, showCompleted]);

  // Group tasks by status for board view
  const tasksByStatus = useMemo(() => {
    const grouped: Record<MetaTaskStatus, (MetaTask | MetaTaskSummary)[]> = {
      proposed: [],
      recruiting: [],
      accepted: [],
      active: [],
      reviewing: [],
      completed: [],
      cancelled: [],
      failed: [],
    };

    for (const task of filteredTasks) {
      grouped[task.status].push(task);
    }

    return grouped;
  }, [filteredTasks]);

  // Stats
  const stats = useMemo(() => {
    const total = tasks.length;
    const active = tasks.filter(
      (t) => t.status === "active" || t.status === "reviewing"
    ).length;
    const open = tasks.filter(
      (t) => t.status === "proposed" || t.status === "recruiting"
    ).length;
    const completed = tasks.filter((t) => t.status === "completed").length;

    return { total, active, open, completed };
  }, [tasks]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Collaboration Board</h2>
          <p className="text-sm text-gray-500">
            {stats.total} tasks ({stats.open} open, {stats.active} active, {stats.completed} completed)
          </p>
        </div>
        <div className="flex items-center gap-2">
          {onCreateTask && (
            <button
              onClick={onCreateTask}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center gap-2"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              New Task
            </button>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4 flex-wrap">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px] max-w-md">
          <input
            type="text"
            placeholder="Search tasks..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        {/* Category Filter */}
        <select
          value={categoryFilter}
          onChange={(e) =>
            setCategoryFilter(e.target.value as MetaTaskCategory | "all")
          }
          className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
        >
          {categoryOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        {/* Show Completed Toggle */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={showCompleted}
            onChange={(e) => setShowCompleted(e.target.checked)}
            className="rounded text-blue-500 focus:ring-blue-500"
          />
          <span className="text-sm text-gray-600">Show completed</span>
        </label>

        {/* View Mode Toggle */}
        <div className="flex items-center border rounded-lg overflow-hidden">
          <button
            onClick={() => setViewMode("board")}
            className={`px-3 py-2 text-sm ${
              viewMode === "board"
                ? "bg-blue-500 text-white"
                : "bg-white text-gray-600 hover:bg-gray-50"
            }`}
            aria-label="Board view"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2"
              />
            </svg>
          </button>
          <button
            onClick={() => setViewMode("list")}
            className={`px-3 py-2 text-sm ${
              viewMode === "list"
                ? "bg-blue-500 text-white"
                : "bg-white text-gray-600 hover:bg-gray-50"
            }`}
            aria-label="List view"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 10h16M4 14h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Board View */}
      {viewMode === "board" && (
        <div className="flex gap-4 overflow-x-auto pb-4">
          {statusColumns.map((column) => {
            const columnTasks = tasksByStatus[column.status];
            if (
              !showCompleted &&
              ["completed", "cancelled", "failed"].includes(column.status) &&
              columnTasks.length === 0
            ) {
              return null;
            }

            return (
              <div
                key={column.status}
                className={`flex-shrink-0 w-80 bg-gray-50 rounded-lg border-t-4 ${column.color}`}
              >
                <div className="p-3 border-b bg-white rounded-t">
                  <div className="flex items-center justify-between">
                    <h3 className="font-medium">{column.label}</h3>
                    <span className="px-2 py-0.5 text-xs bg-gray-200 rounded-full">
                      {columnTasks.length}
                    </span>
                  </div>
                </div>
                <div className="p-2 space-y-2 max-h-[600px] overflow-y-auto">
                  {columnTasks.length === 0 ? (
                    <p className="text-sm text-gray-400 text-center py-4">
                      No tasks
                    </p>
                  ) : (
                    columnTasks.map((task) => (
                      <MetaTaskCard
                        key={task.id}
                        task={task}
                        compact
                        onView={() => onViewTask?.(task.id)}
                        onJoin={() => onJoinTask?.(task.id)}
                      />
                    ))
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* List View */}
      {viewMode === "list" && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredTasks.length === 0 ? (
            <div className="col-span-full text-center py-12">
              <p className="text-gray-500">No tasks match your filters</p>
            </div>
          ) : (
            filteredTasks.map((task) => (
              <MetaTaskCard
                key={task.id}
                task={task}
                onView={() => onViewTask?.(task.id)}
                onJoin={() => onJoinTask?.(task.id)}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default CollaborationBoard;
