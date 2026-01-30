/**
 * MetaTaskCard
 *
 * Displays a meta-task summary with status, participants, and progress.
 * Used in the CollaborationBoard and task listings.
 */

import React from "react";
import {
  MetaTask,
  MetaTaskStatus,
  MetaTaskCategory,
} from "@/lib/meta/collaboration/types";

interface MetaTaskCardProps {
  task: MetaTask | MetaTaskSummary;
  onJoin?: () => void;
  onView?: () => void;
  showActions?: boolean;
  compact?: boolean;
}

// Summary version for list views
export interface MetaTaskSummary {
  id: string;
  title: string;
  description: string;
  category: MetaTaskCategory;
  status: MetaTaskStatus;
  leadLab: string;
  participantCount: number;
  objectiveCount: number;
  completedObjectives: number;
  visibility: "public" | "private" | "invite_only";
  tags: string[];
  domains: string[];
  createdAt: string;
  updatedAt: string;
}

function isFullTask(task: MetaTask | MetaTaskSummary): task is MetaTask {
  return "participants" in task && Array.isArray(task.participants);
}

const statusColors: Record<MetaTaskStatus, string> = {
  proposed: "bg-gray-100 text-gray-700",
  recruiting: "bg-blue-100 text-blue-700",
  accepted: "bg-indigo-100 text-indigo-700",
  active: "bg-green-100 text-green-700",
  reviewing: "bg-yellow-100 text-yellow-700",
  completed: "bg-emerald-100 text-emerald-700",
  cancelled: "bg-red-100 text-red-700",
  failed: "bg-red-100 text-red-700",
};

const categoryIcons: Record<MetaTaskCategory, string> = {
  exploration: "compass",
  integration: "puzzle-piece",
  benchmark: "chart-bar",
  dataset: "database",
  replication: "copy",
  extension: "arrows-expand",
  application: "code",
};

const categoryLabels: Record<MetaTaskCategory, string> = {
  exploration: "Exploration",
  integration: "Integration",
  benchmark: "Benchmark",
  dataset: "Dataset",
  replication: "Replication",
  extension: "Extension",
  application: "Application",
};

export function MetaTaskCard({
  task,
  onJoin,
  onView,
  showActions = true,
  compact = false,
}: MetaTaskCardProps) {
  const participantCount = isFullTask(task)
    ? task.participants.filter((p) => p.status === "active").length
    : task.participantCount;

  const objectiveCount = isFullTask(task)
    ? task.objectives.length
    : task.objectiveCount;

  const completedObjectives = isFullTask(task)
    ? task.objectives.filter((o) => o.status === "completed").length
    : task.completedObjectives;

  const progress =
    objectiveCount > 0
      ? Math.round((completedObjectives / objectiveCount) * 100)
      : 0;

  const leadLab = isFullTask(task) ? task.lead.labName : task.leadLab;

  const canJoin =
    task.status === "proposed" ||
    task.status === "recruiting";

  if (compact) {
    return (
      <div
        className="p-3 border rounded-lg hover:border-blue-300 cursor-pointer transition-colors"
        onClick={onView}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && onView?.()}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`px-2 py-0.5 text-xs rounded-full ${statusColors[task.status]}`}
            >
              {task.status}
            </span>
            <h4 className="font-medium text-sm truncate max-w-[200px]">
              {task.title}
            </h4>
          </div>
          <span className="text-xs text-gray-500">
            {participantCount} participants
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border rounded-lg shadow-sm hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span
                className={`px-2 py-0.5 text-xs rounded-full ${statusColors[task.status]}`}
              >
                {task.status}
              </span>
              <span className="text-xs text-gray-500">
                {categoryLabels[task.category]}
              </span>
              {task.visibility !== "public" && (
                <span className="text-xs text-gray-400">
                  {task.visibility === "private" ? "Private" : "Invite Only"}
                </span>
              )}
            </div>
            <h3 className="font-semibold text-lg">{task.title}</h3>
            <p className="text-gray-600 text-sm mt-1 line-clamp-2">
              {task.description}
            </p>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-4">
        {/* Lead & Participants */}
        <div className="flex items-center justify-between text-sm">
          <div>
            <span className="text-gray-500">Lead:</span>{" "}
            <span className="font-medium">{leadLab}</span>
          </div>
          <div className="flex items-center gap-1">
            <svg
              className="w-4 h-4 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
              />
            </svg>
            <span>{participantCount} participants</span>
          </div>
        </div>

        {/* Progress */}
        {objectiveCount > 0 && (
          <div>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-gray-500">Progress</span>
              <span className="font-medium">
                {completedObjectives}/{objectiveCount} objectives ({progress}%)
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Tags */}
        {task.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {task.tags.slice(0, 5).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded"
              >
                {tag}
              </span>
            ))}
            {task.tags.length > 5 && (
              <span className="text-xs text-gray-400">
                +{task.tags.length - 5} more
              </span>
            )}
          </div>
        )}

        {/* Domains */}
        {task.domains.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {task.domains.map((domain) => (
              <span
                key={domain}
                className="px-2 py-0.5 text-xs bg-blue-50 text-blue-600 rounded"
              >
                {domain}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Footer Actions */}
      {showActions && (
        <div className="px-4 py-3 bg-gray-50 border-t flex items-center justify-between rounded-b-lg">
          <span className="text-xs text-gray-400">
            Updated {formatRelativeTime(task.updatedAt)}
          </span>
          <div className="flex gap-2">
            {onView && (
              <button
                onClick={onView}
                className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
              >
                View Details
              </button>
            )}
            {canJoin && onJoin && (
              <button
                onClick={onJoin}
                className="px-3 py-1.5 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
              >
                Join Task
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

export default MetaTaskCard;
