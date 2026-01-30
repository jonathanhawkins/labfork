/**
 * MetaTaskProgress
 *
 * Displays detailed progress tracking for a meta-task including
 * objectives, milestones, participant activity, and alerts.
 */

"use client";

import React, { useState } from "react";
import {
  MetaTask,
  JointObjective,
  ObjectiveStatus,
  Participant,
} from "@/lib/meta/collaboration/types";
import {
  ProgressTracker,
  ProgressAlert,
} from "@/lib/meta/collaboration/workflow";

interface MetaTaskProgressProps {
  task: MetaTask;
  tracker?: ProgressTracker;
  onUpdateProgress?: (objectiveId: string, progress: number) => void;
  onAcknowledgeAlert?: (alertId: string) => void;
  isParticipant?: boolean;
}

const statusColors: Record<ObjectiveStatus, string> = {
  pending: "bg-gray-200",
  in_progress: "bg-blue-500",
  blocked: "bg-red-500",
  completed: "bg-green-500",
  deferred: "bg-yellow-500",
};

const healthColors: Record<ProgressTracker["healthStatus"], string> = {
  healthy: "text-green-600",
  at_risk: "text-yellow-600",
  blocked: "text-red-600",
  off_track: "text-orange-600",
};

const alertSeverityColors = {
  info: "bg-blue-50 border-blue-200 text-blue-700",
  warning: "bg-yellow-50 border-yellow-200 text-yellow-700",
  error: "bg-red-50 border-red-200 text-red-700",
};

export function MetaTaskProgress({
  task,
  tracker,
  onUpdateProgress,
  onAcknowledgeAlert,
  isParticipant = false,
}: MetaTaskProgressProps) {
  const [expandedObjective, setExpandedObjective] = useState<string | null>(null);
  const [showAllAlerts, setShowAllAlerts] = useState(false);

  const overallProgress = tracker?.overallProgress ?? calculateProgress(task);
  const healthStatus = tracker?.healthStatus ?? "healthy";

  const alerts = tracker?.alerts ?? [];
  const unacknowledgedAlerts = alerts.filter((a) => !a.acknowledged);
  const displayedAlerts = showAllAlerts
    ? alerts
    : unacknowledgedAlerts.slice(0, 3);

  return (
    <div className="space-y-6">
      {/* Overall Progress */}
      <div className="bg-white border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Overall Progress</h3>
          <span className={`font-medium ${healthColors[healthStatus]}`}>
            {healthStatus.replace("_", " ").toUpperCase()}
          </span>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="w-full bg-gray-200 rounded-full h-4">
              <div
                className="bg-blue-500 h-4 rounded-full transition-all"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>
          <span className="text-2xl font-bold">{overallProgress}%</span>
        </div>

        <div className="grid grid-cols-3 gap-4 mt-4 text-center">
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">
              {task.objectives.filter((o) => o.status === "completed").length}
            </div>
            <div className="text-sm text-gray-500">Completed</div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">
              {task.objectives.filter((o) => o.status === "in_progress").length}
            </div>
            <div className="text-sm text-gray-500">In Progress</div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">
              {task.objectives.filter((o) => o.status === "pending").length}
            </div>
            <div className="text-sm text-gray-500">Pending</div>
          </div>
        </div>
      </div>

      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="bg-white border rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Alerts</h3>
            {alerts.length > 3 && (
              <button
                onClick={() => setShowAllAlerts(!showAllAlerts)}
                className="text-sm text-blue-500 hover:text-blue-600"
              >
                {showAllAlerts ? "Show Less" : `Show All (${alerts.length})`}
              </button>
            )}
          </div>

          <div className="space-y-2">
            {displayedAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`flex items-start justify-between p-3 border rounded-lg ${
                  alertSeverityColors[alert.severity]
                } ${alert.acknowledged ? "opacity-50" : ""}`}
              >
                <div className="flex items-start gap-3">
                  <AlertIcon severity={alert.severity} />
                  <div>
                    <p className="font-medium">{alert.message}</p>
                    <p className="text-xs opacity-75 mt-1">
                      {formatRelativeTime(alert.createdAt)}
                    </p>
                  </div>
                </div>
                {!alert.acknowledged && onAcknowledgeAlert && (
                  <button
                    onClick={() => onAcknowledgeAlert(alert.id)}
                    className="text-xs hover:underline"
                  >
                    Dismiss
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Objectives */}
      <div className="bg-white border rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Objectives</h3>

        <div className="space-y-3">
          {task.objectives.map((objective) => (
            <ObjectiveRow
              key={objective.id}
              objective={objective}
              isExpanded={expandedObjective === objective.id}
              onToggle={() =>
                setExpandedObjective(
                  expandedObjective === objective.id ? null : objective.id
                )
              }
              onUpdateProgress={
                isParticipant && onUpdateProgress
                  ? (progress) => onUpdateProgress(objective.id, progress)
                  : undefined
              }
              participants={task.participants}
            />
          ))}

          {task.objectives.length === 0 && (
            <p className="text-center text-gray-500 py-8">
              No objectives defined yet
            </p>
          )}
        </div>
      </div>

      {/* Participant Activity */}
      {tracker?.participantProgress && (
        <div className="bg-white border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Participant Activity</h3>

          <div className="space-y-3">
            {Object.entries(tracker.participantProgress).map(([id, progress]) => {
              const participant = task.participants.find((p) => p.id === id);
              if (!participant) return null;

              return (
                <div
                  key={id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-blue-600 font-medium">
                        {participant.labName.charAt(0)}
                      </span>
                    </div>
                    <div>
                      <div className="font-medium">{participant.labName}</div>
                      <div className="text-sm text-gray-500">
                        {participant.role} - {progress.contributionsCount} contributions
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm">
                      {progress.completedObjectives}/{progress.assignedObjectives} objectives
                    </div>
                    <div className="text-xs text-gray-500">
                      Last active: {formatRelativeTime(progress.lastActive)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

interface ObjectiveRowProps {
  objective: JointObjective;
  isExpanded: boolean;
  onToggle: () => void;
  onUpdateProgress?: (progress: number) => void;
  participants: Participant[];
}

function ObjectiveRow({
  objective,
  isExpanded,
  onToggle,
  onUpdateProgress,
  participants,
}: ObjectiveRowProps) {
  const [editProgress, setEditProgress] = useState(objective.progress);

  const assignedParticipants = objective.assignedTo
    .map((id) => participants.find((p) => p.id === id))
    .filter(Boolean) as Participant[];

  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span
            className={`w-3 h-3 rounded-full ${statusColors[objective.status]}`}
          />
          <div className="text-left">
            <div className="font-medium">{objective.title}</div>
            <div className="text-sm text-gray-500">
              {objective.priority} priority
              {objective.deadline && ` - Due ${formatDate(objective.deadline)}`}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <span className="font-medium">{objective.progress}%</span>
          </div>
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${
              isExpanded ? "rotate-180" : ""
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4 border-t bg-gray-50 space-y-4">
          <p className="text-gray-600 mt-4">{objective.description}</p>

          {/* Progress Bar */}
          <div>
            <div className="flex items-center justify-between text-sm mb-2">
              <span>Progress</span>
              <span>{objective.progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${statusColors[objective.status]}`}
                style={{ width: `${objective.progress}%` }}
              />
            </div>
          </div>

          {/* Update Progress */}
          {onUpdateProgress && (
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={100}
                value={editProgress}
                onChange={(e) => setEditProgress(parseInt(e.target.value, 10))}
                className="flex-1"
              />
              <span className="w-12 text-center">{editProgress}%</span>
              <button
                onClick={() => onUpdateProgress(editProgress)}
                disabled={editProgress === objective.progress}
                className="px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Update
              </button>
            </div>
          )}

          {/* Assigned Participants */}
          {assignedParticipants.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Assigned To</div>
              <div className="flex flex-wrap gap-2">
                {assignedParticipants.map((p) => (
                  <span
                    key={p.id}
                    className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-sm"
                  >
                    {p.labName}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Milestones */}
          {objective.milestones.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Milestones</div>
              <div className="space-y-2">
                {objective.milestones.map((milestone) => (
                  <div
                    key={milestone.id}
                    className={`flex items-center gap-2 text-sm ${
                      milestone.completed ? "text-green-600" : "text-gray-600"
                    }`}
                  >
                    {milestone.completed ? (
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
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <circle cx="12" cy="12" r="10" strokeWidth={2} />
                      </svg>
                    )}
                    <span>{milestone.title}</span>
                    <span className="text-gray-400 text-xs">
                      Due: {formatDate(milestone.dueDate)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Dependencies */}
          {objective.dependencies.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Dependencies</div>
              <div className="text-sm text-gray-500">
                Depends on: {objective.dependencies.join(", ")}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AlertIcon({ severity }: { severity: ProgressAlert["severity"] }) {
  if (severity === "error") {
    return (
      <svg
        className="w-5 h-5 text-red-500 flex-shrink-0"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    );
  }
  if (severity === "warning") {
    return (
      <svg
        className="w-5 h-5 text-yellow-500 flex-shrink-0"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
    );
  }
  return (
    <svg
      className="w-5 h-5 text-blue-500 flex-shrink-0"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  );
}

function calculateProgress(task: MetaTask): number {
  if (task.objectives.length === 0) return 0;
  const total = task.objectives.reduce((sum, o) => sum + o.progress, 0);
  return Math.round(total / task.objectives.length);
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

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString();
}

export default MetaTaskProgress;
