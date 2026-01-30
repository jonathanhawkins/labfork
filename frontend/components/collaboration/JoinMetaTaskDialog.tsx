/**
 * JoinMetaTaskDialog
 *
 * Modal dialog for joining a meta-task with commitment details.
 */

"use client";

import React, { useState } from "react";
import {
  MetaTask,
  ParticipantRole,
  ParticipantCommitment,
} from "@/lib/meta/collaboration/types";

interface JoinMetaTaskDialogProps {
  task: MetaTask;
  isOpen: boolean;
  onClose: () => void;
  onJoin: (data: JoinData) => Promise<void>;
  labId: string;
  labName: string;
  labExpertise?: string[];
}

export interface JoinData {
  labId: string;
  labName: string;
  role: ParticipantRole;
  expertise: string[];
  commitment: ParticipantCommitment;
}

const availableRoles: { value: ParticipantRole; label: string; description: string }[] = [
  {
    value: "contributor",
    label: "Contributor",
    description: "Actively contribute to objectives and deliverables",
  },
  {
    value: "advisor",
    label: "Advisor",
    description: "Provide guidance and expertise when needed",
  },
  {
    value: "reviewer",
    label: "Reviewer",
    description: "Review work and provide feedback",
  },
  {
    value: "observer",
    label: "Observer",
    description: "Follow progress without active participation",
  },
];

export function JoinMetaTaskDialog({
  task,
  isOpen,
  onClose,
  onJoin,
  labId,
  labName,
  labExpertise = [],
}: JoinMetaTaskDialogProps) {
  const [role, setRole] = useState<ParticipantRole>("contributor");
  const [expertise, setExpertise] = useState<string[]>(labExpertise);
  const [newExpertise, setNewExpertise] = useState("");
  const [hoursPerWeek, setHoursPerWeek] = useState(5);
  const [durationWeeks, setDurationWeeks] = useState(4);
  const [resources, setResources] = useState<string[]>([]);
  const [newResource, setNewResource] = useState("");
  const [responsibilities, setResponsibilities] = useState<string[]>([]);
  const [newResponsibility, setNewResponsibility] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleAddExpertise = () => {
    if (newExpertise.trim() && !expertise.includes(newExpertise.trim())) {
      setExpertise([...expertise, newExpertise.trim()]);
      setNewExpertise("");
    }
  };

  const handleRemoveExpertise = (exp: string) => {
    setExpertise(expertise.filter((e) => e !== exp));
  };

  const handleAddResource = () => {
    if (newResource.trim() && !resources.includes(newResource.trim())) {
      setResources([...resources, newResource.trim()]);
      setNewResource("");
    }
  };

  const handleRemoveResource = (res: string) => {
    setResources(resources.filter((r) => r !== res));
  };

  const handleAddResponsibility = () => {
    if (newResponsibility.trim() && !responsibilities.includes(newResponsibility.trim())) {
      setResponsibilities([...responsibilities, newResponsibility.trim()]);
      setNewResponsibility("");
    }
  };

  const handleRemoveResponsibility = (resp: string) => {
    setResponsibilities(responsibilities.filter((r) => r !== resp));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await onJoin({
        labId,
        labName,
        role,
        expertise,
        commitment: {
          hoursPerWeek,
          durationWeeks,
          resources,
          responsibilities,
        },
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to join task");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Dialog */}
      <div className="relative bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold">Join Meta-Task</h2>
            <p className="text-sm text-gray-500">{task.title}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            aria-label="Close"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Error */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              {error}
            </div>
          )}

          {/* Lab Info */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">
              Joining as <span className="font-medium">{labName}</span>
            </p>
          </div>

          {/* Role Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Role
            </label>
            <div className="grid gap-2">
              {availableRoles.map((r) => (
                <label
                  key={r.value}
                  className={`flex items-start p-3 border rounded-lg cursor-pointer transition-colors ${
                    role === r.value
                      ? "border-blue-500 bg-blue-50"
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                >
                  <input
                    type="radio"
                    name="role"
                    value={r.value}
                    checked={role === r.value}
                    onChange={(e) => setRole(e.target.value as ParticipantRole)}
                    className="mt-0.5 text-blue-500"
                  />
                  <div className="ml-3">
                    <span className="font-medium">{r.label}</span>
                    <p className="text-sm text-gray-500">{r.description}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Expertise */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Expertise
            </label>
            <div className="flex flex-wrap gap-2 mb-2">
              {expertise.map((exp) => (
                <span
                  key={exp}
                  className="px-2 py-1 bg-blue-100 text-blue-700 rounded-full text-sm flex items-center gap-1"
                >
                  {exp}
                  <button
                    type="button"
                    onClick={() => handleRemoveExpertise(exp)}
                    className="hover:text-blue-900"
                  >
                    &times;
                  </button>
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={newExpertise}
                onChange={(e) => setNewExpertise(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), handleAddExpertise())}
                placeholder="Add expertise..."
                className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="button"
                onClick={handleAddExpertise}
                className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                Add
              </button>
            </div>
            {/* Suggested from task requirements */}
            {task.requirements.requiredExpertise.length > 0 && (
              <div className="mt-2">
                <p className="text-xs text-gray-500 mb-1">Required by task:</p>
                <div className="flex flex-wrap gap-1">
                  {task.requirements.requiredExpertise
                    .filter((e) => !expertise.includes(e))
                    .map((exp) => (
                      <button
                        key={exp}
                        type="button"
                        onClick={() => setExpertise([...expertise, exp])}
                        className="px-2 py-0.5 text-xs bg-gray-100 hover:bg-gray-200 rounded transition-colors"
                      >
                        + {exp}
                      </button>
                    ))}
                </div>
              </div>
            )}
          </div>

          {/* Commitment */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hours per Week
              </label>
              <input
                type="number"
                min={1}
                max={40}
                value={hoursPerWeek}
                onChange={(e) => setHoursPerWeek(parseInt(e.target.value, 10))}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Duration (weeks)
              </label>
              <input
                type="number"
                min={1}
                max={52}
                value={durationWeeks}
                onChange={(e) => setDurationWeeks(parseInt(e.target.value, 10))}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Resources */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Resources You Can Provide (optional)
            </label>
            <div className="flex flex-wrap gap-2 mb-2">
              {resources.map((res) => (
                <span
                  key={res}
                  className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-sm flex items-center gap-1"
                >
                  {res}
                  <button
                    type="button"
                    onClick={() => handleRemoveResource(res)}
                    className="hover:text-green-900"
                  >
                    &times;
                  </button>
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={newResource}
                onChange={(e) => setNewResource(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), handleAddResource())}
                placeholder="e.g., GPU compute, datasets..."
                className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="button"
                onClick={handleAddResource}
                className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                Add
              </button>
            </div>
          </div>

          {/* Responsibilities */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Specific Responsibilities (optional)
            </label>
            <div className="flex flex-wrap gap-2 mb-2">
              {responsibilities.map((resp) => (
                <span
                  key={resp}
                  className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-sm flex items-center gap-1"
                >
                  {resp}
                  <button
                    type="button"
                    onClick={() => handleRemoveResponsibility(resp)}
                    className="hover:text-purple-900"
                  >
                    &times;
                  </button>
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={newResponsibility}
                onChange={(e) => setNewResponsibility(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), handleAddResponsibility())}
                placeholder="e.g., Lead data preprocessing..."
                className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="button"
                onClick={handleAddResponsibility}
                className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                Add
              </button>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {loading && (
                <svg
                  className="animate-spin h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              )}
              Join Task
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default JoinMetaTaskDialog;
