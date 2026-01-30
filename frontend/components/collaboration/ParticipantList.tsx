/**
 * ParticipantList
 *
 * Displays the list of participants in a meta-task with their roles,
 * status, expertise, and contribution summary.
 */

"use client";

import React, { useState } from "react";
import {
  Participant,
  ParticipantRole,
  ParticipantStatus,
  ParticipantContribution,
} from "@/lib/meta/collaboration/types";

interface ParticipantListProps {
  participants: Participant[];
  leadId?: string;
  currentLabId?: string;
  onRemoveParticipant?: (participantId: string) => void;
  onUpdateRole?: (participantId: string, newRole: ParticipantRole) => void;
  canManage?: boolean;
}

const roleColors: Record<ParticipantRole, string> = {
  lead: "bg-purple-100 text-purple-700",
  "co-lead": "bg-indigo-100 text-indigo-700",
  contributor: "bg-blue-100 text-blue-700",
  advisor: "bg-green-100 text-green-700",
  reviewer: "bg-yellow-100 text-yellow-700",
  observer: "bg-gray-100 text-gray-600",
};

const roleLabels: Record<ParticipantRole, string> = {
  lead: "Lead",
  "co-lead": "Co-Lead",
  contributor: "Contributor",
  advisor: "Advisor",
  reviewer: "Reviewer",
  observer: "Observer",
};

const statusColors: Record<ParticipantStatus, string> = {
  invited: "text-blue-500",
  pending: "text-yellow-500",
  active: "text-green-500",
  paused: "text-orange-500",
  withdrawn: "text-red-500",
  completed: "text-emerald-500",
};

export function ParticipantList({
  participants,
  leadId,
  currentLabId,
  onRemoveParticipant,
  onUpdateRole,
  canManage = false,
}: ParticipantListProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [roleMenuId, setRoleMenuId] = useState<string | null>(null);

  // Sort: lead first, then by status, then by join date
  const sortedParticipants = [...participants].sort((a, b) => {
    if (a.role === "lead") return -1;
    if (b.role === "lead") return 1;
    if (a.role === "co-lead" && b.role !== "co-lead") return -1;
    if (b.role === "co-lead" && a.role !== "co-lead") return 1;
    if (a.status === "active" && b.status !== "active") return -1;
    if (b.status === "active" && a.status !== "active") return 1;
    return new Date(a.joinedAt).getTime() - new Date(b.joinedAt).getTime();
  });

  const activeCount = participants.filter((p) => p.status === "active").length;

  return (
    <div className="bg-white border rounded-lg">
      {/* Header */}
      <div className="px-4 py-3 border-b flex items-center justify-between">
        <div>
          <h3 className="font-semibold">Participants</h3>
          <p className="text-sm text-gray-500">
            {activeCount} active of {participants.length} total
          </p>
        </div>
      </div>

      {/* List */}
      <div className="divide-y">
        {sortedParticipants.map((participant) => (
          <ParticipantRow
            key={participant.id}
            participant={participant}
            isLead={participant.id === leadId || participant.role === "lead"}
            isCurrentUser={participant.labId === currentLabId}
            isExpanded={expandedId === participant.id}
            onToggleExpand={() =>
              setExpandedId(expandedId === participant.id ? null : participant.id)
            }
            showRoleMenu={roleMenuId === participant.id}
            onToggleRoleMenu={() =>
              setRoleMenuId(roleMenuId === participant.id ? null : participant.id)
            }
            onRemove={
              canManage && participant.role !== "lead" && onRemoveParticipant
                ? () => onRemoveParticipant(participant.id)
                : undefined
            }
            onUpdateRole={
              canManage && participant.role !== "lead" && onUpdateRole
                ? (role) => {
                    onUpdateRole(participant.id, role);
                    setRoleMenuId(null);
                  }
                : undefined
            }
          />
        ))}

        {participants.length === 0 && (
          <div className="p-8 text-center text-gray-500">
            No participants yet
          </div>
        )}
      </div>
    </div>
  );
}

interface ParticipantRowProps {
  participant: Participant;
  isLead: boolean;
  isCurrentUser: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
  showRoleMenu: boolean;
  onToggleRoleMenu: () => void;
  onRemove?: () => void;
  onUpdateRole?: (role: ParticipantRole) => void;
}

function ParticipantRow({
  participant,
  isLead,
  isCurrentUser,
  isExpanded,
  onToggleExpand,
  showRoleMenu,
  onToggleRoleMenu,
  onRemove,
  onUpdateRole,
}: ParticipantRowProps) {
  return (
    <div className={`${isCurrentUser ? "bg-blue-50" : ""}`}>
      {/* Main Row */}
      <div className="p-4 flex items-center gap-4">
        {/* Avatar */}
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center ${
            isLead ? "bg-purple-100" : "bg-gray-100"
          }`}
        >
          <span
            className={`font-medium ${isLead ? "text-purple-600" : "text-gray-600"}`}
          >
            {participant.labName.charAt(0).toUpperCase()}
          </span>
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium truncate">{participant.labName}</span>
            {isCurrentUser && (
              <span className="text-xs text-blue-500">(You)</span>
            )}
            <span className={`w-2 h-2 rounded-full ${statusColors[participant.status]}`} />
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span
              className={`px-2 py-0.5 text-xs rounded-full ${roleColors[participant.role]}`}
            >
              {roleLabels[participant.role]}
            </span>
            {participant.expertise.slice(0, 2).map((exp) => (
              <span
                key={exp}
                className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-600 rounded"
              >
                {exp}
              </span>
            ))}
            {participant.expertise.length > 2 && (
              <span className="text-xs text-gray-400">
                +{participant.expertise.length - 2}
              </span>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="text-right text-sm">
          <div className="text-gray-500">
            {participant.contributions.length} contributions
          </div>
          <div className="text-xs text-gray-400">
            Active {formatRelativeTime(participant.lastActiveAt)}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {onUpdateRole && (
            <div className="relative">
              <button
                onClick={onToggleRoleMenu}
                className="p-1.5 hover:bg-gray-100 rounded transition-colors"
                aria-label="Change role"
              >
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
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
              </button>
              {showRoleMenu && (
                <RoleMenu
                  currentRole={participant.role}
                  onSelect={onUpdateRole}
                  onClose={onToggleRoleMenu}
                />
              )}
            </div>
          )}

          {onRemove && (
            <button
              onClick={onRemove}
              className="p-1.5 hover:bg-red-100 rounded text-red-500 transition-colors"
              aria-label="Remove participant"
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
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>
          )}

          <button
            onClick={onToggleExpand}
            className="p-1.5 hover:bg-gray-100 rounded transition-colors"
            aria-label={isExpanded ? "Collapse" : "Expand"}
          >
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${
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
          </button>
        </div>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-4 pb-4 bg-gray-50 space-y-4">
          {/* Commitment */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Commitment:</span>{" "}
              <span className="font-medium">
                {participant.commitment.hoursPerWeek}h/week for{" "}
                {participant.commitment.durationWeeks} weeks
              </span>
            </div>
            <div>
              <span className="text-gray-500">Joined:</span>{" "}
              <span className="font-medium">
                {new Date(participant.joinedAt).toLocaleDateString()}
              </span>
            </div>
          </div>

          {/* Expertise */}
          {participant.expertise.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Expertise</div>
              <div className="flex flex-wrap gap-1">
                {participant.expertise.map((exp) => (
                  <span
                    key={exp}
                    className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-sm"
                  >
                    {exp}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Resources */}
          {participant.commitment.resources.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Resources</div>
              <div className="flex flex-wrap gap-1">
                {participant.commitment.resources.map((res) => (
                  <span
                    key={res}
                    className="px-2 py-1 bg-green-100 text-green-700 rounded text-sm"
                  >
                    {res}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Recent Contributions */}
          {participant.contributions.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Recent Contributions</div>
              <div className="space-y-2">
                {participant.contributions.slice(-3).map((contrib) => (
                  <ContributionItem key={contrib.id} contribution={contrib} />
                ))}
              </div>
            </div>
          )}

          {/* Assigned Objectives */}
          {participant.assignedObjectives.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">
                Assigned Objectives ({participant.assignedObjectives.length})
              </div>
              <div className="text-sm text-gray-600">
                {participant.assignedObjectives.join(", ")}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface RoleMenuProps {
  currentRole: ParticipantRole;
  onSelect: (role: ParticipantRole) => void;
  onClose: () => void;
}

function RoleMenu({ currentRole, onSelect, onClose }: RoleMenuProps) {
  const roles: ParticipantRole[] = [
    "co-lead",
    "contributor",
    "advisor",
    "reviewer",
    "observer",
  ];

  return (
    <>
      <div className="fixed inset-0 z-10" onClick={onClose} />
      <div className="absolute right-0 top-full mt-1 w-40 bg-white border rounded-lg shadow-lg z-20">
        {roles.map((role) => (
          <button
            key={role}
            onClick={() => onSelect(role)}
            disabled={role === currentRole}
            className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 first:rounded-t-lg last:rounded-b-lg ${
              role === currentRole
                ? "bg-blue-50 text-blue-600"
                : "text-gray-700"
            }`}
          >
            {roleLabels[role]}
          </button>
        ))}
      </div>
    </>
  );
}

function ContributionItem({ contribution }: { contribution: ParticipantContribution }) {
  return (
    <div className="flex items-start gap-2 text-sm">
      <span className="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
        {contribution.type}
      </span>
      <div className="flex-1">
        <span className="text-gray-700">{contribution.description}</span>
        <span className="text-gray-400 text-xs ml-2">
          {formatRelativeTime(contribution.timestamp)}
        </span>
      </div>
      {contribution.impactScore > 0 && (
        <span className="text-xs text-green-600">
          +{contribution.impactScore}
        </span>
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

export default ParticipantList;
