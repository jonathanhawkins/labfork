"use client";

/**
 * SynergyCard Component
 *
 * Displays a synergy proposal between two techniques with scores,
 * justification, and accept/reject actions.
 */

import React, { useState } from "react";

interface TechniqueSummary {
  id: string;
  name: string;
  domains?: string[];
}

interface SynergyScore {
  overall: number;
  components: {
    similarity: number;
    complementarity: number;
    novelty: number;
    feasibility: number;
    impact: number;
  };
  confidence: number;
}

interface CombinationAspect {
  fromA: string;
  fromB: string;
  combination: string;
  benefit: string;
}

interface ExpectedOutcome {
  metric: string;
  baseline: number;
  expected: number;
  unit: string;
  confidence: number;
}

interface SynergyProposalData {
  id: string;
  techniqueA: TechniqueSummary;
  techniqueB: TechniqueSummary;
  score: SynergyScore;
  justification: string;
  combinationAspects: CombinationAspect[];
  expectedOutcomes: ExpectedOutcome[];
  status: string;
  createdAt: string | Date;
  notes?: string;
}

interface SynergyCardProps {
  proposal: SynergyProposalData;
  onAccept?: (id: string, notes?: string) => void;
  onReject?: (id: string, notes?: string) => void;
  onViewDetails?: (id: string) => void;
  className?: string;
  showActions?: boolean;
  compact?: boolean;
}

/**
 * Score indicator component
 */
function ScoreIndicator({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const percentage = Math.round(value * 100);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-600">{label}</span>
        <span className="font-medium">{percentage}%</span>
      </div>
      <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Main SynergyCard component
 */
export function SynergyCard({
  proposal,
  onAccept,
  onReject,
  onViewDetails,
  className = "",
  showActions = true,
  compact = false,
}: SynergyCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [notes, setNotes] = useState("");
  const [showNotes, setShowNotes] = useState(false);

  const overallScore = Math.round(proposal.score.overall * 100);
  const confidence = Math.round(proposal.score.confidence * 100);

  const getScoreColor = (score: number): string => {
    if (score >= 0.7) return "bg-green-500";
    if (score >= 0.5) return "bg-yellow-500";
    return "bg-red-500";
  };

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      pending: "bg-gray-100 text-gray-700",
      accepted: "bg-green-100 text-green-700",
      rejected: "bg-red-100 text-red-700",
      exploring: "bg-blue-100 text-blue-700",
      validated: "bg-emerald-100 text-emerald-700",
      invalidated: "bg-orange-100 text-orange-700",
    };

    return (
      <span
        className={`px-2 py-0.5 text-xs font-medium rounded-full ${styles[status] || styles.pending}`}
      >
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

  const handleAccept = () => {
    onAccept?.(proposal.id, notes || undefined);
  };

  const handleReject = () => {
    onReject?.(proposal.id, notes || undefined);
  };

  return (
    <div
      className={`bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow ${className}`}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            {/* Technique combination visualization */}
            <div className="flex items-center gap-2 mb-2">
              <div className="px-3 py-1.5 bg-blue-50 rounded-lg">
                <span className="text-sm font-medium text-blue-700">
                  {proposal.techniqueA.name}
                </span>
              </div>
              <div className="flex items-center justify-center w-8 h-8 bg-purple-100 rounded-full">
                <svg
                  className="w-4 h-4 text-purple-600"
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
              </div>
              <div className="px-3 py-1.5 bg-green-50 rounded-lg">
                <span className="text-sm font-medium text-green-700">
                  {proposal.techniqueB.name}
                </span>
              </div>
            </div>

            {/* Domains */}
            {!compact && (proposal.techniqueA.domains || proposal.techniqueB.domains) && (
              <div className="flex flex-wrap gap-1 mt-2">
                {[
                  ...(proposal.techniqueA.domains || []),
                  ...(proposal.techniqueB.domains || []),
                ]
                  .filter((d, i, arr) => arr.indexOf(d) === i)
                  .map((domain) => (
                    <span
                      key={domain}
                      className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded"
                    >
                      {domain}
                    </span>
                  ))}
              </div>
            )}
          </div>

          {/* Score and status */}
          <div className="flex flex-col items-end gap-2">
            <div
              className={`flex items-center justify-center w-14 h-14 rounded-full ${getScoreColor(proposal.score.overall)}`}
            >
              <span className="text-lg font-bold text-white">
                {overallScore}
              </span>
            </div>
            {getStatusBadge(proposal.status)}
          </div>
        </div>
      </div>

      {/* Justification */}
      <div className="px-4 py-3 bg-gray-50">
        <p className="text-sm text-gray-700">{proposal.justification}</p>
        <div className="mt-2 text-xs text-gray-500">
          Confidence: {confidence}%
        </div>
      </div>

      {/* Score breakdown (collapsible) */}
      {!compact && (
        <div className="px-4 py-3 border-t border-gray-100">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center justify-between w-full text-sm font-medium text-gray-700 hover:text-gray-900"
          >
            <span>Score Breakdown</span>
            <svg
              className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}
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

          {expanded && (
            <div className="mt-3 space-y-2">
              <ScoreIndicator
                label="Similarity"
                value={proposal.score.components.similarity}
                color="bg-blue-500"
              />
              <ScoreIndicator
                label="Complementarity"
                value={proposal.score.components.complementarity}
                color="bg-purple-500"
              />
              <ScoreIndicator
                label="Novelty"
                value={proposal.score.components.novelty}
                color="bg-yellow-500"
              />
              <ScoreIndicator
                label="Feasibility"
                value={proposal.score.components.feasibility}
                color="bg-green-500"
              />
              <ScoreIndicator
                label="Impact"
                value={proposal.score.components.impact}
                color="bg-red-500"
              />
            </div>
          )}
        </div>
      )}

      {/* Combination aspects */}
      {expanded && proposal.combinationAspects.length > 0 && (
        <div className="px-4 py-3 border-t border-gray-100">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Combination Aspects
          </h4>
          <div className="space-y-2">
            {proposal.combinationAspects.map((aspect, i) => (
              <div
                key={i}
                className="p-2 bg-purple-50 rounded text-sm"
              >
                <div className="font-medium text-purple-700">
                  {aspect.combination}
                </div>
                <div className="text-purple-600 text-xs mt-1">
                  {aspect.benefit}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expected outcomes */}
      {expanded && proposal.expectedOutcomes.length > 0 && (
        <div className="px-4 py-3 border-t border-gray-100">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Expected Outcomes
          </h4>
          <div className="grid grid-cols-2 gap-2">
            {proposal.expectedOutcomes.map((outcome, i) => (
              <div
                key={i}
                className="p-2 bg-green-50 rounded text-sm"
              >
                <div className="text-gray-600">{outcome.metric}</div>
                <div className="font-medium text-green-700">
                  {outcome.baseline.toFixed(1)} → {outcome.expected.toFixed(1)}{" "}
                  {outcome.unit}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      {showActions && proposal.status === "pending" && (
        <div className="px-4 py-3 border-t border-gray-100">
          {showNotes && (
            <div className="mb-3">
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add notes (optional)..."
                className="w-full px-3 py-2 text-sm border border-gray-200 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={2}
              />
            </div>
          )}

          <div className="flex gap-2">
            <button
              onClick={() => setShowNotes(!showNotes)}
              className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded transition-colors"
            >
              {showNotes ? "Hide Notes" : "Add Notes"}
            </button>
            <div className="flex-1" />
            <button
              onClick={handleReject}
              className="px-4 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50 rounded transition-colors"
            >
              Reject
            </button>
            <button
              onClick={handleAccept}
              className="px-4 py-1.5 text-sm font-medium text-white bg-green-600 hover:bg-green-700 rounded transition-colors"
            >
              Accept
            </button>
          </div>
        </div>
      )}

      {/* View details link */}
      {onViewDetails && (
        <div className="px-4 py-2 border-t border-gray-100 bg-gray-50">
          <button
            onClick={() => onViewDetails(proposal.id)}
            className="text-sm text-blue-600 hover:text-blue-700 hover:underline"
          >
            View Full Details →
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * Compact synergy card for lists
 */
export function SynergyCardCompact({
  proposal,
  onClick,
  className = "",
}: {
  proposal: SynergyProposalData;
  onClick?: (id: string) => void;
  className?: string;
}) {
  const overallScore = Math.round(proposal.score.overall * 100);

  return (
    <div
      onClick={() => onClick?.(proposal.id)}
      className={`flex items-center gap-3 p-3 bg-white rounded-lg border border-gray-200 hover:border-gray-300 cursor-pointer transition-colors ${className}`}
    >
      <div
        className={`flex items-center justify-center w-10 h-10 rounded-full text-white font-bold text-sm ${
          overallScore >= 70
            ? "bg-green-500"
            : overallScore >= 50
              ? "bg-yellow-500"
              : "bg-red-500"
        }`}
      >
        {overallScore}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1 text-sm">
          <span className="font-medium text-blue-600 truncate">
            {proposal.techniqueA.name}
          </span>
          <span className="text-gray-400">+</span>
          <span className="font-medium text-green-600 truncate">
            {proposal.techniqueB.name}
          </span>
        </div>
        <p className="text-xs text-gray-500 truncate mt-0.5">
          {proposal.justification}
        </p>
      </div>

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
          d="M9 5l7 7-7 7"
        />
      </svg>
    </div>
  );
}

export default SynergyCard;
