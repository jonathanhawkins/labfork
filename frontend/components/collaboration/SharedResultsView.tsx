/**
 * SharedResultsView
 *
 * Displays shared results from a meta-task with quality metrics,
 * contributors, and status information.
 */

"use client";

import React, { useState } from "react";
import {
  SharedResult,
  ResultType,
  ResultQuality,
  Participant,
  ParticipantId,
} from "@/lib/meta/collaboration/types";

interface SharedResultsViewProps {
  results: SharedResult[];
  participants: Participant[];
  onApprove?: (resultId: string) => void;
  onReview?: (resultId: string, quality: Partial<ResultQuality>) => void;
  onMerge?: (resultIds: string[], title: string, description: string) => void;
  canReview?: boolean;
  canMerge?: boolean;
}

const statusColors = {
  draft: "bg-gray-100 text-gray-600",
  submitted: "bg-blue-100 text-blue-700",
  reviewed: "bg-yellow-100 text-yellow-700",
  approved: "bg-green-100 text-green-700",
  published: "bg-emerald-100 text-emerald-700",
};

const typeIcons: Record<ResultType, string> = {
  finding: "lightbulb",
  technique: "cog",
  dataset: "database",
  model: "cube",
  benchmark: "chart-bar",
  paper: "document-text",
  prototype: "code",
};

const typeLabels: Record<ResultType, string> = {
  finding: "Finding",
  technique: "Technique",
  dataset: "Dataset",
  model: "Model",
  benchmark: "Benchmark",
  paper: "Paper",
  prototype: "Prototype",
};

export function SharedResultsView({
  results,
  participants,
  onApprove,
  onReview,
  onMerge,
  canReview = false,
  canMerge = false,
}: SharedResultsViewProps) {
  const [selectedForMerge, setSelectedForMerge] = useState<Set<string>>(new Set());
  const [showMergeDialog, setShowMergeDialog] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const toggleMergeSelection = (resultId: string) => {
    const newSelection = new Set(selectedForMerge);
    if (newSelection.has(resultId)) {
      newSelection.delete(resultId);
    } else {
      newSelection.add(resultId);
    }
    setSelectedForMerge(newSelection);
  };

  const getParticipantName = (id: ParticipantId): string => {
    const participant = participants.find((p) => p.id === id);
    return participant?.labName || id;
  };

  // Group by objective
  const resultsByObjective = results.reduce((acc, result) => {
    const key = result.objectiveId;
    if (!acc[key]) acc[key] = [];
    acc[key].push(result);
    return acc;
  }, {} as Record<string, SharedResult[]>);

  // Stats
  const stats = {
    total: results.length,
    approved: results.filter((r) => r.status === "approved" || r.status === "published").length,
    pending: results.filter((r) => r.status === "draft" || r.status === "submitted").length,
    avgQuality: results.length > 0
      ? Math.round(
          results.reduce((sum, r) => sum + r.quality.overallScore, 0) / results.length * 100
        ) / 100
      : 0,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white border rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Shared Results</h3>
          {canMerge && selectedForMerge.size >= 2 && (
            <button
              onClick={() => setShowMergeDialog(true)}
              className="px-3 py-1.5 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors"
            >
              Merge Selected ({selectedForMerge.size})
            </button>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-4 text-center">
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">{stats.total}</div>
            <div className="text-sm text-gray-500">Total</div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{stats.approved}</div>
            <div className="text-sm text-gray-500">Approved</div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">{stats.pending}</div>
            <div className="text-sm text-gray-500">Pending</div>
          </div>
          <div className="p-3 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold">{stats.avgQuality}</div>
            <div className="text-sm text-gray-500">Avg Quality</div>
          </div>
        </div>
      </div>

      {/* Results List */}
      {Object.entries(resultsByObjective).map(([objectiveId, objResults]) => (
        <div key={objectiveId} className="bg-white border rounded-lg">
          <div className="px-4 py-3 border-b bg-gray-50">
            <h4 className="font-medium text-sm text-gray-600">
              Objective: {objectiveId}
            </h4>
          </div>
          <div className="divide-y">
            {objResults.map((result) => (
              <ResultRow
                key={result.id}
                result={result}
                isExpanded={expandedId === result.id}
                onToggleExpand={() =>
                  setExpandedId(expandedId === result.id ? null : result.id)
                }
                isSelectedForMerge={selectedForMerge.has(result.id)}
                onToggleMerge={canMerge ? () => toggleMergeSelection(result.id) : undefined}
                getParticipantName={getParticipantName}
                onApprove={canReview && onApprove ? () => onApprove(result.id) : undefined}
                onReview={canReview && onReview ? (q) => onReview(result.id, q) : undefined}
              />
            ))}
          </div>
        </div>
      ))}

      {results.length === 0 && (
        <div className="bg-white border rounded-lg p-12 text-center text-gray-500">
          No results submitted yet
        </div>
      )}

      {/* Merge Dialog */}
      {showMergeDialog && onMerge && (
        <MergeDialog
          selectedIds={Array.from(selectedForMerge)}
          results={results.filter((r) => selectedForMerge.has(r.id))}
          onMerge={(title, description) => {
            onMerge(Array.from(selectedForMerge), title, description);
            setSelectedForMerge(new Set());
            setShowMergeDialog(false);
          }}
          onClose={() => setShowMergeDialog(false)}
        />
      )}
    </div>
  );
}

interface ResultRowProps {
  result: SharedResult;
  isExpanded: boolean;
  onToggleExpand: () => void;
  isSelectedForMerge: boolean;
  onToggleMerge?: () => void;
  getParticipantName: (id: ParticipantId) => string;
  onApprove?: () => void;
  onReview?: (quality: Partial<ResultQuality>) => void;
}

function ResultRow({
  result,
  isExpanded,
  onToggleExpand,
  isSelectedForMerge,
  onToggleMerge,
  getParticipantName,
  onApprove,
  onReview,
}: ResultRowProps) {
  const [reviewScore, setReviewScore] = useState(result.quality.overallScore * 100);

  return (
    <div className={isSelectedForMerge ? "bg-blue-50" : ""}>
      {/* Main Row */}
      <div className="p-4 flex items-center gap-4">
        {onToggleMerge && (
          <input
            type="checkbox"
            checked={isSelectedForMerge}
            onChange={onToggleMerge}
            className="rounded text-blue-500"
          />
        )}

        {/* Type Icon */}
        <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
          <TypeIcon type={result.type} />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium truncate">{result.title}</span>
            <span className={`px-2 py-0.5 text-xs rounded-full ${statusColors[result.status]}`}>
              {result.status}
            </span>
            {result.mergedFrom && (
              <span className="text-xs text-purple-500">Merged</span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-1 text-sm text-gray-500">
            <span>{typeLabels[result.type]}</span>
            <span>-</span>
            <span>
              {result.contributors.map(getParticipantName).join(", ")}
            </span>
          </div>
        </div>

        {/* Quality Score */}
        <div className="text-right">
          <QualityBadge quality={result.quality} />
          <div className="text-xs text-gray-400 mt-1">
            {formatRelativeTime(result.updatedAt)}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {onApprove && result.status === "reviewed" && (
            <button
              onClick={onApprove}
              className="px-2 py-1 bg-green-500 text-white text-xs rounded hover:bg-green-600 transition-colors"
            >
              Approve
            </button>
          )}
          <button
            onClick={onToggleExpand}
            className="p-1.5 hover:bg-gray-100 rounded transition-colors"
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
          {/* Description */}
          <div>
            <div className="text-sm font-medium mb-1">Description</div>
            <p className="text-gray-600">{result.description}</p>
          </div>

          {/* Summary */}
          <div>
            <div className="text-sm font-medium mb-1">Summary</div>
            <p className="text-gray-600">{result.data.summary}</p>
          </div>

          {/* Metrics */}
          {result.data.metrics && Object.keys(result.data.metrics).length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Metrics</div>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(result.data.metrics).map(([key, value]) => (
                  <div key={key} className="p-2 bg-white rounded border">
                    <div className="text-xs text-gray-500">{key}</div>
                    <div className="font-medium">{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quality Breakdown */}
          <div>
            <div className="text-sm font-medium mb-2">Quality Assessment</div>
            <div className="grid grid-cols-5 gap-2">
              {(["reproducibility", "novelty", "significance", "completeness"] as const).map(
                (metric) => (
                  <div key={metric} className="text-center">
                    <div className="text-lg font-bold">
                      {Math.round(result.quality[metric] * 100)}
                    </div>
                    <div className="text-xs text-gray-500 capitalize">{metric}</div>
                  </div>
                )
              )}
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">
                  {Math.round(result.quality.overallScore * 100)}
                </div>
                <div className="text-xs text-gray-500">Overall</div>
              </div>
            </div>
          </div>

          {/* Review Notes */}
          {result.quality.reviewNotes.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Review Notes</div>
              <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                {result.quality.reviewNotes.map((note, i) => (
                  <li key={i}>{note}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Artifacts */}
          {result.data.artifacts.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">Artifacts</div>
              <div className="flex flex-wrap gap-2">
                {result.data.artifacts.map((artifact) => (
                  <a
                    key={artifact.id}
                    href={artifact.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-3 py-1.5 bg-white border rounded-lg text-sm hover:border-blue-300 transition-colors flex items-center gap-2"
                  >
                    <span>{artifact.name}</span>
                    <span className="text-xs text-gray-400">{artifact.type}</span>
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* References */}
          {result.data.references.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">References</div>
              <ul className="text-sm text-gray-600 space-y-1">
                {result.data.references.map((ref, i) => (
                  <li key={i} className="truncate">
                    {ref}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Quick Review */}
          {onReview && result.status === "submitted" && (
            <div className="pt-4 border-t">
              <div className="text-sm font-medium mb-2">Quick Review</div>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={reviewScore}
                  onChange={(e) => setReviewScore(parseInt(e.target.value, 10))}
                  className="flex-1"
                />
                <span className="w-12 text-center">{reviewScore}%</span>
                <button
                  onClick={() =>
                    onReview({
                      overallScore: reviewScore / 100,
                      reproducibility: reviewScore / 100,
                      novelty: reviewScore / 100,
                      significance: reviewScore / 100,
                      completeness: reviewScore / 100,
                    })
                  }
                  className="px-3 py-1 bg-yellow-500 text-white text-sm rounded hover:bg-yellow-600"
                >
                  Submit Review
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TypeIcon({ type }: { type: ResultType }) {
  const icons: Record<ResultType, React.ReactNode> = {
    finding: (
      <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    technique: (
      <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
    dataset: (
      <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
      </svg>
    ),
    model: (
      <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
      </svg>
    ),
    benchmark: (
      <svg className="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    paper: (
      <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    prototype: (
      <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
  };

  return icons[type];
}

function QualityBadge({ quality }: { quality: ResultQuality }) {
  const score = Math.round(quality.overallScore * 100);
  let color = "bg-gray-100 text-gray-600";

  if (score >= 80) color = "bg-green-100 text-green-700";
  else if (score >= 60) color = "bg-yellow-100 text-yellow-700";
  else if (score >= 40) color = "bg-orange-100 text-orange-700";
  else if (score > 0) color = "bg-red-100 text-red-700";

  return (
    <span className={`px-2 py-1 rounded text-sm font-medium ${color}`}>
      {score > 0 ? `${score}%` : "Unrated"}
    </span>
  );
}

interface MergeDialogProps {
  selectedIds: string[];
  results: SharedResult[];
  onMerge: (title: string, description: string) => void;
  onClose: () => void;
}

function MergeDialog({ selectedIds, results, onMerge, onClose }: MergeDialogProps) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative bg-white rounded-lg shadow-xl max-w-lg w-full mx-4 p-6">
        <h3 className="text-lg font-semibold mb-4">Merge Results</h3>

        <div className="mb-4">
          <p className="text-sm text-gray-600 mb-2">
            Merging {selectedIds.length} results:
          </p>
          <ul className="text-sm list-disc list-inside">
            {results.map((r) => (
              <li key={r.id}>{r.title}</li>
            ))}
          </ul>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Merged Result Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter title for merged result..."
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe the merged result..."
              rows={3}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={() => onMerge(title, description)}
            disabled={!title.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            Merge
          </button>
        </div>
      </div>
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

export default SharedResultsView;
