"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Lightbulb,
  Sparkles,
  Bug,
  CheckCircle2,
  Clock,
  ArrowUpCircle,
  CircleDot,
  RefreshCw,
  ExternalLink,
  ThumbsUp,
} from "lucide-react";

interface Suggestion {
  id: string;
  title: string;
  description: string;
  category: "feature" | "improvement" | "bug";
  status: "pending" | "approved" | "in-progress" | "done";
  votes: number;
  submittedAt: string;
  taskId?: string;
}

interface SuggestionListProps {
  showPending?: boolean; // For admin view
  compact?: boolean;
  limit?: number;
  onRefresh?: () => void;
  allowVoting?: boolean;
}

const STATUS_CONFIG = {
  pending: {
    label: "Pending Review",
    icon: Clock,
    color: "text-yellow-400",
    bgColor: "bg-yellow-400/10",
    borderColor: "border-yellow-400/30",
  },
  approved: {
    label: "Approved",
    icon: CheckCircle2,
    color: "text-green-400",
    bgColor: "bg-green-400/10",
    borderColor: "border-green-400/30",
  },
  "in-progress": {
    label: "In Progress",
    icon: CircleDot,
    color: "text-blue-400",
    bgColor: "bg-blue-400/10",
    borderColor: "border-blue-400/30",
  },
  done: {
    label: "Completed",
    icon: CheckCircle2,
    color: "text-foreground",
    bgColor: "bg-foreground/10",
    borderColor: "border-foreground/30",
  },
};

const CATEGORY_CONFIG = {
  feature: {
    label: "Feature",
    icon: Sparkles,
    color: "text-purple-400",
  },
  improvement: {
    label: "Improvement",
    icon: ArrowUpCircle,
    color: "text-cyan-400",
  },
  bug: {
    label: "Bug Fix",
    icon: Bug,
    color: "text-red-400",
  },
};

export function SuggestionList({
  showPending = false,
  compact = false,
  limit = 20,
  onRefresh,
  allowVoting = true,
}: SuggestionListProps) {
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [selectedStatus, setSelectedStatus] = useState<string | null>(null);
  const [votingId, setVotingId] = useState<string | null>(null);
  const [votedIds, setVotedIds] = useState<Set<string>>(new Set());

  const handleVote = useCallback(async (suggestionId: string) => {
    if (votingId || votedIds.has(suggestionId)) return;

    setVotingId(suggestionId);
    try {
      const response = await fetch("/api/suggestions/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: suggestionId }),
      });

      if (!response.ok) {
        const data = await response.json();
        if (data.error?.includes("already voted")) {
          // Mark as voted locally
          setVotedIds(prev => {
            const newSet = new Set(prev);
            newSet.add(suggestionId);
            return newSet;
          });
        }
        return;
      }

      const data = await response.json();

      // Update local state
      setSuggestions(prev =>
        prev.map(s =>
          s.id === suggestionId ? { ...s, votes: data.votes } : s
        )
      );
      setVotedIds(prev => {
        const newSet = new Set(prev);
        newSet.add(suggestionId);
        return newSet;
      });
    } catch (err) {
      console.error("Vote failed:", err);
    } finally {
      setVotingId(null);
    }
  }, [votingId, votedIds]);

  const fetchSuggestions = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (selectedStatus) params.set("status", selectedStatus);
      if (showPending) params.set("status", "pending");
      params.set("limit", String(limit));

      const response = await fetch(`/api/suggestions?${params}`);

      if (!response.ok) {
        throw new Error("Failed to fetch suggestions");
      }

      const data = await response.json();
      setSuggestions(data.suggestions || []);
      setLastUpdated(new Date(data.lastUpdated));
    } catch (err) {
      setError("Unable to load suggestions");
      setSuggestions([]);
    } finally {
      setIsLoading(false);
    }
  }, [selectedStatus, showPending, limit]);

  useEffect(() => {
    fetchSuggestions();
  }, [fetchSuggestions]);

  const handleRefresh = useCallback(() => {
    fetchSuggestions();
    onRefresh?.();
  }, [fetchSuggestions, onRefresh]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffHours < 1) return "Just now";
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  // Group suggestions by status
  const groupedSuggestions = suggestions.reduce(
    (acc, s) => {
      if (!acc[s.status]) acc[s.status] = [];
      acc[s.status].push(s);
      return acc;
    },
    {} as Record<string, Suggestion[]>
  );

  if (isLoading) {
    return (
      <div className="text-sm text-muted-foreground text-center py-8">
        Loading suggestions...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-sm text-red-400 text-center py-8">{error}</div>
    );
  }

  if (suggestions.length === 0) {
    return (
      <div className="text-center py-8">
        <Lightbulb className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
        <p className="text-sm text-muted-foreground">No suggestions yet</p>
        <p className="text-xs text-foreground-subtle mt-1">
          Be the first to suggest a feature!
        </p>
      </div>
    );
  }

  if (compact) {
    // Compact view - simple list
    return (
      <div className="space-y-2">
        {suggestions.slice(0, 5).map((suggestion) => {
          const CategoryIcon = CATEGORY_CONFIG[suggestion.category].icon;
          const statusConfig = STATUS_CONFIG[suggestion.status];

          return (
            <div
              key={suggestion.id}
              className={`p-2 rounded border ${statusConfig.borderColor} ${statusConfig.bgColor}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <CategoryIcon
                    className={`w-3 h-3 flex-shrink-0 ${
                      CATEGORY_CONFIG[suggestion.category].color
                    }`}
                  />
                  <span className="text-sm text-foreground truncate">
                    {suggestion.title}
                  </span>
                </div>
                <span className={`text-xs ${statusConfig.color} flex-shrink-0`}>
                  {statusConfig.label}
                </span>
              </div>
            </div>
          );
        })}
        {suggestions.length > 5 && (
          <div className="text-xs text-muted-foreground text-center pt-1">
            +{suggestions.length - 5} more suggestions
          </div>
        )}
      </div>
    );
  }

  // Full view with grouping
  return (
    <div className="space-y-6">
      {/* Header with filters */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            {suggestions.length} suggestion{suggestions.length !== 1 ? "s" : ""}
          </span>
        </div>
        <button
          onClick={handleRefresh}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <RefreshCw className={`w-3 h-3 ${isLoading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Status filter */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        <button
          onClick={() => setSelectedStatus(null)}
          className={`px-3 py-1 text-xs rounded border transition-colors flex-shrink-0 ${
            !selectedStatus
              ? "border-foreground text-foreground"
              : "border-border text-muted-foreground hover:border-foreground-muted"
          }`}
        >
          All
        </button>
        {(["approved", "in-progress", "done"] as const).map((status) => {
          const config = STATUS_CONFIG[status];
          return (
            <button
              key={status}
              onClick={() => setSelectedStatus(status)}
              className={`px-3 py-1 text-xs rounded border transition-colors flex-shrink-0 ${
                selectedStatus === status
                  ? `${config.borderColor} ${config.color}`
                  : "border-border text-muted-foreground hover:border-foreground-muted"
              }`}
            >
              {config.label}
            </button>
          );
        })}
      </div>

      {/* Suggestion list */}
      <div className="space-y-3">
        {suggestions.map((suggestion) => {
          const CategoryIcon = CATEGORY_CONFIG[suggestion.category].icon;
          const statusConfig = STATUS_CONFIG[suggestion.status];
          const StatusIcon = statusConfig.icon;

          return (
            <div
              key={suggestion.id}
              className={`p-4 rounded border ${statusConfig.borderColor} transition-colors hover:bg-foreground/5`}
            >
              {/* Header */}
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  <CategoryIcon
                    className={`w-4 h-4 ${
                      CATEGORY_CONFIG[suggestion.category].color
                    }`}
                  />
                  <h3 className="text-sm text-foreground-bright font-medium">
                    {suggestion.title}
                  </h3>
                </div>
                <div
                  className={`flex items-center gap-1 px-2 py-0.5 rounded ${statusConfig.bgColor}`}
                >
                  <StatusIcon className={`w-3 h-3 ${statusConfig.color}`} />
                  <span className={`text-xs ${statusConfig.color}`}>
                    {statusConfig.label}
                  </span>
                </div>
              </div>

              {/* Description */}
              <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                {suggestion.description}
              </p>

              {/* Footer */}
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-3 text-foreground-subtle">
                  <span>{formatDate(suggestion.submittedAt)}</span>
                  <span
                    className={CATEGORY_CONFIG[suggestion.category].color}
                  >
                    {CATEGORY_CONFIG[suggestion.category].label}
                  </span>
                  {/* Vote button and count */}
                  {allowVoting && (
                    <button
                      onClick={() => handleVote(suggestion.id)}
                      disabled={votingId === suggestion.id || votedIds.has(suggestion.id)}
                      className={`flex items-center gap-1 px-2 py-0.5 rounded transition-colors ${
                        votedIds.has(suggestion.id)
                          ? "bg-foreground/20 text-foreground"
                          : "hover:bg-foreground/10 text-muted-foreground hover:text-foreground"
                      } disabled:opacity-50`}
                    >
                      <ThumbsUp className="w-3 h-3" />
                      <span>{suggestion.votes}</span>
                    </button>
                  )}
                </div>
                {suggestion.taskId && (
                  <a
                    href={`/lab#task-${suggestion.taskId}`}
                    className="flex items-center gap-1 text-foreground-muted hover:text-foreground transition-colors"
                  >
                    <span>Task #{suggestion.taskId}</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Last updated */}
      {lastUpdated && (
        <div className="text-xs text-foreground-subtle text-center">
          Last updated: {lastUpdated.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}

export default SuggestionList;
