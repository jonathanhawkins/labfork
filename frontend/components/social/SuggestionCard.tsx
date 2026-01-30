"use client";

/**
 * SuggestionCard Component
 *
 * Displays a single suggestion with voting and status.
 */

import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  ChevronUp,
  ChevronDown,
  MessageCircle,
  Pin,
  Compass,
  TrendingUp,
  Bug,
  Lightbulb,
  HelpCircle,
  Users,
} from "lucide-react";
import type {
  Suggestion,
  SuggestionCategory,
} from "@/lib/social/suggestions/types";
import {
  CATEGORY_LABELS,
  STATUS_LABELS,
  STATUS_COLORS,
  PRIORITY_LABELS,
  PRIORITY_COLORS,
  getNetVotes,
} from "@/lib/social/suggestions/types";

export interface SuggestionCardProps {
  /** Suggestion data */
  suggestion: Suggestion;
  /** User's current vote (1, -1, or null) */
  userVote?: 1 | -1 | null;
  /** Upvote click handler */
  onUpvote?: () => void;
  /** Downvote click handler */
  onDownvote?: () => void;
  /** Card click handler */
  onClick?: () => void;
  /** Compact mode */
  compact?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Get category icon
 */
function getCategoryIcon(category: SuggestionCategory) {
  switch (category) {
    case "research_direction":
      return Compass;
    case "improvement":
      return TrendingUp;
    case "bug_report":
      return Bug;
    case "feature_request":
      return Lightbulb;
    case "question":
      return HelpCircle;
    case "collaboration":
      return Users;
    default:
      return Lightbulb;
  }
}

/**
 * Format relative time
 */
function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
  return date.toLocaleDateString();
}

export function SuggestionCard({
  suggestion,
  userVote,
  onUpvote,
  onDownvote,
  onClick,
  compact = false,
  className,
}: SuggestionCardProps) {
  const Icon = getCategoryIcon(suggestion.category);
  const netVotes = getNetVotes(suggestion.stats);

  return (
    <div
      className={cn(
        "group flex gap-4 rounded-lg border border-border transition-colors hover:border-foreground-muted",
        compact ? "p-3" : "p-4",
        onClick && "cursor-pointer",
        className
      )}
      onClick={onClick}
    >
      {/* Voting */}
      <div className="flex flex-col items-center gap-1">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onUpvote?.();
          }}
          className={cn(
            "p-1 rounded transition-colors",
            userVote === 1
              ? "text-green-400 bg-green-500/10"
              : "text-foreground-subtle hover:text-foreground-muted hover:bg-foreground-muted/10"
          )}
          title="Upvote"
        >
          <ChevronUp className="w-5 h-5" />
        </button>
        <span
          className={cn(
            "text-sm font-medium",
            netVotes > 0 && "text-green-400",
            netVotes < 0 && "text-red-400",
            netVotes === 0 && "text-foreground-muted"
          )}
        >
          {netVotes}
        </span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onDownvote?.();
          }}
          className={cn(
            "p-1 rounded transition-colors",
            userVote === -1
              ? "text-red-400 bg-red-500/10"
              : "text-foreground-subtle hover:text-foreground-muted hover:bg-foreground-muted/10"
          )}
          title="Downvote"
        >
          <ChevronDown className="w-5 h-5" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Header */}
        <div className="flex items-start gap-2 flex-wrap">
          {/* Pinned indicator */}
          {suggestion.isPinned && (
            <span className="text-yellow-400" title="Pinned">
              <Pin className="w-4 h-4" />
            </span>
          )}

          {/* Title */}
          <h3
            className={cn(
              "font-medium text-foreground flex-1 group-hover:text-foreground-bright transition-colors",
              compact ? "text-sm" : "text-base"
            )}
          >
            {suggestion.title}
          </h3>

          {/* Status badge */}
          <span
            className={cn(
              "text-xs px-2 py-0.5 rounded",
              STATUS_COLORS[suggestion.status]
            )}
          >
            {STATUS_LABELS[suggestion.status]}
          </span>
        </div>

        {/* Description (not in compact mode) */}
        {!compact && (
          <p className="mt-1 text-sm text-foreground-muted line-clamp-2">
            {suggestion.description}
          </p>
        )}

        {/* Meta info */}
        <div
          className={cn(
            "flex items-center gap-3 flex-wrap text-xs text-foreground-subtle",
            compact ? "mt-1" : "mt-2"
          )}
        >
          {/* Category */}
          <span className="flex items-center gap-1">
            <Icon className="w-3.5 h-3.5" />
            {CATEGORY_LABELS[suggestion.category]}
          </span>

          {/* Priority */}
          <span className={PRIORITY_COLORS[suggestion.priority]}>
            {PRIORITY_LABELS[suggestion.priority]}
          </span>

          {/* Comments */}
          <span className="flex items-center gap-1">
            <MessageCircle className="w-3.5 h-3.5" />
            {suggestion.stats.comments}
          </span>

          {/* Author */}
          <span>by {suggestion.author.displayName}</span>

          {/* Time */}
          <span>{formatRelativeTime(suggestion.createdAt)}</span>
        </div>

        {/* Tags */}
        {!compact && suggestion.tags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {suggestion.tags.slice(0, 4).map((tag) => (
              <span
                key={tag}
                className="px-1.5 py-0.5 rounded text-[10px] bg-foreground-muted/10 text-foreground-subtle"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default SuggestionCard;
