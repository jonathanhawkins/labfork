"use client";

/**
 * ResultCard Component
 *
 * Displays a research result in a card format with metrics,
 * media preview, and engagement actions.
 */

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { cn } from "@/lib/utils";
import {
  Heart,
  MessageCircle,
  Share2,
  Bookmark,
  Eye,
  Clock,
  ArrowUp,
  ArrowDown,
  Box,
  Play,
  Lightbulb,
  GitCompare,
  Database,
  FileText,
  MoreHorizontal,
  ExternalLink,
} from "lucide-react";
import type {
  Result,
  ResultType,
  ResultMetric,
} from "@/lib/social/results/types";
import {
  getResultPath,
  RESULT_TYPE_LABELS,
  formatMetricValue,
  calculateImprovement,
} from "@/lib/social/results/types";

export interface ResultCardProps {
  /** Result data */
  result: Result;
  /** Compact mode (smaller card) */
  compact?: boolean;
  /** Show like button */
  showLike?: boolean;
  /** Is liked by current user */
  isLiked?: boolean;
  /** Like click handler */
  onLikeClick?: () => void;
  /** Show save button */
  showSave?: boolean;
  /** Is saved by current user */
  isSaved?: boolean;
  /** Save click handler */
  onSaveClick?: () => void;
  /** Share click handler */
  onShareClick?: () => void;
  /** Card click handler (instead of navigation) */
  onClick?: () => void;
  /** Is selected */
  isSelected?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Get result type icon component
 */
function getTypeIcon(type: ResultType) {
  switch (type) {
    case "model":
      return Box;
    case "demo":
      return Play;
    case "finding":
      return Lightbulb;
    case "comparison":
      return GitCompare;
    case "dataset":
      return Database;
    case "paper":
      return FileText;
    default:
      return Box;
  }
}

/**
 * Get type color classes
 */
function getTypeColor(type: ResultType): string {
  switch (type) {
    case "model":
      return "text-blue-400 bg-blue-500/10";
    case "demo":
      return "text-green-400 bg-green-500/10";
    case "finding":
      return "text-yellow-400 bg-yellow-500/10";
    case "comparison":
      return "text-purple-400 bg-purple-500/10";
    case "dataset":
      return "text-orange-400 bg-orange-500/10";
    case "paper":
      return "text-pink-400 bg-pink-500/10";
    default:
      return "text-foreground-muted bg-foreground-muted/10";
  }
}

/**
 * Format relative time
 */
function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)}mo ago`;
  return `${Math.floor(diffDays / 365)}y ago`;
}

/**
 * Format large numbers
 */
function formatCount(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return n.toString();
}

/**
 * Metric improvement badge
 */
function MetricBadge({ metric }: { metric: ResultMetric }) {
  const improvement = calculateImprovement(metric);

  if (improvement === null) {
    return (
      <span className="text-foreground">{formatMetricValue(metric)}</span>
    );
  }

  const isPositive = improvement > 0;
  const Icon = isPositive ? ArrowUp : ArrowDown;

  return (
    <span className="flex items-center gap-1">
      <span className="text-foreground">{formatMetricValue(metric)}</span>
      <span
        className={cn(
          "flex items-center text-xs",
          isPositive ? "text-green-400" : "text-red-400"
        )}
      >
        <Icon className="w-3 h-3" />
        {Math.abs(improvement).toFixed(1)}%
      </span>
    </span>
  );
}

export function ResultCard({
  result,
  compact = false,
  showLike = true,
  isLiked = false,
  onLikeClick,
  showSave = true,
  isSaved = false,
  onSaveClick,
  onShareClick,
  onClick,
  isSelected = false,
  className,
}: ResultCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const TypeIcon = getTypeIcon(result.type);
  const typeColor = getTypeColor(result.type);
  const resultPath = getResultPath(result.id);

  // Get first image for preview
  const previewImage = result.media.find((m) => m.type === "image");

  // Get primary metrics (first 3)
  const primaryMetrics = result.metrics?.primary.slice(0, 3) || [];

  const CardContent = (
    <div
      className={cn(
        "group relative flex flex-col rounded-lg border transition-all duration-200",
        isSelected
          ? "border-foreground-bright ring-2 ring-foreground-bright/20"
          : "border-border hover:border-foreground-muted",
        isHovered && !isSelected && "shadow-lg",
        compact ? "p-3" : "p-4",
        className
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          {/* Type Icon */}
          <div
            className={cn(
              "flex-shrink-0 rounded-lg flex items-center justify-center",
              typeColor,
              compact ? "w-8 h-8" : "w-10 h-10"
            )}
          >
            <TypeIcon className={cn(compact ? "w-4 h-4" : "w-5 h-5")} />
          </div>

          {/* Title and Author */}
          <div className="min-w-0 flex-1">
            <h3
              className={cn(
                "font-medium text-foreground truncate group-hover:text-foreground-bright transition-colors",
                compact ? "text-sm" : "text-base"
              )}
            >
              {result.title}
            </h3>
            <p className="text-xs text-foreground-muted truncate">
              {result.author.displayName}
            </p>
          </div>
        </div>

        {/* More menu */}
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          className="flex-shrink-0 p-1.5 rounded-lg text-foreground-subtle hover:text-foreground-muted transition-colors opacity-0 group-hover:opacity-100"
        >
          <MoreHorizontal className="w-4 h-4" />
        </button>
      </div>

      {/* Description (not in compact mode) */}
      {!compact && result.description && (
        <p className="mt-3 text-sm text-foreground-muted line-clamp-2">
          {result.description}
        </p>
      )}

      {/* Media Preview (not in compact mode) */}
      {!compact && previewImage && (
        <div className="mt-3 rounded-lg overflow-hidden bg-background-darker aspect-video">
          <Image
            src={previewImage.url}
            alt={previewImage.alt || result.title}
            fill
            className="object-cover"
            unoptimized
          />
        </div>
      )}

      {/* Metrics */}
      {primaryMetrics.length > 0 && (
        <div
          className={cn(
            "flex flex-wrap gap-3 text-sm",
            compact ? "mt-2" : "mt-4"
          )}
        >
          {primaryMetrics.map((metric, index) => (
            <div
              key={metric.name + index}
              className="flex flex-col gap-0.5"
            >
              <span className="text-[10px] text-foreground-subtle uppercase tracking-wider">
                {metric.name}
              </span>
              <MetricBadge metric={metric} />
            </div>
          ))}
        </div>
      )}

      {/* Tags (not in compact mode) */}
      {!compact && result.tags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {result.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="px-1.5 py-0.5 rounded text-[10px] bg-foreground-muted/10 text-foreground-subtle"
            >
              {tag}
            </span>
          ))}
          {result.tags.length > 4 && (
            <span className="text-[10px] text-foreground-subtle">
              +{result.tags.length - 4}
            </span>
          )}
        </div>
      )}

      {/* Engagement Stats */}
      <div
        className={cn(
          "flex items-center gap-4 text-xs text-foreground-subtle",
          compact ? "mt-2" : "mt-4 pt-3 border-t border-border"
        )}
      >
        {/* Like button */}
        {showLike && (
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onLikeClick?.();
            }}
            className={cn(
              "flex items-center gap-1 transition-colors",
              isLiked
                ? "text-red-400 hover:text-red-300"
                : "hover:text-foreground-muted"
            )}
            title={isLiked ? "Unlike" : "Like"}
          >
            <Heart
              className={cn("w-3.5 h-3.5", isLiked && "fill-current")}
            />
            <span>{formatCount(result.stats.likes)}</span>
          </button>
        )}

        {/* Comments */}
        <span className="flex items-center gap-1">
          <MessageCircle className="w-3.5 h-3.5" />
          <span>{formatCount(result.stats.comments)}</span>
        </span>

        {/* Views (not in compact mode) */}
        {!compact && (
          <span className="flex items-center gap-1">
            <Eye className="w-3.5 h-3.5" />
            <span>{formatCount(result.stats.views)}</span>
          </span>
        )}

        <span className="flex-1" />

        {/* Save button */}
        {showSave && (
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onSaveClick?.();
            }}
            className={cn(
              "p-1 transition-colors",
              isSaved
                ? "text-blue-400 hover:text-blue-300"
                : "hover:text-foreground-muted"
            )}
            title={isSaved ? "Unsave" : "Save"}
          >
            <Bookmark
              className={cn("w-3.5 h-3.5", isSaved && "fill-current")}
            />
          </button>
        )}

        {/* Share button */}
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onShareClick?.();
          }}
          className="p-1 hover:text-foreground-muted transition-colors"
          title="Share"
        >
          <Share2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Footer with time and type (not in compact mode) */}
      {!compact && (
        <div className="mt-2 flex items-center justify-between">
          <span className="flex items-center gap-1 text-xs text-foreground-subtle">
            <Clock className="w-3 h-3" />
            {formatRelativeTime(result.publishedAt || result.createdAt)}
          </span>

          {/* Type badge */}
          <span className={cn("text-xs px-2 py-0.5 rounded", typeColor)}>
            {RESULT_TYPE_LABELS[result.type]}
          </span>
        </div>
      )}

      {/* Pinned indicator */}
      {result.isPinned && (
        <div className="absolute -top-2 -right-2 w-4 h-4 rounded-full bg-yellow-500 flex items-center justify-center">
          <span className="text-[8px]">*</span>
        </div>
      )}
    </div>
  );

  // If onClick handler is provided, use button instead of link
  if (onClick) {
    return (
      <button onClick={onClick} className="w-full text-left">
        {CardContent}
      </button>
    );
  }

  // Default: wrap in link
  return (
    <Link href={resultPath} className="block">
      {CardContent}
    </Link>
  );
}

export default ResultCard;
