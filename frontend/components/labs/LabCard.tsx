"use client";

/**
 * LabCard Component
 *
 * Displays a lab in a card format for listings and grids.
 */

import { useState } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Star,
  GitFork,
  Activity,
  Clock,
  Eye,
  Layers,
  Mic,
  LineChart,
  Bot,
  Dna,
  Sparkles,
  Loader2,
  Lightbulb,
} from "lucide-react";
import type { Lab } from "@/lib/labs/types";
import { getLabPath, formatLabStats } from "@/lib/labs/types";

export interface LabCardProps {
  /** Lab data */
  lab: Lab;
  /** Compact mode (smaller card) */
  compact?: boolean;
  /** Show star button */
  showStar?: boolean;
  /** Is starred by current user */
  isStarred?: boolean;
  /** Is star action loading */
  isStarLoading?: boolean;
  /** Star click handler */
  onStarClick?: () => void;
  /** Card click handler (instead of navigation) */
  onClick?: () => void;
  /** Is selected */
  isSelected?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Get domain icon
 */
function getDomainIcon(domainSlug: string) {
  switch (domainSlug) {
    case "voice-clone":
      return Mic;
    case "quant-trading":
      return LineChart;
    case "robotics":
      return Bot;
    case "biotech":
      return Dna;
    case "firefly-network":
      return Sparkles;
    default:
      return Layers;
  }
}

/**
 * Get domain color
 */
function getDomainColor(domainSlug: string): string {
  switch (domainSlug) {
    case "voice-clone":
      return "text-blue-400 bg-blue-500/10";
    case "quant-trading":
      return "text-green-400 bg-green-500/10";
    case "robotics":
      return "text-purple-400 bg-purple-500/10";
    case "biotech":
      return "text-pink-400 bg-pink-500/10";
    case "firefly-network":
      return "text-amber-400 bg-amber-500/10";
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
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
  return `${Math.floor(diffDays / 365)} years ago`;
}

export function LabCard({
  lab,
  compact = false,
  showStar = true,
  isStarred = false,
  isStarLoading = false,
  onStarClick,
  onClick,
  isSelected = false,
  className,
}: LabCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const DomainIcon = getDomainIcon(lab.domainSlug);
  const domainColor = getDomainColor(lab.domainSlug);
  const labPath = getLabPath(lab.owner.username, lab.slug);
  const isIdea = lab.status === "idea";
  const allStatsZero = lab.stats.stars === 0 && lab.stats.forks === 0 && lab.stats.tasks === 0;

  const CardContent = (
    <div
      className={cn(
        "group relative flex flex-col rounded-lg border transition-all duration-200",
        isIdea ? "border-dashed" : "",
        isSelected
          ? "border-foreground-bright ring-2 ring-foreground-bright/20"
          : "border-border hover:border-foreground-muted",
        isHovered && !isSelected && "shadow-lg",
        isIdea && "opacity-80",
        compact ? "p-3" : "p-4",
        className
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          {/* Domain Icon */}
          <div
            className={cn(
              "flex-shrink-0 rounded-lg flex items-center justify-center",
              domainColor,
              compact ? "w-8 h-8" : "w-10 h-10"
            )}
          >
            <DomainIcon className={cn(compact ? "w-4 h-4" : "w-5 h-5")} />
          </div>

          {/* Name and Owner */}
          <div className="min-w-0 flex-1">
            <h3
              className={cn(
                "font-medium text-foreground truncate group-hover:text-foreground-bright transition-colors",
                compact ? "text-sm" : "text-base"
              )}
            >
              {lab.name}
            </h3>
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-foreground-muted truncate">
                {lab.owner.displayName}
              </p>
              {isIdea && (
                <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full text-[10px] font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20">
                  <Lightbulb className="w-2.5 h-2.5" />
                  Idea
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Star Button */}
        {showStar && (
          <button
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              if (!isStarLoading) {
                onStarClick?.();
              }
            }}
            disabled={isStarLoading}
            className={cn(
              "flex-shrink-0 min-w-[44px] min-h-[44px] flex items-center justify-center rounded-lg transition-colors",
              isStarLoading && "opacity-50 cursor-wait",
              isStarred
                ? "text-yellow-400 hover:text-yellow-300"
                : "text-foreground-subtle hover:text-foreground-muted hover:bg-foreground-muted/10"
            )}
            aria-label={isStarLoading ? "Updating star..." : isStarred ? `Unstar ${lab.name}` : `Star ${lab.name}`}
            aria-pressed={isStarred}
            aria-busy={isStarLoading}
          >
            {isStarLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Star className={cn("w-4 h-4", isStarred && "fill-current")} />
            )}
          </button>
        )}
      </div>

      {/* Description (not in compact mode) */}
      {!compact && lab.description && (
        <p className="mt-3 text-sm text-foreground-muted line-clamp-2">
          {lab.description}
        </p>
      )}

      {/* Stats - hidden when all are zero */}
      {!allStatsZero && (
        <div
          className={cn(
            "flex items-center gap-4 text-xs text-foreground-subtle",
            compact ? "mt-2" : "mt-4"
          )}
        >
          {lab.stats.stars > 0 && (
            <span className="flex items-center gap-1">
              <Star className="w-3.5 h-3.5" />
              {lab.stats.stars}
            </span>
          )}
          {lab.stats.forks > 0 && (
            <span className="flex items-center gap-1">
              <GitFork className="w-3.5 h-3.5" />
              {lab.stats.forks}
            </span>
          )}
          {!compact && (
            <>
              {lab.stats.tasks > 0 && (
                <span className="flex items-center gap-1">
                  <Activity className="w-3.5 h-3.5" />
                  {lab.stats.tasks}
                </span>
              )}
              {lab.stats.viewers > 0 && (
                <span className="flex items-center gap-1 text-green-400">
                  <Eye className="w-3.5 h-3.5" />
                  {lab.stats.viewers}
                </span>
              )}
            </>
          )}
        </div>
      )}

      {/* Tags (not in compact mode) */}
      {!compact && lab.tags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {lab.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="px-1.5 py-0.5 rounded text-[10px] bg-foreground-muted/10 text-foreground-subtle"
            >
              {tag}
            </span>
          ))}
          {lab.tags.length > 3 && (
            <span className="text-[10px] text-foreground-subtle">
              +{lab.tags.length - 3}
            </span>
          )}
        </div>
      )}

      {/* Footer (not in compact mode) */}
      {!compact && (
        <div className="mt-3 pt-3 border-t border-border flex items-center justify-between">
          <span className="flex items-center gap-1 text-xs text-foreground-subtle">
            <Clock className="w-3 h-3" />
            Updated {formatRelativeTime(lab.lastActivityAt)}
          </span>

          {/* Domain badge */}
          <span className={cn("text-xs px-2 py-0.5 rounded", domainColor)}>
            {lab.domainName}
          </span>
        </div>
      )}

      {/* Forked from indicator */}
      {lab.forkedFrom && (
        <div className="mt-2 text-xs text-foreground-subtle flex items-center gap-1">
          <GitFork className="w-3 h-3" />
          Forked from {lab.forkedFrom.sourceOwner}/{lab.forkedFrom.sourceSlug}
        </div>
      )}
    </div>
  );

  // If onClick handler is provided, use button instead of link
  if (onClick) {
    return (
      <button
        onClick={onClick}
        className="w-full text-left"
      >
        {CardContent}
      </button>
    );
  }

  // Default: wrap in link
  return (
    <Link href={labPath} className="block">
      {CardContent}
    </Link>
  );
}

export default LabCard;
