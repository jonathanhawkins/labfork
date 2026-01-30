"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  FileText,
  Users,
  Calendar,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Check,
  X,
  Loader2,
  AlertCircle,
  Sparkles,
  Clock,
  Quote,
} from "lucide-react";
import type { Paper } from "@/lib/papers/types";
import {
  getStatusDisplayInfo,
  getComplexityDisplayInfo,
  getRelevanceDisplayInfo,
  getSourceDisplayInfo,
  formatAuthors,
} from "@/lib/papers/types";

export interface PaperCardProps {
  /** Paper data */
  paper: Paper;
  /** Click handler for the card */
  onClick?: () => void;
  /** Accept handler */
  onAccept?: (paperId: string) => void;
  /** Reject handler */
  onReject?: (paperId: string) => void;
  /** Analyze handler */
  onAnalyze?: (paperId: string) => void;
  /** Whether to show expanded view */
  expanded?: boolean;
  /** Whether card is selected */
  isSelected?: boolean;
  /** Compact mode for list views */
  compact?: boolean;
  /** Whether actions are loading */
  isLoading?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * PaperCard - Displays a paper in a card format
 */
export function PaperCard({
  paper,
  onClick,
  onAccept,
  onReject,
  onAnalyze,
  expanded = false,
  isSelected = false,
  compact = false,
  isLoading = false,
  className,
}: PaperCardProps) {
  const [isExpanded, setIsExpanded] = useState(expanded);

  const statusInfo = getStatusDisplayInfo(paper.status);
  const sourceInfo = getSourceDisplayInfo(paper.metadata.source);

  // Get analysis info if available
  const analysis = paper.analysis;
  const complexityInfo = analysis
    ? getComplexityDisplayInfo(analysis.complexity)
    : null;
  const relevanceInfo = analysis
    ? getRelevanceDisplayInfo(analysis.relevanceScore)
    : null;

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else {
      setIsExpanded(!isExpanded);
    }
  };

  const handleAccept = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onAccept && !isLoading) {
      onAccept(paper.id);
    }
  };

  const handleReject = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onReject && !isLoading) {
      onReject(paper.id);
    }
  };

  const handleAnalyze = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onAnalyze && !isLoading) {
      onAnalyze(paper.id);
    }
  };

  // Format date
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return null;
    try {
      return new Date(dateStr).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
      });
    } catch {
      return null;
    }
  };

  return (
    <div
      className={cn(
        "relative group rounded-lg border transition-all duration-200",
        "bg-background-card hover:bg-background-elevated",
        isSelected
          ? "border-foreground-bright ring-1 ring-foreground-bright/50"
          : "border-border hover:border-foreground-muted",
        onClick && "cursor-pointer",
        className
      )}
    >
      {/* Main content */}
      <div
        className={cn("p-4", compact && "p-3")}
        onClick={handleClick}
        role={onClick ? "button" : undefined}
        tabIndex={onClick ? 0 : undefined}
      >
        {/* Header row */}
        <div className="flex items-start gap-3">
          {/* Expand/collapse icon */}
          {!compact && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsExpanded(!isExpanded);
              }}
              className="flex-shrink-0 mt-1 text-foreground-muted hover:text-foreground"
            >
              {isExpanded ? (
                <ChevronDown className="w-4 h-4" />
              ) : (
                <ChevronRight className="w-4 h-4" />
              )}
            </button>
          )}

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Title row */}
            <div className="flex items-start justify-between gap-2">
              <h3
                className={cn(
                  "font-normal text-foreground-bright group-hover:text-white transition-colors",
                  compact ? "text-sm line-clamp-1" : "text-base line-clamp-2"
                )}
              >
                {paper.metadata.title}
              </h3>

              {/* External link */}
              <a
                href={paper.metadata.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="flex-shrink-0 text-foreground-muted hover:text-foreground"
                title="Open paper"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>

            {/* Meta row */}
            <div className="flex flex-wrap items-center gap-2 mt-2 text-xs text-foreground-muted">
              {/* Source badge */}
              <span
                className={cn(
                  "px-1.5 py-0.5 rounded text-[10px]",
                  sourceInfo.bgColor,
                  sourceInfo.color
                )}
              >
                {sourceInfo.label}
              </span>

              {/* Status badge */}
              <span
                className={cn(
                  "px-1.5 py-0.5 rounded text-[10px]",
                  statusInfo.bgColor,
                  statusInfo.color
                )}
              >
                {statusInfo.label}
              </span>

              {/* Relevance score (if analyzed) */}
              {relevanceInfo && (
                <span
                  className={cn(
                    "px-1.5 py-0.5 rounded text-[10px] flex items-center gap-1",
                    relevanceInfo.bgColor,
                    relevanceInfo.color
                  )}
                >
                  <Sparkles className="w-3 h-3" />
                  {analysis!.relevanceScore}
                </span>
              )}

              {/* Complexity (if analyzed) */}
              {complexityInfo && (
                <span
                  className={cn(
                    "px-1.5 py-0.5 rounded text-[10px]",
                    complexityInfo.bgColor,
                    complexityInfo.color
                  )}
                >
                  {complexityInfo.label}
                </span>
              )}

              {/* Authors */}
              <span className="flex items-center gap-1">
                <Users className="w-3 h-3" />
                {formatAuthors(paper.metadata.authors, 2)}
              </span>

              {/* Date */}
              {paper.metadata.publishedDate && (
                <span className="flex items-center gap-1">
                  <Calendar className="w-3 h-3" />
                  {formatDate(paper.metadata.publishedDate)}
                </span>
              )}

              {/* Citations */}
              {paper.metadata.citationCount !== undefined &&
                paper.metadata.citationCount > 0 && (
                  <span className="flex items-center gap-1">
                    <Quote className="w-3 h-3" />
                    {paper.metadata.citationCount}
                  </span>
                )}
            </div>

            {/* Error message */}
            {paper.status === "error" && paper.error && (
              <div className="flex items-center gap-2 mt-2 text-xs text-red-400">
                <AlertCircle className="w-3 h-3" />
                {paper.error}
              </div>
            )}
          </div>
        </div>

        {/* Expanded content */}
        {isExpanded && !compact && (
          <div className="mt-4 space-y-3 border-t border-border pt-4">
            {/* Abstract */}
            <div>
              <h4 className="text-xs text-foreground-muted mb-1">Abstract</h4>
              <p className="text-sm text-foreground line-clamp-4">
                {paper.metadata.abstract}
              </p>
            </div>

            {/* Analysis results */}
            {analysis && (
              <>
                {/* Relevance */}
                <div>
                  <h4 className="text-xs text-foreground-muted mb-1">
                    Relevance Analysis
                  </h4>
                  <p className="text-sm text-foreground">
                    {analysis.relevanceReason}
                  </p>
                </div>

                {/* Novelty */}
                <div>
                  <h4 className="text-xs text-foreground-muted mb-1">
                    Key Contribution
                  </h4>
                  <p className="text-sm text-foreground">{analysis.novelty}</p>
                </div>

                {/* Techniques */}
                {analysis.techniques.length > 0 && (
                  <div>
                    <h4 className="text-xs text-foreground-muted mb-1">
                      Techniques
                    </h4>
                    <div className="flex flex-wrap gap-1">
                      {analysis.techniques.map((tech, i) => (
                        <span
                          key={i}
                          className={cn(
                            "px-2 py-0.5 rounded text-xs",
                            tech.isMainContribution
                              ? "bg-foreground-bright/20 text-foreground-bright"
                              : "bg-foreground-muted/10 text-foreground-muted"
                          )}
                        >
                          {tech.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Complexity explanation */}
                <div>
                  <h4 className="text-xs text-foreground-muted mb-1 flex items-center gap-2">
                    <Clock className="w-3 h-3" />
                    Implementation Complexity
                  </h4>
                  <p className="text-sm text-foreground">
                    {analysis.complexityReason}
                  </p>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Action buttons */}
      {!compact && (
        <div className="flex items-center gap-2 px-4 py-3 border-t border-border bg-background/50">
          {/* Analyze button (for fetched papers) */}
          {paper.status === "fetched" && onAnalyze && (
            <button
              onClick={handleAnalyze}
              disabled={isLoading}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
                "bg-foreground-bright/10 text-foreground-bright",
                "hover:bg-foreground-bright/20",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              Analyze
            </button>
          )}

          {/* Accept/Reject buttons (for analyzed papers) */}
          {paper.status === "analyzed" && (
            <>
              {onAccept && (
                <button
                  onClick={handleAccept}
                  disabled={isLoading}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
                    "bg-green-500/10 text-green-400",
                    "hover:bg-green-500/20",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Check className="w-4 h-4" />
                  )}
                  Accept
                </button>
              )}

              {onReject && (
                <button
                  onClick={handleReject}
                  disabled={isLoading}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
                    "bg-red-500/10 text-red-400",
                    "hover:bg-red-500/20",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  <X className="w-4 h-4" />
                  Reject
                </button>
              )}
            </>
          )}

          {/* Retry for errors */}
          {paper.status === "error" && onAnalyze && (
            <button
              onClick={handleAnalyze}
              disabled={isLoading}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
                "bg-yellow-500/10 text-yellow-400",
                "hover:bg-yellow-500/20",
                "disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <AlertCircle className="w-4 h-4" />
              )}
              Retry
            </button>
          )}

          {/* View tasks button (for implemented papers) */}
          {paper.status === "implemented" && paper.taskIds && (
            <span className="text-sm text-green-400 flex items-center gap-2">
              <Check className="w-4 h-4" />
              {paper.taskIds.length} tasks created
            </span>
          )}

          {/* Analyzing status */}
          {paper.status === "analyzing" && (
            <span className="text-sm text-yellow-400 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing...
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default PaperCard;
