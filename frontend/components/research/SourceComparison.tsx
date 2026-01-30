"use client";

import { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  ArrowUpDown,
  Check,
  X,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  GitCompare,
  BookOpen,
  Users,
  Calendar,
  Hash,
  Star,
} from "lucide-react";
import type { Paper } from "@/lib/papers/types";

export interface SourceComparisonProps {
  /** Papers to compare */
  papers: Paper[];
  /** Select paper handler */
  onSelect?: (paper: Paper) => void;
  /** Dismiss paper handler */
  onDismiss?: (paper: Paper) => void;
  /** Custom class name */
  className?: string;
}

type SortField = "title" | "date" | "citations" | "authors" | "source";
type SortDirection = "asc" | "desc";

interface ComparisonField {
  key: string;
  label: string;
  icon: React.ReactNode;
  getValue: (paper: Paper) => string | number | null;
  format?: (value: string | number | null) => string;
}

const COMPARISON_FIELDS: ComparisonField[] = [
  {
    key: "title",
    label: "Title",
    icon: <BookOpen className="w-4 h-4" />,
    getValue: (paper) => paper.metadata.title,
  },
  {
    key: "authors",
    label: "Authors",
    icon: <Users className="w-4 h-4" />,
    getValue: (paper) =>
      paper.metadata.authors.length > 0
        ? paper.metadata.authors.map((a) => a.name).join(", ")
        : null,
    format: (value) => {
      if (!value) return "Unknown";
      const authors = String(value).split(", ");
      if (authors.length <= 3) return String(value);
      return `${authors.slice(0, 3).join(", ")} +${authors.length - 3}`;
    },
  },
  {
    key: "date",
    label: "Published",
    icon: <Calendar className="w-4 h-4" />,
    getValue: (paper) => paper.metadata.publishedDate || null,
    format: (value) => {
      if (!value) return "Unknown";
      try {
        return new Date(String(value)).toLocaleDateString("en-US", {
          year: "numeric",
          month: "short",
        });
      } catch {
        return String(value);
      }
    },
  },
  {
    key: "citations",
    label: "Citations",
    icon: <Hash className="w-4 h-4" />,
    getValue: (paper) => paper.metadata.citationCount ?? null,
    format: (value) =>
      value !== null ? Number(value).toLocaleString() : "N/A",
  },
  {
    key: "source",
    label: "Source",
    icon: <Star className="w-4 h-4" />,
    getValue: (paper) => paper.metadata.source,
    format: (value) => {
      const sourceLabels: Record<string, string> = {
        arxiv: "arXiv",
        "semantic-scholar": "Semantic Scholar",
        github: "GitHub",
        pdf: "PDF Upload",
        doi: "DOI",
        manual: "Manual Entry",
      };
      return sourceLabels[String(value)] || String(value);
    },
  },
];

/**
 * SourceComparison - Side-by-side comparison of papers from different sources
 */
export function SourceComparison({
  papers,
  onSelect,
  onDismiss,
  className,
}: SourceComparisonProps) {
  const [sortField, setSortField] = useState<SortField>("citations");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [expandedPapers, setExpandedPapers] = useState<Set<string>>(new Set());

  // Sort papers
  const sortedPapers = useMemo(() => {
    return [...papers].sort((a, b) => {
      const field = COMPARISON_FIELDS.find((f) => f.key === sortField);
      if (!field) return 0;

      const aValue = field.getValue(a);
      const bValue = field.getValue(b);

      // Handle nulls
      if (aValue === null && bValue === null) return 0;
      if (aValue === null) return sortDirection === "asc" ? -1 : 1;
      if (bValue === null) return sortDirection === "asc" ? 1 : -1;

      // Compare values
      if (typeof aValue === "number" && typeof bValue === "number") {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
      }

      const aStr = String(aValue).toLowerCase();
      const bStr = String(bValue).toLowerCase();
      return sortDirection === "asc"
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });
  }, [papers, sortField, sortDirection]);

  // Toggle sort
  const handleSort = useCallback(
    (field: SortField) => {
      if (sortField === field) {
        setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
      } else {
        setSortField(field);
        setSortDirection("desc");
      }
    },
    [sortField]
  );

  // Toggle expansion
  const toggleExpanded = useCallback((paperId: string) => {
    setExpandedPapers((prev) => {
      const next = new Set(prev);
      if (next.has(paperId)) {
        next.delete(paperId);
      } else {
        next.add(paperId);
      }
      return next;
    });
  }, []);

  if (papers.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center py-12 text-center",
          className
        )}
      >
        <GitCompare className="w-12 h-12 text-foreground-muted mb-4" />
        <p className="text-foreground-muted">No papers to compare</p>
        <p className="text-sm text-foreground-subtle mt-1">
          Add papers from different sources to compare them
        </p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-foreground-muted" />
          <h3 className="text-lg text-foreground-bright">
            Compare {papers.length} Papers
          </h3>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-foreground-muted">Sort by:</span>
          <select
            value={sortField}
            onChange={(e) => handleSort(e.target.value as SortField)}
            className={cn(
              "px-2 py-1 text-sm rounded",
              "bg-background border border-border",
              "text-foreground focus:outline-none focus:border-foreground-muted"
            )}
          >
            <option value="citations">Citations</option>
            <option value="date">Date</option>
            <option value="title">Title</option>
            <option value="authors">Authors</option>
            <option value="source">Source</option>
          </select>
          <button
            onClick={() =>
              setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"))
            }
            className="p-1 rounded hover:bg-foreground-muted/10"
            title={sortDirection === "asc" ? "Ascending" : "Descending"}
          >
            <ArrowUpDown className="w-4 h-4 text-foreground-muted" />
          </button>
        </div>
      </div>

      {/* Comparison table */}
      <div className="border border-border rounded-lg overflow-hidden">
        {/* Table header */}
        <div className="bg-background-card border-b border-border">
          <div className="grid grid-cols-12 gap-2 px-4 py-3 text-xs text-foreground-muted">
            <div className="col-span-5">Paper</div>
            <div className="col-span-2">Source</div>
            <div className="col-span-2">Date</div>
            <div className="col-span-2 text-right">Citations</div>
            <div className="col-span-1"></div>
          </div>
        </div>

        {/* Table body */}
        <div className="divide-y divide-border">
          {sortedPapers.map((paper) => {
            const isExpanded = expandedPapers.has(paper.id);

            return (
              <div key={paper.id} className="bg-background">
                {/* Main row */}
                <div className="grid grid-cols-12 gap-2 px-4 py-3 items-center">
                  {/* Title and authors */}
                  <div className="col-span-5">
                    <p className="text-sm text-foreground line-clamp-1">
                      {paper.metadata.title}
                    </p>
                    <p className="text-xs text-foreground-muted line-clamp-1 mt-0.5">
                      {paper.metadata.authors
                        .slice(0, 2)
                        .map((a) => a.name)
                        .join(", ")}
                      {paper.metadata.authors.length > 2 && " et al."}
                    </p>
                  </div>

                  {/* Source */}
                  <div className="col-span-2">
                    <span
                      className={cn(
                        "px-2 py-0.5 text-xs rounded",
                        paper.metadata.source === "arxiv" &&
                          "bg-red-500/10 text-red-400",
                        paper.metadata.source === "semantic-scholar" &&
                          "bg-blue-500/10 text-blue-400",
                        paper.metadata.source === "github" &&
                          "bg-purple-500/10 text-purple-400",
                        paper.metadata.source === "pdf" &&
                          "bg-orange-500/10 text-orange-400",
                        paper.metadata.source === "doi" &&
                          "bg-green-500/10 text-green-400"
                      )}
                    >
                      {COMPARISON_FIELDS.find((f) => f.key === "source")
                        ?.format?.(paper.metadata.source) || paper.metadata.source}
                    </span>
                  </div>

                  {/* Date */}
                  <div className="col-span-2 text-sm text-foreground-muted">
                    {COMPARISON_FIELDS.find((f) => f.key === "date")?.format?.(
                      paper.metadata.publishedDate || null
                    )}
                  </div>

                  {/* Citations */}
                  <div className="col-span-2 text-right">
                    <span className="text-sm text-foreground">
                      {paper.metadata.citationCount?.toLocaleString() ?? "N/A"}
                    </span>
                  </div>

                  {/* Actions */}
                  <div className="col-span-1 flex items-center justify-end gap-1">
                    <button
                      onClick={() => toggleExpanded(paper.id)}
                      className="p-1 rounded hover:bg-foreground-muted/10"
                      title={isExpanded ? "Collapse" : "Expand"}
                    >
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-foreground-muted" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-foreground-muted" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Expanded details */}
                {isExpanded && (
                  <div className="px-4 pb-4 pt-2 border-t border-border/50 bg-background-card/30">
                    {/* Abstract */}
                    <div className="mb-4">
                      <h4 className="text-xs text-foreground-muted mb-1">
                        Abstract
                      </h4>
                      <p className="text-sm text-foreground line-clamp-4">
                        {paper.metadata.abstract || "No abstract available"}
                      </p>
                    </div>

                    {/* Additional metadata */}
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      {paper.metadata.doi && (
                        <div>
                          <span className="text-foreground-muted">DOI: </span>
                          <a
                            href={`https://doi.org/${paper.metadata.doi}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-foreground-bright hover:underline"
                          >
                            {paper.metadata.doi}
                          </a>
                        </div>
                      )}
                      {paper.metadata.url && (
                        <div>
                          <a
                            href={paper.metadata.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1 text-foreground-bright hover:underline"
                          >
                            <ExternalLink className="w-3 h-3" />
                            View Paper
                          </a>
                        </div>
                      )}
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2 mt-4 pt-4 border-t border-border/50">
                      {onSelect && (
                        <button
                          onClick={() => onSelect(paper)}
                          className={cn(
                            "flex items-center gap-2 px-3 py-1.5 text-sm rounded",
                            "bg-foreground-bright text-background hover:bg-white"
                          )}
                        >
                          <Check className="w-4 h-4" />
                          Select
                        </button>
                      )}
                      {onDismiss && (
                        <button
                          onClick={() => onDismiss(paper)}
                          className={cn(
                            "flex items-center gap-2 px-3 py-1.5 text-sm rounded",
                            "text-foreground-muted hover:text-foreground",
                            "border border-border hover:border-foreground-muted"
                          )}
                        >
                          <X className="w-4 h-4" />
                          Dismiss
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Summary */}
      <div className="text-xs text-foreground-subtle text-center">
        Showing {papers.length} papers from{" "}
        {new Set(papers.map((p) => p.metadata.source)).size} sources
      </div>
    </div>
  );
}

export default SourceComparison;
