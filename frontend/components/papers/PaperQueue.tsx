"use client";

import { useState, useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  Search,
  Filter,
  RefreshCw,
  ChevronDown,
  FileText,
  Loader2,
  Sparkles,
  X,
} from "lucide-react";
import type { Paper, PaperStatus, PaperSource } from "@/lib/papers/types";
import { PaperCard } from "./PaperCard";

export interface PaperQueueProps {
  /** Initial papers (optional, will fetch if not provided) */
  initialPapers?: Paper[];
  /** Domain slug for filtering */
  domainSlug?: string;
  /** Paper click handler */
  onPaperClick?: (paper: Paper) => void;
  /** Paper accepted handler */
  onPaperAccepted?: (paper: Paper) => void;
  /** Paper rejected handler */
  onPaperRejected?: (paper: Paper) => void;
  /** Whether to auto-refresh */
  autoRefresh?: boolean;
  /** Refresh interval in ms */
  refreshInterval?: number;
  /** Custom class name */
  className?: string;
}

type SortField = "addedAt" | "relevanceScore" | "publishedDate";
type SortOrder = "asc" | "desc";

const STATUS_OPTIONS: { value: PaperStatus | "all"; label: string }[] = [
  { value: "all", label: "All Status" },
  { value: "fetched", label: "Ready" },
  { value: "analyzing", label: "Analyzing" },
  { value: "analyzed", label: "Analyzed" },
  { value: "accepted", label: "Accepted" },
  { value: "rejected", label: "Rejected" },
  { value: "implemented", label: "Implemented" },
  { value: "error", label: "Error" },
];

const SORT_OPTIONS: { value: SortField; label: string }[] = [
  { value: "addedAt", label: "Date Added" },
  { value: "relevanceScore", label: "Relevance" },
  { value: "publishedDate", label: "Published Date" },
];

/**
 * PaperQueue - List of papers with filtering and sorting
 */
export function PaperQueue({
  initialPapers,
  domainSlug,
  onPaperClick,
  onPaperAccepted,
  onPaperRejected,
  autoRefresh = true,
  refreshInterval = 10000,
  className,
}: PaperQueueProps) {
  // State
  const [papers, setPapers] = useState<Paper[]>(initialPapers || []);
  const [isLoading, setIsLoading] = useState(!initialPapers);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<PaperStatus | "all">("all");
  const [sortField, setSortField] = useState<SortField>("addedAt");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [showFilters, setShowFilters] = useState(false);

  // Action loading states
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>(
    {}
  );

  // Fetch papers
  const fetchPapers = useCallback(
    async (showLoading = true) => {
      if (showLoading) setIsLoading(true);
      else setIsRefreshing(true);

      try {
        let url = `/api/papers?sortBy=${sortField}&sortOrder=${sortOrder}`;

        if (domainSlug) {
          url += `&domain=${domainSlug}`;
        }

        if (statusFilter !== "all") {
          url += `&status=${statusFilter}`;
        }

        if (search) {
          url += `&search=${encodeURIComponent(search)}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        if (data.papers) {
          setPapers(data.papers);
          setError(null);
        } else {
          setError("Failed to fetch papers");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch papers");
      } finally {
        setIsLoading(false);
        setIsRefreshing(false);
      }
    },
    [domainSlug, statusFilter, sortField, sortOrder, search]
  );

  // Initial fetch
  useEffect(() => {
    if (!initialPapers) {
      fetchPapers(true);
    }
  }, [fetchPapers, initialPapers]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const timer = setInterval(() => {
      fetchPapers(false);
    }, refreshInterval);

    return () => clearInterval(timer);
  }, [autoRefresh, refreshInterval, fetchPapers]);

  // Handle analyze
  const handleAnalyze = useCallback(
    async (paperId: string) => {
      setActionLoading((prev) => ({ ...prev, [paperId]: true }));

      try {
        const response = await fetch("/api/papers/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ paperId, domainSlug }),
        });

        const data = await response.json();

        if (data.success && data.paper) {
          setPapers((prev) =>
            prev.map((p) => (p.id === paperId ? data.paper : p))
          );
        }
      } catch (err) {
        console.error("Analyze failed:", err);
      } finally {
        setActionLoading((prev) => ({ ...prev, [paperId]: false }));
      }
    },
    [domainSlug]
  );

  // Handle accept
  const handleAccept = useCallback(
    async (paperId: string) => {
      setActionLoading((prev) => ({ ...prev, [paperId]: true }));

      try {
        const response = await fetch("/api/papers", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: paperId, status: "accepted" }),
        });

        const data = await response.json();

        if (data.success && data.paper) {
          setPapers((prev) =>
            prev.map((p) => (p.id === paperId ? data.paper : p))
          );
          if (onPaperAccepted) {
            onPaperAccepted(data.paper);
          }
        }
      } catch (err) {
        console.error("Accept failed:", err);
      } finally {
        setActionLoading((prev) => ({ ...prev, [paperId]: false }));
      }
    },
    [onPaperAccepted]
  );

  // Handle reject
  const handleReject = useCallback(
    async (paperId: string) => {
      setActionLoading((prev) => ({ ...prev, [paperId]: true }));

      try {
        const response = await fetch("/api/papers", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: paperId, status: "rejected" }),
        });

        const data = await response.json();

        if (data.success && data.paper) {
          setPapers((prev) =>
            prev.map((p) => (p.id === paperId ? data.paper : p))
          );
          if (onPaperRejected) {
            onPaperRejected(data.paper);
          }
        }
      } catch (err) {
        console.error("Reject failed:", err);
      } finally {
        setActionLoading((prev) => ({ ...prev, [paperId]: false }));
      }
    },
    [onPaperRejected]
  );

  // Filter papers client-side for search
  const filteredPapers = papers.filter((paper) => {
    if (!search) return true;
    const searchLower = search.toLowerCase();
    return (
      paper.metadata.title.toLowerCase().includes(searchLower) ||
      paper.metadata.abstract.toLowerCase().includes(searchLower) ||
      paper.metadata.authors.some((a) =>
        a.name.toLowerCase().includes(searchLower)
      )
    );
  });

  // Stats
  const stats = {
    total: papers.length,
    pending:
      papers.filter((p) => p.status === "fetched" || p.status === "pending")
        .length +
      papers.filter((p) => p.status === "analyzing").length,
    analyzed: papers.filter((p) => p.status === "analyzed").length,
    accepted: papers.filter((p) => p.status === "accepted").length,
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-muted" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search papers..."
            className={cn(
              "w-full pl-10 pr-4 py-2 text-sm rounded-lg",
              "bg-background-card border border-border",
              "text-foreground placeholder:text-foreground-subtle",
              "focus:outline-none focus:border-foreground-muted"
            )}
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-foreground-muted hover:text-foreground"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {/* Filter toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              "flex items-center gap-2 px-3 py-2 text-sm rounded-lg",
              "border transition-colors",
              showFilters
                ? "bg-foreground-bright/10 border-foreground-bright/50 text-foreground-bright"
                : "bg-background-card border-border text-foreground-muted hover:text-foreground"
            )}
          >
            <Filter className="w-4 h-4" />
            Filters
            {statusFilter !== "all" && (
              <span className="w-2 h-2 rounded-full bg-foreground-bright" />
            )}
          </button>

          {/* Refresh */}
          <button
            onClick={() => fetchPapers(false)}
            disabled={isRefreshing}
            className={cn(
              "p-2 rounded-lg border border-border",
              "bg-background-card text-foreground-muted hover:text-foreground",
              "disabled:opacity-50"
            )}
          >
            <RefreshCw
              className={cn("w-4 h-4", isRefreshing && "animate-spin")}
            />
          </button>
        </div>
      </div>

      {/* Filters panel */}
      {showFilters && (
        <div className="flex flex-wrap items-center gap-3 p-4 rounded-lg bg-background-card border border-border">
          {/* Status filter */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-foreground-muted">Status:</label>
            <select
              value={statusFilter}
              onChange={(e) =>
                setStatusFilter(e.target.value as PaperStatus | "all")
              }
              className={cn(
                "px-3 py-1.5 text-sm rounded-lg",
                "bg-background border border-border",
                "text-foreground",
                "focus:outline-none focus:border-foreground-muted"
              )}
            >
              {STATUS_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-foreground-muted">Sort:</label>
            <select
              value={sortField}
              onChange={(e) => setSortField(e.target.value as SortField)}
              className={cn(
                "px-3 py-1.5 text-sm rounded-lg",
                "bg-background border border-border",
                "text-foreground",
                "focus:outline-none focus:border-foreground-muted"
              )}
            >
              {SORT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
              className="px-2 py-1.5 text-sm rounded-lg bg-background border border-border text-foreground-muted hover:text-foreground"
            >
              {sortOrder === "desc" ? "Newest" : "Oldest"}
            </button>
          </div>

          {/* Clear filters */}
          {(statusFilter !== "all" || search) && (
            <button
              onClick={() => {
                setStatusFilter("all");
                setSearch("");
              }}
              className="text-sm text-foreground-muted hover:text-foreground"
            >
              Clear filters
            </button>
          )}
        </div>
      )}

      {/* Stats bar */}
      <div className="flex items-center gap-4 text-xs text-foreground-muted">
        <span>{stats.total} papers</span>
        {stats.pending > 0 && (
          <span className="flex items-center gap-1 text-yellow-400">
            <Sparkles className="w-3 h-3" />
            {stats.pending} pending
          </span>
        )}
        {stats.analyzed > 0 && (
          <span className="text-purple-400">{stats.analyzed} analyzed</span>
        )}
        {stats.accepted > 0 && (
          <span className="text-green-400">{stats.accepted} accepted</span>
        )}
      </div>

      {/* Papers list */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 text-foreground-muted animate-spin" />
        </div>
      ) : error ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <p className="text-foreground-muted">{error}</p>
          <button
            onClick={() => fetchPapers(true)}
            className="mt-2 text-sm text-foreground-bright hover:underline"
          >
            Try again
          </button>
        </div>
      ) : filteredPapers.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <FileText className="w-12 h-12 text-foreground-muted/30 mb-4" />
          <p className="text-foreground-muted">No papers found</p>
          {(search || statusFilter !== "all") && (
            <button
              onClick={() => {
                setSearch("");
                setStatusFilter("all");
              }}
              className="mt-2 text-sm text-foreground-bright hover:underline"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {filteredPapers.map((paper) => (
            <PaperCard
              key={paper.id}
              paper={paper}
              onClick={onPaperClick ? () => onPaperClick(paper) : undefined}
              onAnalyze={handleAnalyze}
              onAccept={handleAccept}
              onReject={handleReject}
              isLoading={actionLoading[paper.id]}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default PaperQueue;
