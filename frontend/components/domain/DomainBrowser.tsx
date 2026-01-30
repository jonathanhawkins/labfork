"use client";

import { useState, useEffect, useMemo } from "react";
import { cn } from "@/lib/utils";
import { DomainCard, DomainCardProps } from "./DomainCard";
import { Search, Filter, X, Loader2, RefreshCw } from "lucide-react";

/**
 * Domain summary from API
 */
export interface DomainSummary {
  name: string;
  slug: string;
  description: string;
  difficulty?: "beginner" | "intermediate" | "advanced";
  primaryColor: string;
  accentColor: string;
  backgroundStyle: string;
  tags?: string[];
  propsCount: number;
  metricsCount: number;
}

/**
 * Props for DomainBrowser component
 */
export interface DomainBrowserProps {
  /** Initial domains to display (optional, will fetch if not provided) */
  initialDomains?: DomainSummary[];
  /** Called when a domain is selected */
  onSelectDomain?: (slug: string) => void;
  /** Currently selected domain slug */
  selectedDomain?: string;
  /** Show compact cards */
  compact?: boolean;
  /** Maximum domains to show (for preview mode) */
  maxDomains?: number;
  /** Hide filters */
  hideFilters?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Extract unique categories from tags
 */
function extractCategories(domains: DomainSummary[]): string[] {
  const categories = new Set<string>();
  for (const domain of domains) {
    for (const tag of domain.tags || []) {
      categories.add(tag);
    }
  }
  return Array.from(categories).sort();
}

/**
 * DomainBrowser - Grid view of all available domains
 *
 * Features:
 * - Responsive grid layout
 * - Search by name/description
 * - Filter by category/tag
 * - Filter by difficulty
 * - Loading and error states
 */
export function DomainBrowser({
  initialDomains,
  onSelectDomain,
  selectedDomain,
  compact = false,
  maxDomains,
  hideFilters = false,
  className,
}: DomainBrowserProps) {
  const [domains, setDomains] = useState<DomainSummary[]>(initialDomains || []);
  const [isLoading, setIsLoading] = useState(!initialDomains);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>("");

  // Fetch domains on mount
  useEffect(() => {
    if (initialDomains) return;

    async function fetchDomains() {
      try {
        setIsLoading(true);
        setError(null);

        const params = new URLSearchParams();
        // We'll filter client-side for better UX

        const response = await fetch(`/api/domains?${params}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch domains: ${response.statusText}`);
        }

        const data = await response.json();
        setDomains(data.domains || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load domains");
      } finally {
        setIsLoading(false);
      }
    }

    fetchDomains();
  }, [initialDomains]);

  // Extract categories for filter dropdown
  const categories = useMemo(() => extractCategories(domains), [domains]);

  // Filter domains
  const filteredDomains = useMemo(() => {
    let result = domains;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (d) =>
          d.name.toLowerCase().includes(query) ||
          d.description.toLowerCase().includes(query) ||
          d.tags?.some((t) => t.toLowerCase().includes(query))
      );
    }

    // Category filter
    if (selectedCategory) {
      result = result.filter((d) =>
        d.tags?.some((t) => t.toLowerCase() === selectedCategory.toLowerCase())
      );
    }

    // Difficulty filter
    if (selectedDifficulty) {
      result = result.filter((d) => d.difficulty === selectedDifficulty);
    }

    // Limit if specified
    if (maxDomains) {
      result = result.slice(0, maxDomains);
    }

    return result;
  }, [domains, searchQuery, selectedCategory, selectedDifficulty, maxDomains]);

  // Clear all filters
  const clearFilters = () => {
    setSearchQuery("");
    setSelectedCategory("");
    setSelectedDifficulty("");
  };

  const hasActiveFilters = searchQuery || selectedCategory || selectedDifficulty;

  // Refresh domains
  const refresh = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/domains");
      const data = await response.json();
      setDomains(data.domains || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to refresh");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Filters */}
      {!hideFilters && (
        <div className="flex flex-col sm:flex-row gap-3">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-muted" />
            <input
              type="text"
              placeholder="Search domains..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={cn(
                "w-full pl-9 pr-4 py-2 text-sm",
                "bg-background-card border border-border rounded-lg",
                "text-foreground placeholder:text-foreground-muted",
                "focus:outline-none focus:border-foreground-muted"
              )}
            />
          </div>

          {/* Category filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-muted" />
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className={cn(
                "pl-9 pr-8 py-2 text-sm appearance-none",
                "bg-background-card border border-border rounded-lg",
                "text-foreground",
                "focus:outline-none focus:border-foreground-muted"
              )}
            >
              <option value="">All Categories</option>
              {categories.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </div>

          {/* Difficulty filter */}
          <select
            value={selectedDifficulty}
            onChange={(e) => setSelectedDifficulty(e.target.value)}
            className={cn(
              "px-4 py-2 text-sm appearance-none",
              "bg-background-card border border-border rounded-lg",
              "text-foreground",
              "focus:outline-none focus:border-foreground-muted"
            )}
          >
            <option value="">All Levels</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>

          {/* Clear filters */}
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1.5 px-3 py-2 text-sm text-foreground-muted hover:text-foreground"
            >
              <X className="w-4 h-4" />
              Clear
            </button>
          )}

          {/* Refresh */}
          <button
            onClick={refresh}
            disabled={isLoading}
            className="p-2 text-foreground-muted hover:text-foreground disabled:opacity-50"
            title="Refresh domains"
          >
            <RefreshCw className={cn("w-4 h-4", isLoading && "animate-spin")} />
          </button>
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 text-foreground-muted animate-spin" />
          <span className="ml-2 text-sm text-foreground-muted">
            Loading domains...
          </span>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4">
          <p className="text-sm text-red-400">{error}</p>
          <button
            onClick={refresh}
            className="mt-2 text-xs text-red-400 hover:text-red-300 underline"
          >
            Try again
          </button>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !error && filteredDomains.length === 0 && (
        <div className="text-center py-12">
          <p className="text-foreground-muted">No domains found</p>
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="mt-2 text-sm text-foreground-muted hover:text-foreground underline"
            >
              Clear filters
            </button>
          )}
        </div>
      )}

      {/* Domain grid */}
      {!isLoading && !error && filteredDomains.length > 0 && (
        <div
          className={cn(
            "grid gap-4",
            compact
              ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
              : "grid-cols-1 md:grid-cols-2 xl:grid-cols-3"
          )}
        >
          {filteredDomains.map((domain) => (
            <DomainCard
              key={domain.slug}
              name={domain.name}
              slug={domain.slug}
              description={domain.description}
              difficulty={domain.difficulty}
              primaryColor={domain.primaryColor}
              accentColor={domain.accentColor}
              backgroundStyle={domain.backgroundStyle}
              tags={domain.tags}
              propsCount={domain.propsCount}
              metricsCount={domain.metricsCount}
              isSelected={selectedDomain === domain.slug}
              onClick={onSelectDomain ? () => onSelectDomain(domain.slug) : undefined}
              compact={compact}
            />
          ))}
        </div>
      )}

      {/* Results count */}
      {!isLoading && !error && !hideFilters && (
        <div className="text-xs text-foreground-muted text-center">
          Showing {filteredDomains.length} of {domains.length} domains
        </div>
      )}
    </div>
  );
}

export default DomainBrowser;
