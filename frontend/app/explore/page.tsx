"use client";

/**
 * Explore Labs Page
 *
 * Browse and discover public labs with:
 * - Search and filters
 * - Domain categories
 * - Sort options
 * - Featured labs
 */

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Search,
  Filter,
  SlidersHorizontal,
  Star,
  TrendingUp,
  Clock,
  Activity,
  Mic,
  LineChart,
  Bot,
  Dna,
  Layers,
  ChevronDown,
  X,
  Loader2,
} from "lucide-react";
import { LabCard } from "@/components/labs/LabCard";
import type { Lab } from "@/lib/labs/types";
import { Button } from "@/components/ui/button";

// Domain options
const DOMAINS = [
  { slug: "all", name: "All Domains", icon: Layers, color: "text-foreground-muted" },
  { slug: "firefly-network", name: "Firefly Network", icon: Star, color: "text-amber-400", featured: true },
  { slug: "voice-clone", name: "Voice Clone", icon: Mic, color: "text-blue-400" },
  { slug: "quant-trading", name: "Quant Trading", icon: LineChart, color: "text-green-400" },
  { slug: "robotics", name: "Robotics", icon: Bot, color: "text-purple-400" },
  { slug: "biotech", name: "Biotech", icon: Dna, color: "text-pink-400" },
];

// Sort options
const SORT_OPTIONS = [
  { value: "popular", label: "Most Popular", icon: Star },
  { value: "trending", label: "Trending", icon: TrendingUp },
  { value: "recent", label: "Recently Updated", icon: Clock },
  { value: "active", label: "Most Active", icon: Activity },
];

export default function ExplorePage() {
  const [labs, setLabs] = useState<Lab[]>([]);
  const [featuredLabs, setFeaturedLabs] = useState<Lab[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDomain, setSelectedDomain] = useState("all");
  const [sortBy, setSortBy] = useState("popular");
  const [showFilters, setShowFilters] = useState(false);

  // Pagination
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);

  // Fetch labs
  const fetchLabs = useCallback(async (reset = false) => {
    try {
      if (reset) {
        setPage(1);
        setIsLoading(true);
      }

      const params = new URLSearchParams({
        visibility: "public",
        sortBy,
        page: reset ? "1" : String(page),
        limit: "12",
      });

      if (searchQuery) {
        params.set("search", searchQuery);
      }
      if (selectedDomain !== "all") {
        params.set("domain", selectedDomain);
      }

      const response = await fetch(`/api/labs?${params}`);
      const data = await response.json();

      if (data.success) {
        if (reset) {
          setLabs(data.labs);
        } else {
          setLabs((prev) => [...prev, ...data.labs]);
        }
        setTotal(data.total);
        setHasMore(data.labs.length === 12 && data.labs.length < data.total);
        setError(null);
        setRetryCount(0); // Reset retry count on success
      } else {
        setError(data.error || "Failed to load labs. Please try again.");
        setRetryCount((c) => c + 1);
      }
    } catch (err) {
      console.error("Failed to fetch labs:", err);
      // Distinguish between network errors and API errors
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError("Network error - please check your connection and try again");
      } else if (err instanceof Error) {
        setError(`Failed to load labs: ${err.message}`);
      } else {
        setError("Failed to load labs. Please try again.");
      }
      setRetryCount((c) => c + 1);
    } finally {
      setIsLoading(false);
    }
  }, [searchQuery, selectedDomain, sortBy, page]);

  // Fetch featured labs
  const fetchFeaturedLabs = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        visibility: "public",
        sortBy: "popular",
        limit: "4",
      });

      const response = await fetch(`/api/labs?${params}`);
      const data = await response.json();

      if (data.success) {
        // Mark first few as "featured" for display
        setFeaturedLabs(data.labs.slice(0, 3));
      }
    } catch (err) {
      console.error("Failed to fetch featured labs:", err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchLabs(true);
    fetchFeaturedLabs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchFeaturedLabs]); // fetchLabs excluded intentionally - filter useEffect handles it

  // Refetch on filter change
  useEffect(() => {
    fetchLabs(true);
  }, [fetchLabs]);

  // Load more
  const loadMore = () => {
    if (!isLoading && hasMore) {
      setPage((p) => p + 1);
    }
  };

  useEffect(() => {
    if (page > 1) {
      fetchLabs(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]); // fetchLabs excluded - we only want this to fire on page change

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="border-b border-border bg-gradient-to-b from-foreground-muted/5 to-transparent">
        <div className="max-w-7xl mx-auto px-4 py-12">
          <h1 className="text-3xl font-bold text-foreground-bright mb-2">
            Explore Labs
          </h1>
          <p className="text-lg text-foreground-muted max-w-2xl">
            Discover cutting-edge research labs from the community. Star your favorites, fork to build upon, or create your own.
          </p>

          {/* Search Bar */}
          <div className="mt-6 flex gap-3">
            <div className="relative flex-1 max-w-xl">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-foreground-subtle" />
              <input
                type="text"
                placeholder="Search labs by name, description, or tags..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className={cn(
                  "w-full pl-10 pr-4 py-3 rounded-lg text-sm",
                  "bg-background border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-foreground-muted/10"
                >
                  <X className="w-4 h-4 text-foreground-subtle" />
                </button>
              )}
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={cn(
                "flex items-center gap-2 px-4 py-3 rounded-lg border transition-colors",
                showFilters
                  ? "border-foreground-bright bg-foreground-bright/10 text-foreground-bright"
                  : "border-border hover:bg-foreground-muted/10 text-foreground-muted"
              )}
            >
              <SlidersHorizontal className="w-4 h-4" />
              Filters
            </button>
          </div>

          {/* Filter Panel */}
          {showFilters && (
            <div className="mt-4 p-4 rounded-lg border border-border bg-background">
              <div className="flex flex-wrap gap-6">
                {/* Domain filter */}
                <div>
                  <label className="block text-xs text-foreground-muted mb-2">Domain</label>
                  <div className="flex flex-wrap gap-2">
                    {DOMAINS.map((domain) => (
                      <button
                        key={domain.slug}
                        onClick={() => setSelectedDomain(domain.slug)}
                        className={cn(
                          "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm border transition-colors",
                          selectedDomain === domain.slug
                            ? "border-foreground-bright bg-foreground-bright/10 text-foreground-bright"
                            : "border-border hover:bg-foreground-muted/10 text-foreground-muted"
                        )}
                      >
                        <domain.icon className={cn("w-4 h-4", domain.color)} />
                        {domain.name}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Sort */}
                <div>
                  <label className="block text-xs text-foreground-muted mb-2">Sort by</label>
                  <div className="flex flex-wrap gap-2">
                    {SORT_OPTIONS.map((option) => (
                      <button
                        key={option.value}
                        onClick={() => setSortBy(option.value)}
                        className={cn(
                          "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm border transition-colors",
                          sortBy === option.value
                            ? "border-foreground-bright bg-foreground-bright/10 text-foreground-bright"
                            : "border-border hover:bg-foreground-muted/10 text-foreground-muted"
                        )}
                      >
                        <option.icon className="w-4 h-4" />
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Featured Labs (only show when no filters) */}
        {!searchQuery && selectedDomain === "all" && featuredLabs.length > 0 && (
          <div className="mb-12">
            <h2 className="text-lg font-medium text-foreground-bright mb-4 flex items-center gap-2">
              <Star className="w-5 h-5 text-yellow-400" />
              Featured Labs
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {featuredLabs.map((lab) => (
                <LabCard key={lab.id} lab={lab} />
              ))}
            </div>
          </div>
        )}

        {/* Results Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium text-foreground">
            {searchQuery
              ? `Search results for "${searchQuery}"`
              : selectedDomain !== "all"
              ? DOMAINS.find((d) => d.slug === selectedDomain)?.name
              : "All Labs"}
            {total > 0 && (
              <span className="ml-2 text-sm text-foreground-muted font-normal">
                ({total} labs)
              </span>
            )}
          </h2>
        </div>

        {/* Loading State */}
        {isLoading && labs.length === 0 && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="text-center py-12">
            <p className="text-red-400 mb-2">{error}</p>
            {retryCount > 1 && (
              <p className="text-sm text-foreground-muted mb-4">
                Tried {retryCount} times. If this persists, the service may be temporarily unavailable.
              </p>
            )}
            <Button
              variant="outline"
              onClick={() => {
                setRetryCount((c) => c + 1);
                fetchLabs(true);
              }}
            >
              Try Again
            </Button>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !error && labs.length === 0 && (
          <div className="text-center py-12">
            <Layers className="w-12 h-12 mx-auto text-foreground-subtle mb-4" />
            <h3 className="text-lg font-medium text-foreground-bright mb-2">
              No labs found
            </h3>
            <p className="text-sm text-foreground-muted max-w-md mx-auto">
              {searchQuery
                ? "Try adjusting your search or filters"
                : "Be the first to create a lab in this domain!"}
            </p>
          </div>
        )}

        {/* Labs Grid */}
        {labs.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {labs.map((lab) => (
              <LabCard key={lab.id} lab={lab} />
            ))}
          </div>
        )}

        {/* Load More */}
        {hasMore && labs.length > 0 && (
          <div className="mt-8 text-center">
            <button
              onClick={loadMore}
              disabled={isLoading}
              className={cn(
                "px-6 py-3 rounded-lg border border-border text-sm transition-colors",
                isLoading
                  ? "opacity-50 cursor-not-allowed"
                  : "hover:bg-foreground-muted/10"
              )}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading...
                </span>
              ) : (
                "Load More"
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
