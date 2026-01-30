"use client";

/**
 * ActivityFeed Component
 *
 * Displays a list of activities with infinite scroll and live updates.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { cn } from "@/lib/utils";
import { Loader2, RefreshCw, Bell } from "lucide-react";
import type {
  AggregatedActivity,
  ActivityType,
  ActivityFeedResult,
} from "@/lib/social/activity/types";
import { groupActivitiesByDate, getDateGroupLabel } from "@/lib/social/activity/types";
import { FeedItem } from "./FeedItem";

export interface ActivityFeedProps {
  /** Lab ID to filter by */
  labId?: string;
  /** User ID to filter by */
  userId?: string;
  /** Activity types to show */
  types?: ActivityType[];
  /** Enable live updates */
  liveUpdates?: boolean;
  /** Update interval in ms (default 30000) */
  updateInterval?: number;
  /** Compact mode */
  compact?: boolean;
  /** Group by date */
  groupByDate?: boolean;
  /** Show empty state */
  showEmpty?: boolean;
  /** Empty state message */
  emptyMessage?: string;
  /** Custom class name */
  className?: string;
}

export function ActivityFeed({
  labId,
  userId,
  types,
  liveUpdates = true,
  updateInterval = 30000,
  compact = false,
  groupByDate = true,
  showEmpty = true,
  emptyMessage = "No activity yet",
  className,
}: ActivityFeedProps) {
  const [activities, setActivities] = useState<AggregatedActivity[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [page, setPage] = useState(1);
  const [newCount, setNewCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const latestTimestampRef = useRef<string | null>(null);
  const loadMoreRef = useRef<HTMLDivElement>(null);

  // Build query string
  const buildQueryString = useCallback(
    (pageNum: number, since?: string) => {
      const params = new URLSearchParams();

      if (labId) params.set("labId", labId);
      if (userId) params.set("userId", userId);
      if (types && types.length > 0) params.set("types", types.join(","));
      if (since) {
        params.set("since", since);
      } else {
        params.set("page", pageNum.toString());
        params.set("limit", "20");
        params.set("aggregate", "true");
      }

      return params.toString();
    },
    [labId, userId, types]
  );

  // Fetch activities
  const fetchActivities = useCallback(
    async (pageNum = 1, append = false) => {
      try {
        if (pageNum === 1 && !append) {
          setIsLoading(true);
        } else {
          setIsLoadingMore(true);
        }

        const query = buildQueryString(pageNum);
        const response = await fetch(`/api/activity?${query}`);

        if (!response.ok) {
          throw new Error("Failed to fetch activities");
        }

        const data: ActivityFeedResult = await response.json();

        if (append) {
          setActivities((prev) => [...prev, ...data.activities]);
        } else {
          setActivities(data.activities);
        }

        setHasMore(data.hasMore);
        setPage(data.page);

        if (data.latestTimestamp) {
          latestTimestampRef.current = data.latestTimestamp;
        }

        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setIsLoading(false);
        setIsLoadingMore(false);
      }
    },
    [buildQueryString]
  );

  // Check for new activities
  const checkForNew = useCallback(async () => {
    if (!latestTimestampRef.current) return;

    try {
      const query = buildQueryString(1, latestTimestampRef.current);
      const response = await fetch(`/api/activity?${query}`);

      if (!response.ok) return;

      const data = await response.json();

      if (data.count > 0) {
        setNewCount(data.count);
      }
    } catch {
      // Silently fail polling
    }
  }, [buildQueryString]);

  // Load new activities
  const loadNew = useCallback(() => {
    setNewCount(0);
    fetchActivities(1);
  }, [fetchActivities]);

  // Load more activities
  const loadMore = useCallback(() => {
    if (!isLoadingMore && hasMore) {
      fetchActivities(page + 1, true);
    }
  }, [fetchActivities, isLoadingMore, hasMore, page]);

  // Initial fetch
  useEffect(() => {
    fetchActivities(1);
  }, [fetchActivities]);

  // Live updates polling
  useEffect(() => {
    if (!liveUpdates) return;

    const interval = setInterval(checkForNew, updateInterval);
    return () => clearInterval(interval);
  }, [liveUpdates, updateInterval, checkForNew]);

  // Infinite scroll observer
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoadingMore) {
          loadMore();
        }
      },
      { threshold: 0.5 }
    );

    if (loadMoreRef.current) {
      observer.observe(loadMoreRef.current);
    }

    return () => observer.disconnect();
  }, [hasMore, isLoadingMore, loadMore]);

  // Group activities by date if enabled
  const groupedActivities = groupByDate
    ? groupActivitiesByDate(activities)
    : null;

  // Loading state
  if (isLoading) {
    return (
      <div className={cn("flex items-center justify-center py-8", className)}>
        <Loader2 className="w-6 h-6 animate-spin text-foreground-muted" />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={cn("text-center py-8", className)}>
        <p className="text-red-400 text-sm mb-2">{error}</p>
        <button
          onClick={() => fetchActivities(1)}
          className="text-sm text-foreground-muted hover:text-foreground transition-colors"
        >
          Try again
        </button>
      </div>
    );
  }

  // Empty state
  if (activities.length === 0 && showEmpty) {
    return (
      <div className={cn("text-center py-8", className)}>
        <p className="text-foreground-muted text-sm">{emptyMessage}</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-1", className)}>
      {/* New activities banner */}
      {newCount > 0 && (
        <button
          onClick={loadNew}
          className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-blue-500/10 text-blue-400 text-sm hover:bg-blue-500/20 transition-colors"
        >
          <Bell className="w-4 h-4" />
          {newCount} new {newCount === 1 ? "activity" : "activities"}
        </button>
      )}

      {/* Activities list */}
      {groupByDate && groupedActivities ? (
        // Grouped by date
        Array.from(groupedActivities.entries()).map(([date, dateActivities]) => (
          <div key={date}>
            <div className="sticky top-0 bg-background/95 backdrop-blur-sm py-2 z-10">
              <span className="text-xs font-medium text-foreground-muted">
                {getDateGroupLabel(date)}
              </span>
            </div>
            <div className="space-y-1">
              {dateActivities.map((activity) => (
                <FeedItem
                  key={activity.id}
                  activity={activity as AggregatedActivity}
                  compact={compact}
                  showTime={true}
                />
              ))}
            </div>
          </div>
        ))
      ) : (
        // Flat list
        activities.map((activity) => (
          <FeedItem
            key={activity.id}
            activity={activity}
            compact={compact}
            showTime={true}
          />
        ))
      )}

      {/* Load more trigger */}
      <div ref={loadMoreRef} className="py-4">
        {isLoadingMore && (
          <div className="flex items-center justify-center">
            <Loader2 className="w-5 h-5 animate-spin text-foreground-muted" />
          </div>
        )}
      </div>

      {/* Manual load more button */}
      {hasMore && !isLoadingMore && (
        <button
          onClick={loadMore}
          className="w-full py-2 text-sm text-foreground-muted hover:text-foreground transition-colors"
        >
          Load more
        </button>
      )}
    </div>
  );
}

export default ActivityFeed;
