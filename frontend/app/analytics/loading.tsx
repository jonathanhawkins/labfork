/**
 * Loading State for Analytics Page
 *
 * Displayed while the /analytics page is loading.
 * Shows skeleton loaders that match the NetworkHealth component layout.
 */

import {
  NetworkHealthSkeleton,
  NetworkStatsSkeleton,
  ContributorMapSkeleton,
} from "@/components/compute/ComputeLoading";

export default function AnalyticsLoading() {
  return (
    <div className="min-h-screen bg-background">
      {/* Page Header Skeleton */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 bg-muted/50 rounded animate-pulse" />
                <div className="h-7 w-40 bg-muted/50 rounded animate-pulse" />
              </div>
              <div className="h-4 w-64 bg-muted/50 rounded animate-pulse" />
            </div>

            <div className="flex items-center gap-3">
              <div className="h-9 w-[160px] bg-muted/50 rounded animate-pulse" />
              <div className="h-9 w-24 bg-muted/50 rounded animate-pulse" />
            </div>
          </div>

          <div className="flex items-center gap-2 mt-4">
            <div className="w-2 h-2 rounded-full bg-muted/50 animate-pulse" />
            <div className="h-3 w-32 bg-muted/50 rounded animate-pulse" />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-6">
            <NetworkHealthSkeleton />

            {/* Live Dashboard Skeleton */}
            <div className="bg-card border border-border rounded-xl p-6">
              <div className="h-4 w-24 bg-muted/50 rounded animate-pulse mb-4" />
              <NetworkStatsSkeleton />
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            <ContributorMapSkeleton />

            {/* Quick Stats Skeleton */}
            <div className="bg-card border border-border rounded-xl p-6">
              <div className="h-4 w-32 bg-muted/50 rounded animate-pulse mb-4" />
              <div className="space-y-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <div className="h-3 w-24 bg-muted/50 rounded animate-pulse" />
                    <div className="h-5 w-16 bg-muted/50 rounded animate-pulse" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Performance Insights Skeleton */}
        <div className="mt-6 bg-card border border-border rounded-xl p-6">
          <div className="h-4 w-32 bg-muted/50 rounded animate-pulse mb-4" />
          <div className="grid grid-cols-3 gap-6">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="space-y-2">
                <div className="h-3 w-32 bg-muted/50 rounded animate-pulse" />
                <div className="h-8 w-20 bg-muted/50 rounded animate-pulse" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
