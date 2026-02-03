"use client";

/**
 * FeedPageClient Component
 *
 * Client-side wrapper for the feed page that handles:
 * - Auto-seeding demo data on first visit
 * - Filter tab state
 * - Refresh on seed completion
 */

import { useState, useCallback, useEffect } from "react";
import { Suspense } from "react";
import { useRouter } from "next/navigation";
import { Loader2, FlaskConical, Bell, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { ActivityFeed } from "./ActivityFeed";
import { FeedSeeder } from "./FeedSeeder";
import type { ActivityType } from "@/lib/social/activity";

type FilterTab = "all" | "results" | "comments" | "labs";

const FILTER_TABS: { id: FilterTab; label: string; types?: ActivityType[] }[] = [
  { id: "all", label: "All Activity" },
  { id: "results", label: "Results", types: ["result_created", "result_published", "result_liked"] },
  { id: "comments", label: "Comments", types: ["comment_added", "result_commented"] },
  { id: "labs", label: "Labs", types: ["lab_starred", "lab_forked", "lab_created"] },
];

// Fallback demo data for suggested labs
const DEMO_LABS = [
  { id: "1", name: "Firefly Network", owner: "spark_research", stars: 128, slug: "firefly-network" },
  { id: "2", name: "Voice Clone Lab", owner: "voice_pioneer", stars: 89, slug: "voice-clone" },
  { id: "3", name: "Quant Trading", owner: "ai_researcher", stars: 67, slug: "quant-trading" },
];

/**
 * Suggested labs sidebar
 */
function SuggestedLabs() {
  const [suggestedLabs, setSuggestedLabs] = useState<Array<{
    id: string;
    name: string;
    owner: string;
    stars: number;
    slug: string;
  }>>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchSuggestedLabs() {
      try {
        const response = await fetch("/api/labs?sortBy=stars&limit=3&visibility=public");
        if (!response.ok) {
          setSuggestedLabs(DEMO_LABS);
          return;
        }

        const data = await response.json();
        if (data.success && data.labs && data.labs.length > 0) {
          const mapped = data.labs.map((lab: {
            id: string;
            name: string;
            slug: string;
            owner: { username: string };
            stats: { stars: number };
          }) => ({
            id: lab.id,
            name: lab.name,
            owner: lab.owner.username,
            stars: lab.stats.stars,
            slug: lab.slug,
          }));
          setSuggestedLabs(mapped);
        } else {
          setSuggestedLabs(DEMO_LABS);
        }
      } catch (error) {
        console.error("Failed to fetch suggested labs:", error);
        setSuggestedLabs(DEMO_LABS);
      } finally {
        setIsLoading(false);
      }
    }

    fetchSuggestedLabs();
  }, []);

  if (isLoading) {
    return (
      <div className="p-4 rounded-lg border border-border">
        <h3 className="text-sm font-medium text-foreground mb-3">
          Suggested Labs
        </h3>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-5 h-5 animate-spin text-foreground-muted" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 rounded-lg border border-border">
      <h3 className="text-sm font-medium text-foreground mb-3">
        Suggested Labs
      </h3>
      <div className="space-y-3">
        {suggestedLabs.map((lab) => (
          <a
            key={`${lab.owner}/${lab.slug}`}
            href={`/labs/${lab.owner}/${lab.slug}`}
            className="flex items-center gap-3 group"
          >
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center text-amber-400">
              <FlaskConical className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate group-hover:text-foreground-bright transition-colors">
                {lab.name}
              </p>
              <p className="text-xs text-foreground-muted">
                @{lab.owner} · {lab.stars} stars
              </p>
            </div>
            <button
              className="px-3 py-1 rounded text-xs min-h-[44px] min-w-[44px] bg-foreground-bright/10 text-foreground-bright hover:bg-foreground-bright/20 transition-colors"
              aria-label={`Star ${lab.name}`}
              onClick={(e) => {
                e.preventDefault();
                // TODO: Implement star functionality
              }}
            >
              Star
            </button>
          </a>
        ))}
      </div>
    </div>
  );
}

/**
 * Trending topics sidebar
 * NOTE: These are curated editorial topics until we have enough real activity
 * to compute trending tags dynamically from lab tags and activity data.
 */
function TrendingTopics() {
  // Curated trending topics (editorial)
  // TODO: Replace with API call once we have /api/topics/trending
  // which aggregates tags from recent labs, results, and activity
  const topics = [
    { tag: "mppt-algorithm", count: 34 },
    { tag: "mesh-networking", count: 28 },
    { tag: "voice-cloning", count: 23 },
    { tag: "emotion-tts", count: 18 },
    { tag: "zero-shot", count: 12 },
  ];

  return (
    <div className="p-4 rounded-lg border border-border">
      <h3 className="text-sm font-medium text-foreground mb-3">
        Trending Topics
      </h3>
      <div className="space-y-2">
        {topics.map((topic) => (
          <a
            key={topic.tag}
            href={`/explore?tag=${topic.tag}`}
            className="flex items-center justify-between text-sm hover:bg-foreground-muted/5 px-2 py-1 rounded -mx-2 transition-colors"
          >
            <span className="text-foreground-muted hover:text-foreground transition-colors">
              #{topic.tag}
            </span>
            <span className="text-xs text-foreground-subtle">
              {topic.count} posts
            </span>
          </a>
        ))}
      </div>
    </div>
  );
}

export function FeedPageClient() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<FilterTab>("all");
  const [refreshKey, setRefreshKey] = useState(0);

  // When seeding completes, refresh the feed
  const handleSeeded = useCallback(() => {
    setRefreshKey((k) => k + 1);
    router.refresh();
  }, [router]);

  // Get filter types for current tab
  const filterTypes = FILTER_TABS.find((t) => t.id === activeTab)?.types;

  return (
    <div className="min-h-screen bg-background">
      {/* Auto-seed on first visit */}
      <FeedSeeder onSeeded={handleSeeded} />

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Main Feed */}
          <div className="lg:col-span-3 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-foreground">
                  Activity Feed
                </h1>
                <p className="text-foreground-muted text-sm mt-1">
                  Latest updates from labs and researchers
                </p>
              </div>

              <div className="flex items-center gap-2">
                <a
                  href="/notifications"
                  className="p-2 min-w-[44px] min-h-[44px] rounded-lg text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10 transition-colors flex items-center justify-center"
                  aria-label="View notifications"
                >
                  <Bell className="w-5 h-5" />
                </a>
                <button
                  className="p-2 min-w-[44px] min-h-[44px] rounded-lg text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10 transition-colors cursor-not-allowed opacity-50"
                  aria-label="Feed settings (coming soon)"
                  title="Feed settings coming soon"
                  disabled
                >
                  <Settings className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Filter tabs */}
            <div className="flex items-center gap-1 p-1 rounded-lg bg-background-darker border border-border overflow-x-auto">
              {FILTER_TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    "px-3 py-1.5 min-h-[44px] rounded-lg text-sm whitespace-nowrap transition-colors",
                    activeTab === tab.id
                      ? "bg-foreground-bright/10 text-foreground font-medium"
                      : "text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10"
                  )}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Feed content */}
            <Suspense
              fallback={
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 animate-spin text-foreground-muted" />
                </div>
              }
            >
              <ActivityFeed
                key={refreshKey}
                liveUpdates={true}
                updateInterval={30000}
                groupByDate={true}
                types={filterTypes}
                emptyMessage={
                  activeTab === "all"
                    ? "No activity yet. Star some labs to see their updates here!"
                    : `No ${activeTab} activity to show.`
                }
              />
            </Suspense>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <SuggestedLabs />
            <TrendingTopics />

            {/* Footer links */}
            <div className="text-xs text-foreground-subtle space-y-1">
              <p className="font-medium">LabFork</p>
              <p>Fork. Watch. Discover.</p>
              <div className="flex gap-2 pt-2">
                <a href="/about" className="hover:text-foreground-muted min-h-[44px] flex items-center">
                  About
                </a>
                <a href="/help" className="hover:text-foreground-muted min-h-[44px] flex items-center">
                  Help
                </a>
                <a href="/privacy" className="hover:text-foreground-muted min-h-[44px] flex items-center">
                  Privacy
                </a>
                <a href="/terms" className="hover:text-foreground-muted min-h-[44px] flex items-center">
                  Terms
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FeedPageClient;
