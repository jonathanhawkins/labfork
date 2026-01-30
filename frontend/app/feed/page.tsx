/**
 * Activity Feed Page
 *
 * Shows a personalized feed of activity from labs and users the viewer follows.
 */

import { Suspense } from "react";
import { Loader2, FlaskConical, Bell, Settings } from "lucide-react";
import { ActivityFeed } from "@/components/social";

export const metadata = {
  title: "Activity Feed | LabFork",
  description: "Stay updated with the latest research activity from labs you follow",
};

/**
 * Feed filter tabs
 */
function FeedTabs() {
  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-background-darker border border-border">
      <button className="px-3 py-1.5 rounded-lg bg-foreground-bright/10 text-foreground text-sm font-medium">
        All Activity
      </button>
      <button className="px-3 py-1.5 rounded-lg text-foreground-muted text-sm hover:text-foreground hover:bg-foreground-muted/10 transition-colors">
        Results
      </button>
      <button className="px-3 py-1.5 rounded-lg text-foreground-muted text-sm hover:text-foreground hover:bg-foreground-muted/10 transition-colors">
        Comments
      </button>
      <button className="px-3 py-1.5 rounded-lg text-foreground-muted text-sm hover:text-foreground hover:bg-foreground-muted/10 transition-colors">
        Labs
      </button>
    </div>
  );
}

/**
 * Suggested labs sidebar
 */
function SuggestedLabs() {
  // Mock data - in production this would be fetched
  const suggestedLabs = [
    { name: "Prosody Research", owner: "team-speech", stars: 42 },
    { name: "Emotion TTS", owner: "ml-audio", stars: 38 },
    { name: "Voice Cloning v2", owner: "synthesizer", stars: 27 },
  ];

  return (
    <div className="p-4 rounded-lg border border-border">
      <h3 className="text-sm font-medium text-foreground mb-3">
        Suggested Labs
      </h3>
      <div className="space-y-3">
        {suggestedLabs.map((lab) => (
          <div
            key={`${lab.owner}/${lab.name}`}
            className="flex items-center gap-3"
          >
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400">
              <FlaskConical className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                {lab.name}
              </p>
              <p className="text-xs text-foreground-muted">
                @{lab.owner} - {lab.stars} stars
              </p>
            </div>
            <button className="px-2 py-1 rounded text-xs bg-foreground-bright/10 text-foreground-bright hover:bg-foreground-bright/20 transition-colors">
              Star
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Trending topics sidebar
 */
function TrendingTopics() {
  const topics = [
    { tag: "prosody-control", count: 23 },
    { tag: "emotion-tts", count: 18 },
    { tag: "voice-cloning", count: 15 },
    { tag: "speaker-adaptation", count: 12 },
    { tag: "zero-shot", count: 9 },
  ];

  return (
    <div className="p-4 rounded-lg border border-border">
      <h3 className="text-sm font-medium text-foreground mb-3">
        Trending Topics
      </h3>
      <div className="space-y-2">
        {topics.map((topic) => (
          <div
            key={topic.tag}
            className="flex items-center justify-between text-sm"
          >
            <span className="text-foreground-muted hover:text-foreground transition-colors cursor-pointer">
              #{topic.tag}
            </span>
            <span className="text-xs text-foreground-subtle">
              {topic.count} posts
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function FeedPage() {
  return (
    <div className="min-h-screen bg-background">
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
                  Latest updates from labs and researchers you follow
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button className="p-2 rounded-lg text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10 transition-colors">
                  <Bell className="w-5 h-5" />
                </button>
                <button className="p-2 rounded-lg text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10 transition-colors">
                  <Settings className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Filter tabs */}
            <FeedTabs />

            {/* Feed content */}
            <Suspense
              fallback={
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 animate-spin text-foreground-muted" />
                </div>
              }
            >
              <ActivityFeed
                liveUpdates={true}
                updateInterval={30000}
                groupByDate={true}
                emptyMessage="No activity yet. Star some labs to see their updates here!"
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
                <a href="#" className="hover:text-foreground-muted">
                  About
                </a>
                <a href="#" className="hover:text-foreground-muted">
                  Help
                </a>
                <a href="#" className="hover:text-foreground-muted">
                  Privacy
                </a>
                <a href="#" className="hover:text-foreground-muted">
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
