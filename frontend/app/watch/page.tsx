"use client";

import { useState, useCallback, useEffect } from "react";
import PublicLabView from "@/components/lab/PublicLabView";
import SuggestionForm from "@/components/lab/SuggestionForm";
import SuggestionList from "@/components/lab/SuggestionList";
import {
  Lightbulb,
  ChevronDown,
  ChevronUp,
  MessageSquarePlus,
  ListChecks,
  Github,
  Eye,
} from "lucide-react";

/**
 * Hook to track and display live viewer count
 * Fetches real viewer count from /api/viewers
 */
function useViewerCount() {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    // Fetch initial count
    const fetchCount = async () => {
      try {
        const response = await fetch("/api/viewers");
        const data = await response.json();
        setCount(data.count);
      } catch {
        // No fake counts - show 0 if API unavailable
        setCount(0);
      }
    };

    fetchCount();

    // Poll every 30 seconds
    const interval = setInterval(fetchCount, 30000);
    return () => clearInterval(interval);
  }, []);

  return count;
}

/**
 * Public AI Lab Viewer
 *
 * This page is designed for public deployment (e.g., Vercel)
 * It shows:
 * - Live 3D visualization of AI agents working
 * - Sanitized GPU stats (no IPs or sensitive info)
 * - Community suggestion submission
 * - Public suggestion queue
 *
 * No admin controls, no task creation, read-only view.
 */
export default function WatchPage() {
  const [suggestionPanelOpen, setSuggestionPanelOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"suggest" | "queue">("queue");
  const [refreshKey, setRefreshKey] = useState(0);
  const viewerCount = useViewerCount();

  const handleSuggestionSubmitted = useCallback(() => {
    // Refresh the list when a new suggestion is submitted
    setRefreshKey((k) => k + 1);
    // Switch to queue tab to show the new suggestion
    setActiveTab("queue");
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Public Header */}
      <header className="border-b border-border bg-background-elevated">
        <div className="max-w-[1600px] mx-auto px-3 sm:px-4 py-2 sm:py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 sm:gap-3">
              <div className="w-6 h-6 sm:w-8 sm:h-8 rounded border border-border flex items-center justify-center">
                <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-foreground animate-pulse" />
              </div>
              <div>
                <h1 className="text-sm sm:text-base font-semibold bg-gradient-to-r from-blue-400 to-purple-400 text-transparent bg-clip-text">
                  LabFork
                </h1>
                <p className="text-[10px] sm:text-xs text-muted-foreground">
                  Live Research Lab View
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-4">
              {/* Live viewer count */}
              {viewerCount !== null && (
                <div className="flex items-center gap-1.5 px-2 py-1 text-xs text-muted-foreground">
                  <Eye className="w-3.5 h-3.5" />
                  <span className="tabular-nums">{viewerCount}</span>
                  <span className="hidden sm:inline">watching</span>
                </div>
              )}

              {/* Toggle suggestion panel */}
              <button
                onClick={() => setSuggestionPanelOpen(!suggestionPanelOpen)}
                className={`flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded border transition-colors ${
                  suggestionPanelOpen
                    ? "border-foreground text-foreground"
                    : "border-border text-muted-foreground hover:text-foreground hover:border-foreground-muted"
                }`}
              >
                <Lightbulb className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span className="text-xs sm:text-sm hidden sm:inline">Suggestions</span>
                {suggestionPanelOpen ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
              </button>

              {/* GitHub link */}
              <a
                href="https://github.com/your-org/voice-clone-pipeline"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded border border-border text-muted-foreground hover:text-foreground hover:border-foreground-muted transition-colors"
              >
                <Github className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span className="text-xs sm:text-sm hidden sm:inline">GitHub</span>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Suggestion Panel (collapsible) */}
      {suggestionPanelOpen && (
        <div className="border-b border-border bg-background-elevated animate-in slide-in-from-top duration-200">
          <div className="max-w-[1600px] mx-auto px-4 py-4">
            <div className="flex flex-col lg:flex-row gap-6">
              {/* Tab navigation */}
              <div className="flex lg:flex-col gap-2 lg:w-[180px]">
                <button
                  onClick={() => setActiveTab("queue")}
                  className={`flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors ${
                    activeTab === "queue"
                      ? "bg-foreground/10 text-foreground-bright"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <ListChecks className="w-4 h-4" />
                  <span>Suggestion Queue</span>
                </button>
                <button
                  onClick={() => setActiveTab("suggest")}
                  className={`flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors ${
                    activeTab === "suggest"
                      ? "bg-foreground/10 text-foreground-bright"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <MessageSquarePlus className="w-4 h-4" />
                  <span>Submit Suggestion</span>
                </button>
              </div>

              {/* Content area */}
              <div className="flex-1 min-w-0">
                {activeTab === "queue" ? (
                  <div className="max-h-[400px] overflow-y-auto pr-2">
                    <SuggestionList
                      key={refreshKey}
                      limit={10}
                      onRefresh={() => setRefreshKey((k) => k + 1)}
                    />
                  </div>
                ) : (
                  <div className="max-w-lg">
                    <div className="mb-4">
                      <h3 className="text-sm text-foreground-bright">
                        Have an idea?
                      </h3>
                      <p className="text-xs text-muted-foreground mt-1">
                        Submit a feature request, improvement, or bug report.
                        Approved suggestions get added to the task queue for our
                        AI agents to implement.
                      </p>
                    </div>
                    <SuggestionForm onSubmitted={handleSuggestionSubmitted} />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Lab View */}
      <main className="flex-1">
        <PublicLabView />
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-background-elevated">
        <div className="max-w-[1600px] mx-auto px-4 py-3">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-4">
              <span className="font-medium">LabFork</span>
              <span className="text-foreground-subtle">|</span>
              <span>Fork. Watch. Discover.</span>
            </div>
            <div className="flex items-center gap-4">
              <a
                href="/lab"
                className="hover:text-foreground transition-colors"
              >
                Full Lab (Admin)
              </a>
              <a
                href="https://github.com/labfork"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                Source Code
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
