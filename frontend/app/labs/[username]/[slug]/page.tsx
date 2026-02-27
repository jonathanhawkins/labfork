"use client";

/**
 * Lab Portal Page
 *
 * Main lab viewing page with:
 * - Lab header with stats and actions
 * - 3D visualization (if available)
 * - Activity feed
 * - Tasks list
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth/hooks";
import { cn } from "@/lib/utils";
import {
  Activity,
  Clock,
  FileText,
  GitFork,
  ListTodo,
  Loader2,
  MessageSquare,
  Settings,
  Terminal,
  Lightbulb,
} from "lucide-react";
import { LabHeader } from "@/components/labs/LabHeader";
import { ShareDialog } from "@/components/labs/ShareDialog";
import { quickForkLab } from "@/components/labs/ForkDialog";
import { FireflyLabContent } from "@/components/labs/FireflyLabContent";
import { LiveLabViewer } from "@/components/labs/LiveLabViewer";
import { ActivityFeed } from "@/components/social/ActivityFeed";
import { LabDemosSection } from "@/components/demos";
import type { Lab } from "@/lib/labs/types";
import {
  CheckCircle2,
  Circle,
  AlertCircle,
} from "lucide-react";

interface Task {
  id: string;
  subject: string;
  description?: string;
  status: "pending" | "in_progress" | "completed";
  owner?: string;
  blockedBy?: string[];
  blocks?: string[];
  activeForm?: string;
  createdAt?: string;
  updatedAt?: string;
  /** Summary from Nudge Engine result */
  resultSummary?: string;
  /** When the task completed */
  completedAt?: string;
  /** Whether the task failed (shown as completed with error) */
  isFailed?: boolean;
  /** Nudge Engine task ID */
  nudgeTaskId?: string;
  /** Parent task ID (subtask indicator) */
  parentTaskId?: string;
}

/** Format relative time, e.g. "5m ago", "2h ago" */
function relativeTime(iso: string | undefined): string {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

interface LabPortalPageProps {
  params: {
    username: string;
    slug: string;
  };
}

type TabId = "overview" | "tasks" | "activity" | "settings";

export default function LabPortalPage({ params }: LabPortalPageProps) {
  const { username, slug } = params;
  const router = useRouter();

  // Authentication (works with Clerk on prod, mock in dev)
  const { user: currentUser, isLoaded: isUserLoaded } = useAuth();
  const currentUserRef = useRef(currentUser);
  currentUserRef.current = currentUser;

  const [lab, setLab] = useState<Lab | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStarred, setIsStarred] = useState(false);
  const [isOwner, setIsOwner] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  // Tasks state
  const [tasks, setTasks] = useState<Task[]>([]);
  const [tasksLoading, setTasksLoading] = useState(false);

  // Fetch lab data (stable callback — uses ref for currentUser to avoid re-fetches)
  const fetchLab = useCallback(async () => {
    try {
      setError(null);

      // First, find the lab by username/slug
      const searchParams = new URLSearchParams({
        owner: username,
        slug: slug,
      });
      const response = await fetch(`/api/labs?${searchParams}`);
      const data = await response.json();

      if (!data.success || data.labs.length === 0) {
        setError("Lab not found");
        return;
      }

      const foundLab = data.labs[0];

      // Fetch full lab details with social stats
      const detailResponse = await fetch(`/api/labs/${foundLab.id}`);
      const detailData = await detailResponse.json();

      if (detailData.success) {
        setLab(detailData.lab);
        setIsStarred(detailData.social?.isStarred || false);

        // Check if current user is owner using ref (avoids callback invalidation)
        setIsOwner(currentUserRef.current?.id === detailData.lab.owner.id);
      } else {
        setError(detailData.error || "Failed to load lab");
      }
    } catch (err) {
      console.error("Failed to fetch lab:", err);
      setError("Failed to load lab");
    } finally {
      setIsLoading(false);
    }
  }, [username, slug]);

  // Fetch lab once auth is ready (stable deps — won't re-fire on currentUser change)
  useEffect(() => {
    if (isUserLoaded) {
      fetchLab();
    }
  }, [isUserLoaded, fetchLab]);

  // Fetch research tasks for this lab from Nudge Engine
  const fetchTasks = useCallback(async () => {
    if (!lab) return;

    try {
      setTasksLoading(true);
      const response = await fetch(`/api/labs/${lab.id}/research`);
      const data = await response.json();

      if (data.success && data.tasks) {
        setTasks(data.tasks);
        // Update sidebar stats with real task count
        if (data.total !== undefined) {
          setLab((prev) =>
            prev
              ? { ...prev, stats: { ...prev.stats, tasks: data.total } }
              : prev
          );
        }
      }
    } catch (err) {
      console.error("Failed to fetch tasks:", err);
    } finally {
      setTasksLoading(false);
    }
  }, [lab?.id]);

  // Always fetch tasks on initial load (needed for Overview + Tasks tab)
  useEffect(() => {
    if (lab) {
      fetchTasks();
    }
  }, [lab, fetchTasks]);

  // Poll for live task updates every 10 seconds when on Tasks tab
  useEffect(() => {
    if (activeTab !== "tasks" || !lab) return;
    const interval = setInterval(fetchTasks, 10_000);
    return () => clearInterval(interval);
  }, [activeTab, lab, fetchTasks]);

  // Handle star toggle
  const handleStarToggle = (starred: boolean, count: number) => {
    setIsStarred(starred);
    if (lab) {
      setLab({
        ...lab,
        stats: { ...lab.stats, stars: count },
      });
    }
  };

  // Handle fork success
  const handleForkSuccess = (forkedLab: Lab) => {
    router.push(`/labs/${forkedLab.owner.username}/${forkedLab.slug}`);
  };

  // Skeleton layout while loading — matches final page structure to prevent flash
  if (!isUserLoaded || isLoading) {
    return (
      <div className="min-h-screen bg-background">
        {/* Header skeleton */}
        <div className="border-b border-border">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <div className="flex items-start gap-3 sm:gap-4">
              <div className="w-10 h-10 sm:w-14 sm:h-14 rounded-lg sm:rounded-xl bg-foreground-muted/10 animate-pulse" />
              <div className="flex-1 space-y-2">
                <div className="h-6 sm:h-7 w-48 bg-foreground-muted/10 rounded animate-pulse" />
                <div className="h-4 w-32 bg-foreground-muted/10 rounded animate-pulse" />
              </div>
            </div>
          </div>
        </div>
        {/* Tab bar skeleton */}
        <div className="border-b border-border">
          <div className="max-w-7xl mx-auto px-4">
            <div className="flex gap-4 py-3">
              <div className="h-4 w-20 bg-foreground-muted/10 rounded animate-pulse" />
              <div className="h-4 w-16 bg-foreground-muted/10 rounded animate-pulse" />
              <div className="h-4 w-18 bg-foreground-muted/10 rounded animate-pulse" />
            </div>
          </div>
        </div>
        {/* Content skeleton */}
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8">
            <div className="lg:col-span-2 space-y-6">
              <div className="h-[200px] sm:h-[280px] lg:h-[400px] rounded-lg bg-foreground-muted/10 animate-pulse" />
              <div className="h-32 rounded-lg bg-foreground-muted/10 animate-pulse" />
            </div>
            <div className="space-y-6">
              <div className="h-24 rounded-lg bg-foreground-muted/10 animate-pulse" />
              <div className="h-16 rounded-lg bg-foreground-muted/10 animate-pulse" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !lab) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-foreground-bright mb-2">Lab not found</h1>
          <p className="text-foreground-muted mb-4">
            The lab "{username}/{slug}" does not exist or you do not have access.
          </p>
          <button
            onClick={() => router.push("/explore")}
            className="px-4 py-2 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors"
          >
            Browse Labs
          </button>
        </div>
      </div>
    );
  }

  const isIdea = lab.status === "idea";
  const allStatsZero = lab.stats.stars === 0 && lab.stats.forks === 0 && lab.stats.tasks === 0 && lab.stats.papers === 0 && lab.stats.experiments === 0;

  // Tab content
  const tabs: { id: TabId; label: string; icon: typeof Activity }[] = [
    { id: "overview", label: "Overview", icon: FileText },
    { id: "tasks", label: "Tasks", icon: ListTodo },
    ...(isIdea ? [] : [{ id: "activity" as const, label: "Activity", icon: Activity }]),
    ...(isOwner ? [{ id: "settings" as const, label: "Settings", icon: Settings }] : []),
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Lab Header */}
      <LabHeader
        lab={lab}
        isStarred={isStarred}
        isOwner={isOwner}
        onStarToggle={handleStarToggle}
        onForkSuccess={handleForkSuccess}
        onShareClick={() => setShowShareDialog(true)}
        onSettingsClick={() => setActiveTab("settings")}
      />

      {/* Tab Navigation */}
      <div className="border-b border-border">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 -mb-px">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex items-center gap-2 px-4 py-3 text-sm border-b-2 transition-colors",
                  activeTab === tab.id
                    ? "border-foreground-bright text-foreground-bright"
                    : "border-transparent text-foreground-muted hover:text-foreground hover:border-foreground-muted/50"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div className="space-y-8">
            {/* Firefly Lab Special Content */}
            {lab.domainSlug === "firefly-network" && (
              <div className="p-6 rounded-lg border border-amber-500/30 bg-amber-500/5">
                <div className="flex items-center gap-2 mb-4">
                  <span className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
                  <span className="text-sm font-medium text-amber-400">
                    Live Research Lab
                  </span>
                </div>
                <FireflyLabContent />
              </div>
            )}

            {/* Idea CTA */}
            {isIdea && (
              <div className="p-6 rounded-lg border border-dashed border-amber-500/30 bg-amber-500/5 text-center">
                <Lightbulb className="w-10 h-10 mx-auto text-amber-400 mb-3" />
                <h3 className="text-lg font-medium text-foreground-bright mb-2">
                  This is an open research idea
                </h3>
                <p className="text-sm text-foreground-muted mb-4 max-w-md mx-auto">
                  Fork this lab to start building. Your work will contribute to solving real problems.
                </p>
                <button
                  onClick={async () => {
                    const result = await quickForkLab(lab);
                    if (result.success && result.lab) {
                      handleForkSuccess(result.lab);
                    }
                  }}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-foreground text-background hover:bg-foreground-bright transition-colors font-medium text-sm"
                >
                  <GitFork className="w-4 h-4" />
                  Fork this lab to start building
                </button>
              </div>
            )}

            {/* Standard lab content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8">
              {/* Main content */}
              <div className="lg:col-span-2 space-y-6">
                {/* Research Demos — primary content above the 3D viewer */}
                <LabDemosSection labSlug={lab.slug} />

                {/* Lab 3D Viewer — compact on mobile, full on desktop */}
                <div className="rounded-lg border border-border bg-background-elevated overflow-hidden">
                  <LiveLabViewer
                    lab={lab}
                    readOnly={!isOwner}
                    showViewers={true}
                    showActivity={true}
                    allowFullscreen={true}
                    compact={true}
                    className="h-[200px] sm:h-[280px] lg:h-[400px]"
                  />
                </div>

                {/* Recent Research Tasks (on Overview) */}
                {tasks.length > 0 && (
                  <div className="p-4 sm:p-6 rounded-lg border border-border">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-base sm:text-lg font-medium text-foreground-bright">
                        Research Progress
                      </h3>
                      <button
                        onClick={() => setActiveTab("tasks")}
                        className="text-xs text-foreground-muted hover:text-foreground transition-colors"
                      >
                        View all ({tasks.length})
                      </button>
                    </div>
                    <div className="space-y-3">
                      {tasks.slice(0, 5).map((task) => (
                        <div
                          key={task.id}
                          className={cn(
                            "p-3 rounded-lg border transition-colors",
                            task.status === "in_progress"
                              ? "border-blue-500/40 bg-blue-500/5"
                              : task.isFailed
                                ? "border-red-500/30 bg-red-500/5"
                                : task.status === "completed"
                                  ? "border-green-500/20 bg-green-500/5"
                                  : "border-border"
                          )}
                        >
                          <div className="flex items-start gap-2.5">
                            <div className="mt-0.5 flex-shrink-0">
                              {task.isFailed ? (
                                <AlertCircle className="w-4 h-4 text-red-400" />
                              ) : task.status === "completed" ? (
                                <CheckCircle2 className="w-4 h-4 text-green-400" />
                              ) : task.status === "in_progress" ? (
                                <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                              ) : (
                                <Circle className="w-4 h-4 text-foreground-subtle" />
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <h4 className="text-sm font-medium text-foreground-bright truncate">
                                {task.subject}
                              </h4>
                              {task.resultSummary && task.status === "completed" && !task.isFailed && (
                                <p className="text-xs text-green-300/80 mt-1 line-clamp-2">
                                  {task.resultSummary}
                                </p>
                              )}
                              {task.status === "in_progress" && (
                                <p className="text-xs text-blue-400/80 mt-1 italic">
                                  AI agent working...
                                </p>
                              )}
                              <span className="text-[11px] text-foreground-subtle mt-1 block">
                                {relativeTime(task.completedAt || task.createdAt)}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* README/Description */}
                {lab.description && (
                  <div className="p-4 sm:p-6 rounded-lg border border-border">
                    <h3 className="text-base sm:text-lg font-medium text-foreground-bright mb-3 sm:mb-4">About</h3>
                    <p className="text-sm text-foreground-muted whitespace-pre-wrap">
                      {lab.description}
                    </p>
                  </div>
                )}

                {/* README (if available) */}
                {lab.readme && (
                  <div className="p-4 sm:p-6 rounded-lg border border-border">
                    <h3 className="text-base sm:text-lg font-medium text-foreground-bright mb-3 sm:mb-4">README</h3>
                    <div className="prose prose-sm prose-invert max-w-none">
                      <pre className="text-sm text-foreground-muted whitespace-pre-wrap font-sans">
                        {lab.readme}
                      </pre>
                    </div>
                  </div>
                )}
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                {/* Quick Stats */}
                {!allStatsZero && (
                  <div className="p-4 rounded-lg border border-border">
                    <h3 className="text-sm font-medium text-foreground-bright mb-3">Stats</h3>
                    <div className="space-y-3">
                      {lab.stats.tasks > 0 && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-foreground-muted">Tasks</span>
                          <span className="text-foreground">{lab.stats.tasks}</span>
                        </div>
                      )}
                      {lab.stats.papers > 0 && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-foreground-muted">Papers</span>
                          <span className="text-foreground">{lab.stats.papers}</span>
                        </div>
                      )}
                      {lab.stats.experiments > 0 && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-foreground-muted">Experiments</span>
                          <span className="text-foreground">{lab.stats.experiments}</span>
                        </div>
                      )}
                      {lab.stats.viewers > 0 && (
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-foreground-muted">Viewers</span>
                          <span className="text-green-400">{lab.stats.viewers} watching</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Activity Summary */}
                <div className="p-4 rounded-lg border border-border">
                  <h3 className="text-sm font-medium text-foreground-bright mb-3">Activity</h3>
                  <div className="flex items-center gap-2 text-sm text-foreground-muted">
                    <Clock className="w-4 h-4" />
                    <span>
                      Last updated{" "}
                      {new Date(lab.lastActivityAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                {/* Forked from */}
                {lab.forkedFrom && (
                  <div className="p-4 rounded-lg border border-border">
                    <h3 className="text-sm font-medium text-foreground-bright mb-3">Forked from</h3>
                    <a
                      href={`/labs/${lab.forkedFrom.sourceOwner}/${lab.forkedFrom.sourceSlug}`}
                      className="flex items-center gap-2 text-sm text-foreground-muted hover:text-foreground transition-colors"
                    >
                      <GitFork className="w-4 h-4" />
                      {lab.forkedFrom.sourceOwner}/{lab.forkedFrom.sourceSlug}
                    </a>
                  </div>
                )}

                {/* Domain-specific info for Firefly */}
                {lab.domainSlug === "firefly-network" && (
                  <div className="p-4 rounded-lg border border-amber-500/20 bg-amber-500/5">
                    <h3 className="text-sm font-medium text-amber-400 mb-3">Firefly Network</h3>
                    <p className="text-xs text-foreground-muted mb-3">
                      This lab is part of the Firefly Network project, bringing solar-powered mesh lights to 1 billion people.
                    </p>
                    <a
                      href="/projects/firefly-network"
                      className="text-xs text-amber-400 hover:text-amber-300 transition-colors"
                    >
                      Learn more about the project
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tasks Tab */}
        {activeTab === "tasks" && (
          <div className="max-w-3xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-foreground-bright">
                Tasks ({tasks.length})
              </h2>
              {isOwner && (
                <button className="px-3 py-1.5 text-sm rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors">
                  New Task
                </button>
              )}
            </div>

            {tasksLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
              </div>
            ) : tasks.length > 0 ? (
              <div className="space-y-3">
                {tasks.map((task) => (
                  <div
                    key={task.id}
                    className={cn(
                      "p-4 rounded-lg border transition-colors",
                      task.status === "in_progress"
                        ? "border-blue-500/40 bg-blue-500/5"
                        : task.isFailed
                          ? "border-red-500/30 bg-red-500/5"
                          : task.status === "completed"
                            ? "border-green-500/20"
                            : "border-border hover:border-foreground-muted/50"
                    )}
                  >
                    <div className="flex items-start gap-3">
                      {/* Status Icon */}
                      <div className="mt-0.5">
                        {task.isFailed ? (
                          <AlertCircle className="w-5 h-5 text-red-400" />
                        ) : task.status === "completed" ? (
                          <CheckCircle2 className="w-5 h-5 text-green-400" />
                        ) : task.status === "in_progress" ? (
                          <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                        ) : (
                          <Circle className="w-5 h-5 text-foreground-subtle" />
                        )}
                      </div>

                      {/* Task Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-sm font-medium text-foreground-bright">
                            {task.subject}
                          </h3>
                          {task.parentTaskId && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-foreground-subtle/20 text-foreground-muted">
                              Subtask
                            </span>
                          )}
                        </div>
                        {task.description && (
                          <p className="text-xs text-foreground-muted mb-2 line-clamp-2">
                            {task.description}
                          </p>
                        )}
                        {task.status === "in_progress" && (
                          <p className="text-xs text-blue-400 italic">
                            AI agent working...
                          </p>
                        )}
                        {task.resultSummary && task.status === "completed" && !task.isFailed && (
                          <div className="mt-2 p-2 rounded bg-green-500/10 border border-green-500/20">
                            <p className="text-xs text-green-300 line-clamp-3">
                              {task.resultSummary}
                            </p>
                          </div>
                        )}
                        <div className="flex items-center gap-3 mt-2 text-xs text-foreground-subtle">
                          <span className={cn(
                            "capitalize",
                            task.isFailed && "text-red-400"
                          )}>
                            {task.isFailed ? "failed" : task.status.replace("_", " ")}
                          </span>
                          {task.owner && <span>{task.owner}</span>}
                          {(task.createdAt || task.completedAt) && (
                            <span>{relativeTime(task.completedAt || task.createdAt)}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 border border-dashed border-border rounded-lg">
                <ListTodo className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
                <p className="text-sm text-foreground-muted">
                  No research tasks yet
                </p>
                <p className="text-xs text-foreground-subtle mt-1">
                  Fork this lab to activate AI research agents
                </p>
              </div>
            )}
          </div>
        )}

        {/* Activity Tab */}
        {activeTab === "activity" && (
          <div className="max-w-3xl">
            <h2 className="text-lg font-medium text-foreground-bright mb-6">
              Recent Activity
            </h2>

            <ActivityFeed
              labId={lab.id}
              liveUpdates={true}
              updateInterval={30000}
              groupByDate={true}
              emptyMessage="No activity yet. Activity will appear here as work progresses on this lab."
            />
          </div>
        )}

        {/* Settings Tab (owner only) */}
        {activeTab === "settings" && isOwner && (
          <div className="max-w-2xl">
            <h2 className="text-lg font-medium text-foreground-bright mb-6">
              Lab Settings
            </h2>

            <div className="space-y-6">
              {/* General Settings */}
              <div className="p-6 rounded-lg border border-border">
                <h3 className="text-sm font-medium text-foreground-bright mb-4">General</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs text-foreground-muted mb-1">Lab Name</label>
                    <input
                      type="text"
                      defaultValue={lab.name}
                      className="w-full px-3 py-2 rounded-lg text-sm bg-background border border-border text-foreground"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-foreground-muted mb-1">Description</label>
                    <textarea
                      rows={3}
                      defaultValue={lab.description || ""}
                      className="w-full px-3 py-2 rounded-lg text-sm bg-background border border-border text-foreground"
                    />
                  </div>
                </div>
              </div>

              {/* Visibility */}
              <div className="p-6 rounded-lg border border-border">
                <h3 className="text-sm font-medium text-foreground-bright mb-4">Visibility</h3>
                <div className="space-y-2">
                  {["public", "unlisted", "private"].map((visibility) => (
                    <label
                      key={visibility}
                      className={cn(
                        "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                        lab.visibility === visibility
                          ? "border-foreground-bright bg-foreground-bright/5"
                          : "border-border hover:bg-foreground-muted/5"
                      )}
                    >
                      <input
                        type="radio"
                        name="visibility"
                        value={visibility}
                        defaultChecked={lab.visibility === visibility}
                        className="sr-only"
                      />
                      <div>
                        <span className="text-sm font-medium text-foreground capitalize">
                          {visibility}
                        </span>
                        <p className="text-xs text-foreground-muted mt-0.5">
                          {visibility === "public" && "Anyone can see this lab"}
                          {visibility === "unlisted" && "Only people with the link can see"}
                          {visibility === "private" && "Only you can see this lab"}
                        </p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Danger Zone */}
              <div className="p-6 rounded-lg border border-red-500/20 bg-red-500/5">
                <h3 className="text-sm font-medium text-red-400 mb-4">Danger Zone</h3>
                <button className="px-4 py-2 rounded-lg border border-red-500/30 text-red-400 text-sm hover:bg-red-500/10 transition-colors">
                  Delete Lab
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Share Dialog */}
      <ShareDialog
        lab={lab}
        isOpen={showShareDialog}
        onClose={() => setShowShareDialog(false)}
      />
    </div>
  );
}
