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

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
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
} from "lucide-react";
import { LabHeader } from "@/components/labs/LabHeader";
import { ShareDialog } from "@/components/labs/ShareDialog";
import { FireflyLabContent } from "@/components/labs/FireflyLabContent";
import { LiveLabViewer } from "@/components/labs/LiveLabViewer";
import type { Lab } from "@/lib/labs/types";
import { getCurrentUser } from "@/lib/auth/mock-user";

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

  const [lab, setLab] = useState<Lab | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStarred, setIsStarred] = useState(false);
  const [isOwner, setIsOwner] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  // Fetch lab data
  const fetchLab = useCallback(async () => {
    try {
      setIsLoading(true);
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

        // Check if current user is owner
        const currentUser = getCurrentUser();
        setIsOwner(currentUser?.id === detailData.lab.owner.id);
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

  useEffect(() => {
    fetchLab();
  }, [fetchLab]);

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

  // Tab content
  const tabs: { id: TabId; label: string; icon: typeof Activity }[] = [
    { id: "overview", label: "Overview", icon: FileText },
    { id: "tasks", label: "Tasks", icon: ListTodo },
    { id: "activity", label: "Activity", icon: Activity },
    ...(isOwner ? [{ id: "settings" as const, label: "Settings", icon: Settings }] : []),
  ];

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
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

            {/* Standard lab content */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Main content */}
              <div className="lg:col-span-2 space-y-6">
                {/* Lab 3D Viewer */}
                <div className="rounded-lg border border-border bg-background-elevated overflow-hidden">
                  <LiveLabViewer
                    lab={lab}
                    readOnly={!isOwner}
                    showViewers={true}
                    showActivity={true}
                    allowFullscreen={true}
                    className="h-[400px]"
                  />
                </div>

                {/* README/Description */}
                {lab.description && (
                  <div className="p-6 rounded-lg border border-border">
                    <h3 className="text-lg font-medium text-foreground-bright mb-4">About</h3>
                    <p className="text-sm text-foreground-muted whitespace-pre-wrap">
                      {lab.description}
                    </p>
                  </div>
                )}

                {/* README (if available) */}
                {lab.readme && (
                  <div className="p-6 rounded-lg border border-border">
                    <h3 className="text-lg font-medium text-foreground-bright mb-4">README</h3>
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
                <div className="p-4 rounded-lg border border-border">
                  <h3 className="text-sm font-medium text-foreground-bright mb-3">Stats</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground-muted">Tasks</span>
                      <span className="text-foreground">{lab.stats.tasks}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground-muted">Papers</span>
                      <span className="text-foreground">{lab.stats.papers}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground-muted">Experiments</span>
                      <span className="text-foreground">{lab.stats.experiments}</span>
                    </div>
                    {lab.stats.viewers > 0 && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-foreground-muted">Viewers</span>
                        <span className="text-green-400">{lab.stats.viewers} watching</span>
                      </div>
                    )}
                  </div>
                </div>

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
                Tasks ({lab.stats.tasks})
              </h2>
              {isOwner && (
                <button className="px-3 py-1.5 text-sm rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors">
                  New Task
                </button>
              )}
            </div>

            <div className="text-center py-12 border border-border rounded-lg">
              <ListTodo className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
              <p className="text-sm text-foreground-muted">
                No tasks yet
              </p>
              <p className="text-xs text-foreground-subtle mt-1">
                Tasks will appear here when created
              </p>
            </div>
          </div>
        )}

        {/* Activity Tab */}
        {activeTab === "activity" && (
          <div className="max-w-3xl">
            <h2 className="text-lg font-medium text-foreground-bright mb-6">
              Recent Activity
            </h2>

            <div className="text-center py-12 border border-border rounded-lg">
              <Activity className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
              <p className="text-sm text-foreground-muted">
                No activity yet
              </p>
              <p className="text-xs text-foreground-subtle mt-1">
                Activity will appear here as work progresses
              </p>
            </div>
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
