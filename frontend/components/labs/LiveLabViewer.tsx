"use client";

/**
 * LiveLabViewer Component
 *
 * Embeddable 3D lab viewer for public lab pages:
 * - Displays Lab3D in read-only mode
 * - Shows real-time agent activity
 * - Auto-refreshes agent status
 * - Minimal UI for embedded contexts
 */

import { useState, useEffect, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";
import {
  Activity,
  Eye,
  Maximize2,
  Minimize2,
  RefreshCw,
  Users,
  Loader2,
} from "lucide-react";
import type { Lab } from "@/lib/labs/types";

// Dynamically import Lab3D to avoid SSR issues with Three.js
const Lab3D = dynamic(() => import("@/components/Lab3D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-background-elevated">
      <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
    </div>
  ),
});

export interface LiveLabViewerProps {
  /** Lab data */
  lab: Lab;
  /** Is read-only mode (for public viewers) */
  readOnly?: boolean;
  /** Auto-refresh interval in ms (0 to disable) */
  refreshInterval?: number;
  /** Show viewer count */
  showViewers?: boolean;
  /** Show activity feed */
  showActivity?: boolean;
  /** Allow fullscreen */
  allowFullscreen?: boolean;
  /** Compact mode (minimal UI) */
  compact?: boolean;
  /** Custom class name */
  className?: string;
}

interface AgentStatus {
  id: string;
  name: string;
  status: "idle" | "working" | "thinking";
  task?: string;
  lastActivity?: string;
}

interface ActivityEntry {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  agentId?: string;
}

export function LiveLabViewer({
  lab,
  readOnly = true,
  refreshInterval = 5000,
  showViewers = true,
  showActivity = true,
  allowFullscreen = true,
  compact = false,
  className,
}: LiveLabViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [viewerCount, setViewerCount] = useState(lab.stats.viewers);
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [activities, setActivities] = useState<ActivityEntry[]>([]);
  const [showActivityPanel, setShowActivityPanel] = useState(false);

  // Fetch agent status
  const fetchAgentStatus = useCallback(async () => {
    try {
      setIsRefreshing(true);
      const response = await fetch(`/api/lab/agent-status?labId=${lab.id}`);
      const data = await response.json();

      if (data.success && data.agents) {
        setAgents(data.agents);
      }
    } catch (error) {
      console.error("Failed to fetch agent status:", error);
    } finally {
      setIsRefreshing(false);
    }
  }, [lab.id]);

  // Fetch recent activity
  const fetchActivity = useCallback(async () => {
    try {
      const response = await fetch(`/api/lab/activities?labId=${lab.id}&limit=5`);
      const data = await response.json();

      if (data.success && data.activities) {
        setActivities(data.activities);
      }
    } catch (error) {
      console.error("Failed to fetch activities:", error);
    }
  }, [lab.id]);

  // Fetch viewer count
  const fetchViewerCount = useCallback(async () => {
    try {
      const response = await fetch(`/api/labs/${lab.id}/stats`);
      const data = await response.json();

      if (data.success && data.stats) {
        setViewerCount(data.stats.viewers);
      }
    } catch (error) {
      console.error("Failed to fetch viewer count:", error);
    }
  }, [lab.id]);

  // Initial fetch
  useEffect(() => {
    fetchAgentStatus();
    fetchActivity();
    fetchViewerCount();
  }, [fetchAgentStatus, fetchActivity, fetchViewerCount]);

  // Auto-refresh
  useEffect(() => {
    if (refreshInterval <= 0) return;

    const interval = setInterval(() => {
      fetchAgentStatus();
      fetchViewerCount();
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [refreshInterval, fetchAgentStatus, fetchViewerCount]);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current || !allowFullscreen) return;

    if (!isFullscreen) {
      containerRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, [isFullscreen, allowFullscreen]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  // Convert agents to Lab3D format
  const lab3DAgents = agents.map((agent) => ({
    id: agent.id,
    name: agent.name,
    color: 0x4ecdc4, // Default color
    position: [0, 0, 0] as [number, number, number],
    task: agent.task,
    status: agent.status,
  }));

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative rounded-lg overflow-hidden border border-border bg-background-elevated",
        isFullscreen && "fixed inset-0 z-50 rounded-none",
        className
      )}
    >
      {/* 3D Viewer */}
      <div className="w-full h-full min-h-[300px]">
        <Lab3D
          agents={lab3DAgents.length > 0 ? lab3DAgents : undefined}
          onAgentClick={readOnly ? undefined : (agent) => console.log("Agent clicked:", agent)}
        />
      </div>

      {/* Overlay Controls */}
      <div className="absolute top-0 left-0 right-0 p-3 flex items-start justify-between pointer-events-none">
        {/* Left: Lab info */}
        <div className="pointer-events-auto">
          {!compact && (
            <div className="px-3 py-1.5 rounded-lg bg-background/80 backdrop-blur-sm border border-border">
              <h3 className="text-sm font-medium text-foreground truncate max-w-[200px]">
                {lab.name}
              </h3>
              <p className="text-xs text-foreground-muted truncate max-w-[200px]">
                {lab.owner.displayName}
              </p>
            </div>
          )}
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-2 pointer-events-auto">
          {/* Viewers */}
          {showViewers && viewerCount > 0 && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-background/80 backdrop-blur-sm border border-border">
              <Eye className="w-3.5 h-3.5 text-green-400" />
              <span className="text-xs text-foreground">{viewerCount}</span>
            </div>
          )}

          {/* Active agents */}
          {agents.filter((a) => a.status === "working").length > 0 && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-background/80 backdrop-blur-sm border border-border">
              <Users className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-xs text-foreground">
                {agents.filter((a) => a.status === "working").length} active
              </span>
            </div>
          )}

          {/* Refresh */}
          <button
            onClick={() => {
              fetchAgentStatus();
              fetchActivity();
            }}
            disabled={isRefreshing}
            className={cn(
              "p-1.5 rounded-lg bg-background/80 backdrop-blur-sm border border-border transition-colors",
              isRefreshing ? "opacity-50" : "hover:bg-background"
            )}
            title="Refresh"
          >
            <RefreshCw className={cn("w-3.5 h-3.5 text-foreground-muted", isRefreshing && "animate-spin")} />
          </button>

          {/* Activity toggle */}
          {showActivity && (
            <button
              onClick={() => setShowActivityPanel(!showActivityPanel)}
              className={cn(
                "p-1.5 rounded-lg bg-background/80 backdrop-blur-sm border transition-colors",
                showActivityPanel
                  ? "border-foreground-bright bg-foreground-bright/10"
                  : "border-border hover:bg-background"
              )}
              title="Activity feed"
            >
              <Activity className={cn("w-3.5 h-3.5", showActivityPanel ? "text-foreground-bright" : "text-foreground-muted")} />
            </button>
          )}

          {/* Fullscreen */}
          {allowFullscreen && (
            <button
              onClick={toggleFullscreen}
              className="p-1.5 rounded-lg bg-background/80 backdrop-blur-sm border border-border hover:bg-background transition-colors"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? (
                <Minimize2 className="w-3.5 h-3.5 text-foreground-muted" />
              ) : (
                <Maximize2 className="w-3.5 h-3.5 text-foreground-muted" />
              )}
            </button>
          )}
        </div>
      </div>

      {/* Activity Panel */}
      {showActivity && showActivityPanel && (
        <div className="absolute bottom-0 left-0 right-0 max-h-[40%] overflow-y-auto p-3">
          <div className="rounded-lg bg-background/90 backdrop-blur-sm border border-border p-3">
            <h4 className="text-xs font-medium text-foreground-bright mb-2">Recent Activity</h4>
            {activities.length === 0 ? (
              <p className="text-xs text-foreground-muted">No recent activity</p>
            ) : (
              <div className="space-y-2">
                {activities.map((activity) => (
                  <div key={activity.id} className="flex items-start gap-2">
                    <Activity className="w-3 h-3 text-foreground-subtle mt-0.5 flex-shrink-0" />
                    <div className="min-w-0">
                      <p className="text-xs text-foreground truncate">{activity.description}</p>
                      <p className="text-[10px] text-foreground-subtle">
                        {new Date(activity.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Read-only indicator */}
      {readOnly && !compact && (
        <div className="absolute bottom-3 left-3">
          <div className="px-2 py-1 rounded bg-background/80 backdrop-blur-sm border border-border">
            <span className="text-[10px] text-foreground-subtle">View only</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default LiveLabViewer;
