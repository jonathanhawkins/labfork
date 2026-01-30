"use client";

/**
 * Labs Overview Page
 *
 * Displays all research labs from the research manager with live status.
 * Auto-refreshes every 30 seconds to show real-time progress.
 */

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Activity,
  Bot,
  CheckCircle2,
  Clock,
  FileText,
  Loader2,
  PlayCircle,
  Settings,
  Zap,
} from "lucide-react";

interface LabAgent {
  id: string;
  name?: string;
  displayName?: string;
  type?: string;
  status: string;
  currentTask?: string;
  progress?: number;
}

interface LabStatus {
  id: string;
  name: string;
  description: string;
  active: boolean;
  domain?: string;
  settings?: {
    maxAgents?: number;
    autoSpawn?: boolean;
    researchInterval?: number;
  };
  agents: {
    running: number;
    total: number;
    list: LabAgent[];
  };
  tasks: {
    total: number;
    pending: number;
    in_progress: number;
    completed: number;
  };
  proposals: {
    total: number;
    pending: number;
  };
  createdAt: string;
  updatedAt: string;
}

export default function LabsOverviewPage() {
  const [labs, setLabs] = useState<LabStatus[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchLabs = useCallback(async () => {
    try {
      const response = await fetch("/api/labs/research");
      const data = await response.json();

      if (data.success) {
        setLabs(data.labs);
        setError(null);
        setLastUpdated(new Date());
      } else {
        setError(data.error || "Failed to load labs");
      }
    } catch (err) {
      setError("Failed to connect to server");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLabs();

    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchLabs, 30000);
    return () => clearInterval(interval);
  }, [fetchLabs]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-foreground-muted mx-auto mb-4" />
          <p className="text-foreground-muted">Loading research labs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground-bright">
                Research Labs
              </h1>
              <p className="text-foreground-muted mt-2">
                Live status of AI research labs and their agents
              </p>
            </div>

            {lastUpdated && (
              <div className="flex items-center gap-2 text-sm text-foreground-subtle">
                <Clock className="w-4 h-4" />
                <span>
                  Updated {lastUpdated.toLocaleTimeString()}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {error ? (
          <div className="text-center py-12">
            <p className="text-red-400 mb-4">{error}</p>
            <button
              onClick={fetchLabs}
              className="px-4 py-2 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors"
            >
              Retry
            </button>
          </div>
        ) : labs.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 mx-auto text-foreground-subtle mb-4" />
            <h2 className="text-xl font-medium text-foreground-bright mb-2">
              No Research Labs Found
            </h2>
            <p className="text-foreground-muted">
              Research labs will appear here when created via the research manager.
            </p>
          </div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2">
            {labs.map((lab) => (
              <LabCard key={lab.id} lab={lab} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function LabCard({ lab }: { lab: LabStatus }) {
  const hasActiveAgents = lab.agents.running > 0;
  const hasPendingProposals = lab.proposals.pending > 0;
  const completionRate =
    lab.tasks.total > 0
      ? Math.round((lab.tasks.completed / lab.tasks.total) * 100)
      : 0;

  return (
    <Link
      href={`/labs/research/${lab.id}`}
      className={cn(
        "block p-6 rounded-lg border transition-colors hover:border-foreground-muted",
        hasActiveAgents
          ? "border-green-500/30 bg-green-500/5"
          : "border-border"
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground-bright">
            {lab.name}
          </h2>
          <p className="text-sm text-foreground-muted mt-1 line-clamp-2">
            {lab.description}
          </p>
        </div>

        {lab.active ? (
          hasActiveAgents ? (
            <span className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-green-500/20 text-green-400">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              Active
            </span>
          ) : (
            <span className="px-2 py-1 rounded text-xs bg-foreground-subtle/20 text-foreground-muted">
              Idle
            </span>
          )
        ) : (
          <span className="px-2 py-1 rounded text-xs bg-amber-500/20 text-amber-400">
            Paused
          </span>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <div className="flex items-center gap-1 text-xs text-foreground-muted mb-1">
            <Bot className="w-3 h-3" />
            Agents
          </div>
          <div className="text-lg font-semibold text-foreground-bright">
            {lab.agents.running}
            <span className="text-foreground-subtle text-sm">
              /{lab.agents.total}
            </span>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-1 text-xs text-foreground-muted mb-1">
            <Activity className="w-3 h-3" />
            Tasks
          </div>
          <div className="text-lg font-semibold text-foreground-bright">
            {lab.tasks.in_progress > 0 && (
              <span className="text-blue-400 mr-1">{lab.tasks.in_progress}</span>
            )}
            {lab.tasks.completed}
            <span className="text-foreground-subtle text-sm">
              /{lab.tasks.total}
            </span>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-1 text-xs text-foreground-muted mb-1">
            <FileText className="w-3 h-3" />
            Proposals
          </div>
          <div className="text-lg font-semibold text-foreground-bright">
            {hasPendingProposals ? (
              <span className="text-amber-400">{lab.proposals.pending}</span>
            ) : (
              lab.proposals.total
            )}
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      {lab.tasks.total > 0 && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs text-foreground-muted mb-1">
            <span>Progress</span>
            <span>{completionRate}% complete</span>
          </div>
          <div className="h-2 bg-foreground-subtle/20 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-green-500 transition-all"
              style={{ width: `${completionRate}%` }}
            />
          </div>
        </div>
      )}

      {/* Active Agents */}
      {hasActiveAgents && (
        <div className="pt-4 border-t border-border">
          <div className="text-xs text-foreground-muted mb-2">
            Active Agents
          </div>
          <div className="space-y-2">
            {lab.agents.list
              .filter((a) => a.status === "running" || a.status === "working")
              .slice(0, 2)
              .map((agent) => (
                <div
                  key={agent.id}
                  className="flex items-center gap-2 text-sm"
                >
                  <Zap className="w-3 h-3 text-amber-400" />
                  <span className="text-foreground">
                    {agent.displayName || agent.name || agent.id}
                  </span>
                  {agent.currentTask && (
                    <span className="text-foreground-subtle text-xs truncate flex-1">
                      - {agent.currentTask}
                    </span>
                  )}
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Pending Proposals Alert */}
      {hasPendingProposals && (
        <div className="mt-4 p-3 rounded bg-amber-500/10 border border-amber-500/20">
          <div className="flex items-center gap-2 text-sm text-amber-400">
            <FileText className="w-4 h-4" />
            {lab.proposals.pending} proposal{lab.proposals.pending > 1 ? "s" : ""}{" "}
            pending review
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-4 border-t border-border flex items-center justify-between text-xs text-foreground-subtle">
        <span>
          Domain: {lab.domain || "N/A"}
        </span>
        <span>
          Updated {new Date(lab.updatedAt).toLocaleDateString()}
        </span>
      </div>
    </Link>
  );
}
