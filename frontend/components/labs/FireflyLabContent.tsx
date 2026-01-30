"use client";

/**
 * FireflyLabContent Component
 *
 * Rich content component for the Firefly Lab portal showing:
 * - Live AI agent status with progress
 * - Task completion tracker
 * - Research papers with implementation status
 * - Published results with comments
 * - BOM tracker
 */

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Activity,
  BookOpen,
  Bot,
  CheckCircle2,
  Circle,
  Clock,
  DollarSign,
  ExternalLink,
  FileText,
  Loader2,
  MessageSquare,
  PlayCircle,
  Sparkles,
  Star,
  ThumbsUp,
  Zap,
} from "lucide-react";

interface LabPaper {
  id: string;
  title: string;
  authors: string[];
  category: string;
  status: string;
  progress: number;
  assignedAgent?: string;
}

interface LabTask {
  id: string;
  subject: string;
  type: string;
  status: string;
  priority: string;
  progress: number;
  assignedAgent?: string;
}

interface LabAgent {
  id: string;
  name: string;
  displayName: string;
  type: string;
  status: string;
  currentTask?: string;
  progress: number;
  tokensGenerated: number;
  costEstimate: number;
  color: number;
}

interface LabResult {
  id: string;
  title: string;
  description: string;
  type: string;
  likes: number;
  comments: { id: string; content: string; username: string }[];
}

interface BOMItem {
  id: string;
  name: string;
  totalCost: number;
  category: string;
  status: string;
}

interface FireflyLabData {
  papers: LabPaper[];
  paperStats: { total: number; implemented: number; implementing: number };
  tasks: LabTask[];
  taskStats: { total: number; completed: number; inProgress: number };
  agents: LabAgent[];
  agentStats: { total: number; working: number; totalCost: number };
  results: LabResult[];
  resultStats: { total: number; totalLikes: number; totalComments: number };
  bom: BOMItem[];
  bomSummary: { totalCost: number; targetCost: number };
}

export function FireflyLabContent() {
  const [data, setData] = useState<FireflyLabData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<
    "agents" | "tasks" | "papers" | "results" | "bom"
  >("agents");

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch("/api/labs/firefly");
      const result = await response.json();
      if (result.success) {
        setData(result);
        setError(null);
      } else {
        setError(result.error || "Failed to load data");
      }
    } catch (err) {
      setError("Failed to load lab data");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <p className="text-foreground-muted">{error || "No data available"}</p>
        <button
          onClick={fetchData}
          className="mt-4 px-4 py-2 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="p-4 rounded-lg border border-border bg-background-elevated">
          <div className="flex items-center gap-2 mb-2">
            <Bot className="w-4 h-4 text-green-400" />
            <span className="text-sm text-foreground-muted">Agents</span>
          </div>
          <div className="text-2xl font-bold text-foreground-bright">
            {data.agentStats.working}/{data.agentStats.total}
          </div>
          <div className="text-xs text-foreground-subtle">working</div>
        </div>

        <div className="p-4 rounded-lg border border-border bg-background-elevated">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-foreground-muted">Tasks</span>
          </div>
          <div className="text-2xl font-bold text-foreground-bright">
            {data.taskStats.completed}/{data.taskStats.total}
          </div>
          <div className="text-xs text-foreground-subtle">completed</div>
        </div>

        <div className="p-4 rounded-lg border border-border bg-background-elevated">
          <div className="flex items-center gap-2 mb-2">
            <BookOpen className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-foreground-muted">Papers</span>
          </div>
          <div className="text-2xl font-bold text-foreground-bright">
            {data.paperStats.implemented}/{data.paperStats.total}
          </div>
          <div className="text-xs text-foreground-subtle">implemented</div>
        </div>

        <div className="p-4 rounded-lg border border-border bg-background-elevated">
          <div className="flex items-center gap-2 mb-2">
            <ThumbsUp className="w-4 h-4 text-amber-400" />
            <span className="text-sm text-foreground-muted">Engagement</span>
          </div>
          <div className="text-2xl font-bold text-foreground-bright">
            {data.resultStats.totalLikes}
          </div>
          <div className="text-xs text-foreground-subtle">likes</div>
        </div>

        <div className="p-4 rounded-lg border border-border bg-background-elevated">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-sm text-foreground-muted">BOM</span>
          </div>
          <div className="text-2xl font-bold text-foreground-bright">
            ${data.bomSummary.totalCost.toFixed(0)}
          </div>
          <div className="text-xs text-foreground-subtle">
            target: ${data.bomSummary.targetCost}
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 border-b border-border">
        {[
          { id: "agents", label: "Agents", icon: Bot },
          { id: "tasks", label: "Tasks", icon: Activity },
          { id: "papers", label: "Papers", icon: BookOpen },
          { id: "results", label: "Results", icon: FileText },
          { id: "bom", label: "BOM", icon: DollarSign },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={cn(
              "flex items-center gap-2 px-4 py-2 text-sm border-b-2 -mb-px transition-colors",
              activeTab === tab.id
                ? "border-foreground-bright text-foreground-bright"
                : "border-transparent text-foreground-muted hover:text-foreground"
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {/* Agents Tab */}
        {activeTab === "agents" && (
          <div className="grid md:grid-cols-3 gap-4">
            {data.agents.map((agent) => (
              <div
                key={agent.id}
                className={cn(
                  "p-4 rounded-lg border transition-colors",
                  agent.status === "working"
                    ? "border-green-500/30 bg-green-500/5"
                    : "border-border"
                )}
              >
                <div className="flex items-center gap-3 mb-3">
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold"
                    style={{ backgroundColor: `#${agent.color.toString(16)}` }}
                  >
                    {agent.name[0]}
                  </div>
                  <div>
                    <div className="font-medium text-foreground-bright">
                      {agent.name}
                    </div>
                    <div className="text-xs text-foreground-subtle">
                      {agent.displayName}
                    </div>
                  </div>
                  {agent.status === "working" && (
                    <span className="ml-auto flex items-center gap-1 text-xs text-green-400">
                      <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                      Active
                    </span>
                  )}
                </div>

                {agent.currentTask && (
                  <div className="text-sm text-foreground-muted mb-3">
                    {agent.currentTask}
                  </div>
                )}

                <div className="flex items-center gap-2 mb-2">
                  <div className="flex-1 h-2 bg-foreground-subtle/20 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-foreground-bright transition-all"
                      style={{ width: `${agent.progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-foreground-muted">
                    {agent.progress}%
                  </span>
                </div>

                <div className="flex items-center justify-between text-xs text-foreground-subtle">
                  <span>{agent.type === "ollama" ? "FREE (Ollama)" : agent.type}</span>
                  {agent.costEstimate > 0 && (
                    <span>${agent.costEstimate.toFixed(2)}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Tasks Tab */}
        {activeTab === "tasks" && (
          <div className="space-y-3">
            {data.tasks.map((task) => (
              <div
                key={task.id}
                className={cn(
                  "p-4 rounded-lg border transition-colors",
                  task.status === "in_progress"
                    ? "border-blue-500/30 bg-blue-500/5"
                    : task.status === "completed"
                    ? "border-green-500/30 bg-green-500/5"
                    : "border-border"
                )}
              >
                <div className="flex items-start gap-3">
                  {task.status === "completed" ? (
                    <CheckCircle2 className="w-5 h-5 text-green-400 mt-0.5" />
                  ) : task.status === "in_progress" ? (
                    <PlayCircle className="w-5 h-5 text-blue-400 mt-0.5" />
                  ) : (
                    <Circle className="w-5 h-5 text-foreground-subtle mt-0.5" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground-bright">
                      {task.subject}
                    </div>
                    <div className="flex flex-wrap items-center gap-2 mt-1">
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded text-xs",
                          task.priority === "critical"
                            ? "bg-red-500/20 text-red-400"
                            : task.priority === "high"
                            ? "bg-orange-500/20 text-orange-400"
                            : "bg-foreground-subtle/20 text-foreground-muted"
                        )}
                      >
                        {task.priority}
                      </span>
                      <span className="px-2 py-0.5 rounded text-xs bg-foreground-subtle/20 text-foreground-muted">
                        {task.type}
                      </span>
                      {task.assignedAgent && (
                        <span className="text-xs text-foreground-subtle">
                          Agent: {task.assignedAgent}
                        </span>
                      )}
                    </div>
                    {task.progress > 0 && task.status !== "completed" && (
                      <div className="flex items-center gap-2 mt-2">
                        <div className="flex-1 h-1.5 bg-foreground-subtle/20 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-400 transition-all"
                            style={{ width: `${task.progress}%` }}
                          />
                        </div>
                        <span className="text-xs text-foreground-muted">
                          {task.progress}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Papers Tab */}
        {activeTab === "papers" && (
          <div className="space-y-3">
            {data.papers.map((paper) => (
              <div
                key={paper.id}
                className={cn(
                  "p-4 rounded-lg border transition-colors",
                  paper.status === "implemented"
                    ? "border-green-500/30 bg-green-500/5"
                    : paper.status === "implementing"
                    ? "border-purple-500/30 bg-purple-500/5"
                    : "border-border"
                )}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-foreground-bright">
                      {paper.title}
                    </div>
                    <div className="text-sm text-foreground-muted mt-1">
                      {paper.authors.join(", ")}
                    </div>
                    <div className="flex flex-wrap items-center gap-2 mt-2">
                      <span className="px-2 py-0.5 rounded text-xs bg-foreground-subtle/20 text-foreground-muted">
                        {paper.category}
                      </span>
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded text-xs",
                          paper.status === "implemented"
                            ? "bg-green-500/20 text-green-400"
                            : paper.status === "implementing"
                            ? "bg-purple-500/20 text-purple-400"
                            : paper.status === "reading"
                            ? "bg-blue-500/20 text-blue-400"
                            : "bg-foreground-subtle/20 text-foreground-muted"
                        )}
                      >
                        {paper.status}
                      </span>
                      {paper.assignedAgent && (
                        <span className="text-xs text-foreground-subtle">
                          Agent: {paper.assignedAgent}
                        </span>
                      )}
                    </div>
                  </div>
                  {paper.progress > 0 && (
                    <div className="text-right">
                      <div className="text-2xl font-bold text-foreground-bright">
                        {paper.progress}%
                      </div>
                      <div className="text-xs text-foreground-subtle">progress</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Results Tab */}
        {activeTab === "results" && (
          <div className="space-y-4">
            {data.results.map((result) => (
              <div key={result.id} className="p-4 rounded-lg border border-border">
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div>
                    <div className="font-medium text-foreground-bright">
                      {result.title}
                    </div>
                    <div className="text-sm text-foreground-muted mt-1">
                      {result.description}
                    </div>
                  </div>
                  <span
                    className={cn(
                      "px-2 py-0.5 rounded text-xs",
                      result.type === "paper"
                        ? "bg-purple-500/20 text-purple-400"
                        : result.type === "code"
                        ? "bg-green-500/20 text-green-400"
                        : "bg-blue-500/20 text-blue-400"
                    )}
                  >
                    {result.type}
                  </span>
                </div>

                <div className="flex items-center gap-4 text-sm text-foreground-muted">
                  <span className="flex items-center gap-1">
                    <ThumbsUp className="w-4 h-4" />
                    {result.likes}
                  </span>
                  <span className="flex items-center gap-1">
                    <MessageSquare className="w-4 h-4" />
                    {result.comments.length}
                  </span>
                </div>

                {result.comments.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-border space-y-3">
                    {result.comments.slice(0, 2).map((comment) => (
                      <div key={comment.id} className="text-sm">
                        <span className="font-medium text-foreground">
                          @{comment.username}:
                        </span>{" "}
                        <span className="text-foreground-muted">
                          {comment.content.slice(0, 100)}
                          {comment.content.length > 100 && "..."}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* BOM Tab */}
        {activeTab === "bom" && (
          <div>
            <div className="mb-6 p-4 rounded-lg border border-amber-500/30 bg-amber-500/5">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm text-foreground-muted">Current BOM Cost</div>
                  <div className="text-3xl font-bold text-amber-400">
                    ${data.bomSummary.totalCost.toFixed(2)}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-foreground-muted">Target</div>
                  <div className="text-xl font-bold text-foreground-bright">
                    ${data.bomSummary.targetCost.toFixed(2)}
                  </div>
                </div>
              </div>
              <div className="mt-3">
                <div className="flex items-center justify-between text-xs text-foreground-muted mb-1">
                  <span>Progress to target</span>
                  <span>
                    ${(data.bomSummary.totalCost - data.bomSummary.targetCost).toFixed(2)} over
                  </span>
                </div>
                <div className="h-2 bg-foreground-subtle/20 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-500"
                    style={{
                      width: `${Math.min(100, (data.bomSummary.targetCost / data.bomSummary.totalCost) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {data.bom.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center justify-between p-3 rounded-lg border border-border"
                >
                  <div>
                    <div className="font-medium text-foreground">{item.name}</div>
                    <div className="text-xs text-foreground-subtle">
                      {item.category}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className={cn(
                        "px-2 py-0.5 rounded text-xs",
                        item.status === "selected"
                          ? "bg-green-500/20 text-green-400"
                          : item.status === "optimizing"
                          ? "bg-amber-500/20 text-amber-400"
                          : "bg-blue-500/20 text-blue-400"
                      )}
                    >
                      {item.status}
                    </span>
                    <span className="font-mono text-foreground-bright">
                      ${item.totalCost.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default FireflyLabContent;
