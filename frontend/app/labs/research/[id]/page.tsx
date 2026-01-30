"use client";

/**
 * Research Lab Detail Page
 *
 * Displays detailed research state for a specific lab including:
 * - Active agents and their current tasks
 * - Task list with completion status
 * - Proposals with approval buttons
 * - Research documents (markdown)
 * - Progress history
 *
 * Auto-refreshes every 30 seconds.
 */

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Activity,
  ArrowLeft,
  Bot,
  CheckCircle2,
  Circle,
  Clock,
  FileText,
  Loader2,
  PlayCircle,
  RefreshCw,
  Settings,
  ThumbsDown,
  ThumbsUp,
  Zap,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

interface PageProps {
  params: Promise<{ id: string }>;
}

interface Agent {
  id: string;
  name?: string;
  displayName?: string;
  type?: string;
  model?: string;
  status: string;
  currentTask?: string;
  currentTaskId?: string;
  progress?: number;
  tokensGenerated?: number;
  costEstimate?: number;
  startedAt?: string;
  lastActivityAt?: string;
}

interface Proposal {
  id: string;
  title?: string;
  storyId?: string;
  status: string;
  summary?: string;
  recommendation?: string;
  filePath?: string;
  createdAt?: string;
  updatedAt?: string;
}

interface Task {
  id: string;
  subject: string;
  description?: string;
  status: "pending" | "in_progress" | "completed";
  priority?: string;
  owner?: string;
  activeForm?: string;
  blockedBy?: string[];
  blocks?: string[];
  createdAt?: string;
  updatedAt?: string;
  completedAt?: string;
}

interface ProgressEntry {
  timestamp: string;
  type: string;
  taskId?: string;
  subject?: string;
  agent?: string;
  output?: string;
}

interface ResearchDocument {
  name: string;
  path: string;
  content: string;
  size: number;
  modified: string;
}

interface LabData {
  lab: {
    id: string;
    name: string;
    description: string;
    active: boolean;
    domain?: string;
    taskListId: string;
    settings?: {
      maxAgents?: number;
      autoSpawn?: boolean;
      researchInterval?: number;
    };
    createdAt: string;
    updatedAt: string;
  };
  agents: {
    list: Agent[];
    running: number;
    total: number;
  };
  proposals: {
    list: Proposal[];
    pending: number;
    total: number;
  };
  tasks: {
    list: Task[];
    stats: {
      total: number;
      pending: number;
      in_progress: number;
      completed: number;
    };
  };
  progress: {
    recent: ProgressEntry[];
    total: number;
  };
  documents: ResearchDocument[];
  outputs: string[];
}

type TabId = "overview" | "tasks" | "proposals" | "documents" | "activity";

export default function ResearchLabDetailPage({ params }: PageProps) {
  const router = useRouter();
  const [labId, setLabId] = useState<string | null>(null);
  const [data, setData] = useState<LabData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [selectedDocument, setSelectedDocument] = useState<ResearchDocument | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Unwrap params
  useEffect(() => {
    params.then((p) => setLabId(p.id));
  }, [params]);

  const fetchData = useCallback(async () => {
    if (!labId) return;

    try {
      setIsRefreshing(true);
      const response = await fetch(`/api/labs/research/${labId}`);
      const result = await response.json();

      if (result.success) {
        setData(result);
        setError(null);
      } else {
        setError(result.error || "Failed to load lab");
      }
    } catch (err) {
      setError("Failed to connect to server");
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [labId]);

  useEffect(() => {
    if (labId) {
      fetchData();
      // Auto-refresh every 30 seconds
      const interval = setInterval(fetchData, 30000);
      return () => clearInterval(interval);
    }
  }, [labId, fetchData]);

  if (isLoading || !labId) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-foreground-bright mb-2">
            {error || "Lab not found"}
          </h1>
          <p className="text-foreground-muted mb-4">
            Could not load lab details.
          </p>
          <div className="flex gap-2 justify-center">
            <button
              onClick={fetchData}
              className="px-4 py-2 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors"
            >
              Retry
            </button>
            <Link
              href="/labs"
              className="px-4 py-2 rounded-lg border border-border text-foreground hover:bg-foreground-muted/10 transition-colors"
            >
              Back to Labs
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const { lab, agents, proposals, tasks, progress, documents } = data;

  const tabs: { id: TabId; label: string; icon: typeof Activity; badge?: number }[] = [
    { id: "overview", label: "Overview", icon: Activity },
    { id: "tasks", label: "Tasks", icon: CheckCircle2, badge: tasks.stats.in_progress },
    { id: "proposals", label: "Proposals", icon: FileText, badge: proposals.pending },
    { id: "documents", label: "Documents", icon: FileText, badge: documents.length },
    { id: "activity", label: "Activity", icon: Clock },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center gap-4 mb-4">
            <Link
              href="/labs"
              className="p-2 rounded-lg hover:bg-foreground-muted/10 transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-foreground-muted" />
            </Link>
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold text-foreground-bright">
                  {lab.name}
                </h1>
                {lab.active ? (
                  agents.running > 0 ? (
                    <span className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-green-500/20 text-green-400">
                      <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                      {agents.running} agent{agents.running > 1 ? "s" : ""} active
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
              <p className="text-foreground-muted mt-1">{lab.description}</p>
            </div>
            <button
              onClick={fetchData}
              disabled={isRefreshing}
              className={cn(
                "p-2 rounded-lg hover:bg-foreground-muted/10 transition-colors",
                isRefreshing && "animate-spin"
              )}
            >
              <RefreshCw className="w-5 h-5 text-foreground-muted" />
            </button>
          </div>

          {/* Tab Navigation */}
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
                {tab.badge !== undefined && tab.badge > 0 && (
                  <span className="px-1.5 py-0.5 text-xs rounded-full bg-foreground-bright/20">
                    {tab.badge}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Active Agents */}
              <div className="p-6 rounded-lg border border-border">
                <h3 className="text-lg font-medium text-foreground-bright mb-4">
                  Active Agents
                </h3>
                {agents.list.length === 0 ? (
                  <p className="text-foreground-muted">No agents registered</p>
                ) : (
                  <div className="space-y-4">
                    {agents.list.map((agent) => (
                      <AgentCard key={agent.id} agent={agent} />
                    ))}
                  </div>
                )}
              </div>

              {/* Recent Tasks */}
              <div className="p-6 rounded-lg border border-border">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-foreground-bright">
                    Recent Tasks
                  </h3>
                  <button
                    onClick={() => setActiveTab("tasks")}
                    className="text-sm text-foreground-muted hover:text-foreground"
                  >
                    View all
                  </button>
                </div>
                {tasks.list.length === 0 ? (
                  <p className="text-foreground-muted">No tasks</p>
                ) : (
                  <div className="space-y-2">
                    {tasks.list.slice(0, 5).map((task) => (
                      <TaskRow key={task.id} task={task} />
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Stats */}
              <div className="p-4 rounded-lg border border-border">
                <h3 className="text-sm font-medium text-foreground-bright mb-3">
                  Statistics
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground-muted">Tasks</span>
                    <span className="text-foreground">
                      {tasks.stats.completed}/{tasks.stats.total} completed
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground-muted">In Progress</span>
                    <span className="text-blue-400">{tasks.stats.in_progress}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground-muted">Proposals</span>
                    <span className="text-foreground">{proposals.total}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground-muted">Documents</span>
                    <span className="text-foreground">{documents.length}</span>
                  </div>
                </div>
              </div>

              {/* Pending Proposals */}
              {proposals.pending > 0 && (
                <div className="p-4 rounded-lg border border-amber-500/30 bg-amber-500/5">
                  <h3 className="text-sm font-medium text-amber-400 mb-3">
                    Pending Review
                  </h3>
                  <p className="text-sm text-foreground-muted mb-3">
                    {proposals.pending} proposal{proposals.pending > 1 ? "s" : ""} awaiting review
                  </p>
                  <button
                    onClick={() => setActiveTab("proposals")}
                    className="w-full px-3 py-2 text-sm rounded-lg bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors"
                  >
                    Review Proposals
                  </button>
                </div>
              )}

              {/* Lab Settings */}
              <div className="p-4 rounded-lg border border-border">
                <h3 className="text-sm font-medium text-foreground-bright mb-3">
                  Configuration
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-foreground-muted">Domain</span>
                    <span className="text-foreground">{lab.domain || "N/A"}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-foreground-muted">Task List</span>
                    <span className="text-foreground font-mono text-xs">
                      {lab.taskListId}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-foreground-muted">Max Agents</span>
                    <span className="text-foreground">
                      {lab.settings?.maxAgents || 3}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-foreground-muted">Auto Spawn</span>
                    <span className="text-foreground">
                      {lab.settings?.autoSpawn ? "Yes" : "No"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tasks Tab */}
        {activeTab === "tasks" && (
          <div className="max-w-3xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-medium text-foreground-bright">
                Tasks ({tasks.stats.total})
              </h2>
              <div className="flex gap-2 text-sm">
                <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-400">
                  {tasks.stats.in_progress} in progress
                </span>
                <span className="px-2 py-1 rounded bg-green-500/20 text-green-400">
                  {tasks.stats.completed} completed
                </span>
              </div>
            </div>

            {tasks.list.length === 0 ? (
              <div className="text-center py-12 border border-border rounded-lg">
                <CheckCircle2 className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
                <p className="text-foreground-muted">No tasks yet</p>
              </div>
            ) : (
              <div className="space-y-2">
                {tasks.list.map((task) => (
                  <TaskCard key={task.id} task={task} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Proposals Tab */}
        {activeTab === "proposals" && (
          <div className="max-w-3xl">
            <h2 className="text-lg font-medium text-foreground-bright mb-6">
              Research Proposals ({proposals.total})
            </h2>

            {proposals.list.length === 0 ? (
              <div className="text-center py-12 border border-border rounded-lg">
                <FileText className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
                <p className="text-foreground-muted">No proposals yet</p>
              </div>
            ) : (
              <div className="space-y-4">
                {proposals.list.map((proposal) => (
                  <ProposalCard key={proposal.id} proposal={proposal} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Documents Tab */}
        {activeTab === "documents" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Document List */}
            <div className="space-y-2">
              <h2 className="text-lg font-medium text-foreground-bright mb-4">
                Research Documents
              </h2>
              {documents.length === 0 ? (
                <p className="text-foreground-muted">No documents found</p>
              ) : (
                documents.map((doc) => (
                  <button
                    key={doc.path}
                    onClick={() => setSelectedDocument(doc)}
                    className={cn(
                      "w-full text-left p-3 rounded-lg border transition-colors",
                      selectedDocument?.path === doc.path
                        ? "border-foreground-bright bg-foreground-bright/5"
                        : "border-border hover:border-foreground-muted"
                    )}
                  >
                    <div className="font-medium text-foreground text-sm">
                      {doc.name}
                    </div>
                    <div className="text-xs text-foreground-subtle mt-1">
                      {(doc.size / 1024).toFixed(1)} KB - {new Date(doc.modified).toLocaleDateString()}
                    </div>
                  </button>
                ))
              )}
            </div>

            {/* Document Viewer */}
            <div className="lg:col-span-2">
              {selectedDocument ? (
                <div className="p-6 rounded-lg border border-border">
                  <h3 className="text-lg font-medium text-foreground-bright mb-4">
                    {selectedDocument.name}
                  </h3>
                  <div className="prose prose-sm prose-invert max-w-none">
                    <ReactMarkdown>{selectedDocument.content}</ReactMarkdown>
                  </div>
                </div>
              ) : (
                <div className="p-6 rounded-lg border border-border text-center">
                  <FileText className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
                  <p className="text-foreground-muted">
                    Select a document to view
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Activity Tab */}
        {activeTab === "activity" && (
          <div className="max-w-3xl">
            <h2 className="text-lg font-medium text-foreground-bright mb-6">
              Recent Activity ({progress.total} events)
            </h2>

            {progress.recent.length === 0 ? (
              <div className="text-center py-12 border border-border rounded-lg">
                <Activity className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
                <p className="text-foreground-muted">No activity yet</p>
              </div>
            ) : (
              <div className="space-y-3">
                {progress.recent
                  .slice()
                  .reverse()
                  .map((entry, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-lg border border-border"
                    >
                      <div className="flex items-start gap-3">
                        <Activity className="w-4 h-4 text-foreground-muted mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm text-foreground">
                            {entry.subject || entry.type}
                          </div>
                          {entry.agent && (
                            <div className="text-xs text-foreground-subtle">
                              Agent: {entry.agent}
                            </div>
                          )}
                          <div className="text-xs text-foreground-subtle mt-1">
                            {new Date(entry.timestamp).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function AgentCard({ agent }: { agent: Agent }) {
  const isActive = agent.status === "running" || agent.status === "working";

  return (
    <div
      className={cn(
        "p-4 rounded-lg border transition-colors",
        isActive ? "border-green-500/30 bg-green-500/5" : "border-border"
      )}
    >
      <div className="flex items-center gap-3 mb-3">
        <div className="w-10 h-10 rounded-full bg-foreground-subtle/20 flex items-center justify-center">
          <Bot className="w-5 h-5 text-foreground-muted" />
        </div>
        <div className="flex-1">
          <div className="font-medium text-foreground-bright">
            {agent.displayName || agent.name || agent.id}
          </div>
          <div className="text-xs text-foreground-subtle">
            {agent.type} - {agent.model || "unknown model"}
          </div>
        </div>
        {isActive && (
          <span className="flex items-center gap-1 text-xs text-green-400">
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

      {agent.progress !== undefined && agent.progress > 0 && (
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-foreground-subtle/20 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all"
              style={{ width: `${agent.progress}%` }}
            />
          </div>
          <span className="text-xs text-foreground-muted">{agent.progress}%</span>
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-foreground-subtle mt-3">
        <span>
          {agent.type === "ollama" ? "FREE (Ollama)" : `$${(agent.costEstimate || 0).toFixed(2)}`}
        </span>
        {agent.tokensGenerated !== undefined && (
          <span>{agent.tokensGenerated.toLocaleString()} tokens</span>
        )}
      </div>
    </div>
  );
}

function TaskRow({ task }: { task: Task }) {
  return (
    <div className="flex items-center gap-3 py-2">
      {task.status === "completed" ? (
        <CheckCircle2 className="w-4 h-4 text-green-400" />
      ) : task.status === "in_progress" ? (
        <PlayCircle className="w-4 h-4 text-blue-400" />
      ) : (
        <Circle className="w-4 h-4 text-foreground-subtle" />
      )}
      <span
        className={cn(
          "flex-1 text-sm truncate",
          task.status === "completed"
            ? "text-foreground-muted line-through"
            : "text-foreground"
        )}
      >
        {task.subject}
      </span>
      {task.owner && (
        <span className="text-xs text-foreground-subtle">{task.owner}</span>
      )}
    </div>
  );
}

function TaskCard({ task }: { task: Task }) {
  return (
    <div
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
          <div className="font-medium text-foreground-bright">{task.subject}</div>
          {task.description && (
            <p className="text-sm text-foreground-muted mt-1 line-clamp-2">
              {task.description}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-2 mt-2">
            {task.priority && (
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
            )}
            {task.owner && (
              <span className="text-xs text-foreground-subtle">
                Owner: {task.owner}
              </span>
            )}
            {task.activeForm && task.status === "in_progress" && (
              <span className="text-xs text-blue-400 italic">
                {task.activeForm}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function ProposalCard({ proposal }: { proposal: Proposal }) {
  const isPending = proposal.status === "pending" || proposal.status === "pending_review";

  return (
    <div
      className={cn(
        "p-4 rounded-lg border transition-colors",
        isPending
          ? "border-amber-500/30 bg-amber-500/5"
          : "border-border"
      )}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="font-medium text-foreground-bright">
            {proposal.title || proposal.storyId || proposal.id}
          </div>
          {proposal.summary && (
            <p className="text-sm text-foreground-muted mt-2 line-clamp-3">
              {proposal.summary}
            </p>
          )}
          {proposal.recommendation && (
            <p className="text-sm text-foreground mt-2">
              <strong>Recommendation:</strong> {proposal.recommendation}
            </p>
          )}
          <div className="flex items-center gap-2 mt-3">
            <span
              className={cn(
                "px-2 py-0.5 rounded text-xs",
                isPending
                  ? "bg-amber-500/20 text-amber-400"
                  : proposal.status === "approved"
                  ? "bg-green-500/20 text-green-400"
                  : "bg-red-500/20 text-red-400"
              )}
            >
              {proposal.status}
            </span>
            {proposal.createdAt && (
              <span className="text-xs text-foreground-subtle">
                {new Date(proposal.createdAt).toLocaleDateString()}
              </span>
            )}
          </div>
        </div>

        {isPending && (
          <div className="flex gap-2">
            <button className="p-2 rounded-lg bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-colors">
              <ThumbsUp className="w-4 h-4" />
            </button>
            <button className="p-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors">
              <ThumbsDown className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
