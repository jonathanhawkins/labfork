"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Lab3D from "@/components/Lab3D";
import { GpuStatsDialog } from "@/components/lab/GpuStatsDialog";
import { TokenUsageWidget } from "@/components/lab/TokenUsageWidget";
import {
  Brain,
  Cpu,
  Search,
  ListTodo,
  Activity,
  CheckCircle2,
  Circle,
  PlayCircle,
  Clock,
  Plus,
  X,
  Play,
  Check,
  Maximize2,
  Minimize2,
  ChevronLeft,
  ChevronRight,
  Mic,
  Volume2,
  Server,
  Zap,
} from "lucide-react";
import { useLabActivities, activitiesToLog } from "@/components/lab/activities";

interface Agent {
  id: string;
  name: string;
  color: number;
  position: [number, number, number];
  task?: string;
  status: "idle" | "working" | "thinking";
  iconKey?: string;
}

interface Task {
  id: string;
  subject: string;
  description?: string;
  status: "pending" | "in_progress" | "completed";
  owner?: string;
  blockedBy?: string[];
  activeForm?: string;
}

interface ResearchAgent {
  name: string;
  type: string;
  task: string;
  status: string;
  started_at: string;
}

interface AgentMessage {
  agent: string;
  message: string;
  timestamp: string;
  type: string;
}

// Pastel colors matching Lab3D
const COLORS = {
  codex: 0xffb3ba,
  opus: 0xbae1ff,
  explorer: 0xffffba,
  planner: 0xbaffc9,
  ollama: 0xd4baff, // Purple pastel for local AI
  labManager: 0x4ecdc4, // Teal/cyan - 4090 lab-manager
};

const TYPE_COLORS: Record<string, number> = {
  codex: COLORS.codex,
  opus: COLORS.opus,
  explorer: COLORS.explorer,
  planner: COLORS.planner,
  ollama: COLORS.ollama,
  "lab-manager": COLORS.labManager,
  agent: 0xcbd5f5,
};

// 4090 Agent status from backend
interface Agent4090Status {
  id: string;
  name: string;
  status: string;
  task?: string;
  lastOutput?: string;
  tokensGenerated?: number;
  elapsedTime?: string;
}

const AGENT_ICONS: Record<string, typeof Brain> = {
  codex: Cpu,
  opus: Brain,
  explorer: Search,
  planner: ListTodo,
  ollama: Zap, // Lightning bolt for fast local inference
  "lab-manager": Activity,
};

const ACTIVITY_ICONS: Record<string, typeof Mic> = {
  training: Cpu,
  recording: Mic,
  generation: Volume2,
  "live-transform": Zap,
  inference: Server,
  processing: Activity,
  task: ListTodo,
};

// ============== Section Component ==============

function Section({
  title,
  defaultOpen = false,
  badge,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-4 text-foreground-bright hover:text-foreground transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm">{title}</span>
          {badge}
        </div>
        <span className="text-muted-foreground">{isOpen ? "-" : "+"}</span>
      </button>
      {isOpen && <div className="px-4 pb-4 animate-fade-in">{children}</div>}
    </div>
  );
}

// ============== StatRow Component ==============

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
}

export default function LabPage() {
  const [agent4090Status, setAgent4090Status] = useState<Agent4090Status[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [researchAgents, setResearchAgents] = useState<
    Record<string, ResearchAgent>
  >({});
  const [showTaskForm, setShowTaskForm] = useState(false);
  const [newTaskSubject, setNewTaskSubject] = useState("");
  const [newTaskDescription, setNewTaskDescription] = useState("");

  // Error states for fetch operations
  const [fetchErrors, setFetchErrors] = useState<{
    tasks: string | null;
    agents: string | null;
    messages: string | null;
    health: string | null;
  }>({ tasks: null, agents: null, messages: null, health: null });

  const setFetchError = useCallback((key: "tasks" | "agents" | "messages" | "health", error: string | null) => {
    setFetchErrors(prev => ({ ...prev, [key]: error }));
  }, []);

  const {
    activities,
    activeCount,
    isLoading: activitiesLoading,
    lastUpdated,
  } = useLabActivities({ pollInterval: 3000 });

  const agents = useMemo<Agent[]>(() => {
    const normalizeAgentId = (raw?: string) =>
      (raw || "").trim().replace(/^rm:/, "");

    const summarizeTask = (taskText?: string) => {
      if (!taskText) return "Working...";
      const taskLine = taskText.match(/TASK #(\d+):\s*([^\n]+)/);
      if (taskLine) return `Task #${taskLine[1]}: ${taskLine[2]}`;
      // Skip raw prompts that start with rules/warnings
      if (taskText.startsWith('⚠') || taskText.includes('CRITICAL')) return "Researching...";
      const firstLine = taskText.split("\n").find((line) => line.trim());
      return (firstLine || taskText).slice(0, 80);
    };

    const inferType = (name: string, type?: string) => {
      const lower = `${name} ${type || ""}`.toLowerCase();
      if (lower.includes("codex")) return "codex";
      if (lower.includes("ollama")) return "ollama";
      if (lower.includes("opus")) return "opus";
      if (lower.includes("planner")) return "planner";
      if (lower.includes("explorer")) return "explorer";
      return type || "agent";
    };

    const getShortId = (name: string) => {
      const match = name.match(/task-(\d+)/);
      if (match) return `#${match[1]}`;
      if (name.length <= 8) return name;
      return name.slice(-6);
    };

    const FUN_NAMES = [
      "Opus Popus", "Noodle", "Bloop", "Sprocket", "Brainy",
      "Sparky", "Cruncher", "Scouty", "Wobbles", "Fizz",
      "Pixel", "Ziggy", "Turbo", "Nibbles", "Blinky",
      "Cosmo", "Doodle", "Gizmo", "Jinx", "Mochi",
    ];

    const hashStr = (s: string) => {
      let h = 0;
      for (let i = 0; i < s.length; i++) {
        h = ((h << 5) - h + s.charCodeAt(i)) | 0;
      }
      return Math.abs(h);
    };

    const formatName = (name: string, type: string) => {
      const funName = FUN_NAMES[hashStr(name) % FUN_NAMES.length];
      const label = type.toUpperCase();
      return `${funName}\n${label}`;
    };

    const POSITION_PRESETS: [number, number, number][] = [
      [-3, 0, -2],
      [3, 0, -2],
      [-3, 0, 2],
      [3, 0, 2],
      [-1.5, 0, -1.5],
      [1.5, 0, -1.5],
      [-1.5, 0, 1.5],
      [1.5, 0, 1.5],
      [-5, 0, 0],
      [5, 0, 0],
    ];

    const getPositionForIndex = (index: number): [number, number, number] => {
      if (index < POSITION_PRESETS.length) return POSITION_PRESETS[index];
      const ringIndex = index - POSITION_PRESETS.length;
      const slots = 8;
      const ring = Math.floor(ringIndex / slots) + 1;
      const angle = ((ringIndex % slots) / slots) * Math.PI * 2;
      const radius = 3.5 + ring * 1.3;
      return [Math.cos(angle) * radius, 0, Math.sin(angle) * radius];
    };

    const taskByOwner = new Map<string, Task>();
    tasks
      .filter((task) => task.status === "in_progress" && task.owner)
      .forEach((task) => {
        const owner = normalizeAgentId(task.owner);
        if (owner) taskByOwner.set(owner, task);
      });

    const dynamicAgents: Agent[] = Object.entries(researchAgents)
      .filter(([, agent]) => agent?.status === "running")
      .filter(([name]) => !name.includes("manager") && !name.includes("auto-improver") && !name.includes("loop"))
      .map(([name, agent], index) => {
        const agentId = normalizeAgentId(agent?.name || name) || name;
        const agentType = inferType(agentId, agent?.type);
        const assignedTask = taskByOwner.get(agentId);
        return {
          id: agentId,
          name: formatName(agentId, agentType),
          iconKey: agentType,
          color: TYPE_COLORS[agentType] || TYPE_COLORS.agent,
          position: getPositionForIndex(index),
          task: assignedTask?.activeForm || assignedTask?.subject || summarizeTask(agent?.task),
          status: assignedTask ? "working" : "thinking",
        };
      });

    // Add 4090 agents with real status (only if actually running)
    const agents4090: Agent[] = agent4090Status
      .filter(a => a.status === 'working')
      .map((a, index) => {
        const agentType = inferType(a.name || a.id, (a as any).type);
        return {
          id: a.id || a.name,
          name: formatName(a.name || a.id, agentType),
          iconKey: agentType,
          color: TYPE_COLORS[agentType] || TYPE_COLORS.agent,
          position: getPositionForIndex(dynamicAgents.length + index),
          task: a.task || a.lastOutput || "Working...",
          status: "working" as const,
        };
      });

    return [...dynamicAgents, ...agents4090];
  }, [agent4090Status, researchAgents, tasks]);

  const activityLog = useMemo(() => {
    return activitiesToLog(activities);
  }, [activities]);

  const sortedTasks = useMemo(() => {
    const statusOrder = { in_progress: 0, pending: 1, completed: 2 };
    return [...tasks].sort((a, b) => {
      const statusDiff = statusOrder[a.status] - statusOrder[b.status];
      if (statusDiff !== 0) return statusDiff;
      return parseInt(b.id) - parseInt(a.id);
    });
  }, [tasks]);

  const [isCreatingTask, setIsCreatingTask] = useState(false);

  const [autoSpawnEnabled, setAutoSpawnEnabled] = useState(true);
  const [autoSpawnStatus, setAutoSpawnStatus] = useState<{
    pendingTasks: number;
    runningAgents: number;
    canSpawn: boolean;
    lastSpawn?: string;
    research?: {
      hasResearcher: boolean;
      isDue: boolean;
      nextTopic: string;
      topicIndex: number;
      totalTopics: number;
    };
  } | null>(null);

  const [agentMessages, setAgentMessages] = useState<AgentMessage[]>([]);

  const [healthStatus, setHealthStatus] = useState<{
    healthy: boolean;
    runningCount: number;
    stuckCount: number;
    errorCount: number;
    recommendations: string[];
    stuckAgents: string[];
  } | null>(null);

  const [progressData, setProgressData] = useState<{
    agents: {
      name: string;
      taskId?: string;
      progressScore: number;
      status: string;
      statusReason: string;
      toolCalls: number;
      filesWritten: number;
    }[];
    stats: {
      totalAttempts: number;
      completed: number;
      stuck: number;
      avgProgressScore: number;
    };
  } | null>(null);

  const [metrics, setMetrics] = useState<{
    totalTasksCompleted: number;
    totalTasksFailed: number;
    successRate: number;
    avgCompletionTime: number;
    last24h: { completed: number; failed: number };
    byOutcome: { completed: number; stuck: number; error: number; timeout: number };
    estimatedCostToday: number;
    orchestratorUptime: number | null;
  } | null>(null);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [gpuStatsOpen, setGpuStatsOpen] = useState(false);
  const [agentOutputs, setAgentOutputs] = useState<Record<string, { lines: string[]; file: string }>>({});
  const [demoResult, setDemoResult] = useState<{
    agent: string;
    task: string;
    response: string;
    duration_ms: number;
    simulated: boolean;
  } | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [selectedDemoAgent, setSelectedDemoAgent] = useState("synergy-detector");
  const fullscreenRef = useRef<HTMLDivElement>(null);

  const toggleFullscreen = useCallback(async () => {
    if (!isFullscreen) {
      try {
        if (fullscreenRef.current?.requestFullscreen) {
          await fullscreenRef.current.requestFullscreen();
        }
      } catch (err) {
        console.error("Fullscreen request failed:", err);
      }
    } else {
      try {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        }
      } catch (err) {
        console.error("Exit fullscreen failed:", err);
      }
    }
  }, [isFullscreen]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFullscreen) {
        toggleFullscreen();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen, toggleFullscreen]);

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const response = await fetch("/api/tasks");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.tasks) {
          setTasks(data.tasks);
        }
        if (data.agents) {
          setResearchAgents(data.agents);
        }
        setFetchError("tasks", null);
      } catch (error) {
        console.error("Failed to fetch tasks:", error);
        setFetchError("tasks", "Unable to load tasks");
      }
    };

    fetchTasks();
    const interval = setInterval(fetchTasks, 5000);
    return () => clearInterval(interval);
  }, [setFetchError]);

  useEffect(() => {
    const checkAutoSpawn = async () => {
      try {
        const response = await fetch("/api/lab/auto-spawn");
        const status = await response.json();
        setAutoSpawnStatus(status);

        if (autoSpawnEnabled && status.canSpawn && status.pendingTasks > 0) {
          console.log("[Lab] Auto-spawning agent for pending task...");
          const spawnResponse = await fetch("/api/lab/auto-spawn", {
            method: "POST",
          });
          const result = await spawnResponse.json();
          if (result.success) {
            console.log("[Lab] Agent spawned:", result.assignedTask?.subject);
            setAutoSpawnStatus((prev) => prev ? {
              ...prev,
              lastSpawn: new Date().toLocaleTimeString(),
              runningAgents: result.runningAgents,
              pendingTasks: result.pendingTasks,
            } : null);
          }
        }
      } catch (error) {
        console.error("Auto-spawn check failed:", error);
      }
    };

    checkAutoSpawn();
    const interval = setInterval(checkAutoSpawn, 30000);
    return () => clearInterval(interval);
  }, [autoSpawnEnabled]);

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await fetch("/api/lab/agent-messages");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.messages) {
          setAgentMessages(
            data.messages.map((m: AgentMessage) => ({
              ...m,
              timestamp: new Date(m.timestamp).toLocaleTimeString(),
            }))
          );
        }
        setFetchError("messages", null);
      } catch (error) {
        console.error("Failed to fetch agent messages:", error);
        setFetchError("messages", "Unable to load agent messages");
      }
    };

    fetchMessages();
    const interval = setInterval(fetchMessages, 5000);
    return () => clearInterval(interval);
  }, [setFetchError]);

  // Fetch 4090 agent status (lab-manager)
  useEffect(() => {
    const fetch4090Status = async () => {
      try {
        const response = await fetch("/api/lab/agent-status");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.agents) {
          setAgent4090Status(data.agents);
        }
        setFetchError("agents", null);
      } catch (error) {
        console.error("Failed to fetch 4090 agent status:", error);
        setFetchError("agents", "Unable to load agent status");
      }
    };

    fetch4090Status();
    const interval = setInterval(fetch4090Status, 3000); // Poll every 3 seconds
    return () => clearInterval(interval);
  }, [setFetchError]);

  // Fetch agent output from 4090 monitors
  useEffect(() => {
    const fetchAgentOutput = async () => {
      try {
        const response = await fetch("/api/lab/agent-output");
        const data = await response.json();
        if (data.outputs) {
          setAgentOutputs(data.outputs);
        }
      } catch (error) {
        console.error("Failed to fetch agent output:", error);
      }
    };

    fetchAgentOutput();
    const interval = setInterval(fetchAgentOutput, 8000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch("/api/lab/health");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        setHealthStatus(data);
        setFetchError("health", null);

        if (data.stuckCount > 0 && autoSpawnEnabled) {
          console.log("[Lab] Detected stuck agents, triggering cleanup...");
          await fetch("/api/lab/health", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "kill-stuck" }),
          });
        }
      } catch (error) {
        console.error("Health check failed:", error);
        setFetchError("health", "Unable to check system health");
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, [autoSpawnEnabled, setFetchError]);

  useEffect(() => {
    const fetchProgress = async () => {
      try {
        const response = await fetch("/api/lab/progress");
        const data = await response.json();
        setProgressData(data);
      } catch (error) {
        console.error("Progress fetch failed:", error);
      }
    };

    fetchProgress();
    const interval = setInterval(fetchProgress, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch("/api/lab/metrics");
        const data = await response.json();
        setMetrics(data);
      } catch (error) {
        console.error("Metrics fetch failed:", error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAgentClick = (agent: Agent) => {
    setSelectedAgent(agent);
  };

  const handleCreateTask = async () => {
    if (!newTaskSubject.trim()) return;

    setIsCreatingTask(true);
    try {
      const response = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          subject: newTaskSubject,
          description: newTaskDescription,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setTasks((prev) => [...prev, data.task]);
        setNewTaskSubject("");
        setNewTaskDescription("");
        setShowTaskForm(false);
      }
    } catch (error) {
      console.error("Failed to create task:", error);
    } finally {
      setIsCreatingTask(false);
    }
  };

  const handleUpdateTaskStatus = async (
    taskId: string,
    newStatus: "pending" | "in_progress" | "completed"
  ) => {
    try {
      const response = await fetch("/api/tasks", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: taskId, status: newStatus }),
      });

      if (response.ok) {
        setTasks((prev) =>
          prev.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t))
        );
      }
    } catch (error) {
      console.error("Failed to update task:", error);
    }
  };

  const getAgentColorClass = (agentId: string) => {
    switch (agentId) {
      case "codex":
        return "bg-foreground/20 text-foreground";
      case "opus":
        return "bg-foreground/15 text-foreground";
      case "explorer":
        return "bg-foreground/10 text-foreground";
      case "planner":
        return "bg-foreground/25 text-foreground";
      default:
        return "bg-muted-foreground/20 text-foreground";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "working":
        return (
          <span className="text-xs text-foreground-bright flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-foreground animate-pulse" />
            Working
          </span>
        );
      case "thinking":
        return (
          <span className="text-xs text-foreground flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />
            Thinking
          </span>
        );
      default:
        return (
          <span className="text-xs text-muted-foreground flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50" />
            Idle
          </span>
        );
    }
  };

  const getTaskStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-4 h-4 text-foreground" />;
      case "in_progress":
        return <PlayCircle className="w-4 h-4 text-foreground-bright" />;
      default:
        return <Circle className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const renderTaskList = (compact = false) => (
    <>
      {showTaskForm && (
        <div className="mb-4 p-4 border border-border rounded">
          <div className="space-y-3">
            <input
              type="text"
              placeholder="Task subject..."
              value={newTaskSubject}
              onChange={(e) => setNewTaskSubject(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:border-foreground"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleCreateTask();
                }
              }}
            />
            <textarea
              placeholder="Description (optional)..."
              value={newTaskDescription}
              onChange={(e) => setNewTaskDescription(e.target.value)}
              rows={2}
              className="w-full px-3 py-2 bg-background border border-border rounded text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:border-foreground resize-none"
            />
            <div className="flex gap-2">
              <button
                onClick={handleCreateTask}
                disabled={!newTaskSubject.trim() || isCreatingTask}
                className="flex-1 py-2 px-4 bg-foreground text-background rounded text-sm hover:bg-foreground-bright disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
              >
                {isCreatingTask ? (
                  "Creating..."
                ) : (
                  <>
                    <Plus className="w-4 h-4" /> Add Task
                  </>
                )}
              </button>
              <button
                onClick={() => {
                  setShowTaskForm(false);
                  setNewTaskSubject("");
                  setNewTaskDescription("");
                }}
                className="py-2 px-4 border border-border text-foreground rounded text-sm hover:bg-accent transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {sortedTasks.length > 0 ? (
        <div className="space-y-2">
          {sortedTasks.slice(0, compact ? 5 : undefined).map((task) => (
            <div
              key={task.id}
              className={`p-3 rounded border transition-colors ${
                task.status === "in_progress"
                  ? "border-foreground/30 bg-foreground/5"
                  : task.status === "completed"
                  ? "border-foreground/20 bg-foreground/3"
                  : "border-border"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2 flex-1 min-w-0">
                  {getTaskStatusIcon(task.status)}
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-foreground truncate">
                      #{task.id} {task.subject}
                    </div>
                    {task.activeForm && task.status === "in_progress" && (
                      <div className="text-xs text-foreground-bright mt-1 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {task.activeForm}
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1 flex-shrink-0">
                  {task.status !== "in_progress" && (
                    <button
                      onClick={() => handleUpdateTaskStatus(task.id, "in_progress")}
                      className="p-1 rounded border border-border hover:bg-accent text-foreground transition-colors"
                      title="Start task"
                    >
                      <Play className="w-3 h-3" />
                    </button>
                  )}
                  {task.status !== "completed" && (
                    <button
                      onClick={() => handleUpdateTaskStatus(task.id, "completed")}
                      className="p-1 rounded border border-border hover:bg-accent text-foreground transition-colors"
                      title="Complete task"
                    >
                      <Check className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
          {compact && sortedTasks.length > 5 && (
            <div className="text-xs text-muted-foreground text-center pt-1">
              +{sortedTasks.length - 5} more tasks
            </div>
          )}
        </div>
      ) : (
        <div className="text-muted-foreground text-center py-4 text-sm">
          No tasks yet. Click + to add.
        </div>
      )}
    </>
  );

  // Fullscreen layout
  if (isFullscreen) {
    return (
      <div
        ref={fullscreenRef}
        className="relative w-full h-screen bg-background overflow-hidden"
      >
        <div className="absolute inset-0">
          <Lab3D
            agents={agents}
            activities={activities}
            onAgentClick={handleAgentClick}
            onComputerClick={() => setGpuStatsOpen(true)}
          />
        </div>

        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3">
          <div className="flex items-center gap-2 px-4 py-2 bg-background-elevated/90 rounded border border-border">
            <span className="text-sm text-foreground-bright">Research Lab</span>
          </div>
          <button
            onClick={toggleFullscreen}
            className="p-2.5 bg-background-elevated/90 rounded border border-border hover:bg-accent transition-colors"
            title="Exit fullscreen"
          >
            <Minimize2 className="w-5 h-5 text-foreground" />
          </button>
        </div>

        <button
          onClick={() => setLeftPanelOpen(!leftPanelOpen)}
          className={`absolute top-1/2 -translate-y-1/2 z-30 p-2 bg-background-elevated/90 border border-border hover:bg-accent transition-all duration-300 ${
            leftPanelOpen ? "left-[320px] rounded-r" : "left-0 rounded-r"
          }`}
        >
          {leftPanelOpen ? (
            <ChevronLeft className="w-5 h-5 text-foreground" />
          ) : (
            <ChevronRight className="w-5 h-5 text-foreground" />
          )}
        </button>

        <div
          className={`absolute top-0 left-0 h-full w-[320px] z-20 transition-transform duration-300 ${
            leftPanelOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="h-full p-4 bg-background-elevated/95 border-r border-border overflow-y-auto">
            <Section title="Agent Details" defaultOpen>
              {selectedAgent ? (
                <div className="p-3 border border-border rounded">
                  <div className="flex items-center gap-2 mb-2">
                    {(() => {
                      const Icon = AGENT_ICONS[selectedAgent.iconKey || selectedAgent.id] || Brain;
                      return (
                        <div
                          className={`p-2 rounded ${getAgentColorClass(
                            selectedAgent.id
                          )}`}
                        >
                          <Icon className="w-4 h-4" />
                        </div>
                      );
                    })()}
                    <div>
                      <div className="text-sm text-foreground-bright">
                        {selectedAgent.name}
                      </div>
                      {getStatusBadge(selectedAgent.status)}
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {selectedAgent.task}
                  </div>
                  {/* Agent terminal output (fullscreen) */}
                  {(() => {
                    const output = Object.entries(agentOutputs).find(
                      ([name]) => name.toLowerCase().includes(selectedAgent.id.toLowerCase()) ||
                        selectedAgent.id.toLowerCase().includes(name.toLowerCase().replace(/[^a-z0-9]/g, ''))
                    );
                    if (!output) return null;
                    const [, data] = output;
                    return (
                      <div className="mt-3 rounded overflow-hidden border border-border">
                        <div
                          className="p-2 max-h-[160px] overflow-y-auto"
                          style={{ background: '#1a1a2e' }}
                        >
                          <pre className="text-xs leading-relaxed whitespace-pre-wrap break-all" style={{ color: '#00ff88', fontFamily: 'monospace' }}>
                            {data.lines.join('\n')}
                          </pre>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              ) : (
                <div className="text-xs text-muted-foreground text-center py-3">
                  Click an agent to see details
                </div>
              )}
            </Section>

            <Section title="Tasks" defaultOpen>
              <div className="flex items-center justify-end mb-2">
                <button
                  onClick={() => setShowTaskForm(!showTaskForm)}
                  className="p-1.5 rounded border border-border hover:bg-accent text-foreground transition-colors"
                >
                  {showTaskForm ? (
                    <X className="w-3 h-3" />
                  ) : (
                    <Plus className="w-3 h-3" />
                  )}
                </button>
              </div>
              {renderTaskList(true)}
            </Section>
          </div>
        </div>

        <button
          onClick={() => setRightPanelOpen(!rightPanelOpen)}
          className={`absolute top-1/2 -translate-y-1/2 z-30 p-2 bg-background-elevated/90 border border-border hover:bg-accent transition-all duration-300 ${
            rightPanelOpen ? "right-[280px] rounded-l" : "right-0 rounded-l"
          }`}
        >
          {rightPanelOpen ? (
            <ChevronRight className="w-5 h-5 text-foreground" />
          ) : (
            <ChevronLeft className="w-5 h-5 text-foreground" />
          )}
        </button>

        <div
          className={`absolute top-0 right-0 h-full w-[280px] z-20 transition-transform duration-300 ${
            rightPanelOpen ? "translate-x-0" : "translate-x-full"
          }`}
        >
          <div className="h-full p-4 bg-background-elevated/95 border-l border-border overflow-y-auto">
            <Section title="Agents" defaultOpen>
              <div className="space-y-2">
                {agents.map((agent) => {
                  const Icon = AGENT_ICONS[agent.iconKey || agent.id] || Brain;
                  return (
                    <div
                      key={agent.id}
                      className="flex items-center justify-between p-2 border border-border rounded cursor-pointer hover:bg-accent transition-colors"
                      onClick={() => setSelectedAgent(agent)}
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className={`p-1.5 rounded ${getAgentColorClass(
                            agent.id
                          )}`}
                        >
                          <Icon className="w-3 h-3" />
                        </div>
                        <span className="text-sm text-foreground">
                          {agent.name}
                        </span>
                      </div>
                      {getStatusBadge(agent.status)}
                    </div>
                  );
                })}
              </div>
            </Section>

            <Section title="Activity">
              <div className="space-y-1 max-h-[200px] overflow-y-auto">
                {activityLog.length > 0 ? (
                  activityLog.slice(0, 5).map((entry, idx) => (
                    <div
                      key={idx}
                      className="text-xs p-2 border border-border rounded"
                    >
                      <span className="text-muted-foreground">
                        {entry.time.toLocaleTimeString()}
                      </span>
                      <span className="text-muted-foreground mx-1">|</span>
                      <span className="text-foreground">{entry.agent}</span>
                      <div className="text-muted-foreground truncate">
                        {entry.action}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-muted-foreground text-center py-3 text-xs">
                    Waiting for activity...
                  </div>
                )}
              </div>
            </Section>
          </div>
        </div>

        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
          <div className="px-4 py-2 bg-background-elevated/90 rounded border border-border text-xs text-muted-foreground">
            Drag to rotate | Scroll to zoom | Click agents | Press ESC to exit
          </div>
        </div>
      </div>
    );
  }

  // Normal layout
  return (
    <div ref={fullscreenRef} className="min-h-screen bg-background flex">
      {/* Left Sidebar */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Agent Details" defaultOpen>
          {selectedAgent ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                {(() => {
                  const Icon = AGENT_ICONS[selectedAgent.iconKey || selectedAgent.id] || Brain;
                  return (
                    <div
                      className={`p-3 rounded ${getAgentColorClass(
                        selectedAgent.id
                      )}`}
                    >
                      <Icon className="w-5 h-5" />
                    </div>
                  );
                })()}
                <div>
                  <h3 className="text-sm text-foreground-bright">
                    {selectedAgent.name}
                  </h3>
                  {getStatusBadge(selectedAgent.status)}
                </div>
              </div>
              <div className="border border-border rounded p-3">
                <div className="text-xs text-muted-foreground mb-1">
                  Current Task
                </div>
                <div className="text-sm text-foreground">{selectedAgent.task}</div>
              </div>
              {/* Agent terminal output */}
              {(() => {
                const output = Object.entries(agentOutputs).find(
                  ([name]) => name.toLowerCase().includes(selectedAgent.id.toLowerCase()) ||
                    selectedAgent.id.toLowerCase().includes(name.toLowerCase().replace(/[^a-z0-9]/g, ''))
                );
                if (!output) return null;
                const [, data] = output;
                return (
                  <div className="border border-border rounded overflow-hidden">
                    <div className="text-xs text-muted-foreground px-3 py-1.5 border-b border-border flex items-center justify-between">
                      <span>Terminal Output</span>
                      <span className="text-foreground-subtle truncate ml-2 max-w-[140px]">{data.file.split('/').pop()}</span>
                    </div>
                    <div
                      className="p-3 max-h-[200px] overflow-y-auto"
                      style={{ background: '#1a1a2e' }}
                    >
                      <pre className="text-xs leading-relaxed whitespace-pre-wrap break-all" style={{ color: '#00ff88', fontFamily: 'monospace' }}>
                        {data.lines.join('\n')}
                      </pre>
                    </div>
                  </div>
                );
              })()}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground text-center py-8">
              Click on an agent to see details
            </div>
          )}
        </Section>

        <Section title="Auto-Spawn">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Auto-assign tasks</span>
              <button
                onClick={() => setAutoSpawnEnabled(!autoSpawnEnabled)}
                className={`px-3 py-1 rounded text-xs transition-colors ${
                  autoSpawnEnabled
                    ? "bg-foreground text-background"
                    : "border border-border text-muted-foreground"
                }`}
              >
                {autoSpawnEnabled ? "Enabled" : "Disabled"}
              </button>
            </div>

            {healthStatus && (
              <div
                className={`p-2 rounded text-xs flex items-center justify-between ${
                  healthStatus.healthy
                    ? "border border-foreground/20"
                    : "border border-muted-foreground"
                }`}
              >
                <span className={healthStatus.healthy ? "text-foreground" : "text-muted-foreground"}>
                  {healthStatus.healthy
                    ? "System Healthy"
                    : `${healthStatus.stuckCount} Stuck Agents`}
                </span>
                <span
                  className={`w-2 h-2 rounded-full ${
                    healthStatus.healthy ? "bg-foreground" : "bg-muted-foreground animate-pulse"
                  }`}
                />
              </div>
            )}

            {autoSpawnStatus && (
              <div className="space-y-2">
                <StatRow label="Pending tasks" value={autoSpawnStatus.pendingTasks} />
                <StatRow label="Running agents" value={autoSpawnStatus.runningAgents} />
                {autoSpawnStatus.lastSpawn && (
                  <StatRow label="Last spawn" value={autoSpawnStatus.lastSpawn} />
                )}
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={async () => {
                  try {
                    const response = await fetch("/api/lab/auto-spawn", {
                      method: "POST",
                    });
                    const result = await response.json();
                    if (result.success) {
                      setAutoSpawnStatus((prev) => ({
                        ...prev!,
                        lastSpawn: new Date().toLocaleTimeString(),
                        runningAgents: result.runningAgents,
                        pendingTasks: result.pendingTasks,
                      }));
                    }
                  } catch (error) {
                    console.error("Manual spawn failed:", error);
                  }
                }}
                disabled={!autoSpawnStatus?.canSpawn}
                className="flex-1 py-2 px-3 rounded text-sm border border-border text-foreground hover:bg-accent disabled:opacity-50 transition-colors"
              >
                Spawn Task
              </button>
              <button
                onClick={async () => {
                  try {
                    const response = await fetch("/api/lab/auto-spawn?research=true", {
                      method: "POST",
                    });
                    const result = await response.json();
                    if (result.success) {
                      setAutoSpawnStatus((prev) => ({
                        ...prev!,
                        lastSpawn: new Date().toLocaleTimeString(),
                        runningAgents: result.runningAgents,
                      }));
                    }
                  } catch (error) {
                    console.error("Research spawn failed:", error);
                  }
                }}
                disabled={
                  autoSpawnStatus?.research?.hasResearcher ||
                  (autoSpawnStatus?.runningAgents ?? 0) >= 3
                }
                className="flex-1 py-2 px-3 rounded text-sm border border-border text-foreground hover:bg-accent disabled:opacity-50 transition-colors flex items-center justify-center gap-1"
              >
                <Search className="w-3 h-3" />
                Research
              </button>
            </div>
          </div>
        </Section>

        {metrics && (
          <Section title="Metrics">
            <div className="space-y-2">
              <StatRow label="Success Rate" value={`${metrics.successRate ?? 0}%`} />
              <StatRow label="Completed (24h)" value={metrics.last24h?.completed ?? 0} />
              <StatRow label="Failed (24h)" value={metrics.last24h?.failed ?? 0} />
              <StatRow label="Avg time" value={`${metrics.avgCompletionTime ?? 0}m`} />
              <div className="pt-2 mt-2 border-t border-border">
                <StatRow label="Est. cost today" value={`$${(metrics.estimatedCostToday ?? 0).toFixed(2)}`} />
              </div>
            </div>
          </Section>
        )}

        <Section title="AI Agent Demo" defaultOpen>
          <div className="space-y-3">
            <div className="text-xs text-muted-foreground mb-2">
              Test Ollama-powered agents
            </div>
            <select
              value={selectedDemoAgent}
              onChange={(e) => setSelectedDemoAgent(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-border rounded text-sm text-foreground focus:outline-none focus:border-foreground"
            >
              <option value="synergy-detector">Synergy Detector</option>
              <option value="pattern-recognizer">Pattern Recognizer</option>
              <option value="gap-analyzer">Gap Analyzer</option>
              <option value="evolution-engine">Evolution Engine</option>
              <option value="transfer-agent">Transfer Agent</option>
              <option value="lab-manager">Lab Manager</option>
            </select>
            <button
              onClick={async () => {
                setDemoLoading(true);
                setDemoResult(null);
                try {
                  const response = await fetch("/api/agents/demo", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ agent: selectedDemoAgent }),
                  });
                  const data = await response.json();
                  setDemoResult(data);
                } catch (error) {
                  console.error("Demo failed:", error);
                } finally {
                  setDemoLoading(false);
                }
              }}
              disabled={demoLoading}
              className="w-full py-2 px-4 bg-foreground text-background rounded text-sm hover:bg-foreground-bright disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
              {demoLoading ? (
                <>
                  <span className="w-4 h-4 border-2 border-background/30 border-t-background rounded-full animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Run Demo Task
                </>
              )}
            </button>
            {demoResult && (
              <div className="border border-border rounded overflow-hidden">
                <div className="px-3 py-2 border-b border-border bg-background-elevated flex items-center justify-between">
                  <span className="text-xs text-foreground-bright">{demoResult.agent}</span>
                  <span className="text-xs text-muted-foreground">
                    {demoResult.simulated ? "Simulated" : `${demoResult.duration_ms}ms`}
                  </span>
                </div>
                <div className="p-3 max-h-[200px] overflow-y-auto text-xs">
                  <div className="text-muted-foreground mb-2 italic">{demoResult.task}</div>
                  <div className="text-foreground whitespace-pre-wrap">{demoResult.response}</div>
                </div>
              </div>
            )}
          </div>
        </Section>

        <Section title="Token Usage" defaultOpen>
          <TokenUsageWidget />
        </Section>
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-lg text-foreground-bright">Research Lab</h1>
              <p className="text-sm text-muted-foreground">
                Watch the research agents collaborate in their 3D world
              </p>
            </div>
            <button
              onClick={toggleFullscreen}
              className="flex items-center gap-2 px-4 py-2 border border-border text-foreground rounded hover:bg-accent transition-colors"
            >
              <Maximize2 className="w-4 h-4" />
              <span className="text-sm">Fullscreen</span>
            </button>
          </div>

          {/* Error Banner */}
          {Object.values(fetchErrors).some(Boolean) && (
            <div className="mb-4 p-3 border border-amber-500/30 bg-amber-500/10 rounded-lg">
              <div className="flex items-center gap-2 text-amber-400 text-sm">
                <Activity className="w-4 h-4" />
                <span>Some data could not be loaded:</span>
              </div>
              <ul className="mt-2 text-xs text-amber-300/80 space-y-1 pl-6">
                {fetchErrors.tasks && <li>{fetchErrors.tasks}</li>}
                {fetchErrors.agents && <li>{fetchErrors.agents}</li>}
                {fetchErrors.messages && <li>{fetchErrors.messages}</li>}
                {fetchErrors.health && <li>{fetchErrors.health}</li>}
              </ul>
            </div>
          )}

          {/* 3D Visualization */}
          <div className="border border-border rounded overflow-hidden mb-4">
            <div className="h-[700px]">
              <Lab3D
                agents={agents}
                activities={activities}
                onAgentClick={handleAgentClick}
                onComputerClick={() => setGpuStatsOpen(true)}
              />
            </div>
          </div>

          <div className="text-center text-xs text-muted-foreground mb-6">
            Drag to rotate | Scroll to zoom | Click on an agent to see details | Click the supercomputer for GPU stats
          </div>

          {/* Task List */}
          <div className="border border-border rounded p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <ListTodo className="w-4 h-4 text-muted-foreground" />
                <h2 className="text-sm text-foreground-bright">Shared Task List</h2>
                <span className="text-xs text-muted-foreground">labfork</span>
              </div>
              <button
                onClick={() => setShowTaskForm(!showTaskForm)}
                className="p-2 rounded border border-border hover:bg-accent text-foreground transition-colors"
              >
                {showTaskForm ? (
                  <X className="w-4 h-4" />
                ) : (
                  <Plus className="w-4 h-4" />
                )}
              </button>
            </div>
            {renderTaskList()}
          </div>
        </div>
      </main>

      {/* Right Sidebar */}
      <aside className="w-[280px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section
          title="Active Activities"
          defaultOpen
          badge={
            activeCount > 0 ? (
              <span className="text-xs text-foreground-bright">{activeCount}</span>
            ) : undefined
          }
        >
          {activities.filter((a) => a.active).length > 0 ? (
            <div className="space-y-3">
              {activities
                .filter((a) => a.active)
                .map((activity) => {
                  const Icon = ACTIVITY_ICONS[activity.type] || Activity;
                  return (
                    <div
                      key={activity.id}
                      className="p-3 rounded border border-border"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <Icon className="w-4 h-4 text-foreground" />
                          <span className="text-sm text-foreground-bright">
                            {activity.config.name}
                          </span>
                        </div>
                        {activity.progress !== undefined && (
                          <span className="text-xs text-muted-foreground">
                            {activity.progress}%
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground truncate">
                        {activity.message}
                      </div>
                      {activity.assignedAgent && (
                        <div className="text-xs text-foreground-subtle mt-1">
                          Agent: {activity.assignedAgent}
                        </div>
                      )}
                      {activity.progress !== undefined && (
                        <div className="mt-2 h-1 bg-border rounded-full overflow-hidden">
                          <div
                            className="h-full bg-foreground transition-all duration-300"
                            style={{ width: `${activity.progress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          ) : (
            <div className="text-muted-foreground text-center py-4 text-sm">
              {activitiesLoading ? "Loading activities..." : "No active activities"}
            </div>
          )}
          {lastUpdated && (
            <div className="text-xs text-foreground-subtle text-center mt-3">
              Updated {lastUpdated.toLocaleTimeString()}
            </div>
          )}
        </Section>

        {Object.keys(researchAgents).length > 0 && (
          <Section title="Research Agents">
            <div className="space-y-3">
              {Object.entries(researchAgents).map(([name, agent]) => (
                <div
                  key={name}
                  className={`p-3 rounded border ${
                    agent.status === "running"
                      ? "border-foreground/30"
                      : "border-border"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-foreground-bright">{name}</span>
                    <span
                      className={`text-xs ${
                        agent.status === "running"
                          ? "text-foreground"
                          : "text-muted-foreground"
                      }`}
                    >
                      {agent.status}
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Type: {agent.type}
                  </div>
                  <div className="text-xs text-foreground-subtle truncate mt-1">
                    {agent.task?.substring(0, 50)}...
                  </div>
                </div>
              ))}
            </div>
          </Section>
        )}

        {progressData?.agents?.length > 0 && (
          <Section title="Agent Progress">
            <div className="space-y-3">
              {progressData.agents.map((agent) => (
                <div
                  key={agent.name}
                  className={`p-3 rounded border ${
                    agent.status === "error"
                      ? "border-muted-foreground"
                      : agent.status === "stuck"
                      ? "border-muted-foreground/50"
                      : "border-border"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-foreground-bright">
                      {agent.taskId ? `Task #${agent.taskId}` : agent.name.split("-")[0]}
                    </span>
                    <span className="text-xs text-muted-foreground">{agent.status}</span>
                  </div>
                  <div className="flex items-center gap-2 mb-1">
                    <div className="flex-1 h-1 bg-border rounded-full overflow-hidden">
                      <div
                        className="h-full bg-foreground transition-all duration-500"
                        style={{ width: `${agent.progressScore}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground w-8">
                      {agent.progressScore}%
                    </span>
                  </div>
                  <div className="text-xs text-foreground-subtle">
                    {agent.toolCalls} tools | {agent.filesWritten} files written
                  </div>
                  {agent.statusReason && (
                    <div className="text-xs text-muted-foreground mt-1 italic">
                      {agent.statusReason}
                    </div>
                  )}
                </div>
              ))}
            </div>
            {progressData.stats.totalAttempts > 0 && (
              <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground">
                History: {progressData.stats.completed}/{progressData.stats.totalAttempts} completed
                | Avg score: {progressData.stats.avgProgressScore}%
              </div>
            )}
          </Section>
        )}

        <Section title="3D Agent Status">
          <div className="space-y-3">
            {agents.map((agent) => {
              const Icon = AGENT_ICONS[agent.iconKey || agent.id] || Brain;
              return (
                <div
                  key={agent.id}
                  className="flex items-center justify-between p-3 border border-border rounded cursor-pointer hover:bg-accent transition-colors"
                  onClick={() => setSelectedAgent(agent)}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded ${getAgentColorClass(agent.id)}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="text-sm text-foreground">{agent.name}</span>
                  </div>
                  {getStatusBadge(agent.status)}
                </div>
              );
            })}
          </div>
        </Section>

        <Section title="Activity Log">
          <div className="flex items-center gap-1 mb-2">
            <span className="w-2 h-2 rounded-full bg-foreground animate-pulse" />
            <span className="text-xs text-muted-foreground">Live</span>
          </div>
          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {activityLog.length > 0 && (
              <>
                <div className="text-xs text-foreground-subtle mb-1">Active Tasks</div>
                {activityLog.map((entry, idx) => (
                  <div
                    key={`activity-${idx}`}
                    className="text-xs p-2 border-l-2 border-foreground/30 bg-foreground/5"
                  >
                    <span className="text-muted-foreground">
                      {entry.time.toLocaleTimeString()}
                    </span>
                    <span className="text-muted-foreground mx-2">|</span>
                    <span className="text-foreground">{entry.agent}</span>
                    <span className="text-muted-foreground mx-2">-</span>
                    <span className="text-foreground">{entry.action}</span>
                  </div>
                ))}
              </>
            )}

            {agentMessages.length > 0 && (
              <>
                <div className="text-xs text-foreground-subtle mt-3 mb-1">Agent Output</div>
                {agentMessages.slice(0, 10).map((msg, idx) => (
                  <div
                    key={`msg-${idx}`}
                    className="text-xs p-2 border-l-2 border-border"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-muted-foreground">{msg.timestamp}</span>
                      <span className="text-foreground">{msg.agent}</span>
                      <span className="text-xs text-muted-foreground">{msg.type}</span>
                    </div>
                    <div className="text-foreground truncate">{msg.message}</div>
                  </div>
                ))}
              </>
            )}

            {activityLog.length === 0 && agentMessages.length === 0 && (
              <div className="text-muted-foreground text-center py-4">
                Waiting for activity...
              </div>
            )}
          </div>
        </Section>
      </aside>

      {/* GPU Stats Dialog */}
      <GpuStatsDialog open={gpuStatsOpen} onOpenChange={setGpuStatsOpen} />
    </div>
  );
}
