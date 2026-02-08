"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Lab3D from "@/components/Lab3D";
import { useLabActivities, activitiesToLog } from "@/components/lab/activities";
import {
  useAgentStatus,
  useAgentWorkLog,
  workLogToActivityLog,
  type AgentStatus as WorkersAgentStatus,
} from "@/hooks/useAgentStatus";
import {
  Brain,
  Cpu,
  Search,
  ListTodo,
  Activity,
  ThermometerSun,
  HardDrive,
  Server,
  Monitor,
  Zap,
  AlertCircle,
  CheckCircle2,
  Clock,
} from "lucide-react";
import { useCompletedTasks } from "@/hooks/useCompletedTasks";

// ============== Fun Agent Names ==============

const FUN_NAMES = [
  "Opus Popus", "Noodle", "Bloop", "Sprocket", "Brainy",
  "Sparky", "Cruncher", "Scouty", "Wobbles", "Fizz",
  "Pixel", "Ziggy", "Turbo", "Nibbles", "Blinky",
  "Cosmo", "Doodle", "Gizmo", "Jinx", "Mochi",
];

function getFunName(id: string): string {
  let h = 0;
  for (let i = 0; i < id.length; i++) {
    h = ((h << 5) - h + id.charCodeAt(i)) | 0;
  }
  return FUN_NAMES[Math.abs(h) % FUN_NAMES.length];
}

// ============== Data Sanitization Utilities ==============

/**
 * Patterns that should be sanitized from public display
 */
const SENSITIVE_PATTERNS = [
  // IP addresses (IPv4)
  /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g,
  // File paths
  /\/Users\/[^\s]+/g,
  /\/home\/[^\s]+/g,
  /~\/[^\s]+/g,
  /C:\\[^\s]+/gi,
  // SSH commands
  /ssh\s+\S+@\S+/g,
  // Environment variables
  /\$\{?[A-Z_]+\}?/g,
  // Secrets/tokens
  /token[=:]\s*\S+/gi,
  /password[=:]\s*\S+/gi,
  /api[_-]?key[=:]\s*\S+/gi,
];

/**
 * Sanitize a string by removing sensitive information
 */
function sanitizeString(str: string): string {
  let result = str;

  for (const pattern of SENSITIVE_PATTERNS) {
    result = result.replace(pattern, "[redacted]");
  }

  // Replace known IPs with friendly names
  result = result.replace(/100\.83\.78\.111/g, "Training Cluster");
  result = result.replace(/localhost:\d+/g, "local service");
  result = result.replace(/127\.0\.0\.1:\d+/g, "local service");

  return result;
}

/**
 * Sanitize task/activity messages for public display
 */
function sanitizeMessage(message: string): string {
  // First apply general sanitization
  let result = sanitizeString(message);

  // Remove config file paths but keep the action
  result = result.replace(/config\/\S+\.yaml/g, "training config");
  result = result.replace(/checkpoints?\/\S+/g, "model checkpoint");

  return result;
}

/**
 * Sanitize GPU stats for public display
 */
interface RawGpuStats {
  connected: boolean;
  error?: string;
  timestamp: string;
  gpu: {
    name: string;
    driverVersion: string;
    cudaVersion: string;
    utilization: number;
    memoryUsed: number;
    memoryTotal: number;
    memoryPercent: number;
    temperature: number;
    powerDraw: number;
    powerLimit: number;
    fanSpeed?: number;
  } | null;
  processes: Array<{
    pid: string;
    name: string;
    memoryUsed: string;
    script?: string;
    session?: string;
    config?: string;
    progress?: string;
  }>;
}

interface SanitizedGpuStats {
  connected: boolean;
  clusterName: string;
  gpu: {
    name: string;
    utilization: number;
    memoryPercent: number;
    temperature: number;
    powerPercent: number;
  } | null;
  hasActiveTraining: boolean;
  trainingStatus?: string;
}

function sanitizeGpuStats(raw: RawGpuStats | null): SanitizedGpuStats {
  if (!raw || !raw.connected || !raw.gpu) {
    return {
      connected: false,
      clusterName: "Training Cluster",
      gpu: null,
      hasActiveTraining: false,
    };
  }

  // Extract training status without revealing paths
  let trainingStatus: string | undefined;
  const hasTraining = raw.processes && raw.processes.length > 0;

  if (hasTraining) {
    const proc = raw.processes[0];
    if (proc.progress) {
      // Sanitize progress string (e.g., "Epoch 22 - loss: 0.5734")
      trainingStatus = sanitizeMessage(proc.progress);
    } else {
      trainingStatus = "Training in progress";
    }
  }

  return {
    connected: true,
    clusterName: "Training Cluster",
    gpu: {
      name: raw.gpu.name.replace(/NVIDIA\s*/i, "").trim(),
      utilization: raw.gpu.utilization,
      memoryPercent: raw.gpu.memoryPercent,
      temperature: raw.gpu.temperature,
      powerPercent: Math.round((raw.gpu.powerDraw / raw.gpu.powerLimit) * 100),
    },
    hasActiveTraining: hasTraining,
    trainingStatus,
  };
}

// ============== Agent Types ==============

interface Agent {
  id: string;
  name: string;
  color: number;
  position: [number, number, number];
  task?: string;
  status: "idle" | "working" | "thinking";
}

interface Agent4090Status {
  id: string;
  name: string;
  status: string;
  task?: string;
  lastOutput?: string;
  tokensGenerated?: number;
  elapsedTime?: string;
}

// Pastel colors matching Lab3D
const COLORS = {
  codex: 0xffb3ba,
  opus: 0xbae1ff,
  explorer: 0xffffba,
  planner: 0xbaffc9,
  labManager: 0x4ecdc4,
  researcher: 0xbae1ff,
  developer: 0xffb3ba,
  coordinator: 0xbaffc9,
};

const AGENT_ICONS: Record<string, typeof Brain> = {
  codex: Cpu,
  opus: Brain,
  explorer: Search,
  planner: ListTodo,
  "lab-manager": Activity,
  researcher: Brain,
  developer: Cpu,
  coordinator: ListTodo,
};

// Helper to get agent color based on type
function getAgentColor(type: string): number {
  const typeMap: Record<string, number> = {
    researcher: COLORS.researcher,
    developer: COLORS.developer,
    explorer: COLORS.explorer,
    coordinator: COLORS.coordinator,
    codex: COLORS.codex,
    opus: COLORS.opus,
    planner: COLORS.planner,
  };
  return typeMap[type.toLowerCase()] || COLORS.labManager;
}

// Helper to map agent status from Workers API format
function mapAgentStatus(status: string): "idle" | "working" | "thinking" {
  switch (status) {
    case 'working':
      return 'working';
    case 'blocked':
      return 'thinking';
    case 'idle':
    default:
      return 'idle';
  }
}

// ============== Public Lab View Component ==============

interface PublicLabViewProps {
  showSuggestions?: boolean;
}

// Compute network stats interface
interface ComputeNetworkStats {
  totalDevices: number;
  busyDevices: number;
  tierCounts: { power: number; standard: number; crowd: number };
  totalCompute: number; // TFLOPS
}

// ============== Demo Data (Stable Constants) ==============

// Demo agents to show when no real agents are running
const DEMO_AGENTS: Agent[] = [
  { id: "opus", name: "Opus", color: COLORS.opus, position: [-3, 0, -2], task: "Analyzing research papers", status: "working" },
  { id: "codex", name: "Codex", color: COLORS.codex, position: [3, 0, -2], task: "Implementing prosody model", status: "working" },
  { id: "explorer", name: "Scout", color: COLORS.explorer, position: [-3, 0, 2], task: "Searching for synergies", status: "thinking" },
  { id: "planner", name: "Planner", color: COLORS.planner, position: [3, 0, 2], task: "Scheduling training runs", status: "idle" },
];

// Demo activities when no real activities
const DEMO_ACTIVITIES = [
  { time: new Date(Date.now() - 30000), agent: "Opus", action: "Completed analysis of voice prosody patterns" },
  { time: new Date(Date.now() - 60000), agent: "Codex", action: "Training step 2847/5000 - loss: 0.0234" },
  { time: new Date(Date.now() - 120000), agent: "Scout", action: "Found synergy: EmoProsody + StyleTransfer" },
  { time: new Date(Date.now() - 180000), agent: "Planner", action: "Scheduled overnight training batch" },
  { time: new Date(Date.now() - 240000), agent: "Opus", action: "Reviewing MaskGCT paper implementation" },
];

// Demo compute stats as fallback
const DEMO_COMPUTE_STATS: ComputeNetworkStats = {
  totalDevices: 12,
  busyDevices: 8,
  tierCounts: { power: 2, standard: 5, crowd: 5 },
  totalCompute: 45.2,
};

export function PublicLabView({ showSuggestions = false }: PublicLabViewProps) {
  const [agent4090Status, setAgent4090Status] = useState<Agent4090Status[]>([]);
  const [gpuStats, setGpuStats] = useState<SanitizedGpuStats | null>(null);
  const [computeNetworkStats, setComputeNetworkStats] = useState<ComputeNetworkStats | null>(null);

  // Fetch agent status from Cloudflare Workers API (new autonomous agent system)
  const {
    agents: workersAgents,
    isLoading: workersAgentsLoading,
    isDemo: isWorkersDemo,
    lastUpdated: workersLastUpdated,
  } = useAgentStatus({ pollInterval: 5000 });

  // Fetch work log from Cloudflare Workers API
  const {
    entries: workLogEntries,
    isLoading: workLogLoading,
    isDemo: isWorkLogDemo,
  } = useAgentWorkLog({ limit: 20, pollInterval: 5000 });

  // Fetch completed tasks from Firefly Network project
  const {
    completedTasks,
    taskSummary,
    isLoading: tasksLoading,
  } = useCompletedTasks({ projectId: "firefly-network", pollInterval: 15000 });

  // Legacy activities hook (for backward compatibility)
  const {
    activities,
    activeCount: legacyActiveCount,
    isLoading: activitiesLoading,
    lastUpdated,
  } = useLabActivities({ pollInterval: 5000 });

  // Calculate active agent count from workers API
  const workersActiveCount = useMemo(() => {
    return workersAgents.filter((a) => a.status === 'working').length;
  }, [workersAgents]);

  // Use workers count if available, otherwise fall back to legacy
  const activeCount = workersActiveCount > 0 ? workersActiveCount : legacyActiveCount;

  // Build agent list from real running agents, fall back to demo agents
  const agents = useMemo<Agent[]>(() => {
    const POSITION_PRESETS: [number, number, number][] = [
      [-3, 0, -2], [3, 0, -2], [-3, 0, 2], [3, 0, 2],
      [-1.5, 0, -1.5], [1.5, 0, -1.5], [-1.5, 0, 1.5], [1.5, 0, 1.5],
    ];

    const getPosition = (i: number): [number, number, number] => {
      if (i < POSITION_PRESETS.length) return POSITION_PRESETS[i];
      const angle = ((i % 8) / 8) * Math.PI * 2;
      const radius = 4;
      return [Math.cos(angle) * radius, 0, Math.sin(angle) * radius];
    };

    // Priority 1: Try Workers API agents (new autonomous agent system)
    if (workersAgents.length > 0 && !isWorkersDemo) {
      return workersAgents.map((a, index) => ({
        id: a.id,
        name: a.name || getFunName(a.id),
        color: getAgentColor(a.type),
        position: getPosition(index),
        task: sanitizeMessage(a.current_task?.title || "Working..."),
        status: mapAgentStatus(a.status),
      }));
    }

    // Priority 2: Try 4090 agent status (legacy system)
    const realAgents = agent4090Status
      .filter(a => a.status === 'working')
      .map((a, index) => ({
        id: a.id || a.name,
        name: getFunName(a.name || a.id),
        color: COLORS.labManager,
        position: getPosition(index),
        task: sanitizeMessage(a.task || a.lastOutput || "Working..."),
        status: "working" as const,
      }));

    // Priority 3: Fall back to demo agents
    return realAgents.length > 0 ? realAgents : DEMO_AGENTS;
  }, [workersAgents, isWorkersDemo, agent4090Status]);

  // Sanitized activity log with demo fallback
  const activityLog = useMemo(() => {
    // Priority 1: Use work log from Workers API if available
    if (workLogEntries.length > 0 && !isWorkLogDemo) {
      const workLogActivities = workLogToActivityLog(workLogEntries, workersAgents);
      return workLogActivities.map((entry) => ({
        ...entry,
        action: sanitizeMessage(entry.action),
      }));
    }

    // Priority 2: Use legacy activities
    const realActivities = activitiesToLog(activities).map((entry) => ({
      ...entry,
      action: sanitizeMessage(entry.action),
    }));

    // Priority 3: Fall back to demo activities
    return realActivities.length > 0 ? realActivities : DEMO_ACTIVITIES;
  }, [workLogEntries, isWorkLogDemo, workersAgents, activities]);

  // Fetch 4090 agent status
  useEffect(() => {
    const fetch4090Status = async () => {
      try {
        const response = await fetch("/api/lab/agent-status");
        const data = await response.json();
        if (data.agents) {
          setAgent4090Status(data.agents);
        }
      } catch (error) {
        console.error("Failed to fetch agent status:", error);
      }
    };

    fetch4090Status();
    const interval = setInterval(fetch4090Status, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch GPU stats (sanitized)
  useEffect(() => {
    const fetchGpuStats = async () => {
      try {
        // Use public endpoint that returns sanitized data
        const response = await fetch("/api/public/gpu-stats");
        const data = await response.json();
        setGpuStats(data);
      } catch (error) {
        setGpuStats({
          connected: false,
          clusterName: "Training Cluster",
          gpu: null,
          hasActiveTraining: false,
        });
      }
    };

    fetchGpuStats();
    const interval = setInterval(fetchGpuStats, 10000);
    return () => clearInterval(interval);
  }, []);

  // Fetch compute network stats
  useEffect(() => {
    const fetchComputeStats = async () => {
      try {
        const response = await fetch("/api/compute/devices");
        const data = await response.json();

        if (data.devices && data.devices.length > 0) {
          setComputeNetworkStats({
            totalDevices: data.count || data.devices.length,
            busyDevices: data.devices.filter((d: { status: string }) => d.status === 'busy').length,
            tierCounts: data.byTier || {
              power: data.devices.filter((d: { tier: string }) => d.tier === 'power').length,
              standard: data.devices.filter((d: { tier: string }) => d.tier === 'standard').length,
              crowd: data.devices.filter((d: { tier: string }) => d.tier === 'crowd').length,
            },
            totalCompute: data.totalCompute || data.devices.reduce(
              (acc: number, d: { capabilities?: { compute?: number } }) =>
                acc + (d.capabilities?.compute || 0), 0
            ),
          });
        } else {
          // No real devices, use demo stats
          setComputeNetworkStats(DEMO_COMPUTE_STATS);
        }
      } catch {
        // API failed, use demo stats
        setComputeNetworkStats(DEMO_COMPUTE_STATS);
      }
    };

    fetchComputeStats();
    const interval = setInterval(fetchComputeStats, 15000);
    return () => clearInterval(interval);
  }, []);

  // Agent click handler (no-op in public view, just for visual feedback)
  const handleAgentClick = useCallback((agent: Agent) => {
    // In public view, we might show a tooltip but no editing
    console.log(`[Public] Viewing agent: ${agent.name}`);
  }, []);

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

  // Mobile tab state for bottom navigation
  const [mobileTab, setMobileTab] = useState<'view' | 'agents' | 'activity' | 'tasks'>('view');

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Compact Status Bar - Mobile optimized */}
      <div className="border-b border-border bg-background-elevated">
        <div className="max-w-[1400px] mx-auto px-3 py-2">
          <div className="flex items-center justify-between">
            {/* Left: Quick stats */}
            <div className="flex items-center gap-3">
              {/* Active agents indicator */}
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${isWorkersDemo ? 'bg-amber-500' : 'bg-green-500'} animate-pulse`} />
                <span className="text-xs text-foreground-bright font-medium">
                  {activeCount} agents
                </span>
              </div>

              {/* Compute network - condensed on mobile */}
              {computeNetworkStats && (
                <div className="hidden xs:flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Zap className="w-3 h-3 text-amber-400" />
                  <span>{computeNetworkStats.totalCompute.toFixed(0)} TF</span>
                </div>
              )}
            </div>

            {/* Right: Task progress */}
            {taskSummary && (
              <div className="flex items-center gap-2">
                <div className="hidden sm:flex items-center gap-1 text-xs text-muted-foreground">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  <span>{taskSummary.completed}/{taskSummary.total}</span>
                </div>
                {/* Mini progress bar - always visible */}
                <div className="w-16 sm:w-24 h-1.5 bg-background rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 transition-all"
                    style={{ width: `${(taskSummary.completed / taskSummary.total) * 100}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content - Tab-based on mobile, side-by-side on desktop */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* 3D View - Full height on mobile when 'view' tab active */}
        <div className={`flex-1 relative ${mobileTab !== 'view' ? 'hidden lg:block' : ''} min-h-0`}>
          <Lab3D
            agents={agents}
            activities={activities}
            onAgentClick={handleAgentClick}
            showDemoProps={true}
          />

          {/* Controls hint - hide on mobile to maximize view */}
          <div className="absolute bottom-16 lg:bottom-4 left-1/2 -translate-x-1/2 z-10 hidden sm:block">
            <div className="px-4 py-2 bg-background/80 backdrop-blur rounded border border-border text-xs text-muted-foreground">
              Drag to rotate | Scroll to zoom | Click agents to view status
            </div>
          </div>
        </div>

        {/* Right Sidebar - Tab content on mobile, always visible on desktop */}
        <aside className={`w-full lg:w-[280px] xl:w-[300px] border-t lg:border-t-0 lg:border-l border-border bg-background-elevated overflow-y-auto ${mobileTab === 'view' ? 'hidden lg:block' : ''} flex-1 lg:flex-none pb-16 lg:pb-0`}>
          {/* GPU Status Card - Show on agents tab on mobile, always on desktop */}
          {gpuStats?.connected && gpuStats.gpu && (
            <div className={`border-b border-border p-4 ${mobileTab !== 'agents' ? 'hidden lg:block' : ''}`}>
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-foreground-bright">
                  {gpuStats.clusterName}
                </span>
                <span className="w-2 h-2 rounded-full bg-foreground animate-pulse" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">GPU</div>
                  <div className="text-lg text-foreground font-medium">{gpuStats.gpu.utilization}%</div>
                </div>
                <div className="p-3 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">VRAM</div>
                  <div className="text-lg text-foreground font-medium">{gpuStats.gpu.memoryPercent}%</div>
                </div>
                <div className="p-3 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">Temp</div>
                  <div className="text-lg text-foreground font-medium">{gpuStats.gpu.temperature}C</div>
                </div>
                <div className="p-3 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">Power</div>
                  <div className="text-lg text-foreground font-medium">{gpuStats.gpu.powerPercent}%</div>
                </div>
              </div>
              {gpuStats.hasActiveTraining && gpuStats.trainingStatus && (
                <div className="mt-3 p-3 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">Training</div>
                  <div className="text-sm text-foreground-bright">
                    {gpuStats.trainingStatus}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Compute Network Stats - Show on agents tab on mobile */}
          <div className={`border-b border-border p-4 ${mobileTab !== 'agents' ? 'hidden lg:block' : ''}`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Monitor className="w-4 h-4 text-green-500" />
                <span className="text-sm text-foreground-bright">
                  Compute Network
                </span>
              </div>
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-background rounded border border-border">
                <div className="text-xs text-muted-foreground">Devices</div>
                <div className="text-lg text-foreground flex items-center gap-1">
                  <span className="text-green-400 font-medium">
                    {computeNetworkStats?.totalDevices || '...'}
                  </span>
                  <span className="text-xs text-muted-foreground">online</span>
                </div>
              </div>
              <div className="p-3 bg-background rounded border border-border">
                <div className="text-xs text-muted-foreground">Active</div>
                <div className="text-lg text-foreground flex items-center gap-1">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span className="text-amber-400 font-medium">
                    {computeNetworkStats?.busyDevices || '...'}
                  </span>
                </div>
              </div>
            </div>
            {computeNetworkStats && (
              <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                <span className="px-2 py-1 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                  {computeNetworkStats.tierCounts.power} power
                </span>
                <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  {computeNetworkStats.tierCounts.standard} standard
                </span>
                <span className="px-2 py-1 rounded bg-green-500/20 text-green-400 border border-green-500/30">
                  {computeNetworkStats.tierCounts.crowd} crowd
                </span>
              </div>
            )}
            {computeNetworkStats && computeNetworkStats.totalCompute > 0 && (
              <div className="mt-3 text-sm text-muted-foreground">
                <span className="text-foreground-bright font-medium">{computeNetworkStats.totalCompute.toFixed(1)}</span> TFLOPS total compute
              </div>
            )}
          </div>

          {/* Agent List - Show on agents tab on mobile */}
          <div className={`p-4 border-b border-border ${mobileTab !== 'agents' ? 'hidden lg:block' : ''}`}>
            <h3 className="text-sm text-foreground-bright mb-3">Agents</h3>
            <div className="flex flex-col gap-2">
              {agents.map((agent) => {
                const Icon = AGENT_ICONS[agent.id] || Brain;
                return (
                  <div
                    key={agent.id}
                    className="flex items-center justify-between p-3 border border-border rounded min-h-[48px] active:bg-foreground/5 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Icon className="w-5 h-5 text-muted-foreground" />
                      <div>
                        <span className="text-sm text-foreground font-medium">{agent.name}</span>
                        {agent.task && (
                          <p className="text-xs text-muted-foreground truncate max-w-[150px]">
                            {agent.task}
                          </p>
                        )}
                      </div>
                    </div>
                    {getStatusBadge(agent.status)}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Activity Log - Show on activity tab on mobile */}
          <div className={`p-4 border-b border-border ${mobileTab !== 'activity' ? 'hidden lg:block' : ''}`}>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm text-foreground-bright">Activity</h3>
              <div className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${isWorkLogDemo ? 'bg-amber-500' : 'bg-green-500'} animate-pulse`} />
                <span className="text-xs text-muted-foreground">
                  {isWorkLogDemo ? 'Demo' : 'Live'}
                </span>
              </div>
            </div>
            <div className="space-y-2 max-h-[400px] lg:max-h-[300px] overflow-y-auto">
              {activityLog.length > 0 ? (
                activityLog.slice(0, 12).map((entry, idx) => (
                  <div
                    key={idx}
                    className="text-xs p-3 border-l-2 border-foreground/30 bg-foreground/5 rounded-r"
                  >
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <span suppressHydrationWarning>{entry.time.toLocaleTimeString()}</span>
                      <span>|</span>
                      <span className="text-foreground font-medium">{entry.agent}</span>
                    </div>
                    <div className="text-foreground mt-1">
                      {entry.action}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-muted-foreground text-center py-8 text-sm">
                  Waiting for activity...
                </div>
              )}
            </div>
            {(workersLastUpdated || lastUpdated) && (
              <div className="text-xs text-foreground-subtle text-center mt-3" suppressHydrationWarning>
                Updated {(workersLastUpdated || lastUpdated)?.toLocaleTimeString()}
              </div>
            )}
          </div>

          {/* Completed Tasks - Show on tasks tab on mobile */}
          <div className={`p-4 ${mobileTab !== 'tasks' ? 'hidden lg:block' : ''}`}>
            <div className="flex items-center justify-between mb-2 sm:mb-3">
              <h3 className="text-xs sm:text-sm text-foreground-bright flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                Completed Tasks
              </h3>
              {taskSummary && (
                <span className="text-[10px] sm:text-xs px-1.5 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30">
                  {taskSummary.completed}/{taskSummary.total}
                </span>
              )}
            </div>

            {/* Task Progress Bar */}
            {taskSummary && taskSummary.total > 0 && (
              <div className="mb-4">
                <div className="h-2 bg-background rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 to-emerald-400 transition-all duration-500"
                    style={{ width: `${(taskSummary.completed / taskSummary.total) * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>{taskSummary.pending} pending</span>
                  <span>{taskSummary.in_progress} active</span>
                </div>
              </div>
            )}

            {/* Recent Completed Tasks List */}
            <div className="space-y-2 max-h-[400px] lg:max-h-[180px] overflow-y-auto">
              {completedTasks.length > 0 ? (
                completedTasks.slice(0, 10).map((task) => (
                  <div
                    key={task.id}
                    className="text-xs p-3 bg-green-500/5 border border-green-500/20 rounded min-h-[56px]"
                  >
                    <div className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="text-foreground font-medium">
                          {task.title}
                        </div>
                        <div className="flex items-center gap-2 text-muted-foreground mt-1">
                          <Clock className="w-3 h-3" />
                          <span suppressHydrationWarning>
                            {new Date(task.completed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                          {task.assigned_agent && (
                            <>
                              <span>•</span>
                              <span className="truncate">{task.assigned_agent}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : tasksLoading ? (
                <div className="text-muted-foreground text-center py-8 text-sm">
                  Loading tasks...
                </div>
              ) : (
                <div className="text-muted-foreground text-center py-8 text-sm">
                  No completed tasks yet
                </div>
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* Mobile Bottom Navigation - Touch-friendly tabs */}
      <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-background-elevated border-t border-border safe-area-pb" role="navigation" aria-label="Main navigation">
        <div className="flex items-stretch">
          <button
            onClick={() => setMobileTab('view')}
            aria-label="View 3D Lab"
            aria-current={mobileTab === 'view' ? 'page' : undefined}
            className={`flex-1 flex flex-col items-center justify-center py-3 min-h-[56px] transition-colors active:bg-foreground/10 ${
              mobileTab === 'view' ? 'text-foreground-bright bg-foreground/5' : 'text-muted-foreground'
            }`}
          >
            <Monitor className="w-5 h-5 mb-0.5" />
            <span className="text-[10px]">Lab</span>
          </button>
          <button
            onClick={() => setMobileTab('agents')}
            aria-label={`View Agents${activeCount > 0 ? `, ${activeCount} active` : ''}`}
            aria-current={mobileTab === 'agents' ? 'page' : undefined}
            className={`relative flex-1 flex flex-col items-center justify-center py-3 min-h-[56px] transition-colors active:bg-foreground/10 ${
              mobileTab === 'agents' ? 'text-foreground-bright bg-foreground/5' : 'text-muted-foreground'
            }`}
          >
            <Brain className="w-5 h-5 mb-0.5" />
            <span className="text-[10px]">Agents</span>
            {activeCount > 0 && (
              <span className="absolute top-1 right-1/4 w-4 h-4 bg-green-500 rounded-full text-[9px] text-white flex items-center justify-center font-medium" aria-hidden="true">
                {activeCount}
              </span>
            )}
          </button>
          <button
            onClick={() => setMobileTab('activity')}
            aria-label="View Activity Feed"
            aria-current={mobileTab === 'activity' ? 'page' : undefined}
            className={`flex-1 flex flex-col items-center justify-center py-3 min-h-[56px] transition-colors active:bg-foreground/10 ${
              mobileTab === 'activity' ? 'text-foreground-bright bg-foreground/5' : 'text-muted-foreground'
            }`}
          >
            <Activity className="w-5 h-5 mb-0.5" />
            <span className="text-[10px]">Activity</span>
          </button>
          <button
            onClick={() => setMobileTab('tasks')}
            aria-label={`View Completed Tasks${taskSummary && taskSummary.completed > 0 ? `, ${taskSummary.completed} completed` : ''}`}
            aria-current={mobileTab === 'tasks' ? 'page' : undefined}
            className={`relative flex-1 flex flex-col items-center justify-center py-3 min-h-[56px] transition-colors active:bg-foreground/10 ${
              mobileTab === 'tasks' ? 'text-foreground-bright bg-foreground/5' : 'text-muted-foreground'
            }`}
          >
            <CheckCircle2 className="w-5 h-5 mb-0.5" />
            <span className="text-[10px]">Tasks</span>
            {taskSummary && taskSummary.completed > 0 && (
              <span className="absolute top-1 right-1/4 w-4 h-4 bg-green-500 rounded-full text-[9px] text-white flex items-center justify-center font-medium" aria-hidden="true">
                {taskSummary.completed}
              </span>
            )}
          </button>
        </div>
      </nav>

      {/* Spacer for mobile bottom nav */}
      <div className="h-14 lg:hidden" />
    </div>
  );
}

export default PublicLabView;
