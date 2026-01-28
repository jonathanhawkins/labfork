"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Lab3D from "@/components/Lab3D";
import { useLabActivities, activitiesToLog } from "@/components/lab/activities";
import {
  Brain,
  Cpu,
  Search,
  ListTodo,
  Activity,
  ThermometerSun,
  HardDrive,
  Server,
} from "lucide-react";

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
};

const AGENT_ICONS: Record<string, typeof Brain> = {
  codex: Cpu,
  opus: Brain,
  explorer: Search,
  planner: ListTodo,
  "lab-manager": Activity,
};

// ============== Public Lab View Component ==============

interface PublicLabViewProps {
  showSuggestions?: boolean;
}

export function PublicLabView({ showSuggestions = false }: PublicLabViewProps) {
  const [agent4090Status, setAgent4090Status] = useState<Agent4090Status[]>([]);
  const [gpuStats, setGpuStats] = useState<SanitizedGpuStats | null>(null);

  const {
    activities,
    activeCount,
    isLoading: activitiesLoading,
    lastUpdated,
  } = useLabActivities({ pollInterval: 5000 });

  // Build agent list from real running agents only
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

    // Only show 4090 agents that are actually working
    return agent4090Status
      .filter(a => a.status === 'working')
      .map((a, index) => ({
        id: a.id || a.name,
        name: a.name,
        color: COLORS.labManager,
        position: getPosition(index),
        task: sanitizeMessage(a.task || a.lastOutput || "Working..."),
        status: "working" as const,
      }));
  }, [agent4090Status]);

  // Sanitized activity log
  const activityLog = useMemo(() => {
    return activitiesToLog(activities).map((entry) => ({
      ...entry,
      action: sanitizeMessage(entry.action),
    }));
  }, [activities]);

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

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <div className="border-b border-border bg-background-elevated">
        <div className="max-w-[1400px] mx-auto px-3 sm:px-4 py-2 sm:py-3">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-base sm:text-lg text-foreground-bright">AI Research Lab</h1>
              <p className="text-[10px] sm:text-xs text-muted-foreground">
                Watch AI agents collaborate in real-time
              </p>
            </div>
              <div className="flex items-center gap-2 sm:gap-3">
                {/* GPU stats - hide on mobile, show condensed on tablet */}
                {gpuStats?.connected && gpuStats.gpu && (
                <div className="hidden sm:flex items-center gap-2 md:gap-4 text-xs text-muted-foreground">
                  <div className="hidden md:flex items-center gap-1">
                    <Server className="w-3 h-3" />
                    <span>{gpuStats.clusterName}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Activity className="w-3 h-3" />
                    <span>{gpuStats.gpu.utilization}%</span>
                  </div>
                  <div className="hidden md:flex items-center gap-1">
                    <ThermometerSun className="w-3 h-3" />
                    <span>{gpuStats.gpu.temperature}C</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <HardDrive className="w-3 h-3" />
                    <span>{gpuStats.gpu.memoryPercent}%</span>
                  </div>
                </div>
                )}
                <div className="flex items-center gap-1.5 sm:gap-2">
                  <span className="text-[10px] sm:text-xs text-muted-foreground">
                    {activeCount} active
                  </span>
                  <span className="w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-foreground animate-pulse" />
                </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - stack on mobile, side-by-side on desktop */}
      <div className="flex-1 flex flex-col lg:flex-row">
        {/* 3D View */}
        <div className="flex-1 relative min-h-[50vh] lg:min-h-0">
          <Lab3D
            agents={agents}
            activities={activities}
            onAgentClick={handleAgentClick}
            showDemoProps={true}
          />

          {/* Controls hint - simpler on mobile */}
          <div className="absolute bottom-2 sm:bottom-4 left-1/2 -translate-x-1/2 z-10">
            <div className="px-2 sm:px-4 py-1.5 sm:py-2 bg-background/80 backdrop-blur rounded border border-border text-[10px] sm:text-xs text-muted-foreground">
              <span className="hidden sm:inline">Drag to rotate | Scroll to zoom | Click agents to view status</span>
              <span className="sm:hidden">Drag to rotate | Pinch to zoom</span>
            </div>
          </div>
        </div>

        {/* Right Sidebar - full width on mobile, fixed width on desktop */}
        <aside className="w-full lg:w-[280px] xl:w-[300px] border-t lg:border-t-0 lg:border-l border-border bg-background-elevated overflow-y-auto max-h-[50vh] lg:max-h-none">
          {/* GPU Status Card */}
          {gpuStats?.connected && gpuStats.gpu && (
            <div className="border-b border-border p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm text-foreground-bright">
                  {gpuStats.clusterName}
                </span>
                <span className="w-2 h-2 rounded-full bg-foreground animate-pulse" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-2 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">GPU</div>
                  <div className="text-sm text-foreground">{gpuStats.gpu.utilization}%</div>
                </div>
                <div className="p-2 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">VRAM</div>
                  <div className="text-sm text-foreground">{gpuStats.gpu.memoryPercent}%</div>
                </div>
                <div className="p-2 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">Temp</div>
                  <div className="text-sm text-foreground">{gpuStats.gpu.temperature}C</div>
                </div>
                <div className="p-2 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">Power</div>
                  <div className="text-sm text-foreground">{gpuStats.gpu.powerPercent}%</div>
                </div>
              </div>
              {gpuStats.hasActiveTraining && gpuStats.trainingStatus && (
                <div className="mt-3 p-2 bg-background rounded border border-border">
                  <div className="text-xs text-muted-foreground">Training</div>
                  <div className="text-sm text-foreground-bright">
                    {gpuStats.trainingStatus}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Agent List - horizontal scroll on mobile, vertical on desktop */}
          <div className="p-3 sm:p-4 border-b border-border">
            <h3 className="text-xs sm:text-sm text-foreground-bright mb-2 sm:mb-3">Agents</h3>
            <div className="flex lg:flex-col gap-2 overflow-x-auto lg:overflow-x-visible pb-2 lg:pb-0">
              {agents.map((agent) => {
                const Icon = AGENT_ICONS[agent.id] || Brain;
                return (
                  <div
                    key={agent.id}
                    className="flex items-center justify-between p-1.5 sm:p-2 border border-border rounded min-w-[140px] lg:min-w-0 flex-shrink-0 lg:flex-shrink"
                  >
                    <div className="flex items-center gap-1.5 sm:gap-2">
                      <Icon className="w-3 h-3 sm:w-4 sm:h-4 text-muted-foreground" />
                      <span className="text-xs sm:text-sm text-foreground">{agent.name}</span>
                    </div>
                    {getStatusBadge(agent.status)}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Activity Log */}
          <div className="p-3 sm:p-4">
            <div className="flex items-center justify-between mb-2 sm:mb-3">
              <h3 className="text-xs sm:text-sm text-foreground-bright">Activity</h3>
              <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-foreground animate-pulse" />
                <span className="text-[10px] sm:text-xs text-muted-foreground">Live</span>
              </div>
            </div>
            <div className="space-y-1.5 sm:space-y-2 max-h-[200px] lg:max-h-[300px] overflow-y-auto">
              {activityLog.length > 0 ? (
                activityLog.slice(0, 8).map((entry, idx) => (
                  <div
                    key={idx}
                    className="text-[10px] sm:text-xs p-1.5 sm:p-2 border-l-2 border-foreground/30 bg-foreground/5"
                  >
                    <div className="flex items-center gap-1.5 sm:gap-2 text-muted-foreground">
                      <span>{entry.time.toLocaleTimeString()}</span>
                      <span className="hidden sm:inline">|</span>
                      <span className="text-foreground">{entry.agent}</span>
                    </div>
                    <div className="text-foreground truncate mt-0.5 sm:mt-1">
                      {entry.action}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-muted-foreground text-center py-3 sm:py-4 text-[10px] sm:text-xs">
                  Waiting for activity...
                </div>
              )}
            </div>
            {lastUpdated && (
              <div className="text-[10px] sm:text-xs text-foreground-subtle text-center mt-2 sm:mt-3">
                Updated {lastUpdated.toLocaleTimeString()}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default PublicLabView;
