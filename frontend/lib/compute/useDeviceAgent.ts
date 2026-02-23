/**
 * React Hook for Device Agent
 *
 * Provides easy integration of the device agent into React components.
 * Handles initialization, lifecycle, and state management.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import {
  DeviceAgent,
  getDeviceAgent,
  type AgentStatus,
  type AgentConfig,
  type DeviceStats,
} from "./device-agent";
import type { ComputeDevice, ComputeTask } from "./types";
import type { LoadProgress } from "./webllm-engine";

/**
 * Hook state
 */
export interface DeviceAgentState {
  /** Current agent status */
  status: AgentStatus;
  /** Registered device (if any) */
  device: ComputeDevice | null;
  /** Current task being executed */
  currentTask: ComputeTask | null;
  /** Current task progress (0-100) */
  taskProgress: number;
  /** Device statistics */
  stats: DeviceStats;
  /** Last error */
  error: Error | null;
  /** Whether the agent is ready */
  isReady: boolean;
  /** Whether the agent is busy */
  isBusy: boolean;
  /** Whether the agent is paused */
  isPaused: boolean;
}

/**
 * Hook actions
 */
export interface DeviceAgentActions {
  /** Start the agent */
  start: () => Promise<void>;
  /** Pause the agent */
  pause: () => void;
  /** Resume the agent */
  resume: () => void;
  /** Stop the agent */
  stop: () => void;
  /** Clear error */
  clearError: () => void;
}

/**
 * Hook return type
 */
export interface UseDeviceAgentReturn extends DeviceAgentState, DeviceAgentActions {
  /** The agent instance (for advanced usage) */
  agent: DeviceAgent;
  /** Whether the WebLLM model is loaded */
  isModelLoaded: boolean;
  /** Load the WebLLM AI model for real compute */
  loadModel: () => Promise<void>;
  /** Model load progress (0-100) */
  modelLoadProgress: number;
}

/**
 * Hook options
 */
export interface UseDeviceAgentOptions extends AgentConfig {
  /** Whether to auto-start on mount (default: true) */
  autoStart?: boolean;
  /** Whether to auto-stop on unmount (default: true) */
  autoStop?: boolean;
}

/**
 * React hook for device agent
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const {
 *     status,
 *     device,
 *     stats,
 *     currentTask,
 *     taskProgress,
 *     isReady,
 *     isBusy,
 *     isPaused,
 *     start,
 *     pause,
 *     resume,
 *     stop,
 *   } = useDeviceAgent({
 *     deviceName: "My Device",
 *     autoStart: true,
 *   });
 *
 *   return (
 *     <div>
 *       <p>Status: {status}</p>
 *       {device && <p>Tier: {device.tier}</p>}
 *       {currentTask && <p>Task: {currentTask.id}</p>}
 *       {isBusy && <p>Progress: {taskProgress}%</p>}
 *       <button onClick={isPaused ? resume : pause}>
 *         {isPaused ? "Resume" : "Pause"}
 *       </button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useDeviceAgent(options: UseDeviceAgentOptions = {}): UseDeviceAgentReturn {
  const {
    autoStart = true,
    autoStop = true,
    ...agentConfig
  } = options;

  // Get or create agent instance
  const agentRef = useRef<DeviceAgent>(getDeviceAgent(agentConfig));
  const agent = agentRef.current;

  // State
  const [status, setStatus] = useState<AgentStatus>(agent.getStatus());
  const [device, setDevice] = useState<ComputeDevice | null>(agent.getDevice());
  const [currentTask, setCurrentTask] = useState<ComputeTask | null>(agent.getCurrentTask());
  const [taskProgress, setTaskProgress] = useState<number>(0);
  const [stats, setStats] = useState<DeviceStats>(agent.getStats());
  const [error, setError] = useState<Error | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(agent.isModelLoaded());
  const [modelLoadProgress, setModelLoadProgress] = useState<number>(0);

  // Derived state
  const isReady = status === "online";
  const isBusy = currentTask !== null;
  const isPaused = status === "paused";

  // Actions
  const start = useCallback(async () => {
    try {
      await agent.start();
    } catch (err) {
      console.error("Failed to start agent:", err);
    }
  }, [agent]);

  const pause = useCallback(() => {
    agent.pause();
  }, [agent]);

  const resume = useCallback(() => {
    agent.resume();
  }, [agent]);

  const stop = useCallback(() => {
    agent.stop();
  }, [agent]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const loadModel = useCallback(async () => {
    try {
      await agent.loadModel();
      setIsModelLoaded(true);
    } catch (err) {
      console.error("Failed to load model:", err);
      setError(err instanceof Error ? err : new Error(String(err)));
    }
  }, [agent]);

  // Set up event listeners
  useEffect(() => {
    const handleStatusChange = (newStatus: AgentStatus) => {
      setStatus(newStatus);
    };

    const handleDeviceRegistered = (newDevice: ComputeDevice) => {
      setDevice(newDevice);
    };

    const handleTaskReceived = (task: ComputeTask) => {
      setCurrentTask(task);
      setTaskProgress(0);
    };

    const handleTaskStarted = (taskId: string) => {
      // Task already set by handleTaskReceived
      setTaskProgress(0);
    };

    const handleTaskCompleted = (taskId: string, success: boolean) => {
      setCurrentTask(null);
      setTaskProgress(0);
    };

    const handleTaskProgress = (taskId: string, progress: number) => {
      setTaskProgress(progress);
    };

    const handleError = (err: Error) => {
      setError(err);
      console.error("Agent error:", err);
    };

    const handleStatsUpdated = (newStats: DeviceStats) => {
      setStats(newStats);
    };

    const handleModelLoadProgress = (progress: LoadProgress) => {
      // LoadProgress.progress is 0-100
      setModelLoadProgress(Math.round(progress.progress));
    };

    const handleModelLoaded = () => {
      setIsModelLoaded(true);
      setModelLoadProgress(100);
    };

    // Register listeners
    agent.on("statusChange", handleStatusChange);
    agent.on("deviceRegistered", handleDeviceRegistered);
    agent.on("taskReceived", handleTaskReceived);
    agent.on("taskStarted", handleTaskStarted);
    agent.on("taskCompleted", handleTaskCompleted);
    agent.on("taskProgress", handleTaskProgress);
    agent.on("error", handleError);
    agent.on("statsUpdated", handleStatsUpdated);
    agent.on("modelLoadProgress", handleModelLoadProgress);
    agent.on("modelLoaded", handleModelLoaded);

    // Cleanup
    return () => {
      agent.off("statusChange", handleStatusChange);
      agent.off("deviceRegistered", handleDeviceRegistered);
      agent.off("taskReceived", handleTaskReceived);
      agent.off("taskStarted", handleTaskStarted);
      agent.off("taskCompleted", handleTaskCompleted);
      agent.off("taskProgress", handleTaskProgress);
      agent.off("error", handleError);
      agent.off("statsUpdated", handleStatsUpdated);
      agent.off("modelLoadProgress", handleModelLoadProgress);
      agent.off("modelLoaded", handleModelLoaded);
    };
  }, [agent]);

  // Auto-start on mount
  useEffect(() => {
    if (autoStart && status === "initializing") {
      start();
    }
  }, [autoStart, status, start]);

  // Auto-stop on unmount
  useEffect(() => {
    return () => {
      if (autoStop) {
        agent.stop();
      }
    };
  }, [agent, autoStop]);

  // Periodic stats update (every second when online)
  useEffect(() => {
    if (status === "online" || status === "paused") {
      const interval = setInterval(() => {
        setStats(agent.getStats());
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [agent, status]);

  return {
    // State
    status,
    device,
    currentTask,
    taskProgress,
    stats,
    error,
    isReady,
    isBusy,
    isPaused,
    isModelLoaded,
    modelLoadProgress,

    // Actions
    start,
    pause,
    resume,
    stop,
    clearError,
    loadModel,

    // Agent instance
    agent,
  };
}

/**
 * Hook for device agent stats only
 * Useful for displaying stats without managing the full agent lifecycle
 */
export function useDeviceAgentStats(): DeviceStats {
  const agent = getDeviceAgent();
  const [stats, setStats] = useState<DeviceStats>(agent.getStats());

  useEffect(() => {
    const handleStatsUpdated = (newStats: DeviceStats) => {
      setStats(newStats);
    };

    agent.on("statsUpdated", handleStatsUpdated);

    // Update stats every second
    const interval = setInterval(() => {
      setStats(agent.getStats());
    }, 1000);

    return () => {
      agent.off("statsUpdated", handleStatsUpdated);
      clearInterval(interval);
    };
  }, [agent]);

  return stats;
}

/**
 * Hook for device info only
 * Useful for displaying device tier without managing the agent lifecycle
 */
export function useDeviceInfo(): ComputeDevice | null {
  const agent = getDeviceAgent();
  const [device, setDevice] = useState<ComputeDevice | null>(agent.getDevice());

  useEffect(() => {
    const handleDeviceRegistered = (newDevice: ComputeDevice) => {
      setDevice(newDevice);
    };

    agent.on("deviceRegistered", handleDeviceRegistered);

    return () => {
      agent.off("deviceRegistered", handleDeviceRegistered);
    };
  }, [agent]);

  return device;
}

/**
 * Hook for current task only
 * Useful for displaying task progress without managing the agent lifecycle
 */
export function useCurrentTask(): {
  task: ComputeTask | null;
  progress: number;
} {
  const agent = getDeviceAgent();
  const [task, setTask] = useState<ComputeTask | null>(agent.getCurrentTask());
  const [progress, setProgress] = useState<number>(0);

  useEffect(() => {
    const handleTaskReceived = (newTask: ComputeTask) => {
      setTask(newTask);
      setProgress(0);
    };

    const handleTaskCompleted = () => {
      setTask(null);
      setProgress(0);
    };

    const handleTaskProgress = (taskId: string, newProgress: number) => {
      setProgress(newProgress);
    };

    agent.on("taskReceived", handleTaskReceived);
    agent.on("taskCompleted", handleTaskCompleted);
    agent.on("taskProgress", handleTaskProgress);

    return () => {
      agent.off("taskReceived", handleTaskReceived);
      agent.off("taskCompleted", handleTaskCompleted);
      agent.off("taskProgress", handleTaskProgress);
    };
  }, [agent]);

  return { task, progress };
}

/**
 * Format uptime for display
 */
export function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}

/**
 * Format compute time for display
 */
export function formatComputeTime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }
}

/**
 * Format credits with proper decimals
 */
export function formatCredits(credits: number): string {
  if (credits >= 1000) {
    return `${(credits / 1000).toFixed(1)}K`;
  }
  return credits.toFixed(1);
}

/**
 * Get status color for UI
 */
export function getStatusColor(status: AgentStatus): string {
  switch (status) {
    case "online":
      return "text-green-400";
    case "paused":
      return "text-yellow-400";
    case "connecting":
    case "initializing":
      return "text-blue-400";
    case "offline":
      return "text-gray-400";
    case "error":
      return "text-red-400";
    default:
      return "text-gray-400";
  }
}

/**
 * Get status label for UI
 */
export function getStatusLabel(status: AgentStatus): string {
  switch (status) {
    case "initializing":
      return "Initializing...";
    case "connecting":
      return "Connecting...";
    case "online":
      return "Online";
    case "paused":
      return "Paused";
    case "offline":
      return "Offline";
    case "error":
      return "Error";
    default:
      return "Unknown";
  }
}
