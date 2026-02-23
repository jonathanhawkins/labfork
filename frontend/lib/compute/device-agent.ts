/**
 * WebGPU Device Agent
 *
 * Client-side agent that connects devices to the LabFork distributed compute network.
 * Implements singleton pattern to ensure one agent per browser tab.
 */

import type {
  ComputeDevice,
  ComputeTask,
  DeviceStatus,
  RegisterDeviceRequest,
  TaskResult,
  DeviceCapabilities,
  DeviceAvailability,
} from "./types";
import { DEFAULT_AVAILABILITY } from "./types";
import { detectGPU, runBenchmark } from "./gpu-detect";
import { getWebLLMEngine, WebLLMEngine, ModelId, MODEL_INFO, type LoadProgress } from "./webllm-engine";
import { getDraftGenerator } from "./draft-generator";
import { getDraftVerifier } from "./draft-verifier";
import type { DraftSequence } from "./speculative-decoding";

/**
 * Agent status
 */
export type AgentStatus = "initializing" | "connecting" | "online" | "paused" | "offline" | "error";

/**
 * Agent events
 */
export interface AgentEvents {
  statusChange: (status: AgentStatus) => void;
  deviceRegistered: (device: ComputeDevice) => void;
  taskReceived: (task: ComputeTask) => void;
  taskStarted: (taskId: string) => void;
  taskCompleted: (taskId: string, success: boolean) => void;
  taskProgress: (taskId: string, progress: number) => void;
  error: (error: Error) => void;
  statsUpdated: (stats: DeviceStats) => void;
  modelLoadProgress: (progress: LoadProgress) => void;
  modelLoaded: (modelId: ModelId) => void;
}

/**
 * Device statistics
 */
export interface DeviceStats {
  tasksCompleted: number;
  creditsEarned: number;
  totalComputeTime: number;
  uptimeSeconds: number;
  lastTaskCompletedAt?: Date;
}

/**
 * Agent configuration
 */
export interface AgentConfig {
  /** Device display name */
  deviceName?: string;
  /** Availability preferences */
  availability?: Partial<DeviceAvailability>;
  /** Heartbeat interval in ms (default: 5000) */
  heartbeatInterval?: number;
  /** Auto-start on initialization */
  autoStart?: boolean;
  /** API base URL */
  apiBaseUrl?: string;
  /** Model to load (auto-selected based on tier if not specified) */
  modelId?: ModelId;
  /** Use mock inference instead of WebLLM (for testing) */
  useMockInference?: boolean;
}

/**
 * Task execution context
 */
interface TaskExecutionContext {
  task: ComputeTask;
  startTime: number;
  abortController: AbortController;
}

/**
 * WebGPU Device Agent
 * Singleton class that manages device registration, heartbeat, and task execution
 */
// Storage keys for auth persistence
const AUTH_TOKEN_KEY = "labfork_device_auth_token";
const DEVICE_ID_KEY = "labfork_device_id";

export class DeviceAgent {
  private static instance: DeviceAgent | null = null;

  private status: AgentStatus = "initializing";
  private device: ComputeDevice | null = null;
  private currentTask: TaskExecutionContext | null = null;
  private listeners: Map<keyof AgentEvents, Set<Function>> = new Map();
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private stats: DeviceStats = {
    tasksCompleted: 0,
    creditsEarned: 0,
    totalComputeTime: 0,
    uptimeSeconds: 0,
  };
  private startTime: number = 0;
  private uptimeTimer: NodeJS.Timeout | null = null;
  private reconnectAttempts: number = 0;
  private readonly maxReconnectAttempts: number = 5;
  private readonly reconnectBackoff: number = 1000; // Start with 1s

  // Authentication
  private authToken: string | null = null;

  // WebLLM engine
  private webllmEngine: WebLLMEngine | null = null;
  private loadedModelId: ModelId | null = null;

  // Configuration
  private readonly config: Required<AgentConfig>;

  // Task polling
  private pollTimer: NodeJS.Timeout | null = null;
  private readonly pollInterval = 3000; // Poll every 3 seconds when idle

  private constructor(config: AgentConfig = {}) {
    this.config = {
      deviceName: config.deviceName || this.generateDeviceName(),
      availability: { ...DEFAULT_AVAILABILITY, ...config.availability },
      heartbeatInterval: config.heartbeatInterval || 5000,
      autoStart: config.autoStart ?? true,
      apiBaseUrl: config.apiBaseUrl || (
        // Use Workers API for distributed compute network
        process.env.NEXT_PUBLIC_WORKERS_API_URL ||
        "https://labfork-agents.jonathan-hawkins.workers.dev/api"
      ) + "/compute",
      modelId: config.modelId,
      useMockInference: config.useMockInference ?? false,
    };

    // Initialize listeners map
    const eventKeys: (keyof AgentEvents)[] = [
      "statusChange",
      "deviceRegistered",
      "taskReceived",
      "taskStarted",
      "taskCompleted",
      "taskProgress",
      "error",
      "statsUpdated",
      "modelLoadProgress",
      "modelLoaded",
    ];
    eventKeys.forEach((key) => this.listeners.set(key, new Set()));

    // Load persisted auth token
    this.loadAuthToken();
  }

  /**
   * Load auth token from localStorage
   */
  private loadAuthToken(): void {
    if (typeof window === "undefined") return;
    try {
      this.authToken = localStorage.getItem(AUTH_TOKEN_KEY);
      const deviceId = localStorage.getItem(DEVICE_ID_KEY);
      if (this.authToken && deviceId) {
        console.log(`[DeviceAgent] Loaded existing auth for device: ${deviceId.slice(0, 12)}...`);
      }
    } catch (e) {
      console.warn("[DeviceAgent] Failed to load auth token:", e);
    }
  }

  /**
   * Save auth token to localStorage
   */
  private saveAuthToken(token: string, deviceId: string): void {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(AUTH_TOKEN_KEY, token);
      localStorage.setItem(DEVICE_ID_KEY, deviceId);
      this.authToken = token;
      console.log(`[DeviceAgent] Saved auth token for device: ${deviceId.slice(0, 12)}...`);
    } catch (e) {
      console.warn("[DeviceAgent] Failed to save auth token:", e);
    }
  }

  /**
   * Clear auth token from localStorage
   */
  private clearAuthToken(): void {
    if (typeof window === "undefined") return;
    try {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(DEVICE_ID_KEY);
      this.authToken = null;
    } catch (e) {
      console.warn("[DeviceAgent] Failed to clear auth token:", e);
    }
  }

  /**
   * Make an authenticated fetch request
   */
  private async authFetch(url: string, options: RequestInit = {}): Promise<Response> {
    const headers = new Headers(options.headers);
    headers.set("Content-Type", "application/json");

    if (this.authToken) {
      headers.set("Authorization", `Bearer ${this.authToken}`);
    }

    return fetch(url, {
      ...options,
      headers,
    });
  }

  /**
   * Get singleton instance
   */
  public static getInstance(config?: AgentConfig): DeviceAgent {
    if (!DeviceAgent.instance) {
      DeviceAgent.instance = new DeviceAgent(config);
    }
    return DeviceAgent.instance;
  }

  /**
   * Reset singleton instance (for testing)
   */
  public static resetInstance(): void {
    if (DeviceAgent.instance) {
      DeviceAgent.instance.destroy();
      DeviceAgent.instance = null;
    }
  }

  /**
   * Initialize and start the agent
   */
  public async start(): Promise<void> {
    if (this.status === "online" || this.status === "connecting") {
      console.warn("Agent already started");
      return;
    }

    try {
      this.updateStatus("connecting");
      this.startTime = Date.now();
      this.startUptimeTracking();

      // Detect GPU capabilities
      const gpuInfo = await detectGPU();
      if (!gpuInfo.available) {
        throw new Error("WebGPU not available on this device");
      }

      // Run benchmark
      console.log("Running GPU benchmark...");
      const benchmark = await runBenchmark();
      console.log(`Benchmark complete: ${benchmark.tflops.toFixed(3)} TFLOPS`);

      // Build device capabilities
      const capabilities: DeviceCapabilities = {
        compute: benchmark.tflops,
        memory: (gpuInfo.estimatedMemoryMB || 2048) / 1024, // Convert to GB
        bandwidth: await this.estimateBandwidth(),
        platform: "webgpu",
        gpuName: gpuInfo.adapterInfo?.description || "Unknown GPU",
        cachedModels: [],
      };

      // Register device (include platform for proper tier classification)
      const registerRequest = {
        name: this.config.deviceName,
        platform: "webgpu",
        capabilities: {
          compute: capabilities.compute,
          memory: capabilities.memory,
          models: capabilities.cachedModels || [],
        },
        availability: this.config.availability,
      };

      const response = await fetch(`${this.config.apiBaseUrl}/devices`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(registerRequest),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Failed to register device");
      }

      const { device, authToken } = await response.json();
      this.device = device;
      this.reconnectAttempts = 0;

      // Save auth token for subsequent requests
      if (authToken) {
        this.saveAuthToken(authToken, device.id);
      }

      this.emit("deviceRegistered", device);
      console.log(`Device registered: ${device.id} (${device.tier} tier)`);

      // Start heartbeat
      this.startHeartbeat();

      // Start polling for tasks
      this.startTaskPolling();

      this.updateStatus("online");
    } catch (error) {
      console.error("Agent start failed:", error);
      this.updateStatus("error");
      this.emit("error", error instanceof Error ? error : new Error(String(error)));

      // Attempt reconnection
      this.scheduleReconnect();
    }
  }

  /**
   * Pause the agent (stop accepting new tasks)
   */
  public pause(): void {
    if (this.status !== "online") {
      console.warn("Agent not online, cannot pause");
      return;
    }

    this.updateStatus("paused");
    console.log("Agent paused");

    // Abort current task if any
    if (this.currentTask) {
      this.currentTask.abortController.abort();
      this.currentTask = null;
    }
  }

  /**
   * Resume the agent
   */
  public resume(): void {
    if (this.status !== "paused") {
      console.warn("Agent not paused, cannot resume");
      return;
    }

    this.updateStatus("online");
    console.log("Agent resumed");
  }

  /**
   * Stop the agent completely
   */
  public stop(): void {
    console.log("Stopping agent...");

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.uptimeTimer) {
      clearInterval(this.uptimeTimer);
      this.uptimeTimer = null;
    }

    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }

    if (this.currentTask) {
      this.currentTask.abortController.abort();
      this.currentTask = null;
    }

    this.updateStatus("offline");
  }

  /**
   * Destroy the agent and clean up resources
   */
  public destroy(): void {
    this.stop();
    this.listeners.clear();
    this.device = null;
    this.stats = {
      tasksCompleted: 0,
      creditsEarned: 0,
      totalComputeTime: 0,
      uptimeSeconds: 0,
    };
  }

  /**
   * Get current status
   */
  public getStatus(): AgentStatus {
    return this.status;
  }

  /**
   * Get registered device
   */
  public getDevice(): ComputeDevice | null {
    return this.device;
  }

  /**
   * Get current task
   */
  public getCurrentTask(): ComputeTask | null {
    return this.currentTask?.task || null;
  }

  /**
   * Get device statistics
   */
  public getStats(): DeviceStats {
    return {
      ...this.stats,
      uptimeSeconds: Math.floor((Date.now() - this.startTime) / 1000),
    };
  }

  /**
   * Add event listener
   */
  public on<K extends keyof AgentEvents>(event: K, listener: AgentEvents[K]): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.add(listener);
    }
  }

  /**
   * Remove event listener
   */
  public off<K extends keyof AgentEvents>(event: K, listener: AgentEvents[K]): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.delete(listener);
    }
  }

  /**
   * Emit event to all listeners
   */
  private emit<K extends keyof AgentEvents>(
    event: K,
    ...args: Parameters<AgentEvents[K]>
  ): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.forEach((listener) => {
        try {
          (listener as any)(...args);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  /**
   * Update agent status and emit event
   */
  private updateStatus(status: AgentStatus): void {
    if (this.status !== status) {
      this.status = status;
      this.emit("statusChange", status);
    }
  }

  /**
   * Start heartbeat loop
   */
  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(() => {
      this.sendHeartbeat();
    }, this.config.heartbeatInterval);

    // Send initial heartbeat immediately
    this.sendHeartbeat();
  }

  /**
   * Start task polling loop
   */
  private startTaskPolling(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
    }

    this.pollTimer = setInterval(() => {
      this.pollForTask();
    }, this.pollInterval);

    // Poll immediately
    this.pollForTask();
  }

  /**
   * Poll orchestrator for available tasks
   */
  private async pollForTask(): Promise<void> {
    if (!this.device) {
      return;
    }

    // Don't poll if we're not online or already have a task
    if (this.status !== "online" || this.currentTask) {
      return;
    }

    try {
      const response = await this.authFetch(`${this.config.apiBaseUrl}/tasks/assign`, {
        method: "POST",
        body: JSON.stringify({
          deviceId: this.device.id,
          capabilities: {
            compute: this.device.capabilities.compute,
            memory: this.device.capabilities.memory,
            cachedModels: this.device.capabilities.cachedModels,
          },
        }),
      });

      if (!response.ok) {
        console.warn("Task poll failed:", response.statusText);
        return;
      }

      const data = await response.json();

      if (data.hasWork && data.task) {
        console.log("Received task from polling:", data.task.id);
        this.handleTaskAssignment(data.task);
      }
    } catch (error) {
      // Don't log every poll failure, just silently retry
      console.debug("Task poll error:", error);
    }
  }

  /**
   * Send heartbeat to orchestrator
   */
  private async sendHeartbeat(): Promise<void> {
    if (!this.device) {
      console.warn("Cannot send heartbeat: device not registered");
      return;
    }

    if (this.status === "offline" || this.status === "error") {
      return;
    }

    try {
      const currentStatus: DeviceStatus = this.status === "paused" ? "paused" :
                                          this.currentTask ? "busy" : "online";

      const response = await this.authFetch(`${this.config.apiBaseUrl}/devices/${this.device.id}`, {
        method: "PATCH",
        body: JSON.stringify({
          status: currentStatus,
          taskProgress: this.currentTask ? 50 : undefined, // Mock progress for now
        }),
      });

      if (!response.ok) {
        // Try to get error message from response body
        let errorMessage = `Heartbeat failed (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.error) {
            errorMessage = `Heartbeat failed: ${errorData.error}`;
          }
        } catch {
          // If we can't parse the body, use status text as fallback
          if (response.statusText) {
            errorMessage = `Heartbeat failed: ${response.statusText}`;
          }
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();

      // Check for task assignment
      if (data.task && !this.currentTask && this.status === "online") {
        this.handleTaskAssignment(data.task);
      }

      this.reconnectAttempts = 0; // Reset on successful heartbeat
    } catch (error) {
      console.error("Heartbeat error:", error);
      this.emit("error", error instanceof Error ? error : new Error(String(error)));

      // Schedule reconnect if we've lost connection
      this.scheduleReconnect();
    }
  }

  /**
   * Handle task assignment from orchestrator
   */
  private async handleTaskAssignment(task: ComputeTask): Promise<void> {
    if (this.currentTask) {
      console.warn("Already processing a task, ignoring new assignment");
      return;
    }

    console.log(`Task assigned: ${task.id} (${task.type})`);
    this.emit("taskReceived", task);

    const abortController = new AbortController();
    this.currentTask = {
      task,
      startTime: Date.now(),
      abortController,
    };

    this.emit("taskStarted", task.id);

    try {
      // Execute task
      const result = await this.executeTask(task, abortController.signal);

      // Report completion and get actual credits awarded by server
      const creditsAwarded = await this.completeTask(task.id, true, result);

      // Update stats
      const computeTime = Date.now() - this.currentTask.startTime;
      this.stats.tasksCompleted++;
      this.stats.totalComputeTime += computeTime / 1000;
      this.stats.creditsEarned += creditsAwarded || task.reward || 1;
      this.stats.lastTaskCompletedAt = new Date();
      this.emit("statsUpdated", this.getStats());

      this.emit("taskCompleted", task.id, true);
      console.log(`Task completed: ${task.id}`);
    } catch (error) {
      console.error("Task execution error:", error);

      // Report failure (if not aborted)
      if (!abortController.signal.aborted) {
        await this.completeTask(task.id, false, undefined, String(error));
        this.emit("taskCompleted", task.id, false);
      }
    } finally {
      this.currentTask = null;
    }
  }

  /**
   * Execute a compute task using WebLLM
   */
  private async executeTask(task: ComputeTask, signal: AbortSignal): Promise<TaskResult> {
    // Handle speculative decoding tasks
    if (task.type === "draft_generation") {
      return this.executeDraftGeneration(task, signal);
    }

    if (task.type === "draft_verification") {
      return this.executeDraftVerification(task, signal);
    }

    // Check if using mock inference
    if (this.config.useMockInference || !this.webllmEngine?.isModelLoaded()) {
      return this.executeMockTask(task, signal);
    }

    // Use WebLLM for real inference
    const startTime = Date.now();

    // Start progress tracking
    const progressInterval = setInterval(() => {
      if (!signal.aborted && this.currentTask) {
        const elapsed = Date.now() - startTime;
        // Estimate progress based on typical inference time
        const estimatedTime = 5000; // 5 seconds typical
        const progress = Math.min(90, (elapsed / estimatedTime) * 100);
        this.emit("taskProgress", task.id, progress);
      }
    }, 200);

    try {
      if (signal.aborted) {
        throw new Error("Task aborted");
      }

      // Execute task with WebLLM
      const result = await this.webllmEngine.executeTask(task);
      result.computeMode = 'webllm';

      clearInterval(progressInterval);
      this.emit("taskProgress", task.id, 100);

      return result;
    } catch (error) {
      clearInterval(progressInterval);
      throw error;
    }
  }

  /**
   * Execute draft generation task (for crowd/standard tier devices)
   */
  private async executeDraftGeneration(
    task: ComputeTask,
    signal: AbortSignal
  ): Promise<TaskResult> {
    const startTime = Date.now();

    if (signal.aborted) {
      throw new Error("Task aborted");
    }

    const draftGenerator = getDraftGenerator();

    // Get config
    const draftCount = task.config.speculativeDecoding?.draftCount || 8;
    const draftModelId = (task.config.speculativeDecoding?.draftModelId || "Qwen2-0.5B") as ModelId;
    const temperature = task.config.temperature || 0.8;

    // Initialize if needed
    if (!draftGenerator.isReady() || draftGenerator.getCurrentModel() !== draftModelId) {
      await draftGenerator.initialize(draftModelId);
    }

    // Generate draft
    const result = await draftGenerator.generateDraft({
      context: task.input.prompt || "",
      draftCount,
      modelId: draftModelId,
      temperature,
      deviceId: this.device?.id || "unknown",
    });

    const computeTime = Date.now() - startTime;

    return {
      text: result.draft.tokens.map((t) => t.text).join(""),
      // Store draft in a special field for verification
      embedding: JSON.stringify(result.draft) as any, // Hack: use embedding field
      computeMode: 'webllm',
      metrics: {
        computeTime,
        tokensPerSecond: result.tokensPerSecond,
      },
    };
  }

  /**
   * Execute draft verification task (for power tier devices)
   */
  private async executeDraftVerification(
    task: ComputeTask,
    signal: AbortSignal
  ): Promise<TaskResult> {
    const startTime = Date.now();

    if (signal.aborted) {
      throw new Error("Task aborted");
    }

    const draftVerifier = getDraftVerifier();

    // Get draft from task input (hack: stored in embedding field)
    const draft: DraftSequence = JSON.parse(task.input.embedding as any);

    // Get config
    const verifyModelId = (task.config.speculativeDecoding?.verifyModelId || "Phi-3-mini") as ModelId;
    const acceptanceThreshold = task.config.speculativeDecoding?.acceptanceThreshold || 0.8;
    const temperature = task.config.temperature || 0.8;

    // Initialize if needed
    if (!draftVerifier.isReady() || draftVerifier.getCurrentModel() !== verifyModelId) {
      await draftVerifier.initialize(verifyModelId);
    }

    // Verify draft
    const result = await draftVerifier.verifyDraft({
      draft,
      modelId: verifyModelId,
      acceptanceThreshold,
      deviceId: this.device?.id || "unknown",
      temperature,
    });

    const computeTime = Date.now() - startTime;

    return {
      text: result.finalText,
      // Store verification result
      embedding: JSON.stringify(result) as any,
      computeMode: 'webllm',
      metrics: {
        computeTime,
        tokensPerSecond: result.tokensPerSecond,
      },
    };
  }

  /**
   * Execute a mock task (for testing or when WebLLM unavailable)
   */
  private async executeMockTask(task: ComputeTask, signal: AbortSignal): Promise<TaskResult> {
    const computeTimeMs = Math.random() * 2000 + 1000; // 1-3 seconds

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      if (!signal.aborted && this.currentTask) {
        const elapsed = Date.now() - this.currentTask.startTime;
        const progress = Math.min(95, (elapsed / computeTimeMs) * 100);
        this.emit("taskProgress", task.id, progress);
      }
    }, 200);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        clearInterval(progressInterval);

        if (signal.aborted) {
          reject(new Error("Task aborted"));
          return;
        }

        // Mock result based on task type
        const result: TaskResult = {
          computeMode: 'mock',
          metrics: {
            computeTime: computeTimeMs,
          },
        };

        if (task.type === "full_inference" || task.type === "shard_inference") {
          result.tokens = Array(task.config.maxTokens)
            .fill(0)
            .map(() => Math.floor(Math.random() * 50000));
          result.text = "Mock inference result";
          result.metrics.tokensPerSecond = (task.config.maxTokens / computeTimeMs) * 1000;
        } else if (task.type === "embedding") {
          // Return undefined instead of fake random vectors
          // Server validates dimensions, and fake data pollutes indexes
          result.embedding = undefined;
        }

        resolve(result);
      }, computeTimeMs);

      // Handle abort — use { once: true } to prevent memory leak
      signal.addEventListener("abort", () => {
        clearTimeout(timeout);
        clearInterval(progressInterval);
        reject(new Error("Task aborted"));
      }, { once: true });
    });
  }

  /**
   * Load WebLLM model
   */
  public async loadModel(modelId?: ModelId): Promise<void> {
    if (!this.webllmEngine) {
      this.webllmEngine = getWebLLMEngine();
    }

    // Determine model to load
    const targetModel = modelId || this.config.modelId ||
      WebLLMEngine.getRecommendedModel(this.device?.tier || "crowd");

    console.log(`Loading model: ${targetModel}`);

    // Set up progress listener
    const progressHandler = (progress: LoadProgress) => {
      this.emit("modelLoadProgress", progress);
    };

    this.webllmEngine.on("loadProgress", progressHandler);

    try {
      await this.webllmEngine.loadModel(targetModel);
      this.loadedModelId = targetModel;
      this.emit("modelLoaded", targetModel);
    } finally {
      this.webllmEngine.off("loadProgress", progressHandler);
    }
  }

  /**
   * Get current loaded model
   */
  public getLoadedModel(): ModelId | null {
    return this.loadedModelId;
  }

  /**
   * Check if model is loaded
   */
  public isModelLoaded(): boolean {
    return this.webllmEngine?.isModelLoaded() || false;
  }

  /**
   * Complete a task and report to orchestrator
   */
  private async completeTask(
    taskId: string,
    success: boolean,
    result?: TaskResult,
    error?: string
  ): Promise<number> {
    if (!this.device) {
      console.warn("Cannot complete task: device not registered");
      return 0;
    }

    try {
      const response = await this.authFetch(`${this.config.apiBaseUrl}/tasks/${taskId}`, {
        method: "POST",
        body: JSON.stringify({
          deviceId: this.device.id,
          success,
          result,
          error,
        }),
      });

      if (!response.ok) {
        // Try to get error message from response body
        let errorMessage = `Failed to complete task (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.error) {
            errorMessage = `Failed to complete task: ${errorData.error}`;
          }
        } catch {
          if (response.statusText) {
            errorMessage = `Failed to complete task: ${response.statusText}`;
          }
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      const credits = data.creditsAwarded || 0;
      if (credits) {
        console.log(`Earned ${credits} credits`);
      }
      return credits;
    } catch (error) {
      console.error("Error reporting task completion:", error);
      this.emit("error", error instanceof Error ? error : new Error(String(error)));
      return 0;
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error("Max reconnection attempts reached");
      this.updateStatus("error");
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectBackoff * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

    setTimeout(async () => {
      if (this.status === "error" || this.status === "offline") {
        console.log("Attempting to reconnect...");

        // If we already have a device + auth token, try a heartbeat first
        // to re-establish the connection without creating a duplicate device
        if (this.device && this.authToken) {
          try {
            const response = await this.authFetch(
              `${this.config.apiBaseUrl}/devices/${this.device.id}`,
              {
                method: "PATCH",
                body: JSON.stringify({ status: "online" }),
              }
            );
            if (response.ok) {
              console.log("Reconnected via heartbeat");
              this.updateStatus("online");
              this.startHeartbeat();
              this.startTaskPolling();
              this.reconnectAttempts = 0;
              return;
            }
          } catch {
            // Heartbeat failed, fall through to full re-registration
          }
        }

        this.start();
      }
    }, delay);
  }

  /**
   * Start tracking uptime
   */
  private startUptimeTracking(): void {
    if (this.uptimeTimer) {
      clearInterval(this.uptimeTimer);
    }

    this.uptimeTimer = setInterval(() => {
      this.stats.uptimeSeconds = Math.floor((Date.now() - this.startTime) / 1000);
    }, 1000);
  }

  /**
   * Estimate network bandwidth (simple implementation)
   */
  private async estimateBandwidth(): Promise<number> {
    // Mock bandwidth - in production, could do a real bandwidth test
    // For now, assume reasonable broadband speeds
    return 100; // Mbps
  }

  /**
   * Generate a device name
   */
  private generateDeviceName(): string {
    const platform = typeof navigator !== "undefined" ? navigator.platform : "Unknown";
    const timestamp = new Date().toISOString().split("T")[0];
    return `${platform}-${timestamp}`;
  }
}

/**
 * Export singleton getter for convenience
 */
export function getDeviceAgent(config?: AgentConfig): DeviceAgent {
  return DeviceAgent.getInstance(config);
}
